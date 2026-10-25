from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from squat_analysis_api.assistant_client import AssistantClient
from squat_analysis_api.video_processor import process_video

try:
    from fastapi import (
        BackgroundTasks,
        Depends,
        FastAPI,
        File,
        Form,
        HTTPException,
        UploadFile,
        status,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError:
    BackgroundTasks = Depends = FastAPI = File = Form = HTTPException = UploadFile = None
    status = None
    CORSMiddleware = FileResponse = None


DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_ASSISTANT_URL = "http://127.0.0.1:7862"
DEFAULT_JOBS_DIR = Path("jobs")
DEFAULT_JOBS_TTL_HOURS = 24.0
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
INTERRUPTED_JOB_ERROR = "Interrupted by server restart"


logger = logging.getLogger(__name__)


@dataclass
class JobState:
    job_id: str
    status: str
    mode: str
    input_filename: str
    error: str | None = None
    report_url: str | None = None
    original_video_url: str | None = None
    annotated_video_url: str | None = None
    video_seconds: float | None = None
    assistant_seconds: float | None = None


class JobStore:
    def __init__(self, persist_dir: Path | None = None) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._persist_dir = persist_dir

    @classmethod
    def load(cls, persist_dir: Path) -> "JobStore":
        """Build a store that persists to disk and rehydrate any prior jobs so
        status survives a server restart."""
        store = cls(persist_dir=persist_dir)
        store._rehydrate()
        return store

    def create(self, *, job_id: str, mode: str, input_filename: str) -> JobState:
        job = JobState(
            job_id=job_id,
            status="queued",
            mode=mode,
            input_filename=input_filename,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
        return job

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> JobState:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            self._persist(job)
            return job

    def delete(self, job_id: str) -> None:
        """Forget a job whose durable evaluation could not be created."""
        with self._lock:
            self._jobs.pop(job_id, None)

    def interrupted_job_ids(self) -> tuple[str, ...]:
        """Return jobs that were left in flight when this store rehydrated."""
        with self._lock:
            return tuple(
                job.job_id
                for job in self._jobs.values()
                if job.status == "failed" and job.error == INTERRUPTED_JOB_ERROR
            )

    def _job_file(self, job_id: str) -> Path:
        return self._persist_dir / job_id / "job.json"  # type: ignore[union-attr]

    def _persist(self, job: JobState) -> None:
        if self._persist_dir is None:
            return
        path = self._job_file(job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(job)), encoding="utf-8")

    def _rehydrate(self) -> None:
        if self._persist_dir is None or not self._persist_dir.exists():
            return
        for job_file in sorted(self._persist_dir.glob("*/job.json")):
            try:
                data = json.loads(job_file.read_text(encoding="utf-8"))
                job = JobState(**data)
            except (json.JSONDecodeError, TypeError, OSError):
                continue
            # A job left mid-flight by a restart can never resume: the background
            # task that owned it is gone. Surface it as failed rather than leaving
            # the client polling a "processing" status forever.
            if job.status in ("queued", "processing"):
                job.status = "failed"
                job.error = INTERRUPTED_JOB_ERROR
            self._jobs[job.job_id] = job
            self._persist(job)


def create_app(
    *,
    jobs_dir: Path | None = None,
    assistant_url: str | None = None,
    job_store: JobStore | None = None,
    database_healthcheck: Callable[[], str] | None = None,
    session_factory: Callable[[], Any] | None = None,
):
    if FastAPI is None:
        raise RuntimeError("Install assistant extras before running the API server: uv sync --extra assistant")

    jobs_root = (jobs_dir or Path(os.getenv("SQUAT_ANALYSIS_JOBS_DIR", DEFAULT_JOBS_DIR))).resolve()
    configured_assistant_url = assistant_url if assistant_url is not None else os.getenv("SQUAT_ASSISTANT_URL", DEFAULT_ASSISTANT_URL)
    jobs_root.mkdir(parents=True, exist_ok=True)
    _cleanup_jobs(jobs_root, _jobs_ttl_hours())
    store = job_store or JobStore.load(jobs_root)

    from sqlalchemy import select
    from sqlalchemy.exc import SQLAlchemyError

    from squat_analysis_api.auth import get_current_user
    from squat_analysis_api.auth_routes import router as auth_router
    from squat_analysis_api.database import (
        check_database_connection,
        get_db,
        get_session_factory,
    )
    from squat_analysis_api.evaluation_routes import router as evaluation_router
    from squat_analysis_api.gamification_routes import router as gamification_router
    from squat_analysis_api.evaluation_service import (
        create_queued_evaluation,
        fail_interrupted_evaluations,
    )
    from squat_analysis_api.models import Evaluation

    configured_database_healthcheck = (
        database_healthcheck or check_database_connection
    )
    configured_session_factory = session_factory
    if configured_session_factory is None:
        configured_session_factory = lambda: get_session_factory()()

    interrupted_job_ids = store.interrupted_job_ids()
    if interrupted_job_ids:
        try:
            fail_interrupted_evaluations(
                configured_session_factory,
                job_ids=interrupted_job_ids,
                error=INTERRUPTED_JOB_ERROR,
            )
        except Exception:
            # Database availability should not prevent the API from starting.
            # The helper has rolled back; keep the disk-backed job failure and
            # make the reconciliation problem explicit in server logs.
            logger.exception(
                "Could not reconcile %d interrupted evaluation(s)",
                len(interrupted_job_ids),
            )

    app = FastAPI(title="Squat Analysis Upload API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth_router)
    app.include_router(evaluation_router)
    app.include_router(gamification_router)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "database": {"status": configured_database_healthcheck()},
            "assistant_url": configured_assistant_url,
            "jobs_dir": str(jobs_root),
        }

    @app.post("/api/jobs")
    def create_job(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        mode: str = Form(...),
        current_user=Depends(get_current_user),
        db=Depends(get_db),
    ) -> dict[str, Any]:
        if mode not in {"beginner", "pro"}:
            raise HTTPException(status_code=400, detail="mode must be beginner or pro")

        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="video must be .mp4, .mov, or .avi")
        _validate_video_upload(video, suffix)

        job_id = uuid.uuid4().hex
        job_dir = jobs_root / job_id
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_name = _safe_filename(video.filename or f"upload{suffix}")
        input_path = input_dir / input_name
        with input_path.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)

        original_video_url = _file_url(job_id, input_name)
        job = store.create(job_id=job_id, mode=mode, input_filename=input_name)
        store.update(job_id, original_video_url=original_video_url)
        try:
            create_queued_evaluation(
                db,
                user_id=current_user.id,
                job_id=job_id,
                mode=mode,
                input_filename=input_name,
                original_video_url=original_video_url,
            )
        except SQLAlchemyError as exc:
            db.rollback()
            store.delete(job_id)
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.exception("Could not persist queued evaluation for job %s", job_id)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database is unavailable",
            ) from exc
        background_tasks.add_task(
            _process_job,
            store=store,
            jobs_root=jobs_root,
            job_id=job_id,
            input_path=input_path,
            mode=mode,
            assistant_url=configured_assistant_url,
            session_factory=configured_session_factory,
        )
        return _job_payload(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(
        job_id: str,
        current_user=Depends(get_current_user),
        db=Depends(get_db),
    ) -> dict[str, Any]:
        evaluation = db.scalar(
            select(Evaluation.id).where(
                Evaluation.job_id == job_id,
                Evaluation.user_id == current_user.id,
            )
        )
        if evaluation is None:
            raise HTTPException(status_code=404, detail="job not found")
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_payload(job)

    @app.get("/api/jobs/{job_id}/report")
    def get_report(
        job_id: str,
        current_user=Depends(get_current_user),
        db=Depends(get_db),
    ) -> dict[str, Any]:
        evaluation = db.scalar(
            select(Evaluation.id).where(
                Evaluation.job_id == job_id,
                Evaluation.user_id == current_user.id,
            )
        )
        if evaluation is None:
            raise HTTPException(status_code=404, detail="job not found")
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "completed":
            raise HTTPException(status_code=409, detail=f"job is {job.status}")

        report_path = _resolve_report_path(jobs_root, job_id)
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="report not found")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return _web_report(job_id, report)

    @app.get("/api/jobs/{job_id}/files/{filename}")
    def get_file(job_id: str, filename: str):
        path = _resolve_job_file(jobs_root, job_id, filename)
        if path is None:
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(path)

    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install assistant extras before running the API server: uv sync --extra assistant") from exc

    host = os.getenv("SQUAT_ANALYSIS_API_HOST", DEFAULT_API_HOST)
    port = int(os.getenv("SQUAT_ANALYSIS_API_PORT", str(DEFAULT_API_PORT)))
    uvicorn.run(create_app(), host=host, port=port)


def _process_job(
    *,
    store: JobStore,
    jobs_root: Path,
    job_id: str,
    input_path: Path,
    mode: str,
    assistant_url: str,
    session_factory: Callable[[], Any],
) -> None:
    from squat_analysis_api.evaluation_service import (
        complete_evaluation,
        fail_evaluation,
        mark_evaluation_processing,
    )

    output_dir = jobs_root / job_id / "output"
    frame_dir = jobs_root / job_id / "frames"
    output_path = output_dir / f"{input_path.stem}_analyzed.mp4"
    report_path = output_dir / "analysis.json"
    try:
        # A single worker claims queued work. Retries that find it already
        # processing or completed stop here, which also prevents double-counting
        # progress for the same evaluation.
        if not mark_evaluation_processing(session_factory, job_id=job_id):
            logger.info(
                "Skipping duplicate processing attempt for job_id=%s",
                job_id,
            )
            return
        store.update(job_id, status="processing", error=None)
        started = time.perf_counter()
        process_video(
            input_path,
            output_path,
            mode,
            write_json_report=True,
            assistant_url=assistant_url,
            assistant_client=AssistantClient(assistant_url),
            report_path=report_path,
            frame_dir=frame_dir,
        )
        total_seconds = time.perf_counter() - started
        report = json.loads(report_path.read_text(encoding="utf-8"))

        # The assistant (Falcon + Gemma) latency is recorded in the report. The
        # remaining time is the MediaPipe analysis + annotated-video encode.
        assistant_seconds = _assistant_latency_from_report(report)
        video_seconds = total_seconds - (assistant_seconds or 0.0)
        persisted_video_seconds = round(max(0.0, video_seconds), 1)
        persisted_assistant_seconds = (
            round(assistant_seconds, 1) if assistant_seconds is not None else None
        )
        original_video_url = _file_url(job_id, input_path.name)
        annotated_video_url = _file_url(job_id, output_path.name)

        complete_evaluation(
            session_factory,
            job_id=job_id,
            report_json=report,
            original_video_url=original_video_url,
            annotated_video_url=annotated_video_url,
            video_seconds=persisted_video_seconds,
            assistant_seconds=persisted_assistant_seconds,
        )
        store.update(
            job_id,
            status="completed",
            report_url=f"/api/jobs/{job_id}/report",
            original_video_url=original_video_url,
            annotated_video_url=annotated_video_url,
            video_seconds=persisted_video_seconds,
            assistant_seconds=persisted_assistant_seconds,
        )
    except Exception as exc:
        logger.exception("Job processing failed for job_id=%s", job_id)
        should_mark_job_failed = True
        try:
            # False means a concurrent worker already committed completion. Do
            # not let a later duplicate failure downgrade that successful job.
            should_mark_job_failed = fail_evaluation(
                session_factory,
                job_id=job_id,
                error=str(exc),
            )
        except Exception:
            # The persistence helper already logs and rolls back. Preserve the
            # operational failure in the existing job store as a fallback.
            logger.exception(
                "Could not persist failure state for job_id=%s",
                job_id,
            )
        if should_mark_job_failed:
            store.update(job_id, status="failed", error=str(exc))


def _assistant_latency(report_path: Path) -> float | None:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _assistant_latency_from_report(report)


def _assistant_latency_from_report(report: dict[str, Any]) -> float | None:
    assistant = report.get("assistant", {})
    if not isinstance(assistant, dict):
        return None
    latency = assistant.get("latency_seconds")
    return float(latency) if isinstance(latency, (int, float)) else None


def _job_payload(job: JobState) -> dict[str, Any]:
    return asdict(job)


def _web_report(job_id: str, report: dict[str, Any]) -> dict[str, Any]:
    video = report.get("video", {})
    for key in ("input_path", "output_path"):
        path = video.get(key)
        if path:
            video[key] = _file_url(job_id, Path(path).name)

    for rep in report.get("reps", []):
        key_frames = rep.get("key_frames", {})
        for role, path in list(key_frames.items()):
            if path:
                key_frames[role] = _file_url(job_id, Path(path).name)
    return report


def _resolve_job_file(jobs_root: Path, job_id: str, filename: str) -> Path | None:
    if Path(filename).name != filename:
        return None

    job_dir = (jobs_root / job_id).resolve()
    if not _is_relative_to(job_dir, jobs_root):
        return None

    for subdir in ("input", "output", "frames"):
        candidate = (job_dir / subdir / filename).resolve()
        if _is_relative_to(candidate, job_dir) and candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_report_path(jobs_root: Path, job_id: str) -> Path:
    output_dir = jobs_root / job_id / "output"
    default_path = output_dir / "analysis.json"
    if default_path.exists():
        return default_path

    fallback_reports = sorted(output_dir.glob("*_analysis.json"))
    if fallback_reports:
        return fallback_reports[0]
    return default_path


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.replace(" ", "_")
    return name or "upload.mp4"


def _validate_video_upload(video: UploadFile, suffix: str) -> None:
    """Reject empty files and obvious extension/container mismatches.

    This intentionally validates only stable container signatures. Full codec
    decoding still belongs to OpenCV/FFmpeg in the background worker.
    """

    try:
        header = video.file.read(32)
        video.file.seek(0)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="video could not be read") from exc

    if not header:
        raise HTTPException(status_code=400, detail="video must not be empty")

    if suffix in {".mp4", ".mov"}:
        # ISO BMFF/QuickTime commonly starts with ftyp, but valid camera files
        # may start with another top-level atom before metadata is rearranged.
        valid_container = len(header) >= 8 and header[4:8] in {
            b"ftyp",
            b"moov",
            b"mdat",
            b"free",
            b"wide",
        }
    else:
        valid_container = (
            len(header) >= 12
            and header[:4] == b"RIFF"
            and header[8:12] == b"AVI "
        )

    if not valid_container:
        raise HTTPException(
            status_code=400,
            detail="video content does not match its .mp4, .mov, or .avi container",
        )


def _file_url(job_id: str, filename: str) -> str:
    return f"/api/jobs/{job_id}/files/{Path(filename).name}"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _jobs_ttl_hours() -> float:
    raw = os.getenv("SQUAT_ANALYSIS_JOBS_TTL_HOURS")
    if not raw:
        return DEFAULT_JOBS_TTL_HOURS
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_JOBS_TTL_HOURS


def _cleanup_jobs(jobs_root: Path, ttl_hours: float) -> None:
    """Remove job directories older than the TTL so jobs/ does not grow without
    bound. A non-positive TTL disables cleanup."""
    if ttl_hours <= 0 or not jobs_root.exists():
        return
    cutoff = time.time() - ttl_hours * 3600
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        try:
            if job_dir.stat().st_mtime < cutoff:
                shutil.rmtree(job_dir, ignore_errors=True)
        except OSError:
            continue


def _cors_origins() -> list[str]:
    raw = os.getenv("SQUAT_ANALYSIS_CORS_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return ["http://127.0.0.1:5173", "http://localhost:5173"]


if __name__ == "__main__":
    main()
