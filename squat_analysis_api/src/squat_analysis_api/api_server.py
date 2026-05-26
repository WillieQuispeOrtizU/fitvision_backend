from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from squat_analysis_api.assistant_client import AssistantClient
from squat_analysis_api.video_processor import process_video

try:
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError:
    BackgroundTasks = FastAPI = File = Form = HTTPException = UploadFile = None
    CORSMiddleware = FileResponse = None


DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_ASSISTANT_URL = "http://127.0.0.1:7862"
DEFAULT_JOBS_DIR = Path("jobs")
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


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


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self, *, job_id: str, mode: str, input_filename: str) -> JobState:
        job = JobState(
            job_id=job_id,
            status="queued",
            mode=mode,
            input_filename=input_filename,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> JobState:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            return job


def create_app(
    *,
    jobs_dir: Path | None = None,
    assistant_url: str | None = None,
    job_store: JobStore | None = None,
):
    if FastAPI is None:
        raise RuntimeError("Install assistant extras before running the API server: uv sync --extra assistant")

    jobs_root = (jobs_dir or Path(os.getenv("SQUAT_ANALYSIS_JOBS_DIR", DEFAULT_JOBS_DIR))).resolve()
    configured_assistant_url = assistant_url if assistant_url is not None else os.getenv("SQUAT_ASSISTANT_URL", DEFAULT_ASSISTANT_URL)
    store = job_store or JobStore()
    jobs_root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Squat Analysis Upload API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "assistant_url": configured_assistant_url,
            "jobs_dir": str(jobs_root),
        }

    @app.post("/api/jobs")
    def create_job(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        mode: str = Form("beginner"),
    ) -> dict[str, Any]:
        if mode not in {"beginner", "pro"}:
            raise HTTPException(status_code=400, detail="mode must be beginner or pro")

        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="video must be .mp4, .mov, or .avi")

        job_id = uuid.uuid4().hex
        job_dir = jobs_root / job_id
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_name = _safe_filename(video.filename or f"upload{suffix}")
        input_path = input_dir / input_name
        with input_path.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)

        job = store.create(job_id=job_id, mode=mode, input_filename=input_name)
        store.update(job_id, original_video_url=_file_url(job_id, input_name))
        background_tasks.add_task(
            _process_job,
            store=store,
            jobs_root=jobs_root,
            job_id=job_id,
            input_path=input_path,
            mode=mode,
            assistant_url=configured_assistant_url,
        )
        return _job_payload(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_payload(job)

    @app.get("/api/jobs/{job_id}/report")
    def get_report(job_id: str) -> dict[str, Any]:
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
) -> None:
    output_dir = jobs_root / job_id / "output"
    frame_dir = jobs_root / job_id / "frames"
    output_path = output_dir / f"{input_path.stem}_analyzed.mp4"
    report_path = output_dir / "analysis.json"
    try:
        store.update(job_id, status="processing", error=None)
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
        store.update(
            job_id,
            status="completed",
            report_url=f"/api/jobs/{job_id}/report",
            annotated_video_url=_file_url(job_id, output_path.name),
        )
    except Exception as exc:
        store.update(job_id, status="failed", error=str(exc))


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


def _file_url(job_id: str, filename: str) -> str:
    return f"/api/jobs/{job_id}/files/{Path(filename).name}"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _cors_origins() -> list[str]:
    raw = os.getenv("SQUAT_ANALYSIS_CORS_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return ["http://127.0.0.1:5173", "http://localhost:5173"]


if __name__ == "__main__":
    main()
