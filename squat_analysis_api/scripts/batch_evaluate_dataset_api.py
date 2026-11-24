from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import http.client
import ipaddress
import json
import os
import secrets
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit


EXPECTED_ACCOUNT_EMAIL = "imanolquispe@hotmail.com"
DEFAULT_BASE_URL = "http://127.0.0.1:18000"
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = BACKEND_ROOT.parent / "dataset"
DEFAULT_STATE_FILE = BACKEND_ROOT / "batch_results" / "dataset_state.json"
DEFAULT_CSV_FILE = BACKEND_ROOT / "batch_results" / "dataset_results.csv"
DEFAULT_REPORTS_DIR = BACKEND_ROOT / "batch_results" / "reports"
STATE_VERSION = 2
MAX_VIDEO_BYTES = 500 * 1024 * 1024
TERMINAL_JOB_STATUSES = {"completed", "failed"}
CSV_FIELDS = (
    "file_name",
    "category",
    "mode",
    "job_id",
    "evaluation_status",
    "total_reps",
    "correct_reps",
    "incorrect_reps",
    "system_classification",
    "processing_seconds",
    "error_message",
)


class BatchError(RuntimeError):
    pass


class DatasetValidationError(BatchError):
    pass


class ApiError(BatchError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"FastAPI respondió HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class ApiProtocolError(BatchError):
    pass


class AmbiguousSubmissionError(BatchError):
    pass


@dataclass(frozen=True)
class VideoItem:
    path: Path
    relative_path: str
    file_name: str
    category: str
    size_bytes: int
    sha256: str


def expected_video_paths(dataset_root: Path) -> list[tuple[str, Path]]:
    paths = [
        ("GOOD", dataset_root / "Good" / "GOOD001.mp4"),
        ("BAD", dataset_root / "Bad" / "BAD001.mp4"),
    ]
    paths.extend(
        ("GOOD", dataset_root / "Good" / f"GOOD{index:03d}.mp4")
        for index in range(2, 6)
    )
    paths.extend(
        ("BAD", dataset_root / "Bad" / f"BAD{index:03d}.mp4")
        for index in range(2, 21)
    )
    paths.extend(
        ("GOOD", dataset_root / "Good" / f"GOOD{index:03d}.mp4")
        for index in range(6, 21)
    )
    return paths


def validate_dataset(dataset_root: Path) -> list[VideoItem]:
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise DatasetValidationError(f"No existe el dataset: {root}")

    expected = expected_video_paths(root)
    expected_files = {path.resolve() for _, path in expected}
    actual_files: set[Path] = set()
    for directory_name in ("Good", "Bad"):
        directory = root / directory_name
        if not directory.is_dir():
            raise DatasetValidationError(f"Falta el directorio {directory}")
        actual_files.update(path.resolve() for path in directory.iterdir() if path.is_file())

    missing = sorted(path for path in expected_files if not path.is_file())
    extras = sorted(actual_files - expected_files)
    if missing or extras:
        details: list[str] = []
        if missing:
            details.append("faltan: " + ", ".join(path.name for path in missing))
        if extras:
            details.append("sobran: " + ", ".join(path.name for path in extras))
        raise DatasetValidationError("Inventario inválido; " + "; ".join(details))

    items: list[VideoItem] = []
    for category, path in expected:
        resolved = path.resolve()
        stat = resolved.stat()
        if resolved.suffix.lower() != ".mp4":
            raise DatasetValidationError(f"Formato no permitido: {resolved}")
        if stat.st_size <= 0:
            raise DatasetValidationError(f"Video vacío: {resolved}")
        if stat.st_size > MAX_VIDEO_BYTES:
            raise DatasetValidationError(
                f"Video mayor a 500 MiB ({stat.st_size} bytes): {resolved}"
            )
        with resolved.open("rb") as handle:
            header = handle.read(64)
        if len(header) < 8 or header[4:8] not in {
            b"ftyp",
            b"moov",
            b"mdat",
            b"free",
            b"wide",
        }:
            raise DatasetValidationError(
                f"El contenido no corresponde a un contenedor MP4/MOV: {resolved}"
            )
        items.append(
            VideoItem(
                path=resolved,
                relative_path=resolved.relative_to(root).as_posix(),
                file_name=resolved.name,
                category=category,
                size_bytes=stat.st_size,
                sha256=_file_sha256(resolved),
            )
        )
    return items


def select_videos(items: list[VideoItem], max_videos: int | None) -> list[VideoItem]:
    if max_videos is None:
        return list(items)
    if max_videos < 1 or max_videos > len(items):
        raise BatchError(f"--max-videos debe estar entre 1 y {len(items)}")
    # The manifest is priority ordered; max_videos always selects its prefix.
    return list(items[:max_videos])


def validate_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BatchError("--base-url debe ser una URL HTTP(S) válida")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BatchError("--base-url no debe contener credenciales, query ni fragmento")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise BatchError(
            "Se rechaza HTTP hacia un host no local. Usa el túnel SSH en "
            "http://127.0.0.1:18000 o configura HTTPS."
        )
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


class ApiClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 60.0) -> None:
        self.base_url = validate_base_url(base_url)
        self._parsed = urlsplit(self.base_url)
        self._timeout_seconds = timeout_seconds
        self._token: str | None = None

    def login(self, email: str, password: str) -> dict[str, Any]:
        payload = self._json_request(
            "POST",
            "/api/auth/login",
            payload={"email": email, "password": password},
            authenticated=False,
        )
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise ApiProtocolError("El login no devolvió access_token")
        self._token = token
        return payload

    def get_current_user(self) -> dict[str, Any]:
        return self._json_request("GET", "/api/users/me")

    def list_evaluations(self) -> list[dict[str, Any]]:
        payload = self._json_request("GET", "/api/evaluations")
        if not isinstance(payload, list):
            raise ApiProtocolError("GET /api/evaluations no devolvió una lista")
        return [row for row in payload if isinstance(row, dict)]

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._json_request("GET", f"/api/jobs/{quote(job_id, safe='')}")

    def get_evaluation_detail(self, evaluation_id: int) -> dict[str, Any]:
        payload = self._json_request("GET", f"/api/evaluations/{evaluation_id}")
        if not isinstance(payload, dict):
            raise ApiProtocolError(
                "GET /api/evaluations/{id} no devolvió un objeto JSON"
            )
        return payload

    def upload_video(self, video: VideoItem, mode: str) -> dict[str, Any]:
        self._require_token()
        boundary = f"----FitVisionBatch{secrets.token_hex(16)}"
        before_mode = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="mode"\r\n\r\n'
            f"{mode}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="video"; filename="{video.file_name}"\r\n'
            "Content-Type: video/mp4\r\n\r\n"
        ).encode("utf-8")
        closing = f"\r\n--{boundary}--\r\n".encode("ascii")
        content_length = len(before_mode) + video.size_bytes + len(closing)
        connection = self._connection(timeout_seconds=max(120.0, self._timeout_seconds))
        started = False
        try:
            connection.putrequest("POST", self._target("/api/jobs"))
            connection.putheader("Accept", "application/json")
            connection.putheader("Authorization", f"Bearer {self._token}")
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(content_length))
            connection.endheaders()
            started = True
            connection.send(before_mode)
            with video.path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    connection.send(chunk)
            connection.send(closing)
            response = connection.getresponse()
            status = response.status
            body = response.read()
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            if started:
                raise AmbiguousSubmissionError(
                    "Se perdió la respuesta después de iniciar el upload; "
                    "no se reintentará sin reconciliar /api/evaluations."
                ) from exc
            raise BatchError("No se pudo conectar con FastAPI") from exc
        finally:
            connection.close()

        try:
            payload = _decode_json(body)
        except ApiProtocolError:
            if status >= 500:
                raise AmbiguousSubmissionError(
                    f"FastAPI respondió HTTP {status} sin JSON después del upload; "
                    "el estado queda ambiguo y no se reintentará automáticamente."
                )
            raise
        if status >= 500:
            raise AmbiguousSubmissionError(
                f"FastAPI respondió HTTP {status} después del upload; "
                "el estado queda ambiguo y no se reintentará automáticamente."
            )
        if status < 200 or status >= 300:
            raise ApiError(status, _error_detail(payload))
        if not isinstance(payload, dict) or not isinstance(payload.get("job_id"), str):
            raise AmbiguousSubmissionError(
                "FastAPI aceptó el upload pero no devolvió un job_id reconocible."
            )
        return payload

    def _json_request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
            headers["Content-Length"] = str(len(body))
        if authenticated:
            self._require_token()
            headers["Authorization"] = f"Bearer {self._token}"
        connection = self._connection()
        try:
            connection.request(method, self._target(path), body=body, headers=headers)
            response = connection.getresponse()
            status = response.status
            raw = response.read()
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise BatchError("No se pudo completar la solicitud a FastAPI") from exc
        finally:
            connection.close()
        decoded = _decode_json(raw)
        if status < 200 or status >= 300:
            raise ApiError(status, _error_detail(decoded))
        return decoded

    def _connection(self, *, timeout_seconds: float | None = None):
        timeout = timeout_seconds or self._timeout_seconds
        host = self._parsed.hostname or ""
        if self._parsed.scheme == "https":
            return http.client.HTTPSConnection(host, self._parsed.port, timeout=timeout)
        return http.client.HTTPConnection(host, self._parsed.port, timeout=timeout)

    def _target(self, path: str) -> str:
        prefix = self._parsed.path.rstrip("/")
        return f"{prefix}/{path.lstrip('/')}"

    def _require_token(self) -> None:
        if not self._token:
            raise BatchError("No existe una sesión autenticada en memoria")


def authenticate_expected_account(client: ApiClient) -> dict[str, Any]:
    typed_email = input(
        f"Correo FitVision [{EXPECTED_ACCOUNT_EMAIL}]: "
    ).strip() or EXPECTED_ACCOUNT_EMAIL
    if typed_email.lower() != EXPECTED_ACCOUNT_EMAIL:
        raise BatchError(
            f"Esta ejecución está restringida a {EXPECTED_ACCOUNT_EMAIL}"
        )
    password = getpass.getpass("Contraseña FitVision (no se guardará): ")
    if not password:
        raise BatchError("La contraseña no puede estar vacía")
    client.login(typed_email, password)
    user = client.get_current_user()
    authenticated_email = str(user.get("email", "")).strip().lower()
    if authenticated_email != EXPECTED_ACCOUNT_EMAIL:
        raise BatchError(
            "La cuenta autenticada no coincide con la cuenta autorizada; se aborta."
        )
    return user


def execute_batch(
    *,
    client: ApiClient,
    user: dict[str, Any],
    all_items: list[VideoItem],
    selected_items: list[VideoItem],
    mode: str,
    state_path: Path,
    csv_path: Path,
    reports_dir: Path,
    resume: bool,
    reconcile_job_id: str | None,
    confirm_no_job: bool,
    poll_interval_seconds: float,
    job_timeout_seconds: float,
) -> None:
    if resume:
        if not state_path.is_file():
            raise BatchError(f"No existe el checkpoint para --resume: {state_path}")
        state = _load_json(state_path)
        _validate_state(
            state,
            all_items=all_items,
            mode=mode,
            base_url=client.base_url,
            user=user,
            reports_dir=reports_dir,
        )
    else:
        existing_reports = [
            _report_path(reports_dir, item)
            for item in all_items
            if _report_path(reports_dir, item).exists()
        ]
        if state_path.exists() or csv_path.exists() or existing_reports:
            raise BatchError(
                "Ya existe un checkpoint, CSV o reporte JSON. Usa --resume o "
                "elige rutas nuevas para evitar evaluaciones duplicadas o "
                "sobrescribir resultados."
            )
        state = _new_state(
            all_items=all_items,
            mode=mode,
            base_url=client.base_url,
            user=user,
            reports_dir=reports_dir,
        )
        _ensure_report_directories(reports_dir)
        _save_progress(state, state_path, csv_path)

    if state.get("uncertain_upload"):
        _reconcile_uncertain(
            client,
            state,
            state_path,
            csv_path,
            approved_job_id=reconcile_job_id,
            confirm_no_job=confirm_no_job,
        )

    records = {record["relative_path"]: record for record in state["videos"]}
    for position, item in enumerate(selected_items, start=1):
        record = records[item.relative_path]
        report_path = _report_path(reports_dir, item)
        if record.get("finalized"):
            if (
                record.get("evaluation_status") == "rejected"
                and not record.get("job_id")
            ):
                print(f"[{position}/{len(selected_items)}] rechazado: {item.relative_path}")
                continue
            if _restore_saved_report(record, report_path):
                print(f"[{position}/{len(selected_items)}] omitido: {item.relative_path}")
                continue
            record["finalized"] = False
            _save_progress(state, state_path, csv_path)
        elif record.get("job_id") and _restore_saved_report(record, report_path):
            _save_progress(state, state_path, csv_path)
            print(
                f"[{position}/{len(selected_items)}] reporte recuperado: "
                f"{item.relative_path}"
            )
            continue

        job_id = record.get("job_id")
        if not job_id:
            print(f"[{position}/{len(selected_items)}] subiendo: {item.relative_path}")
            baseline = [
                row.get("job_id")
                for row in client.list_evaluations()
                if isinstance(row.get("job_id"), str)
            ]
            state["uncertain_upload"] = {
                "relative_path": item.relative_path,
                "file_name": item.file_name,
                "mode": mode,
                "started_at": _utc_iso(),
                "baseline_job_ids": baseline,
            }
            record["local_status"] = "uploading"
            _save_progress(state, state_path, csv_path)
            try:
                job = client.upload_video(item, mode)
            except ApiError as exc:
                state["uncertain_upload"] = None
                if exc.status == 401:
                    record["local_status"] = "pending"
                    _save_progress(state, state_path, csv_path)
                    raise
                record.update(
                    {
                        "local_status": "rejected",
                        "evaluation_status": "rejected",
                        "error_message": str(exc),
                        "finalized": True,
                    }
                )
                _save_progress(state, state_path, csv_path)
                continue
            except AmbiguousSubmissionError:
                record["local_status"] = "ambiguous"
                _save_progress(state, state_path, csv_path)
                raise

            job_id = job["job_id"]
            record.update(
                {
                    "job_id": job_id,
                    "local_status": str(job.get("status") or "submitted"),
                    "error_message": "",
                }
            )
            state["uncertain_upload"] = None
            _save_progress(state, state_path, csv_path)
        else:
            print(f"[{position}/{len(selected_items)}] reanudando job {job_id}")

        _poll_and_finalize(
            client,
            state,
            record,
            item,
            state_path,
            csv_path,
            reports_dir,
            poll_interval_seconds=poll_interval_seconds,
            job_timeout_seconds=job_timeout_seconds,
        )


def _poll_and_finalize(
    client: ApiClient,
    state: dict[str, Any],
    record: dict[str, Any],
    item: VideoItem,
    state_path: Path,
    csv_path: Path,
    reports_dir: Path,
    *,
    poll_interval_seconds: float,
    job_timeout_seconds: float,
) -> None:
    job_id = str(record["job_id"])
    deadline = time.monotonic() + job_timeout_seconds
    while True:
        try:
            job = client.get_job(job_id)
        except ApiError as exc:
            if exc.status != 404:
                raise
            persisted = _find_evaluation(client.list_evaluations(), job_id)
            if persisted is None or persisted.get("status") not in TERMINAL_JOB_STATUSES:
                raise
            _finalize_with_report(
                client,
                state,
                record,
                item,
                persisted,
                state_path,
                csv_path,
                reports_dir,
            )
            print(
                f"  recuperado desde historial: status={record['evaluation_status']} "
                f"reps={record.get('total_reps')} correct={record.get('correct')}"
            )
            return
        status = str(job.get("status", ""))
        if status not in {"queued", "processing", "completed", "failed"}:
            raise ApiProtocolError(f"Estado de job desconocido: {status!r}")
        record["local_status"] = status
        record["error_message"] = str(job.get("error") or "")
        record["video_seconds"] = job.get("video_seconds")
        record["assistant_seconds"] = job.get("assistant_seconds")
        _save_progress(state, state_path, csv_path)
        if status in TERMINAL_JOB_STATUSES:
            break
        if time.monotonic() >= deadline:
            raise BatchError(
                f"El job {job_id} no terminó dentro del tiempo configurado; "
                "el checkpoint permite continuar con --resume."
            )
        time.sleep(poll_interval_seconds)

    evaluation = _find_evaluation(client.list_evaluations(), job_id)
    if evaluation is None:
        raise ApiProtocolError(
            f"El job {job_id} terminó pero no aparece en /api/evaluations; "
            "no se volverá a subir el video."
        )
    _finalize_with_report(
        client,
        state,
        record,
        item,
        evaluation,
        state_path,
        csv_path,
        reports_dir,
    )
    print(
        f"  terminado: status={record['evaluation_status']} "
        f"reps={record.get('total_reps')} correct={record.get('correct')}"
    )


def _reconcile_uncertain(
    client: ApiClient,
    state: dict[str, Any],
    state_path: Path,
    csv_path: Path,
    *,
    approved_job_id: str | None,
    confirm_no_job: bool,
) -> None:
    uncertain = state["uncertain_upload"]
    baseline = set(uncertain.get("baseline_job_ids") or [])
    candidates = []
    for evaluation in client.list_evaluations():
        job_id = evaluation.get("job_id")
        original_url = evaluation.get("original_video_url")
        filename = Path(urlsplit(str(original_url or "")).path).name
        if (
            isinstance(job_id, str)
            and job_id not in baseline
            and evaluation.get("mode") == uncertain.get("mode")
            and filename == uncertain.get("file_name")
        ):
            candidates.append(evaluation)
    records = {record["relative_path"]: record for record in state["videos"]}
    record = records[uncertain["relative_path"]]
    if confirm_no_job:
        if candidates:
            candidate_ids = ", ".join(str(row["job_id"]) for row in candidates)
            raise AmbiguousSubmissionError(
                "No se puede confirmar ausencia de job porque existen candidatos: "
                f"{candidate_ids}"
            )
        record["job_id"] = ""
        record["local_status"] = "pending"
        state["uncertain_upload"] = None
        _save_progress(state, state_path, csv_path)
        print("Conciliación confirmada sin job; se permite un nuevo upload explícito.")
        return

    if approved_job_id is None:
        candidate_ids = ", ".join(str(row["job_id"]) for row in candidates) or "ninguno"
        raise AmbiguousSubmissionError(
            "La subida requiere conciliación explícita antes de continuar. "
            f"Candidatos: {candidate_ids}. Reejecuta con --resume y "
            "--reconcile-job-id JOB_ID, o usa --confirm-no-job solo después "
            "de comprobar que no se creó evaluación."
        )

    approved = [row for row in candidates if row.get("job_id") == approved_job_id]
    if len(approved) != 1:
        raise AmbiguousSubmissionError(
            f"El job aprobado {approved_job_id!r} no coincide de forma única "
            "con la subida ambigua. No se reintentará el archivo."
        )

    record["job_id"] = approved[0]["job_id"]
    record["local_status"] = str(approved[0].get("status") or "submitted")
    state["uncertain_upload"] = None
    _save_progress(state, state_path, csv_path)
    print(f"Upload ambiguo reconciliado con job {record['job_id']}")


def _new_state(
    *,
    all_items: list[VideoItem],
    mode: str,
    base_url: str,
    user: dict[str, Any],
    reports_dir: Path,
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "created_at": _utc_iso(),
        "dataset_root": str(all_items[0].path.parents[1]),
        "base_url": base_url,
        "mode": mode,
        "reports_dir": str(reports_dir),
        "user_id": user.get("id"),
        "account_email_sha256": hashlib.sha256(
            EXPECTED_ACCOUNT_EMAIL.encode("utf-8")
        ).hexdigest(),
        "uncertain_upload": None,
        "videos": [
            {
                "relative_path": item.relative_path,
                "file_name": item.file_name,
                "category": item.category,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "mode": mode,
                "job_id": "",
                "evaluation_id": None,
                "local_status": "pending",
                "evaluation_status": "pending",
                "total_reps": None,
                "correct_reps": None,
                "incorrect_reps": None,
                "correct": None,
                "video_seconds": None,
                "assistant_seconds": None,
                "error_message": "",
                "report_file": "",
                "report_sha256": "",
                "report_saved_at": None,
                "finalized": False,
            }
            for item in all_items
        ],
    }


def _validate_state(
    state: dict[str, Any],
    *,
    all_items: list[VideoItem],
    mode: str,
    base_url: str,
    user: dict[str, Any],
    reports_dir: Path,
) -> None:
    if state.get("version") != STATE_VERSION:
        raise BatchError("Versión de checkpoint no compatible")
    if state.get("mode") != mode:
        raise BatchError(
            f"El checkpoint usa mode={state.get('mode')}; ejecuta --mode {state.get('mode')}"
        )
    if state.get("base_url") != base_url:
        raise BatchError("El checkpoint pertenece a otra API_BASE_URL")
    if state.get("reports_dir") != str(reports_dir):
        raise BatchError("El checkpoint utiliza otro directorio de reportes")
    if state.get("user_id") != user.get("id"):
        raise BatchError("El checkpoint pertenece a otra cuenta FitVision")
    records = {record.get("relative_path"): record for record in state.get("videos", [])}
    if set(records) != {item.relative_path for item in all_items}:
        raise BatchError("El manifiesto del checkpoint no contiene los mismos 40 videos")
    for item in all_items:
        record = records[item.relative_path]
        if record.get("size_bytes") != item.size_bytes or record.get("sha256") != item.sha256:
            raise BatchError(
                f"El video cambió desde la ejecución anterior: {item.relative_path}"
            )


def _save_progress(
    state: dict[str, Any], state_path: Path, csv_path: Path
) -> None:
    state["updated_at"] = _utc_iso()
    _atomic_json_write(state_path, state)
    _atomic_csv_write(csv_path, state["videos"])


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_csv_write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8-sig",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            for record in records:
                writer.writerow(_csv_row(record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    status = record.get("evaluation_status", "")
    classification = (
        _classification(record.get("correct"))
        if status in TERMINAL_JOB_STATUSES
        else ""
    )
    return {
        "file_name": record.get("file_name", ""),
        "category": record.get("category", ""),
        "mode": record.get("mode", ""),
        "job_id": record.get("job_id", ""),
        "evaluation_status": record.get("evaluation_status", ""),
        "total_reps": _csv_value(record.get("total_reps")),
        "correct_reps": _csv_value(record.get("correct_reps")),
        "incorrect_reps": _csv_value(record.get("incorrect_reps")),
        "system_classification": classification,
        "processing_seconds": _processing_seconds(record),
        "error_message": record.get("error_message", ""),
    }


def _classification(value: Any) -> str:
    if value is True:
        return "correct"
    if value is False:
        return "incorrect"
    return "unclassified"


def _processing_seconds(record: dict[str, Any]) -> str:
    values = [record.get("video_seconds"), record.get("assistant_seconds")]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return "" if not numeric else f"{sum(numeric):.1f}"


def _find_evaluation(
    evaluations: list[dict[str, Any]], job_id: str
) -> dict[str, Any] | None:
    return next((row for row in evaluations if row.get("job_id") == job_id), None)


def _ensure_report_directories(reports_dir: Path) -> None:
    (reports_dir / "Good").mkdir(parents=True, exist_ok=True)
    (reports_dir / "Bad").mkdir(parents=True, exist_ok=True)


def _report_path(reports_dir: Path, item: VideoItem) -> Path:
    folder = {"GOOD": "Good", "BAD": "Bad"}.get(item.category)
    if folder is None:
        raise BatchError(f"Categoría de video no válida: {item.category!r}")
    stem = Path(item.file_name).stem
    if Path(item.file_name).name != item.file_name or not stem:
        raise BatchError(f"Nombre de video no seguro: {item.file_name!r}")
    return reports_dir / folder / f"{stem}.json"


def _validate_evaluation_detail(
    detail: dict[str, Any], *, evaluation_id: int, job_id: str
) -> None:
    if type(detail.get("id")) is not int or detail.get("id") != evaluation_id:
        raise ApiProtocolError("El detalle pertenece a otra evaluación")
    if detail.get("job_id") != job_id:
        raise ApiProtocolError("El detalle pertenece a otro job")
    if detail.get("status") not in TERMINAL_JOB_STATUSES:
        raise ApiProtocolError("El detalle de evaluación aún no es terminal")
    if not isinstance(detail.get("report_json"), dict):
        raise ApiProtocolError("El detalle no contiene report_json completo")


def _restore_saved_report(record: dict[str, Any], report_path: Path) -> bool:
    evaluation_id = record.get("evaluation_id")
    job_id = record.get("job_id")
    if not isinstance(evaluation_id, int) or not isinstance(job_id, str) or not job_id:
        return False
    if not report_path.is_file():
        return False
    actual_hash = _file_sha256(report_path)
    expected_hash = record.get("report_sha256")
    if isinstance(expected_hash, str) and expected_hash and expected_hash != actual_hash:
        return False
    try:
        detail = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(detail, dict):
            return False
        _validate_evaluation_detail(
            detail,
            evaluation_id=evaluation_id,
            job_id=job_id,
        )
    except (OSError, json.JSONDecodeError, ApiProtocolError):
        return False
    _apply_evaluation(record, detail)
    record.update(
        {
            "report_file": f"{report_path.parent.name}/{report_path.name}",
            "report_sha256": actual_hash,
            "finalized": True,
        }
    )
    return True


def _apply_evaluation(
    record: dict[str, Any], evaluation: dict[str, Any]
) -> None:
    record.update(
        {
            "evaluation_status": evaluation.get("status"),
            "total_reps": evaluation.get("total_reps"),
            "correct_reps": evaluation.get("correct_reps"),
            "incorrect_reps": evaluation.get("incorrect_reps"),
            "correct": evaluation.get("correct"),
            "video_seconds": evaluation.get("video_seconds"),
            "assistant_seconds": evaluation.get("assistant_seconds"),
            "local_status": str(
                evaluation.get("status") or record.get("local_status") or ""
            ),
            "finalized": False,
        }
    )


def _finalize_with_report(
    client: ApiClient,
    state: dict[str, Any],
    record: dict[str, Any],
    item: VideoItem,
    evaluation: dict[str, Any],
    state_path: Path,
    csv_path: Path,
    reports_dir: Path,
) -> None:
    evaluation_id = evaluation.get("id")
    if type(evaluation_id) is not int:
        raise ApiProtocolError(
            f"La evaluación del job {record.get('job_id')} no contiene un id válido"
        )
    record["evaluation_id"] = evaluation_id
    _apply_evaluation(record, evaluation)
    _save_progress(state, state_path, csv_path)

    detail = client.get_evaluation_detail(evaluation_id)
    _validate_evaluation_detail(
        detail,
        evaluation_id=evaluation_id,
        job_id=str(record["job_id"]),
    )
    report_path = _report_path(reports_dir, item)
    _atomic_json_write(report_path, detail)
    report_sha256 = _file_sha256(report_path)

    _apply_evaluation(record, detail)
    record.update(
        {
            "evaluation_id": evaluation_id,
            "report_file": report_path.relative_to(reports_dir).as_posix(),
            "report_sha256": report_sha256,
            "report_saved_at": _utc_iso(),
            "finalized": True,
        }
    )
    _save_progress(state, state_path, csv_path)


def _decode_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiProtocolError("FastAPI devolvió una respuesta JSON inválida") from exc


def _error_detail(payload: Any) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
    return "error sin detalle"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchError(f"No se pudo leer el checkpoint {path}") from exc
    if not isinstance(payload, dict):
        raise BatchError("El checkpoint no contiene un objeto JSON")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Carga secuencial y reanudable del dataset FitVision mediante FastAPI."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida y lista videos sin autenticar ni subir.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Habilita explícitamente login y uploads reales.",
    )
    parser.add_argument("--resume", action="store_true", help="Reanuda el checkpoint.")
    reconciliation = parser.add_mutually_exclusive_group()
    reconciliation.add_argument(
        "--reconcile-job-id",
        help="Asocia explícitamente un job verificado a una subida ambigua.",
    )
    reconciliation.add_argument(
        "--confirm-no-job",
        action="store_true",
        help="Confirma explícitamente que una subida ambigua no creó evaluación.",
    )
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--mode", choices=("beginner", "pro"), default="beginner")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--csv-file", type=Path, default=DEFAULT_CSV_FILE)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--job-timeout", type=float, default=3600.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        if args.resume and not args.execute:
            raise BatchError("--resume requiere --execute")
        if (args.reconcile_job_id or args.confirm_no_job) and not args.resume:
            raise BatchError(
                "--reconcile-job-id y --confirm-no-job requieren --execute --resume"
            )
        if args.poll_interval <= 0 or args.job_timeout <= 0 or args.request_timeout <= 0:
            raise BatchError("Los timeouts e intervalo de polling deben ser positivos")

        items = validate_dataset(args.dataset_root)
        selected = select_videos(items, args.max_videos)
        print(
            f"Dataset válido: {len(items)}/40 videos; "
            f"seleccionados para esta ejecución: {len(selected)}"
        )
        for item in selected:
            print(
                f"  {item.relative_path} | {item.size_bytes} bytes | "
                f"sha256={item.sha256[:12]}…"
            )

        if args.dry_run:
            print("Dry-run finalizado: no se inició sesión, no se creó CSV y no hubo uploads.")
            return 0

        base_url = validate_base_url(args.base_url)
        client = ApiClient(base_url, timeout_seconds=args.request_timeout)
        user = authenticate_expected_account(client)
        print(
            f"Cuenta verificada: {EXPECTED_ACCOUNT_EMAIL}; "
            "el token permanece solo en memoria."
        )
        execute_batch(
            client=client,
            user=user,
            all_items=items,
            selected_items=selected,
            mode=args.mode,
            state_path=args.state_file.expanduser().resolve(),
            csv_path=args.csv_file.expanduser().resolve(),
            reports_dir=args.reports_dir.expanduser().resolve(),
            resume=args.resume,
            reconcile_job_id=args.reconcile_job_id,
            confirm_no_job=args.confirm_no_job,
            poll_interval_seconds=args.poll_interval,
            job_timeout_seconds=args.job_timeout,
        )
        print(f"Checkpoint: {args.state_file.expanduser().resolve()}")
        print(f"CSV: {args.csv_file.expanduser().resolve()}")
        print(f"Reportes JSON: {args.reports_dir.expanduser().resolve()}")
        return 0
    except AmbiguousSubmissionError as exc:
        print(f"DETENIDO POR SUBIDA AMBIGUA: {exc}", file=sys.stderr)
        return 3
    except (BatchError, ApiError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Interrumpido; el checkpoint conserva el estado. Si ocurrió durante "
            "el upload, --resume exigirá conciliación antes de reintentar.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
