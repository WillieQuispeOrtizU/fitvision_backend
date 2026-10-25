from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from squat_analysis_api.api_server import JobStore, create_app
from squat_analysis_api.auth import get_current_user
from squat_analysis_api.database import get_db
from squat_analysis_api.models import (
    AppUser,
    AuthSession,
    Base,
    Evaluation,
    UserProgress,
)


VALID_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00"


class EvaluationsApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.jobs_dir = Path(self._tmp.name)
        self.engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        self.store = JobStore()
        app = create_app(
            jobs_dir=self.jobs_dir,
            assistant_url="http://assistant.test",
            job_store=self.store,
            database_healthcheck=lambda: "ok",
            session_factory=self.session_factory,
        )
        app.dependency_overrides[get_db] = override_get_db
        self.app = app
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.engine.dispose()
        self._tmp.cleanup()

    def test_job_without_token_is_unauthorized(self):
        response = self.client.post(
            "/api/jobs",
            files={"video": ("squat.mp4", VALID_MP4_BYTES, "video/mp4")},
            data={"mode": "beginner"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")
        with self.session_factory() as db:
            self.assertIsNone(db.scalar(select(Evaluation)))

    def test_job_creates_queued_evaluation_linked_to_current_user(self):
        token = self._register("owner@example.com")

        with patch.object(BackgroundTasks, "add_task"):
            response = self._post_job(token, filename="front squat.mp4", mode="pro")

        self.assertEqual(response.status_code, 200)
        job = response.json()
        self.assertEqual(job["status"], "queued")
        with self.session_factory() as db:
            user = db.scalar(select(AppUser).where(AppUser.email == "owner@example.com"))
            evaluation = db.scalar(
                select(Evaluation).where(Evaluation.job_id == job["job_id"])
            )
            self.assertIsNotNone(evaluation)
            self.assertEqual(evaluation.user_id, user.id)
            self.assertEqual(evaluation.exercise, "squat")
            self.assertEqual(evaluation.mode, "pro")
            self.assertEqual(evaluation.status, "queued")
            self.assertEqual(evaluation.input_filename, "front_squat.mp4")
            self.assertEqual(
                evaluation.original_video_url,
                f"/api/jobs/{job['job_id']}/files/front_squat.mp4",
            )
            self.assertEqual(evaluation.report_json, {})

    def test_database_failure_rolls_back_job_before_background_is_scheduled(self):
        class FailingSession:
            def __init__(self) -> None:
                self.rolled_back = False

            def add(self, _value: Any) -> None:
                pass

            def commit(self) -> None:
                raise SQLAlchemyError("database unavailable")

            def rollback(self) -> None:
                self.rolled_back = True

        failing_session = FailingSession()

        def override_failing_db():
            yield failing_session

        original_db_override = self.app.dependency_overrides[get_db]
        self.app.dependency_overrides[get_db] = override_failing_db
        self.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id=123
        )
        try:
            with (
                patch(
                    "squat_analysis_api.api_server.uuid.uuid4",
                    return_value=SimpleNamespace(hex="dbfailure"),
                ),
                patch.object(BackgroundTasks, "add_task") as add_task,
            ):
                response = self._post_job("unused-token")
        finally:
            self.app.dependency_overrides[get_db] = original_db_override
            self.app.dependency_overrides.pop(get_current_user, None)

        self.assertEqual(response.status_code, 503)
        self.assertTrue(failing_session.rolled_back)
        self.assertIsNone(self.store.get("dbfailure"))
        self.assertFalse((self.jobs_dir / "dbfailure").exists())
        add_task.assert_not_called()

    def test_user_cannot_read_another_users_job_or_evaluation(self):
        owner_token = self._register("owner@example.com")
        other_token = self._register("other@example.com")
        with patch.object(BackgroundTasks, "add_task"):
            created = self._post_job(owner_token)
        job_id = created.json()["job_id"]

        with self.session_factory() as db:
            evaluation_id = db.scalar(
                select(Evaluation.id).where(Evaluation.job_id == job_id)
            )

        owner_job = self.client.get(
            f"/api/jobs/{job_id}", headers=self._auth(owner_token)
        )
        other_job = self.client.get(
            f"/api/jobs/{job_id}", headers=self._auth(other_token)
        )
        other_report = self.client.get(
            f"/api/jobs/{job_id}/report", headers=self._auth(other_token)
        )
        other_detail = self.client.get(
            f"/api/evaluations/{evaluation_id}", headers=self._auth(other_token)
        )

        self.assertEqual(owner_job.status_code, 200)
        self.assertEqual(other_job.status_code, 404)
        self.assertEqual(other_report.status_code, 404)
        self.assertEqual(other_detail.status_code, 404)

    def test_completed_job_persists_authoritative_report_and_metrics(self):
        token = self._register("completed@example.com")
        report = self._completed_report()

        with patch(
            "squat_analysis_api.api_server.process_video",
            side_effect=self._write_report(report),
        ):
            response = self._post_job(token)

        self.assertEqual(response.status_code, 200)
        job_id = response.json()["job_id"]
        with self.session_factory() as db:
            evaluation = db.scalar(
                select(Evaluation).where(Evaluation.job_id == job_id)
            )
            progress = db.get(UserProgress, evaluation.user_id)

            self.assertEqual(evaluation.status, "completed")
            self.assertIs(evaluation.correct, False)
            self.assertEqual(evaluation.total_reps, 3)
            self.assertEqual(evaluation.correct_reps, 2)
            self.assertEqual(evaluation.incorrect_reps, 1)
            self.assertEqual(
                evaluation.original_video_url,
                f"/api/jobs/{job_id}/files/squat.mp4",
            )
            self.assertEqual(
                evaluation.annotated_video_url,
                f"/api/jobs/{job_id}/files/squat_analyzed.mp4",
            )
            self.assertEqual(float(evaluation.video_seconds), 0.0)
            self.assertEqual(float(evaluation.assistant_seconds), 1.3)
            self.assertEqual(evaluation.report_json, report)
            self.assertIsNotNone(evaluation.completed_at)
            self.assertIsNotNone(evaluation.updated_at)
            self.assertEqual(progress.total_evaluations, 1)
            self.assertEqual(progress.total_correct_reps, 2)
            self.assertEqual(progress.total_incorrect_reps, 1)

    def test_processing_status_is_persisted_before_analyzer_runs(self):
        token = self._register("processing@example.com")
        report = self._completed_report(correct=1, incorrect=0)
        write_report = self._write_report(report)
        observed_statuses: list[str] = []

        def inspect_then_process(
            input_path: Path,
            output_path: Path,
            mode: str,
            **kwargs: Any,
        ) -> None:
            job_id = input_path.parents[1].name
            with self.session_factory() as db:
                observed_statuses.append(
                    db.scalar(
                        select(Evaluation.status).where(
                            Evaluation.job_id == job_id
                        )
                    )
                )
            write_report(input_path, output_path, mode, **kwargs)

        with patch(
            "squat_analysis_api.api_server.process_video",
            side_effect=inspect_then_process,
        ):
            response = self._post_job(token)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(observed_statuses, ["processing"])

    def test_failed_job_persists_error_without_updating_progress(self):
        token = self._register("failed@example.com")

        with patch(
            "squat_analysis_api.api_server.process_video",
            side_effect=RuntimeError("processing exploded"),
        ):
            response = self._post_job(token)

        job_id = response.json()["job_id"]
        with self.session_factory() as db:
            evaluation = db.scalar(
                select(Evaluation).where(Evaluation.job_id == job_id)
            )
            progress = db.get(UserProgress, evaluation.user_id)

            self.assertEqual(evaluation.status, "failed")
            self.assertIn("processing exploded", evaluation.error)
            self.assertEqual(progress.total_evaluations, 0)
            self.assertEqual(progress.total_correct_reps, 0)
            self.assertEqual(progress.total_incorrect_reps, 0)

    def test_history_is_user_scoped_and_sorted_newest_first(self):
        token = self._register("history@example.com")
        other_token = self._register("history-other@example.com")
        now = datetime.now(timezone.utc)
        with self.session_factory() as db:
            owner_id = db.scalar(
                select(AppUser.id).where(AppUser.email == "history@example.com")
            )
            other_id = db.scalar(
                select(AppUser.id).where(AppUser.email == "history-other@example.com")
            )
            db.add_all(
                [
                    self._evaluation(
                        owner_id, "older", created_at=now - timedelta(days=2)
                    ),
                    self._evaluation(owner_id, "newer", created_at=now),
                    self._evaluation(
                        other_id, "other", created_at=now + timedelta(days=1)
                    ),
                ]
            )
            db.commit()

        response = self.client.get(
            "/api/evaluations", headers=self._auth(token)
        )
        other_response = self.client.get(
            "/api/evaluations", headers=self._auth(other_token)
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual([item["job_id"] for item in payload], ["newer", "older"])
        self.assertEqual(
            set(payload[0]),
            {
                "id",
                "job_id",
                "exercise",
                "mode",
                "status",
                "correct",
                "total_reps",
                "correct_reps",
                "incorrect_reps",
                "original_video_url",
                "annotated_video_url",
                "video_seconds",
                "assistant_seconds",
                "xp_earned",
                "created_at",
                "completed_at",
            },
        )
        self.assertNotIn("other", [item["job_id"] for item in payload])
        self.assertEqual(
            [item["job_id"] for item in other_response.json()], ["other"]
        )

    def test_evaluation_detail_returns_report_only_to_owner(self):
        token = self._register("detail@example.com")
        other_token = self._register("detail-other@example.com")
        report = {"stats": {"correct": 1, "incorrect": 0, "total_reps": 1}}
        with self.session_factory() as db:
            owner_id = db.scalar(
                select(AppUser.id).where(AppUser.email == "detail@example.com")
            )
            evaluation = self._evaluation(
                owner_id,
                "detail-job",
                created_at=datetime.now(timezone.utc),
                report_json=report,
            )
            db.add(evaluation)
            db.commit()
            evaluation_id = evaluation.id

        detail = self.client.get(
            f"/api/evaluations/{evaluation_id}", headers=self._auth(token)
        )
        forbidden = self.client.get(
            f"/api/evaluations/{evaluation_id}", headers=self._auth(other_token)
        )
        missing = self.client.get(
            "/api/evaluations/999999", headers=self._auth(token)
        )

        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["job_id"], "detail-job")
        self.assertEqual(detail.json()["report_json"], report)
        self.assertEqual(forbidden.status_code, 404)
        self.assertEqual(missing.status_code, 404)

    def test_duplicate_processing_increments_progress_only_once(self):
        token = self._register("idempotent@example.com")
        scheduled: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

        def capture_task(
            _tasks: BackgroundTasks,
            function: Any,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            scheduled.append((function, args, kwargs))

        with patch.object(BackgroundTasks, "add_task", new=capture_task):
            response = self._post_job(token)
        self.assertEqual(len(scheduled), 1)

        report = self._completed_report(correct=3, incorrect=0)
        function, args, kwargs = scheduled[0]
        with patch(
            "squat_analysis_api.api_server.process_video",
            side_effect=self._write_report(report),
        ):
            function(*args, **kwargs)
            function(*args, **kwargs)

        job_id = response.json()["job_id"]
        with self.session_factory() as db:
            evaluation = db.scalar(
                select(Evaluation).where(Evaluation.job_id == job_id)
            )
            progress = db.get(UserProgress, evaluation.user_id)

            self.assertEqual(evaluation.status, "completed")
            self.assertIs(evaluation.correct, True)
            self.assertEqual(progress.total_evaluations, 1)
            self.assertEqual(progress.total_correct_reps, 3)
            self.assertEqual(progress.total_incorrect_reps, 0)
            self.assertEqual(progress.total_xp, 100)
            self.assertEqual(progress.level, 1)
            self.assertEqual(progress.current_streak, 1)
            self.assertEqual(progress.longest_streak, 1)

    def _register(self, email: str) -> str:
        response = self.client.post(
            "/api/auth/register",
            json={
                "email": email,
                "password": "password",
                "full_name": "Test User",
                "training_level": "beginner",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["access_token"]

    def _post_job(
        self,
        token: str,
        *,
        filename: str = "squat.mp4",
        mode: str = "beginner",
    ):
        return self.client.post(
            "/api/jobs",
            headers=self._auth(token),
            files={"video": (filename, VALID_MP4_BYTES, "video/mp4")},
            data={"mode": mode},
        )

    @staticmethod
    def _auth(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _completed_report(
        *, correct: int = 2, incorrect: int = 1
    ) -> dict[str, Any]:
        repetitions = [
            {"rep_id": index + 1, "correct": index >= incorrect}
            for index in range(correct + incorrect)
        ]
        return {
            "video": {
                "input_path": "input/squat.mp4",
                "output_path": "output/squat_analyzed.mp4",
            },
            "stats": {
                "correct": correct,
                "incorrect": incorrect,
                "total_reps": correct + incorrect,
            },
            "reps": repetitions,
            "assistant": {"status": "completed", "latency_seconds": 1.3},
        }

    @staticmethod
    def _write_report(report: dict[str, Any]):
        def fake_process(input_path: Path, output_path: Path, mode: str, **kwargs):
            del input_path, mode
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"annotated video")
            report_path = kwargs["report_path"]
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report), encoding="utf-8")

        return fake_process

    @staticmethod
    def _evaluation(
        user_id: int,
        job_id: str,
        *,
        created_at: datetime,
        report_json: dict[str, Any] | None = None,
    ) -> Evaluation:
        return Evaluation(
            user_id=user_id,
            job_id=job_id,
            exercise="squat",
            mode="beginner",
            status="completed",
            input_filename="squat.mp4",
            correct=True,
            total_reps=1,
            correct_reps=1,
            incorrect_reps=0,
            original_video_url=f"/api/jobs/{job_id}/files/squat.mp4",
            annotated_video_url=f"/api/jobs/{job_id}/files/squat_analyzed.mp4",
            video_seconds=1.0,
            assistant_seconds=0.5,
            report_json=report_json or {},
            created_at=created_at,
            updated_at=created_at,
            completed_at=created_at,
        )


if __name__ == "__main__":
    unittest.main()
