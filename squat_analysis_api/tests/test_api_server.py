from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from squat_analysis_api.api_server import JobStore, create_app


class ApiServerTest(unittest.TestCase):
    def test_upload_creates_completed_job_and_serves_report_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp)

            def fake_process(input_path, output_path, mode, **kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                kwargs["frame_dir"].mkdir(parents=True, exist_ok=True)
                self.assertEqual(kwargs["report_path"].name, "analysis.json")
                output_path.write_bytes(b"video")
                bottom = kwargs["frame_dir"] / "rep_001_bottom.jpg"
                bottom.write_bytes(b"image")
                report = {
                    "video": {
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                    },
                    "stats": {"correct": 1, "incorrect": 0, "total_reps": 1},
                    "reps": [
                        {
                            "rep_id": 1,
                            "key_frames": {"bottom": str(bottom)},
                            "assistant_feedback": {"summary": "Good.", "cues": [], "severity": "info"},
                        }
                    ],
                    "assistant": {"status": "completed"},
                }
                kwargs["report_path"].parent.mkdir(parents=True, exist_ok=True)
                kwargs["report_path"].write_text(json.dumps(report), encoding="utf-8")

            with patch("squat_analysis_api.api_server.process_video", side_effect=fake_process):
                client = TestClient(create_app(jobs_dir=jobs_dir, assistant_url="http://assistant.test"))
                response = client.post(
                    "/api/jobs",
                    files={"video": ("squat.mp4", b"fake video", "video/mp4")},
                    data={"mode": "beginner"},
                )

            self.assertEqual(response.status_code, 200)
            job = response.json()
            self.assertEqual(job["status"], "queued")
            job_id = job["job_id"]
            poll = client.get(f"/api/jobs/{job_id}").json()
            self.assertEqual(poll["status"], "completed")
            self.assertTrue((jobs_dir / job_id / "input" / "squat.mp4").exists())

            report_response = client.get(f"/api/jobs/{job_id}/report")
            self.assertEqual(report_response.status_code, 200)
            report = report_response.json()
            self.assertEqual(report["stats"]["total_reps"], 1)
            self.assertEqual(report["video"]["output_path"], f"/api/jobs/{job_id}/files/squat_analyzed.mp4")
            self.assertEqual(report["reps"][0]["key_frames"]["bottom"], f"/api/jobs/{job_id}/files/rep_001_bottom.jpg")

            file_response = client.get(f"/api/jobs/{job_id}/files/squat_analyzed.mp4")
            self.assertEqual(file_response.status_code, 200)
            self.assertEqual(file_response.content, b"video")

    def test_completed_job_reads_legacy_stemmed_report_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp)
            job_id = "legacy"
            output_dir = jobs_dir / job_id / "output"
            output_dir.mkdir(parents=True)
            (output_dir / "squat_3_analysis.json").write_text(
                json.dumps(
                    {
                        "video": {"input_path": "input/squat_3.mp4", "output_path": "output/squat_3_analyzed.mp4"},
                        "stats": {"correct": 1, "incorrect": 0, "total_reps": 1},
                        "reps": [],
                        "assistant": {"status": "completed"},
                    }
                ),
                encoding="utf-8",
            )
            store = JobStore()
            store.create(job_id=job_id, mode="beginner", input_filename="squat_3.mp4")
            store.update(job_id, status="completed")
            client = TestClient(create_app(jobs_dir=jobs_dir, assistant_url="http://assistant.test", job_store=store))

            response = client.get(f"/api/jobs/{job_id}/report")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["stats"]["total_reps"], 1)

    def test_failed_processing_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake_process(*args, **kwargs):
                raise RuntimeError("processing exploded")

            with patch("squat_analysis_api.api_server.process_video", side_effect=fake_process):
                client = TestClient(create_app(jobs_dir=Path(tmp), assistant_url="http://assistant.test"))
                response = client.post(
                    "/api/jobs",
                    files={"video": ("squat.mp4", b"fake video", "video/mp4")},
                    data={"mode": "beginner"},
                )

            job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
            self.assertEqual(job["status"], "failed")
            self.assertIn("processing exploded", job["error"])

    def test_file_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore()
            store.create(job_id="abc", mode="beginner", input_filename="squat.mp4")
            client = TestClient(create_app(jobs_dir=Path(tmp), assistant_url="http://assistant.test", job_store=store))

            response = client.get("/api/jobs/abc/files/..%2Fsecret.txt")

            self.assertEqual(response.status_code, 404)

    def test_job_status_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp)

            # First "process": persist a completed job and an interrupted one.
            store = JobStore.load(jobs_dir)
            store.create(job_id="done", mode="beginner", input_filename="a.mp4")
            store.update("done", status="completed")
            store.create(job_id="stuck", mode="beginner", input_filename="b.mp4")
            store.update("stuck", status="processing")

            # Simulate a server restart: a brand new store loads from disk.
            reloaded = JobStore.load(jobs_dir)
            self.assertEqual(reloaded.get("done").status, "completed")
            stuck = reloaded.get("stuck")
            self.assertEqual(stuck.status, "failed")
            self.assertIn("restart", stuck.error)

    def test_old_jobs_are_cleaned_up_on_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp)
            old_job = jobs_dir / "old"
            old_job.mkdir(parents=True)
            (old_job / "marker").write_bytes(b"x")
            # Backdate the directory well past the TTL.
            past = time.time() - 48 * 3600
            os.utime(old_job, (past, past))

            client = TestClient(create_app(jobs_dir=jobs_dir, assistant_url="http://assistant.test"))
            client.get("/api/health")

            self.assertFalse(old_job.exists())

    def test_invalid_upload_mode_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(jobs_dir=Path(tmp), assistant_url="http://assistant.test"))

            response = client.post(
                "/api/jobs",
                files={"video": ("squat.mp4", b"fake video", "video/mp4")},
                data={"mode": "advanced"},
            )

            self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
