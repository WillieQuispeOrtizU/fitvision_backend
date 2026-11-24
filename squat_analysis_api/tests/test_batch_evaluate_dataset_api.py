from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.batch_evaluate_dataset_api import (
    AmbiguousSubmissionError,
    ApiClient,
    ApiProtocolError,
    BatchError,
    DatasetValidationError,
    _csv_row,
    execute_batch,
    main,
    select_videos,
    validate_base_url,
    validate_dataset,
)


MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * 128)
EXPECTED_RELATIVE_ORDER = (
    ["Good/GOOD001.mp4", "Bad/BAD001.mp4"]
    + [f"Good/GOOD{index:03d}.mp4" for index in range(2, 6)]
    + [f"Bad/BAD{index:03d}.mp4" for index in range(2, 21)]
    + [f"Good/GOOD{index:03d}.mp4" for index in range(6, 21)]
)


class FakeApiClient:
    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:18000"
        self.uploaded: list[str] = []
        self.jobs: dict[str, dict[str, object]] = {}
        self.evaluations: list[dict[str, object]] = []
        self.details: dict[int, dict[str, object]] = {}
        self.detail_calls: list[int] = []
        self.fail_ambiguously = False
        self.fail_detail = False
        self.wrong_detail_job_id: str | None = None

    def list_evaluations(self):
        return list(reversed(self.evaluations))

    def upload_video(self, video, mode: str):
        self.uploaded.append(video.relative_path)
        if self.fail_ambiguously:
            raise AmbiguousSubmissionError("simulated lost response")
        job_id = f"job-{len(self.uploaded):03d}"
        correct = video.category == "GOOD"
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "error": None,
            "video_seconds": 1.2,
            "assistant_seconds": 0.3,
        }
        evaluation_id = len(self.uploaded)
        summary = {
            "id": evaluation_id,
            "job_id": job_id,
            "exercise": "squat",
            "mode": mode,
            "status": "completed",
            "correct": correct,
            "total_reps": 2,
            "correct_reps": 2 if correct else 0,
            "incorrect_reps": 0 if correct else 2,
            "original_video_url": f"/api/jobs/{job_id}/files/{video.file_name}",
            "annotated_video_url": f"/api/jobs/{job_id}/files/analyzed.mp4",
            "video_seconds": 1.2,
            "assistant_seconds": 0.3,
            "xp_earned": 100 if correct else 70,
            "created_at": "2026-11-15T00:00:00Z",
            "completed_at": "2026-11-15T00:00:02Z",
        }
        self.evaluations.append(summary)
        self.details[evaluation_id] = {
            **summary,
            "report_json": {
                "video": {"frames": 120, "fps": 30.0},
                "reps": [
                    {
                        "rep_id": 1,
                        "correct": correct,
                        "angles": {"hip": 91.5, "knee": 88.0, "ankle": 72.0},
                        "issue_tags": [] if correct else ["bend_forward"],
                        "worst_issue": None if correct else "bend_forward",
                    }
                ],
                "assistant": {
                    "overall_summary": "Resumen autoritativo",
                    "rep_feedback": [
                        {
                            "rep_id": 1,
                            "summary": "Detalle",
                            "cues": ["Mantén el control."],
                            "severity": "info",
                        }
                    ],
                },
            },
        }
        return {"job_id": job_id, "status": "queued"}

    def get_job(self, job_id: str):
        return self.jobs[job_id]

    def get_evaluation_detail(self, evaluation_id: int):
        self.detail_calls.append(evaluation_id)
        if self.fail_detail:
            raise BatchError("simulated detail endpoint failure")
        detail = json.loads(json.dumps(self.details[evaluation_id]))
        if self.wrong_detail_job_id is not None:
            detail["job_id"] = self.wrong_detail_job_id
        return detail


class FakeHttpResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body


class RecordingHttpConnection:
    def __init__(self, response: FakeHttpResponse) -> None:
        self.response = response
        self.method = ""
        self.target = ""
        self.headers: dict[str, str] = {}
        self.body = bytearray()
        self.closed = False

    def putrequest(self, method: str, target: str) -> None:
        self.method = method
        self.target = target

    def putheader(self, name: str, value: str) -> None:
        self.headers[name.lower()] = value

    def endheaders(self) -> None:
        return None

    def send(self, data: bytes) -> None:
        self.body.extend(data)

    def getresponse(self) -> FakeHttpResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


class BatchDatasetScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "dataset"
        self.reports_dir = Path(self._tmp.name) / "reports"
        self._create_dataset()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_inventory_contains_exactly_40_videos_in_priority_order(self):
        items = validate_dataset(self.root)

        self.assertEqual(len(items), 40)
        self.assertEqual(
            [item.relative_path for item in items],
            EXPECTED_RELATIVE_ORDER,
        )
        self.assertTrue(all(len(item.sha256) == 64 for item in items))

    def test_inventory_rejects_missing_extra_empty_and_invalid_container(self):
        target = self.root / "Good" / "GOOD020.mp4"
        original = target.read_bytes()
        target.unlink()
        with self.assertRaises(DatasetValidationError):
            validate_dataset(self.root)
        target.write_bytes(original)

        extra = self.root / "Bad" / "EXTRA.mp4"
        extra.write_bytes(MP4_BYTES)
        with self.assertRaises(DatasetValidationError):
            validate_dataset(self.root)
        extra.unlink()

        target.write_bytes(b"")
        with self.assertRaises(DatasetValidationError):
            validate_dataset(self.root)
        target.write_bytes(b"not-an-mp4")
        with self.assertRaises(DatasetValidationError):
            validate_dataset(self.root)

    def test_max_two_is_good001_and_bad001(self):
        selected = select_videos(validate_dataset(self.root), 2)
        self.assertEqual(
            [item.relative_path for item in selected],
            ["Good/GOOD001.mp4", "Bad/BAD001.mp4"],
        )

    def test_max_six_and_twenty_five_follow_priority_prefixes(self):
        items = validate_dataset(self.root)

        self.assertEqual(
            [item.relative_path for item in select_videos(items, 6)],
            EXPECTED_RELATIVE_ORDER[:6],
        )
        self.assertEqual(
            [item.relative_path for item in select_videos(items, 25)],
            EXPECTED_RELATIVE_ORDER[:25],
        )
        self.assertEqual(
            [item.relative_path for item in select_videos(items, None)],
            EXPECTED_RELATIVE_ORDER,
        )
        self.assertEqual(
            {path for path in EXPECTED_RELATIVE_ORDER[:25] if path.startswith("Good/")},
            {f"Good/GOOD{index:03d}.mp4" for index in range(1, 6)},
        )
        self.assertEqual(
            {path for path in EXPECTED_RELATIVE_ORDER[:25] if path.startswith("Bad/")},
            {f"Bad/BAD{index:03d}.mp4" for index in range(1, 21)},
        )

    def test_dry_run_does_not_create_checkpoint_or_csv(self):
        state = Path(self._tmp.name) / "state.json"
        output = Path(self._tmp.name) / "results.csv"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "--dry-run",
                    "--max-videos",
                    "2",
                    "--dataset-root",
                    str(self.root),
                    "--state-file",
                    str(state),
                    "--csv-file",
                    str(output),
                    "--reports-dir",
                    str(self.reports_dir),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Good/GOOD001.mp4", stdout.getvalue())
        self.assertIn("Bad/BAD001.mp4", stdout.getvalue())
        self.assertFalse(state.exists())
        self.assertFalse(output.exists())
        self.assertFalse(self.reports_dir.exists())

    def test_legacy_checkpoint_resumes_in_new_order_without_duplicates(self):
        items = validate_dataset(self.root)
        state_path = Path(self._tmp.name) / "state.json"
        csv_path = Path(self._tmp.name) / "results.csv"
        client = FakeApiClient()
        user = {"id": 77, "email": "imanolquispe@hotmail.com"}

        execute_batch(
            client=client,
            user=user,
            all_items=items,
            selected_items=select_videos(items, 2),
            mode="beginner",
            state_path=state_path,
            csv_path=csv_path,
            reports_dir=self.reports_dir,
            resume=False,
            reconcile_job_id=None,
            confirm_no_job=False,
            poll_interval_seconds=0.001,
            job_timeout_seconds=1,
        )
        self.assertEqual(
            client.uploaded,
            ["Good/GOOD001.mp4", "Bad/BAD001.mp4"],
        )
        good_report = self.reports_dir / "Good" / "GOOD001.json"
        bad_report = self.reports_dir / "Bad" / "BAD001.json"
        self.assertTrue(good_report.is_file())
        self.assertTrue(bad_report.is_file())
        self.assertEqual(
            json.loads(good_report.read_text(encoding="utf-8")),
            client.details[1],
        )
        self.assertEqual(
            json.loads(bad_report.read_text(encoding="utf-8")),
            client.details[2],
        )
        initial_report_bytes = {
            good_report: good_report.read_bytes(),
            bad_report: bad_report.read_bytes(),
        }
        initial_detail_calls = list(client.detail_calls)

        checkpoint_before_resume = json.loads(state_path.read_text(encoding="utf-8"))
        records_by_path = {
            record["relative_path"]: record
            for record in checkpoint_before_resume["videos"]
        }
        legacy_interleaved_order = [
            path
            for index in range(1, 21)
            for path in (
                f"Good/GOOD{index:03d}.mp4",
                f"Bad/BAD{index:03d}.mp4",
            )
        ]
        checkpoint_before_resume["videos"] = [
            records_by_path[path] for path in legacy_interleaved_order
        ]
        state_path.write_text(
            json.dumps(checkpoint_before_resume, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        execute_batch(
            client=client,
            user=user,
            all_items=items,
            selected_items=items,
            mode="beginner",
            state_path=state_path,
            csv_path=csv_path,
            reports_dir=self.reports_dir,
            resume=True,
            reconcile_job_id=None,
            confirm_no_job=False,
            poll_interval_seconds=0.001,
            job_timeout_seconds=1,
        )
        self.assertEqual(len(client.uploaded), 40)
        self.assertEqual(len(set(client.uploaded)), 40)
        self.assertEqual(client.uploaded, EXPECTED_RELATIVE_ORDER)
        self.assertEqual(len(list(self.reports_dir.rglob("*.json"))), 40)
        self.assertEqual(client.detail_calls[:2], initial_detail_calls)
        self.assertEqual(client.detail_calls.count(1), 1)
        self.assertEqual(client.detail_calls.count(2), 1)
        for path, contents in initial_report_bytes.items():
            self.assertEqual(path.read_bytes(), contents)

        checkpoint_text = state_path.read_text(encoding="utf-8")
        self.assertNotIn("Bearer", checkpoint_text)
        self.assertNotIn("password", checkpoint_text.lower())
        checkpoint = json.loads(checkpoint_text)
        self.assertEqual(len(checkpoint["videos"]), 40)
        self.assertTrue(all(row["finalized"] for row in checkpoint["videos"]))

        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows[0]["system_classification"], "correct")
        self.assertEqual(rows[1]["system_classification"], "incorrect")

    def test_ambiguous_upload_blocks_resume_and_never_duplicates(self):
        items = validate_dataset(self.root)
        state_path = Path(self._tmp.name) / "state.json"
        csv_path = Path(self._tmp.name) / "results.csv"
        client = FakeApiClient()
        client.fail_ambiguously = True
        user = {"id": 77, "email": "imanolquispe@hotmail.com"}

        with self.assertRaises(AmbiguousSubmissionError):
            execute_batch(
                client=client,
                user=user,
                all_items=items,
                selected_items=select_videos(items, 2),
                mode="beginner",
                state_path=state_path,
                csv_path=csv_path,
                reports_dir=self.reports_dir,
                resume=False,
                reconcile_job_id=None,
                confirm_no_job=False,
                poll_interval_seconds=0.001,
                job_timeout_seconds=1,
            )
        self.assertEqual(len(client.uploaded), 1)
        client.fail_ambiguously = False
        with self.assertRaises(AmbiguousSubmissionError):
            execute_batch(
                client=client,
                user=user,
                all_items=items,
                selected_items=items,
                mode="beginner",
                state_path=state_path,
                csv_path=csv_path,
                reports_dir=self.reports_dir,
                resume=True,
                reconcile_job_id=None,
                confirm_no_job=False,
                poll_interval_seconds=0.001,
                job_timeout_seconds=1,
            )
        self.assertEqual(len(client.uploaded), 1)

    def test_ambiguous_upload_requires_explicit_matching_job_id(self):
        items = validate_dataset(self.root)
        state_path = Path(self._tmp.name) / "state.json"
        csv_path = Path(self._tmp.name) / "results.csv"
        client = FakeApiClient()
        client.fail_ambiguously = True
        user = {"id": 77, "email": "imanolquispe@hotmail.com"}
        with self.assertRaises(AmbiguousSubmissionError):
            execute_batch(
                client=client,
                user=user,
                all_items=items,
                selected_items=items[:1],
                mode="beginner",
                state_path=state_path,
                csv_path=csv_path,
                reports_dir=self.reports_dir,
                resume=False,
                reconcile_job_id=None,
                confirm_no_job=False,
                poll_interval_seconds=0.001,
                job_timeout_seconds=1,
            )

        client.fail_ambiguously = False
        client.jobs["recovered-job"] = {
            "job_id": "recovered-job",
            "status": "completed",
            "error": None,
            "video_seconds": 1.0,
            "assistant_seconds": 0.2,
        }
        recovered = {
            "id": 101,
            "job_id": "recovered-job",
            "mode": "beginner",
            "status": "completed",
            "correct": True,
            "total_reps": 1,
            "correct_reps": 1,
            "incorrect_reps": 0,
            "original_video_url": (
                "/api/jobs/recovered-job/files/GOOD001.mp4"
            ),
            "video_seconds": 1.0,
            "assistant_seconds": 0.2,
        }
        client.evaluations.append(recovered)
        client.details[101] = {**recovered, "report_json": {"reps": []}}
        execute_batch(
            client=client,
            user=user,
            all_items=items,
            selected_items=items[:1],
            mode="beginner",
            state_path=state_path,
            csv_path=csv_path,
            reports_dir=self.reports_dir,
            resume=True,
            reconcile_job_id="recovered-job",
            confirm_no_job=False,
            poll_interval_seconds=0.001,
            job_timeout_seconds=1,
        )
        self.assertEqual(client.uploaded, ["Good/GOOD001.mp4"])
        checkpoint = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["videos"][0]["job_id"], "recovered-job")
        self.assertTrue(checkpoint["videos"][0]["finalized"])
        self.assertTrue((self.reports_dir / "Good" / "GOOD001.json").is_file())

    def test_missing_and_corrupt_reports_are_recovered_without_reupload(self):
        items = validate_dataset(self.root)
        state_path = Path(self._tmp.name) / "state.json"
        csv_path = Path(self._tmp.name) / "results.csv"
        client = FakeApiClient()
        user = {"id": 77, "email": "imanolquispe@hotmail.com"}

        self._execute_one(client, user, items, state_path, csv_path, resume=False)
        report_path = self.reports_dir / "Good" / "GOOD001.json"
        self.assertTrue(report_path.is_file())
        self.assertEqual(client.uploaded, ["Good/GOOD001.mp4"])

        report_path.unlink()
        self._execute_one(client, user, items, state_path, csv_path, resume=True)
        self.assertEqual(client.uploaded, ["Good/GOOD001.mp4"])
        self.assertEqual(client.detail_calls.count(1), 2)
        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8")),
            client.details[1],
        )

        report_path.write_text('{"corrupt": true}', encoding="utf-8")
        self._execute_one(client, user, items, state_path, csv_path, resume=True)
        self.assertEqual(client.uploaded, ["Good/GOOD001.mp4"])
        self.assertEqual(client.detail_calls.count(1), 3)
        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8")),
            client.details[1],
        )

    def test_detail_failure_is_resumable_without_reupload(self):
        items = validate_dataset(self.root)
        state_path = Path(self._tmp.name) / "state.json"
        csv_path = Path(self._tmp.name) / "results.csv"
        client = FakeApiClient()
        client.fail_detail = True
        user = {"id": 77, "email": "imanolquispe@hotmail.com"}

        with self.assertRaisesRegex(BatchError, "detail endpoint failure"):
            self._execute_one(client, user, items, state_path, csv_path, resume=False)
        checkpoint = json.loads(state_path.read_text(encoding="utf-8"))
        record = checkpoint["videos"][0]
        self.assertEqual(record["job_id"], "job-001")
        self.assertEqual(record["evaluation_id"], 1)
        self.assertFalse(record["finalized"])
        self.assertFalse((self.reports_dir / "Good" / "GOOD001.json").exists())

        client.fail_detail = False
        self._execute_one(client, user, items, state_path, csv_path, resume=True)
        self.assertEqual(client.uploaded, ["Good/GOOD001.mp4"])
        self.assertTrue((self.reports_dir / "Good" / "GOOD001.json").is_file())

    def test_mismatched_detail_is_rejected_and_not_finalized(self):
        items = validate_dataset(self.root)
        state_path = Path(self._tmp.name) / "state.json"
        csv_path = Path(self._tmp.name) / "results.csv"
        client = FakeApiClient()
        client.wrong_detail_job_id = "another-job"
        user = {"id": 77, "email": "imanolquispe@hotmail.com"}

        with self.assertRaisesRegex(ApiProtocolError, "otro job"):
            self._execute_one(client, user, items, state_path, csv_path, resume=False)
        checkpoint = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertFalse(checkpoint["videos"][0]["finalized"])
        self.assertEqual(client.uploaded, ["Good/GOOD001.mp4"])
        self.assertFalse((self.reports_dir / "Good" / "GOOD001.json").exists())

    def test_new_run_refuses_to_overwrite_orphan_report(self):
        items = validate_dataset(self.root)
        report_path = self.reports_dir / "Good" / "GOOD001.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(BatchError, "reporte JSON"):
            execute_batch(
                client=FakeApiClient(),
                user={"id": 77, "email": "imanolquispe@hotmail.com"},
                all_items=items,
                selected_items=items[:1],
                mode="beginner",
                state_path=Path(self._tmp.name) / "state.json",
                csv_path=Path(self._tmp.name) / "results.csv",
                reports_dir=self.reports_dir,
                resume=False,
                reconcile_job_id=None,
                confirm_no_job=False,
                poll_interval_seconds=0.001,
                job_timeout_seconds=1,
            )

    def test_csv_preserves_three_classification_states(self):
        base = {
            "evaluation_status": "completed",
            "video_seconds": 1,
            "assistant_seconds": 0.5,
        }
        self.assertEqual(_csv_row({**base, "correct": True})["system_classification"], "correct")
        self.assertEqual(_csv_row({**base, "correct": False})["system_classification"], "incorrect")
        self.assertEqual(_csv_row({**base, "correct": None})["system_classification"], "unclassified")
        self.assertEqual(_csv_row(base)["processing_seconds"], "1.5")

    def test_public_http_is_rejected_but_tunnel_and_https_are_allowed(self):
        self.assertEqual(
            validate_base_url("http://127.0.0.1:18000"),
            "http://127.0.0.1:18000",
        )
        self.assertEqual(
            validate_base_url("https://api.example.test"),
            "https://api.example.test",
        )
        with self.assertRaises(BatchError):
            validate_base_url("http://34.83.250.8")

    def test_http_client_streams_expected_authenticated_multipart(self):
        video = validate_dataset(self.root)[0]
        connection = RecordingHttpConnection(
            FakeHttpResponse(200, b'{"job_id":"job-1","status":"queued"}')
        )
        client = ApiClient("http://127.0.0.1:18000")
        client._token = "memory-only-token"
        client._connection = lambda **kwargs: connection

        response = client.upload_video(video, "beginner")

        self.assertEqual(response["job_id"], "job-1")
        self.assertEqual(connection.method, "POST")
        self.assertEqual(connection.target, "/api/jobs")
        self.assertEqual(
            connection.headers["authorization"], "Bearer memory-only-token"
        )
        self.assertIn("multipart/form-data; boundary=", connection.headers["content-type"])
        self.assertEqual(int(connection.headers["content-length"]), len(connection.body))
        self.assertIn(b'name="mode"\r\n\r\nbeginner', connection.body)
        self.assertIn(
            b'name="video"; filename="GOOD001.mp4"', connection.body
        )
        self.assertIn(video.path.read_bytes(), connection.body)
        self.assertNotIn(b"memory-only-token", connection.body)
        self.assertTrue(connection.closed)

    def test_non_json_server_error_after_upload_is_ambiguous(self):
        video = validate_dataset(self.root)[0]
        connection = RecordingHttpConnection(
            FakeHttpResponse(502, b"Bad Gateway")
        )
        client = ApiClient("http://127.0.0.1:18000")
        client._token = "memory-only-token"
        client._connection = lambda **kwargs: connection

        with self.assertRaises(AmbiguousSubmissionError):
            client.upload_video(video, "beginner")

    def _create_dataset(self) -> None:
        for category, prefix in (("Good", "GOOD"), ("Bad", "BAD")):
            directory = self.root / category
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(1, 21):
                (directory / f"{prefix}{index:03d}.mp4").write_bytes(
                    MP4_BYTES + bytes([index])
                )

    def _execute_one(
        self,
        client,
        user,
        items,
        state_path,
        csv_path,
        *,
        resume: bool,
    ) -> None:
        execute_batch(
            client=client,
            user=user,
            all_items=items,
            selected_items=items[:1],
            mode="beginner",
            state_path=state_path,
            csv_path=csv_path,
            reports_dir=self.reports_dir,
            resume=resume,
            reconcile_job_id=None,
            confirm_no_job=False,
            poll_interval_seconds=0.001,
            job_timeout_seconds=1,
        )


if __name__ == "__main__":
    unittest.main()
