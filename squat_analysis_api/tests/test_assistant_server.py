from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from squat_analysis_api.assistant_server import _select_key_frames, analyze_squat_payload


class _FakeFalcon:
    model_id = "fake-falcon"

    def detect(self, frame_paths):
        return [
            {
                "frame_path": frame_paths[0] if frame_paths else "none",
                "query": "person",
                "bbox": {"xy": {"x": 0.5, "y": 0.5}, "hw": {"h": 0.8, "w": 0.4}},
                "confidence": None,
            }
        ]


class _FakeGemma:
    model_id = "fake-gemma"

    def generate(self, payload, detections):
        return {
            "overall_summary": "One rep reviewed.",
            "rep_feedback": [
                {
                    "rep_id": 1,
                    "summary": "Rep reviewed.",
                    "cues": ["Brace."],
                    "severity": "info",
                }
            ],
        }


class AssistantServerTest(unittest.TestCase):
    def test_analyze_squat_payload_uses_adapters(self):
        payload = {
            "stats": {"total_reps": 1, "correct": 1, "incorrect": 0},
            "reps": [{"rep_id": 1, "key_frames": {"bottom": "/missing.jpg"}}],
        }

        result = analyze_squat_payload(payload, falcon_adapter=_FakeFalcon(), gemma_adapter=_FakeGemma())

        self.assertEqual(result["overall_summary"], "One rep reviewed.")
        self.assertEqual(result["model_names"]["falcon"], "fake-falcon")
        self.assertEqual(result["model_names"]["gemma"], "fake-gemma")
        self.assertEqual(result["rep_feedback"][0]["rep_id"], 1)


class SelectKeyFramesTest(unittest.TestCase):
    def test_prefers_worst_issue_dedupes_and_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in ("start", "bottom", "worst", "r2bottom"):
                path = root / f"{name}.jpg"
                path.write_bytes(b"x")
                paths[name] = str(path)

            reps = [
                {"key_frames": {"start": paths["start"], "bottom": paths["bottom"], "worst_issue": paths["worst"]}},
                {"key_frames": {"start": paths["start"], "bottom": paths["r2bottom"]}},
            ]

            # limit=1 keeps only the first rep's most informative (worst_issue) frame.
            self.assertEqual(_select_key_frames(reps, 1), [paths["worst"]])
            # limit=5 falls back to bottom for rep 2 and skips the duplicate start frame.
            self.assertEqual(_select_key_frames(reps, 5), [paths["worst"], paths["r2bottom"]])

    def test_skips_missing_paths(self):
        reps = [{"key_frames": {"bottom": "/does/not/exist.jpg"}}]
        self.assertEqual(_select_key_frames(reps, 4), [])


if __name__ == "__main__":
    unittest.main()

