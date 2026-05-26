from __future__ import annotations

import unittest

from squat_analysis_api.assistant_server import analyze_squat_payload


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


if __name__ == "__main__":
    unittest.main()

