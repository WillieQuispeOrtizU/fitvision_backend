from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from squat_analysis_api.analyzer import SquatFrameAnalyzer
from squat_analysis_api.assistant_client import AssistantClientError
from squat_analysis_api.reporting import VideoMetadata, build_report, write_report
from squat_analysis_api.thresholds import get_thresholds
from squat_analysis_api.video_processor import (
    _apply_assistant_feedback,
    _save_key_frame_images,
)


class RepExtractionTest(unittest.TestCase):
    def test_correct_rep_event_from_synthetic_states(self):
        analyzer = SquatFrameAnalyzer(get_thresholds("beginner"))
        self._feed_states(
            analyzer,
            [
                ("s2", 20, 35, 20),
                ("s3", 25, 80, 20),
                ("s2", 20, 35, 20),
                ("s1", 20, 20, 20),
            ],
        )

        self.assertEqual(analyzer.stats.correct, 1)
        self.assertEqual(analyzer.stats.incorrect, 0)
        self.assertEqual(len(analyzer.reps), 1)
        rep = analyzer.reps[0]
        self.assertTrue(rep.correct)
        self.assertEqual(rep.bottom_frame, 1)
        self.assertEqual(rep.angles["knee"]["max"], 80.0)
        self.assertEqual(rep.issue_tags, [])

    def test_incorrect_rep_records_issue_tags(self):
        analyzer = SquatFrameAnalyzer(get_thresholds("beginner"))
        self._feed_states(
            analyzer,
            [
                ("s2", 20, 35, 20),
                ("s3", 25, 80, 50),
                ("s2", 20, 35, 20),
                ("s1", 20, 20, 20),
            ],
        )

        self.assertEqual(analyzer.stats.correct, 0)
        self.assertEqual(analyzer.stats.incorrect, 1)
        rep = analyzer.reps[0]
        self.assertFalse(rep.correct)
        self.assertIn("knee_over_toe", rep.issue_tags)
        self.assertEqual(rep.worst_issue, "knee_over_toe")

    def test_bend_forward_issue_makes_complete_rep_incorrect(self):
        analyzer = SquatFrameAnalyzer(get_thresholds("beginner"))
        self._feed_states(
            analyzer,
            [
                ("s2", 5, 35, 20),
                ("s3", 25, 80, 20),
                ("s2", 20, 35, 20),
                ("s1", 20, 20, 20),
            ],
        )

        self.assertEqual(analyzer.stats.correct, 0)
        self.assertEqual(analyzer.stats.incorrect, 1)
        self.assertFalse(analyzer.reps[0].correct)
        self.assertIn("bend_forward", analyzer.reps[0].issue_tags)

    def test_torso_within_threshold_does_not_mark_bend_forward(self):
        analyzer = SquatFrameAnalyzer(get_thresholds("beginner"))
        analyzer.state_tracker["state_seq"] = ["s2"]

        issues = analyzer._update_feedback(20, 35, 20)

        self.assertNotIn("bend_forward", issues)
        self.assertFalse(analyzer.state_tracker["INCORRECT_POSTURE"])

    def test_transient_feedback_can_be_cleared_between_repetitions(self):
        analyzer = SquatFrameAnalyzer(get_thresholds("beginner"))
        analyzer.state_tracker["DISPLAY_TEXT"].fill(True)
        analyzer.state_tracker["COUNT_FRAMES"].fill(7)

        analyzer._clear_transient_feedback()

        self.assertFalse(analyzer.state_tracker["DISPLAY_TEXT"].any())
        self.assertFalse(analyzer.state_tracker["COUNT_FRAMES"].any())

    def _feed_states(self, analyzer, states):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        for idx, (state, hip, knee, ankle) in enumerate(states):
            analyzer.state_tracker["curr_state"] = state
            analyzer._update_state_sequence(state)
            issues = analyzer._update_feedback(hip, knee, ankle)
            analyzer._track_rep_frame(
                frame=frame,
                frame_index=idx,
                timestamp=float(idx),
                state=state,
                hip_angle=float(hip),
                knee_angle=float(knee),
                ankle_angle=float(ankle),
                issues=issues,
            )
            completed = analyzer._update_counters(state)
            if completed is not None:
                analyzer._finish_current_rep(
                    correct=completed,
                    frame=frame,
                    frame_index=idx,
                    timestamp=float(idx),
                )


class ReportWritingTest(unittest.TestCase):
    def test_report_and_key_frames_are_written(self):
        analyzer = SquatFrameAnalyzer(get_thresholds("beginner"))
        RepExtractionTest()._feed_states(
            analyzer,
            [
                ("s2", 20, 35, 20),
                ("s3", 25, 80, 20),
                ("s2", 20, 35, 20),
                ("s1", 20, 20, 20),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reps = analyzer.reps
            _save_key_frame_images(
                key_frame_images=analyzer.key_frame_images,
                reps=reps,
                frame_dir=tmp_path / "frames",
            )
            report = build_report(
                video=VideoMetadata(
                    input_path="input.mp4",
                    output_path="output.mp4",
                    fps=30.0,
                    width=640,
                    height=480,
                    frame_count=120,
                    duration=4.0,
                ),
                stats=analyzer.stats,
                reps=reps,
            )
            report_path = tmp_path / "analysis.json"
            write_report(report_path, report)

            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["stats"]["total_reps"], 1)
            self.assertTrue(Path(saved["reps"][0]["key_frames"]["bottom"]).exists())
            self.assertIsNotNone(cv2.imread(saved["reps"][0]["key_frames"]["bottom"]))


class AssistantFallbackTest(unittest.TestCase):
    def test_assistant_outage_uses_deterministic_feedback_without_losing_report(self):
        report = {
            "video": {"input_path": "input.mp4", "output_path": "output.mp4"},
            "stats": {"correct": 0, "incorrect": 1, "total_reps": 1},
            "reps": [
                {
                    "rep_id": 1,
                    "correct": False,
                    "issue_tags": ["future_issue"],
                    "assistant_feedback": None,
                }
            ],
        }
        client = Mock()
        client.analyze.side_effect = AssistantClientError("assistant unavailable")

        with self.assertLogs(
            "squat_analysis_api.video_processor", level="WARNING"
        ):
            result = _apply_assistant_feedback(
                report,
                client=client,
                service_url="http://assistant.test",
            )

        self.assertEqual(result["stats"]["incorrect"], 1)
        self.assertEqual(result["assistant"]["status"], "fallback")
        self.assertEqual(result["assistant"]["errors"], ["assistant unavailable"])
        self.assertIsNone(result["assistant"]["latency_seconds"])
        self.assertTrue(result["reps"][0]["assistant_feedback"]["cues"])


if __name__ == "__main__":
    unittest.main()

