from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from squat_analysis_api.analyzer import SquatFrameAnalyzer
from squat_analysis_api.reporting import VideoMetadata, build_report, write_report
from squat_analysis_api.thresholds import get_thresholds
from squat_analysis_api.video_processor import _save_key_frame_images


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


if __name__ == "__main__":
    unittest.main()

