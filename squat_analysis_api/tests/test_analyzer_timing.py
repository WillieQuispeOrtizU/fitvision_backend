from __future__ import annotations

import time
import unittest

from squat_analysis_api.analyzer import SquatFrameAnalyzer
from squat_analysis_api.thresholds import get_thresholds


def _build_analyzer() -> SquatFrameAnalyzer:
    return SquatFrameAnalyzer(get_thresholds("beginner"))


class InactivityTimingTest(unittest.TestCase):
    """The inactivity reset must be driven by the video timestamp (self._now),
    never by the wall clock, so analysis is deterministic across machines."""

    def test_reset_fires_on_video_time_not_wall_time(self):
        analyzer = _build_analyzer()
        analyzer.state_tracker["curr_state"] = "s1"
        analyzer.state_tracker["prev_state"] = "s1"
        analyzer.state_tracker["SQUAT_COUNT"] = 5
        analyzer.state_tracker["IMPROPER_SQUAT"] = 2

        # First tick at t=0: no elapsed video time, so no reset.
        analyzer._now = 0.0
        self.assertFalse(analyzer._update_inactivity())

        # Jump video time past the inactivity threshold: reset fires.
        analyzer._now = analyzer.thresholds["INACTIVE_THRESH"] + 1.0
        self.assertTrue(analyzer._update_inactivity())
        self.assertEqual(analyzer.state_tracker["SQUAT_COUNT"], 0)
        self.assertEqual(analyzer.state_tracker["IMPROPER_SQUAT"], 0)

    def test_wall_clock_does_not_trigger_reset(self):
        analyzer = _build_analyzer()
        analyzer.state_tracker["curr_state"] = "s1"
        analyzer.state_tracker["prev_state"] = "s1"
        analyzer.state_tracker["SQUAT_COUNT"] = 3

        # Hold video time constant while real time elapses; the threshold must
        # never be crossed no matter how long the wall clock runs.
        analyzer._now = 1.0
        start = time.perf_counter()
        while time.perf_counter() - start < 0.05:
            self.assertFalse(analyzer._update_inactivity())
        self.assertEqual(analyzer.state_tracker["SQUAT_COUNT"], 3)


class AuthoritativeStatsTest(unittest.TestCase):
    """stats must come from the completed reps, so a live-overlay counter reset
    cannot corrupt the totals written to the report."""

    def _feed_one_correct_rep(self, analyzer: SquatFrameAnalyzer) -> None:
        import numpy as np

        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        states = [
            ("s2", 20, 35, 20),
            ("s3", 25, 80, 20),
            ("s2", 20, 35, 20),
            ("s1", 20, 20, 20),
        ]
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
                analyzer._finish_current_rep(correct=completed, frame=frame, frame_index=idx, timestamp=float(idx))

    def test_stats_survive_live_counter_reset(self):
        analyzer = _build_analyzer()
        self._feed_one_correct_rep(analyzer)
        self.assertEqual(analyzer.stats.correct, 1)

        # Simulate the inactivity overlay zeroing the live counters mid-video.
        analyzer.state_tracker["SQUAT_COUNT"] = 0
        analyzer.state_tracker["IMPROPER_SQUAT"] = 0

        # The report-facing total must still reflect the completed rep.
        self.assertEqual(analyzer.stats.correct, 1)
        self.assertEqual(analyzer.stats.incorrect, 0)


if __name__ == "__main__":
    unittest.main()
