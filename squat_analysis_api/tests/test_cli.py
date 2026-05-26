from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from squat_analysis_api.analyzer import SquatStats
from squat_analysis_api.assistant_client import AssistantClientError
from squat_analysis_api.cli import main
from squat_analysis_api.video_processor import VideoResult


class CliTest(unittest.TestCase):
    @patch("squat_analysis_api.cli.process_video")
    @patch("squat_analysis_api.cli.discover_videos")
    def test_default_processing_remains_supported(self, discover_videos, process_video):
        discover_videos.return_value = [Path("input_video/a.mp4")]
        process_video.return_value = VideoResult(
            input_path=Path("input_video/a.mp4"),
            output_path=Path("output_video/a_analyzed.mp4"),
            frames=10,
            stats=SquatStats(correct=1, incorrect=0),
        )

        exit_code = main([])

        self.assertEqual(exit_code, 0)
        process_video.assert_called_once()
        self.assertFalse(process_video.call_args.kwargs["write_json_report"])
        self.assertIsNone(process_video.call_args.kwargs["assistant_url"])

    @patch("squat_analysis_api.cli.process_video")
    def test_report_flag_writes_report(self, process_video):
        process_video.return_value = VideoResult(
            input_path=Path("input_video/a.mp4"),
            output_path=Path("output_video/a_analyzed.mp4"),
            frames=10,
            stats=SquatStats(correct=1, incorrect=0),
            report_path=Path("output_video/a_analysis.json"),
        )

        exit_code = main(["--file", "input_video/a.mp4", "--report"])

        self.assertEqual(exit_code, 0)
        self.assertTrue(process_video.call_args.kwargs["write_json_report"])

    @patch("squat_analysis_api.cli.process_video")
    @patch("squat_analysis_api.cli.AssistantClient")
    def test_assistant_unavailable_fails_before_processing(self, assistant_client_cls, process_video):
        assistant = Mock()
        assistant.health.side_effect = AssistantClientError("down")
        assistant_client_cls.return_value = assistant

        exit_code = main(["--file", "input_video/a.mp4", "--assistant-url", "http://127.0.0.1:7862"])

        self.assertEqual(exit_code, 1)
        process_video.assert_not_called()


if __name__ == "__main__":
    unittest.main()

