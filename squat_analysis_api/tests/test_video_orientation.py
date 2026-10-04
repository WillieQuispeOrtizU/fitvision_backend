from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import cv2
import numpy as np

from squat_analysis_api.analyzer import SquatStats
from squat_analysis_api import video_processor


class RotationMetadataParsingTest(unittest.TestCase):
    def test_no_rotation_metadata_is_zero_regardless_of_dimensions(self):
        for width, height in ((1080, 1920), (1920, 1080)):
            with self.subTest(width=width, height=height):
                self.assertEqual(
                    video_processor._rotation_from_probe_payload(
                        {"streams": [{"width": width, "height": height}]}
                    ),
                    0,
                )

    def test_display_matrix_rotation_90_is_detected(self):
        payload = {
            "streams": [
                {
                    "side_data_list": [
                        {"side_data_type": "Display Matrix", "rotation": 90}
                    ]
                }
            ]
        }

        self.assertEqual(video_processor._rotation_from_probe_payload(payload), 90)

    def test_rotate_tag_180_is_detected(self):
        payload = {"streams": [{"tags": {"rotate": "180"}}]}

        self.assertEqual(video_processor._rotation_from_probe_payload(payload), 180)

    def test_rotate_tag_270_is_detected(self):
        payload = {"streams": [{"tags": {"rotate": "270"}}]}

        self.assertEqual(video_processor._rotation_from_probe_payload(payload), 270)

    def test_negative_90_is_canonicalized_and_side_data_has_precedence(self):
        payload = {
            "streams": [
                {
                    "side_data_list": [
                        {"side_data_type": "Display Matrix", "rotation": -90.0}
                    ],
                    "tags": {"rotate": "90"},
                }
            ]
        }

        self.assertEqual(video_processor._rotation_from_probe_payload(payload), 270)


class RotationProbeTest(unittest.TestCase):
    def test_missing_ffprobe_warns_and_falls_back(self):
        input_path = Path("mobile.mp4")
        with (
            patch.object(video_processor.shutil, "which", return_value=None),
            patch.object(video_processor.subprocess, "run") as run,
            self.assertLogs(video_processor.logger, level="WARNING") as logs,
        ):
            rotation = video_processor._probe_video_rotation(input_path)

        self.assertIsNone(rotation)
        run.assert_not_called()
        self.assertIn("ffprobe is unavailable", "\n".join(logs.output))

    def test_failed_ffprobe_warns_and_falls_back_without_shell(self):
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=1,
            stdout="",
            stderr="probe exploded",
        )
        with (
            patch.object(video_processor.shutil, "which", return_value="ffprobe"),
            patch.object(
                video_processor.subprocess, "run", return_value=completed
            ) as run,
            self.assertLogs(video_processor.logger, level="WARNING") as logs,
        ):
            rotation = video_processor._probe_video_rotation(Path("mobile.mp4"))

        self.assertIsNone(rotation)
        command = run.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertIn("probe exploded", "\n".join(logs.output))

    def test_invalid_ffprobe_json_warns_and_falls_back(self):
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout="not-json",
            stderr="",
        )
        with (
            patch.object(video_processor.shutil, "which", return_value="ffprobe"),
            patch.object(video_processor.subprocess, "run", return_value=completed),
            self.assertLogs(video_processor.logger, level="WARNING") as logs,
        ):
            rotation = video_processor._probe_video_rotation(Path("mobile.mp4"))

        self.assertIsNone(rotation)
        self.assertIn("unusable orientation data", "\n".join(logs.output))

    def test_probe_reads_display_matrix_from_json(self):
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "side_data_list": [
                                {
                                    "side_data_type": "Display Matrix",
                                    "rotation": 90,
                                }
                            ]
                        }
                    ]
                }
            ),
            stderr="",
        )
        with (
            patch.object(video_processor.shutil, "which", return_value="ffprobe"),
            patch.object(
                video_processor.subprocess, "run", return_value=completed
            ) as run,
        ):
            rotation = video_processor._probe_video_rotation(Path("mobile.mp4"))

        self.assertEqual(rotation, 90)
        command = run.call_args.args[0]
        self.assertIn("-show_streams", command)
        self.assertEqual(command[-1], "mobile.mp4")
        self.assertFalse(run.call_args.kwargs["shell"])


class NormalizedVideoContextTest(unittest.TestCase):
    def test_zero_rotation_does_not_create_or_transcode_an_intermediate(self):
        input_path = Path("already-upright.mp4")
        with (
            patch.object(video_processor, "_probe_video_rotation", return_value=0),
            patch.object(video_processor.shutil, "which") as which,
            patch.object(
                video_processor, "_normalize_video_orientation"
            ) as normalize,
        ):
            with video_processor._normalized_video_for_analysis(input_path) as path:
                self.assertEqual(path, input_path)

        which.assert_not_called()
        normalize.assert_not_called()

    def test_missing_ffmpeg_warns_and_uses_original(self):
        input_path = Path("rotated.mp4")
        with (
            patch.object(video_processor, "_probe_video_rotation", return_value=90),
            patch.object(video_processor.shutil, "which", return_value=None),
            self.assertLogs(video_processor.logger, level="WARNING") as logs,
        ):
            with video_processor._normalized_video_for_analysis(input_path) as path:
                self.assertEqual(path, input_path)

        self.assertIn("ffmpeg is unavailable", "\n".join(logs.output))

    def test_normalized_temporary_is_removed_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "original.mp4"
            input_path.write_bytes(b"original")

            def normalize(_input, output, **_kwargs):
                output.write_bytes(b"normalized")
                return True

            with (
                patch.object(
                    video_processor, "_probe_video_rotation", return_value=90
                ),
                patch.object(
                    video_processor.shutil, "which", return_value="ffmpeg"
                ),
                patch.object(
                    video_processor,
                    "_normalize_video_orientation",
                    side_effect=normalize,
                ),
            ):
                with video_processor._normalized_video_for_analysis(
                    input_path
                ) as normalized_path:
                    temporary_parent = normalized_path.parent
                    self.assertNotEqual(normalized_path, input_path)
                    self.assertEqual(normalized_path.read_bytes(), b"normalized")
                    self.assertTrue(temporary_parent.exists())

                self.assertFalse(normalized_path.exists())
                self.assertFalse(temporary_parent.exists())
                self.assertEqual(input_path.read_bytes(), b"original")

    def test_normalized_temporary_is_removed_when_consumer_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "original.mp4"
            input_path.write_bytes(b"original")
            yielded_path: Path | None = None

            def normalize(_input, output, **_kwargs):
                output.write_bytes(b"normalized")
                return True

            with (
                patch.object(
                    video_processor, "_probe_video_rotation", return_value=270
                ),
                patch.object(
                    video_processor.shutil, "which", return_value="ffmpeg"
                ),
                patch.object(
                    video_processor,
                    "_normalize_video_orientation",
                    side_effect=normalize,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                    with video_processor._normalized_video_for_analysis(
                        input_path
                    ) as path:
                        yielded_path = path
                        raise RuntimeError("consumer failed")

            self.assertIsNotNone(yielded_path)
            self.assertFalse(yielded_path.exists())
            self.assertEqual(input_path.read_bytes(), b"original")

    def test_ffmpeg_failure_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "original.mp4"
            output_path = Path(tmp) / "normalized.mp4"
            input_path.write_bytes(b"original")

            def fail(command, **_kwargs):
                Path(command[-1]).write_bytes(b"partial")
                return subprocess.CompletedProcess(
                    command, 1, stdout=b"", stderr=b"encode failed"
                )

            with (
                patch.object(video_processor.subprocess, "run", side_effect=fail) as run,
                self.assertLogs(video_processor.logger, level="WARNING") as logs,
            ):
                succeeded = video_processor._normalize_video_orientation(
                    input_path,
                    output_path,
                    ffmpeg="ffmpeg",
                    rotation=180,
                )

            self.assertFalse(succeeded)
            self.assertFalse(output_path.exists())
            self.assertEqual(input_path.read_bytes(), b"original")
            self.assertFalse(run.call_args.kwargs["shell"])
            self.assertIn("encode failed", "\n".join(logs.output))

    def test_successful_normalization_uses_autorotate_and_clears_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "original.mp4"
            output_path = Path(tmp) / "normalized.mp4"
            input_path.write_bytes(b"original")

            def succeed(command, **_kwargs):
                Path(command[-1]).write_bytes(b"normalized")
                return subprocess.CompletedProcess(
                    command, 0, stdout=b"", stderr=b""
                )

            with patch.object(
                video_processor.subprocess, "run", side_effect=succeed
            ) as run:
                succeeded = video_processor._normalize_video_orientation(
                    input_path,
                    output_path,
                    ffmpeg="ffmpeg",
                    rotation=90,
                )

            self.assertTrue(succeeded)
            self.assertEqual(output_path.read_bytes(), b"normalized")
            command = run.call_args.args[0]
            self.assertLess(command.index("-autorotate"), command.index("-i"))
            self.assertEqual(command[command.index("-map_metadata") + 1], "-1")
            self.assertEqual(
                command[command.index("-metadata:s:v:0") + 1], "rotate=0"
            )
            self.assertFalse(run.call_args.kwargs["shell"])


class ProcessVideoOrientationTest(unittest.TestCase):
    def test_process_uses_normalized_path_dimensions_and_report_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "original.mp4"
            normalized_path = root / "normalized.mp4"
            output_path = root / "output" / "analyzed.mp4"
            report_path = root / "output" / "analysis.json"
            input_path.write_bytes(b"original upload")
            normalized_path.write_bytes(b"normalized pixels")

            capture = MagicMock()
            capture.isOpened.return_value = True
            capture.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FRAME_WIDTH: 720.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 1280.0,
            }[prop]
            frame = np.zeros((1280, 720, 3), dtype=np.uint8)
            capture.read.side_effect = [(True, frame), (True, frame), (False, None)]

            writer = MagicMock()
            writer.isOpened.return_value = True

            class FakeAnalyzer:
                stats = SquatStats(correct=0, incorrect=0)
                reps: list[object] = []
                key_frame_images: dict[int, dict[str, object]] = {}

                def process(self, rgb_frame, _pose, **_kwargs):
                    return rgb_frame

            analyzer = FakeAnalyzer()

            @contextmanager
            def normalized_context(_input_path):
                yield normalized_path

            with (
                patch.object(
                    video_processor,
                    "_normalized_video_for_analysis",
                    side_effect=normalized_context,
                ) as normalized,
                patch.object(
                    video_processor.cv2, "VideoCapture", return_value=capture
                ) as video_capture,
                patch.object(
                    video_processor.cv2, "VideoWriter", return_value=writer
                ) as video_writer,
                patch.object(
                    video_processor, "SquatFrameAnalyzer", return_value=analyzer
                ),
                patch.object(
                    video_processor,
                    "get_mediapipe_pose",
                    return_value=nullcontext(object()),
                ),
                patch.object(video_processor, "_transcode_to_h264") as transcode,
            ):
                result = video_processor.process_video(
                    input_path,
                    output_path,
                    write_json_report=True,
                    report_path=report_path,
                )

            normalized.assert_called_once_with(input_path)
            video_capture.assert_called_once_with(str(normalized_path))
            video_writer.assert_called_once_with(
                str(output_path), ANY, 25.0, (720, 1280)
            )
            self.assertEqual(writer.write.call_count, 2)
            capture.release.assert_called_once()
            writer.release.assert_called_once()
            transcode.assert_called_once_with(output_path)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["video"]["input_path"], str(input_path))
            self.assertEqual(report["video"]["output_path"], str(output_path))
            self.assertEqual(report["video"]["fps"], 25.0)
            self.assertEqual(report["video"]["width"], 720)
            self.assertEqual(report["video"]["height"], 1280)
            self.assertEqual(report["video"]["frame_count"], 2)
            self.assertAlmostEqual(report["video"]["duration"], 0.08)
            self.assertEqual(result.input_path, input_path)
            self.assertEqual(result.frames, 2)
            self.assertEqual(input_path.read_bytes(), b"original upload")


class AnnotatedVideoTranscodeTest(unittest.TestCase):
    def test_transcode_neutralizes_rotation_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzed.mp4"
            path.write_bytes(b"mp4v")

            def succeed(command, **_kwargs):
                Path(command[-1]).write_bytes(b"h264")
                return subprocess.CompletedProcess(
                    command, 0, stdout=b"", stderr=b""
                )

            with (
                patch.object(video_processor.shutil, "which", return_value="ffmpeg"),
                patch.object(
                    video_processor.subprocess, "run", side_effect=succeed
                ) as run,
            ):
                video_processor._transcode_to_h264(path)

            self.assertEqual(path.read_bytes(), b"h264")
            command = run.call_args.args[0]
            self.assertLess(command.index("-noautorotate"), command.index("-i"))
            self.assertEqual(command[command.index("-map_metadata") + 1], "-1")
            self.assertEqual(
                command[command.index("-metadata:s:v:0") + 1], "rotate=0"
            )
            self.assertFalse(run.call_args.kwargs["shell"])
            self.assertFalse(path.with_name("analyzed.h264.mp4").exists())

    def test_failed_transcode_keeps_annotated_video_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzed.mp4"
            path.write_bytes(b"mp4v")

            def fail(command, **_kwargs):
                Path(command[-1]).write_bytes(b"partial")
                return subprocess.CompletedProcess(
                    command, 1, stdout=b"", stderr=b"transcode failed"
                )

            with (
                patch.object(video_processor.shutil, "which", return_value="ffmpeg"),
                patch.object(video_processor.subprocess, "run", side_effect=fail),
                self.assertLogs(video_processor.logger, level="WARNING") as logs,
            ):
                video_processor._transcode_to_h264(path)

            self.assertEqual(path.read_bytes(), b"mp4v")
            self.assertFalse(path.with_name("analyzed.h264.mp4").exists())
            self.assertIn("transcode failed", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
