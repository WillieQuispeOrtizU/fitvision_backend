from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2

from squat_analysis_api.assistant_client import AssistantClient, apply_assistant_response
from squat_analysis_api.analyzer import SquatFrameAnalyzer, SquatStats
from squat_analysis_api.reporting import VideoMetadata, build_report, empty_assistant_status, write_report
from squat_analysis_api.thresholds import get_thresholds
from squat_analysis_api.utils import get_mediapipe_pose


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoResult:
    input_path: Path
    output_path: Path
    frames: int
    stats: SquatStats
    report_path: Path | None = None


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_analyzed.mp4"


def report_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_analysis.json"


def discover_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")


def process_video(
    input_path: Path,
    output_path: Path,
    mode: str = "beginner",
    *,
    write_json_report: bool = False,
    assistant_url: str | None = None,
    assistant_client: AssistantClient | None = None,
    report_path: Path | None = None,
    frame_dir: Path | None = None,
) -> VideoResult:
    # Mobile videos may store landscape pixels plus a display matrix instead
    # of physically rotating the frames. Materialize that presentation before
    # OpenCV/MediaPipe so angle calculations use the visual upright image.
    with _normalized_video_for_analysis(input_path) as analysis_path:
        capture = cv2.VideoCapture(str(analysis_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open input video: {input_path}")

        writer = None
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise RuntimeError(f"Could not read video dimensions: {input_path}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")

            analyzer = SquatFrameAnalyzer(thresholds=get_thresholds(mode))
            frame_count = 0

            with get_mediapipe_pose() as pose:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = frame_count / fps if fps > 0 else 0.0
                    processed_rgb = analyzer.process(
                        rgb_frame,
                        pose,
                        frame_index=frame_count,
                        timestamp=timestamp,
                    )
                    writer.write(cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR))
                    frame_count += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

    # OpenCV writes mp4v (MPEG-4 Part 2), which browsers will not play inline.
    # Re-encode to H.264 so the annotated video plays in the web frontend.
    _transcode_to_h264(output_path)

    result_report_path = report_path
    if write_json_report or assistant_url:
        result_report_path = result_report_path or report_path_for(input_path, output_path.parent)
        reps = analyzer.reps
        _save_key_frame_images(
            key_frame_images=analyzer.key_frame_images,
            reps=reps,
            frame_dir=frame_dir or output_path.parent / f"{input_path.stem}_frames",
        )
        video_metadata = VideoMetadata(
            input_path=str(input_path),
            output_path=str(output_path),
            fps=float(fps),
            width=width,
            height=height,
            frame_count=frame_count,
            duration=frame_count / fps if fps > 0 else 0.0,
        )
        report = build_report(
            video=video_metadata,
            stats=analyzer.stats,
            reps=reps,
            assistant=empty_assistant_status(enabled=bool(assistant_url), service_url=assistant_url),
        )
        if assistant_url:
            client = assistant_client or AssistantClient(assistant_url)
            response, latency = client.analyze(report)
            report = apply_assistant_response(
                report,
                response,
                service_url=assistant_url,
                latency_seconds=latency,
            )
        write_report(result_report_path, report)

    return VideoResult(
        input_path=input_path,
        output_path=output_path,
        frames=frame_count,
        stats=analyzer.stats,
        report_path=result_report_path,
    )


@contextmanager
def _normalized_video_for_analysis(input_path: Path) -> Iterator[Path]:
    """Yield pixels in their intended visual orientation for OpenCV.

    The uploaded file is read-only. A temporary H.264 copy is created only
    when ffprobe reports non-zero presentation rotation. Any probe/encode
    failure degrades to the legacy behavior and is logged explicitly.
    """

    rotation = _probe_video_rotation(input_path)
    if rotation is None or rotation == 0:
        yield input_path
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning(
            "ffmpeg is unavailable; analyzing video without applying %s-degree "
            "orientation metadata: %s",
            rotation,
            input_path,
        )
        yield input_path
        return

    try:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="fitvision-orientation-"
        )
    except OSError as exc:
        logger.warning(
            "Could not create an orientation-normalization directory for %s: %s; "
            "using the original video",
            input_path,
            exc,
        )
        yield input_path
        return

    with temporary_directory as directory:
        normalized_path = Path(directory) / "normalized.mp4"
        if not _normalize_video_orientation(
            input_path,
            normalized_path,
            ffmpeg=ffmpeg,
            rotation=rotation,
        ):
            yield input_path
            return

        logger.info(
            "Using physically normalized video for analysis (rotation=%s): %s",
            rotation,
            input_path,
        )
        yield normalized_path


def _probe_video_rotation(input_path: Path) -> int | None:
    """Return presentation rotation normalized to ``0..359`` degrees.

    Modern mobile files expose rotation through Display Matrix side data;
    older files commonly use a ``rotate`` stream tag. Side data is treated as
    authoritative when both are present. ``None`` means probing was not
    possible and callers should preserve the previous OpenCV behavior.
    """

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        logger.warning(
            "ffprobe is unavailable; video orientation metadata cannot be "
            "inspected: %s",
            input_path,
        )
        return None

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            shell=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "ffprobe could not inspect video orientation for %s: %s; using the "
            "original video",
            input_path,
            exc,
        )
        return None

    if result.returncode != 0:
        logger.warning(
            "ffprobe failed while inspecting video orientation for %s%s; using "
            "the original video",
            input_path,
            _subprocess_detail(result.stderr),
        )
        return None

    try:
        payload = json.loads(result.stdout)
        return _rotation_from_probe_payload(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "ffprobe returned unusable orientation data for %s: %s; using the "
            "original video",
            input_path,
            exc,
        )
        return None


def _rotation_from_probe_payload(payload: object) -> int:
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("no video stream was reported")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise ValueError("invalid video stream data")

    malformed_rotation = False
    side_data_list = stream.get("side_data_list")
    if isinstance(side_data_list, list):
        for side_data in side_data_list:
            if not isinstance(side_data, dict) or "rotation" not in side_data:
                continue
            rotation = _parse_rotation(side_data["rotation"])
            if rotation is not None:
                return rotation
            malformed_rotation = True

    tags = stream.get("tags")
    if isinstance(tags, dict):
        for key, value in tags.items():
            if str(key).lower() != "rotate":
                continue
            rotation = _parse_rotation(value)
            if rotation is not None:
                return rotation
            malformed_rotation = True

    if malformed_rotation:
        raise ValueError("rotation metadata is not numeric")
    return 0


def _parse_rotation(value: object) -> int | None:
    try:
        rotation = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rotation):
        return None
    return int(round(rotation)) % 360


def _normalize_video_orientation(
    input_path: Path,
    output_path: Path,
    *,
    ffmpeg: str,
    rotation: int,
) -> bool:
    """Materialize FFmpeg's presentation transform into encoded pixels."""

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-map_metadata",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            shell=False,
        )
    except OSError as exc:
        _remove_temporary_file(output_path, description="partial normalized video")
        logger.warning(
            "ffmpeg could not normalize %s-degree orientation for %s: %s; "
            "using the original video",
            rotation,
            input_path,
            exc,
        )
        return False

    if result.returncode == 0 and _is_nonempty_file(output_path):
        return True

    _remove_temporary_file(output_path, description="normalized video")
    logger.warning(
        "ffmpeg failed to normalize %s-degree orientation for %s%s; using the "
        "original video",
        rotation,
        input_path,
        _subprocess_detail(result.stderr),
    )
    return False


def _transcode_to_h264(path: Path) -> None:
    """Re-encode an mp4 in place to H.264/yuv420p with a web-streaming layout.

    OpenCV's mp4v output is MPEG-4 Part 2, which browsers refuse to play inline.
    Requires ffmpeg on PATH; if it is missing or the encode fails, the original
    file is left untouched so analysis still completes (the video just may not
    play in some browsers).
    """
    if not path.exists():
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning(
            "ffmpeg is unavailable; annotated video remains in its original "
            "OpenCV codec: %s",
            path,
        )
        return

    transcoded = path.with_name(f"{path.stem}.h264{path.suffix}")
    try:
        # Do not mistake output left by an interrupted older run for the result
        # of this invocation if starting ffmpeg itself fails.
        transcoded.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "Could not prepare temporary H.264 output %s: %s; keeping %s",
            transcoded,
            exc,
            path,
        )
        return

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        # OpenCV already wrote physically oriented frames. Never apply stale
        # presentation metadata during the final browser-compatible encode.
        "-noautorotate",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-map_metadata",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
        "-movflags",
        "+faststart",
        "-an",
        str(transcoded),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            shell=False,
        )
    except OSError as exc:
        _remove_temporary_file(transcoded, description="partial H.264 video")
        logger.warning("Could not transcode annotated video %s: %s", path, exc)
        return

    if result.returncode == 0 and _is_nonempty_file(transcoded):
        try:
            os.replace(transcoded, path)
            return
        except OSError as exc:
            _remove_temporary_file(
                transcoded,
                description="completed H.264 video that could not be installed",
            )
            logger.warning(
                "Could not replace annotated video %s with its H.264 copy: %s; "
                "keeping the OpenCV output",
                path,
                exc,
            )
            return

    _remove_temporary_file(transcoded, description="partial H.264 video")
    logger.warning(
        "ffmpeg failed to transcode annotated video %s%s",
        path,
        _subprocess_detail(result.stderr),
    )


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _remove_temporary_file(path: Path, *, description: str) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove %s %s: %s", description, path, exc)


def _subprocess_detail(stderr: object) -> str:
    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    elif isinstance(stderr, str):
        text = stderr
    else:
        text = ""
    text = " ".join(text.strip().split())
    return f" ({text[-500:]})" if text else ""


def _save_key_frame_images(
    *,
    key_frame_images: dict[int, dict[str, object]],
    reps: list,
    frame_dir: Path,
) -> None:
    if not key_frame_images:
        return

    frame_dir.mkdir(parents=True, exist_ok=True)
    reps_by_id = {rep.rep_id: rep for rep in reps}
    for rep_id, frames_by_role in key_frame_images.items():
        rep = reps_by_id.get(rep_id)
        if rep is None:
            continue

        for role, rgb_frame in frames_by_role.items():
            path = frame_dir / f"rep_{rep_id:03d}_{role}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            rep.key_frames[role] = str(path)
