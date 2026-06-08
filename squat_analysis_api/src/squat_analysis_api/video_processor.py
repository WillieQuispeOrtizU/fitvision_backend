from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from squat_analysis_api.assistant_client import AssistantClient, apply_assistant_response
from squat_analysis_api.analyzer import SquatFrameAnalyzer, SquatStats
from squat_analysis_api.reporting import VideoMetadata, build_report, empty_assistant_status, write_report
from squat_analysis_api.thresholds import get_thresholds
from squat_analysis_api.utils import get_mediapipe_pose


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
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video dimensions: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    analyzer = SquatFrameAnalyzer(thresholds=get_thresholds(mode))
    frame_count = 0

    try:
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


def _transcode_to_h264(path: Path) -> None:
    """Re-encode an mp4 in place to H.264/yuv420p with a web-streaming layout.

    OpenCV's mp4v output is MPEG-4 Part 2, which browsers refuse to play inline.
    Requires ffmpeg on PATH; if it is missing or the encode fails, the original
    file is left untouched so analysis still completes (the video just may not
    play in some browsers).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not path.exists():
        return

    transcoded = path.with_name(f"{path.stem}.h264{path.suffix}")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(transcoded),
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError:
        return

    if result.returncode == 0 and transcoded.exists():
        os.replace(transcoded, path)
    else:
        transcoded.unlink(missing_ok=True)


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
