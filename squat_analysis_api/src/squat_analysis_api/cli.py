from __future__ import annotations

import argparse
from pathlib import Path

from squat_analysis_api.assistant_client import AssistantClient, AssistantClientError
from squat_analysis_api.video_processor import discover_videos, output_path_for, process_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="squat-analysis",
        description="Analyze MP4 squat videos and write annotated output videos.",
    )
    parser.add_argument(
        "--mode",
        choices=("beginner", "pro"),
        default="beginner",
        help="Threshold set to use for squat analysis.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input_video"),
        help="Directory containing .mp4 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_video"),
        help="Directory where analyzed videos will be written.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Process one .mp4 file instead of every .mp4 in --input.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a JSON analysis report and sampled key-frame images.",
    )
    parser.add_argument(
        "--assistant-url",
        help="Enable Falcon/Gemma feedback through a local assistant service URL.",
    )
    parser.add_argument(
        "--assistant-timeout",
        type=float,
        help="Seconds to wait for assistant analysis. Defaults to SQUAT_ASSISTANT_CLIENT_TIMEOUT or 600.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_dir = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        videos = [args.file]
    else:
        videos = discover_videos(input_dir)

    if not videos:
        print(f"No .mp4 files found in {input_dir}")
        return 0

    assistant_client = None
    if args.assistant_url:
        assistant_client = AssistantClient(args.assistant_url, timeout=args.assistant_timeout)
        try:
            assistant_client.health()
        except AssistantClientError as exc:
            print(f"Assistant unavailable at {args.assistant_url}: {exc}")
            return 1

    exit_code = 0
    for input_path in videos:
        if input_path.suffix.lower() != ".mp4":
            print(f"Skipping non-mp4 file: {input_path}")
            continue

        output_path = output_path_for(input_path, output_dir)
        try:
            result = process_video(
                input_path,
                output_path,
                args.mode,
                write_json_report=args.report,
                assistant_url=args.assistant_url,
                assistant_client=assistant_client,
            )
        except Exception as exc:
            exit_code = 1
            print(f"Failed: {input_path} ({exc})")
            continue

        message = (
            "Processed "
            f"input={result.input_path} "
            f"output={result.output_path} "
            f"frames={result.frames} "
            f"correct={result.stats.correct} "
            f"incorrect={result.stats.incorrect}"
        )
        if result.report_path:
            message += f" report={result.report_path}"
        print(message)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
