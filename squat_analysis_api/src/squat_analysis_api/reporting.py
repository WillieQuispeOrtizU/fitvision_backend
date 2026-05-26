from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RepAnalysis:
    rep_id: int
    start_frame: int
    end_frame: int
    bottom_frame: int
    start_time: float
    end_time: float
    correct: bool
    issue_tags: list[str]
    worst_issue: str | None
    angles: dict[str, dict[str, float]]
    key_frames: dict[str, str] = field(default_factory=dict)
    assistant_feedback: dict[str, Any] | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_report_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


@dataclass
class VideoMetadata:
    input_path: str
    output_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float


def empty_assistant_status(*, enabled: bool, service_url: str | None = None) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "service_url": service_url,
        "status": "not_requested" if not enabled else "pending",
        "model_names": {},
        "latency_seconds": None,
        "errors": [],
        "overall_summary": None,
        "detections": [],
    }


def build_report(
    *,
    video: VideoMetadata,
    stats: Any,
    reps: list[RepAnalysis],
    assistant: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "video": asdict(video),
        "stats": {
            "correct": int(stats.correct),
            "incorrect": int(stats.incorrect),
            "total_reps": len(reps),
        },
        "reps": [rep.to_report_dict() for rep in reps],
        "assistant": assistant or empty_assistant_status(enabled=False),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

