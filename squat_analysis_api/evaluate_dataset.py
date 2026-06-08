"""Evaluate the MediaPipe squat analyzer against a labelled dataset.

The dataset is a folder of subfolders, one per ground-truth class, e.g.

    squats/
      good/   *.mp4 *.MOV ...
      bad/    *.mp4 *.MOV ...

Each video carries a single video-level label (its folder name), but the
analyzer produces a per-rep verdict (correct/incorrect) plus a rep count. This
script reconciles the two:

  * Form classification (video level): each video's per-rep verdicts are
    aggregated into one predicted label, then compared to the folder label.
    Reported as a confusion matrix + accuracy / precision / recall / F1, with
    "bad" treated as the positive class (the thing we want to catch).

  * Rep detection: videos where the analyzer found 0 reps are reported as a
    separate failure bucket. True rep-counting accuracy needs ground-truth
    counts, which folder labels do not provide -- pass --counts a CSV of
    `filename,reps` to also get mean-absolute-error and exact-match on counts.

This runs the analyzer ONLY: no annotated-video encode, no Falcon/Gemma. So it
is fast and isolates what MediaPipe + the rep state machine actually decide.

Usage:
    uv run python evaluate_dataset.py ../squats
    uv run python evaluate_dataset.py ../squats --mode pro --rule all
    uv run python evaluate_dataset.py ../squats --counts truth.csv --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from squat_analysis_api.analyzer import SquatFrameAnalyzer
from squat_analysis_api.thresholds import get_thresholds
from squat_analysis_api.utils import get_mediapipe_pose

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v", ".mkv"}

# Which folder name counts as the positive class for precision/recall. "bad" is
# positive because catching bad form is the safety-relevant outcome.
POSITIVE_LABEL = "bad"
NEGATIVE_LABEL = "good"


@dataclass
class VideoEvaluation:
    path: Path
    label: str
    total_reps: int
    correct_reps: int
    incorrect_reps: int
    predicted: str  # "good" | "bad" | "no_reps"
    issue_tags: list[str]
    seconds: float


@dataclass
class CountTruth:
    by_name: dict[str, int] = field(default_factory=dict)

    def get(self, path: Path) -> int | None:
        return self.by_name.get(path.name)


def discover_dataset(root: Path) -> dict[str, list[Path]]:
    """Map each class folder name -> sorted list of video files inside it."""
    classes: dict[str, list[Path]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        videos = sorted(
            p for p in child.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if videos:
            classes[child.name] = videos
    return classes


def analyze_video(path: Path, pose, mode: str) -> tuple[int, int, int, list[str]]:
    """Run the analyzer over one video; return (total, correct, incorrect, issues)."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    analyzer = SquatFrameAnalyzer(thresholds=get_thresholds(mode))
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            analyzer.process(
                rgb,
                pose,
                frame_index=frame_index,
                timestamp=frame_index / fps,
            )
            frame_index += 1
    finally:
        capture.release()

    reps = analyzer.reps
    stats = analyzer.stats
    issues: list[str] = []
    for rep in reps:
        issues.extend(rep.issue_tags)
    return len(reps), stats.correct, stats.incorrect, issues


def predict_label(correct: int, incorrect: int, rule: str) -> str:
    """Collapse per-rep verdicts into a single video-level prediction."""
    total = correct + incorrect
    if total == 0:
        return "no_reps"
    if rule == "all":
        # Strict: a single flagged rep condemns the whole video.
        return NEGATIVE_LABEL if incorrect == 0 else POSITIVE_LABEL
    # "majority": the video is bad only if most reps are flagged. Ties -> good.
    return POSITIVE_LABEL if incorrect > correct else NEGATIVE_LABEL


def load_counts(path: Path) -> CountTruth:
    truth = CountTruth()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            name, raw = row[0].strip(), row[1].strip()
            if not name or name.lower() in {"filename", "video", "name"}:
                continue
            try:
                truth.by_name[name] = int(raw)
            except ValueError:
                continue
    return truth


def evaluate(
    root: Path,
    *,
    mode: str,
    rule: str,
    counts: CountTruth | None,
) -> tuple[list[VideoEvaluation], dict]:
    classes = discover_dataset(root)
    if not classes:
        raise RuntimeError(f"No class subfolders with videos found under {root}")

    results: list[VideoEvaluation] = []
    with get_mediapipe_pose() as pose:
        for label, videos in classes.items():
            for video in videos:
                started = time.perf_counter()
                total, correct, incorrect, issues = analyze_video(video, pose, mode)
                elapsed = time.perf_counter() - started
                predicted = predict_label(correct, incorrect, rule)
                results.append(
                    VideoEvaluation(
                        path=video,
                        label=label,
                        total_reps=total,
                        correct_reps=correct,
                        incorrect_reps=incorrect,
                        predicted=predicted,
                        issue_tags=issues,
                        seconds=elapsed,
                    )
                )
                print(
                    f"  [{label:>4}] {video.name:<22} "
                    f"reps={total:<2} ok={correct:<2} bad={incorrect:<2} "
                    f"-> {predicted:<7} ({elapsed:4.1f}s)",
                    flush=True,
                )

    summary = summarize(results, counts=counts, rule=rule, mode=mode)
    return results, summary


def summarize(
    results: list[VideoEvaluation],
    *,
    counts: CountTruth | None,
    rule: str,
    mode: str,
) -> dict:
    labels = sorted({r.label for r in results})
    only_binary = set(labels) <= {POSITIVE_LABEL, NEGATIVE_LABEL}

    # Confusion matrix over scored videos (those with >=1 detected rep).
    scored = [r for r in results if r.predicted != "no_reps"]
    no_reps = [r for r in results if r.predicted == "no_reps"]

    tp = fp = tn = fn = 0
    correct_videos = 0
    for r in scored:
        if r.predicted == r.label:
            correct_videos += 1
        if only_binary:
            if r.label == POSITIVE_LABEL and r.predicted == POSITIVE_LABEL:
                tp += 1
            elif r.label == NEGATIVE_LABEL and r.predicted == POSITIVE_LABEL:
                fp += 1
            elif r.label == NEGATIVE_LABEL and r.predicted == NEGATIVE_LABEL:
                tn += 1
            elif r.label == POSITIVE_LABEL and r.predicted == NEGATIVE_LABEL:
                fn += 1

    accuracy = correct_videos / len(scored) if scored else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        "mode": mode,
        "aggregation_rule": rule,
        "positive_label": POSITIVE_LABEL,
        "videos_total": len(results),
        "videos_scored": len(scored),
        "videos_no_reps": len(no_reps),
        "no_reps_files": [r.path.name for r in no_reps],
        "video_classification": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "binary": only_binary,
        },
        "rep_detection": {
            "total_reps_detected": sum(r.total_reps for r in results),
            "reps_per_video_mean": round(
                sum(r.total_reps for r in results) / len(results), 2
            ) if results else 0.0,
        },
    }

    if counts is not None:
        metrics["rep_counting"] = score_counts(results, counts)
    return metrics


def score_counts(results: list[VideoEvaluation], counts: CountTruth) -> dict:
    matched: list[tuple[int, int]] = []
    for r in results:
        truth = counts.get(r.path)
        if truth is not None:
            matched.append((truth, r.total_reps))
    if not matched:
        return {"matched_videos": 0, "note": "no filename overlap with --counts CSV"}
    abs_errors = [abs(t - p) for t, p in matched]
    exact = sum(1 for t, p in matched if t == p)
    return {
        "matched_videos": len(matched),
        "exact_match": exact,
        "exact_match_rate": round(exact / len(matched), 4),
        "mae": round(sum(abs_errors) / len(abs_errors), 3),
        "max_abs_error": max(abs_errors),
    }


def print_summary(summary: dict, results: list[VideoEvaluation]) -> None:
    vc = summary["video_classification"]
    print("\n" + "=" * 64)
    print(f"DATASET EVALUATION  (mode={summary['mode']}, rule={summary['aggregation_rule']})")
    print("=" * 64)
    print(
        f"videos: {summary['videos_total']} total, "
        f"{summary['videos_scored']} scored, "
        f"{summary['videos_no_reps']} with 0 reps detected"
    )
    if summary["videos_no_reps"]:
        print(f"  0-rep videos: {', '.join(summary['no_reps_files'])}")

    print("\nVideo-level form classification (positive = '%s'):" % summary["positive_label"])
    print(f"  accuracy    : {vc['accuracy']:.3f}")
    if vc["binary"]:
        cm = vc["confusion_matrix"]
        print(f"  precision   : {vc['precision']:.3f}")
        print(f"  recall      : {vc['recall']:.3f}   (bad caught / all bad)")
        print(f"  specificity : {vc['specificity']:.3f}   (good kept / all good)")
        print(f"  f1          : {vc['f1']:.3f}")
        print(f"  confusion   : TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")
        print("                (positive = bad form)")
    else:
        print("  (precision/recall shown only for a binary good/bad dataset)")

    rd = summary["rep_detection"]
    print("\nRep detection:")
    print(f"  total reps detected : {rd['total_reps_detected']}")
    print(f"  reps / video (mean) : {rd['reps_per_video_mean']}")

    if "rep_counting" in summary:
        rc = summary["rep_counting"]
        print("\nRep counting vs ground truth:")
        if rc.get("matched_videos"):
            print(f"  matched videos  : {rc['matched_videos']}")
            print(f"  exact-count rate: {rc['exact_match_rate']:.3f} ({rc['exact_match']} exact)")
            print(f"  mean abs error  : {rc['mae']}")
            print(f"  worst abs error : {rc['max_abs_error']}")
        else:
            print(f"  {rc.get('note', 'no matches')}")

    # Per-video misclassifications, so the failures are inspectable.
    misses = [r for r in results if r.predicted != r.label]
    if misses:
        print("\nMisclassified / undetected videos:")
        for r in misses:
            issues = ",".join(sorted(set(r.issue_tags))) or "-"
            print(
                f"  {r.label:>4} {r.path.name:<22} -> {r.predicted:<7} "
                f"(reps={r.total_reps} ok={r.correct_reps} bad={r.incorrect_reps}; {issues})"
            )
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Folder with class subfolders (e.g. squats/)")
    parser.add_argument("--mode", choices=["beginner", "pro"], default="beginner")
    parser.add_argument(
        "--rule",
        choices=["majority", "all"],
        default="majority",
        help="How per-rep verdicts collapse to a video verdict. "
        "'majority': bad only if most reps flagged. 'all': bad if any rep flagged.",
    )
    parser.add_argument("--counts", type=Path, help="CSV of 'filename,reps' ground-truth counts")
    parser.add_argument("--json", type=Path, help="Write the full report (per-video + summary) here")
    args = parser.parse_args()

    counts = load_counts(args.counts) if args.counts else None

    print(f"Evaluating {args.dataset} (mode={args.mode}, rule={args.rule})...\n")
    results, summary = evaluate(args.dataset, mode=args.mode, rule=args.rule, counts=counts)
    print_summary(summary, results)

    if args.json:
        payload = {
            "summary": summary,
            "videos": [
                {
                    "file": r.path.name,
                    "label": r.label,
                    "predicted": r.predicted,
                    "total_reps": r.total_reps,
                    "correct_reps": r.correct_reps,
                    "incorrect_reps": r.incorrect_reps,
                    "issue_tags": sorted(set(r.issue_tags)),
                    "seconds": round(r.seconds, 2),
                }
                for r in results
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote report to {args.json}")


if __name__ == "__main__":
    main()
