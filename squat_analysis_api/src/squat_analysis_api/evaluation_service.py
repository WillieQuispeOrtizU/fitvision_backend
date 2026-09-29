from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from squat_analysis_api.models import Evaluation, UserProgress


logger = logging.getLogger(__name__)

SessionFactory = Callable[[], Session]


class EvaluationPersistenceError(RuntimeError):
    """Raised when an evaluation cannot be persisted consistently."""


class EvaluationNotFoundError(EvaluationPersistenceError):
    """Raised when a background transition cannot find its evaluation."""


def create_queued_evaluation(
    db: Session,
    *,
    user_id: int,
    job_id: str,
    mode: str,
    input_filename: str,
    original_video_url: str,
) -> Evaluation:
    """Create and commit the evaluation before its background job is scheduled."""

    now = _utc_now()
    evaluation = Evaluation(
        user_id=user_id,
        job_id=job_id,
        exercise="squat",
        mode=mode,
        status="queued",
        input_filename=input_filename,
        original_video_url=original_video_url,
        report_json={},
        created_at=now,
        updated_at=now,
    )
    try:
        db.add(evaluation)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to create queued evaluation for job_id=%s", job_id)
        raise
    return evaluation


def mark_evaluation_processing(
    session_factory: SessionFactory,
    *,
    job_id: str,
) -> bool:
    """Move a queued evaluation to processing in a short-lived transaction."""

    def transition(db: Session) -> bool:
        evaluation = _locked_evaluation(db, job_id)
        if evaluation.status != "queued":
            return False
        evaluation.status = "processing"
        evaluation.error = None
        evaluation.updated_at = _utc_now()
        return True

    return _run_background_transaction(
        session_factory,
        job_id=job_id,
        operation="mark evaluation processing",
        transition=transition,
    )


def complete_evaluation(
    session_factory: SessionFactory,
    *,
    job_id: str,
    report_json: dict[str, Any],
    original_video_url: str,
    annotated_video_url: str,
    video_seconds: float,
    assistant_seconds: float | None,
) -> bool:
    """Persist an authoritative report and increment progress exactly once.

    The evaluation row is locked before checking its terminal state. Evaluation
    completion and the three progress counters are committed together, so a
    duplicate worker cannot account for the same evaluation twice.
    """

    def transition(db: Session) -> bool:
        evaluation = _locked_evaluation(db, job_id)
        if evaluation.status == "completed":
            return False
        correct_reps, incorrect_reps, total_reps = _authoritative_counts(
            report_json
        )

        progress = db.scalar(
            select(UserProgress)
            .where(UserProgress.user_id == evaluation.user_id)
            .with_for_update()
        )
        if progress is None:
            raise EvaluationPersistenceError(
                f"user progress not found for evaluation job_id={job_id}"
            )

        now = _utc_now()
        evaluation.status = "completed"
        evaluation.correct = incorrect_reps == 0
        evaluation.total_reps = total_reps
        evaluation.correct_reps = correct_reps
        evaluation.incorrect_reps = incorrect_reps
        evaluation.original_video_url = original_video_url
        evaluation.annotated_video_url = annotated_video_url
        evaluation.video_seconds = _seconds_decimal(video_seconds)
        evaluation.assistant_seconds = _seconds_decimal(assistant_seconds)
        evaluation.error = None
        evaluation.report_json = report_json
        evaluation.completed_at = now
        evaluation.updated_at = now

        progress.total_evaluations += 1
        progress.total_correct_reps += correct_reps
        progress.total_incorrect_reps += incorrect_reps
        progress.updated_at = now
        return True

    return _run_background_transaction(
        session_factory,
        job_id=job_id,
        operation="complete evaluation",
        transition=transition,
    )


def fail_evaluation(
    session_factory: SessionFactory,
    *,
    job_id: str,
    error: str,
) -> bool:
    """Persist a processing failure without downgrading a completed evaluation."""

    def transition(db: Session) -> bool:
        evaluation = _locked_evaluation(db, job_id)
        if evaluation.status == "completed":
            return False
        evaluation.status = "failed"
        evaluation.error = error
        evaluation.updated_at = _utc_now()
        return True

    return _run_background_transaction(
        session_factory,
        job_id=job_id,
        operation="fail evaluation",
        transition=transition,
    )


def fail_interrupted_evaluations(
    session_factory: SessionFactory,
    *,
    job_ids: Iterable[str],
    error: str,
) -> int:
    """Reconcile disk jobs interrupted by a prior server process.

    Legacy job files may not have a matching evaluation, so missing rows are
    intentionally ignored. Completed and already-failed evaluations are never
    downgraded or rewritten.
    """

    unique_job_ids = tuple(dict.fromkeys(job_ids))
    if not unique_job_ids:
        return 0

    try:
        db = session_factory()
    except Exception:
        logger.exception("Failed to open a session for interrupted evaluations")
        raise

    with db:
        try:
            with db.begin():
                evaluations = db.scalars(
                    select(Evaluation)
                    .where(
                        Evaluation.job_id.in_(unique_job_ids),
                        Evaluation.status.in_(("queued", "processing")),
                    )
                    .with_for_update()
                ).all()
                now = _utc_now()
                for evaluation in evaluations:
                    evaluation.status = "failed"
                    evaluation.error = error
                    evaluation.updated_at = now
                return len(evaluations)
        except Exception:
            db.rollback()
            logger.exception(
                "Failed to reconcile interrupted evaluations for %d job(s)",
                len(unique_job_ids),
            )
            raise


def _locked_evaluation(db: Session, job_id: str) -> Evaluation:
    evaluation = db.scalar(
        select(Evaluation)
        .where(Evaluation.job_id == job_id)
        .with_for_update()
    )
    if evaluation is None:
        raise EvaluationNotFoundError(
            f"evaluation not found for job_id={job_id}"
        )
    return evaluation


def _run_background_transaction(
    session_factory: SessionFactory,
    *,
    job_id: str,
    operation: str,
    transition: Callable[[Session], bool],
) -> bool:
    try:
        db = session_factory()
    except Exception:
        logger.exception("Failed to open a session to %s for job_id=%s", operation, job_id)
        raise

    with db:
        try:
            with db.begin():
                return transition(db)
        except Exception:
            # Session.begin() already rolls back, and this explicit rollback also
            # protects custom Session implementations used by deployments/tests.
            db.rollback()
            logger.exception("Failed to %s for job_id=%s", operation, job_id)
            raise


def _authoritative_counts(report_json: dict[str, Any]) -> tuple[int, int, int]:
    try:
        stats = report_json["stats"]
        correct_reps = _nonnegative_int(stats["correct"], "stats.correct")
        incorrect_reps = _nonnegative_int(stats["incorrect"], "stats.incorrect")
        total_reps = _nonnegative_int(stats["total_reps"], "stats.total_reps")
    except (KeyError, TypeError) as exc:
        raise EvaluationPersistenceError(
            "analysis report is missing authoritative repetition counts"
        ) from exc

    if total_reps != correct_reps + incorrect_reps:
        raise EvaluationPersistenceError(
            "analysis report repetition counts are inconsistent"
        )
    return correct_reps, incorrect_reps, total_reps


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationPersistenceError(
            f"analysis report field {field} must be a non-negative integer"
        )
    return value


def _seconds_decimal(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
