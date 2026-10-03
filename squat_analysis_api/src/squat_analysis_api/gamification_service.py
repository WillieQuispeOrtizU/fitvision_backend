from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import and_, case, delete, func, or_, select
from sqlalchemy.orm import Session

from squat_analysis_api.gamification_catalog import ensure_gamification_catalog
from squat_analysis_api.models import (
    Achievement,
    AppUser,
    Challenge,
    Evaluation,
    UserAchievement,
    UserChallenge,
    UserProgress,
    XpTransaction,
)


XP_PER_LEVEL = 250
CORRECT_EVALUATION_XP = 100
NEEDS_IMPROVEMENT_EVALUATION_XP = 70
LIMA_TIMEZONE = ZoneInfo("America/Lima")


def level_for_xp(total_xp: int) -> int:
    return (total_xp // XP_PER_LEVEL) + 1


def initialize_user_gamification(
    db: Session,
    *,
    user_id: int,
    started_at: datetime | None = None,
) -> None:
    """Ensure canonical catalogs and active challenge rows exist."""

    now = _as_utc(started_at or datetime.now(timezone.utc))
    ensure_gamification_catalog(db)
    _get_or_create_progress(db, user_id=user_id, now=now)
    _ensure_user_challenges(db, user_id=user_id, started_at=now)


def apply_completed_evaluation(
    db: Session,
    *,
    evaluation: Evaluation,
    progress: UserProgress,
    occurred_at: datetime,
    historical_replay: bool = False,
) -> int:
    """Apply all derived gamification for one completed evaluation.

    The caller owns the transaction and must already hold locks for the
    evaluation and user_progress rows. Nothing is committed here.
    """

    ensure_gamification_catalog(db)
    occurred_at = _as_utc(occurred_at)
    history = _history_before(
        db,
        evaluation=evaluation,
        occurred_at=occurred_at,
        chronological_only=historical_replay,
    )
    previous = history["previous"]

    progress.total_evaluations = history["total_evaluations"] + 1
    progress.total_correct_reps = (
        history["correct_repetitions"] + int(evaluation.correct_reps or 0)
    )
    progress.total_incorrect_reps = (
        history["incorrect_repetitions"] + int(evaluation.incorrect_reps or 0)
    )
    _update_streak(progress, previous=previous, occurred_at=occurred_at)

    current_is_correct = evaluation.correct is True
    current_is_eligible = int(evaluation.total_reps or 0) > 0
    correct_evaluations = history["correct_evaluations"] + int(
        current_is_correct
    )
    eligible_evaluations = history["eligible_evaluations"] + int(
        current_is_eligible
    )
    eligible_correct_evaluations = history[
        "eligible_correct_evaluations"
    ] + int(current_is_eligible and current_is_correct)
    completed_pro = history["completed_pro"] or (
        current_is_eligible and evaluation.mode == "pro"
    )
    improved_previous = (
        current_is_eligible
        and current_is_correct
        and previous is not None
        and previous.correct is False
    )

    xp_earned = 0
    if current_is_eligible and evaluation.correct is not None:
        base_xp = (
            CORRECT_EVALUATION_XP
            if current_is_correct
            else NEEDS_IMPROVEMENT_EVALUATION_XP
        )
        reason = (
            "evaluation_completed_correct"
            if current_is_correct
            else "evaluation_completed_needs_improvement"
        )
        _add_xp_transaction(
            db,
            progress=progress,
            evaluation=evaluation,
            amount=base_xp,
            reason=reason,
            occurred_at=occurred_at,
        )
        xp_earned += base_xp

    achievement_values = {
        "first_evaluation": progress.total_evaluations,
        "first_correct": correct_evaluations,
        "five_evaluations": progress.total_evaluations,
        "ten_evaluations": progress.total_evaluations,
        "perfectionist": correct_evaluations,
    }
    _unlock_achievements(
        db,
        user_id=evaluation.user_id,
        evaluation_id=evaluation.id,
        values=achievement_values,
        occurred_at=occurred_at,
    )

    challenge_values = {
        "complete_three": eligible_evaluations,
        "two_correct": eligible_correct_evaluations,
        "try_pro": int(completed_pro),
    }
    xp_earned += _update_challenges(
        db,
        progress=progress,
        evaluation=evaluation,
        values=challenge_values,
        improved_previous=improved_previous,
        occurred_at=occurred_at,
    )

    progress.level = level_for_xp(progress.total_xp)
    progress.updated_at = occurred_at
    return xp_earned


def get_gamification_state(db: Session, *, user_id: int) -> dict[str, Any]:
    """Return the complete authoritative gamification state for one user."""

    now = datetime.now(timezone.utc)
    progress = db.scalar(
        select(UserProgress)
        .where(UserProgress.user_id == user_id)
        .with_for_update()
    )
    if progress is None:
        progress = _get_or_create_progress(db, user_id=user_id, now=now)
    ensure_gamification_catalog(db)
    _ensure_user_challenges(db, user_id=user_id, started_at=now)
    db.flush()

    aggregate = db.execute(
        select(
            func.count(Evaluation.id),
            func.coalesce(
                func.sum(case((Evaluation.correct.is_(True), 1), else_=0)), 0
            ),
            func.coalesce(
                func.sum(case((Evaluation.correct.is_(False), 1), else_=0)), 0
            ),
            func.coalesce(func.sum(Evaluation.correct_reps), 0),
            func.coalesce(func.sum(Evaluation.incorrect_reps), 0),
        ).where(
            Evaluation.user_id == user_id,
            Evaluation.status == "completed",
        )
    ).one()
    total_evaluations = int(aggregate[0] or 0)
    correct_evaluations = int(aggregate[1] or 0)
    needs_improvement_evaluations = int(aggregate[2] or 0)
    correct_repetitions = int(aggregate[3] or 0)
    incorrect_repetitions = int(aggregate[4] or 0)

    achievement_rows = db.execute(
        select(Achievement, UserAchievement)
        .outerjoin(
            UserAchievement,
            and_(
                UserAchievement.achievement_id == Achievement.id,
                UserAchievement.user_id == user_id,
            ),
        )
        .where(Achievement.is_active.is_(True))
        .order_by(Achievement.id)
    ).all()
    achievement_metrics = {
        "total_evaluations": total_evaluations,
        "correct_evaluations": correct_evaluations,
    }
    achievements = []
    for achievement, unlocked in achievement_rows:
        criteria = achievement.criteria_json or {}
        target = int(criteria.get("target", 1))
        metric = str(criteria.get("metric", "total_evaluations"))
        current = int(achievement_metrics.get(metric, 0))
        achievements.append(
            {
                "code": achievement.code,
                "name": achievement.name,
                "description": achievement.description or "",
                "target": target,
                "current": min(current, target),
                "xp_reward": achievement.xp_reward,
                "unlocked": unlocked is not None,
                "unlocked_at": None if unlocked is None else unlocked.unlocked_at,
            }
        )

    challenge_rows = db.execute(
        select(Challenge, UserChallenge)
        .join(
            UserChallenge,
            and_(
                UserChallenge.challenge_id == Challenge.id,
                UserChallenge.user_id == user_id,
            ),
        )
        .where(Challenge.is_active.is_(True))
        .order_by(Challenge.id)
    ).all()
    challenges = [
        {
            "code": challenge.code,
            "title": challenge.title,
            "description": challenge.description or "",
            "target": challenge.target_value,
            "current": min(user_challenge.progress_value, challenge.target_value),
            "xp_reward": challenge.xp_reward,
            "status": user_challenge.status,
            "completed_at": user_challenge.completed_at,
        }
        for challenge, user_challenge in challenge_rows
    ]

    xp_at_level_start = (progress.level - 1) * XP_PER_LEVEL
    return {
        "progress": {
            "total_xp": progress.total_xp,
            "level": progress.level,
            "current_streak": progress.current_streak,
            "longest_streak": progress.longest_streak,
            "total_evaluations": total_evaluations,
            "correct_evaluations": correct_evaluations,
            "needs_improvement_evaluations": needs_improvement_evaluations,
            "total_repetitions": correct_repetitions + incorrect_repetitions,
            "correct_repetitions": correct_repetitions,
            "incorrect_repetitions": incorrect_repetitions,
            "xp_at_level_start": xp_at_level_start,
            "xp_for_next_level": progress.level * XP_PER_LEVEL,
            "xp_within_level": progress.total_xp - xp_at_level_start,
            "xp_required_within_level": XP_PER_LEVEL,
        },
        "achievements": achievements,
        "challenges": challenges,
    }


def rebuild_user_gamification(db: Session, *, user_id: int) -> int:
    """Rebuild one user's derived state from completed evaluations.

    The caller owns the transaction. Evaluation rows and their reports are
    read-only throughout the replay.
    """

    now = datetime.now(timezone.utc)
    evaluations = list(
        db.scalars(
            select(Evaluation)
            .where(
                Evaluation.user_id == user_id,
                Evaluation.status == "completed",
            )
            .order_by(_evaluation_time_expression(), Evaluation.id)
            .with_for_update()
        ).all()
    )
    progress = db.scalar(
        select(UserProgress)
        .where(UserProgress.user_id == user_id)
        .with_for_update()
    )
    if progress is None:
        progress = UserProgress(user_id=user_id)
        db.add(progress)
        db.flush()
    ensure_gamification_catalog(db)

    db.execute(delete(XpTransaction).where(XpTransaction.user_id == user_id))
    db.execute(delete(UserAchievement).where(UserAchievement.user_id == user_id))
    db.execute(delete(UserChallenge).where(UserChallenge.user_id == user_id))
    progress.total_xp = 0
    progress.level = 1
    progress.current_streak = 0
    progress.longest_streak = 0
    progress.total_evaluations = 0
    progress.total_correct_reps = 0
    progress.total_incorrect_reps = 0
    progress.updated_at = now
    db.flush()

    started_at = _evaluation_time(evaluations[0]) if evaluations else now
    _ensure_user_challenges(db, user_id=user_id, started_at=started_at)
    for evaluation in evaluations:
        apply_completed_evaluation(
            db,
            evaluation=evaluation,
            progress=progress,
            occurred_at=_evaluation_time(evaluation),
            historical_replay=True,
        )
    return len(evaluations)


def all_user_ids(db: Session) -> list[int]:
    return list(db.scalars(select(AppUser.id).order_by(AppUser.id)).all())


def evaluation_xp_subquery():
    return (
        select(func.coalesce(func.sum(XpTransaction.amount), 0))
        .where(XpTransaction.evaluation_id == Evaluation.id)
        .correlate(Evaluation)
        .scalar_subquery()
    )


def _history_before(
    db: Session,
    *,
    evaluation: Evaluation,
    occurred_at: datetime,
    chronological_only: bool,
) -> dict[str, Any]:
    event_time = _evaluation_time_expression()
    comparison_time = (
        occurred_at.replace(tzinfo=None)
        if db.get_bind().dialect.name == "sqlite"
        else occurred_at
    )
    before = or_(
        event_time < comparison_time,
        and_(event_time == comparison_time, Evaluation.id < evaluation.id),
    )
    common_conditions = [
        Evaluation.user_id == evaluation.user_id,
        Evaluation.status == "completed",
        Evaluation.id != evaluation.id,
    ]
    if chronological_only:
        common_conditions.append(before)
    common = tuple(common_conditions)
    aggregate = db.execute(
        select(
            func.count(Evaluation.id),
            func.coalesce(
                func.sum(case((Evaluation.correct.is_(True), 1), else_=0)), 0
            ),
            func.coalesce(func.sum(Evaluation.correct_reps), 0),
            func.coalesce(func.sum(Evaluation.incorrect_reps), 0),
            func.coalesce(
                func.sum(case((Evaluation.total_reps > 0, 1), else_=0)), 0
            ),
            func.coalesce(
                func.sum(
                    case(
                        (
                            and_(
                                Evaluation.total_reps > 0,
                                Evaluation.correct.is_(True),
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ),
                0,
            ),
            func.coalesce(
                func.sum(
                    case(
                        (
                            and_(
                                Evaluation.total_reps > 0,
                                Evaluation.mode == "pro",
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ),
                0,
            ),
        ).where(*common)
    ).one()
    previous = db.scalar(
        select(Evaluation)
        .where(*common)
        .order_by(event_time.desc(), Evaluation.id.desc())
        .limit(1)
    )
    return {
        "total_evaluations": int(aggregate[0] or 0),
        "correct_evaluations": int(aggregate[1] or 0),
        "correct_repetitions": int(aggregate[2] or 0),
        "incorrect_repetitions": int(aggregate[3] or 0),
        "eligible_evaluations": int(aggregate[4] or 0),
        "eligible_correct_evaluations": int(aggregate[5] or 0),
        "completed_pro": bool(aggregate[6]),
        "previous": previous,
    }


def _unlock_achievements(
    db: Session,
    *,
    user_id: int,
    evaluation_id: int,
    values: dict[str, int],
    occurred_at: datetime,
) -> None:
    achievements = list(
        db.scalars(
            select(Achievement).where(Achievement.is_active.is_(True))
        ).all()
    )
    unlocked_ids = set(
        db.scalars(
            select(UserAchievement.achievement_id).where(
                UserAchievement.user_id == user_id
            )
        ).all()
    )
    for achievement in achievements:
        criteria = achievement.criteria_json or {}
        target = int(criteria.get("target", 1))
        if achievement.id in unlocked_ids or values.get(achievement.code, 0) < target:
            continue
        db.add(
            UserAchievement(
                user_id=user_id,
                achievement_id=achievement.id,
                evaluation_id=evaluation_id,
                unlocked_at=occurred_at,
            )
        )


def _update_challenges(
    db: Session,
    *,
    progress: UserProgress,
    evaluation: Evaluation,
    values: dict[str, int],
    improved_previous: bool,
    occurred_at: datetime,
) -> int:
    rows = _ensure_user_challenges(
        db,
        user_id=evaluation.user_id,
        started_at=occurred_at,
    )
    earned = 0
    for challenge, user_challenge in rows:
        if challenge.code == "improve_previous":
            value = max(user_challenge.progress_value, int(improved_previous))
        else:
            value = values.get(challenge.code, user_challenge.progress_value)
        user_challenge.progress_value = min(value, challenge.target_value)
        user_challenge.updated_at = occurred_at
        if (
            user_challenge.status == "active"
            and user_challenge.progress_value >= challenge.target_value
        ):
            user_challenge.status = "completed"
            user_challenge.completed_at = occurred_at
            if challenge.xp_reward > 0:
                _add_xp_transaction(
                    db,
                    progress=progress,
                    evaluation=evaluation,
                    amount=challenge.xp_reward,
                    reason=f"challenge:{challenge.code}",
                    occurred_at=occurred_at,
                )
                earned += challenge.xp_reward
    return earned


def _ensure_user_challenges(
    db: Session,
    *,
    user_id: int,
    started_at: datetime,
) -> list[tuple[Challenge, UserChallenge]]:
    challenges = list(
        db.scalars(
            select(Challenge)
            .where(Challenge.is_active.is_(True))
            .order_by(Challenge.id)
        ).all()
    )
    existing = {
        value.challenge_id: value
        for value in db.scalars(
            select(UserChallenge).where(UserChallenge.user_id == user_id)
        ).all()
    }
    result: list[tuple[Challenge, UserChallenge]] = []
    for challenge in challenges:
        state = existing.get(challenge.id)
        if state is None:
            state = UserChallenge(
                user_id=user_id,
                challenge_id=challenge.id,
                progress_value=0,
                status="active",
                started_at=started_at,
                updated_at=started_at,
            )
            db.add(state)
        result.append((challenge, state))
    db.flush()
    return result


def _get_or_create_progress(
    db: Session,
    *,
    user_id: int,
    now: datetime,
) -> UserProgress:
    progress = db.get(UserProgress, user_id)
    if progress is None:
        progress = UserProgress(
            user_id=user_id,
            total_xp=0,
            level=1,
            current_streak=0,
            longest_streak=0,
            total_evaluations=0,
            total_correct_reps=0,
            total_incorrect_reps=0,
            updated_at=now,
        )
        db.add(progress)
        db.flush()
    return progress


def _add_xp_transaction(
    db: Session,
    *,
    progress: UserProgress,
    evaluation: Evaluation,
    amount: int,
    reason: str,
    occurred_at: datetime,
) -> None:
    if amount == 0:
        return
    db.add(
        XpTransaction(
            user_id=evaluation.user_id,
            evaluation_id=evaluation.id,
            amount=amount,
            reason=reason,
            created_at=occurred_at,
        )
    )
    progress.total_xp += amount


def _update_streak(
    progress: UserProgress,
    *,
    previous: Evaluation | None,
    occurred_at: datetime,
) -> None:
    if previous is None:
        progress.current_streak = 1
    else:
        current_day = _lima_day(occurred_at)
        previous_day = _lima_day(_evaluation_time(previous))
        delta = (current_day - previous_day).days
        if delta == 0:
            progress.current_streak = max(progress.current_streak, 1)
        elif delta == 1:
            progress.current_streak += 1
        else:
            progress.current_streak = 1
    progress.longest_streak = max(
        progress.longest_streak,
        progress.current_streak,
    )


def _evaluation_time_expression():
    return func.coalesce(
        Evaluation.completed_at,
        Evaluation.updated_at,
        Evaluation.created_at,
    )


def _evaluation_time(evaluation: Evaluation) -> datetime:
    return _as_utc(
        evaluation.completed_at or evaluation.updated_at or evaluation.created_at
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _lima_day(value: datetime):
    return _as_utc(value).astimezone(LIMA_TIMEZONE).date()
