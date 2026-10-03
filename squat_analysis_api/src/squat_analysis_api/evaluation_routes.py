from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from squat_analysis_api.auth import get_current_user
from squat_analysis_api.database import get_db
from squat_analysis_api.gamification_service import evaluation_xp_subquery
from squat_analysis_api.models import AppUser, Evaluation
from squat_analysis_api.schemas import (
    EvaluationDetailResponse,
    EvaluationSummaryResponse,
)


router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


@router.get("", response_model=list[EvaluationSummaryResponse])
def list_evaluations(
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    rows = db.execute(
        select(Evaluation, evaluation_xp_subquery().label("xp_earned"))
        .where(Evaluation.user_id == user.id)
        .order_by(Evaluation.created_at.desc(), Evaluation.id.desc())
    ).all()
    return [_evaluation_payload(evaluation, xp_earned) for evaluation, xp_earned in rows]


@router.get("/{evaluation_id}", response_model=EvaluationDetailResponse)
def get_evaluation(
    evaluation_id: int,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    row = db.execute(
        select(Evaluation, evaluation_xp_subquery().label("xp_earned")).where(
            Evaluation.id == evaluation_id,
            Evaluation.user_id == user.id,
        )
    ).one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation not found",
        )
    evaluation, xp_earned = row
    return _evaluation_payload(evaluation, xp_earned, include_report=True)


def _evaluation_payload(
    evaluation: Evaluation,
    xp_earned: int,
    *,
    include_report: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": evaluation.id,
        "job_id": evaluation.job_id,
        "exercise": evaluation.exercise,
        "mode": evaluation.mode,
        "status": evaluation.status,
        "correct": evaluation.correct,
        "total_reps": evaluation.total_reps,
        "correct_reps": evaluation.correct_reps,
        "incorrect_reps": evaluation.incorrect_reps,
        "original_video_url": evaluation.original_video_url,
        "annotated_video_url": evaluation.annotated_video_url,
        "video_seconds": evaluation.video_seconds,
        "assistant_seconds": evaluation.assistant_seconds,
        "xp_earned": int(xp_earned or 0),
        "created_at": evaluation.created_at,
        "completed_at": evaluation.completed_at,
    }
    if include_report:
        payload["report_json"] = evaluation.report_json
    return payload
