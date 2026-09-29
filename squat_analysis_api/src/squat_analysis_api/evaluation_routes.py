from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from squat_analysis_api.auth import get_current_user
from squat_analysis_api.database import get_db
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
) -> list[Evaluation]:
    return list(
        db.scalars(
            select(Evaluation)
            .where(Evaluation.user_id == user.id)
            .order_by(Evaluation.created_at.desc(), Evaluation.id.desc())
        ).all()
    )


@router.get("/{evaluation_id}", response_model=EvaluationDetailResponse)
def get_evaluation(
    evaluation_id: int,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Evaluation:
    evaluation = db.scalar(
        select(Evaluation).where(
            Evaluation.id == evaluation_id,
            Evaluation.user_id == user.id,
        )
    )
    if evaluation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation not found",
        )
    return evaluation
