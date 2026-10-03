from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from squat_analysis_api.auth import get_current_user
from squat_analysis_api.database import get_db
from squat_analysis_api.gamification_service import get_gamification_state
from squat_analysis_api.models import AppUser
from squat_analysis_api.schemas import GamificationResponse


router = APIRouter(prefix="/api/gamification", tags=["gamification"])


@router.get("", response_model=GamificationResponse)
def get_gamification(
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    state = get_gamification_state(db, user_id=user.id)
    # This commit only persists idempotent catalog/user-challenge repair for
    # users created before gamification was introduced.
    db.commit()
    return state
