from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from squat_analysis_api.auth import (
    CurrentAuth,
    get_current_auth,
    get_current_user,
    hash_password,
    issue_session,
    password_needs_rehash,
    utc_now,
    verify_password,
)
from squat_analysis_api.database import get_db
from squat_analysis_api.gamification_service import initialize_user_gamification
from squat_analysis_api.models import AppUser, UserProgress
from squat_analysis_api.schemas import (
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    UserResponse,
    UserUpdateRequest,
)


router = APIRouter(prefix="/api")


@router.post(
    "/auth/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["authentication"],
)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> AuthResponse:
    email = str(payload.email).strip().lower()
    if _find_user(db, email) is not None:
        raise _duplicate_email()

    user = AppUser(
        email=email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        birth_date=payload.birth_date,
        height_cm=_decimal(payload.height_cm),
        weight_kg=_decimal(payload.weight_kg),
        training_level=payload.training_level,
    )
    try:
        db.add(user)
        db.flush()
        progress = UserProgress(
            user_id=user.id,
            total_xp=0,
            level=1,
            current_streak=0,
            longest_streak=0,
            total_evaluations=0,
            total_correct_reps=0,
            total_incorrect_reps=0,
        )
        db.add(progress)
        db.flush()
        initialize_user_gamification(
            db,
            user_id=user.id,
            started_at=utc_now(),
        )
        issued = issue_session(user.id)
        db.add(issued.record)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if _is_duplicate_email_error(exc):
            raise _duplicate_email() from exc
        raise

    db.refresh(user)
    return AuthResponse(
        user=UserResponse.model_validate(user),
        access_token=issued.token,
        expires_at=issued.record.expires_at,
    )


@router.post(
    "/auth/login",
    response_model=AuthResponse,
    tags=["authentication"],
)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    email = str(payload.email).strip().lower()
    user = _find_user(db, email)
    password_is_valid = verify_password(
        user.password_hash if user is not None and user.is_active else None,
        payload.password,
    )
    if user is None or not user.is_active or not password_is_valid:
        raise _invalid_credentials()

    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)
        user.updated_at = utc_now()

    issued = issue_session(user.id)
    db.add(issued.record)
    db.commit()
    return AuthResponse(
        user=UserResponse.model_validate(user),
        access_token=issued.token,
        expires_at=issued.record.expires_at,
    )


@router.post(
    "/auth/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["authentication"],
)
def logout(
    current: CurrentAuth = Depends(get_current_auth),
    db: Session = Depends(get_db),
) -> Response:
    current.session.revoked_at = utc_now()
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/users/me",
    response_model=UserResponse,
    tags=["users"],
)
def get_me(user: AppUser = Depends(get_current_user)) -> AppUser:
    return user


@router.patch(
    "/users/me",
    response_model=UserResponse,
    tags=["users"],
)
def update_me(
    payload: UserUpdateRequest,
    user: AppUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AppUser:
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        if field in {"height_cm", "weight_kg"}:
            value = _decimal(value)
        setattr(user, field, value)

    if updates:
        user.updated_at = utc_now()
        db.commit()
        db.refresh(user)
    return user


def _find_user(db: Session, email: str) -> AppUser | None:
    return db.scalar(select(AppUser).where(func.lower(AppUser.email) == email))


def _decimal(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _is_duplicate_email_error(error: IntegrityError) -> bool:
    diagnostic = getattr(error.orig, "diag", None)
    constraint_name = getattr(diagnostic, "constraint_name", None)
    details = str(error.orig).lower()
    return constraint_name == "ux_app_user_email" or any(
        marker in details for marker in ("ux_app_user_email", "app_user.email")
    )


def _duplicate_email() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="an account with this email already exists",
    )


def _invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )
