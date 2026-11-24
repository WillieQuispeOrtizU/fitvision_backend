from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


TrainingLevel = Literal["beginner", "intermediate", "advanced"]
EvaluationExercise = Literal["squat"]
EvaluationMode = Literal["beginner", "pro"]
EvaluationStatus = Literal["queued", "processing", "completed", "failed"]
ChallengeStatus = Literal["active", "completed", "claimed", "expired"]


def _normalize_email(value: object) -> object:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _normalize_optional_name(value: object) -> object:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    return normalized or None


def _validate_birth_date(value: date | None) -> date | None:
    if value is not None and value > date.today():
        raise ValueError("birth_date must not be in the future")
    return value


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=1024)
    full_name: str | None = Field(default=None, max_length=150)
    birth_date: date | None = None
    height_cm: float | None = Field(default=None, gt=0, le=999.99)
    weight_kg: float | None = Field(default=None, gt=0, le=999.99)
    training_level: TrainingLevel | None = None

    _normalize_email = field_validator("email", mode="before")(_normalize_email)
    _normalize_name = field_validator("full_name", mode="before")(
        _normalize_optional_name
    )
    _birth_date_not_in_future = field_validator("birth_date")(
        _validate_birth_date
    )


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=1, max_length=1024)

    _normalize_email = field_validator("email", mode="before")(_normalize_email)


class UserUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(default=None, max_length=150)
    birth_date: date | None = None
    height_cm: float | None = Field(default=None, gt=0, le=999.99)
    weight_kg: float | None = Field(default=None, gt=0, le=999.99)
    training_level: TrainingLevel | None = None

    _normalize_name = field_validator("full_name", mode="before")(
        _normalize_optional_name
    )
    _birth_date_not_in_future = field_validator("birth_date")(
        _validate_birth_date
    )


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str | None
    birth_date: date | None
    height_cm: float | None
    weight_kg: float | None
    training_level: TrainingLevel | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class AuthResponse(BaseModel):
    user: UserResponse
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: datetime


class PasswordResetRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr = Field(max_length=255)

    _normalize_email = field_validator("email", mode="before")(_normalize_email)


class PasswordResetRequestResponse(BaseModel):
    message: str
    request_token: str
    expires_in_seconds: int
    retry_after_seconds: int


class PasswordResetVerifyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_token: str = Field(min_length=32, max_length=512)
    code: str = Field(pattern=r"^\d{6}$")


class PasswordResetVerifyResponse(BaseModel):
    reset_token: str
    expires_in_seconds: int


class PasswordResetCompletePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reset_token: str = Field(min_length=32, max_length=512)
    new_password: str = Field(min_length=8, max_length=1024)


class EvaluationSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: str
    exercise: EvaluationExercise
    mode: EvaluationMode
    status: EvaluationStatus
    correct: bool | None
    total_reps: int | None
    correct_reps: int | None
    incorrect_reps: int | None
    original_video_url: str | None
    annotated_video_url: str | None
    video_seconds: float | None
    assistant_seconds: float | None
    xp_earned: int
    created_at: datetime
    completed_at: datetime | None


class EvaluationDetailResponse(EvaluationSummaryResponse):
    report_json: dict[str, Any]


class GamificationProgressResponse(BaseModel):
    total_xp: int
    level: int
    current_streak: int
    longest_streak: int
    total_evaluations: int
    correct_evaluations: int
    needs_improvement_evaluations: int
    total_repetitions: int
    correct_repetitions: int
    incorrect_repetitions: int
    xp_at_level_start: int
    xp_for_next_level: int
    xp_within_level: int
    xp_required_within_level: int


class GamificationAchievementResponse(BaseModel):
    code: str
    name: str
    description: str
    target: int
    current: int
    xp_reward: int
    unlocked: bool
    unlocked_at: datetime | None


class GamificationChallengeResponse(BaseModel):
    code: str
    title: str
    description: str
    target: int
    current: int
    xp_reward: int
    status: ChallengeStatus
    completed_at: datetime | None


class GamificationResponse(BaseModel):
    progress: GamificationProgressResponse
    achievements: list[GamificationAchievementResponse]
    challenges: list[GamificationChallengeResponse]
