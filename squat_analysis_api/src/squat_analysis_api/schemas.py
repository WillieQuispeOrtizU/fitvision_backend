from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


TrainingLevel = Literal["beginner", "intermediate", "advanced"]


def _normalize_email(value: object) -> object:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _normalize_optional_name(value: object) -> object:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    return normalized or None


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
