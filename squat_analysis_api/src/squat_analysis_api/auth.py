from __future__ import annotations

import hashlib
import math
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from argon2.low_level import Type
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from squat_analysis_api.database import get_db
from squat_analysis_api.models import AppUser, AuthSession


DEFAULT_SESSION_TTL_HOURS = 24 * 30
_password_hasher = PasswordHasher(type=Type.ID)
_dummy_password_hash = _password_hasher.hash("fitvision-invalid-account-password")
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class IssuedSession:
    token: str
    record: AuthSession


@dataclass(frozen=True)
class CurrentAuth:
    user: AppUser
    session: AuthSession


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    candidate_hash = password_hash or _dummy_password_hash
    try:
        return _password_hasher.verify(candidate_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    try:
        return _password_hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return False


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_session(user_id: int) -> IssuedSession:
    token = secrets.token_urlsafe(32)
    now = utc_now()
    record = AuthSession(
        user_id=user_id,
        token_hash=hash_session_token(token),
        created_at=now,
        expires_at=now + _session_ttl(),
    )
    return IssuedSession(token=token, record=record)


def get_current_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> CurrentAuth:
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials
    ):
        raise _unauthorized()

    token_hash = hash_session_token(credentials.credentials)
    row = db.execute(
        select(AuthSession, AppUser)
        .join(AppUser, AppUser.id == AuthSession.user_id)
        .where(
            AuthSession.token_hash == token_hash,
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at > func.now(),
            AppUser.is_active.is_(True),
        )
    ).one_or_none()
    if row is None:
        raise _unauthorized()

    session_record, user = row
    session_record.last_used_at = utc_now()
    db.commit()
    return CurrentAuth(user=user, session=session_record)


def get_current_user(current: CurrentAuth = Depends(get_current_auth)) -> AppUser:
    return current.user


def _session_ttl() -> timedelta:
    raw_value = os.getenv("AUTH_SESSION_TTL_HOURS", "").strip()
    try:
        hours = float(raw_value) if raw_value else DEFAULT_SESSION_TTL_HOURS
    except ValueError:
        hours = DEFAULT_SESSION_TTL_HOURS
    if not math.isfinite(hours) or hours <= 0:
        hours = DEFAULT_SESSION_TTL_HOURS
    return timedelta(hours=hours)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid or expired authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
