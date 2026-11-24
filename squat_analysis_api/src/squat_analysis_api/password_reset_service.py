from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import math
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import Engine, func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from squat_analysis_api.auth import hash_password, utc_now
from squat_analysis_api.models import AppUser, AuthSession, PasswordResetRecord
from squat_analysis_api.password_reset_email import PasswordResetEmailSender


logger = logging.getLogger(__name__)


class PasswordResetConfigurationError(RuntimeError):
    pass


class PasswordResetRateLimited(RuntimeError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("password reset request rate limited")
        self.retry_after_seconds = max(1, retry_after_seconds)


class InvalidPasswordResetCode(RuntimeError):
    pass


class PasswordResetAttemptsExceeded(RuntimeError):
    pass


class InvalidPasswordResetToken(RuntimeError):
    pass


@dataclass(frozen=True)
class PasswordResetSettings:
    pepper: bytes
    code_ttl_seconds: int = 600
    reset_token_ttl_seconds: int = 600
    resend_cooldown_seconds: int = 60
    rate_window_seconds: int = 3600
    max_requests_per_email: int = 5
    max_requests_per_ip: int = 20
    max_attempts: int = 5
    trusted_proxy_ips: frozenset[str] = frozenset()

    @classmethod
    def from_environment(cls) -> PasswordResetSettings:
        pepper_text = os.getenv("PASSWORD_RESET_PEPPER", "").strip()
        pepper = pepper_text.encode("utf-8")
        if len(pepper) < 32 or pepper_text == "replace-with-at-least-32-random-bytes":
            raise PasswordResetConfigurationError(
                "PASSWORD_RESET_PEPPER must contain at least 32 bytes"
            )

        max_attempts = _positive_int(
            "PASSWORD_RESET_MAX_ATTEMPTS", default=5
        )
        if max_attempts > 5:
            raise PasswordResetConfigurationError(
                "PASSWORD_RESET_MAX_ATTEMPTS must not exceed 5"
            )

        proxies = frozenset(
            value.strip()
            for value in os.getenv("PASSWORD_RESET_TRUSTED_PROXY_IPS", "").split(",")
            if value.strip()
        )
        for proxy in proxies:
            try:
                ipaddress.ip_address(proxy)
            except ValueError as exc:
                raise PasswordResetConfigurationError(
                    "PASSWORD_RESET_TRUSTED_PROXY_IPS contains an invalid address"
                ) from exc

        resend_cooldown_seconds = _positive_int(
            "PASSWORD_RESET_RESEND_COOLDOWN_SECONDS", default=60
        )
        if resend_cooldown_seconds < 60:
            raise PasswordResetConfigurationError(
                "PASSWORD_RESET_RESEND_COOLDOWN_SECONDS must be at least 60"
            )

        return cls(
            pepper=pepper,
            code_ttl_seconds=_positive_int(
                "PASSWORD_RESET_CODE_TTL_SECONDS", default=600
            ),
            reset_token_ttl_seconds=_positive_int(
                "PASSWORD_RESET_TOKEN_TTL_SECONDS", default=600
            ),
            resend_cooldown_seconds=resend_cooldown_seconds,
            rate_window_seconds=_positive_int(
                "PASSWORD_RESET_RATE_WINDOW_SECONDS", default=3600
            ),
            max_requests_per_email=_positive_int(
                "PASSWORD_RESET_MAX_REQUESTS_PER_EMAIL", default=5
            ),
            max_requests_per_ip=_positive_int(
                "PASSWORD_RESET_MAX_REQUESTS_PER_IP", default=20
            ),
            max_attempts=max_attempts,
            trusted_proxy_ips=proxies,
        )


@dataclass(frozen=True)
class IssuedPasswordResetRequest:
    request_token: str
    code: str
    recipient: str | None
    record_id: int


@dataclass(frozen=True)
class VerifiedPasswordReset:
    reset_token: str


def get_password_reset_settings() -> PasswordResetSettings:
    return PasswordResetSettings.from_environment()


def resolve_requester_ip(
    *,
    peer_ip: str | None,
    forwarded_for: str | None,
    settings: PasswordResetSettings,
) -> str:
    """Resolve a client IP without trusting arbitrary forwarding headers."""

    peer = _valid_ip(peer_ip) or "unknown"
    if peer in settings.trusted_proxy_ips and forwarded_for:
        forwarded = _valid_ip(forwarded_for.split(",", 1)[0].strip())
        if forwarded is not None:
            return forwarded
    return peer


def issue_password_reset_request(
    db: Session,
    *,
    email: str,
    requester_ip: str,
    settings: PasswordResetSettings,
    now: datetime | None = None,
) -> IssuedPasswordResetRequest:
    now = now or utc_now()
    normalized_email = email.strip().lower()
    email_hash = _fingerprint(settings.pepper, "email", normalized_email)
    ip_hash = _fingerprint(settings.pepper, "ip", requester_ip)

    _acquire_rate_limit_locks(db, email_hash, ip_hash)
    window_start = now - timedelta(seconds=settings.rate_window_seconds)

    email_count = db.scalar(
        select(func.count(PasswordResetRecord.id)).where(
            PasswordResetRecord.email_hash == email_hash,
            PasswordResetRecord.created_at >= window_start,
        )
    ) or 0
    ip_count = db.scalar(
        select(func.count(PasswordResetRecord.id)).where(
            PasswordResetRecord.requester_ip_hash == ip_hash,
            PasswordResetRecord.created_at >= window_start,
        )
    ) or 0
    if email_count >= settings.max_requests_per_email:
        raise PasswordResetRateLimited(settings.rate_window_seconds)
    if ip_count >= settings.max_requests_per_ip:
        raise PasswordResetRateLimited(settings.rate_window_seconds)

    latest_created_at = db.scalar(
        select(PasswordResetRecord.created_at)
        .where(PasswordResetRecord.email_hash == email_hash)
        .order_by(PasswordResetRecord.created_at.desc())
        .limit(1)
    )
    if latest_created_at is not None:
        elapsed = (now - _as_utc(latest_created_at)).total_seconds()
        if elapsed < settings.resend_cooldown_seconds:
            raise PasswordResetRateLimited(
                math.ceil(settings.resend_cooldown_seconds - elapsed)
            )

    user = db.scalar(
        select(AppUser)
        .where(func.lower(AppUser.email) == normalized_email)
        .with_for_update()
    )
    if user is not None and not user.is_active:
        user = None

    db.execute(
        update(PasswordResetRecord)
        .where(
            PasswordResetRecord.email_hash == email_hash,
            PasswordResetRecord.used_at.is_(None),
            PasswordResetRecord.invalidated_at.is_(None),
        )
        .values(invalidated_at=now)
    )
    db.flush()

    request_token = secrets.token_urlsafe(32)
    code = f"{secrets.randbelow(1_000_000):06d}"
    record = PasswordResetRecord(
        user_id=user.id if user is not None else None,
        request_token_hash=_sha256(request_token),
        email_hash=email_hash,
        requester_ip_hash=ip_hash,
        code_hash=_code_digest(settings.pepper, request_token, code),
        failed_attempts=0,
        created_at=now,
        expires_at=now + timedelta(seconds=settings.code_ttl_seconds),
    )
    try:
        db.add(record)
        db.flush()
        record_id = record.id
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        # A concurrent request may have won the one-active-request constraint.
        raise PasswordResetRateLimited(settings.resend_cooldown_seconds) from exc

    return IssuedPasswordResetRequest(
        request_token=request_token,
        code=code,
        recipient=normalized_email if user is not None else None,
        record_id=record_id,
    )


def verify_password_reset_code(
    db: Session,
    *,
    request_token: str,
    code: str,
    settings: PasswordResetSettings,
    now: datetime | None = None,
) -> VerifiedPasswordReset:
    now = now or utc_now()
    record = db.scalar(
        select(PasswordResetRecord)
        .where(PasswordResetRecord.request_token_hash == _sha256(request_token))
        .with_for_update()
    )
    if not _is_verifiable(record, now, settings.max_attempts):
        raise InvalidPasswordResetCode()

    expected = _code_digest(settings.pepper, request_token, code)
    if not hmac.compare_digest(record.code_hash, expected):
        record.failed_attempts += 1
        exhausted = record.failed_attempts >= settings.max_attempts
        if exhausted:
            record.invalidated_at = now
        db.commit()
        if exhausted:
            raise PasswordResetAttemptsExceeded()
        raise InvalidPasswordResetCode()

    reset_token = secrets.token_urlsafe(32)
    record.verified_at = now
    record.reset_token_hash = _sha256(reset_token)
    record.reset_token_expires_at = now + timedelta(
        seconds=settings.reset_token_ttl_seconds
    )
    db.commit()
    return VerifiedPasswordReset(reset_token=reset_token)


def complete_password_reset(
    db: Session,
    *,
    reset_token: str,
    new_password: str,
    now: datetime | None = None,
) -> None:
    now = now or utc_now()
    record = db.scalar(
        select(PasswordResetRecord)
        .where(PasswordResetRecord.reset_token_hash == _sha256(reset_token))
        .with_for_update()
    )
    if not _is_completable(record, now):
        raise InvalidPasswordResetToken()

    user = db.scalar(
        select(AppUser)
        .where(AppUser.id == record.user_id, AppUser.is_active.is_(True))
        .with_for_update()
    )
    if user is None:
        raise InvalidPasswordResetToken()

    user.password_hash = hash_password(new_password)
    user.updated_at = now
    record.used_at = now
    record.invalidated_at = now
    db.execute(
        update(PasswordResetRecord)
        .where(
            PasswordResetRecord.user_id == user.id,
            PasswordResetRecord.id != record.id,
            PasswordResetRecord.used_at.is_(None),
            PasswordResetRecord.invalidated_at.is_(None),
        )
        .values(invalidated_at=now)
    )
    db.execute(
        update(AuthSession)
        .where(
            AuthSession.user_id == user.id,
            AuthSession.revoked_at.is_(None),
        )
        .values(revoked_at=now)
    )
    db.commit()


def deliver_password_reset_email(
    *,
    sender: PasswordResetEmailSender,
    recipient: str | None,
    code: str,
    expires_in_minutes: int,
    record_id: int,
    bind: Engine,
) -> None:
    """Deliver after the HTTP response and safely mark failed deliveries."""

    if recipient is None:
        return
    try:
        sender.send_code(
            recipient=recipient,
            code=code,
            expires_in_minutes=expires_in_minutes,
        )
    except Exception as exc:  # Provider implementations are third-party edges.
        logger.warning(
            "Password reset email delivery failed; record_id=%s error_type=%s",
            record_id,
            type(exc).__name__,
        )
        try:
            with Session(bind=bind) as db:
                now = utc_now()
                record = db.get(PasswordResetRecord, record_id)
                if record is not None:
                    record.delivery_failed_at = now
                    if record.used_at is None and record.invalidated_at is None:
                        record.invalidated_at = now
                    db.commit()
        except SQLAlchemyError as persistence_error:
            logger.error(
                "Could not mark password reset delivery failure; "
                "record_id=%s error_type=%s",
                record_id,
                type(persistence_error).__name__,
            )


def _is_verifiable(
    record: PasswordResetRecord | None, now: datetime, max_attempts: int
) -> bool:
    return bool(
        record is not None
        and record.used_at is None
        and record.invalidated_at is None
        and record.verified_at is None
        and _as_utc(record.expires_at) > now
        and record.failed_attempts < max_attempts
    )


def _is_completable(
    record: PasswordResetRecord | None, now: datetime
) -> bool:
    return bool(
        record is not None
        and record.user_id is not None
        and record.used_at is None
        and record.invalidated_at is None
        and record.verified_at is not None
        and record.reset_token_expires_at is not None
        and _as_utc(record.reset_token_expires_at) > now
    )


def _acquire_rate_limit_locks(
    db: Session, email_hash: str, ip_hash: str
) -> None:
    bind = db.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for digest in sorted({email_hash, ip_hash}):
        lock_key = int.from_bytes(
            bytes.fromhex(digest)[:8], byteorder="big", signed=True
        )
        db.execute(select(func.pg_advisory_xact_lock(lock_key)))


def _fingerprint(pepper: bytes, domain: str, value: str) -> str:
    return hmac.new(
        pepper,
        f"fitvision:{domain}:{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _code_digest(pepper: bytes, request_token: str, code: str) -> str:
    return _fingerprint(pepper, "password-reset-code", f"{request_token}:{code}")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _valid_ip(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return None


def _positive_int(name: str, *, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise PasswordResetConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise PasswordResetConfigurationError(f"{name} must be positive")
    return value
