from __future__ import annotations

import math

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from sqlalchemy.orm import Session

from squat_analysis_api.database import get_db
from squat_analysis_api.password_reset_email import (
    PasswordResetEmailConfigurationError,
    PasswordResetEmailSender,
    get_password_reset_email_sender,
)
from squat_analysis_api.password_reset_service import (
    InvalidPasswordResetCode,
    InvalidPasswordResetToken,
    PasswordResetAttemptsExceeded,
    PasswordResetConfigurationError,
    PasswordResetRateLimited,
    PasswordResetSettings,
    complete_password_reset,
    deliver_password_reset_email,
    get_password_reset_settings,
    issue_password_reset_request,
    resolve_requester_ip,
    verify_password_reset_code,
)
from squat_analysis_api.schemas import (
    PasswordResetCompletePayload,
    PasswordResetRequestPayload,
    PasswordResetRequestResponse,
    PasswordResetVerifyPayload,
    PasswordResetVerifyResponse,
)


router = APIRouter(prefix="/api/auth/password-reset", tags=["authentication"])

GENERIC_REQUEST_MESSAGE = (
    "If an active account exists for that email, a recovery code will be sent."
)


def password_reset_settings_dependency() -> PasswordResetSettings:
    try:
        return get_password_reset_settings()
    except PasswordResetConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="password reset service is unavailable",
        ) from exc


def password_reset_sender_dependency() -> PasswordResetEmailSender:
    try:
        return get_password_reset_email_sender()
    except PasswordResetEmailConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="password reset email service is unavailable",
        ) from exc


@router.post(
    "/request",
    response_model=PasswordResetRequestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_password_reset(
    payload: PasswordResetRequestPayload,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    settings: PasswordResetSettings = Depends(password_reset_settings_dependency),
    sender: PasswordResetEmailSender = Depends(password_reset_sender_dependency),
    db: Session = Depends(get_db),
) -> PasswordResetRequestResponse:
    requester_ip = resolve_requester_ip(
        peer_ip=request.client.host if request.client is not None else None,
        forwarded_for=request.headers.get("x-forwarded-for"),
        settings=settings,
    )
    try:
        issued = issue_password_reset_request(
            db,
            email=str(payload.email),
            requester_ip=requester_ip,
            settings=settings,
        )
    except PasswordResetRateLimited as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="password reset requests are temporarily limited",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc

    background_tasks.add_task(
        deliver_password_reset_email,
        sender=sender,
        recipient=issued.recipient,
        code=issued.code,
        expires_in_minutes=max(1, math.ceil(settings.code_ttl_seconds / 60)),
        record_id=issued.record_id,
        bind=db.get_bind(),
    )
    response.headers["Cache-Control"] = "no-store"
    return PasswordResetRequestResponse(
        message=GENERIC_REQUEST_MESSAGE,
        request_token=issued.request_token,
        expires_in_seconds=settings.code_ttl_seconds,
        retry_after_seconds=settings.resend_cooldown_seconds,
    )


@router.post("/verify", response_model=PasswordResetVerifyResponse)
def verify_password_reset(
    payload: PasswordResetVerifyPayload,
    response: Response,
    settings: PasswordResetSettings = Depends(password_reset_settings_dependency),
    db: Session = Depends(get_db),
) -> PasswordResetVerifyResponse:
    try:
        verified = verify_password_reset_code(
            db,
            request_token=payload.request_token,
            code=payload.code,
            settings=settings,
        )
    except InvalidPasswordResetCode as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid or expired recovery code",
        ) from exc
    except PasswordResetAttemptsExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="maximum recovery-code attempts exceeded",
        ) from exc

    response.headers["Cache-Control"] = "no-store"
    return PasswordResetVerifyResponse(
        reset_token=verified.reset_token,
        expires_in_seconds=settings.reset_token_ttl_seconds,
    )


@router.post("/complete", status_code=status.HTTP_204_NO_CONTENT)
def finish_password_reset(
    payload: PasswordResetCompletePayload,
    db: Session = Depends(get_db),
) -> Response:
    try:
        complete_password_reset(
            db,
            reset_token=payload.reset_token,
            new_password=payload.new_password,
        )
    except InvalidPasswordResetToken as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid or expired password reset token",
        ) from exc
    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={"Cache-Control": "no-store"},
    )
