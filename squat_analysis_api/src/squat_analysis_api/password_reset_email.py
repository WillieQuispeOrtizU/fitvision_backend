from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr
from typing import Protocol


class PasswordResetEmailConfigurationError(RuntimeError):
    """Raised when the selected email provider is not safely configured."""


class PasswordResetEmailDeliveryError(RuntimeError):
    """Credential-free error used when delivery fails at runtime."""


class PasswordResetEmailSender(Protocol):
    def send_code(
        self,
        *,
        recipient: str,
        code: str,
        expires_in_minutes: int,
    ) -> None: ...


@dataclass(frozen=True)
class SmtpSettings:
    host: str
    port: int
    username: str | None
    password: str | None
    from_email: str
    from_name: str
    security: str
    timeout_seconds: float

    @classmethod
    def from_environment(cls) -> SmtpSettings:
        provider = os.getenv("PASSWORD_RESET_EMAIL_PROVIDER", "").strip().lower()
        if provider != "smtp":
            raise PasswordResetEmailConfigurationError(
                "PASSWORD_RESET_EMAIL_PROVIDER must be configured as smtp"
            )

        host = os.getenv("SMTP_HOST", "").strip()
        from_email = os.getenv("SMTP_FROM_EMAIL", "").strip()
        if not host or not from_email:
            raise PasswordResetEmailConfigurationError(
                "SMTP_HOST and SMTP_FROM_EMAIL are required"
            )

        port = _positive_int("SMTP_PORT", default=587)
        timeout_seconds = _positive_float("SMTP_TIMEOUT_SECONDS", default=15.0)
        security = os.getenv("SMTP_SECURITY", "starttls").strip().lower()
        if security not in {"starttls", "ssl", "plain"}:
            raise PasswordResetEmailConfigurationError(
                "SMTP_SECURITY must be starttls, ssl, or plain"
            )

        username = os.getenv("SMTP_USERNAME", "").strip() or None
        password = os.getenv("SMTP_PASSWORD", "") or None
        if (username is None) != (password is None):
            raise PasswordResetEmailConfigurationError(
                "SMTP_USERNAME and SMTP_PASSWORD must be configured together"
            )
        if security == "plain" and (
            host.lower() not in {"localhost", "127.0.0.1", "::1"}
            or username is not None
        ):
            raise PasswordResetEmailConfigurationError(
                "plain SMTP is allowed only on loopback without credentials"
            )

        return cls(
            host=host,
            port=port,
            username=username,
            password=password,
            from_email=from_email,
            from_name=os.getenv("SMTP_FROM_NAME", "FitVision").strip()
            or "FitVision",
            security=security,
            timeout_seconds=timeout_seconds,
        )


class SmtpPasswordResetEmailSender:
    def __init__(self, settings: SmtpSettings) -> None:
        self._settings = settings

    def send_code(
        self,
        *,
        recipient: str,
        code: str,
        expires_in_minutes: int,
    ) -> None:
        message = EmailMessage()
        message["Subject"] = "Código de recuperación de FitVision"
        message["From"] = formataddr(
            (self._settings.from_name, self._settings.from_email)
        )
        message["To"] = recipient
        message.set_content(
            "Usa este código para restablecer tu contraseña de FitVision:\n\n"
            f"{code}\n\n"
            f"El código vence en {expires_in_minutes} minutos. "
            "Si no solicitaste este cambio, ignora este correo."
        )

        try:
            if self._settings.security == "ssl":
                with smtplib.SMTP_SSL(
                    self._settings.host,
                    self._settings.port,
                    timeout=self._settings.timeout_seconds,
                    context=ssl.create_default_context(),
                ) as client:
                    self._authenticate_and_send(client, message)
                return

            with smtplib.SMTP(
                self._settings.host,
                self._settings.port,
                timeout=self._settings.timeout_seconds,
            ) as client:
                client.ehlo()
                if self._settings.security == "starttls":
                    client.starttls(context=ssl.create_default_context())
                    client.ehlo()
                self._authenticate_and_send(client, message)
        except (OSError, smtplib.SMTPException) as exc:
            raise PasswordResetEmailDeliveryError(
                "password reset email delivery failed"
            ) from exc

    def _authenticate_and_send(
        self, client: smtplib.SMTP, message: EmailMessage
    ) -> None:
        if self._settings.username is not None:
            client.login(self._settings.username, self._settings.password or "")
        client.send_message(message)


def get_password_reset_email_sender() -> PasswordResetEmailSender:
    return SmtpPasswordResetEmailSender(SmtpSettings.from_environment())


def _positive_int(name: str, *, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise PasswordResetEmailConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise PasswordResetEmailConfigurationError(f"{name} must be positive")
    return value


def _positive_float(name: str, *, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError as exc:
        raise PasswordResetEmailConfigurationError(f"{name} must be numeric") from exc
    if value <= 0:
        raise PasswordResetEmailConfigurationError(f"{name} must be positive")
    return value
