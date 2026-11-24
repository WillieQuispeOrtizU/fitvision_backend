from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from squat_analysis_api.password_reset_email import (
    PasswordResetEmailConfigurationError,
    PasswordResetEmailDeliveryError,
    SmtpPasswordResetEmailSender,
    SmtpSettings,
)


class FakeSmtpClient:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.ehlo_count = 0
        self.starttls_called = False
        self.login_credentials: tuple[str, str] | None = None
        self.message = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def ehlo(self) -> None:
        self.ehlo_count += 1

    def starttls(self, *, context) -> None:
        self.starttls_called = True

    def login(self, username: str, password: str) -> None:
        self.login_credentials = (username, password)

    def send_message(self, message) -> None:
        self.message = message


class PasswordResetEmailTest(unittest.TestCase):
    def test_missing_provider_configuration_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(PasswordResetEmailConfigurationError):
                SmtpSettings.from_environment()

    def test_plain_smtp_rejects_remote_hosts_and_credentials(self):
        base = {
            "PASSWORD_RESET_EMAIL_PROVIDER": "smtp",
            "SMTP_HOST": "smtp.example.test",
            "SMTP_SECURITY": "plain",
            "SMTP_FROM_EMAIL": "sender@example.test",
        }
        with patch.dict(os.environ, base, clear=True):
            with self.assertRaises(PasswordResetEmailConfigurationError):
                SmtpSettings.from_environment()

        with patch.dict(
            os.environ,
            {
                **base,
                "SMTP_HOST": "127.0.0.1",
                "SMTP_USERNAME": "sender",
                "SMTP_PASSWORD": "secret",
            },
            clear=True,
        ):
            with self.assertRaises(PasswordResetEmailConfigurationError):
                SmtpSettings.from_environment()

        with patch.dict(
            os.environ,
            {**base, "SMTP_HOST": "127.0.0.1"},
            clear=True,
        ):
            settings = SmtpSettings.from_environment()
        self.assertEqual(settings.security, "plain")

    def test_starttls_sender_authenticates_and_sends_real_message_shape(self):
        environment = {
            "PASSWORD_RESET_EMAIL_PROVIDER": "smtp",
            "SMTP_HOST": "smtp.example.test",
            "SMTP_PORT": "587",
            "SMTP_SECURITY": "starttls",
            "SMTP_USERNAME": "sender@example.test",
            "SMTP_PASSWORD": "application-password",
            "SMTP_FROM_EMAIL": "sender@example.test",
            "SMTP_FROM_NAME": "FitVision",
        }
        fake = FakeSmtpClient()
        with patch.dict(os.environ, environment, clear=True):
            settings = SmtpSettings.from_environment()
        with patch(
            "squat_analysis_api.password_reset_email.smtplib.SMTP",
            return_value=fake,
        ):
            SmtpPasswordResetEmailSender(settings).send_code(
                recipient="user@example.test",
                code="012345",
                expires_in_minutes=10,
            )

        self.assertTrue(fake.starttls_called)
        self.assertEqual(fake.ehlo_count, 2)
        self.assertEqual(
            fake.login_credentials,
            ("sender@example.test", "application-password"),
        )
        self.assertEqual(fake.message["To"], "user@example.test")
        self.assertIn("012345", fake.message.get_content())
        self.assertIn("10 minutos", fake.message.get_content())

    def test_runtime_smtp_error_is_wrapped_without_sensitive_details(self):
        settings = SmtpSettings(
            host="smtp.example.test",
            port=587,
            username=None,
            password=None,
            from_email="sender@example.test",
            from_name="FitVision",
            security="starttls",
            timeout_seconds=5,
        )
        with patch(
            "squat_analysis_api.password_reset_email.smtplib.SMTP",
            side_effect=OSError("secret provider detail"),
        ):
            with self.assertRaises(PasswordResetEmailDeliveryError) as captured:
                SmtpPasswordResetEmailSender(settings).send_code(
                    recipient="user@example.test",
                    code="123456",
                    expires_in_minutes=10,
                )
        self.assertNotIn("secret", str(captured.exception))
        self.assertNotIn("user@example.test", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
