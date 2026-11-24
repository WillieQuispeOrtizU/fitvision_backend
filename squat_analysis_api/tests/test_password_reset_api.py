from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from squat_analysis_api.api_server import create_app
from squat_analysis_api.auth import verify_password
from squat_analysis_api.database import get_db
from squat_analysis_api.models import (
    AppUser,
    AuthSession,
    Base,
    PasswordResetRecord,
)
from squat_analysis_api.password_reset_routes import (
    GENERIC_REQUEST_MESSAGE,
    password_reset_sender_dependency,
    password_reset_settings_dependency,
)
from squat_analysis_api.password_reset_service import (
    PasswordResetSettings,
    PasswordResetConfigurationError,
    complete_password_reset,
    resolve_requester_ip,
)


class FakeEmailSender:
    def __init__(self) -> None:
        self.deliveries: list[dict[str, object]] = []
        self.last_attempt: dict[str, object] | None = None
        self.fail = False

    def send_code(
        self,
        *,
        recipient: str,
        code: str,
        expires_in_minutes: int,
    ) -> None:
        self.last_attempt = {
            "recipient": recipient,
            "code": code,
            "expires_in_minutes": expires_in_minutes,
        }
        if self.fail:
            raise RuntimeError(f"{recipient}:{code}:simulated provider outage")
        self.deliveries.append(self.last_attempt)


class PasswordResetApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=Session,
        )
        self.settings = PasswordResetSettings(pepper=b"t" * 32)
        self.sender = FakeEmailSender()

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        self.app = create_app(
            jobs_dir=Path(self._tmp.name),
            assistant_url="http://assistant.test",
            database_healthcheck=lambda: "ok",
        )
        self.app.dependency_overrides[get_db] = override_get_db
        self.app.dependency_overrides[password_reset_settings_dependency] = (
            lambda: self.settings
        )
        self.app.dependency_overrides[password_reset_sender_dependency] = (
            lambda: self.sender
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.engine.dispose()
        self._tmp.cleanup()

    def test_existing_account_receives_six_digit_code_and_only_hashes_persist(self):
        self._register()

        response = self._request_reset(" USER@example.com ")

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["message"], GENERIC_REQUEST_MESSAGE)
        self.assertEqual(payload["expires_in_seconds"], 600)
        self.assertEqual(payload["retry_after_seconds"], 60)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(len(self.sender.deliveries), 1)
        delivery = self.sender.deliveries[0]
        code = str(delivery["code"])
        self.assertRegex(code, r"^\d{6}$")
        self.assertEqual(delivery["recipient"], "user@example.com")

        with self.session_factory() as db:
            record = db.scalar(select(PasswordResetRecord))
            self.assertIsNotNone(record.user_id)
            self.assertNotEqual(record.code_hash, code)
            self.assertEqual(len(record.code_hash), 64)
            self.assertEqual(
                record.request_token_hash,
                hashlib.sha256(payload["request_token"].encode()).hexdigest(),
            )
            self.assertNotEqual(record.request_token_hash, payload["request_token"])
            self.assertNotIn("user@example.com", record.email_hash)

    def test_unknown_account_has_indistinguishable_response_and_dummy_record(self):
        known_registration = self._register()
        self.assertEqual(known_registration.status_code, 201)
        known = self._request_reset("user@example.com")
        unknown = self._request_reset("missing@example.com")

        self.assertEqual(known.status_code, unknown.status_code)
        for key in ("message", "expires_in_seconds", "retry_after_seconds"):
            self.assertEqual(known.json()[key], unknown.json()[key])
        self.assertEqual(set(known.json()), set(unknown.json()))
        self.assertEqual(len(self.sender.deliveries), 1)

        with self.session_factory() as db:
            records = db.scalars(
                select(PasswordResetRecord).order_by(PasswordResetRecord.id)
            ).all()
            self.assertEqual(len(records), 2)
            self.assertIsNotNone(records[0].user_id)
            self.assertIsNone(records[1].user_id)

    def test_complete_reset_changes_argon2_hash_and_revokes_all_old_sessions(self):
        registration = self._register()
        first_token = registration.json()["access_token"]
        second_login = self.client.post(
            "/api/auth/login",
            json={"email": "user@example.com", "password": "password"},
        )
        second_token = second_login.json()["access_token"]

        request_payload, code = self._issue_for_existing_account()
        verification = self.client.post(
            "/api/auth/password-reset/verify",
            json={"request_token": request_payload["request_token"], "code": code},
        )
        self.assertEqual(verification.status_code, 200)
        self.assertEqual(verification.headers["cache-control"], "no-store")
        reset_token = verification.json()["reset_token"]
        with self.session_factory() as db:
            record = db.scalar(select(PasswordResetRecord))
            self.assertEqual(
                record.reset_token_hash,
                hashlib.sha256(reset_token.encode()).hexdigest(),
            )
            self.assertNotEqual(record.reset_token_hash, reset_token)

        completed = self.client.post(
            "/api/auth/password-reset/complete",
            json={"reset_token": reset_token, "new_password": "new-password"},
        )

        self.assertEqual(completed.status_code, 204)
        self.assertEqual(completed.headers["cache-control"], "no-store")
        old_login = self.client.post(
            "/api/auth/login",
            json={"email": "user@example.com", "password": "password"},
        )
        new_login = self.client.post(
            "/api/auth/login",
            json={"email": "user@example.com", "password": "new-password"},
        )
        self.assertEqual(old_login.status_code, 401)
        self.assertEqual(new_login.status_code, 200)
        for old_token in (first_token, second_token):
            me = self.client.get(
                "/api/users/me",
                headers={"Authorization": f"Bearer {old_token}"},
            )
            self.assertEqual(me.status_code, 401)

        with self.session_factory() as db:
            user = db.scalar(select(AppUser))
            self.assertTrue(user.password_hash.startswith("$argon2id$"))
            self.assertTrue(verify_password(user.password_hash, "new-password"))
            sessions = db.scalars(
                select(AuthSession).order_by(AuthSession.id)
            ).all()
            self.assertIsNotNone(sessions[0].revoked_at)
            self.assertIsNotNone(sessions[1].revoked_at)
            record = db.scalar(select(PasswordResetRecord))
            self.assertIsNotNone(record.used_at)

    def test_wrong_code_counts_attempts_and_fifth_failure_invalidates_request(self):
        self._register()
        request_payload, correct_code = self._issue_for_existing_account()
        wrong_code = "000000" if correct_code != "000000" else "999999"

        for _ in range(4):
            response = self.client.post(
                "/api/auth/password-reset/verify",
                json={
                    "request_token": request_payload["request_token"],
                    "code": wrong_code,
                },
            )
            self.assertEqual(response.status_code, 400)
        fifth = self.client.post(
            "/api/auth/password-reset/verify",
            json={
                "request_token": request_payload["request_token"],
                "code": wrong_code,
            },
        )
        after_limit = self.client.post(
            "/api/auth/password-reset/verify",
            json={
                "request_token": request_payload["request_token"],
                "code": correct_code,
            },
        )

        self.assertEqual(fifth.status_code, 429)
        self.assertEqual(after_limit.status_code, 400)
        with self.session_factory() as db:
            record = db.scalar(select(PasswordResetRecord))
            self.assertEqual(record.failed_attempts, 5)
            self.assertIsNotNone(record.invalidated_at)

    def test_expired_code_is_rejected(self):
        self._register()
        request_payload, code = self._issue_for_existing_account()
        with self.session_factory() as db:
            record = db.scalar(select(PasswordResetRecord))
            record.created_at = datetime.now(timezone.utc) - timedelta(minutes=11)
            record.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            db.commit()

        response = self.client.post(
            "/api/auth/password-reset/verify",
            json={"request_token": request_payload["request_token"], "code": code},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "invalid or expired recovery code")

    def test_verified_code_and_reset_token_are_single_use(self):
        self._register()
        request_payload, code = self._issue_for_existing_account()
        first_verify = self.client.post(
            "/api/auth/password-reset/verify",
            json={"request_token": request_payload["request_token"], "code": code},
        )
        repeated_verify = self.client.post(
            "/api/auth/password-reset/verify",
            json={"request_token": request_payload["request_token"], "code": code},
        )
        reset_token = first_verify.json()["reset_token"]
        first_complete = self.client.post(
            "/api/auth/password-reset/complete",
            json={"reset_token": reset_token, "new_password": "new-password"},
        )
        repeated_complete = self.client.post(
            "/api/auth/password-reset/complete",
            json={"reset_token": reset_token, "new_password": "other-password"},
        )

        self.assertEqual(first_verify.status_code, 200)
        self.assertEqual(repeated_verify.status_code, 400)
        self.assertEqual(first_complete.status_code, 204)
        self.assertEqual(repeated_complete.status_code, 400)

    def test_expired_reset_token_is_rejected(self):
        self._register()
        request_payload, code = self._issue_for_existing_account()
        verified = self.client.post(
            "/api/auth/password-reset/verify",
            json={"request_token": request_payload["request_token"], "code": code},
        )
        with self.session_factory() as db:
            record = db.scalar(select(PasswordResetRecord))
            record.reset_token_expires_at = datetime.now(timezone.utc) - timedelta(
                seconds=1
            )
            db.commit()

        response = self.client.post(
            "/api/auth/password-reset/complete",
            json={
                "reset_token": verified.json()["reset_token"],
                "new_password": "new-password",
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_invalid_new_password_does_not_consume_reset_token(self):
        self._register()
        request_payload, code = self._issue_for_existing_account()
        verified = self.client.post(
            "/api/auth/password-reset/verify",
            json={"request_token": request_payload["request_token"], "code": code},
        )
        reset_token = verified.json()["reset_token"]

        invalid = self.client.post(
            "/api/auth/password-reset/complete",
            json={"reset_token": reset_token, "new_password": "short"},
        )
        valid = self.client.post(
            "/api/auth/password-reset/complete",
            json={"reset_token": reset_token, "new_password": "long-enough"},
        )

        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(valid.status_code, 204)

    def test_cooldown_then_reissue_invalidates_previous_code(self):
        self._register()
        first_payload, first_code = self._issue_for_existing_account()

        blocked = self._request_reset("user@example.com")
        self.assertEqual(blocked.status_code, 429)
        self.assertGreaterEqual(int(blocked.headers["retry-after"]), 1)

        with self.session_factory() as db:
            first = db.scalar(select(PasswordResetRecord))
            first.created_at = datetime.now(timezone.utc) - timedelta(seconds=61)
            db.commit()
        second = self._request_reset("user@example.com")
        self.assertEqual(second.status_code, 202)

        old_code = self.client.post(
            "/api/auth/password-reset/verify",
            json={
                "request_token": first_payload["request_token"],
                "code": first_code,
            },
        )
        self.assertEqual(old_code.status_code, 400)
        with self.session_factory() as db:
            records = db.scalars(
                select(PasswordResetRecord).order_by(PasswordResetRecord.id)
            ).all()
            self.assertIsNotNone(records[0].invalidated_at)
            self.assertIsNone(records[1].invalidated_at)

    def test_hourly_limits_apply_per_email_and_per_ip(self):
        self.settings = replace(
            self.settings,
            resend_cooldown_seconds=1,
            max_requests_per_email=2,
            max_requests_per_ip=3,
        )
        self._register()
        self.assertEqual(self._request_reset("user@example.com").status_code, 202)
        self._age_all_requests(seconds=2)
        self.assertEqual(self._request_reset("user@example.com").status_code, 202)
        self._age_all_requests(seconds=2)
        self.assertEqual(self._request_reset("user@example.com").status_code, 429)

        # A fresh test IP is not available through TestClient, so reset the
        # records and demonstrate that distinct emails share the configured IP cap.
        with self.session_factory() as db:
            db.query(PasswordResetRecord).delete()
            db.commit()
        self.settings = replace(
            self.settings,
            max_requests_per_email=5,
            max_requests_per_ip=2,
        )
        self.assertEqual(self._request_reset("one@example.com").status_code, 202)
        self.assertEqual(self._request_reset("two@example.com").status_code, 202)
        self.assertEqual(self._request_reset("three@example.com").status_code, 429)

    def test_email_provider_failure_is_generic_and_invalidates_challenge(self):
        self._register()
        self.sender.fail = True
        with self.assertLogs(
            "squat_analysis_api.password_reset_service", level="WARNING"
        ) as captured_logs:
            existing = self._request_reset("user@example.com")
        unknown = self._request_reset("missing@example.com")

        self.assertEqual(existing.status_code, 202)
        self.assertEqual(unknown.status_code, 202)
        for key in ("message", "expires_in_seconds", "retry_after_seconds"):
            self.assertEqual(existing.json()[key], unknown.json()[key])
        with self.session_factory() as db:
            existing_record = db.scalar(
                select(PasswordResetRecord).where(
                    PasswordResetRecord.user_id.is_not(None)
                )
            )
            self.assertIsNotNone(existing_record.delivery_failed_at)
            self.assertIsNotNone(existing_record.invalidated_at)
        logs = "\n".join(captured_logs.output)
        self.assertNotIn("user@example.com", logs)
        self.assertNotIn(str(self.sender.last_attempt["code"]), logs)

    def test_missing_configuration_returns_same_503_before_account_lookup(self):
        self._register()
        settings_override = self.app.dependency_overrides.pop(
            password_reset_settings_dependency
        )
        sender_override = self.app.dependency_overrides.pop(
            password_reset_sender_dependency
        )
        try:
            with patch.dict(os.environ, {}, clear=True):
                existing = self._request_reset("user@example.com")
                unknown = self._request_reset("missing@example.com")
        finally:
            self.app.dependency_overrides[password_reset_settings_dependency] = (
                settings_override
            )
            self.app.dependency_overrides[password_reset_sender_dependency] = (
                sender_override
            )

        self.assertEqual(existing.status_code, 503)
        self.assertEqual(unknown.status_code, 503)
        self.assertEqual(existing.json(), unknown.json())
        with self.session_factory() as db:
            self.assertEqual(len(db.scalars(select(PasswordResetRecord)).all()), 0)

    def test_settings_reject_weak_pepper_and_cooldown_below_sixty_seconds(self):
        invalid_environments = (
            {},
            {"PASSWORD_RESET_PEPPER": "too-short"},
            {
                "PASSWORD_RESET_PEPPER": "replace-with-at-least-32-random-bytes"
            },
            {
                "PASSWORD_RESET_PEPPER": "x" * 32,
                "PASSWORD_RESET_RESEND_COOLDOWN_SECONDS": "59",
            },
        )
        for environment in invalid_environments:
            with self.subTest(environment=environment):
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(PasswordResetConfigurationError):
                        PasswordResetSettings.from_environment()

    def test_new_request_after_verification_invalidates_previous_reset_token(self):
        self._register()
        first_request, first_code = self._issue_for_existing_account()
        first_verify = self.client.post(
            "/api/auth/password-reset/verify",
            json={
                "request_token": first_request["request_token"],
                "code": first_code,
            },
        )
        with self.session_factory() as db:
            record = db.scalar(select(PasswordResetRecord))
            record.created_at = datetime.now(timezone.utc) - timedelta(seconds=61)
            db.commit()
        second_request = self._request_reset("user@example.com")
        old_complete = self.client.post(
            "/api/auth/password-reset/complete",
            json={
                "reset_token": first_verify.json()["reset_token"],
                "new_password": "new-password",
            },
        )
        self.assertEqual(second_request.status_code, 202)
        self.assertEqual(old_complete.status_code, 400)

    def test_complete_reset_rolls_back_password_session_and_token_on_db_failure(self):
        registration = self._register()
        session_token = registration.json()["access_token"]
        request_payload, code = self._issue_for_existing_account()
        verified = self.client.post(
            "/api/auth/password-reset/verify",
            json={"request_token": request_payload["request_token"], "code": code},
        )
        reset_token = verified.json()["reset_token"]

        with self.session_factory() as db:
            with patch.object(
                db, "commit", side_effect=SQLAlchemyError("simulated commit failure")
            ):
                with self.assertRaises(SQLAlchemyError):
                    complete_password_reset(
                        db,
                        reset_token=reset_token,
                        new_password="new-password",
                    )
            db.rollback()

        with self.session_factory() as db:
            user = db.scalar(select(AppUser))
            session = db.scalar(select(AuthSession))
            record = db.scalar(select(PasswordResetRecord))
            self.assertTrue(verify_password(user.password_hash, "password"))
            self.assertIsNone(session.revoked_at)
            self.assertIsNone(record.used_at)

        # The same reset token still works because the failed transaction made
        # no partial password/session/token change.
        completed = self.client.post(
            "/api/auth/password-reset/complete",
            json={"reset_token": reset_token, "new_password": "new-password"},
        )
        self.assertEqual(completed.status_code, 204)
        self.assertEqual(
            self.client.get(
                "/api/users/me",
                headers={"Authorization": f"Bearer {session_token}"},
            ).status_code,
            401,
        )

    def test_code_schema_requires_exactly_six_digits(self):
        for invalid_code in ("12345", "1234567", "abcdef"):
            with self.subTest(code=invalid_code):
                response = self.client.post(
                    "/api/auth/password-reset/verify",
                    json={"request_token": "x" * 32, "code": invalid_code},
                )
                self.assertEqual(response.status_code, 422)

    def test_forwarded_ip_is_used_only_for_explicitly_trusted_proxy(self):
        direct = resolve_requester_ip(
            peer_ip="127.0.0.1",
            forwarded_for="203.0.113.10",
            settings=self.settings,
        )
        trusted = resolve_requester_ip(
            peer_ip="127.0.0.1",
            forwarded_for="203.0.113.10, 127.0.0.1",
            settings=replace(
                self.settings, trusted_proxy_ips=frozenset({"127.0.0.1"})
            ),
        )
        self.assertEqual(direct, "127.0.0.1")
        self.assertEqual(trusted, "203.0.113.10")

    def _register(self):
        return self.client.post(
            "/api/auth/register",
            json={
                "email": "user@example.com",
                "password": "password",
                "full_name": "Nombre",
                "birth_date": "2002-01-01",
                "height_cm": 173,
                "weight_kg": 65,
                "training_level": "beginner",
            },
        )

    def _request_reset(self, email: str):
        return self.client.post(
            "/api/auth/password-reset/request", json={"email": email}
        )

    def _issue_for_existing_account(self) -> tuple[dict[str, object], str]:
        response = self._request_reset("user@example.com")
        self.assertEqual(response.status_code, 202)
        return response.json(), str(self.sender.deliveries[-1]["code"])

    def _age_all_requests(self, *, seconds: int) -> None:
        with self.session_factory() as db:
            records = db.scalars(select(PasswordResetRecord)).all()
            for record in records:
                record.created_at = datetime.now(timezone.utc) - timedelta(
                    seconds=seconds
                )
            db.commit()


if __name__ == "__main__":
    unittest.main()
