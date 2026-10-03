from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from squat_analysis_api.api_server import create_app
from squat_analysis_api.database import get_db
from squat_analysis_api.models import AppUser, AuthSession, Base, UserProgress


class AuthApiTest(unittest.TestCase):
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

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        app = create_app(
            jobs_dir=Path(self._tmp.name),
            assistant_url="http://assistant.test",
            database_healthcheck=lambda: "ok",
        )
        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.engine.dispose()
        self._tmp.cleanup()

    def test_register_normalizes_email_creates_progress_and_hashes_secrets(self):
        response = self._register(email="  USER@Example.COM ")

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["user"]["email"], "user@example.com")
        self.assertEqual(payload["token_type"], "bearer")
        token = payload["access_token"]

        with self.session_factory() as db:
            user = db.scalar(select(AppUser))
            self.assertIsNotNone(user)
            self.assertNotEqual(user.password_hash, "password")
            self.assertTrue(user.password_hash.startswith("$argon2id$"))

            progress = db.get(UserProgress, user.id)
            self.assertIsNotNone(progress)
            self.assertEqual(progress.total_xp, 0)
            self.assertEqual(progress.level, 1)

            auth_session = db.scalar(select(AuthSession))
            expected_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            self.assertEqual(auth_session.token_hash, expected_hash)
            self.assertNotEqual(auth_session.token_hash, token)

    def test_duplicate_email_is_rejected_case_insensitively(self):
        self.assertEqual(self._register(email="user@example.com").status_code, 201)

        duplicate = self._register(email="USER@EXAMPLE.COM")

        self.assertEqual(duplicate.status_code, 409)

    def test_login_and_current_user(self):
        self._register()

        login = self.client.post(
            "/api/auth/login",
            json={"email": "USER@EXAMPLE.COM", "password": "password"},
        )

        self.assertEqual(login.status_code, 200)
        token = login.json()["access_token"]
        me = self.client.get("/api/users/me", headers=self._auth(token))
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["full_name"], "Nombre")

        with self.session_factory() as db:
            session_count = db.scalar(select(func.count()).select_from(AuthSession))
            self.assertEqual(session_count, 2)

    def test_invalid_login_and_missing_bearer_are_unauthorized(self):
        self._register()

        invalid = self.client.post(
            "/api/auth/login",
            json={"email": "user@example.com", "password": "wrong"},
        )
        missing = self.client.get("/api/users/me")

        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(invalid.headers["www-authenticate"], "Bearer")
        self.assertEqual(missing.status_code, 401)

    def test_patch_updates_only_public_profile_fields(self):
        token = self._register().json()["access_token"]

        updated = self.client.patch(
            "/api/users/me",
            headers=self._auth(token),
            json={
                "full_name": "Nombre Actualizado",
                "height_cm": 175.5,
                "training_level": "intermediate",
            },
        )
        forbidden = self.client.patch(
            "/api/users/me",
            headers=self._auth(token),
            json={"id": 999, "password_hash": "unsafe"},
        )

        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["full_name"], "Nombre Actualizado")
        self.assertEqual(updated.json()["height_cm"], 175.5)
        self.assertEqual(forbidden.status_code, 422)

    def test_logout_revokes_only_the_presented_session(self):
        token = self._register().json()["access_token"]

        logout = self.client.post("/api/auth/logout", headers=self._auth(token))
        after_logout = self.client.get("/api/users/me", headers=self._auth(token))

        self.assertEqual(logout.status_code, 204)
        self.assertEqual(after_logout.status_code, 401)
        with self.session_factory() as db:
            auth_session = db.scalar(select(AuthSession))
            self.assertIsNotNone(auth_session.revoked_at)

    def test_expired_session_is_rejected_and_health_reports_database(self):
        token = self._register().json()["access_token"]
        with self.session_factory() as db:
            auth_session = db.scalar(select(AuthSession))
            now = datetime.now(timezone.utc)
            auth_session.created_at = now - timedelta(hours=2)
            auth_session.expires_at = now - timedelta(hours=1)
            db.commit()

        expired = self.client.get("/api/users/me", headers=self._auth(token))
        health = self.client.get("/api/health")

        self.assertEqual(expired.status_code, 401)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["database"]["status"], "ok")

    def _register(self, *, email: str = "user@example.com"):
        return self.client.post(
            "/api/auth/register",
            json={
                "email": email,
                "password": "password",
                "full_name": "Nombre",
                "birth_date": "2002-01-01",
                "height_cm": 173,
                "weight_kg": 65,
                "training_level": "beginner",
            },
        )

    @staticmethod
    def _auth(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}


if __name__ == "__main__":
    unittest.main()
