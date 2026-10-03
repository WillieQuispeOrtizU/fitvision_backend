from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from squat_analysis_api.api_server import create_app
from squat_analysis_api.database import get_db
from squat_analysis_api.evaluation_service import complete_evaluation
from squat_analysis_api.gamification_catalog import ensure_gamification_catalog
from squat_analysis_api.gamification_service import (
    get_gamification_state,
    initialize_user_gamification,
    level_for_xp,
    rebuild_user_gamification,
)
from squat_analysis_api.models import (
    Achievement,
    AppUser,
    Base,
    Challenge,
    Evaluation,
    UserAchievement,
    UserChallenge,
    UserProgress,
    XpTransaction,
)


class GamificationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
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
        self._sequence = 0

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_new_user_has_zero_progress_and_active_challenges(self):
        user_id = self._create_user("new@example.com")

        state = self._state(user_id)

        self.assertEqual(
            state["progress"],
            {
                "total_xp": 0,
                "level": 1,
                "current_streak": 0,
                "longest_streak": 0,
                "total_evaluations": 0,
                "correct_evaluations": 0,
                "needs_improvement_evaluations": 0,
                "total_repetitions": 0,
                "correct_repetitions": 0,
                "incorrect_repetitions": 0,
                "xp_at_level_start": 0,
                "xp_for_next_level": 250,
                "xp_within_level": 0,
                "xp_required_within_level": 250,
            },
        )
        self.assertEqual(len(state["achievements"]), 5)
        self.assertFalse(any(item["unlocked"] for item in state["achievements"]))
        self.assertEqual(len(state["challenges"]), 4)
        self.assertTrue(
            all(item["status"] == "active" for item in state["challenges"])
        )

    def test_correct_evaluation_awards_100_xp(self):
        user_id = self._create_user("correct@example.com")
        evaluation_id = self._complete(
            user_id,
            when=self._day(1),
            correct=3,
            incorrect=0,
        )

        self.assertEqual(self._state(user_id)["progress"]["total_xp"], 100)
        self.assertEqual(
            self._xp_rows(evaluation_id),
            [(100, "evaluation_completed_correct")],
        )

    def test_needs_improvement_evaluation_awards_70_xp(self):
        user_id = self._create_user("needs@example.com")
        evaluation_id = self._complete(
            user_id,
            when=self._day(1),
            correct=2,
            incorrect=1,
        )

        self.assertEqual(self._state(user_id)["progress"]["total_xp"], 70)
        self.assertEqual(
            self._xp_rows(evaluation_id),
            [(70, "evaluation_completed_needs_improvement")],
        )

    def test_zero_repetitions_keeps_classification_but_awards_no_xp(self):
        user_id = self._create_user("zero@example.com")
        evaluation_id = self._complete(
            user_id,
            when=self._day(1),
            correct=0,
            incorrect=0,
            mode="pro",
        )

        with self.session_factory() as db:
            evaluation = db.get(Evaluation, evaluation_id)
            self.assertIs(evaluation.correct, True)
            self.assertEqual(db.scalar(select(func.count(XpTransaction.id))), 0)
            try_pro = db.scalar(
                select(UserChallenge)
                .join(Challenge)
                .where(
                    UserChallenge.user_id == user_id,
                    Challenge.code == "try_pro",
                )
            )
            self.assertEqual(try_pro.progress_value, 0)
            self.assertEqual(try_pro.status, "active")
        self.assertEqual(self._state(user_id)["progress"]["total_xp"], 0)

    def test_duplicate_completion_does_not_duplicate_any_reward(self):
        user_id = self._create_user("duplicate@example.com")
        when = self._day(1)
        evaluation_id, job_id = self._processing_evaluation(
            user_id,
            when=when,
        )
        report = self._report(correct=1, incorrect=0)

        with patch(
            "squat_analysis_api.evaluation_service._utc_now",
            return_value=when,
        ):
            first = complete_evaluation(
                self.session_factory,
                job_id=job_id,
                report_json=report,
                original_video_url="/original.mp4",
                annotated_video_url="/annotated.mp4",
                video_seconds=1.0,
                assistant_seconds=0.1,
            )
            second = complete_evaluation(
                self.session_factory,
                job_id=job_id,
                report_json=report,
                original_video_url="/original.mp4",
                annotated_video_url="/annotated.mp4",
                video_seconds=1.0,
                assistant_seconds=0.1,
            )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(self._xp_rows(evaluation_id), [(100, "evaluation_completed_correct")])
        self.assertEqual(self._state(user_id)["progress"]["total_evaluations"], 1)

    def test_level_formula_has_exact_250_xp_bands(self):
        self.assertEqual(level_for_xp(0), 1)
        self.assertEqual(level_for_xp(249), 1)
        self.assertEqual(level_for_xp(250), 2)
        self.assertEqual(level_for_xp(499), 2)
        self.assertEqual(level_for_xp(500), 3)

    def test_completed_evaluations_persist_level_after_crossing_250_xp(self):
        user_id = self._create_user("persisted-level@example.com")
        self._complete(user_id, when=self._day(1), correct=1, incorrect=0)
        self._complete(user_id, when=self._day(2), correct=1, incorrect=0)

        with self.session_factory() as db:
            progress = db.get(UserProgress, user_id)
            transaction_sum = db.scalar(
                select(func.coalesce(func.sum(XpTransaction.amount), 0)).where(
                    XpTransaction.user_id == user_id
                )
            )

            # Two 100 XP completions plus the one-time 100 XP two_correct
            # challenge cross the first 250 XP boundary.
            self.assertEqual(transaction_sum, 300)
            self.assertEqual(progress.total_xp, 300)
            self.assertEqual(progress.level, 2)

    def test_streak_same_day_counts_once(self):
        user_id = self._create_user("same-day@example.com")
        self._complete(user_id, when=self._day(1, 10), correct=1, incorrect=0)
        self._complete(user_id, when=self._day(1, 18), correct=0, incorrect=1)

        progress = self._state(user_id)["progress"]
        self.assertEqual(progress["current_streak"], 1)
        self.assertEqual(progress["longest_streak"], 1)

    def test_streak_consecutive_lima_days_increments(self):
        user_id = self._create_user("consecutive@example.com")
        self._complete(user_id, when=self._day(1), correct=1, incorrect=0)
        self._complete(user_id, when=self._day(2), correct=1, incorrect=0)

        progress = self._state(user_id)["progress"]
        self.assertEqual(progress["current_streak"], 2)
        self.assertEqual(progress["longest_streak"], 2)

    def test_streak_uses_lima_calendar_boundary(self):
        user_id = self._create_user("lima-boundary@example.com")
        # 04:30 UTC is 23:30 on the prior calendar day in Lima; 05:30 UTC is
        # 00:30 the next Lima day even though only one real hour elapsed.
        first = datetime(2026, 1, 2, 4, 30, tzinfo=timezone.utc)
        second = datetime(2026, 1, 2, 5, 30, tzinfo=timezone.utc)
        self._complete(user_id, when=first, correct=1, incorrect=0)
        self._complete(user_id, when=second, correct=1, incorrect=0)

        self.assertEqual(self._state(user_id)["progress"]["current_streak"], 2)

    def test_streak_gap_resets_and_longest_is_preserved(self):
        user_id = self._create_user("gap@example.com")
        self._complete(user_id, when=self._day(1), correct=1, incorrect=0)
        self._complete(user_id, when=self._day(2), correct=1, incorrect=0)
        self._complete(user_id, when=self._day(5), correct=1, incorrect=0)

        progress = self._state(user_id)["progress"]
        self.assertEqual(progress["current_streak"], 1)
        self.assertEqual(progress["longest_streak"], 2)

    def test_first_evaluation_and_first_correct_unlock_once(self):
        user_id = self._create_user("first-achievements@example.com")
        self._complete(user_id, when=self._day(1), correct=0, incorrect=1)
        self.assertEqual(self._unlocked_codes(user_id), {"first_evaluation"})

        self._complete(user_id, when=self._day(2), correct=1, incorrect=0)
        self.assertEqual(
            self._unlocked_codes(user_id),
            {"first_evaluation", "first_correct"},
        )

    def test_five_and_ten_evaluation_achievements_unlock(self):
        user_id = self._create_user("volume-achievements@example.com")
        for index in range(5):
            self._complete(
                user_id,
                when=self._day(index + 1),
                correct=0,
                incorrect=1,
            )
        self.assertIn("five_evaluations", self._unlocked_codes(user_id))
        self.assertNotIn("ten_evaluations", self._unlocked_codes(user_id))

        for index in range(5, 10):
            self._complete(
                user_id,
                when=self._day(index + 1),
                correct=0,
                incorrect=1,
            )
        self.assertIn("ten_evaluations", self._unlocked_codes(user_id))

    def test_perfectionist_unlocks_after_five_correct_evaluations(self):
        user_id = self._create_user("perfectionist@example.com")
        for index in range(5):
            self._complete(
                user_id,
                when=self._day(index + 1),
                correct=1,
                incorrect=0,
            )

        self.assertIn("perfectionist", self._unlocked_codes(user_id))

    def test_complete_three_reward_is_granted_exactly_once(self):
        user_id = self._create_user("complete-three@example.com")
        ids = [
            self._complete(
                user_id,
                when=self._day(index + 1),
                correct=0,
                incorrect=1,
            )
            for index in range(4)
        ]

        self.assertEqual(self._reason_count(user_id, "challenge:complete_three"), 1)
        self.assertIn((120, "challenge:complete_three"), self._xp_rows(ids[2]))
        self.assertNotIn((120, "challenge:complete_three"), self._xp_rows(ids[3]))

    def test_two_correct_reward_is_granted_exactly_once(self):
        user_id = self._create_user("two-correct@example.com")
        self._complete(user_id, when=self._day(1), correct=1, incorrect=0)
        second = self._complete(user_id, when=self._day(2), correct=1, incorrect=0)
        self._complete(user_id, when=self._day(3), correct=1, incorrect=0)

        self.assertEqual(self._reason_count(user_id, "challenge:two_correct"), 1)
        self.assertIn((100, "challenge:two_correct"), self._xp_rows(second))

    def test_try_pro_reward_is_granted_exactly_once(self):
        user_id = self._create_user("try-pro@example.com")
        first = self._complete(
            user_id,
            when=self._day(1),
            correct=1,
            incorrect=0,
            mode="pro",
        )
        self._complete(
            user_id,
            when=self._day(2),
            correct=1,
            incorrect=0,
            mode="pro",
        )

        self.assertEqual(self._reason_count(user_id, "challenge:try_pro"), 1)
        self.assertIn((80, "challenge:try_pro"), self._xp_rows(first))

    def test_improve_previous_uses_immediately_previous_classification_once(self):
        user_id = self._create_user("improve@example.com")
        self._complete(user_id, when=self._day(1), correct=0, incorrect=1)
        improved = self._complete(
            user_id,
            when=self._day(2),
            correct=1,
            incorrect=0,
        )
        self._complete(user_id, when=self._day(3), correct=1, incorrect=0)

        self.assertEqual(
            self._reason_count(user_id, "challenge:improve_previous"), 1
        )
        self.assertIn((150, "challenge:improve_previous"), self._xp_rows(improved))

    def test_gamification_failure_rolls_back_evaluation_and_all_rewards(self):
        user_id = self._create_user("rollback@example.com")
        when = self._day(1)
        _, job_id = self._processing_evaluation(user_id, when=when)

        def fail_after_mutation(db, *, evaluation, progress, occurred_at):
            progress.total_xp += 999
            db.add(
                XpTransaction(
                    user_id=evaluation.user_id,
                    evaluation_id=evaluation.id,
                    amount=999,
                    reason="test_failure",
                    created_at=occurred_at,
                )
            )
            achievement = db.scalar(
                select(Achievement).where(
                    Achievement.code == "first_evaluation"
                )
            )
            challenge_state = db.scalar(
                select(UserChallenge)
                .join(Challenge)
                .where(
                    UserChallenge.user_id == evaluation.user_id,
                    Challenge.code == "complete_three",
                )
            )
            db.add(
                UserAchievement(
                    user_id=evaluation.user_id,
                    achievement_id=achievement.id,
                    evaluation_id=evaluation.id,
                    unlocked_at=occurred_at,
                )
            )
            challenge_state.progress_value = 3
            challenge_state.status = "completed"
            challenge_state.completed_at = occurred_at
            challenge_state.updated_at = occurred_at
            db.flush()
            raise RuntimeError("gamification exploded")

        with (
            patch(
                "squat_analysis_api.evaluation_service._utc_now",
                return_value=when,
            ),
            patch(
                "squat_analysis_api.evaluation_service.apply_completed_evaluation",
                side_effect=fail_after_mutation,
            ),
            self.assertRaisesRegex(RuntimeError, "gamification exploded"),
        ):
            complete_evaluation(
                self.session_factory,
                job_id=job_id,
                report_json=self._report(correct=1, incorrect=0),
                original_video_url="/original.mp4",
                annotated_video_url="/annotated.mp4",
                video_seconds=1.0,
                assistant_seconds=0.1,
            )

        with self.session_factory() as db:
            evaluation = db.scalar(
                select(Evaluation).where(Evaluation.job_id == job_id)
            )
            progress = db.get(UserProgress, user_id)
            self.assertEqual(evaluation.status, "processing")
            self.assertEqual(progress.total_xp, 0)
            self.assertEqual(progress.total_evaluations, 0)
            self.assertEqual(db.scalar(select(func.count(XpTransaction.id))), 0)
            self.assertEqual(db.scalar(select(func.count(UserAchievement.id))), 0)
            complete_three = db.scalar(
                select(UserChallenge)
                .join(Challenge)
                .where(
                    UserChallenge.user_id == user_id,
                    Challenge.code == "complete_three",
                )
            )
            self.assertEqual(complete_three.progress_value, 0)
            self.assertEqual(complete_three.status, "active")
            self.assertIsNone(complete_three.completed_at)

    def test_catalog_seed_and_backfill_are_idempotent(self):
        user_id = self._create_user("backfill@example.com")
        self._historical_evaluation(
            user_id,
            when=self._day(1),
            correct=False,
            correct_reps=0,
            incorrect_reps=1,
        )
        self._historical_evaluation(
            user_id,
            when=self._day(2),
            correct=True,
            correct_reps=1,
            incorrect_reps=0,
        )

        with self.session_factory() as db, db.begin():
            ensure_gamification_catalog(db)
            first_ids = list(db.scalars(select(Achievement.id).order_by(Achievement.code)))
            ensure_gamification_catalog(db)
            second_ids = list(db.scalars(select(Achievement.id).order_by(Achievement.code)))
            self.assertEqual(first_ids, second_ids)

        first = self._run_backfill(user_id)
        second = self._run_backfill(user_id)

        self.assertEqual(first, second)
        self.assertEqual(first["progress"]["total_xp"], 320)
        self.assertEqual(self._reason_count(user_id, "challenge:improve_previous"), 1)
        with self.session_factory() as db:
            transaction_sum = db.scalar(
                select(func.coalesce(func.sum(XpTransaction.amount), 0)).where(
                    XpTransaction.user_id == user_id
                )
            )
        self.assertEqual(transaction_sum, first["progress"]["total_xp"])

    def _create_user(self, email: str) -> int:
        now = self._day(0)
        with self.session_factory() as db:
            user = AppUser(
                email=email,
                password_hash="not-used",
                full_name="Test User",
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            db.add(user)
            db.flush()
            db.add(
                UserProgress(
                    user_id=user.id,
                    total_xp=0,
                    level=1,
                    current_streak=0,
                    longest_streak=0,
                    total_evaluations=0,
                    total_correct_reps=0,
                    total_incorrect_reps=0,
                    updated_at=now,
                )
            )
            db.flush()
            initialize_user_gamification(db, user_id=user.id, started_at=now)
            db.commit()
            return user.id

    def _complete(
        self,
        user_id: int,
        *,
        when: datetime,
        correct: int,
        incorrect: int,
        mode: str = "beginner",
    ) -> int:
        evaluation_id, job_id = self._processing_evaluation(
            user_id,
            when=when,
            mode=mode,
        )
        with patch(
            "squat_analysis_api.evaluation_service._utc_now",
            return_value=when,
        ):
            completed = complete_evaluation(
                self.session_factory,
                job_id=job_id,
                report_json=self._report(correct=correct, incorrect=incorrect),
                original_video_url="/original.mp4",
                annotated_video_url="/annotated.mp4",
                video_seconds=1.0,
                assistant_seconds=0.1,
            )
        self.assertTrue(completed)
        return evaluation_id

    def _processing_evaluation(
        self,
        user_id: int,
        *,
        when: datetime,
        mode: str = "beginner",
    ) -> tuple[int, str]:
        self._sequence += 1
        job_id = f"job-{self._sequence}"
        with self.session_factory() as db:
            evaluation = Evaluation(
                user_id=user_id,
                job_id=job_id,
                exercise="squat",
                mode=mode,
                status="processing",
                input_filename="squat.mp4",
                original_video_url="/original.mp4",
                report_json={},
                created_at=when,
                updated_at=when,
            )
            db.add(evaluation)
            db.commit()
            return evaluation.id, job_id

    def _historical_evaluation(
        self,
        user_id: int,
        *,
        when: datetime,
        correct: bool,
        correct_reps: int,
        incorrect_reps: int,
    ) -> None:
        self._sequence += 1
        with self.session_factory() as db:
            db.add(
                Evaluation(
                    user_id=user_id,
                    job_id=f"historical-{self._sequence}",
                    exercise="squat",
                    mode="beginner",
                    status="completed",
                    correct=correct,
                    total_reps=correct_reps + incorrect_reps,
                    correct_reps=correct_reps,
                    incorrect_reps=incorrect_reps,
                    report_json={},
                    created_at=when,
                    updated_at=when,
                    completed_at=when,
                )
            )
            db.commit()

    def _state(self, user_id: int):
        with self.session_factory() as db:
            state = get_gamification_state(db, user_id=user_id)
            db.commit()
            return state

    def _run_backfill(self, user_id: int):
        with self.session_factory() as db, db.begin():
            rebuild_user_gamification(db, user_id=user_id)
        return self._state(user_id)

    def _xp_rows(self, evaluation_id: int) -> list[tuple[int, str]]:
        with self.session_factory() as db:
            return list(
                db.execute(
                    select(XpTransaction.amount, XpTransaction.reason)
                    .where(XpTransaction.evaluation_id == evaluation_id)
                    .order_by(XpTransaction.id)
                ).all()
            )

    def _reason_count(self, user_id: int, reason: str) -> int:
        with self.session_factory() as db:
            return int(
                db.scalar(
                    select(func.count(XpTransaction.id)).where(
                        XpTransaction.user_id == user_id,
                        XpTransaction.reason == reason,
                    )
                )
                or 0
            )

    def _unlocked_codes(self, user_id: int) -> set[str]:
        with self.session_factory() as db:
            return set(
                db.scalars(
                    select(Achievement.code)
                    .join(UserAchievement)
                    .where(UserAchievement.user_id == user_id)
                ).all()
            )

    @staticmethod
    def _report(*, correct: int, incorrect: int):
        return {
            "stats": {
                "correct": correct,
                "incorrect": incorrect,
                "total_reps": correct + incorrect,
            },
            "reps": [],
        }

    @staticmethod
    def _day(day: int, hour: int = 12) -> datetime:
        return datetime(2026, 1, 1, hour, tzinfo=timezone.utc) + timedelta(days=day)


class GamificationApiTest(unittest.TestCase):
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
            session_factory=self.session_factory,
        )
        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.engine.dispose()
        self._tmp.cleanup()

    def test_gamification_requires_bearer_and_is_scoped_to_current_user(self):
        token_a, user_a = self._register("a@example.com")
        token_b, _ = self._register("b@example.com")
        helper = GamificationServiceTest(methodName="runTest")
        helper.session_factory = self.session_factory
        helper._sequence = 100
        helper._complete(
            user_a,
            when=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
            correct=1,
            incorrect=0,
        )

        missing = self.client.get("/api/gamification")
        response_a = self.client.get(
            "/api/gamification", headers=self._auth(token_a)
        )
        response_b = self.client.get(
            "/api/gamification", headers=self._auth(token_b)
        )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(response_a.status_code, 200)
        self.assertEqual(response_a.json()["progress"]["total_xp"], 100)
        self.assertEqual(response_b.status_code, 200)
        self.assertEqual(response_b.json()["progress"]["total_xp"], 0)

    def test_evaluation_list_and_detail_include_total_triggered_xp(self):
        token, user_id = self._register("xp-evaluation@example.com")
        helper = GamificationServiceTest(methodName="runTest")
        helper.session_factory = self.session_factory
        helper._sequence = 200
        evaluation_id = helper._complete(
            user_id,
            when=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
            correct=1,
            incorrect=0,
            mode="pro",
        )

        history = self.client.get("/api/evaluations", headers=self._auth(token))
        detail = self.client.get(
            f"/api/evaluations/{evaluation_id}", headers=self._auth(token)
        )

        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()[0]["xp_earned"], 180)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["xp_earned"], 180)

    def _register(self, email: str) -> tuple[str, int]:
        response = self.client.post(
            "/api/auth/register",
            json={
                "email": email,
                "password": "password",
                "full_name": "Test User",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["access_token"], response.json()["user"]["id"]

    @staticmethod
    def _auth(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}


if __name__ == "__main__":
    unittest.main()
