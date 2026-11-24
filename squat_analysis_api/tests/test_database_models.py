from __future__ import annotations

import unittest

from squat_analysis_api.models import Base


class DatabaseModelsTest(unittest.TestCase):
    def test_mapped_tables_and_columns_match_schema_dump(self):
        expected = {
            "achievement": {
                "id",
                "code",
                "name",
                "description",
                "xp_reward",
                "criteria_json",
                "is_active",
                "created_at",
            },
            "app_user": {
                "id",
                "email",
                "password_hash",
                "full_name",
                "birth_date",
                "height_cm",
                "weight_kg",
                "training_level",
                "is_active",
                "created_at",
                "updated_at",
            },
            "auth_session": {
                "id",
                "user_id",
                "token_hash",
                "created_at",
                "expires_at",
                "last_used_at",
                "revoked_at",
            },
            "challenge": {
                "id",
                "code",
                "title",
                "description",
                "target_type",
                "target_value",
                "xp_reward",
                "criteria_json",
                "start_at",
                "end_at",
                "is_active",
                "created_at",
            },
            "evaluation": {
                "id",
                "user_id",
                "job_id",
                "exercise",
                "mode",
                "status",
                "input_filename",
                "correct",
                "total_reps",
                "correct_reps",
                "incorrect_reps",
                "original_video_url",
                "annotated_video_url",
                "video_seconds",
                "assistant_seconds",
                "error",
                "report_json",
                "created_at",
                "updated_at",
                "completed_at",
            },
            "password_reset_request": {
                "id",
                "user_id",
                "request_token_hash",
                "email_hash",
                "requester_ip_hash",
                "code_hash",
                "failed_attempts",
                "created_at",
                "expires_at",
                "verified_at",
                "reset_token_hash",
                "reset_token_expires_at",
                "used_at",
                "invalidated_at",
                "delivery_failed_at",
            },
            "user_achievement": {
                "id",
                "user_id",
                "achievement_id",
                "evaluation_id",
                "unlocked_at",
            },
            "user_challenge": {
                "id",
                "user_id",
                "challenge_id",
                "progress_value",
                "status",
                "started_at",
                "completed_at",
                "claimed_at",
                "updated_at",
            },
            "user_progress": {
                "user_id",
                "total_xp",
                "level",
                "current_streak",
                "longest_streak",
                "total_evaluations",
                "total_correct_reps",
                "total_incorrect_reps",
                "updated_at",
            },
            "xp_transaction": {
                "id",
                "user_id",
                "evaluation_id",
                "amount",
                "reason",
                "created_at",
            },
        }

        self.assertEqual(set(Base.metadata.tables), set(expected))
        for table_name, columns in expected.items():
            with self.subTest(table=table_name):
                self.assertEqual(
                    set(Base.metadata.tables[table_name].columns.keys()),
                    columns,
                )


if __name__ == "__main__":
    unittest.main()
