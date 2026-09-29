from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# PostgreSQL uses bigint throughout the existing schema. The SQLite variant is
# only for isolated tests, where INTEGER is required for autoincrementing PKs.
BIGINT = BigInteger().with_variant(Integer(), "sqlite")
JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")


class AppUser(Base):
    __tablename__ = "app_user"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(150))
    birth_date: Mapped[date | None] = mapped_column(Date)
    height_cm: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    weight_kg: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    training_level: Mapped[str | None] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    sessions: Mapped[list[AuthSession]] = relationship(
        back_populates="user", passive_deletes=True
    )
    progress: Mapped[UserProgress | None] = relationship(
        back_populates="user", uselist=False, passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint(
            "height_cm IS NULL OR height_cm > 0", name="chk_app_user_height"
        ),
        CheckConstraint(
            "training_level IS NULL OR training_level IN "
            "('beginner', 'intermediate', 'advanced')",
            name="chk_app_user_training_level",
        ),
        CheckConstraint(
            "weight_kg IS NULL OR weight_kg > 0", name="chk_app_user_weight"
        ),
        Index("ux_app_user_email", func.lower(email), unique=True),
    )


class AuthSession(Base):
    __tablename__ = "auth_session"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("app_user.id", ondelete="CASCADE"),
        nullable=False,
    )
    token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[AppUser] = relationship(back_populates="sessions")

    __table_args__ = (
        CheckConstraint("expires_at > created_at", name="chk_auth_session_dates"),
        UniqueConstraint("token_hash", name="auth_session_token_hash_key"),
        Index("ix_auth_session_expires_at", "expires_at"),
        Index("ix_auth_session_user_id", "user_id"),
    )


class Evaluation(Base):
    __tablename__ = "evaluation"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("app_user.id", ondelete="CASCADE"),
        nullable=False,
    )
    job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    exercise: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default=text("'squat'")
    )
    mode: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'beginner'")
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'queued'")
    )
    input_filename: Mapped[str | None] = mapped_column(Text)
    correct: Mapped[bool | None] = mapped_column(Boolean)
    total_reps: Mapped[int | None] = mapped_column(Integer)
    correct_reps: Mapped[int | None] = mapped_column(Integer)
    incorrect_reps: Mapped[int | None] = mapped_column(Integer)
    original_video_url: Mapped[str | None] = mapped_column(Text)
    annotated_video_url: Mapped[str | None] = mapped_column(Text)
    video_seconds: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    assistant_seconds: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    error: Mapped[str | None] = mapped_column(Text)
    report_json: Mapped[dict[str, Any]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
        # PostgreSQL implicitly casts this literal to the existing JSONB
        # column. Keeping the portable literal also permits isolated SQLite
        # integration tests; production never creates or alters the table.
        server_default=text("'{}'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "correct_reps IS NULL OR correct_reps >= 0",
            name="chk_evaluation_correct_reps",
        ),
        CheckConstraint("exercise = 'squat'", name="chk_evaluation_exercise"),
        CheckConstraint(
            "incorrect_reps IS NULL OR incorrect_reps >= 0",
            name="chk_evaluation_incorrect_reps",
        ),
        CheckConstraint(
            "mode IN ('beginner', 'pro')", name="chk_evaluation_mode"
        ),
        CheckConstraint(
            "total_reps IS NULL OR correct_reps IS NULL OR "
            "incorrect_reps IS NULL OR total_reps = correct_reps + incorrect_reps",
            name="chk_evaluation_rep_totals",
        ),
        CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed')",
            name="chk_evaluation_status",
        ),
        CheckConstraint(
            "total_reps IS NULL OR total_reps >= 0",
            name="chk_evaluation_total_reps",
        ),
        UniqueConstraint("job_id", name="evaluation_job_id_key"),
        Index("ix_evaluation_report_json", "report_json", postgresql_using="gin"),
        Index("ix_evaluation_status", "status"),
    )


Index(
    "ix_evaluation_user_created",
    Evaluation.user_id,
    Evaluation.created_at.desc(),
)


class UserProgress(Base):
    __tablename__ = "user_progress"

    user_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("app_user.id", ondelete="CASCADE"),
        primary_key=True,
        autoincrement=False,
    )
    total_xp: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    level: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    current_streak: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    longest_streak: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    total_evaluations: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    total_correct_reps: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    total_incorrect_reps: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    user: Mapped[AppUser] = relationship(back_populates="progress")

    __table_args__ = (
        CheckConstraint(
            "total_evaluations >= 0 AND total_correct_reps >= 0 AND "
            "total_incorrect_reps >= 0",
            name="chk_user_progress_counts",
        ),
        CheckConstraint("level >= 1", name="chk_user_progress_level"),
        CheckConstraint(
            "current_streak >= 0 AND longest_streak >= 0",
            name="chk_user_progress_streak",
        ),
        CheckConstraint("total_xp >= 0", name="chk_user_progress_xp"),
    )


class XpTransaction(Base):
    __tablename__ = "xp_transaction"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("app_user.id", ondelete="CASCADE"),
        nullable=False,
    )
    evaluation_id: Mapped[int | None] = mapped_column(
        BIGINT, ForeignKey("evaluation.id", ondelete="SET NULL")
    )
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("amount <> 0", name="chk_xp_transaction_amount"),
        Index("ix_xp_transaction_evaluation", "evaluation_id"),
    )


Index(
    "ix_xp_transaction_user_created",
    XpTransaction.user_id,
    XpTransaction.created_at.desc(),
)


class Achievement(Base):
    __tablename__ = "achievement"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(60), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    xp_reward: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    criteria_json: Mapped[dict[str, Any]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("xp_reward >= 0", name="chk_achievement_xp"),
        UniqueConstraint("code", name="achievement_code_key"),
    )


class UserAchievement(Base):
    __tablename__ = "user_achievement"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("app_user.id", ondelete="CASCADE"),
        nullable=False,
    )
    achievement_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("achievement.id", ondelete="CASCADE"),
        nullable=False,
    )
    evaluation_id: Mapped[int | None] = mapped_column(
        BIGINT, ForeignKey("evaluation.id", ondelete="SET NULL")
    )
    unlocked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "achievement_id", name="ux_user_achievement"
        ),
        Index("ix_user_achievement_user", "user_id"),
    )


class Challenge(Base):
    __tablename__ = "challenge"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(60), nullable=False)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    target_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_value: Mapped[int] = mapped_column(Integer, nullable=False)
    xp_reward: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    criteria_json: Mapped[dict[str, Any]] = mapped_column(
        JSON_DOCUMENT,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "start_at IS NULL OR end_at IS NULL OR end_at > start_at",
            name="chk_challenge_dates",
        ),
        CheckConstraint("target_value > 0", name="chk_challenge_target"),
        CheckConstraint("xp_reward >= 0", name="chk_challenge_xp"),
        UniqueConstraint("code", name="challenge_code_key"),
    )


class UserChallenge(Base):
    __tablename__ = "user_challenge"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("app_user.id", ondelete="CASCADE"),
        nullable=False,
    )
    challenge_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("challenge.id", ondelete="CASCADE"),
        nullable=False,
    )
    progress_value: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'active'")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "progress_value >= 0", name="chk_user_challenge_progress"
        ),
        CheckConstraint(
            "status IN ('active', 'completed', 'claimed', 'expired')",
            name="chk_user_challenge_status",
        ),
        UniqueConstraint("user_id", "challenge_id", name="ux_user_challenge"),
        Index("ix_user_challenge_status", "status"),
        Index("ix_user_challenge_user", "user_id"),
    )
