from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from squat_analysis_api.models import Achievement, Challenge


ACHIEVEMENT_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "code": "first_evaluation",
        "name": "Primera evaluación",
        "description": "Completa tu primera evaluación.",
        "xp_reward": 0,
        "criteria_json": {"metric": "total_evaluations", "target": 1},
        "is_active": True,
    },
    {
        "code": "first_correct",
        "name": "Primera ejecución correcta",
        "description": "Consigue tu primera evaluación correcta.",
        "xp_reward": 0,
        "criteria_json": {"metric": "correct_evaluations", "target": 1},
        "is_active": True,
    },
    {
        "code": "five_evaluations",
        "name": "Técnica en progreso",
        "description": "Completa 5 evaluaciones.",
        "xp_reward": 0,
        "criteria_json": {"metric": "total_evaluations", "target": 5},
        "is_active": True,
    },
    {
        "code": "ten_evaluations",
        "name": "Constancia",
        "description": "Completa 10 evaluaciones.",
        "xp_reward": 0,
        "criteria_json": {"metric": "total_evaluations", "target": 10},
        "is_active": True,
    },
    {
        "code": "perfectionist",
        "name": "Perfeccionista",
        "description": "Consigue 5 evaluaciones correctas.",
        "xp_reward": 0,
        "criteria_json": {"metric": "correct_evaluations", "target": 5},
        "is_active": True,
    },
)


CHALLENGE_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "code": "complete_three",
        "title": "Tres evaluaciones",
        "description": "Completa 3 evaluaciones de sentadilla.",
        "target_type": "total_evaluations",
        "target_value": 3,
        "xp_reward": 120,
        "criteria_json": {
            "metric": "eligible_evaluations",
            "minimum_total_reps": 1,
        },
        "start_at": None,
        "end_at": None,
        "is_active": True,
    },
    {
        "code": "two_correct",
        "title": "Técnica consistente",
        "description": "Consigue 2 evaluaciones correctas.",
        "target_type": "correct_evaluations",
        "target_value": 2,
        "xp_reward": 100,
        "criteria_json": {
            "metric": "eligible_correct_evaluations",
            "minimum_total_reps": 1,
        },
        "start_at": None,
        "end_at": None,
        "is_active": True,
    },
    {
        "code": "try_pro",
        "title": "Prueba el modo Pro",
        "description": "Completa una evaluación en modo Pro.",
        "target_type": "mode_completed",
        "target_value": 1,
        "xp_reward": 80,
        "criteria_json": {
            "mode": "pro",
            "minimum_total_reps": 1,
        },
        "start_at": None,
        "end_at": None,
        "is_active": True,
    },
    {
        "code": "improve_previous",
        "title": "Superación personal",
        "description": "Mejora una ejecución que necesitaba atención.",
        "target_type": "improve_previous",
        "target_value": 1,
        "xp_reward": 150,
        "criteria_json": {
            "previous_correct": False,
            "current_correct": True,
            "minimum_total_reps": 1,
        },
        "start_at": None,
        "end_at": None,
        "is_active": True,
    },
)


def ensure_gamification_catalog(db: Session) -> None:
    """Idempotently insert or update the canonical catalog by stable code."""

    for definition in ACHIEVEMENT_CATALOG:
        _upsert_by_code(db, Achievement, definition)
    for definition in CHALLENGE_CATALOG:
        _upsert_by_code(db, Challenge, definition)
    db.flush()


def _upsert_by_code(
    db: Session,
    model: type[Achievement] | type[Challenge],
    values: dict[str, Any],
) -> None:
    table = model.__table__
    dialect = db.get_bind().dialect.name
    mutable_values = {key: value for key, value in values.items() if key != "code"}

    if dialect == "postgresql":
        statement = postgresql_insert(table).values(**values)
        db.execute(
            statement.on_conflict_do_update(
                index_elements=[table.c.code],
                set_=mutable_values,
            )
        )
        return

    if dialect == "sqlite":
        statement = sqlite_insert(table).values(**values)
        db.execute(
            statement.on_conflict_do_update(
                index_elements=[table.c.code],
                set_=mutable_values,
            )
        )
        return

    existing = db.scalar(select(model).where(model.code == values["code"]))
    if existing is None:
        db.add(model(**values))
        return
    for field, value in mutable_values.items():
        setattr(existing, field, value)
