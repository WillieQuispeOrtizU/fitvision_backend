from __future__ import annotations

import argparse

from sqlalchemy import select

from squat_analysis_api.database import get_session_factory
from squat_analysis_api.gamification_catalog import ensure_gamification_catalog
from squat_analysis_api.gamification_service import (
    all_user_ids,
    rebuild_user_gamification,
)
from squat_analysis_api.models import AppUser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    session_factory = get_session_factory()

    if args.command == "seed":
        with session_factory() as db, db.begin():
            ensure_gamification_catalog(db)
        print("Gamification catalog seeded successfully.")
        return 0

    if args.all_users:
        with session_factory() as db:
            user_ids = all_user_ids(db)
    else:
        user_ids = list(dict.fromkeys(args.user_id or []))
        with session_factory() as db:
            existing = set(
                db.scalars(select(AppUser.id).where(AppUser.id.in_(user_ids))).all()
            )
        missing = [user_id for user_id in user_ids if user_id not in existing]
        if missing:
            parser.error(f"unknown user id(s): {', '.join(map(str, missing))}")

    total_evaluations = 0
    for user_id in user_ids:
        with session_factory() as db:
            transaction = db.begin()
            try:
                rebuilt = rebuild_user_gamification(db, user_id=user_id)
                if args.dry_run:
                    transaction.rollback()
                else:
                    transaction.commit()
            except Exception:
                transaction.rollback()
                raise
        total_evaluations += rebuilt
        action = "Would rebuild" if args.dry_run else "Rebuilt"
        print(f"{action} user_id={user_id}: {rebuilt} completed evaluation(s).")

    verb = "Dry run checked" if args.dry_run else "Backfill completed for"
    print(
        f"{verb} {len(user_ids)} user(s), "
        f"{total_evaluations} completed evaluation(s)."
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed or rebuild FitVision PostgreSQL gamification state."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("seed", help="Idempotently seed the canonical catalogs.")

    backfill = subparsers.add_parser(
        "backfill",
        help="Rebuild derived gamification from completed evaluations.",
    )
    selection = backfill.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        dest="all_users",
        action="store_true",
        help="Rebuild every user, one transaction per user.",
    )
    selection.add_argument(
        "--user-id",
        action="append",
        type=int,
        help="Rebuild one user; repeat the option for multiple users.",
    )
    backfill.add_argument(
        "--dry-run",
        action="store_true",
        help="Run each replay and roll it back instead of persisting it.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
