from __future__ import annotations

import os
from collections.abc import Generator
from functools import lru_cache

from fastapi import HTTPException, status
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(RuntimeError):
    """Raised when PostgreSQL configuration is missing or unsupported."""


def _database_url() -> URL:
    raw_url = os.getenv("DATABASE_URL", "").strip()
    if not raw_url:
        raise DatabaseConfigurationError("DATABASE_URL is not configured")

    # Some cloud providers still expose the legacy postgres:// scheme. Force
    # SQLAlchemy to use psycopg 3 for both common PostgreSQL URL forms.
    if raw_url.startswith("postgres://"):
        raw_url = f"postgresql://{raw_url.removeprefix('postgres://')}"

    try:
        url = make_url(raw_url)
    except Exception as exc:
        raise DatabaseConfigurationError("DATABASE_URL is invalid") from exc

    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    if url.drivername != "postgresql+psycopg":
        raise DatabaseConfigurationError(
            "DATABASE_URL must use PostgreSQL with the psycopg driver"
        )
    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(_database_url(), pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    """Provide one SQLAlchemy session per request.

    Transactions are committed explicitly by write endpoints. Any uncaught
    error rolls the request transaction back, and the session is always closed.
    """

    try:
        factory = get_session_factory()
    except (DatabaseConfigurationError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database is unavailable",
        ) from exc

    db = factory()
    try:
        yield db
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database is unavailable",
        ) from exc
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_database_connection() -> str:
    """Return a credential-free PostgreSQL health status."""

    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except DatabaseConfigurationError:
        return "not_configured"
    except SQLAlchemyError:
        return "unavailable"
    return "ok"
