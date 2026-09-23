"""Async SQLAlchemy engine and session factory (local SQLite or hosted Postgres)."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# Load .env from project root (parent of backend/), not just CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()


def _to_async_database_url(url: str) -> str:
    """Ensure DATABASE_URL uses an async-compatible SQLAlchemy driver."""
    if url.startswith(("postgres://", "postgresql://", "postgresql+asyncpg://")):
        # asyncpg calls libpq's sslmode "ssl".
        url = url.replace("sslmode=", "ssl=")
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("sqlite+aiosqlite://"):
        return url
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    return url


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in the environment or .env file.")

ASYNC_DATABASE_URL = _to_async_database_url(DATABASE_URL)


def _connect_args(url: str) -> dict:
    """Transaction-mode poolers (Supabase Supavisor/PgBouncer on 6543) hand each
    transaction to any server connection, so asyncpg's cached, named prepared
    statements would collide there."""
    from sqlalchemy.engine import make_url

    parsed = make_url(url)
    if parsed.get_backend_name() == "postgresql" and parsed.port == 6543:
        return {
            "statement_cache_size": 0,
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4()}__",
        }
    return {}


engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args=_connect_args(ASYNC_DATABASE_URL),
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI-compatible async session dependency."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _add_missing_columns(sync_conn) -> None:
    """Idempotent upgrades for databases created before a column existed."""
    from sqlalchemy import inspect, text

    from backend.auth import LOCAL_USER_ID

    columns = {c["name"] for c in inspect(sync_conn).get_columns("datasets")}
    if "owner_id" not in columns:
        # Everything uploaded before users existed belongs to the local user.
        sync_conn.execute(
            text(f"ALTER TABLE datasets ADD COLUMN owner_id VARCHAR(64) NOT NULL DEFAULT '{LOCAL_USER_ID}'")
        )
        sync_conn.execute(text("CREATE INDEX IF NOT EXISTS ix_datasets_owner_id ON datasets (owner_id)"))


def _lock_down_postgres(sync_conn) -> None:
    """Row-level security with no policies on every app table.

    Supabase serves the public schema over its REST API to anyone holding the
    anon key (which ships in the frontend). With RLS on and no policies those
    roles see nothing; this backend connects as the table owner and is unaffected.
    """
    from sqlalchemy import text

    for table in Base.metadata.sorted_tables:
        sync_conn.execute(text(f'ALTER TABLE "{table.name}" ENABLE ROW LEVEL SECURITY'))


async def init_db() -> None:
    """Create missing tables and columns, then index pre-table local model files."""
    from backend import models  # noqa: F401
    from backend.mcp_server.model_registry import import_legacy_models
    from backend.storage import LocalStorage, get_storage

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)
        if conn.dialect.name == "postgresql":
            await conn.run_sync(_lock_down_postgres)

    storage = get_storage()
    if isinstance(storage, LocalStorage):
        async with AsyncSessionLocal() as session:
            await import_legacy_models(session, storage)
