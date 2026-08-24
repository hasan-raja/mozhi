"""Async SQLAlchemy engine + session management.

Key decisions:
- create_async_engine with asyncpg driver
- Session is request-scoped: one session per HTTP request via FastAPI dependency,
  never a module-level shared session (async sessions are not concurrency-safe).
"""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings


def build_engine(database_url: str | None = None) -> AsyncEngine:
    """Build an engine from an explicit URL (tests inject their own)."""
    url = database_url or get_settings().database_url
    return create_async_engine(url, echo=False, pool_pre_ping=True)


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency — yields one session per request, always closes."""
    engine = build_engine()
    try:
        async with build_sessionmaker(engine)() as session:
            yield session
    finally:
        await engine.dispose()
