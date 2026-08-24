"""Tests for the DB layer wiring.

These verify construction/config logic without a live Postgres — integration
tests against real Postgres arrive with Alembic (step 4).
"""

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import get_settings
from app.db import build_engine, build_sessionmaker


def test_build_engine_uses_configured_url() -> None:
    settings = get_settings()
    engine = build_engine()
    assert engine.url.render_as_string(hide_password=False) == settings.database_url


def test_build_engine_accepts_override() -> None:
    engine = build_engine("postgresql+asyncpg://x:x@localhost:1/x")
    assert "localhost:1" in engine.url.render_as_string(hide_password=True)


def test_sessionmaker_built_from_engine() -> None:
    engine = build_engine("postgresql+asyncpg://x:x@localhost:1/x")
    maker = build_sessionmaker(engine)
    assert isinstance(maker, async_sessionmaker)


def test_settings_cached() -> None:
    assert get_settings() is get_settings()


def test_database_url_from_env(monkeypatch) -> None:
    """Env var MOZHI_DATABASE_URL overrides the default (12-factor config)."""
    monkeypatch.setenv("MOZHI_DATABASE_URL", "postgresql+asyncpg://e:e@h:2/e")
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert s.database_url == "postgresql+asyncpg://e:e@h:2/e"
    finally:
        get_settings.cache_clear()
