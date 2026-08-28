"""Settings loading and compatibility tests."""

from app.config import Settings


def test_settings_accepts_existing_mozhi_prefixed_runtime_values(monkeypatch):
    monkeypatch.setenv("MOZHI_DATABASE_URL", "postgresql+asyncpg://db.example/mozhi")
    monkeypatch.setenv("MOZHI_REDIS_URL", "redis://redis.example:6379/4")
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "groq")

    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+asyncpg://db.example/mozhi"
    assert settings.redis_url == "redis://redis.example:6379/4"
    assert settings.engine_mode == "groq"


def test_settings_reads_and_strips_unprefixed_api_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", " key-with-windows-carriage-return\r")

    settings = Settings(_env_file=None)

    assert settings.groq_api_key == "key-with-windows-carriage-return"
