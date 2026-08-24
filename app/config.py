"""Application settings — 12-factor: everything configurable via environment."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MOZHI_", extra="ignore")

    # App
    app_name: str = "mozhi"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://mozhi:mozhi@localhost:5432/mozhi"

    # Broker
    redis_url: str = "redis://localhost:6379/0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
