"""Application settings — 12-factor: everything configurable via environment."""

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = Field(
        default="mozhi", validation_alias=AliasChoices("MOZHI_APP_NAME", "APP_NAME")
    )
    debug: bool = Field(default=False, validation_alias=AliasChoices("MOZHI_DEBUG", "DEBUG"))

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://mozhi:mozhi@localhost:5433/mozhi",
        validation_alias=AliasChoices("MOZHI_DATABASE_URL", "DATABASE_URL"),
    )

    # Broker
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("MOZHI_REDIS_URL", "REDIS_URL"),
    )

    # ML engines (used from Day 3 onward)
    sarvam_api_key: str = Field(
        default="", validation_alias=AliasChoices("SARVAM_API_KEY", "MOZHI_SARVAM_API_KEY")
    )
    openrouter_api_key: str = Field(
        default="", validation_alias=AliasChoices("OPENROUTER_API_KEY", "MOZHI_OPENROUTER_API_KEY")
    )
    groq_api_key: str = Field(
        default="", validation_alias=AliasChoices("GROQ_API_KEY", "MOZHI_GROQ_API_KEY")
    )
    engine_mode: str = Field(
        default="local",
        validation_alias=AliasChoices("MOZHI_ENGINE_MODE", "ENGINE_MODE"),
    )  # local | sarvam | groq | mock

    @field_validator("groq_api_key", "openrouter_api_key", "sarvam_api_key", mode="before")
    @classmethod
    def strip_keys(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


@lru_cache
def get_settings() -> Settings:
    return Settings()
