"""Engine abstraction tests — registry selection, mock determinism, protocols."""

from typing import Any
from unittest.mock import patch

import pytest

from app.engine_registry import (
    MockSpeechEngine,
    MockTranscriptionEngine,
    MockTranslationEngine,
    TranslationFallbackEngine,
    get_engines,
)


def test_mock_mode_returns_mock_triple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "mock")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        engines = get_engines()
        assert isinstance(engines["asr"], MockTranscriptionEngine)
        assert isinstance(engines["translate"], MockTranslationEngine)
        assert isinstance(engines["tts"], MockSpeechEngine)
    finally:
        get_settings.cache_clear()


def test_unknown_mode_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "bogus")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="MOZHI_ENGINE_MODE"):
            get_engines()
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_mock_transcription_is_deterministic() -> None:
    e1 = await MockTranscriptionEngine().transcribe("fake.wav", "en")
    e2 = await MockTranscriptionEngine().transcribe("fake.wav", "en")
    assert e1 == e2  # CI/loadtests need identical outputs


@pytest.mark.asyncio
async def test_mock_translation_tags_target_lang() -> None:
    out = await MockTranslationEngine().translate(["hello", "world"], "ta")
    assert out == ["[ta] hello", "[ta] world"]


@pytest.mark.asyncio
async def test_mock_tts_paths_follow_blob_layout() -> None:
    audio = await MockSpeechEngine().synthesize("text", "ta", 3, "job-123")
    assert audio.audio_path == "jobs/job-123/tts/seg_3.wav"


class _FailingTranslationEngine:
    async def translate(self, texts: list[str], target_lang: str) -> list[str]:
        raise RuntimeError("provider unavailable")


class _WorkingTranslationEngine:
    async def translate(self, texts: list[str], target_lang: str) -> list[str]:
        return [f"fallback:{text}" for text in texts]


@pytest.mark.asyncio
async def test_translation_fallback_uses_secondary_provider_after_primary_failure() -> None:
    engine = TranslationFallbackEngine(
        primary=_FailingTranslationEngine(), fallback=_WorkingTranslationEngine()
    )

    assert await engine.translate(["hello"], "ta") == ["fallback:hello"]


def test_local_mode_uses_groq_then_openrouter_when_both_keys_are_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "local")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        unusable_local = type("UnusableLocalTranslator", (), {"usable": False})()
        with patch(
            "app.engines.local.get_local_engines",
            return_value={"translate": unusable_local},
        ):
            engines: dict[str, Any] = get_engines()

        assert isinstance(engines["translate"], TranslationFallbackEngine)
    finally:
        get_settings.cache_clear()
