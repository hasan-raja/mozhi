"""Engine abstraction tests — registry selection, mock determinism, protocols."""

import pytest

from app.engine_registry import (
    MockSpeechEngine,
    MockTranscriptionEngine,
    MockTranslationEngine,
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
