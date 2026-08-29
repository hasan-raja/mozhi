"""Engine abstraction — ports & adapters for ML providers.

Every ML capability (asr, translate, tts) is a Protocol. Engines implement
them; the registry selects by MOZHI_ENGINE_MODE:
  local  → open-source models (faster-whisper, IndicTrans2, IndicTTS) — dev
  sarvam → Sarvam API — demo/production upgrade
  mock   → deterministic fake outputs — CI and load tests, zero keys

Swap tier = one env var. Nothing else changes.
"""

from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import get_settings


@dataclass(frozen=True)
class TranscriptSegment:
    """Output of ASR: one utterance with timing."""

    index: int
    start_ms: int
    end_ms: int
    text: str
    language: str = "en"


@dataclass(frozen=True)
class SynthAudio:
    """Output of TTS: synthesized speech metadata."""

    segment_index: int
    audio_path: str
    duration_ms: int
    tempo_factor: float = 1.0  # stretch applied to fit original timing


class TranscriptionEngine(Protocol):
    async def transcribe(
        self, audio_path: str, source_lang: str
    ) -> list[TranscriptSegment]: ...


class TranslationEngine(Protocol):
    async def translate(self, texts: list[str], target_lang: str) -> list[str]: ...


class TranslationFallbackEngine:
    """Try the primary translator, then the fallback on provider failures."""

    def __init__(self, primary: TranslationEngine, fallback: TranslationEngine) -> None:
        self.primary = primary
        self.fallback = fallback

    async def translate(self, texts: list[str], target_lang: str) -> list[str]:
        try:
            return await self.primary.translate(texts, target_lang)
        except (httpx.HTTPError, RuntimeError):
            return await self.fallback.translate(texts, target_lang)


class SpeechEngine(Protocol):
    async def synthesize(
        self, text: str, target_lang: str, segment_index: int, job_id: str
    ) -> SynthAudio: ...


# ── Mock implementations (CI / load tests — deterministic, no keys) ──────────


class MockTranscriptionEngine:
    async def transcribe(
        self, audio_path: str, source_lang: str
    ) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(0, 0, 2000, "mock transcript line one", source_lang),
            TranscriptSegment(1, 2000, 4000, "mock transcript line two", source_lang),
        ]


class MockTranslationEngine:
    async def translate(self, texts: list[str], target_lang: str) -> list[str]:
        return [f"[{target_lang}] {t}" for t in texts]


class MockSpeechEngine:
    async def synthesize(
        self, text: str, target_lang: str, segment_index: int, job_id: str
    ) -> SynthAudio:
        return SynthAudio(
            segment_index=segment_index,
            audio_path=f"jobs/{job_id}/tts/seg_{segment_index}.wav",
            duration_ms=len(text) * 50,
        )


# ── Registry ─────────────────────────────────────────────────────────────────


def get_engines() -> dict[str, object]:
    """Return the engine triple for the configured mode.

    Local/Sarvam engines import lazily — CI must not need torch/ffmpeg installed.
    """
    settings = get_settings()
    mode = settings.engine_mode
    if mode == "mock":
        return {
            "asr": MockTranscriptionEngine(),
            "translate": MockTranslationEngine(),
            "tts": MockSpeechEngine(),
        }
    if mode == "local":
        from app.engines.local import get_local_engines

        engines: dict[str, object] = get_local_engines()
        # Free-first fallback ladder: IndicTrans2 → Groq → OpenRouter.
        tr = engines.get("translate")
        if tr is not None and getattr(tr, "usable", True) is False:
            from app.engines.groq import GroqTranslationEngine
            from app.engines.openrouter import OpenRouterTranslationEngine

            if settings.groq_api_key and settings.openrouter_api_key:
                engines["translate"] = TranslationFallbackEngine(
                    GroqTranslationEngine(), OpenRouterTranslationEngine()
                )
            elif settings.openrouter_api_key:
                engines["translate"] = OpenRouterTranslationEngine()
            else:
                engines["translate"] = GroqTranslationEngine()
        return engines
    if mode == "groq":
        # Explicit Groq mode — uses Groq for translation, local for ASR/TTS
        from app.engines.groq import GroqTranslationEngine
        from app.engines.local import get_local_engines

        groq_engines: dict[str, object] = get_local_engines()
        groq_engines["translate"] = GroqTranslationEngine()
        return groq_engines
    if mode == "sarvam":
        from app.engines.sarvam_engine import get_sarvam_engines

        return get_sarvam_engines()
    raise ValueError(f"unknown MOZHI_ENGINE_MODE: {mode!r} (use local|groq|sarvam|mock)")
