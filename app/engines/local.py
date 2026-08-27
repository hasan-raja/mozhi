"""Local open-source engines — free, run on your machine (dev default).

ASR:        faster-whisper
Translation: IndicTrans2 via lazy import; falls back to a stub until weights
            are downloaded (documented in README setup).
TTS:        IndicTTS stub — same interface, real synthesis on Day 4.
"""

import logging
from typing import Any

from app.engine_registry import SynthAudio, TranscriptSegment

logger = logging.getLogger(__name__)

_whisper_model: Any = None  # cached process-wide


def _get_whisper() -> Any:
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        # base model: ~150MB download on first use; CPU-friendly
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


class LocalTranscriptionEngine:
    """faster-whisper ASR — CPU-bound, so callers must run it in an executor."""

    async def transcribe(
        self, audio_path: str, source_lang: str
    ) -> list[TranscriptSegment]:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio_path)

    def _transcribe_sync(self, audio_path: str) -> list[TranscriptSegment]:
        model = _get_whisper()
        segments_iter, info = model.transcribe(audio_path, vad_filter=True)
        out: list[TranscriptSegment] = []
        for i, seg in enumerate(segments_iter):
            out.append(
                TranscriptSegment(
                    index=i,
                    start_ms=int(seg.start * 1000),
                    end_ms=int(seg.end * 1000),
                    text=seg.text.strip(),
                    language=info.language,
                )
            )
        logger.info("local ASR: %d segments from %s", len(out), audio_path)
        return out


class LocalTranslationEngine:
    """IndicTrans2-backed translation.

    `usable = False` until the IndicTrans2 weights are set up — the registry
    sees this and falls back to OpenRouter automatically (free-first ladder).
    """

    usable: bool = False

    async def translate(self, texts: list[str], target_lang: str) -> list[str]:
        try:
            return await self._translate_indictrans2(texts, target_lang)
        except ImportError as exc:
            raise RuntimeError(
                "IndicTrans2 not installed. See README → 'Local engine setup' "
                "for: pip install inditrans / ai4bharat-transliteration, or set "
                "MOZHI_ENGINE_MODE=sarvam|mock."
            ) from exc

    async def _translate_indictrans2(self, texts: list[str], target_lang: str) -> list[str]:
        # Placeholder for the IndicTrans2 pipeline — wired in Day 3 step 14.
        # Interface contract is fixed now so the swap-in changes nothing upstream.
        raise ImportError("IndicTrans2 pipeline lands in step 14")


class LocalSpeechEngine:
    """IndicTTS-backed synthesis — interface fixed now, real voice on Day 4."""

    async def synthesize(
        self, text: str, target_lang: str, segment_index: int, job_id: str
    ) -> SynthAudio:
        # Day 4: real synthesis; deterministic placeholder keeps pipelines testable
        return SynthAudio(
            segment_index=segment_index,
            audio_path=f"jobs/{job_id}/tts/seg_{segment_index}.wav",
            duration_ms=len(text) * 50,
        )


def get_local_engines() -> dict[str, Any]:
    return {
        "asr": LocalTranscriptionEngine(),
        "translate": LocalTranslationEngine(),
        "tts": LocalSpeechEngine(),
    }
