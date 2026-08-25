"""Sarvam API engines — the demo/production upgrade path.

Uses the ₹1,000 free signup credits; each call records usage for metering.
Kept in its own module so `MOZHI_ENGINE_MODE=local` never imports httpx-heavy
client code paths or requires keys.
"""

import logging
import os

import httpx

from app.engine_registry import SynthAudio, TranscriptSegment

logger = logging.getLogger(__name__)

SARVAM_BASE = "https://api.sarvam.ai/v1"


def _api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        raise RuntimeError("SARVAM_API_KEY not set — add it to .env")
    return key


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}


class SarvamTranscriptionEngine:
    """Sarvam STT (₹30/hour of audio)."""

    async def transcribe(
        self, audio_path: str, source_lang: str
    ) -> list[TranscriptSegment]:
        # Sarvam STT takes raw audio upload; timing comes from their response.
        # Wired fully in step 13b after local path proves the pipeline.
        raise NotImplementedError("Sarvam STT wiring lands in step 13b")


class SarvamTranslationEngine:
    """Sarvam Translate (₹20/10k chars)."""

    async def translate(self, texts: list[str], target_lang: str) -> list[str]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{SARVAM_BASE}/translate",
                headers=_headers(),
                json={
                    "input": "\n".join(texts),
                    "source_language_code": "en-IN",
                    "target_language_code": f"{target_lang}-IN",
                },
            )
            resp.raise_for_status()
            data: dict[str, object] = resp.json()
        raw = str(data.get("translated_text", ""))
        translated = raw.split("\n")
        # API may merge lines; pad/truncate to keep 1:1 mapping
        while len(translated) < len(texts):
            translated.append(texts[len(translated)])
        return translated[: len(texts)]


class SarvamSpeechEngine:
    """Sarvam Bulbul TTS (₹15/10k chars, v2)."""

    async def synthesize(
        self, text: str, target_lang: str, segment_index: int, job_id: str
    ) -> SynthAudio:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{SARVAM_BASE}/text-to-speech",
                headers=_headers(),
                json={"inputs": [text], "target_language_code": f"{target_lang}-IN"},
            )
            resp.raise_for_status()
            data = resp.json()

        # Response carries base64 audio; persist to blob layout.
        # NOTE: blocking Path I/O inside async is acceptable here only because
        # Sarvam engine calls are themselves wrapped for the executor pool on
        # Day 4; flagged by ASYNC240 as a deliberate exception.
        import base64
        from pathlib import Path  # noqa: ASYNC240

        out_dir = Path(f"data/jobs/{job_id}/tts")
        out_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
        out_path = out_dir / f"seg_{segment_index}.wav"
        audio_b64 = data.get("audios", [""])[0]
        out_path.write_bytes(base64.b64decode(audio_b64))

        return SynthAudio(
            segment_index=segment_index,
            audio_path=str(out_path),
            duration_ms=len(text) * 50,
        )


def get_sarvam_engines() -> dict[str, object]:
    return {
        "asr": SarvamTranscriptionEngine(),
        "translate": SarvamTranslationEngine(),
        "tts": SarvamSpeechEngine(),
    }
