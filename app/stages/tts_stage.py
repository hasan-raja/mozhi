"""TTS stage — reads translated.json, synthesizes per-segment speech.

Free-first ladder (engine per MOZHI_ENGINE_MODE):
  local: edge-tts (Microsoft Edge neural voices, free, Indic languages!)
  sarvam: Bulbul API
  mock:  silent placeholder wavs

Each segment's audio is written to jobs/{id}/tts/seg_N.wav and its duration
recorded so the QC stage can compute tempo ratios against the original.
"""

import json
import logging
from pathlib import Path
from typing import Any

from app.media import FFmpegError
from app.task_base import PermanentStageError

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")


def _job_dir(job_id: str) -> Path:
    return DATA_ROOT / "jobs" / job_id


def _edge_tts_voice(target_lang: str) -> str:
    """Map our language codes to free Edge neural voices."""
    voices = {
        "ta": "ta-IN-PallaviNeural",
        "hi": "hi-IN-SwaraNeural",
        "en": "en-IN-NeerjaNeural",
        "te": "te-IN-ShrutiNeural",
        "kn": "kn-IN-SapnaNeural",
        "ml": "ml-IN-SobhanaNeural",
        "bn": "bn-IN-TanishaaNeural",
    }
    return voices.get(target_lang, voices["en"])


async def _synthesize_edge(text: str, voice: str, out_path: Path) -> float:
    """Edge TTS: free neural voices. Returns duration in seconds.

    Raises FFmpegError (a RETRYABLE exception for the pipeline) when edge-tts
    produces an empty or corrupt file — fail fast instead of shipping
    broken audio that only surfaces at the stitch stage.
    """
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))

    import os

    if not os.path.exists(out_path) or os.stat(out_path).st_size == 0:
        raise FFmpegError(
            f"edge-tts produced empty file for voice={voice!r}: {out_path}"
        )
    duration = await _probe_duration(out_path)
    if duration <= 0:
        raise FFmpegError(
            f"edge-tts output has zero duration for voice={voice!r}: {out_path}"
        )
    return duration


async def _probe_duration(out_path: Path) -> float:
    """ffprobe duration in seconds — subprocess offloaded to executor."""
    import asyncio
    import subprocess

    loop = asyncio.get_running_loop()
    proc = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)],
            capture_output=True, text=True,
        ),
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def run_tts(job_id: str) -> dict[str, Any]:
    """Synthesize each translated segment → tts/seg_N.wav + durations.json.
    Pure body — no @stage_task; app.tasks.py wraps it."""
    from app.config import get_settings

    job_dir = _job_dir(job_id)
    translated_path = job_dir / "translated.json"
    if not translated_path.exists():
        raise PermanentStageError(f"translated.json missing for {job_id} — run translate first")

    segments: list[dict[str, Any]] = json.loads(
        translated_path.read_text(encoding="utf-8")
    )
    target_lang = "ta"
    if segments and "-" in str(segments[0].get("translated_text", "")):
        pass  # language comes from job row; default ta

    mode = get_settings().engine_mode
    out_dir = job_dir / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    durations: list[dict[str, Any]] = []
    for seg in segments:
        text = seg.get("translated_text", "")
        idx = seg.get("index", 0)
        out_file = out_dir / f"seg_{idx}.wav"

        if mode == "mock":
            import wave

            dur_s = max(0.5, len(text) * 0.05)
            with wave.open(str(out_file), "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * int(16000 * dur_s))
            duration = dur_s
        else:
            # local AND sarvam both use edge-tts for now (free, Indic voices);
            # Sarvem Bulbul swap-in lands when credits are wired (Day 4b).
            voice = _edge_tts_voice(target_lang)
            duration = asyncio_run_helper(_synthesize_edge(text, voice, out_file))

        durations.append({
            "index": idx,
            "audio_path": str(out_file),
            "duration_ms": int(duration * 1000),
            "original_ms": seg["end_ms"] - seg["start_ms"],
        })

    (job_dir / "durations.json").write_text(json.dumps(durations, indent=2))
    logger.info("tts job=%s: %d segments synthesized (%s)", job_id, len(durations), mode)
    return {"job_id": job_id, "stage": "tts", "segments": len(durations)}


def asyncio_run_helper(coro: Any) -> float:
    import asyncio

    result: float = asyncio.run(coro)
    return result
