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


def _load_target_lang(job_id: str) -> str:
    """Read the job's target language from the DB (falls back to 'ta')."""
    try:
        import asyncio
        import uuid

        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.config import get_settings
        from app.repos import JobRepo

        async def _load() -> str:
            engine = create_async_engine(get_settings().database_url)
            maker = async_sessionmaker(engine, expire_on_commit=False)
            try:
                async with maker() as session:
                    job = await JobRepo(session).get(uuid.UUID(job_id))
                    return job.target_lang if job and job.target_lang else "ta"
            finally:
                await engine.dispose()

        return str(asyncio.run(_load()))
    except Exception:
        logger.exception(
            "tts job=%s: failed to load target_lang, defaulting to ta", job_id
        )
        return "ta"


def _edge_tts_voice(target_lang: str, gender: str | None = None) -> str:
    """Map our language codes to free Edge neural voices.

    Args:
        target_lang: Language code (ta, hi, en, te, kn, ml, bn)
        gender: "male", "female", or None (defaults to female for backwards compat)

    Returns:
        Edge TTS voice identifier.
    """
    # Female voices (default, original mapping)
    female_voices = {
        "ta": "ta-IN-PallaviNeural",
        "hi": "hi-IN-SwaraNeural",
        "en": "en-IN-NeerjaNeural",
        "te": "te-IN-ShrutiNeural",
        "kn": "kn-IN-SapnaNeural",
        "ml": "ml-IN-SobhanaNeural",
        "bn": "bn-IN-TanishaaNeural",
    }

    # Male voices for gender-aware dubbing
    male_voices = {
        "ta": "ta-IN-ValluvarNeural",
        "hi": "hi-IN-MadhurNeural",
        "en": "en-IN-PrabhatNeural",
        "te": "te-IN-MohanNeural",
        "kn": "kn-IN-GaganNeural",
        "ml": "ml-IN-MidhunNeural",
        "bn": "bn-IN-BashkarNeural",
    }

    if gender == "male" and target_lang in male_voices:
        return male_voices[target_lang]

    # Default to female for backwards compatibility
    return female_voices.get(target_lang, female_voices["en"])


async def _synthesize_edge(text: str, voice: str, out_path: Path, rate: str = "+0%") -> float:
    """Edge TTS: free neural voices. Returns duration in seconds.

    Raises FFmpegError (a RETRYABLE exception for the pipeline) when edge-tts
    produces an empty or corrupt file — fail fast instead of shipping
    broken audio that only surfaces at the stitch stage.

    The `rate` parameter adjusts speech speed (e.g., "-20%" = slower, "+10%" = faster).
    """
    import asyncio
    import os

    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(str(out_path))

    loop = asyncio.get_running_loop()

    def _validate() -> None:
        if not os.path.exists(out_path) or os.stat(out_path).st_size == 0:
            raise FFmpegError(
                f"edge-tts produced empty file for voice={voice!r}: {out_path}"
            )

    await loop.run_in_executor(None, _validate)

    duration = await _probe_duration(out_path)
    if duration <= 0:
        raise FFmpegError(
            f"edge-tts output has zero duration for voice={voice!r}: {out_path}"
        )
    return duration


async def _ffmpeg_atempo(in_path: Path, atempo: float) -> None:
    """Apply ffmpeg atempo filter to time-stretch audio. atempo range: 0.5–2.0."""
    import asyncio
    import os
    import subprocess
    import tempfile

    loop = asyncio.get_running_loop()

    # ffmpeg atempo only supports 0.5-2.0; chain multiple if outside range
    filters = []
    remaining = atempo
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.3f}")
    filter_chain = ",".join(filters)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["ffmpeg", "-y", "-i", str(in_path), "-af", filter_chain, str(tmp_path)],
                capture_output=True, text=True,
            ),
        )
        if proc.returncode != 0:
            raise FFmpegError(f"ffmpeg atempo failed: {proc.stderr}")

        # Atomic replace
        await loop.run_in_executor(None, lambda: os.replace(tmp_path, in_path))
    finally:
            if os.path.exists(tmp_path):  # noqa: ASYNC240
                try:
                    os.unlink(tmp_path)  # noqa: ASYNC240
                except OSError:
                    pass


async def _synthesize_with_duration_match(
    text: str, voice: str, out_path: Path, target_duration_ms: int
) -> tuple[float, str]:
    """Synthesize with edge-tts, iteratively adjusting rate to hit target duration.

    Tries default rate first, then adjusts toward target duration within ±10% tolerance.
    Falls back to ffmpeg atempo time-stretch when rate hits ±50% cap.
    Returns (achieved_duration_seconds, rate_used_string).
    """
    TOLERANCE = 0.10  # 10% tolerance on duration
    MAX_ATTEMPTS = 5
    MAX_RATE = 50  # edge-tts limit

    if target_duration_ms <= 0:
        # No target to match; just synthesize at default rate
        duration = await _synthesize_edge(text, voice, out_path, rate="+0%")
        return duration, "+0%"

    # Start with default rate
    rate = "+0%"
    for _attempt in range(MAX_ATTEMPTS):
        try:
            duration = await _synthesize_edge(text, voice, out_path, rate=rate)
            duration_ms = int(duration * 1000)
            ratio = duration_ms / target_duration_ms
            drift = abs(ratio - 1.0)

            if drift <= TOLERANCE:
                return duration, rate

            # Adjust rate for next attempt: if too long, speed up; if too short, slow down
            if ratio > 1.0:
                # Too long → need faster speech (positive rate)
                adjustment = int((ratio - 1.0) * 100)
                new_rate = min(adjustment + 10, MAX_RATE)
                rate = f"+{new_rate}%"
            else:
                # Too short → need slower speech (negative rate)
                adjustment = int((1.0 - ratio) * 100)
                new_rate = min(adjustment + 10, MAX_RATE)
                rate = f"-{new_rate}%"

        except Exception:
            # On any error, fall back to default rate result
            break

    # Fallback: ffmpeg atempo time-stretch to exact target
    # Get current duration at default rate
    try:
        duration = await _synthesize_edge(text, voice, out_path, rate="+0%")
    except Exception:
        duration = target_duration_ms / 1000.0

    duration_ms = int(duration * 1000)
    atempo = duration_ms / target_duration_ms  # >1 = faster, <1 = slower
    await _ffmpeg_atempo(out_path, atempo)

    return target_duration_ms / 1000.0, f"atempo={atempo:.3f}"


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
    Pure body — no @stage_task; app.tasks.py wraps it.
    """
    from app.config import get_settings

    job_dir = _job_dir(job_id)
    translated_path = job_dir / "translated.json"
    if not translated_path.exists():
        raise PermanentStageError(f"translated.json missing for {job_id} — run translate first")

    segments: list[dict[str, Any]] = json.loads(
        translated_path.read_text(encoding="utf-8")
    )
    # Target language comes from the JOB ROW, not a hardcoded default — a Hindi
    # job must synthesize Hindi audio, not Tamil. Falls back to 'ta' only if
    # the DB row is unreachable (local/manual runs).
    target_lang = _load_target_lang(job_id)

    # Load diarization for gender-aware voice selection
    diarization_path = job_dir / "diarization.json"
    diarization_segments: list[dict[str, Any]] = []
    if diarization_path.exists():
        try:
            diarization_segments = json.loads(diarization_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("tts job=%s: failed to load diarization.json", job_id)

    # Build a lookup: (start_ms, end_ms) -> gender
    gender_lookup: dict[tuple[int, int], str] = {}
    for dseg in diarization_segments:
        key = (dseg.get("start_ms", 0), dseg.get("end_ms", 0))
        gender_lookup[key] = dseg.get("gender", "unknown")

    mode = get_settings().engine_mode
    out_dir = job_dir / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    durations: list[dict[str, Any]] = []
    for seg in segments:
        text = seg.get("translated_text", "")
        idx = seg.get("index", 0)
        out_file = out_dir / f"seg_{idx}.wav"

        # Get gender for this segment from diarization (match by timing)
        start_ms = seg.get("start_ms", 0)
        end_ms = seg.get("end_ms", 0)
        gender = gender_lookup.get((start_ms, end_ms), "unknown")

        if mode == "mock":
            import wave

            dur_s = max(0.5, len(text) * 0.05)
            with wave.open(str(out_file), "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * int(16000 * dur_s))
            duration = dur_s
            tempo_factor = 1.0
            rate_used = "+0%"
        else:
            # local AND sarvam both use edge-tts for now (free, Indic voices);
            # Sarvem Bulbul swap-in lands when credits are wired (Day 4b).
            voice = _edge_tts_voice(target_lang, gender)
            original_ms = seg["end_ms"] - seg["start_ms"]
            duration, rate_used = asyncio_run_helper(
                _synthesize_with_duration_match(text, voice, out_file, original_ms)
            )
            tempo_factor = duration * 1000 / original_ms if original_ms > 0 else 1.0

        durations.append({
            "index": idx,
            "audio_path": str(out_file),
            "duration_ms": int(duration * 1000),
            "original_ms": seg["end_ms"] - seg["start_ms"],
            "tempo_factor": round(tempo_factor, 3),
            "rate_used": rate_used,
            "gender": gender,
        })

    (job_dir / "durations.json").write_text(json.dumps(durations, indent=2))
    logger.info("tts job=%s: %d segments synthesized (%s)", job_id, len(durations), mode)
    return {"job_id": job_id, "stage": "tts", "segments": len(durations)}


def asyncio_run_helper(coro: Any) -> tuple[float, str]:
    import asyncio

    result = asyncio.run(coro)
    return tuple(result)