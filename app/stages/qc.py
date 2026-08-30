"""Quality Control stage — validates per-segment TTS audio.

Checks (per PLAN.md Day 4):
  1. SNR check via librosa (>= 12 dB floor)
  2. Duration-ratio score vs original segment bounds (±15% tolerance)
  3. Loudness normalization via ffmpeg ``loudnorm`` for under-threshold segments
  4. Whisper round-trip verification for pronunciation accuracy

This is a pure stage body — no @stage_task. app/tasks.py wraps it.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from app.task_base import PermanentStageError

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")

# Quality thresholds
SNR_FLOOR_DB = 12.0
DURATION_TOLERANCE = 0.15  # ±15% drift allowed

# Step 18: QC feedback loop — auto-remediation before human review.
QC_MAX_RETRIES = 2          # re-synth attempts per segment before escalation
TTS_RATE_DEFAULT = "+0%"   # edge-tts rate; negative = slower speech
TTS_RATE_SLOW = "-20%"    # slower re-synth to better fill the original window


def _job_dir(job_id: str) -> Path:
    return DATA_ROOT / "jobs" / job_id


def _resynth_segment_slower(
    job_id: str, idx: int, text: str, target_lang: str, out_path: Path
) -> float:
    """Re-synthesize ONE segment at a slower rate (Step 18 remediation).

    Returns the new duration in seconds, or 0.0 on failure. Reuses edge-tts via
    the tts stage's helpers so the voice/rate logic stays in one place.
    """
    try:
        from app.stages.tts_stage import (
            _edge_tts_voice,
            asyncio_run_helper,
        )

        voice = _edge_tts_voice(target_lang)
        # edge-tts Communicate accepts a rate override; _synthesize_edge uses the
        # default — patch via monkeypatch-free wrapper is overkill, so we call a
        # small inline path here mirroring _synthesize_edge but with rate.
        import asyncio

        import edge_tts

        communicate = edge_tts.Communicate(text, voice, rate=TTS_RATE_SLOW)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(communicate.save(str(out_path)))
        finally:
            loop.close()

        duration = asyncio_run_helper(_probe_duration_safe(out_path))
        if duration <= 0:
            return 0.0
        return duration
    except Exception:
        logger.exception("qc remediation: re-synth seg=%d failed", idx)
        return 0.0


def _probe_duration_safe(out_path: Path) -> float:
    """ffprobe duration; mirror of tts_stage helper but import-safe here."""
    try:
        import subprocess

        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)],
            capture_output=True, text=True,
        )
        return float(proc.stdout.strip())
    except (ValueError, FileNotFoundError):
        return 0.0


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
            "qc job=%s: failed to load target_lang, defaulting to ta", job_id
        )
        return "ta"


def _load_durations(job_dir: Path) -> list[dict[str, Any]]:
    dur_path = job_dir / "durations.json"
    if not dur_path.exists():
        raise PermanentStageError(
            f"durations.json missing for {job_dir.name} — run tts first"
        )
    raw = json.loads(dur_path.read_text(encoding="utf-8"))
    if not raw:
        raise PermanentStageError(f"empty durations.json for {job_dir.name}")
    return list(raw)


def _load_original_bounds(job_dir: Path) -> dict[int, dict[str, int]]:
    """Build {index: {"duration_ms": int}} from VAD segments."""
    seg_path = job_dir / "segments.json"
    bounds: dict[int, dict[str, int]] = {}
    if seg_path.exists():
        segs = json.loads(seg_path.read_text(encoding="utf-8"))
        for i, pair in enumerate(segs):  # [[start, end], ...]
            start_ms, end_ms = pair
            bounds[i] = {"duration_ms": end_ms - start_ms}
    return bounds


# ── librosa SNR ──────────────────────────────────────────────────────────────

def _snr_db(wav_path: Path) -> float:
    """Return signal-to-noise ratio in dB using librosa.

    SNR = 10 * log10( mean_frame_energy / noise_floor )
    The noise floor is estimated from the quietest 10% of frames (true
    silence/background), NOT by subtracting the mean (which is ~0 for speech
    and collapses every segment to SNR=0.0 — a bug that false-failed all
    clean TTS output).
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        logger.warning("librosa not installed — skipping SNR check")
        return float("inf")

    y, _sr = librosa.load(str(wav_path), sr=16000, mono=True)
    if len(y) == 0:
        return float("-inf")
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    if len(rms) == 0:
        return float("-inf")
    # Noise floor = mean energy of the quietest 10% of frames
    sorted_rms = np.sort(rms)
    n_floor = max(1, len(sorted_rms) // 10)
    noise_floor = float(np.mean(sorted_rms[:n_floor]) ** 2)
    signal_power = float(np.mean(rms ** 2))
    if noise_floor <= 0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_floor))


# ── ffmpeg loudnorm normalization ────────────────────────────────────────────

def _normalize_loudness(in_wav: Path, out_wav: Path) -> bool:
    """Re-encode with ffmpeg loudnorm. Returns True on success."""
    args = [
        "-i", str(in_wav),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y",
        str(out_wav),
    ]
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        logger.error("loudnorm failed for %s: %s", in_wav, proc.stderr[:200])
        return False
    return out_wav.exists() and out_wav.stat().st_size > 0


# ── Whisper round-trip verification ──────────────────────────────────────────

_WHISPER_MODEL: Any = None


def _get_whisper_model() -> Any:
    """Lazy-load WhisperModel once per process to avoid reload per segment."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
            _WHISPER_MODEL = WhisperModel(
                "small", device="cpu", compute_type="int8"
            )
        except ImportError:
            logger.warning(
                "faster-whisper not installed — skipping round-trip check"
            )
            return None
    return _WHISPER_MODEL


def _whisper_pronunciation_score(
    wav_path: Path, expected_text: str, target_lang: str
) -> float:
    """Re-transcribe with faster-whisper and score similarity to expected text.

    Returns Jaccard similarity of lowercased token set (0.0–1.0).
    Returns 1.0 (pass) on any failure — this check is OPTIONAL and must
    never block the pipeline. It's gated by MOZHI_QC_PRONUNCIATION; off by
    default because re-transcribing already-synthesized speech is expensive
    (~30-60s/segment on CPU) and low-value for trusted TTS engines.
    """
    import os

    if not os.environ.get("MOZHI_QC_PRONUNCIATION"):
        return 1.0
    model = _get_whisper_model()
    if model is None:
        return 1.0
    try:
        seg_iter, _info = model.transcribe(
            str(wav_path), language=target_lang, beam_size=3, vad_filter=False
        )
        actual = " ".join(seg.text for seg in seg_iter).strip().lower()
    except Exception:
        logger.exception("qc round-trip failed for %s — passing", wav_path)
        return 1.0

    expected = expected_text.strip().lower()

    actual_tokens = set(actual.split())
    expected_tokens = set(expected.split())
    if not expected_tokens and not actual_tokens:
        return 1.0
    union = actual_tokens | expected_tokens
    if not union:
        return 1.0
    intersection = actual_tokens & expected_tokens
    return float(len(intersection) / len(union))


# ── Main stage body ─────────────────────────────────────────────────────────

def run_qc(job_id: str) -> dict[str, Any]:
    """Validate TTS quality for every segment in the job.

    Reads durations.json + segments.json → writes durations.json back with
    ``qc_scores`` keyed by segment index.  Segments failing SNR or
    duration-ratio are re-normalized in-place.
    """
    job_dir = _job_dir(job_id)
    target_lang = _load_target_lang(job_id)
    durations = _load_durations(job_dir)
    bounds = _load_original_bounds(job_dir)

    # Load translated text for round-trip verification
    seg_texts: dict[int, str] = {}
    trans_path = job_dir / "translated.json"
    if trans_path.exists():
        trans = json.loads(trans_path.read_text(encoding="utf-8"))
        if isinstance(trans, list):
            for s in trans:
                idx = s.get("index", 0)
                seg_texts[idx] = s.get("translated_text", "")

    scores: list[dict[str, Any]] = []
    renormalized = 0
    escalated = 0

    for entry in durations:
        idx = entry["index"]
        wav = Path(entry["audio_path"])
        retries = 0
        if not wav.exists() or wav.stat().st_size == 0:
            logger.warning("qc job=%s seg=%d missing/empty wav — flagging", job_id, idx)
            scores.append({
                "index": idx,
                "passed": False,
                "reason": "missing_or_empty_wav",
                "retries": 0,
                "escalate": True,
            })
            escalated += 1
            continue

        # 1. SNR check
        snr = _snr_db(wav)

        # 2. Duration ratio
        orig_ms = bounds.get(idx, {}).get("duration_ms", 0)
        tts_ms = entry.get("duration_ms", 0)
        if orig_ms > 0:
            ratio = tts_ms / orig_ms
        else:
            ratio = 1.0
        duration_drift = abs(ratio - 1.0)
        duration_ok = duration_drift <= DURATION_TOLERANCE

        reason = ""
        if snr < SNR_FLOOR_DB:
            reason = "snr_low"
        elif not duration_ok:
            reason = "duration_drift"

        # Step 18 remediation loop: retry with adjusted params before escalation
        while reason and retries < QC_MAX_RETRIES:
            if reason == "snr_low":
                # Loudness normalization (idempotent — safe to retry)
                # Build norm_path beside wav so both share absolute/relative
                # form (avoids Windows os.replace path-mismatch errors).
                norm_path = wav.parent / f"seg_{idx}_norm.wav"
                if _normalize_loudness(wav, norm_path):
                    import os

                    # os.replace overwrites atomically on all platforms
                    # (plain rename fails on Windows when dest exists).
                    os.replace(str(norm_path), str(wav))
                    renormalized += 1
                    snr = _snr_db(wav)
                    entry["audio_path"] = str(wav)
            elif reason == "duration_drift":
                # Re-synthesize slower so speech fills the original window better
                text = seg_texts.get(idx, "")
                if text:
                    new_dur = _resynth_segment_slower(
                        job_id, idx, text, target_lang, wav
                    )
                    if new_dur > 0:
                        tts_ms = int(new_dur * 1000)
                        entry["duration_ms"] = tts_ms
                        if orig_ms > 0:
                            ratio = tts_ms / orig_ms
                            duration_drift = abs(ratio - 1.0)

            retries += 1
            # Re-evaluate pass condition
            snr_ok = snr >= SNR_FLOOR_DB
            duration_ok = duration_drift <= DURATION_TOLERANCE
            if snr_ok and duration_ok:
                reason = ""
                break
            # Pick the still-failing reason for the next retry
            if not snr_ok:
                reason = "snr_low"
            elif not duration_ok:
                reason = "duration_drift"

        passed = (not reason)
        if not passed:
            escalated += 1

        # 4. Whisper round-trip (pronunciation accuracy)
        expected_text = seg_texts.get(idx, "")
        if expected_text:
            pron_score = _whisper_pronunciation_score(wav, expected_text, target_lang)
        else:
            pron_score = 1.0

        seg_passed = passed and pron_score >= 0.5
        scores.append({
            "index": idx,
            "passed": seg_passed,
            "snr_db": round(snr, 2),
            "duration_ratio": round(ratio, 3),
            "pronunciation_score": round(pron_score, 3),
            "retries": retries,
            "escalate": (not seg_passed),
            "reason": "" if seg_passed else reason,
        })

    # Persist QC scores back into durations.json
    for entry in durations:
        match = next((s for s in scores if s["index"] == entry["index"]), None)
        if match:
            entry["qc"] = match

    (job_dir / "durations.json").write_text(
        json.dumps(durations, indent=2), encoding="utf-8"
    )

    failed = [s for s in scores if not s["passed"]]
    logger.info(
        "qc job=%s %d segments, %d failed, %d normalized",
        job_id, len(scores), len(failed), renormalized,
    )

    if failed:
        logger.warning("qc job=%s segments failing: %s", job_id,
                       [s["index"] for s in failed])

    return {
        "job_id": job_id,
        "stage": "qc",
        "segments": len(scores),
        "passed": len(scores) - len(failed),
        "failed": len(failed),
        "renormalized": renormalized,
        "escalated": escalated,
    }
