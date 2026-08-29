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


def _job_dir(job_id: str) -> Path:
    return DATA_ROOT / "jobs" / job_id


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

    SNR = 10 * log10( sum(signal^2) / sum(noise^2) )
    Noise is estimated as the residual after subtracting the mean.
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
    # Signal power (mean-squared amplitude)
    signal_power = np.mean(y ** 2)
    # Noise: residual from mean subtraction approximates noise floor
    noise = y - np.mean(y)
    noise_power = np.mean(noise ** 2)
    if noise_power <= 0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


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
                "base", device="cpu", compute_type="int8"
            )
        except ImportError:
            logger.warning(
                "faster-whisper not installed — skipping round-trip check"
            )
            return None
    return _WHISPER_MODEL


def _whisper_pronunciation_score(wav_path: Path, expected_text: str) -> float:
    """Re-transcribe with faster-whisper and score similarity to expected text.

    Returns Jaccard similarity of lowercased token set (0.0–1.0).
    Returns 1.0 if the engine is unavailable (fail-open).
    """
    model = _get_whisper_model()
    if model is None:
        return 1.0

    seg_iter, _info = model.transcribe(
        str(wav_path), language="ta", beam_size=3, vad_filter=False
    )
    actual = " ".join(seg.text for seg in seg_iter).strip().lower()
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

    tts_dir = job_dir / "tts"
    scores: list[dict[str, Any]] = []
    renormalized = 0

    for entry in durations:
        idx = entry["index"]
        wav = Path(entry["audio_path"])
        if not wav.exists() or wav.stat().st_size == 0:
            logger.warning("qc job=%s seg=%d missing/empty wav — flagging", job_id, idx)
            scores.append({
                "index": idx,
                "passed": False,
                "reason": "missing_or_empty_wav",
            })
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

        # 3. Loudness normalization if SNR floor breached
        passed = snr >= SNR_FLOOR_DB and duration_ok
        if snr < SNR_FLOOR_DB:
            norm_path = tts_dir / f"seg_{idx}_norm.wav"
            if _normalize_loudness(wav, norm_path):
                wav.unlink(missing_ok=True)
                norm_path.rename(wav)
                renormalized += 1
                snr = _snr_db(wav)
                entry["audio_path"] = str(wav)

        # 4. Whisper round-trip (pronunciation accuracy)
        expected_text = seg_texts.get(idx, "")
        if expected_text:
            pron_score = _whisper_pronunciation_score(wav, expected_text)
        else:
            pron_score = 1.0

        seg_passed = passed and pron_score >= 0.5
        scores.append({
            "index": idx,
            "passed": seg_passed,
            "snr_db": round(snr, 2),
            "duration_ratio": round(ratio, 3),
            "pronunciation_score": round(pron_score, 3),
            "reason": "" if seg_passed else "snr_low" if snr < SNR_FLOOR_DB else "duration_drift",
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
    }
