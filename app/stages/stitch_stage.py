"""Stitch stage — concat TTS segments, mux onto source video.

Reads durations.json to know each segment's real length, then:
  1. Concatenates jobs/{id}/tts/seg_*.wav in order (ffmpeg concat filter)
  2. Replaces the source video's audio track with the stitched wav
  3. Writes final.mp4 to jobs/{id}/final.mp4

Idempotent: if final.mp4 already exists, skip (safe for redelivery).
"""

import json
import logging
from pathlib import Path
from typing import Any

from app.media import ffmpeg_concat_wavs, ffmpeg_mux_audio_on_video
from app.task_base import PermanentStageError

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")


def _job_dir(job_id: str) -> Path:
    return DATA_ROOT / "jobs" / job_id


def run_stitch(job_id: str) -> dict[str, Any]:
    """Concatenate TTS wavs + mux onto source video → final.mp4.

    Pure body — no @stage_task; app/tasks.py wraps it.
    """
    job_dir = _job_dir(job_id)

    final_path = job_dir / "final.mp4"
    if final_path.exists() and final_path.stat().st_size > 0:
        logger.info("stitch job=%s final.mp4 already exists — skipping", job_id)
        return {"job_id": job_id, "stage": "stitch", "output": str(final_path)}

    # Source video for the video track
    from app.tasks import _find_source
    source_video = str(_find_source(job_id))

    # TTS segments
    tts_dir = job_dir / "tts"
    seg_files = sorted(tts_dir.glob("seg_*.wav"))
    if not seg_files:
        raise PermanentStageError(f"no TTS wav files for job {job_id}")

    # Read durations to validate
    dur_path = job_dir / "durations.json"
    durations = []
    if dur_path.exists():
        durations = json.loads(dur_path.read_text(encoding="utf-8"))

    if not durations:
        raise PermanentStageError(
            f"durations.json missing for {job_id} — run tts first"
        )

    # Step 1: concatenate all wav segments
    concatenated = tts_dir / "concatenated.wav"
    logger.info("stitch job=%s concatenating %d wav segments", job_id, len(seg_files))
    ffmpeg_concat_wavs(seg_files, concatenated)
    if not concatenated.exists():
        raise PermanentStageError("ffmpeg concat failed: no output wav")

    # Step 2: mux onto source video
    logger.info("stitch job=%s muxing onto video %s", job_id, source_video)
    ffmpeg_mux_audio_on_video(source_video, str(concatenated), str(final_path))

    if not final_path.exists() or final_path.stat().st_size == 0:
        raise PermanentStageError("mux produced no final.mp4")

    size_mb = final_path.stat().st_size / (1024 * 1024)
    logger.info("stitch job=%s final.mp4 ready (%.2f MB)", job_id, size_mb)

    return {
        "job_id": job_id,
        "stage": "stitch",
        "output": str(final_path),
        "segments": len(seg_files),
        "size_mb": round(size_mb, 2),
    }
