"""Stage task implementations — Day 2 ships the orchestration skeleton with
mock bodies; real ML logic lands on Day 3-4 per stage.

Every stage task is IDEMPOTENT by design: keyed by (job_id, stage). If a task
is redelivered after a worker crash but the stage already completed, re-running
must be a safe no-op that skips straight to the next stage.
"""

import logging
from typing import Any

from app.pipeline_stages import STAGE_NAMES
from app.task_base import PermanentStageError, stage_task

logger = logging.getLogger(__name__)


def _next_stage(current: str) -> str | None:
    i = STAGE_NAMES.index(current)
    return STAGE_NAMES[i + 1] if i + 1 < len(STAGE_NAMES) else None


@stage_task("extract")
def run_extract(self: Any, job_id: str) -> dict[str, Any]:
    """Pull audio track from source video (16k mono wav). Mock body for now."""
    logger.info("extract job=%s", job_id)
    # TODO(Day 3): ffmpeg -i in.mp4 -ac 1 -ar 16000 out.wav
    return {"job_id": job_id, "stage": "extract", "artifact": f"jobs/{job_id}/extract/audio.wav"}


@stage_task("vad")
def run_vad(self: Any, job_id: str) -> dict[str, Any]:
    """Voice activity detection -> segment boundaries."""
    logger.info("vad job=%s", job_id)
    # TODO(Day 3): silero VAD segmentation
    return {"job_id": job_id, "stage": "vad", "segments": []}


@stage_task("asr")
def run_asr(self: Any, job_id: str) -> dict[str, Any]:
    """faster-whisper transcription of VAD segments."""
    logger.info("asr job=%s", job_id)
    # TODO(Day 3): faster-whisper batched inference
    return {"job_id": job_id, "stage": "asr"}


@stage_task("translate")
def run_translate(self: Any, job_id: str) -> dict[str, Any]:
    """Sarvam translate with OpenRouter fallback."""
    logger.info("translate job=%s", job_id)
    # TODO(Day 3): engine abstraction + fallback orchestration
    return {"job_id": job_id, "stage": "translate"}


@stage_task("tts")
def run_tts(self: Any, job_id: str) -> dict[str, Any]:
    """Per-segment TTS synthesis with tempo matching."""
    logger.info("tts job=%s", job_id)
    # TODO(Day 4)
    return {"job_id": job_id, "stage": "tts"}


@stage_task("qc")
def run_qc(self: Any, job_id: str) -> dict[str, Any]:
    """SNR / duration-ratio / loudness scoring; fail -> remediation."""
    logger.info("qc job=%s", job_id)
    # TODO(Day 4): QC scoring + remediation ladder
    return {"job_id": job_id, "stage": "qc", "passed": True}


@stage_task("stitch")
def run_stitch(self: Any, job_id: str) -> dict[str, Any]:
    """Concat audio segments, mux onto video, subtitles."""
    logger.info("stitch job=%s", job_id)
    # TODO(Day 4)
    if not job_id:
        raise PermanentStageError("stitch requires job_id")
    return {"job_id": job_id, "stage": "stitch", "output": f"jobs/{job_id}/final.mp4"}
