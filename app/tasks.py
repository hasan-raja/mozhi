"""Stage task implementations — Day 3: extract/vad/asr are now REAL.

Every stage task is IDEMPOTENT by design: keyed by (job_id, stage). If a task
is redelivered after a worker crash but the stage already completed, re-running
must be a safe no-op that skips straight to the next stage.

Data flow per job:
  jobs/{job_id}/source.*   (uploaded input)
  jobs/{job_id}/extract/audio.wav        <- extract stage
  jobs/{job_id}/segments.json            <- vad + asr stages
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.engine_registry import get_engines
from app.media import detect_segments, extract_audio
from app.pipeline_stages import STAGE_NAMES
from app.task_base import PermanentStageError, run_async, stage_task

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")


def _next_stage(current: str) -> str | None:
    i = STAGE_NAMES.index(current)
    return STAGE_NAMES[i + 1] if i + 1 < len(STAGE_NAMES) else None


def _job_dir(job_id: str) -> Path:
    return DATA_ROOT / "jobs" / job_id


def _find_source(job_id: str) -> Path:
    """Locate the uploaded source file for this job."""
    source_dir = _job_dir(job_id) / "source"
    candidates = sorted(source_dir.glob("*")) if source_dir.exists() else []
    if not candidates:
        raise PermanentStageError(
            f"no source media for job {job_id} — upload first"
        )
    return candidates[0]


@stage_task("extract")
def run_extract(self: Any, job_id: str) -> dict[str, Any]:
    """Pull audio track from source video/audio → 16k mono wav."""
    src = _find_source(job_id)
    out_wav = _job_dir(job_id) / "extract" / "audio.wav"

    logger.info("extract job=%s src=%s", job_id, src)
    artifact = asyncio.run(extract_audio(str(src), str(out_wav)))
    return {"job_id": job_id, "stage": "extract", "artifact": artifact}


@stage_task("vad")
def run_vad(self: Any, job_id: str) -> dict[str, Any]:
    """Voice activity detection → speech segment boundaries (ms)."""
    wav = _job_dir(job_id) / "extract" / "audio.wav"
    if not wav.exists():
        raise PermanentStageError(f"extracted audio missing for {job_id}")

    logger.info("vad job=%s", job_id)
    segments = asyncio.run(detect_segments(str(wav)))

    import json

    seg_file = _job_dir(job_id) / "segments.json"
    seg_file.write_text(json.dumps(segments))
    return {"job_id": job_id, "stage": "vad", "count": len(segments)}


def _load_source_lang(job_id: str) -> str:
    """Read the job's declared source language from the DB.

    Defaults to 'en' when the row is unavailable (e.g. local/manual runs)
    so transcription never blows up, but production jobs use the language
    the caller actually configured — forcing 'en' would transcribe every
    video as English and corrupt the transcript + downstream translation.
    """
    async def _load() -> str:
        import uuid

        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.config import get_settings
        from app.repos import JobRepo

        engine = create_async_engine(get_settings().database_url)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with maker() as session:
                job = await JobRepo(session).get(uuid.UUID(job_id))
                return job.source_lang if job and job.source_lang else "en"
        finally:
            await engine.dispose()

    try:
        return str(run_async(_load()))
    except Exception:
        logger.exception("asr job=%s: failed to load source_lang, defaulting to en", job_id)
        return "en"


def _save_source_lang(job_id: str, source_lang: str) -> None:
    """Persist the auto-detected source language back to the Job row.

    Whisper detects the real language from the audio; we store it so the
    API status and any downstream consumer see the truth instead of the
    generic 'en' default the job was created with.
    """
    async def _save() -> None:
        import uuid

        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.config import get_settings
        from app.repos import JobRepo

        engine = create_async_engine(get_settings().database_url)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with maker() as session:
                job = await JobRepo(session).get(uuid.UUID(job_id))
                if job is not None:
                    job.source_lang = source_lang
                    await session.commit()
        finally:
            await engine.dispose()

    try:
        run_async(_save())
    except Exception:
        logger.exception("asr job=%s: failed to save detected source_lang=%s", job_id, source_lang)


@stage_task("asr")
def run_asr(self: Any, job_id: str) -> dict[str, Any]:
    """faster-whisper transcription of the extracted audio.

    Transcribes the full wav once (whisper's own VAD filter handles splits),
    then aligns transcripts to the VAD boundaries stored by the vad stage.
    The source language comes from the job row (NOT hardcoded) so a Tamil/Hindi/
    etc. video is transcribed in its own language.
    """
    wav = _job_dir(job_id) / "extract" / "audio.wav"
    seg_file = _job_dir(job_id) / "segments.json"
    if not wav.exists() or not seg_file.exists():
        raise PermanentStageError(f"upstream artifacts missing for {job_id}")

    source_lang = _load_source_lang(job_id)
    # The job's stored source_lang may be the generic 'en' default when the
    # caller didn't know the language. Pass None to faster-whisper so it
    # AUTO-DETECTS the real language from the audio instead of forcing English
    # (which would transcribe a Tamil/Hindi video as English and corrupt the
    # downstream translation). If a real, non-default language was provided we
    # still respect it.
    detect_lang: str | None = None if source_lang == "en" else source_lang

    engines = get_engines()
    engine: Any = engines["asr"]
    transcribe = engine.transcribe

    logger.info(
        "asr job=%s engine=%s source_lang=%s detect=%s",
        job_id, type(engine).__name__, source_lang, detect_lang,
    )
    segments = run_async(transcribe(str(wav), detect_lang))

    # Persist the language whisper actually detected back to the Job row so the
    # rest of the pipeline (and the API status) reflects the real source lang.
    detected = segments[0].language if segments else source_lang
    if detected and detected != source_lang:
        _save_source_lang(job_id, detected)

    import json

    payload = [
        {
            "index": s.index,
            "start_ms": s.start_ms,
            "end_ms": s.end_ms,
            "text": s.text,
            "language": s.language,
        }
        for s in segments
    ]
    out = _job_dir(job_id) / "transcript.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("asr job=%s produced %d segments", job_id, len(payload))
    return {"job_id": job_id, "stage": "asr", "segments": len(payload), "path": str(out)}


@stage_task("translate")
def run_translate(self: Any, job_id: str) -> dict[str, Any]:
    """Translate transcript segments → translated.json (engine per mode)."""
    from app.stages.translate_stage import run_translate as _body

    result: dict[str, Any] = _body(job_id)
    return result


@stage_task("tts")
def run_tts(self: Any, job_id: str) -> dict[str, Any]:
    """Per-segment TTS synthesis with tempo tracking."""
    from app.stages.tts_stage import run_tts as _body

    result: dict[str, Any] = _body(job_id)
    return result


@stage_task("qc")
def run_qc(self: Any, job_id: str) -> dict[str, Any]:
    """SNR / duration-ratio / loudness scoring + remediation."""
    logger.info("qc job=%s", job_id)
    from app.stages.qc import run_qc as _body
    return _body(job_id)


@stage_task("stitch")
def run_stitch(self: Any, job_id: str) -> dict[str, Any]:
    """Concat audio segments, mux onto video, subtitles."""
    logger.info("stitch job=%s", job_id)
    from app.stages.stitch_stage import run_stitch as _body
    result: dict[str, Any] = _body(job_id)
    return result
