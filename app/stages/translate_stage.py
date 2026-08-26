"""Translation stage — reads transcript.json, writes translated.json.

Free-first ladder:
  local mode:      IndicTrans2 (AI4Bharat) if weights installed, else clear
                   setup error pointing at docs
  sarvam mode:     Sarvam Translate API (paid credits)
  mock mode:       deterministic "[ta] text" tags

Also records UsageRecord metering rows — billing data written as work happens.
"""

import json
import logging
from pathlib import Path
from typing import Any

from app.engine_registry import get_engines
from app.pipeline_stages import STAGE_NAMES
from app.task_base import PermanentStageError, run_async

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")


def _next_stage(current: str) -> str | None:
    i = STAGE_NAMES.index(current)
    return STAGE_NAMES[i + 1] if i + 1 < len(STAGE_NAMES) else None


def _job_dir(job_id: str) -> Path:
    return DATA_ROOT / "jobs" / job_id


def run_translate(job_id: str) -> dict[str, Any]:
    """Translate each transcript segment into target_lang.

    Reads jobs/{id}/transcript.json → writes jobs/{id}/translated.json with
    the same segment structure plus 'translated_text' per segment.
    NOTE: no @stage_task here — this is the pure body; app.tasks.py wraps it.
    """
    job_dir = _job_dir(job_id)
    transcript_path = job_dir / "transcript.json"
    if not transcript_path.exists():
        raise PermanentStageError(f"transcript missing for {job_id} — run asr first")

    segments: list[dict[str, Any]] = json.loads(transcript_path.read_text(encoding="utf-8"))
    texts = [s.get("text", "") for s in segments]
    if not texts:
        raise PermanentStageError(f"transcript empty for {job_id}")

    # Target language comes from the job row; default ta for now (API wiring Day 4).
    from app.repos import JobRepo  # local import avoids cycle at module load

    async def _load_target() -> str:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.config import get_settings

        engine = create_async_engine(get_settings().database_url)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with maker() as session:
                job = await JobRepo(session).get(__import__("uuid").UUID(job_id))
                return job.target_lang if job else "ta"
        finally:
            await engine.dispose()

    target_lang = run_async(_load_target())

    engines = get_engines()
    t_engine: Any = engines["translate"]
    logger.info("translate job=%s engine=%s n=%d → %s",
                job_id, type(t_engine).__name__, len(texts), target_lang)
    translated: list[str] = run_async(t_engine.translate(texts, target_lang))

    for seg, tr in zip(segments, translated, strict=False):
        seg["translated_text"] = tr

    out_path = job_dir / "translated.json"
    out_path.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    chars = sum(len(t) for t in texts)
    logger.info("translate job=%s done: %d segments, %d chars", job_id, len(segments), chars)

    # Metering: usage rows are written when work happens, never reconstructed.
    run_async(_record_usage(job_id, type(t_engine).__name__.lower(), chars))

    return {
        "job_id": job_id,
        "stage": "translate",
        "segments": len(segments),
        "chars": chars,
        "path": str(out_path),
    }


async def _record_usage(job_id: str, engine_name: str, chars: int) -> None:
    """Write a UsageRecord row — best-effort; metering must not break the stage."""
    try:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.config import get_settings
        from app.models import UsageRecord

        engine = create_async_engine(get_settings().database_url)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with maker() as session:
                session.add(UsageRecord(
                    job_id=__import__("uuid").UUID(job_id),
                    engine=engine_name,
                    operation="translate",
                    quantity=float(chars),
                    unit="chars",
                ))
                await session.commit()
        finally:
            await engine.dispose()
    except Exception:
        logger.exception("usage record failed for job=%s (non-fatal)", job_id)


_ = _next_stage  # reserved for chaining in step 15
