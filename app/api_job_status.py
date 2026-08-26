"""Job status endpoint — read-only view of job state + stage artifacts."""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.tasks import DATA_ROOT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

STAGE_ARTIFACTS = {
    "extract": "extract/audio.wav",
    "vad": "segments.json",
    "asr": "transcript.json",
    "translate": "translated.json",
    "tts": "durations.json",
}


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    import uuid as uuid_mod

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.config import get_settings
    from app.repos import JobRepo

    try:
        job_uuid = uuid_mod.UUID(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid job id format") from exc

    engine = create_async_engine(get_settings().database_url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            job = await JobRepo(session).get(job_uuid)
    finally:
        await engine.dispose()

    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")

    job_dir = DATA_ROOT / "jobs" / job_id
    artifacts: dict[str, bool] = {
        name: (job_dir / rel).exists() for name, rel in STAGE_ARTIFACTS.items()
    }

    transcript_preview: list[dict[str, Any]] | None = None
    translated_preview: list[dict[str, Any]] | None = None

    t_path = job_dir / "transcript.json"
    if t_path.exists():
        transcript_preview = json.loads(t_path.read_text(encoding="utf-8"))[:3]
    tr_path = job_dir / "translated.json"
    if tr_path.exists():
        translated_preview = [
            {k: seg.get(k) for k in ("index", "text", "translated_text")}
            for seg in json.loads(tr_path.read_text(encoding="utf-8"))[:3]
        ]

    return {
        "job_id": job_id,
        "status": job.status.value,
        "source_lang": job.source_lang,
        "target_lang": job.target_lang,
        "attempt": job.attempt,
        "failure_reason": job.failure_reason,
        "artifacts_ready": artifacts,
        "transcript_preview": transcript_preview,
        "translation_preview": translated_preview,
    }
