"""Job submission from a local file path — simpler than multipart upload for
single-user portfolio use.

POST /jobs  {"source_path": "F:/videos/my_speech.mp4", "target_lang": "ta"}
  → copies the file into data/jobs/{id}/source/, creates the Job row,
    dispatches the full Celery chain, returns job_id.

The API validates the path exists and is a media extension; the actual bytes
never pass through the API process (a real multi-tenant deployment would use
multipart upload to blob storage instead — documented in NOTES.md).
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.celery_app import celery_app
from app.pipeline_stages import STAGE_NAMES
from app.tasks import DATA_ROOT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wav", ".mp3", ".m4a"}


class CreateJobRequest(BaseModel):
    source_path: str
    target_lang: str = "ta"
    source_lang: str = "en"


def _dispatch_chain(job_id: str) -> None:
    """Queue extract first; each stage chains the next on success."""
    celery_app.send_task("mozhi.extract.run", args=[job_id], queue="extract")  # type: ignore[no-untyped-call]


@router.post("")
async def create_job(req: CreateJobRequest) -> dict[str, Any]:
    source = Path(req.source_path)
    if not source.exists():  # noqa: ASYNC240 — local disk stat, portfolio scale
        raise HTTPException(
            status_code=404, detail=f"file not found: {req.source_path}"
        )
    if source.suffix.lower() not in ALLOWED_EXTENSIONS:
        allowed = sorted(ALLOWED_EXTENSIONS)
        raise HTTPException(
            status_code=415,
            detail=f"unsupported media type '{source.suffix}'; allowed: {allowed}",
        )

    job_id = uuid.uuid4().hex
    dest_dir = DATA_ROOT / "jobs" / job_id / "source"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"input{source.suffix.lower()}"
    shutil.copy2(source, dest)  # noqa: ASYNC240 — local disk copy, portfolio scale

    # DB row: explicit id so artifact layout (data/jobs/{id}/) lines up.
    # NOTE: we're inside the running event loop here — await directly,
    # never asyncio.run() (that raises RuntimeError in FastAPI handlers).
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.config import get_settings
    from app.models import Job

    engine = create_async_engine(get_settings().database_url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            session.add(Job(
                id=uuid.UUID(job_id),
                source_lang=req.source_lang,
                target_lang=req.target_lang,
            ))
            await session.commit()
    finally:
        await engine.dispose()

    _dispatch_chain(job_id)
    logger.info("job %s created from %s → %s (target=%s)",
                job_id, source, dest, req.target_lang)
    return {
        "job_id": job_id,
        "status": "submitted",
        "source": str(dest),
        "target_lang": req.target_lang,
        "stages": STAGE_NAMES,
        "poll": f"/jobs/{job_id}",
    }
