"""Stuck-job reaper — periodic Beat task that recovers dead-worker jobs.

The problem this solves: a worker grabs a task, starts it (job → RUNNING),
then the worker process is OOM-killed / the pod evicted. With acks_late the
broker *would* redeliver — but only after visibility_timeout (1h). The reaper
catches these much faster by scanning heartbeats: RUNNING jobs whose
heartbeat_at went silent past the TTL are dead.

Policy ladder:
- stale + attempts < max_attempts → FAILED (retry gateway may revive via requeue)
- stale + attempts >= max_attempts → DEAD_LETTERED (human review pile)
"""

import logging
import uuid
from typing import Any

from app.celery_app import celery_app
from app.models import JobStatus

logger = logging.getLogger(__name__)

# Tunables (env-overridable later via settings)
HEARTBEAT_TTL_SECONDS = 120  # heartbeat every ~30s; 2min silence = dead
REAPER_BATCH_LIMIT = 50


def _run_sync(job_ids: list[uuid.UUID], ttl: int) -> dict[str, Any]:
    """Sync bridge — runs inside asyncio.run() from the beat task."""
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.config import get_settings
    from app.repos import JobRepo

    async def inner() -> dict[str, Any]:
        engine = create_async_engine(get_settings().database_url)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with maker() as session:
                repo = JobRepo(session)
                return await _reap(repo)
        finally:
            await engine.dispose()

    async def _reap(repo: JobRepo) -> dict[str, Any]:
        from datetime import UTC, datetime

        # mark_stuck_failed handles FAILED transition; DLQ check per job below
        stale = await repo.find_stale_running(ttl_seconds=ttl, limit=REAPER_BATCH_LIMIT)
        if not stale:
            return {"requeued": 0, "dead_lettered": 0}

        dead_lettered = 0
        requeueable: list[uuid.UUID] = []
        for job in stale:
            if job.attempt >= job.max_attempts:
                await repo.transition(job, JobStatus.DEAD_LETTERED)
                dead_lettered += 1
            else:
                requeueable.append(job.id)

        if requeueable:
            await repo.mark_stuck_failed(requeueable)
            for jid in requeueable:
                logger.warning("reaper: job=%s stuck → FAILED (retryable)", jid)

        _ = datetime.now(UTC)  # keep import honest if logic shifts
        return {"requeued": len(requeueable), "dead_lettered": dead_lettered}

    return asyncio.run(inner())


@celery_app.task(name="mozhi.reaper.sweep")
def sweep_stale_jobs() -> dict[str, Any]:
    """Beat schedule: run every 60s. Never raises — a broken reaper must not
    crash the beat scheduler."""
    try:
        result = _run_sync([], HEARTBEAT_TTL_SECONDS)
        if result["requeued"] or result["dead_lettered"]:
            logger.info("reaper sweep: %s", result)
        return result
    except Exception:
        logger.exception("reaper sweep failed")
        return {"requeued": 0, "dead_lettered": 0, "error": "sweep failed"}
