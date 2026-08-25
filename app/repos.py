"""Job repository — all DB access for jobs lives here.

Repository pattern: Celery tasks and API endpoints share one data vocabulary,
and tests can mock this layer instead of a live database.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (  # noqa: F401 (Segment re-exported)
    VALID_TRANSITIONS,
    Job,
    JobStatus,
    Segment,
)


class JobRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, job_id: uuid.UUID) -> Job | None:
        return await self.session.get(Job, job_id)

    async def create(self, source_lang: str, target_lang: str) -> Job:
        job = Job(source_lang=source_lang, target_lang=target_lang)
        self.session.add(job)
        await self.session.commit()
        return job

    async def transition(self, job: Job, to: JobStatus) -> Job:
        """Guarded state transition (app-level check from Step 3a)."""
        job.transition(to)  # raises InvalidTransition on illegal moves
        await self.session.commit()
        return job

    async def heartbeat(self, job_id: uuid.UUID) -> None:
        """Worker liveness signal — the reaper scans stale heartbeats.

        UPDATE-then-refresh: after commit we expire cached instances so
        subsequent attribute access re-selects from the DB (the session cache
        would otherwise hide the new value written by this very UPDATE).
        """
        await self.session.execute(
            update(Job).where(Job.id == job_id).values(heartbeat_at=datetime.now(UTC))
        )
        await self.session.commit()

    async def find_stale_running(
        self, ttl_seconds: int, limit: int = 50
    ) -> list[Job]:
        """Reaper query: RUNNING jobs whose heartbeat is older than the TTL.

        Served by ix_jobs_status_created — designed back in Step 3a.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=ttl_seconds)
        result = await self.session.execute(
            select(Job)
            .where(
                Job.status == JobStatus.RUNNING,
                Job.heartbeat_at.is_not(None),
                Job.heartbeat_at < cutoff,
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def claim_next_pending(self) -> Job | None:
        """Atomically hand the oldest PENDING job to one worker.

        SELECT ... FOR UPDATE SKIP LOCKED: concurrent workers never grab the
        same row — the lock skips rows another transaction already holds, so
        there's no waiting and no double-claim. This is the standard pattern
        for job queues built on Postgres.
        """
        result = await self.session.execute(
            select(Job)
            .where(Job.status == JobStatus.PENDING)
            .order_by(Job.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = result.scalars().first()
        if job is None:
            return None
        job.transition(JobStatus.RUNNING)
        job.heartbeat_at = datetime.now(UTC)
        await self.session.commit()
        return job

    async def mark_stuck_failed(self, job_ids: list[uuid.UUID]) -> None:
        """Send stuck jobs to FAILED with a reason (retry gateway may revive them)."""
        if not job_ids:
            return
        await self.session.execute(
            update(Job)
            .where(Job.id.in_(job_ids), Job.status == JobStatus.RUNNING)
            .values(
                status=JobStatus.FAILED,
                failure_reason="stuck: heartbeat expired (reaper)",
                finished_at=datetime.now(UTC),
            )
        )
        await self.session.commit()
        # Bulk UPDATE bypasses the ORM identity cache — expire so subsequent
        # repo.get() calls see the new state (same gotcha as heartbeat()).
        self.session.expire_all()


class StageCompletionRepo:
    """Idempotency ledger: one row per completed (job_id, stage).

    The unique constraint is the enforcement — two workers racing to complete
    the same stage cannot both insert; the loser knows it lost and skips.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def try_mark_completed(self, job_id: uuid.UUID, stage: str) -> bool:
        """Returns True if THIS call won the right to have completed the stage.

        False means another execution already completed it — caller must skip
        downstream work and just chain forward (idempotent no-op).
        """
        from app.models import StageCompletion  # local import avoids cycles

        self.session.add(StageCompletion(job_id=job_id, stage=stage))
        try:
            await self.session.commit()
            return True
        except Exception:
            # unique violation = already completed by a duplicate delivery
            await self.session.rollback()
            return False

    async def is_completed(self, job_id: uuid.UUID, stage: str) -> bool:
        from app.models import StageCompletion

        result = await self.session.execute(
            select(StageCompletion).where(
                StageCompletion.job_id == job_id,
                StageCompletion.stage == stage,
            )
        )
        return result.first() is not None
