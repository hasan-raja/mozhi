"""Stage orchestration — connects Celery tasks to the Job state machine.

Flow per stage execution:
1. claim/verify job is in an expected state
2. transition RUNNING (idempotent — already-RUNNING is fine on redelivery)
3. heartbeat while working
4. run stage body
5. mark completion in ledger; chain next stage or finish job

Idempotency contract: if the stage already appears in stage_completions,
this execution is a duplicate delivery → skip work entirely.
"""

import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import InvalidTransition, Job, JobStatus
from app.repos import JobRepo

logger = logging.getLogger(__name__)


class StagePreconditionError(Exception):
    """Job not in a state that allows this stage to run."""


async def begin_stage(session: AsyncSession, job_id: str, stage: str) -> tuple[Job, bool]:
    """Verify state and mark RUNNING. Returns (job, should_run).

    should_run=False means this is a duplicate delivery of a completed stage —
    caller skips work and just chains forward.
    """
    repo = JobRepo(session)
    job = await repo.get(uuid.UUID(job_id))
    if job is None:
        raise StagePreconditionError(f"job {job_id} not found")

    completion_repo = StageCompletionRepoCompat(session)
    if await completion_repo.is_completed(job.id, stage):
        logger.info("stage=%s job=%s already completed — duplicate delivery", stage, job_id)
        return job, False

    if job.status == JobStatus.PENDING:
        await repo.transition(job, JobStatus.RUNNING)
    elif job.status != JobStatus.RUNNING:
        raise StagePreconditionError(f"job {job_id} in {job.status.value}; cannot run {stage}")

    await repo.heartbeat(job.id)
    return job, True


async def complete_stage(
    session: AsyncSession, job_id: str, stage: str, result: dict[str, Any]
) -> bool:
    """Record completion in the idempotency ledger. Returns True if THIS call won."""
    repo = StageCompletionRepoCompat(session)
    won = await repo.try_mark_completed(uuid.UUID(job_id), stage)
    await session.commit()
    return won


class StageCompletionRepoCompat:
    """Thin adapter so orchestration doesn't import repos directly (avoids cycle)."""

    def __init__(self, session: AsyncSession) -> None:
        from app.repos import StageCompletionRepo

        self._repo = StageCompletionRepo(session)

    async def try_mark_completed(self, job_id: uuid.UUID, stage: str) -> bool:
        return await self._repo.try_mark_completed(job_id, stage)

    async def is_completed(self, job_id: uuid.UUID, stage: str) -> bool:
        return await self._repo.is_completed(job_id, stage)


def guarded_transition(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap async ops so InvalidTransition becomes a logged, non-fatal skip."""

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except InvalidTransition as exc:
            logger.warning("transition rejected: %s", exc)
            return None

    return wrapper
