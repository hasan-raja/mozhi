"""Integration tests for the reaper policy ladder (live postgres)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models import Job, JobStatus
from app.repos import JobRepo

pytestmark = pytest.mark.integration


@pytest.fixture
async def db_session():
    engine = create_async_engine(get_settings().database_url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
        await session.rollback()
    await engine.dispose()


def _stale_job(attempts: int) -> Job:
    job = Job(
        source_lang="en",
        target_lang=f"ta{uuid.uuid4().hex[:6]}",
        status=JobStatus.RUNNING,
        attempt=attempts,
        max_attempts=3,
    )
    job.heartbeat_at = datetime.now(UTC) - timedelta(minutes=10)
    return job


async def test_reap_policy_via_repo(db_session) -> None:
    """attempt < max → FAILED (retryable); attempt >= max → DEAD_LETTERED."""
    young = _stale_job(attempts=0)
    old = _stale_job(attempts=3)
    db_session.add_all([young, old])
    await db_session.commit()

    # Capture ids NOW — after expire_all() below, attribute access on these
    # ORM instances would trigger lazy IO outside greenlet context.
    young_id, old_id = young.id, old.id

    repo = JobRepo(db_session)
    stale = await repo.find_stale_running(ttl_seconds=60)

    stale_ids = {j.id for j in stale}
    assert young.id in stale_ids and old.id in stale_ids

    requeueable_ids = [j.id for j in stale if j.attempt < j.max_attempts]
    dead_ids = [j.id for j in stale if j.attempt >= j.max_attempts]

    if requeueable_ids:
        await repo.mark_stuck_failed(requeueable_ids)
    # Re-fetch dead jobs fresh — mark_stuck_failed expired the session cache.
    for did in dead_ids:
        job_obj = await repo.get(did)
        assert job_obj is not None
        await repo.transition(job_obj, JobStatus.DEAD_LETTERED)

    # Re-read via fresh SQL — bulk UPDATE + expire_all invalidate ORM caches.
    from sqlalchemy import select as _select

    young_fresh = (await db_session.execute(
        _select(Job).where(Job.id == young_id)
    )).scalars().first()
    old_fresh = (await db_session.execute(
        _select(Job).where(Job.id == old_id)
    )).scalars().first()
    assert young_fresh is not None and young_fresh.status == JobStatus.FAILED
    assert old_fresh is not None and old_fresh.status == JobStatus.DEAD_LETTERED


async def test_failed_stuck_job_can_be_revived(db_session) -> None:
    """FAILED is the retry gateway: reaper-reaped jobs can go PENDING again."""
    job = _stale_job(attempts=0)
    db_session.add(job)
    await db_session.commit()
    job_id = job.id  # capture before expire_all in mark_stuck_failed

    repo = JobRepo(db_session)
    await repo.mark_stuck_failed([job_id])

    reaped = await repo.get(job_id)
    assert reaped is not None and reaped.status == JobStatus.FAILED
    assert reaped.failure_reason is not None and "reaper" in reaped.failure_reason

    # revival path
    await repo.transition(reaped, JobStatus.PENDING)
    assert reaped.status == JobStatus.PENDING


async def test_healthy_running_never_flagged(db_session) -> None:
    job = Job(source_lang="en", target_lang=f"hi{uuid.uuid4().hex[:6]}",
              status=JobStatus.RUNNING)
    job.heartbeat_at = datetime.now(UTC)  # just beat
    db_session.add(job)
    await db_session.commit()

    repo = JobRepo(db_session)
    stale_ids = {j.id for j in await repo.find_stale_running(ttl_seconds=60)}
    assert job.id not in stale_ids
