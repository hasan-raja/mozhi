"""Integration tests — run against the real docker Postgres (port 5433).

These prove DB-level behavior that unit tests can't: unique constraints,
transactional rollback, and the reaper query. Skipped automatically if the
database isn't reachable (so CI without services still passes unit tests).
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models import Job, JobStatus
from app.repos import JobRepo, StageCompletionRepo

pytestmark = pytest.mark.integration


@pytest.fixture
async def db_session():
    """Session against the live dev database, rolled back after each test."""
    engine = create_async_engine(get_settings().database_url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
        await session.rollback()
    await engine.dispose()


pytestmark_db = None  # marker registration helper


def db_available() -> bool:
    import asyncio

    import asyncpg

    async def probe() -> bool:
        try:
            c = await asyncpg.connect(
                get_settings().database_url.replace("postgresql+asyncpg://", "postgresql://")
            )
            await c.close()
            return True
        except Exception:
            return False

    return asyncio.run(probe())


if not db_available():
    pytestmark = pytest.mark.skip(reason="dev postgres not reachable")


async def test_unique_constraint_blocks_double_completion(db_session) -> None:
    """THE idempotency guarantee: two inserts of same (job_id, stage) → second fails."""
    job = Job(source_lang="en", target_lang="ta")
    db_session.add(job)
    await db_session.flush()

    repo = StageCompletionRepo(db_session)
    first = await repo.try_mark_completed(job.id, "asr")
    assert first is True

    second = await repo.try_mark_completed(job.id, "asr")
    assert second is False  # duplicate delivery loses the race


async def test_different_stages_both_complete(db_session) -> None:
    job = Job(source_lang="en", target_lang="ta")
    db_session.add(job)
    await db_session.flush()

    repo = StageCompletionRepo(db_session)
    assert await repo.try_mark_completed(job.id, "vad") is True
    assert await repo.try_mark_completed(job.id, "asr") is True


async def test_is_completed_reflects_ledger(db_session) -> None:
    job = Job(source_lang="en", target_lang="ta")
    db_session.add(job)
    await db_session.flush()

    repo = StageCompletionRepo(db_session)
    assert await repo.is_completed(job.id, "tts") is False
    await repo.try_mark_completed(job.id, "tts")
    assert await repo.is_completed(job.id, "tts") is True


async def test_reaper_finds_only_stale_running(db_session) -> None:
    job = Job(source_lang="en", target_lang="ta", status=JobStatus.RUNNING)
    job.started_at = datetime.now(UTC)
    # heartbeat 10 minutes ago with a 60s TTL → stale
    job.heartbeat_at = datetime.now(UTC) - timedelta(minutes=10)
    db_session.add(job)

    fresh = Job(source_lang="en", target_lang="hi", status=JobStatus.RUNNING)
    fresh.heartbeat_at = datetime.now(UTC)  # alive — must NOT be flagged
    db_session.add(fresh)

    pending = Job(source_lang="en", target_lang="hi", status=JobStatus.PENDING)
    db_session.add(pending)
    await db_session.commit()

    repo = JobRepo(db_session)
    stale = await repo.find_stale_running(ttl_seconds=60)
    ids = {j.id for j in stale}
    assert job.id in ids
    assert fresh.id not in ids
    assert pending.id not in ids


async def test_heartbeat_updates_timestamp(db_session) -> None:
    job = Job(source_lang="en", target_lang="ta")
    db_session.add(job)
    await db_session.commit()

    old = datetime.now(UTC) - timedelta(hours=1)
    await db_session.execute(
        text("UPDATE jobs SET heartbeat_at = :ts WHERE id = :id"),
        {"ts": old, "id": str(job.id)},
    )
    await db_session.commit()

    repo = JobRepo(db_session)
    await repo.heartbeat(job.id)

    # Re-read via SQL (session cache is bypassed) to verify the UPDATE landed.
    from sqlalchemy import text as _text

    result = await db_session.execute(
        _text("SELECT heartbeat_at FROM jobs WHERE id = :id"),
        {"id": str(job.id)},
    )
    db_val = result.scalar()
    assert db_val is not None
    assert db_val > old
