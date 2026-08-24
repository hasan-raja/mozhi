"""Unit tests for stage orchestration logic (no broker, no DB)."""

import pytest

from app.orchestration import StagePreconditionError
from app.pipeline_stages import STAGE_NAMES


class FakeJob:
    """Minimal job stand-in — state machine behavior comes from the real model."""

    def __init__(self, status) -> None:
        from app.models import Job

        self.id = __import__("uuid").uuid4()
        self.status = status
        self._job_cls = Job


def _make_job(status):
    import uuid as uuid_mod

    from app.models import Job

    job = Job(source_lang="en", target_lang="ta")
    job.id = uuid_mod.uuid4()
    job.status = status
    return job


@pytest.mark.asyncio
async def test_duplicate_delivery_short_circuits(monkeypatch) -> None:
    """If ledger says completed → should_run=False without touching state."""
    from app import orchestration

    class FakeRepo:
        async def get(self, job_id):
            return _make_job(__import__("app.models", fromlist=["JobStatus"]).JobStatus.COMPLETED)

        async def heartbeat(self, job_id):
            return None

    class FakeCompletion:
        async def is_completed(self, job_id, stage):
            return True

    monkeypatch.setattr(orchestration, "JobRepo", lambda s: FakeRepo())
    monkeypatch.setattr(orchestration, "StageCompletionRepoCompat", lambda s: FakeCompletion())

    job, should_run = await orchestration.begin_stage(
        None, "00000000-0000-0000-0000-000000000001", "asr"
    )
    assert should_run is False


@pytest.mark.asyncio
async def test_missing_job_raises_precondition(monkeypatch) -> None:
    from app import orchestration

    class FakeRepo:
        async def get(self, job_id):
            return None

    monkeypatch.setattr(orchestration, "JobRepo", lambda s: FakeRepo())

    with pytest.raises(StagePreconditionError):
        await orchestration.begin_stage(None, "00000000-0000-0000-0000-000000000002", "asr")


@pytest.mark.asyncio
async def test_failed_job_cannot_run_stage(monkeypatch) -> None:
    from app import orchestration
    from app.models import JobStatus

    class FakeRepo:
        async def get(self, job_id):
            return _make_job(JobStatus.FAILED)

    class FakeCompletion:
        async def is_completed(self, job_id, stage):
            return False

    monkeypatch.setattr(orchestration, "JobRepo", lambda s: FakeRepo())
    monkeypatch.setattr(orchestration, "StageCompletionRepoCompat", lambda s: FakeCompletion())

    with pytest.raises(StagePreconditionError):
        await orchestration.begin_stage(None, "00000000-0000-0000-0000-000000000003", "asr")


def test_stage_names_cover_full_chain() -> None:
    assert STAGE_NAMES[0] == "extract" and STAGE_NAMES[-1] == "stitch"
