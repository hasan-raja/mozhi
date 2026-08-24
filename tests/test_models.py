"""Unit tests for the Job state machine — no DB needed."""

import pytest

from app.models import (
    TERMINAL_STATUSES,
    InvalidTransition,
    Job,
    JobStatus,
)


def make_job(status: JobStatus = JobStatus.PENDING) -> Job:
    job = Job(source_lang="en", target_lang="ta")
    job.status = status
    return job


def test_new_job_starts_pending() -> None:
    assert make_job().status == JobStatus.PENDING


def test_pending_to_running_sets_started_at() -> None:
    job = make_job()
    job.transition(JobStatus.RUNNING)
    assert job.status == JobStatus.RUNNING
    assert job.started_at is not None


def test_running_to_completed_sets_finished_at() -> None:
    job = make_job(JobStatus.RUNNING)
    job.transition(JobStatus.COMPLETED)
    assert job.finished_at is not None


def test_failed_can_retry_via_pending() -> None:
    job = make_job(JobStatus.FAILED)
    job.transition(JobStatus.PENDING)  # retry path
    assert job.status == JobStatus.PENDING


def test_illegal_transition_raises() -> None:
    with pytest.raises(InvalidTransition):
        make_job(JobStatus.PENDING).transition(JobStatus.COMPLETED)


def test_terminal_states_are_frozen() -> None:
    for status in TERMINAL_STATUSES:
        with pytest.raises(InvalidTransition):
            make_job(status).transition(JobStatus.RUNNING)


def test_failed_cannot_go_straight_to_dead_lettered_twice() -> None:
    job = make_job(JobStatus.DEAD_LETTERED)
    with pytest.raises(InvalidTransition):
        job.transition(JobStatus.PENDING)


def test_every_status_has_a_transition_map_entry() -> None:
    from app.models import VALID_TRANSITIONS
    assert set(VALID_TRANSITIONS) == set(JobStatus)
