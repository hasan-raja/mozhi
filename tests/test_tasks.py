"""Unit tests for the task fabric — routing, retry policy, stage chain.

These test pure logic without a live broker (Celery tasks are invoked eagerly).
"""

from app.celery_app import STAGE_QUEUES, celery_app
from app.pipeline_stages import STAGE_NAMES, STAGES
from app.tasks import _next_stage


def test_stage_registry_is_ordered_pipeline() -> None:
    assert STAGE_NAMES == ["extract", "vad", "diarize", "asr", "translate", "tts", "qc", "stitch"]


def test_every_stage_has_unique_queue() -> None:
    queues = [s.queue for s in STAGES]
    assert len(queues) == len(set(queues))
    assert set(STAGE_QUEUES) == set(STAGE_NAMES)


def test_celery_routes_tasks_to_stage_queues() -> None:
    routes = celery_app.conf.task_routes
    for stage in STAGE_NAMES:
        key = f"mozhi.{stage}.*"
        assert key in routes, f"missing route for {key}"
        assert routes[key]["queue"] == stage


def test_next_stage_chain() -> None:
    assert _next_stage("extract") == "vad"
    assert _next_stage("vad") == "diarize"
    assert _next_stage("diarize") == "asr"
    assert _next_stage("qc") == "stitch"
    assert _next_stage("stitch") is None  # terminal stage


def test_all_stage_tasks_registered() -> None:
    registered = celery_app.tasks
    for stage in STAGE_NAMES:
        assert f"mozhi.{stage}.run" in registered


def test_reliability_settings() -> None:
    conf = celery_app.conf
    # at-least-once delivery semantics
    assert conf.task_acks_late is True
    assert conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1
    # no pickle — JSON only attack surface
    assert set(conf.accept_content) == {"json"}
