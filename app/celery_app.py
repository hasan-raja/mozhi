"""Celery application — per-stage queues for independent scaling.

Each pipeline stage gets its own queue so stages scale independently:
ASR is GPU-heavy (1-2 workers), TTS is API-bound (many light workers).
"""

from celery import Celery

from app.config import get_settings
from app.pipeline_stages import STAGES

settings = get_settings()

STAGE_QUEUES = [s.queue for s in STAGES]

celery_app = Celery(
    "mozhi",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    # Explicit task registration (no autodiscovery magic — explicit > implicit)
    imports=["app.tasks"],
    task_routes={
        f"mozhi.{stage}.*": {"queue": stage} for stage in STAGE_QUEUES
    },
    # Reliability defaults — at-least-once delivery
    task_acks_late=True,
    task_reject_on_worker_lost=True,  # redeliver if worker is OOM-killed mid-task
    worker_prefetch_multiplier=1,  # fair dispatch; long tasks shouldn't hog queues
    task_track_started=True,
    broker_transport_options={"visibility_timeout": 3600},  # 1h max silent task
    # Serialization: JSON only — no pickle attack surface
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Beat schedule — the reaper sweeps every minute
    beat_schedule={
        "reaper-sweep": {
            "task": "mozhi.reaper.sweep",
            "schedule": 60.0,
        },
    },
)
