"""Task base class with retry/backoff policy.

Celery is untyped; we keep strict mypy by isolating celery-touching code here
with pragmatic ignores (each one commented).
"""

# ruff/mypy: celery lacks type stubs — isolation module keeps the blast radius small.
import asyncio
import logging
import random
from collections.abc import Callable
from typing import Any

from app.celery_app import celery_app

logger = logging.getLogger(__name__)

# Transient errors are worth retrying; permanent ones fail fast to FAILED.
RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 5


class PermanentStageError(Exception):
    """Raised when a stage fails in a way retrying cannot fix."""


def stage_task(stage: str) -> Callable[[Callable[..., Any]], Any]:
    """Decorator for pipeline stage tasks.

    Provides:
    - AUTOMATIC CHAINING: on success, dispatches the next stage to its queue
      (pipeline order comes from pipeline_stages.STAGES — single source of truth)
    - exponential backoff with jitter on transient errors (prevents retry storms:
      a batch of failing tasks must not all retry at the same instant)
    - no retry on permanent errors (fail fast)
    """

    def decorator(fn: Callable[..., Any]) -> Any:
        @celery_app.task(name=f"mozhi.{stage}.run", bind=True, max_retries=MAX_RETRIES)
        def inner(task_self: Any, job_id: str, *args: Any, **kwargs: Any) -> Any:
            # bind=True → Celery passes the task instance as first positional.
            # Stage bodies declare (self, job_id) to match; forward BOTH.
            try:
                result = fn(task_self, job_id, *args, **kwargs)
            except RETRYABLE_EXCEPTIONS as exc:
                countdown = BASE_BACKOFF_SECONDS * (2**task_self.request.retries) * (
                    1 + random.random() / 2
                )
                logger.warning(
                    "stage=%s job=%s transient error=%r — retry %d/%d in %.1fs",
                    stage, job_id, exc, task_self.request.retries + 1, MAX_RETRIES, countdown,
                )
                raise task_self.retry(exc=exc, countdown=countdown) from exc
            except Exception as exc:  # permanent — do not retry
                logger.exception("stage=%s job=%s permanent failure", stage, job_id)
                raise PermanentStageError(f"{stage}: {exc}") from exc

            _chain_next(stage, job_id)
            return result

        return inner

    return decorator


def _chain_next(current_stage: str, job_id: str) -> None:
    """Dispatch the next pipeline stage for this job (no-op on final stage)."""
    from app.pipeline_stages import STAGES

    names = [s.name for s in STAGES]
    i = names.index(current_stage)
    if i + 1 >= len(names):
        logger.info("stage=%s job=%s is terminal — pipeline complete", current_stage, job_id)
        return
    nxt = STAGES[i + 1]
    celery_app.send_task(f"mozhi.{nxt.name}.run", args=[job_id], queue=nxt.queue)
    logger.info("chained job=%s %s → %s", job_id, current_stage, nxt.name)


def run_async(coro: Any) -> Any:
    """Bridge sync Celery workers into async DB code — fresh loop per task."""
    return asyncio.run(coro)
