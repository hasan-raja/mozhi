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

from celery import Task

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
    - exponential backoff with jitter on transient errors (prevents retry storms:
      a batch of failing tasks must not all retry at the same instant)
    - no retry on permanent errors (fail fast)
    """

    def decorator(fn: Callable[..., Any]) -> Any:
        @celery_app.task(name=f"mozhi.{stage}.run", bind=True, max_retries=MAX_RETRIES)
        def inner(self: Task, job_id: str, *args: Any, **kwargs: Any) -> Any:
            try:
                return fn(self, job_id, *args, **kwargs)
            except RETRYABLE_EXCEPTIONS as exc:
                countdown = BASE_BACKOFF_SECONDS * (2**self.request.retries) * (
                    1 + random.random() / 2
                )
                logger.warning(
                    "stage=%s job=%s transient error=%r — retry %d/%d in %.1fs",
                    stage, job_id, exc, self.request.retries + 1, MAX_RETRIES, countdown,
                )
                raise self.retry(exc=exc, countdown=countdown) from exc
            except Exception as exc:  # permanent — do not retry
                logger.exception("stage=%s job=%s permanent failure", stage, job_id)
                raise PermanentStageError(f"{stage}: {exc}") from exc

        return inner

    return decorator


def run_async(coro: Any) -> Any:
    """Bridge sync Celery workers into async DB code — fresh loop per task."""
    return asyncio.run(coro)
