"""Backpressure — admission control at the API edge.

Principle: reject work at ingress (clean 429 the client can retry) rather than
accepting jobs that will OOM workers or sit in queues for hours.

Mechanism: Redis counters of queued task depth per stage. Middleware compares
total depth against a configurable threshold; over threshold → 429 + Retry-After.
"""

import logging
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.celery_app import celery_app

logger = logging.getLogger(__name__)


class QueueDepthGauge:
    """Reads live queue depth from the broker via Celery inspection.

    Note: inspect() is an active round-trip to workers — fine for portfolio
    scale; production would read Redis LLEN per queue instead (cheaper).
    Kept as a class so that swap is one method.
    """

    def __init__(self, app: Any = celery_app) -> None:
        self.app = app

    def total_depth(self) -> int:
        try:
            inspect = self.app.control.inspect()
            reserved: dict[str, list[dict[str, Any]]] | None = inspect.reserved()
            # reserved = tasks picked/queued by each worker but not yet done
            return sum(len(tasks) for tasks in (reserved or {}).values())
        except RuntimeError:
            # Re-raise programming errors from the probe itself? No — the gauge
            # contract is: never raise. Log and fail open.
            logger.exception("queue depth probe raised — failing open")
            return 0
        except Exception:
            logger.exception("queue depth probe failed — failing open")
            return 0  # fail open: don't block traffic because monitoring broke


def make_429(retry_after: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        content={
            "error": "backpressure",
            "detail": "system at capacity; retry later",
            "retry_after_seconds": retry_after,
        },
    )


class BackpressureMiddleware(BaseHTTPMiddleware):
    """Rejects POST /jobs with 429 when the system is saturated.

    Only job-creating routes are gated — reads/health checks stay free.
    """

    def __init__(
        self,
        app: Any,
        max_queue_depth: int = 100,
        retry_after: int = 30,
        gauge: QueueDepthGauge | None = None,
        protected_paths: tuple[str, ...] = ("/jobs",),
    ) -> None:
        super().__init__(app)
        self.max_queue_depth = max_queue_depth
        self.retry_after = retry_after
        self.gauge = gauge or QueueDepthGauge()
        self.protected_paths = protected_paths

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        is_protected = any(
            request.url.path == p or request.url.path.startswith(p + "/")
            for p in self.protected_paths
        )
        if request.method == "POST" and is_protected:
            try:
                depth = self.gauge.total_depth()
            except Exception:
                # Gauge contract violation — still fail open, never take down
                # the API because monitoring is broken.
                logger.exception("gauge raised in dispatch — failing open")
                depth = 0
            if depth >= self.max_queue_depth:
                logger.warning(
                    "backpressure: depth=%d >= %d → 429",
                    depth, self.max_queue_depth,
                )
                return make_429(self.retry_after)
        return await call_next(request)
