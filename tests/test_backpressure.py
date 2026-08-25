"""Backpressure middleware tests — no broker needed (gauge is injected)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.backpressure import BackpressureMiddleware, QueueDepthGauge, make_429


class FixedGauge(QueueDepthGauge):
    """Gauge that always reports a fixed depth."""

    def __init__(self, depth: int) -> None:
        self.depth = depth

    def total_depth(self) -> int:  # type: ignore[override]
        return self.depth


def make_app(depth: int, threshold: int = 10) -> TestClient:
    app = FastAPI()

    @app.post("/jobs")
    async def create_job() -> dict[str, str]:
        return {"status": "created"}

    @app.get("/jobs")
    async def list_jobs() -> dict[str, list[str]]:
        return {"jobs": []}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.add_middleware(
        BackpressureMiddleware,
        max_queue_depth=threshold,
        gauge=FixedGauge(depth),
    )
    return TestClient(app)


def test_under_threshold_allows_post() -> None:
    client = make_app(depth=5, threshold=10)
    resp = client.post("/jobs")
    assert resp.status_code == 200


def test_over_threshold_rejects_with_429_and_retry_after() -> None:
    client = make_app(depth=50, threshold=10)
    resp = client.post("/jobs")
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "30"
    assert resp.json()["error"] == "backpressure"


def test_reads_are_never_gated() -> None:
    client = make_app(depth=999, threshold=10)
    assert client.get("/jobs").status_code == 200
    assert client.get("/healthz").status_code == 200


def test_threshold_boundary_exactly_at_limit_rejects() -> None:
    client = make_app(depth=10, threshold=10)
    assert client.post("/jobs").status_code == 429


def test_make_429_shape() -> None:
    resp = make_429(60)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "60"


def test_gauge_failure_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the broker probe explodes, traffic flows (fail-open)."""

    class ExplodingGauge(FixedGauge):
        def total_depth(self) -> int:
            raise RuntimeError("broker down")

    app = FastAPI()

    @app.post("/jobs")
    async def create_job2() -> dict[str, str]:
        return {"status": "created"}

    app.add_middleware(
        BackpressureMiddleware, max_queue_depth=1,
        gauge=ExplodingGauge(0),  # type: ignore[arg-type]
    )
    client = TestClient(app)
    resp = client.post("/jobs")
    assert resp.status_code == 200  # fail open
