"""Health endpoint smoke test — proves the factory + test client wiring works."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_healthz_returns_ok() -> None:
    client = TestClient(create_app())
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_app_factory_returns_distinct_instances() -> None:
    """Factory pattern: two calls produce independent apps (test isolation)."""
    assert create_app() is not create_app()
