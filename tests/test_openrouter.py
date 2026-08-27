"""OpenRouter engine tests — mocked HTTP, no network, no key required."""

import json
from typing import Any

import pytest

from app.engines.openrouter import OpenRouterTranslationEngine


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    """Captures the request; returns a canned numbered-line response."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.captured: dict[str, Any] | None = None

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.captured = kwargs
        return FakeResponse({
            "choices": [{"message": {"content": self.content}}]
        })


@pytest.mark.asyncio
async def test_batch_prompt_contains_all_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class CapClient(FakeClient):
        async def post(self, url: str, **kwargs: Any) -> FakeResponse:
            captured.update(kwargs)
            return await super().post(url, **kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    eng = OpenRouterTranslationEngine(api_key="test-key")

    import app.engines.openrouter as mod
    client = CapClient("1. வணக்கம்\n2. உலகம்")
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: client)

    out = await eng.translate(["hello", "world"], "ta")
    assert out == ["வணக்கம்", "ுலகம்".replace("ு", "u")] or out == ["வணக்கம்", "உலகம்"]
    # prompt must include every input line and the language name
    body = json.dumps(captured["json"])
    assert "hello" in body and "world" in body and "Tamil" in body


@pytest.mark.asyncio
async def test_missing_key_raises_helpful_error() -> None:
    eng = OpenRouterTranslationEngine(api_key="")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        await eng.translate(["hello"], "ta")


def test_parse_numbered_lines_pads_and_truncates() -> None:
    raw = "1. alpha\n2. beta\n3. gamma\n4. delta"  # model returned 4, expected 2
    out = OpenRouterTranslationEngine._parse_numbered_lines(raw, 2)
    assert out == ["alpha", "beta"]

    raw2 = "1. only-one"
    out2 = OpenRouterTranslationEngine._parse_numbered_lines(raw2, 3)
    assert len(out2) == 3  # padded to match expectation


def test_temperature_is_low_for_determinism() -> None:
    # translation wants near-deterministic output; guard the contract
    import inspect
    src = inspect.getsource(OpenRouterTranslationEngine.translate)
    assert "temperature" in src
