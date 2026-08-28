"""Groq translation engine — free-tier LLM translation via Groq API.

Uses Groq's available OpenAI-compatible models for fast, free translation.
Batch design: ALL segments go in ONE chat completion. Context matters for
dialogue (back-and-forth lines translate wrong in isolation), and one call
beats N calls for rate-limit budgets.

Contract: model returns numbered lines matching input count; we parse and
pad/truncate to guarantee 1:1 mapping with the source segments.
"""

import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"  # Groq free tier model

LANG_NAMES = {
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "en": "English",
}

PROMPT_TEMPLATE = """Translate each numbered line below into {lang_name}.
Rules:
- Return EXACTLY {n} lines, numbered 1..{n}, nothing else.
- Translate meaning naturally (these are dialogue lines from a video).
- Keep names and numbers unchanged.
- No explanations, no extra text.

Lines:
{lines}"""


class GroqTranslationEngine:
    """LLM translation via Groq's free-tier OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None, model: str = MODEL) -> None:
        settings = get_settings()
        self.api_key = settings.groq_api_key if api_key is None else api_key
        self.model = model

    async def translate(self, texts: list[str], target_lang: str) -> list[str]:
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set — add it to .env "
                "(free key at https://console.groq.com/keys)"
            )

        lang_name = LANG_NAMES.get(target_lang, target_lang)
        lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = PROMPT_TEMPLATE.format(lang_name=lang_name, n=len(texts), lines=lines)

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,  # translation wants determinism
                },
            )
            resp.raise_for_status()
            data = resp.json()

        raw: str = data["choices"][0]["message"]["content"]
        translated = self._parse_numbered_lines(raw, len(texts))
        logger.info("groq: translated %d segments → %s", len(translated), target_lang)
        return translated

    @staticmethod
    def _parse_numbered_lines(raw: str, expected: int) -> list[str]:
        """Parse '1. text' style lines; pad/truncate to exactly `expected`."""
        out: dict[int, str] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            for sep in (". ", ") ", ".)", "- "):
                if sep in line[:6]:
                    num_part, _, text = line.partition(sep)
                    if num_part.strip().isdigit():
                        out[int(num_part)] = text.strip()
                    break
        result = [out.get(i + 1, "") for i in range(expected)]
        # fill any gaps from unparsed trailing content
        if any(not r for r in result):
            fallback = [
                ln.strip() for ln in raw.splitlines()
                if ln.strip() and not ln[:3].rstrip(". ").isdigit()
            ]
            for i, v in enumerate(result):
                if not v and fallback:
                    result[i] = fallback.pop(0)
        return result


def get_groq_engines() -> dict[str, Any]:
    return {"translate": GroqTranslationEngine()}