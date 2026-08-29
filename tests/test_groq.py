"""Groq translation engine tests — mocked HTTP, real parsing logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.engines.groq import GroqTranslationEngine


class TestGroqTranslationEngine:
    @pytest.fixture
    def engine(self):
        return GroqTranslationEngine(api_key="test-key")

    @pytest.mark.asyncio
    async def test_translate_empty_list(self, engine):
        assert await engine.translate([], "ta") == []

    @pytest.mark.asyncio
    async def test_translate_no_key_raises(self):
        engine = GroqTranslationEngine(api_key="")
        with pytest.raises(RuntimeError, match="GROQ_API_KEY not set"):
            await engine.translate(["hello"], "ta")

    @pytest.mark.asyncio
    async def test_translate_success(self, engine):
        mock_response = {
            "choices": [{
                "message": {
                    "content": "1. வணக்கம் உலகம்\n2. நான் நன்றாக இருக்கிறேன்\n3. நன்றி"
                }
            }]
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            result = await engine.translate(
                ["Hello world", "I am fine", "Thank you"],
                "ta"
            )

        assert result == ["வணக்கம் உலகம்", "நான் நன்றாக இருக்கிறேன்", "நன்றி"]

    @pytest.mark.asyncio
    async def test_translate_parses_numbered_lines(self, engine):
        """Parser handles various number formats."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": "1) First line\n2. Second line\n3- Third line"
                }
            }]
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            result = await engine.translate(["a", "b", "c"], "ta")

        assert result == ["First line", "Second line", "Third line"]

    @pytest.mark.asyncio
    async def test_translate_pads_missing_lines(self, engine):
        """Parser pads missing lines with empty strings."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": "1. Only one line"
                }
            }]
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            result = await engine.translate(["a", "b", "c"], "ta")

        assert result == ["Only one line", "", ""]

    def test_parser_keeps_missing_numbered_lines_empty(self):
        """Never shift dialogue into a neighbouring source segment."""
        raw = "1) First line\n3- Third line"

        assert GroqTranslationEngine._parse_numbered_lines(raw, 3) == [
            "First line",
            "",
            "Third line",
        ]

    def test_parser_ignores_unnumbered_content_when_indices_are_missing(self):
        raw = "1) First line\nUnnumbered commentary"

        assert GroqTranslationEngine._parse_numbered_lines(raw, 2) == ["First line", ""]

    @pytest.mark.asyncio
    async def test_translate_handles_http_error(self, engine):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient.post", return_value=mock_resp):
            with pytest.raises(Exception, match="401 Unauthorized"):
                await engine.translate(["hello"], "ta")