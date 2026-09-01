"""TTS stage tests — mock mode synthesis + artifact contract (no network)."""

import json
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from app.stages.tts_stage import run_tts
from app.task_base import PermanentStageError


def test_mock_tts_writes_wavs_and_durations(tmp_path: Path, monkeypatch):
    """Mock mode writes silent WAVs + durations.json without network."""
    monkeypatch.setattr("app.stages.tts_stage.DATA_ROOT", tmp_path)
    # Mock _load_target_lang to avoid DB lookup
    monkeypatch.setattr("app.stages.tts_stage._load_target_lang", lambda _: "ta")
    # Mock engine_mode to "mock" 
    from app.config import Settings
    monkeypatch.setattr("app.config.get_settings", lambda: Settings(engine_mode="mock"))
    job_id = str(uuid.uuid4())
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)

    (job_dir / "translated.json").write_text(json.dumps([
        {"index": 0, "start_ms": 0, "end_ms": 1000, "translated_text": "Hello world"},
        {"index": 1, "start_ms": 1000, "end_ms": 2000, "translated_text": "Goodbye"},
    ]))

    result = run_tts(job_id)

    assert result["stage"] == "tts"
    assert result["segments"] == 2
    assert (job_dir / "tts" / "seg_0.wav").exists()
    assert (job_dir / "tts" / "seg_1.wav").exists()

    durations = json.loads((job_dir / "durations.json").read_text())
    assert len(durations) == 2
    for d in durations:
        assert d["duration_ms"] > 0
        assert "original_ms" in d
        assert d["tempo_factor"] == 1.0
        assert d["rate_used"] == "+0%"


def test_missing_translated_raises_permanent(tmp_path: Path, monkeypatch):
    """run_tts without translated.json raises PermanentStageError."""
    monkeypatch.setattr("app.stages.tts_stage.DATA_ROOT", tmp_path)
    job_id = "missing-translated"

    with pytest.raises(PermanentStageError, match="translated.json missing"):
        run_tts(job_id)


@pytest.mark.asyncio
async def test_synthesize_with_duration_match_hits_target_within_tolerance():
    """_synthesize_with_duration_match should adjust rate to hit target duration within 10%."""
    from app.stages.tts_stage import _synthesize_with_duration_match

    call_count = {"n": 0}

    async def mock_synthesize_edge(
        text: str, voice: str, out_path: Path, rate: str = "+0%"
    ) -> float:
        call_count["n"] += 1
        # Simulate: first call at +0% gives 3000ms (50% longer than 2000ms target)
        # Second call at +50% gives 2100ms (within 10% of 2000ms)
        if call_count["n"] == 1:
            return 3.0
        return 2.1

    with patch("app.stages.tts_stage._synthesize_edge", mock_synthesize_edge), \
         patch("app.stages.tts_stage._probe_duration", side_effect=[3.0, 2.1]):
        duration, rate_used = await _synthesize_with_duration_match(
            "test text", "ta-IN-PallaviNeural", Path("/tmp/test.wav"), 2000
        )

    assert abs(duration - 2.0) <= 0.2  # within 10% of 2000ms = 2.0s
    assert rate_used == "+50%"  # 50% drift + 10% buffer = 60% → capped at 50%


@pytest.mark.asyncio
async def test_synthesize_with_duration_match_falls_back_to_atempo_when_rate_capped():
    """When edge-tts rate hits ±50% cap, ffmpeg atempo stretches to target."""
    from app.stages.tts_stage import _synthesize_with_duration_match

    call_count = {"n": 0}

    async def mock_synthesize_edge(
        text: str, voice: str, out_path: Path, rate: str = "+0%"
    ) -> float:
        call_count["n"] += 1
        # Always return 3000ms (too long) regardless of rate
        return 3.0

    with patch("app.stages.tts_stage._synthesize_edge", mock_synthesize_edge), \
         patch("app.stages.tts_stage._probe_duration", return_value=3.0), \
         patch("app.stages.tts_stage._ffmpeg_atempo") as mock_atempo:
        duration, rate_used = await _synthesize_with_duration_match(
            "test text", "ta-IN-PallaviNeural", Path("/tmp/test.wav"), 2000
        )

    assert abs(duration - 2.0) < 0.01  # exact target via atempo
    assert rate_used.startswith("atempo=")
    mock_atempo.assert_called_once()


def test_run_tts_passes_original_ms_to_duration_matcher(tmp_path: Path, monkeypatch):
    """run_tts should call _synthesize_with_duration_match with target_duration_ms."""
    from app.stages import tts_stage

    call_args = {}

    async def mock_duration_match(text, voice, out_path, target_ms):
        call_args["target_ms"] = target_ms
        return 1.5, "+0%"

    monkeypatch.setattr(tts_stage, "_synthesize_with_duration_match", mock_duration_match)
    monkeypatch.setattr("app.stages.tts_stage.DATA_ROOT", tmp_path)
    # Mock _load_target_lang to avoid DB lookup
    monkeypatch.setattr("app.stages.tts_stage._load_target_lang", lambda _: "ta")
    # Use "local" mode so it calls _synthesize_with_duration_match
    from app.config import Settings
    monkeypatch.setattr("app.config.get_settings", lambda: Settings(engine_mode="local"))

    job_id = str(uuid.uuid4())
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "translated.json").write_text(json.dumps([
        {"index": 0, "start_ms": 0, "end_ms": 2500, "translated_text": "Hello"},
    ]))

    run_tts(job_id)

    assert call_args["target_ms"] == 2500