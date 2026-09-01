"""Diarize stage integration tests — mock mode and simple_diarize path."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from app.tasks import run_diarize

ffmpeg_available = shutil.which("ffmpeg") is not None

JOB = "testjob_diarize_001"


def _seed_extracted_audio(tmp_path: Path, job_id: str) -> Path:
    """Create a REAL extracted audio.wav for diarize to process (16kHz mono)."""
    import numpy as np
    import soundfile as sf

    job_dir = tmp_path / "data" / "jobs" / job_id
    extract_dir = job_dir / "extract"
    extract_dir.mkdir(parents=True)
    wav = extract_dir / "audio.wav"

    # Generate 3 seconds of silence at 16kHz mono (proper WAV)
    sr = 16000
    duration_s = 3
    audio_data = np.zeros(sr * duration_s, dtype=np.float32)
    sf.write(str(wav), audio_data, sr)
    return wav


def _run_async_passthrough():
    """Make run_async a direct asyncio.run, skipping DB lookups."""
    import asyncio

    def _passthrough(coro):
        return asyncio.run(coro)
    return _passthrough


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not on PATH")
def test_diarize_simple_mock_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Diarize runs simple_diarize in mock engine mode and writes diarization.json."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "mock")
    from app.config import get_settings
    get_settings.cache_clear()

    _seed_extracted_audio(tmp_path, JOB)

    # Mock run_async to bypass DB
    run_async_pt = _run_async_passthrough()
    with patch("app.tasks.run_async", run_async_pt):
        result = run_diarize.__wrapped__(JOB)  # type: ignore[attr-defined]

    assert result["stage"] == "diarize"
    assert "segments" in result
    assert result["segments"] >= 1

    diarization_file = tmp_path / "data" / "jobs" / JOB / "diarization.json"
    assert diarization_file.exists()

    payload = json.loads(diarization_file.read_text())
    assert isinstance(payload, list)
    assert len(payload) >= 1
    # Each segment has speaker and gender
    for seg in payload:
        assert "speaker" in seg
        assert "gender" in seg
        assert seg["gender"] in ("male", "female", "unknown")


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not on PATH")
def test_diarize_simple_local_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Diarize runs simple_diarize in local mode (no HF_TOKEN)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "local")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("MOZHI_ENABLE_DIARIZATION", raising=False)
    from app.config import get_settings
    get_settings.cache_clear()

    _seed_extracted_audio(tmp_path, JOB)

    run_async_pt = _run_async_passthrough()
    with patch("app.tasks.run_async", run_async_pt):
        result = run_diarize.__wrapped__(JOB)  # type: ignore[attr-defined]

    assert result["stage"] == "diarize"
    assert "segments" in result

    diarization_file = tmp_path / "data" / "jobs" / JOB / "diarization.json"
    assert diarization_file.exists()

    payload = json.loads(diarization_file.read_text())
    assert len(payload) >= 1
    for seg in payload:
        assert "speaker" in seg
        assert "gender" in seg
        assert seg["gender"] in ("male", "female", "unknown")


def test_diarize_missing_audio_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing extracted audio → PermanentStageError."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "mock")
    from app.config import get_settings
    get_settings.cache_clear()

    from app.task_base import PermanentStageError

    with pytest.raises(PermanentStageError, match="extracted audio missing"):
        run_diarize.__wrapped__("ghostjob")  # type: ignore[attr-defined]


def test_diarize_segments_have_speaker_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diarization segments have SPEAKER_XX labels."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "mock")
    from app.config import get_settings
    get_settings.cache_clear()

    _seed_extracted_audio(tmp_path, JOB)

    run_async_pt = _run_async_passthrough()
    with patch("app.tasks.run_async", run_async_pt):
        run_diarize.__wrapped__(JOB)  # type: ignore[attr-defined]

    diarization_file = tmp_path / "data" / "jobs" / JOB / "diarization.json"
    payload = json.loads(diarization_file.read_text())

    for seg in payload:
        assert seg["speaker"].startswith("SPEAKER_")
        assert "start_ms" in seg
        assert "end_ms" in seg
        assert seg["end_ms"] > seg["start_ms"]