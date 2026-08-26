"""TTS stage tests — mock mode synthesis + artifact contract (no network)."""

import json
import uuid
from pathlib import Path

import pytest

from app.stages.tts_stage import run_tts


def _seed_translated(job_id: str, tmp_root: Path) -> None:
    d = tmp_root / "data" / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "translated.json").write_text(json.dumps([
        {"index": 0, "start_ms": 0, "end_ms": 1500,
         "text": "hello", "translated_text": "[ta] vanakkam"},
        {"index": 1, "start_ms": 1500, "end_ms": 3000,
         "text": "world", "translated_text": "[ta] ulagam"},
    ]), encoding="utf-8")


def test_mock_tts_writes_wavs_and_durations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "mock")
    from app.config import get_settings
    get_settings.cache_clear()
    job_id = uuid.uuid4().hex
    _seed_translated(job_id, tmp_path)

    result = run_tts(job_id)

    assert result["segments"] == 2
    job_dir = tmp_path / "data" / "jobs" / job_id
    assert (job_dir / "tts" / "seg_0.wav").exists()
    assert (job_dir / "tts" / "seg_1.wav").exists()

    durations = json.loads((job_dir / "durations.json").read_text())
    assert len(durations) == 2
    # mock duration = len(text)*0.05s floor 0.5 → "[ta] vanakkam" (13ch) = 0.65s
    assert durations[0]["duration_ms"] == 650
    assert durations[0]["original_ms"] == 1500


def test_missing_translated_raises_permanent(tmp_path: Path, monkeypatch) -> None:
    from app.task_base import PermanentStageError

    monkeypatch.chdir(tmp_path)
    with pytest.raises(PermanentStageError):
        run_tts("nosuchjob")
