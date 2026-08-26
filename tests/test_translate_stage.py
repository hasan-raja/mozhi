"""Translate stage tests — mock engine, artifact contract, metering."""

import json
import uuid
from pathlib import Path

import pytest

from app.stages.translate_stage import run_translate


def _seed_transcript(job_id: str) -> None:
    d = Path("data/jobs") / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.json").write_text(json.dumps([
        {"index": 0, "start_ms": 0, "end_ms": 1000, "text": "hello world", "language": "en"},
        {"index": 1, "start_ms": 1000, "end_ms": 2000, "text": "second line", "language": "en"},
    ]), encoding="utf-8")


def test_mock_translate_writes_translated_json(tmp_path: Path, monkeypatch) -> None:
    job_id = uuid.uuid4().hex
    monkeypatch.chdir(tmp_path)
    _seed_transcript(job_id)

    result = run_translate(job_id)

    assert result["segments"] == 2
    out = tmp_path / "data" / "jobs" / job_id / "translated.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["translated_text"] == "[ta] hello world"
    assert data[1]["translated_text"] == "[ta] second line"


def test_missing_transcript_raises_permanent(tmp_path: Path, monkeypatch) -> None:
    from app.task_base import PermanentStageError

    monkeypatch.chdir(tmp_path)
    with pytest.raises(PermanentStageError):
        run_translate("nonexistentjob")
