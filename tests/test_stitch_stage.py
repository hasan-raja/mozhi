"""Stitch stage tests — mocked ffmpeg, real concatenation logic."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.media import FFmpegError
from app.stages.stitch_stage import run_stitch


def _seed_job(tmp_path: Path, job_id: str, segments: int = 3) -> None:
    """Create minimal TTS artifacts so stitch can run."""
    job_dir = tmp_path / "data" / "jobs" / job_id
    tts_dir = job_dir / "tts"
    tts_dir.mkdir(parents=True)

    # Fake wav files (headers don't matter — ffmpeg is mocked)
    for i in range(segments):
        (tts_dir / f"seg_{i}.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    durations = []
    for i in range(segments):
        durations.append({
            "index": i,
            "audio_path": str(tts_dir / f"seg_{i}.wav"),
            "duration_ms": 3000,
            "original_ms": 3000,
        })
    (job_dir / "durations.json").write_text(json.dumps(durations))

    # Fake source video
    (job_dir / "source").mkdir(exist_ok=True)
    (job_dir / "source" / "input.mp4").write_bytes(b"fake video")


def test_stitch_success(tmp_path: Path, monkeypatch) -> None:
    """Happy path: ffmpeg is mocked, final.mp4 is created."""
    job_id = "testjob_stitch_001"
    monkeypatch.chdir(tmp_path)
    _seed_job(tmp_path, job_id, segments=3)

    # Mock ffmpeg calls so we don't spawn real subprocesses
    with patch("app.media.ffmpeg_concat_wavs") as mock_concat, \
         patch("app.media.ffmpeg_mux_audio_on_video") as mock_mux:
        # Simulate ffmpeg writing the output file
        def fake_mux(video, audio, output):
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_bytes(b"final mp4 content")

        mock_mux.side_effect = fake_mux

        result = run_stitch(job_id)

    assert result["stage"] == "stitch"
    assert "final.mp4" in result["output"]
    assert result["segments"] == 3
    assert Path(result["output"]).exists()
    mock_concat.assert_called_once()
    mock_mux.assert_called_once()


def test_stitch_skips_when_already_exists(tmp_path: Path, monkeypatch) -> None:
    """Idempotent: existing final.mp4 means we return without calling ffmpeg."""
    job_id = "testjob_stitch_002"
    monkeypatch.chdir(tmp_path)
    _seed_job(tmp_path, job_id)

    final_path = tmp_path / "data" / "jobs" / job_id / "final.mp4"
    final_path.write_bytes(b"existing")

    with patch("app.media.ffmpeg_concat_wavs") as mock_concat, \
         patch("app.media.ffmpeg_mux_audio_on_video") as mock_mux:
        result = run_stitch(job_id)

    assert result["stage"] == "stitch"
    assert "final.mp4" in result["output"]
    mock_concat.assert_not_called()
    mock_mux.assert_not_called()


def test_stitch_no_tts_files_raises(tmp_path: Path, monkeypatch) -> None:
    """No seg_*.wav → PermanentStageError."""
    job_id = "testjob_stitch_003"
    monkeypatch.chdir(tmp_path)
    _seed_job(tmp_path, job_id, segments=0)

    from app.task_base import PermanentStageError
    with pytest.raises(PermanentStageError, match="no TTS wav"):
        run_stitch(job_id)
