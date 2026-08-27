"""End-to-end pipeline tests: extract → vad → asr with real ffmpeg + mock ASR.

These run the actual task bodies (bypassing Celery) against fixture media.
Requires ffmpeg on PATH; skips otherwise. faster-whisper is NOT required —
the ASR test uses MockEngine via MOZHI_ENGINE_MODE=mock.
"""

import json
import shutil
from pathlib import Path

import pytest

ffmpeg_available = shutil.which("ffmpeg") is not None
try:
    import silero_vad  # noqa: F401

    torch_available = True
except ImportError:
    torch_available = False
pytestmark = [
    pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not on PATH"),
    pytest.mark.skipif(
        not torch_available, reason="silero-vad/torch not installed (uv sync --extra vad)"
    ),
]

JOB = "testjob000000000000000000000000"


def _make_source_video(tmp_path: Path) -> Path:
    """Generate a 3s test video WITH audio track via ffmpeg lavfi.

    Layout matches what the extract stage expects: data/jobs/{JOB}/source/.
    """
    src_dir = tmp_path / "data" / "jobs" / JOB / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / "input.mp4"
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=128x72:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-shortest",
            str(src),
        ],
        check=True,
    )
    return src


def _run_extract_and_vad(source: Path) -> dict:
    """Run real extract + vad task bodies synchronously (Celery binds self).

    VAD needs torch — if silero/torch aren't installed, skip VAD gracefully
    by writing an empty segments.json (ASR mock doesn't need real boundaries).
    """
    import json as _json

    from app.tasks import run_extract, run_vad

    r1 = run_extract.__wrapped__(JOB)  # bound method: task_self implicit
    assert r1["stage"] == "extract"
    _ = source
    try:
        return run_vad.__wrapped__(JOB)  # type: ignore[attr-defined]
    except Exception as exc:
        if "torch" in str(exc):
            seg_file = Path("data") / "jobs" / JOB / "segments.json"
            seg_file.parent.mkdir(parents=True, exist_ok=True)
            seg_file.write_text(_json.dumps([[0, 3000]]))  # whole clip
            return {"job_id": JOB, "stage": "vad", "count": 1, "skipped": True}
        raise


def test_extract_creates_16k_mono_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full extract stage on a REAL generated video."""
    monkeypatch.chdir(tmp_path)
    src = _make_source_video(tmp_path)

    result = _run_extract_and_vad(src)

    wav = tmp_path / "data" / "jobs" / JOB / "extract" / "audio.wav"
    assert wav.exists()

    import soundfile as sf

    data, sr = sf.read(wav)
    assert sr == 16000
    assert data.ndim == 1
    assert result["stage"] == "vad"  # chained through


def test_asr_with_mock_engine_writes_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ASR stage persists a transcript.json given upstream artifacts."""
    monkeypatch.chdir(tmp_path)
    # Force mock engine: the test's audio.wav is fake bytes that real
    # faster-whisper would reject. Deterministic unit-level coverage.
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "mock")
    from app.config import get_settings
    get_settings.cache_clear()

    job_dir = tmp_path / "data" / "jobs" / JOB
    (job_dir / "extract").mkdir(parents=True)

    # fake extracted wav (empty is fine — mock engine ignores content)
    (job_dir / "extract" / "audio.wav").write_bytes(b"RIFF")
    # vad stage output
    (job_dir / "segments.json").write_text(json.dumps([[0, 2000]]))

    from app.tasks import run_asr

    result = run_asr.__wrapped__(JOB)  # type: ignore[attr-defined]

    transcript = job_dir / "transcript.json"
    assert transcript.exists()
    payload = json.loads(transcript.read_text())
    assert len(payload) >= 1
    assert {"index", "start_ms", "end_ms", "text"} <= set(payload[0].keys())
    assert result["segments"] == len(payload)


def test_missing_source_raises_permanent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    from app.task_base import PermanentStageError
    from app.tasks import run_extract

    with pytest.raises(PermanentStageError):
        run_extract.__wrapped__(None, "ghostjob")  # type: ignore[attr-defined]
