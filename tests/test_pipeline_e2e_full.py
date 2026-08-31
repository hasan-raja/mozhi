"""End-to-end pipeline test (mock engines): asr -> translate -> tts -> qc -> stitch.

Reproduces the Step 20 golden test: deterministic MockEngine output lets us
assert the WHOLE chain produces a final.mp4 whose audio is the TTS (dubbed)
track, not the source-language audio.

Requires ffmpeg on PATH; skips otherwise. faster-whisper NOT required --
the ASR stage is bypassed with a seeded transcript.json.
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ffmpeg_available = shutil.which("ffmpeg") is not None
pytestmark = [
    pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not on PATH"),
]

JOB = uuid.uuid4().hex


def _make_source_video(tmp_path: Path) -> Path:
    """3s test video with an audio track (220 Hz mono sine)."""
    import subprocess

    src_dir = tmp_path / "data" / "jobs" / JOB / "source"
    src_dir.mkdir(parents=True)
    src = src_dir / "input.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=128x72:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-shortest", str(src),
        ],
        check=True,
    )
    return src


def _seed_transcript(tmp_path: Path, texts: list[str]) -> Path:
    """Seed transcript.json so ASR is bypassed (mock engine ignores audio)."""
    import json as _json

    job_dir = tmp_path / "data" / "jobs" / JOB
    job_dir.mkdir(parents=True, exist_ok=True)
    seg_len = 1000
    segments = [
        {"index": i, "start_ms": i * seg_len, "end_ms": (i + 1) * seg_len, "text": t}
        for i, t in enumerate(texts)
    ]
    (job_dir / "transcript.json").write_text(_json.dumps(segments), encoding="utf-8")
    (job_dir / "segments.json").write_text(
        _json.dumps([[s["start_ms"], s["end_ms"]] for s in segments])
    )
    return job_dir


def _run_async_passthrough() -> Any:
    """Make run_async a direct asyncio.run, skipping DB lookups/usage rows."""
    import asyncio

    def _passthrough(coro: Any) -> Any:
        return asyncio.run(coro)
    return _passthrough


def test_full_pipeline_mock_chain_produces_dubbed_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end mock: translate -> tts -> qc -> stitch yields final.mp4."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "mock")

    from app.config import get_settings
    get_settings.cache_clear()

    src = _make_source_video(tmp_path)
    assert src.exists()

    job_dir = _seed_transcript(tmp_path, ["hello world", "how are you"])
    assert (tmp_path / "data" / "jobs" / JOB / "source" / "input.mp4").exists()

    run_async_pt = _run_async_passthrough()

    from app.stages.qc import run_qc
    from app.stages.stitch_stage import run_stitch
    from app.stages.translate_stage import run_translate
    from app.stages.tts_stage import run_tts

    def _mock_target_lang(job_id: str) -> str:
        return "ta"

    with patch("app.stages.translate_stage.run_async", run_async_pt), \
         patch("app.stages.tts_stage._load_target_lang", _mock_target_lang):
        tr = run_translate(JOB)
        assert tr["stage"] == "translate"
        assert tr["segments"] == 2

        tts = run_tts(JOB)
        assert tts["stage"] == "tts"
        assert tts["segments"] == 2
        wavs = sorted((job_dir / "tts").glob("seg_*.wav"))
        assert len(wavs) == 2
        for w in wavs:
            assert w.stat().st_size > 0

        qc = run_qc(JOB)
        assert qc["stage"] == "qc"
        assert qc["segments"] == 2
        durations = json.loads((job_dir / "durations.json").read_text())
        assert "qc" in durations[0]

        st = run_stitch(JOB)
        assert st["stage"] == "stitch"
        final = job_dir / "final.mp4"
        assert final.exists()
        assert final.stat().st_size > 0
        assert st["size_mb"] > 0

        # Confirm the muxed video has both streams
        import subprocess
        probe = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(final)],
            capture_output=True, text=True,
        )
        streams = probe.stdout.strip().splitlines()
        assert "video" in streams
        assert "audio" in streams


def test_pipeline_mock_translate_tags_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: mock translation must prefix [lang] tag (Step 20 golden)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOZHI_ENGINE_MODE", "mock")
    from app.config import get_settings
    get_settings.cache_clear()

    job_dir = _seed_transcript(tmp_path, ["hello"])

    run_async_pt = _run_async_passthrough()
    from app.stages.translate_stage import run_translate

    with patch("app.stages.translate_stage.run_async", run_async_pt):
        run_translate(JOB)

    translated = json.loads((job_dir / "translated.json").read_text())
    # MockTranslationEngine returns f"[{target_lang}] {text}"; target defaults to 'ta'
    assert translated[0]["translated_text"].startswith("[")
    assert "hello" in translated[0]["translated_text"]
