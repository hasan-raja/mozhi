"""Tests for media utilities — ffmpeg extraction (skips if no ffmpeg) and
VAD segmentation logic with a synthetic tone file."""

import asyncio
import shutil
import struct
from pathlib import Path

import pytest

from app.media import FFmpegError, detect_segments, extract_audio

ffmpeg_available = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not on PATH")


def _write_tone_wav(path: Path, segments: list[tuple[int, int]], sr: int = 16000,
                    total_s: float = 4.0) -> None:
    """Write a WAV: silence with tone bursts at the given (start_s, end_s)."""
    import wave

    total = int(sr * total_s)
    samples = [0.0] * total
    for start_s, end_s in segments:
        for i in range(int(start_s * sr), min(int(end_s * sr), total)):
            samples[i] = 0.5 * ((i % 100) / 100 - 0.5)  # rough audible buzz
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(struct.pack("<h", int(s * 32767)) for s in samples)
        w.writeframes(frames)


def test_extract_audio_rejects_missing_input(tmp_path: Path) -> None:
    async def run():
        await extract_audio("nope_does_not_exist.mp4", str(tmp_path / "out.wav"))

    with pytest.raises(FFmpegError):
        asyncio.run(run())


def test_extract_audio_converts_stereo_to_mono_16k(tmp_path: Path) -> None:
    import soundfile as sf

    src = tmp_path / "in.wav"
    # 2s stereo 44.1k source
    sf.write(src, [[0.0, 0.0]] * 44100, 44100, subtype="PCM_16")
    out = tmp_path / "out" / "audio.wav"

    asyncio.run(extract_audio(str(src), str(out)))

    data, sr = sf.read(out)
    assert sr == 16000
    assert data.ndim == 1  # mono


@pytest.mark.skip(reason="requires silero-vad + torch installed; run manually")
def test_detect_segments_finds_tone_bursts(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    _write_tone_wav(wav, [(0.5, 1.5), (2.5, 3.5)])
    segs = asyncio.run(detect_segments(str(wav)))
    assert len(segs) >= 2
    assert segs[0][0] < 1500  # first burst starts ~500ms
