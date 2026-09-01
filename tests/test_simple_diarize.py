"""Tests for simple_diarize — zero-dep diarization + gender classification."""

import tempfile
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write as wav_write

from app.engines.simple_diarize import (
    classify_gender_from_pitch,
    load_diarization,
    save_diarization,
    simple_diarize,
)


def test_simple_diarize_returns_segments():
    """simple_diarize returns list of segments with start/end/speaker."""
    # Create a simple test WAV: 2 seconds of silence + 2 seconds of tone
    sr = 16000
    duration_s = 4
    t = np.linspace(0, duration_s, sr * duration_s, endpoint=False)
    # First half: silence, second half: 440Hz tone
    audio = np.zeros_like(t, dtype=np.float32)
    audio[sr * 2:] = 0.5 * np.sin(2 * np.pi * 440 * t[sr * 2:])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_write(f.name, sr, (audio * 32767).astype(np.int16))
        wav_path = f.name

    try:
        segments = simple_diarize(wav_path, num_speakers=2)
        assert len(segments) >= 1
        for seg in segments:
            assert "start_ms" in seg
            assert "end_ms" in seg
            assert "speaker" in seg
            assert "energy" in seg
            assert seg["start_ms"] < seg["end_ms"]
            assert seg["speaker"].startswith("SPEAKER_")
    finally:
        Path(wav_path).unlink()


def test_simple_diarize_empty_audio():
    """Empty audio returns single segment covering full duration."""
    sr = 16000
    audio = np.zeros(sr * 2, dtype=np.float32)  # 2 seconds silence

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_write(f.name, sr, (audio * 32767).astype(np.int16))
        wav_path = f.name

    try:
        segments = simple_diarize(wav_path, num_speakers=2)
        assert len(segments) == 1
        assert segments[0]["speaker"] == "SPEAKER_00"
        assert segments[0]["start_ms"] == 0
        assert segments[0]["end_ms"] > 0
    finally:
        Path(wav_path).unlink()


def test_classify_gender_from_pitch_adds_gender():
    """classify_gender_from_pitch adds gender field to segments."""
    sr = 16000
    duration_s = 3
    t = np.linspace(0, duration_s, sr * duration_s, endpoint=False)
    # Male-like pitch (~120 Hz)
    audio = 0.5 * np.sin(2 * np.pi * 120 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_write(f.name, sr, (audio * 32767).astype(np.int16))
        wav_path = f.name

    try:
        segments = [{"start_ms": 0, "end_ms": 3000, "speaker": "SPEAKER_00", "energy": 0.5}]
        result = classify_gender_from_pitch(wav_path, segments)
        assert len(result) == 1
        assert "gender" in result[0]
        assert result[0]["gender"] in ("male", "female", "unknown")
    finally:
        Path(wav_path).unlink()


def test_save_load_diarization_roundtrip():
    """save/load diarization preserves data."""
    segments = [
        {
            "start_ms": 0,
            "end_ms": 1000,
            "speaker": "SPEAKER_00",
            "energy": 0.5,
            "gender": "male",
        },
        {
            "start_ms": 1000,
            "end_ms": 2000,
            "speaker": "SPEAKER_01",
            "energy": 0.3,
            "gender": "female",
        },
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = f.name

    try:
        save_diarization(segments, json_path)
        loaded = load_diarization(json_path)
        assert loaded == segments
    finally:
        Path(json_path).unlink()


def test_simple_diarize_merges_short_segments():
    """Segments < 1s should be merged into neighbors."""
    # Create audio with many short energy changes
    sr = 16000
    duration_s = 5
    t = np.linspace(0, duration_s, sr * duration_s, endpoint=False)
    # Alternate silence/tone every 200ms (too short, should merge)
    audio = np.zeros_like(t, dtype=np.float32)
    for i in range(int(duration_s / 0.4)):
        start = int(i * 0.4 * sr)
        end = int((i * 0.4 + 0.2) * sr)
        if i % 2 == 0:
            audio[start:end] = 0.5 * np.sin(2 * np.pi * 440 * t[start:end])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_write(f.name, sr, (audio * 32767).astype(np.int16))
        wav_path = f.name

    try:
        segments = simple_diarize(wav_path, num_speakers=2)
        # Should have fewer segments due to merging
        for seg in segments:
            assert seg["end_ms"] - seg["start_ms"] >= 1000 or len(segments) == 1
    finally:
        Path(wav_path).unlink()