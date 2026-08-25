"""Media utilities — ffmpeg audio extraction + silero VAD segmentation.

Stage 1 (extract): pull 16k mono WAV from any input video/audio via ffmpeg.
Stage 2 (vad): silero VAD finds speech regions → segment boundaries.

All heavy work is CPU-bound and runs via run_in_executor — never blocks the
async loop. ffmpeg must be on PATH (documented in README).
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    """ffmpeg exited non-zero; carries stderr for debugging."""


def _run_ffmpeg_sync(args: list[str]) -> str:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg {' '.join(args)} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


async def extract_audio(
    input_path: str, out_wav: str, sample_rate: int = 16000
) -> str:
    """Extract mono 16-bit PCM WAV from any media file. Returns out_wav."""
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-i", input_path,
        "-vn",                      # drop video
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),    # whisper-friendly rate
        "-c:a", "pcm_s16le",
        out_wav,
    ]
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run_ffmpeg_sync, args)
    logger.info("extracted audio %s -> %s", input_path, out_wav)
    return out_wav


# ── VAD ──────────────────────────────────────────────────────────────────────

_vad_model = None
_vad_utils = None


def _get_vad() -> tuple[Any, Any]:
    global _vad_model, _vad_utils
    if _vad_model is None:
        from silero_vad import (
            get_speech_timestamps,
            load_silero_vad,
        )
        _vad_model = load_silero_vad()
        _vad_utils = get_speech_timestamps
    return _vad_model, _vad_utils


def _vad_sync(wav_path: str) -> list[dict[str, int]]:
    import soundfile as sf  # type: ignore[import-untyped]
    import torch

    model, get_ts = _get_vad()
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:  # stereo → mono
        wav = wav.mean(axis=1)
    speech: list[dict[str, int]] = get_ts(
        torch.from_numpy(wav), model, sampling_rate=sr,
        min_speech_duration_ms=250,   # ignore blips
        min_silence_duration_ms=300,  # merge close segments
        speech_pad_ms=100,            # breathing room around words
    )
    return speech  # [{"start": samples, "end": samples}, ...]


async def detect_segments(wav_path: str, sample_rate: int = 16000) -> list[tuple[int, int]]:
    """Return [(start_ms, end_ms), ...] for each detected speech region."""
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(None, _vad_sync, wav_path)
    segments = [
        (int(chunk["start"] * 1000 / sample_rate),
         int(chunk["end"] * 1000 / sample_rate))
        for chunk in raw
    ]
    logger.info("VAD: %d speech segments in %s", len(segments), wav_path)
    return segments
