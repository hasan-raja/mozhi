"""Simple speaker diarization — zero external deps, uses librosa (already in deps).

Energy-based clustering: splits audio into uniform windows, computes RMS energy
per window, uses k-means (k=2) on energy to label speech/non-speech regions.
Then groups contiguous speech windows into speaker segments.

NOT a true speaker diarization — for production use pyannote.audio (optional extra).
This is a fast CPU fallback that works without HF_TOKEN or GPU.
"""

import json
import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.cluster.vq import kmeans2  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

SegmentDict = dict[str, Any]


def simple_diarize(
    audio_path: str | Path, num_speakers: int = 2, window_ms: int = 500
) -> list[SegmentDict]:
    """Diarize audio into speaker-labeled segments.

    Args:
        audio_path: Path to WAV file (16kHz mono preferred)
        num_speakers: Number of speakers to cluster (default 2)
        window_ms: Analysis window size in ms (default 500ms)

    Returns:
        List of segments: {"start_ms": int, "end_ms": int, "speaker": str, "energy": float}
    """
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    if len(y) == 0:
        return []

    window_samples = int(sr * window_ms / 1000)
    hop_samples = window_samples // 2  # 50% overlap

    # Compute RMS energy per window
    energies: list[float] = []
    times: list[float] = []
    for i in range(0, len(y) - window_samples + 1, hop_samples):
        window = y[i:i + window_samples]
        rms = np.sqrt(np.mean(window**2) + 1e-10)
        energies.append(rms)
        times.append(i / sr * 1000)  # ms

    energies_arr = np.array(energies)
    if len(energies_arr) < num_speakers:
        return [{
            "start_ms": 0,
            "end_ms": int(len(y) / sr * 1000),
            "speaker": "SPEAKER_00",
            "energy": float(np.mean(energies_arr)),
        }]

    # K-means on energy (log scale for better separation)
    log_energies = np.log(energies_arr + 1e-10).reshape(-1, 1)
    centroids, labels = kmeans2(log_energies, num_speakers, minit="points")

    # Label 0 = lower energy (quieter speaker), 1 = higher energy
    # Map to SPEAKER_00, SPEAKER_01...
    speaker_labels = [f"SPEAKER_{i:02d}" for i in labels]

    # Group contiguous windows with same speaker
    segments: list[SegmentDict] = []
    current_speaker = speaker_labels[0]
    seg_start_idx = 0

    for i in range(1, len(speaker_labels)):
        if speaker_labels[i] != current_speaker:
            segments.append({
                "start_ms": int(times[seg_start_idx]),
                "end_ms": int(times[i]),
                "speaker": current_speaker,
                "energy": float(np.mean(energies_arr[seg_start_idx:i])),
            })
            current_speaker = speaker_labels[i]
            seg_start_idx = i

    # Final segment
    segments.append({
        "start_ms": int(times[seg_start_idx]),
        "end_ms": int(len(y) / sr * 1000),
        "speaker": current_speaker,
        "energy": float(np.mean(energies_arr[seg_start_idx:])),
    })

    # Merge very short segments (< 1s) into neighbors
    merged: list[SegmentDict] = []
    for seg in segments:
        if merged and (seg["end_ms"] - seg["start_ms"] < 1000):
            merged[-1]["end_ms"] = seg["end_ms"]
        else:
            merged.append(seg)

    logger.info("simple_diarize: %d segments from %s", len(merged), audio_path)
    return merged


def classify_gender_from_pitch(
    audio_path: str | Path, segments: list[SegmentDict]
) -> list[SegmentDict]:
    """Heuristic gender classification per segment using pitch (f0).

    Args:
        audio_path: Path to WAV file
        segments: List of segments from simple_diarize()

    Returns:
        Segments with added "gender" key: "male" | "female" | "unknown"
    """
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    if len(y) == 0:
        for seg in segments:
            seg["gender"] = "unknown"
        return segments

    # Extract pitch using librosa.pyin (robust for speech)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C7")), sr=sr
    )

    if f0 is None:
        for seg in segments:
            seg["gender"] = "unknown"
        return segments

    frame_duration_ms = 512 / sr * 1000  # default hop_length=512

    for seg in segments:
        start_frame = int(seg["start_ms"] / frame_duration_ms)
        end_frame = int(seg["end_ms"] / frame_duration_ms)
        end_frame = min(end_frame, len(f0))

        if start_frame >= end_frame or start_frame >= len(f0):
            seg["gender"] = "unknown"
            continue

        seg_f0 = f0[start_frame:end_frame]
        seg_voiced = voiced_flag[start_frame:end_frame]

        # Median pitch of voiced frames
        voiced_f0 = seg_f0[seg_voiced]
        if len(voiced_f0) == 0:
            seg["gender"] = "unknown"
            continue

        median_pitch = np.median(voiced_f0)

        # Heuristic thresholds (Hz) — typical adult ranges
        # Male: ~85-180 Hz, Female: ~165-255 Hz
        if median_pitch < 155:
            seg["gender"] = "male"
        elif median_pitch > 180:
            seg["gender"] = "female"
        else:
            seg["gender"] = "unknown"  # overlap zone

    logger.info("classify_gender: %s", [s.get("gender") for s in segments])
    return segments


def save_diarization(segments: list[SegmentDict], out_path: str | Path) -> None:
    """Save diarization result to JSON."""
    Path(out_path).write_text(json.dumps(segments, indent=2))


def load_diarization(in_path: str | Path) -> list[SegmentDict]:
    """Load diarization result from JSON."""
    data = json.loads(Path(in_path).read_text())
    return list(data)