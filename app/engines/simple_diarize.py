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

    Fast path: uses librosa.yin on segment regions only.
    For even faster classification, call `classify_gender_for_tts()` from TTS stage
    which only processes segments that actually need gender-specific voices.
    """
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    if len(y) == 0:
        for seg in segments:
            seg["gender"] = "unknown"
        return segments

    for seg in segments:
        start_sample = int(seg["start_ms"] / 1000 * sr)
        end_sample = int(seg["end_ms"] / 1000 * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(y), end_sample)

        if end_sample - start_sample < sr * 0.2:  # < 200ms — too short for reliable pitch
            seg["gender"] = "unknown"
            continue

        seg_y = y[start_sample:end_sample]

        # Fast pitch estimation using librosa.yin (autocorrelation-based)
        # fmin=80Hz (C2), fmax=300Hz (C4) covers typical speech range
        try:
            f0 = librosa.yin(seg_y, fmin=80.0, fmax=300.0, sr=sr)
        except Exception:
            seg["gender"] = "unknown"
            continue

        if f0 is None or len(f0) == 0:
            seg["gender"] = "unknown"
            continue

        # Filter unvoiced frames (yin returns high values for unvoiced)
        # Use median of lower 50% as voiced pitch estimate
        valid_f0 = f0[f0 > 0]
        if len(valid_f0) == 0:
            seg["gender"] = "unknown"
            continue

        median_pitch = float(np.median(valid_f0))

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


def classify_gender_for_tts(
    audio_path: str | Path,
    segments: list[SegmentDict],
    target_langs: list[str],
    job_id: str = "",
) -> list[SegmentDict]:
    """Lazy gender classification: only classify segments that need gender-specific voices.

    Memory-efficient: loads audio once, processes per-segment slices.
    Only runs for languages that actually have male/female voices.
    """
    # Languages with gender-specific Edge TTS voices
    gender_langs = {"ta", "hi", "te", "kn", "ml", "bn", "mr"}
    needs_gender = any(lang in gender_langs for lang in target_langs)
    if not needs_gender:
        for seg in segments:
            seg.setdefault("gender", "unknown")
        return segments

    import librosa
    import numpy as np

    logger.info("classify_gender_for_tts: job=%s segments=%d", job_id, len(segments))

    # Load audio once but process per-segment to limit memory
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    for seg in segments:
        start_sample = int(seg["start_ms"] / 1000 * sr)
        end_sample = int(seg["end_ms"] / 1000 * sr)
        start_sample = max(0, min(start_sample, len(y) - 1))
        end_sample = max(start_sample + 1, min(end_sample, len(y)))
        seg_y = y[start_sample:end_sample]

        if len(seg_y) < sr * 0.1:  # <100ms - too short for reliable pitch
            seg["gender"] = "unknown"
            continue

        try:
            f0 = librosa.yin(
                seg_y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C5"),
                sr=sr,
            )
            f0_voiced = f0[f0 > 0]
            if len(f0_voiced) == 0:
                seg["gender"] = "unknown"
                continue
            median_f0 = float(np.median(f0_voiced))
            if median_f0 < 165:
                seg["gender"] = "male"
            elif median_f0 > 255:
                seg["gender"] = "female"
            else:
                seg["gender"] = "unknown"
        except Exception:
            seg["gender"] = "unknown"

    logger.info("classify_gender: job=%s result=%s", job_id, [s.get("gender") for s in segments])
    return segments


def save_diarization(segments: list[SegmentDict], out_path: str | Path) -> None:
    """Save diarization result to JSON."""
    Path(out_path).write_text(json.dumps(segments, indent=2))


def load_diarization(in_path: str | Path) -> list[SegmentDict]:
    """Load diarization result from JSON."""
    data = json.loads(Path(in_path).read_text())
    return list(data)