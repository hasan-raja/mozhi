"""QC stage tests — Step 18 feedback loop (auto-remediation + escalation).

Heavy deps (librosa, ffmpeg, edge-tts) are mocked so this runs in CI without
models/binaries. We assert the RETRY/ESCALATE behaviour, not audio math.
"""

import json
from pathlib import Path
from unittest.mock import patch

from app.stages import qc as qc_module
from app.stages.qc import QC_MAX_RETRIES, run_qc


def _fake_loudnorm_ok(in_wav, out_wav):
    """Mock ffmpeg loudnorm: write a placeholder file so os.replace succeeds."""
    Path(out_wav).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    return True


def _seed_job(tmp_path: Path, job_id: str, durations: list[dict]) -> Path:
    """Create job dir + translated.json + TTS wavs + durations.json."""
    job_dir = tmp_path / "data" / "jobs" / job_id
    tts_dir = job_dir / "tts"
    tts_dir.mkdir(parents=True)

    translated = []
    for d in durations:
        idx = d["index"]
        (tts_dir / f"seg_{idx}.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        translated.append({"index": idx, "translated_text": d.get("text", "வணக்கம்")})

    # Point durations.json audio_path at the REAL seeded wavs
    for d in durations:
        d["audio_path"] = str(tts_dir / f"seg_{d['index']}.wav")

    (job_dir / "translated.json").write_text(json.dumps(translated))
    # segments.json (original bounds) — one pair per segment
    segs = [[0, d.get("original_ms", 3000)] for d in durations]
    (job_dir / "segments.json").write_text(json.dumps(segs))
    (job_dir / "durations.json").write_text(json.dumps(durations))
    return job_dir


def _durations_spec(index: int, duration_ms: int, original_ms: int = 3000) -> dict:
    return {
        "index": index,
        "duration_ms": duration_ms,
        "original_ms": original_ms,
    }


def test_qc_all_pass_no_retries(tmp_path: Path, monkeypatch) -> None:
    """Clean segments: no remediation, no escalation."""
    job_id = "qc_pass_001"
    job_dir = _seed_job(tmp_path, job_id, [
        _durations_spec(0, 3000), _durations_spec(1, 3000),
    ])
    monkeypatch.chdir(tmp_path)
    with patch.object(qc_module, "_snr_db", return_value=30.0), \
         patch.object(qc_module, "_normalize_loudness", _fake_loudnorm_ok), \
         patch.object(qc_module, "_resynth_segment_slower", return_value=3.0), \
         patch.object(qc_module, "_load_target_lang", return_value="ta"):
        result = run_qc(job_id)

    assert result["segments"] == 2
    assert result["failed"] == 0
    assert result["escalated"] == 0
    saved = json.loads((job_dir / "durations.json").read_text())
    assert all(s["qc"]["retries"] == 0 and s["qc"]["passed"] for s in saved)


def test_qc_snr_remediated_within_retries(tmp_path: Path, monkeypatch) -> None:
    """Low-SNR segment is renormalized and passes after one retry."""
    job_id = "qc_snr_001"
    specs = [_durations_spec(0, 3000)]
    job_dir = _seed_job(tmp_path, job_id, specs)
    monkeypatch.chdir(tmp_path)

    snr_calls = {"n": 0}
    def fake_snr(path):  # first call low, after renorm high
        snr_calls["n"] += 1
        return 5.0 if snr_calls["n"] == 1 else 30.0

    with patch.object(qc_module, "_snr_db", side_effect=fake_snr), \
         patch.object(qc_module, "_normalize_loudness", _fake_loudnorm_ok), \
         patch.object(qc_module, "_resynth_segment_slower", return_value=3.0), \
         patch.object(qc_module, "_load_target_lang", return_value="ta"):
        result = run_qc(job_id)

    assert result["failed"] == 0
    assert result["escalated"] == 0
    saved = json.loads((job_dir / "durations.json").read_text())
    assert saved[0]["qc"]["retries"] >= 1
    assert saved[0]["qc"]["passed"] is True


def test_qc_escalates_after_max_retries(tmp_path: Path, monkeypatch) -> None:
    """Segment that stays failing after QC_MAX_RETRIES is escalated."""
    job_id = "qc_escalate_001"
    specs = [_durations_spec(0, 3000)]
    _seed_job(tmp_path, job_id, specs)
    monkeypatch.chdir(tmp_path)

    # SNR always low, renorm never helps
    with patch.object(qc_module, "_snr_db", return_value=5.0), \
         patch.object(qc_module, "_normalize_loudness", _fake_loudnorm_ok), \
         patch.object(qc_module, "_resynth_segment_slower", return_value=3.0), \
         patch.object(qc_module, "_load_target_lang", return_value="ta"):
        result = run_qc(job_id)

    assert result["failed"] == 1
    assert result["escalated"] == 1
    assert result["renormalized"] == QC_MAX_RETRIES  # renorm attempted each retry


def test_qc_missing_wav_escalates_immediately(tmp_path: Path, monkeypatch) -> None:
    """A segment with no wav file is escalated without retry."""
    job_id = "qc_missing_001"
    specs = [_durations_spec(0, 3000)]
    job_dir = _seed_job(tmp_path, job_id, specs)
    # Remove the wav so it's missing
    (job_dir / "tts" / "seg_0.wav").unlink()
    monkeypatch.chdir(tmp_path)

    with patch.object(qc_module, "_snr_db", return_value=30.0), \
         patch.object(qc_module, "_normalize_loudness", _fake_loudnorm_ok), \
         patch.object(qc_module, "_resynth_segment_slower", return_value=3.0), \
         patch.object(qc_module, "_load_target_lang", return_value="ta"):
        result = run_qc(job_id)

    assert result["failed"] == 1
    assert result["escalated"] == 1
    saved = json.loads((job_dir / "durations.json").read_text())
    assert saved[0]["qc"]["retries"] == 0
    assert saved[0]["qc"]["reason"] == "missing_or_empty_wav"
