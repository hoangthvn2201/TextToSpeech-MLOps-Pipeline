from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from tts_pipeline.data.feature_extractor import extract_mel
from tts_pipeline.data.validator import validate_dataset, _compute_grade


def test_compute_grade_a() -> None:
    assert _compute_grade(0.01, 0.96, 36) == "A"


def test_compute_grade_c() -> None:
    assert _compute_grade(0.08, 0.82, 26) == "C"


def test_compute_grade_f() -> None:
    assert _compute_grade(0.50, 0.50, 10) == "F"


def test_validate_rejects_short_duration(
    synthetic_audio_dir: Path, tmp_path: Path
) -> None:
    audio_norm_dir = tmp_path / "audio_normalized"
    audio_norm_dir.mkdir()
    mel_dir = tmp_path / "mels"
    mel_dir.mkdir()

    # Copy 3 files as normalized audio and mels
    rows = []
    for i in range(3):
        uid = f"{i:04d}"
        src = synthetic_audio_dir / f"{uid}.wav"
        dst = audio_norm_dir / f"{uid}.wav"
        data, sr = sf.read(str(src))
        sf.write(str(dst), data, sr)
        mel = extract_mel(src)
        np.save(mel_dir / f"{uid}.npy", mel)
        duration = 0.05 if i == 0 else 2.0  # first is too short
        rows.append({"id": uid, "audio_path": str(src), "text": f"t {i}", "text_clean": f"t {i}",
                     "phonemes": "a b c", "duration": duration, "sha256": f"h{i}", "split": "train"})

    df = pd.DataFrame(rows)
    accepted, rejected, scorecard = validate_dataset(
        df, audio_norm_dir, mel_dir, min_duration=0.5, max_duration=15.0,
        min_snr_db=0.0, grade_threshold="F", output_dir=tmp_path / "validation"
    )
    # First utterance should be rejected for short duration
    assert len(rejected) >= 1
    assert rejected.iloc[0]["id"] == "0000"
    assert len(accepted) == 2
