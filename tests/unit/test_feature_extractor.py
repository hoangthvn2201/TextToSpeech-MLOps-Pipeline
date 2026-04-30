from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tts_pipeline.data.feature_extractor import compute_stats, extract_mel


def test_mel_shape(synthetic_audio_dir: Path) -> None:
    src = synthetic_audio_dir / "0000.wav"
    mel = extract_mel(src, sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80)
    assert mel.ndim == 2
    assert mel.shape[0] == 80


def test_mel_deterministic(synthetic_audio_dir: Path) -> None:
    src = synthetic_audio_dir / "0001.wav"
    mel1 = extract_mel(src, sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80)
    mel2 = extract_mel(src, sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80)
    np.testing.assert_array_equal(mel1, mel2)


def test_compute_stats(synthetic_audio_dir: Path, tmp_path: Path) -> None:
    mels = {
        f"{i:04d}": extract_mel(
            synthetic_audio_dir / f"{i:04d}.wav",
            sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80,
        )
        for i in range(5)
    }
    stats_path = tmp_path / "stats.json"
    stats = compute_stats(mels, out_path=stats_path)

    assert stats_path.exists()
    loaded = json.loads(stats_path.read_text())
    assert "mel_mean" in loaded
    assert "mel_std" in loaded
    assert loaded["mel_std"] > 0
