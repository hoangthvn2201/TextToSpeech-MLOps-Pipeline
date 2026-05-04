from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_pipeline.data.audio_processor import normalize_audio


def test_normalize_audio_resamples(synthetic_audio_dir: Path, tmp_path: Path) -> None:
    src = synthetic_audio_dir / "0000.wav"
    dst = tmp_path / "normalized.wav"
    normalize_audio(src, dst, target_sr=16000, peak_normalize_db=-3.0, trim_silence=False)
    data, sr = sf.read(str(dst))
    assert sr == 16000
    assert len(data) > 0


def test_normalize_audio_peak(synthetic_audio_dir: Path, tmp_path: Path) -> None:
    src = synthetic_audio_dir / "0001.wav"
    dst = tmp_path / "peak.wav"
    normalize_audio(src, dst, target_sr=22050, peak_normalize_db=-3.0, trim_silence=False)
    data, _ = sf.read(str(dst))
    peak_db = 20 * np.log10(np.abs(data).max() + 1e-10)
    assert abs(peak_db - (-3.0)) < 1.0, f"Peak was {peak_db:.2f} dB, expected ~-3 dB"


def test_normalize_audio_mono(synthetic_audio_dir: Path, tmp_path: Path) -> None:
    src = synthetic_audio_dir / "0002.wav"
    dst = tmp_path / "mono.wav"
    normalize_audio(src, dst, target_sr=22050, trim_silence=False)
    info = sf.info(str(dst))
    assert info.channels == 1
