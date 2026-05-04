"""Shared test fixtures.

The synthetic_dataset fixture creates 50 short utterances in a temp directory
using pure sine waves + deterministic text. No real audio data is required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from tts_pipeline.text.tokenizer import Tokenizer

N_SAMPLES = 50
SR = 22050
DURATION_SEC = 2.0
PHONEMES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


@pytest.fixture(scope="session")
def synthetic_audio_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """50 synthetic 2-second WAV files at 22050 Hz."""
    audio_dir = tmp_path_factory.mktemp("audio")
    n_samples = int(SR * DURATION_SEC)
    for i in range(N_SAMPLES):
        freq = 220 + i * 10
        t = np.linspace(0, DURATION_SEC, n_samples, endpoint=False)
        wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(str(audio_dir / f"{i:04d}.wav"), wave, SR)
    return audio_dir


@pytest.fixture(scope="session")
def synthetic_transcript_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """50 simple text transcripts matching the audio files."""
    txt_dir = tmp_path_factory.mktemp("transcripts")
    for i in range(N_SAMPLES):
        (txt_dir / f"{i:04d}.txt").write_text(f"test utterance {i}")
    return txt_dir


@pytest.fixture(scope="session")
def synthetic_dataset(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_audio_dir: Path,
    synthetic_transcript_dir: Path,
) -> Path:
    """Full synthetic dataset root: audio/, transcripts/, and metadata.csv."""
    root = tmp_path_factory.mktemp("dataset")
    (root / "audio").symlink_to(synthetic_audio_dir)
    (root / "transcripts").symlink_to(synthetic_transcript_dir)

    # Build metadata.csv
    rows = []
    for i in range(N_SAMPLES):
        uid = f"{i:04d}"
        rows.append({
            "id": uid,
            "audio_path": str(synthetic_audio_dir / f"{uid}.wav"),
            "text": f"test utterance {i}",
            "text_clean": f"test utterance {i}",
            "phonemes": " ".join(PHONEMES[:5]),
            "duration": DURATION_SEC,
            "sha256": f"fake_hash_{i}",
            "split": "train" if i < 40 else ("val" if i < 45 else "test"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(root / "metadata.csv", index=False)
    return root


@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_inventory(PHONEMES, add_blank=True)
