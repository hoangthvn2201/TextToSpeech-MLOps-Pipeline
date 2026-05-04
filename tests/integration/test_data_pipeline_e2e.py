"""E2E test: run all data pipeline stages on the synthetic dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from tts_pipeline.data.audio_processor import normalize_audio
from tts_pipeline.data.feature_extractor import compute_stats, extract_features_batch
from tts_pipeline.data.ingestion import build_metadata
from tts_pipeline.data.transcript_cleaner import clean_metadata
from tts_pipeline.data.validator import validate_dataset


@pytest.mark.slow
def test_data_pipeline_e2e(
    synthetic_audio_dir: Path,
    synthetic_transcript_dir: Path,
    tmp_path: Path,
) -> None:
    # ── Stage 1: Ingest ───────────────────────────────────────────────────────
    df = build_metadata(synthetic_audio_dir, synthetic_transcript_dir, val_ratio=0.1, test_ratio=0.1)
    assert len(df) == 50
    assert set(df["split"].unique()) <= {"train", "val", "test"}

    # ── Stage 2: Normalize audio ──────────────────────────────────────────────
    audio_norm_dir = tmp_path / "audio_normalized"
    audio_norm_dir.mkdir()
    for _, row in df.head(10).iterrows():
        src = Path(row["audio_path"])
        dst = audio_norm_dir / f"{row['id']}.wav"
        normalize_audio(src, dst, target_sr=22050, trim_silence=False)
    assert len(list(audio_norm_dir.glob("*.wav"))) == 10

    # ── Stage 3: Clean transcripts ────────────────────────────────────────────
    accepted, rejected = clean_metadata(df.head(10), language="vi")
    assert len(accepted) + len(rejected) == 10

    # ── Stage 4: Phonemize (stub — use raw text as phonemes) ─────────────────
    accepted["phonemes"] = accepted["text"].apply(lambda t: " ".join(t.split()[:5]))

    # ── Stage 5: Extract features ─────────────────────────────────────────────
    audio_paths = [audio_norm_dir / f"{row['id']}.wav" for _, row in accepted.iterrows()]
    audio_paths = [p for p in audio_paths if p.exists()]

    mel_dir = tmp_path / "mels"
    f0_dir = tmp_path / "f0"
    feature_cfg = {"sample_rate": 22050, "n_fft": 512, "hop_length": 128,
                   "win_length": 512, "n_mels": 40, "f_min": 0.0, "f_max": 8000.0}
    mels = extract_features_batch(audio_paths, mel_dir, f0_dir, feature_cfg, n_workers=1)
    assert len(mels) == len(audio_paths)

    stats = compute_stats(mels, out_path=tmp_path / "stats.json")
    assert "mel_mean" in stats

    # ── Stage 6: Validate ─────────────────────────────────────────────────────
    accepted_v, rejected_v, scorecard = validate_dataset(
        df=accepted,
        audio_normalized_dir=audio_norm_dir,
        mel_dir=mel_dir,
        min_duration=0.1,
        max_duration=20.0,
        min_snr_db=0.0,
        grade_threshold="F",
        output_dir=tmp_path / "validation",
    )
    assert (tmp_path / "validation" / "dataset_scorecard.json").exists()
    assert (tmp_path / "validation" / "rejected_samples.csv").exists()
    assert scorecard["grade"] in ("A", "B", "C", "D", "F")
    assert scorecard["n_accepted"] + scorecard["n_rejected"] == len(accepted)
