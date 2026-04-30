from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tts_pipeline.utils.audio_utils import compute_snr_db
from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

_GRADE_ORDER = ["A", "B", "C", "D", "F"]

_GRADE_THRESHOLDS = {
    # (rejection_rate, phoneme_coverage_score, median_snr)
    "A": (0.02, 0.95, 35),
    "B": (0.05, 0.90, 30),
    "C": (0.10, 0.80, 25),
    "D": (0.20, 0.70, 20),
}


def _compute_grade(rejection_rate: float, coverage_score: float, median_snr: float) -> str:
    for grade, (max_rej, min_cov, min_snr) in _GRADE_THRESHOLDS.items():
        if rejection_rate <= max_rej and coverage_score >= min_cov and median_snr >= min_snr:
            return grade
    return "F"


def validate_dataset(
    df: pd.DataFrame,
    audio_normalized_dir: Path,
    mel_dir: Path,
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    min_snr_db: float = 20.0,
    grade_threshold: str = "C",
    output_dir: Path = Path("data/processed/validation"),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:  # type: ignore[type-arg]
    """Validate dataset and emit full data lineage artifacts.

    Returns (accepted_df, rejected_df, scorecard_dict).
    Halts (raises) if computed grade is below grade_threshold.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_rows: list[dict] = []  # type: ignore[type-arg]
    accepted_ids: set[str] = set()
    snr_values: list[float] = []

    for _, row in df.iterrows():
        reasons = []
        file_id = str(row["id"])
        duration = float(row["duration"])

        if duration < min_duration:
            reasons.append(f"duration_too_short:{duration:.2f}s")
        elif duration > max_duration:
            reasons.append(f"duration_too_long:{duration:.2f}s")

        audio_path = audio_normalized_dir / f"{file_id}.wav"
        if audio_path.exists():
            import soundfile as sf

            data, _ = sf.read(str(audio_path))
            snr = compute_snr_db(data.astype(np.float32))
            snr_values.append(snr)
            if snr < min_snr_db:
                reasons.append(f"low_snr:{snr:.1f}dB")
        else:
            reasons.append("normalized_audio_missing")

        mel_path = mel_dir / f"{file_id}.npy"
        if not mel_path.exists():
            reasons.append("mel_missing")

        if reasons:
            rejected_rows.append({"id": file_id, "reason": "; ".join(reasons), "duration": duration})
        else:
            accepted_ids.add(file_id)

    accepted_df = df[df["id"].isin(accepted_ids)].copy()
    rejected_df = pd.DataFrame(rejected_rows)

    # ── Phoneme distribution ──────────────────────────────────────────────────
    phoneme_dist: Counter[str] = Counter()
    if "phonemes" in accepted_df.columns:
        for phoneme_str in accepted_df["phonemes"].dropna():
            phoneme_dist.update(str(phoneme_str).split())

    phoneme_dist_path = output_dir / "phoneme_distribution.json"
    with open(phoneme_dist_path, "w") as f:
        json.dump(dict(phoneme_dist.most_common()), f, indent=2, ensure_ascii=False)

    # ── Duration histogram ────────────────────────────────────────────────────
    durations = accepted_df["duration"].tolist()
    bins = [0, 1, 2, 3, 5, 8, 12, 15, 20]
    hist, _ = np.histogram(durations, bins=bins)
    dur_hist = {f"{bins[i]}-{bins[i+1]}s": int(hist[i]) for i in range(len(hist))}
    with open(output_dir / "duration_histogram.json", "w") as f:
        json.dump(dur_hist, f, indent=2)

    # ── SNR distribution ──────────────────────────────────────────────────────
    snr_stats = {}
    if snr_values:
        snr_arr = np.array(snr_values)
        snr_stats = {
            "mean": float(np.nanmean(snr_arr)),
            "median": float(np.nanmedian(snr_arr)),
            "p10": float(np.nanpercentile(snr_arr, 10)),
            "p90": float(np.nanpercentile(snr_arr, 90)),
        }
    with open(output_dir / "snr_distribution.json", "w") as f:
        json.dump(snr_stats, f, indent=2)

    # ── Rejected samples ──────────────────────────────────────────────────────
    rejected_df.to_csv(output_dir / "rejected_samples.csv", index=False)

    # ── Dataset scorecard ─────────────────────────────────────────────────────
    n_total = len(df)
    n_rejected = len(rejected_df)
    rejection_rate = n_rejected / max(n_total, 1)

    total_phonemes = sum(phoneme_dist.values())
    covered = sum(1 for v in phoneme_dist.values() if v >= 10)
    coverage_score = covered / max(len(phoneme_dist), 1) if phoneme_dist else 1.0

    median_snr = float(snr_stats.get("median", 0.0))
    grade = _compute_grade(rejection_rate, coverage_score, median_snr)

    scorecard = {
        "grade": grade,
        "n_total": n_total,
        "n_accepted": len(accepted_df),
        "n_rejected": n_rejected,
        "rejection_rate": round(rejection_rate, 4),
        "phoneme_coverage_score": round(coverage_score, 4),
        "median_snr_db": round(median_snr, 2),
        "splits": {
            "train": int((accepted_df["split"] == "train").sum()),
            "val": int((accepted_df["split"] == "val").sum()),
            "test": int((accepted_df["split"] == "test").sum()),
        },
    }
    with open(output_dir / "dataset_scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2)

    logger.info(
        "Dataset scorecard: grade=%s, accepted=%d/%d (%.1f%% rejected), SNR_median=%.1f dB",
        grade,
        len(accepted_df),
        n_total,
        rejection_rate * 100,
        median_snr,
    )

    threshold_idx = _GRADE_ORDER.index(grade_threshold)
    grade_idx = _GRADE_ORDER.index(grade)
    if grade_idx > threshold_idx:
        raise RuntimeError(
            f"Dataset quality grade {grade!r} is below threshold {grade_threshold!r}. "
            f"Check {output_dir}/dataset_scorecard.json for details."
        )

    return accepted_df, rejected_df, scorecard
