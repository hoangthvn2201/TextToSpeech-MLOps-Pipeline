"""Data pipeline CLI entry point — called by dvc.yaml stages."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def run_ingest(args: argparse.Namespace) -> None:
    from tts_pipeline.data.ingestion import build_metadata

    df = build_metadata(
        audio_dir=Path("data/raw/audio"),
        transcript_dir=Path("data/raw/transcripts"),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/metadata.csv", index=False)
    print(f"Ingested {len(df)} samples → data/metadata.csv")


def run_normalize(args: argparse.Namespace) -> None:
    from tts_pipeline.data.audio_processor import process_batch
    from tts_pipeline.utils.config import load_params

    cfg = load_params()
    df = pd.read_csv("data/metadata.csv")
    pairs = [
        (Path(row["audio_path"]), Path("data/processed/audio_normalized") / f"{row['id']}.wav")
        for _, row in df.iterrows()
    ]
    process_batch(
        pairs,
        target_sr=cfg.data.sample_rate,
        peak_normalize_db=cfg.data.peak_normalize_db,
        trim_silence=cfg.data.trim_silence,
    )


def run_clean(args: argparse.Namespace) -> None:
    from tts_pipeline.data.transcript_cleaner import clean_metadata
    from tts_pipeline.utils.config import load_params

    cfg = load_params()
    df = pd.read_csv("data/metadata.csv")
    accepted, rejected = clean_metadata(df, language=cfg.data.language)
    Path("data/processed/transcripts_cleaned").mkdir(parents=True, exist_ok=True)
    accepted.to_csv("data/processed/transcripts_cleaned/accepted.csv", index=False)
    rejected.to_csv("data/processed/transcripts_cleaned/rejected.csv", index=False)
    print(f"Cleaned: {len(accepted)} accepted, {len(rejected)} rejected")


def run_phonemize(args: argparse.Namespace) -> None:
    from tts_pipeline.text.phonemizers.espeak_phonemizer import EspeakPhonemizer
    from tts_pipeline.text.phonemizers.vi_phonemizer import ViPhonemizer
    from tts_pipeline.utils.config import load_params

    cfg = load_params()
    df = pd.read_csv("data/processed/transcripts_cleaned/accepted.csv")

    phonemizer_map = {"espeak": EspeakPhonemizer, "vi": ViPhonemizer}
    PhonemClass = phonemizer_map.get(cfg.text.phonemizer, EspeakPhonemizer)
    phonemizer = PhonemClass(language=cfg.text.language)

    texts = df["text_clean"].fillna("").tolist()
    phoneme_lists = phonemizer.phonemize_batch(texts)
    df["phonemes"] = [" ".join(pl) for pl in phoneme_lists]
    df.to_csv("data/processed/phonemes.csv", index=False)
    print(f"Phonemized {len(df)} utterances")


def run_features(args: argparse.Namespace) -> None:
    from tts_pipeline.data.feature_extractor import compute_stats, extract_features_batch
    from tts_pipeline.utils.config import load_params

    cfg = load_params()
    df = pd.read_csv("data/processed/phonemes.csv")
    audio_paths = [Path("data/processed/audio_normalized") / f"{row['id']}.wav" for _, row in df.iterrows()]
    audio_paths = [p for p in audio_paths if p.exists()]

    feature_cfg = {
        "sample_rate": cfg.data.sample_rate,
        "n_fft": cfg.data.n_fft,
        "hop_length": cfg.data.hop_length,
        "win_length": cfg.data.win_length,
        "n_mels": cfg.data.n_mels,
        "f_min": cfg.data.f_min,
        "f_max": cfg.data.f_max,
    }
    mels = extract_features_batch(
        audio_paths,
        mel_dir=Path("data/processed/mels"),
        f0_dir=Path("data/processed/f0"),
        cfg=feature_cfg,
    )
    compute_stats(mels, out_path=Path("data/processed/stats.json"))
    print(f"Extracted features for {len(mels)} files")


def run_validate(args: argparse.Namespace) -> None:
    from tts_pipeline.data.validator import validate_dataset
    from tts_pipeline.utils.config import load_params

    cfg = load_params()
    df = pd.read_csv("data/processed/phonemes.csv")

    accepted, rejected, scorecard = validate_dataset(
        df=df,
        audio_normalized_dir=Path("data/processed/audio_normalized"),
        mel_dir=Path("data/processed/mels"),
        min_duration=cfg.data.min_duration,
        max_duration=cfg.data.max_duration,
        min_snr_db=cfg.data.min_snr_db,
        grade_threshold=cfg.data.dataset_grade_threshold,
        output_dir=Path("data/processed/validation"),
    )

    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        ids = accepted[accepted["split"] == split]["id"].tolist()
        (splits_dir / f"{split}.txt").write_text("\n".join(ids))
    print(f"Validation done: grade={scorecard['grade']}, accepted={scorecard['n_accepted']}/{scorecard['n_total']}")


STAGE_MAP = {
    "ingest": run_ingest,
    "normalize": run_normalize,
    "clean": run_clean,
    "phonemize": run_phonemize,
    "features": run_features,
    "validate": run_validate,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=list(STAGE_MAP.keys()))
    parser.add_argument("--val-ratio", default="0.05")
    parser.add_argument("--test-ratio", default="0.05")
    args = parser.parse_args()
    STAGE_MAP[args.stage](args)


if __name__ == "__main__":
    main()
