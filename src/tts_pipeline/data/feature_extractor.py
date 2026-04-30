from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def extract_mel(
    audio_path: Path,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    waveform, sr = torchaudio.load(str(audio_path))
    assert sr == sample_rate, f"Expected {sample_rate} Hz, got {sr} Hz for {audio_path}"

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=1.0,
        center=True,
        pad_mode="reflect",
    )
    mel = mel_transform(waveform)  # (1, n_mels, T)
    mel_db = T.AmplitudeToDB(stype="magnitude", top_db=80)(mel)
    return mel_db.squeeze(0).numpy()  # (n_mels, T)


def extract_f0(audio_path: Path, sample_rate: int = 22050, hop_length: int = 256) -> np.ndarray:
    """Extract F0 using torchcrepe (GPU-accelerated CREPE)."""
    try:
        import torchcrepe

        waveform, sr = torchaudio.load(str(audio_path))
        assert sr == sample_rate
        f0, _ = torchcrepe.predict(
            waveform,
            sr,
            hop_length=hop_length,
            fmin=50.0,
            fmax=1100.0,
            model="tiny",
            return_periodicity=True,
            batch_size=512,
            device="cpu",
        )
        return f0.squeeze(0).numpy()
    except ImportError:
        # Fallback: librosa PYIN
        import librosa

        y, _ = librosa.load(str(audio_path), sr=sample_rate)
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=50.0,
            fmax=1100.0,
            sr=sample_rate,
            hop_length=hop_length,
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0.astype(np.float32)


def _process_single(
    audio_path: Path,
    mel_out: Path,
    f0_out: Path,
    cfg: dict,  # type: ignore[type-arg]
) -> tuple[str, np.ndarray]:
    if mel_out.exists() and f0_out.exists():
        mel = np.load(mel_out)
        return audio_path.stem, mel

    mel = extract_mel(audio_path, **cfg)
    f0 = extract_f0(audio_path, sample_rate=cfg["sample_rate"], hop_length=cfg["hop_length"])

    mel_out.parent.mkdir(parents=True, exist_ok=True)
    f0_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(mel_out, mel)
    np.save(f0_out, f0)
    return audio_path.stem, mel


def extract_features_batch(
    audio_paths: list[Path],
    mel_dir: Path,
    f0_dir: Path,
    cfg: dict,  # type: ignore[type-arg]
    n_workers: int = 4,
) -> dict[str, np.ndarray]:
    pairs = [
        (p, mel_dir / f"{p.stem}.npy", f0_dir / f"{p.stem}.npy")
        for p in audio_paths
    ]
    mels: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_single, src, mel_out, f0_out, cfg): src.stem
            for src, mel_out, f0_out in pairs
        }
        for future in as_completed(futures):
            stem = futures[future]
            try:
                _, mel = future.result()
                mels[stem] = mel
            except Exception as exc:
                logger.error("Feature extraction failed for %s: %s", stem, exc)

    logger.info("Extracted features for %d / %d files", len(mels), len(audio_paths))
    return mels


def compute_stats(mels: dict[str, np.ndarray], out_path: Path) -> dict[str, float]:
    all_mels = np.concatenate([m.reshape(m.shape[0], -1) for m in mels.values()], axis=1)
    stats = {
        "mel_mean": float(all_mels.mean()),
        "mel_std": float(all_mels.std()),
        "mel_min": float(all_mels.min()),
        "mel_max": float(all_mels.max()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats
