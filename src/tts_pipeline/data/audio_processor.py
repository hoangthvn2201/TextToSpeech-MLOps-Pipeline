from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F

from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_audio(
    src: Path,
    dst: Path,
    target_sr: int = 22050,
    peak_normalize_db: float = -3.0,
    trim_silence: bool = True,
) -> Path:
    waveform, sr = torchaudio.load(str(src))

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

    # Trim silence
    if trim_silence:
        waveform = F.vad(waveform, sample_rate=target_sr)
        # Reverse + trim leading silence from end
        waveform = F.vad(waveform.flip(-1), sample_rate=target_sr).flip(-1)

    # Peak normalize
    peak = waveform.abs().max()
    if peak > 0:
        target_linear = 10 ** (peak_normalize_db / 20)
        waveform = waveform * (target_linear / peak)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), waveform, target_sr, encoding="PCM_S", bits_per_sample=16)
    return dst


def process_batch(
    src_dst_pairs: list[tuple[Path, Path]],
    target_sr: int = 22050,
    peak_normalize_db: float = -3.0,
    trim_silence: bool = True,
    n_workers: int = 4,
) -> list[Path]:
    results: list[Path] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                normalize_audio, src, dst, target_sr, peak_normalize_db, trim_silence
            ): (src, dst)
            for src, dst in src_dst_pairs
        }
        for future in as_completed(futures):
            src, dst = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error("Failed to process %s: %s", src, exc)
    logger.info("Normalized %d / %d audio files", len(results), len(src_dst_pairs))
    return results
