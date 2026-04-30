from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def compute_pesq_stoi(
    synth_paths: list[Path],
    ref_paths: list[Path],
    sample_rate: int = 22050,
) -> dict[str, float]:
    pesq_scores: list[float] = []
    stoi_scores: list[float] = []

    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError as e:
        logger.error("Missing pesq/pystoi: %s", e)
        return {"pesq_wb": 0.0, "pesq_nb": 0.0, "stoi_mean": 0.0}

    for synth_p, ref_p in zip(synth_paths, ref_paths):
        if not synth_p.exists() or not ref_p.exists():
            continue
        try:
            synth, _ = sf.read(str(synth_p))
            ref, _ = sf.read(str(ref_p))
            min_len = min(len(synth), len(ref))
            synth = synth[:min_len].astype(np.float32)
            ref = ref[:min_len].astype(np.float32)

            # PESQ requires 8kHz or 16kHz
            if sample_rate not in (8000, 16000):
                import torchaudio
                import torch
                resample = torchaudio.transforms.Resample(sample_rate, 16000)
                synth_16k = resample(torch.from_numpy(synth).unsqueeze(0)).squeeze().numpy()
                ref_16k = resample(torch.from_numpy(ref).unsqueeze(0)).squeeze().numpy()
                pesq_wb = pesq(16000, ref_16k, synth_16k, "wb")
            else:
                pesq_wb = pesq(sample_rate, ref, synth, "wb" if sample_rate == 16000 else "nb")

            stoi_score = stoi(ref, synth, sample_rate, extended=False)
            pesq_scores.append(float(pesq_wb))
            stoi_scores.append(float(stoi_score))
        except Exception as exc:
            logger.warning("PESQ/STOI failed for %s: %s", synth_p, exc)

    return {
        "pesq_wb": float(np.mean(pesq_scores)) if pesq_scores else 0.0,
        "stoi_mean": float(np.mean(stoi_scores)) if stoi_scores else 0.0,
    }
