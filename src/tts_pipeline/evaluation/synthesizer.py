from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from tts_pipeline.utils.audio_utils import save_audio
from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def synthesize_test_set(
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: object,
    test_file: Path,
    metadata_path: Path,
    output_dir: Path,
    sample_rate: int = 22050,
    n_ode_steps: int = 50,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, dict[str, object]]:
    """Synthesize all utterances in test_file. Returns dict of {id: {wav_path, rtf}}."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    vocoder = vocoder.to(device).eval()

    meta = pd.read_csv(metadata_path).set_index("id")
    test_ids = test_file.read_text().strip().splitlines()
    results: dict[str, dict[str, object]] = {}

    for i in range(0, len(test_ids), batch_size):
        batch_ids = test_ids[i : i + batch_size]
        phoneme_seqs = []
        for uid in batch_ids:
            phoneme_str = str(meta.loc[uid].get("phonemes", ""))
            phonemes = phoneme_str.split()
            ids = tokenizer.encode(phonemes)  # type: ignore[union-attr]
            phoneme_seqs.append(torch.tensor(ids, dtype=torch.long))

        # Pad to same length
        max_len = max(s.shape[0] for s in phoneme_seqs)
        padded = torch.zeros(len(phoneme_seqs), max_len, dtype=torch.long).to(device)
        lengths = torch.zeros(len(phoneme_seqs), dtype=torch.long).to(device)
        for j, seq in enumerate(phoneme_seqs):
            padded[j, : seq.shape[0]] = seq.to(device)
            lengths[j] = seq.shape[0]

        t0 = time.perf_counter()
        with torch.no_grad():
            mel = model.synthesize(padded, lengths, n_steps=n_ode_steps)
            audio = vocoder(mel).squeeze(1).cpu().numpy()
        elapsed = time.perf_counter() - t0

        for j, uid in enumerate(batch_ids):
            audio_j = audio[j]
            duration_sec = audio_j.shape[0] / sample_rate
            rtf = elapsed / max(duration_sec * len(batch_ids), 1e-8) if j == 0 else 0.0
            wav_path = output_dir / f"{uid}.wav"
            save_audio(wav_path, audio_j, sample_rate)
            results[uid] = {"wav_path": str(wav_path), "rtf": rtf, "duration_sec": duration_sec}

    logger.info("Synthesized %d utterances → %s", len(results), output_dir)
    return results
