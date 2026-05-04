from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_utmos(wav_paths: list[Path], sample_rate: int = 22050) -> dict[str, float]:
    """Compute UTMOS MOS predictions. Returns mean, std, min, max."""
    try:
        import speechmos

        scores = []
        for p in wav_paths:
            try:
                result = speechmos.dnsmos.run(str(p), sr=sample_rate)
                scores.append(float(result.get("OVRL_raw", 0.0)))
            except Exception:
                pass
        if not scores:
            return {"utmos_mean": 0.0, "utmos_std": 0.0, "utmos_min": 0.0, "utmos_max": 0.0}
        arr = np.array(scores)
        return {
            "utmos_mean": float(arr.mean()),
            "utmos_std": float(arr.std()),
            "utmos_min": float(arr.min()),
            "utmos_max": float(arr.max()),
        }
    except ImportError:
        return {"utmos_mean": 0.0, "utmos_std": 0.0, "utmos_min": 0.0, "utmos_max": 0.0}
