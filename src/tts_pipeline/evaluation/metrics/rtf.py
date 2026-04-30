from __future__ import annotations

import numpy as np


def compute_rtf_stats(rtf_values: list[float]) -> dict[str, float]:
    if not rtf_values:
        return {"rtf_mean": 0.0, "rtf_p50": 0.0, "rtf_p95": 0.0, "rtf_max": 0.0}
    arr = np.array(rtf_values)
    return {
        "rtf_mean": float(arr.mean()),
        "rtf_p50": float(np.percentile(arr, 50)),
        "rtf_p95": float(np.percentile(arr, 95)),
        "rtf_max": float(arr.max()),
    }
