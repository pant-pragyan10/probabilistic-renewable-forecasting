"""Make raw model output a valid forecast: monotone quantiles, inside [0, 1], zero at night (solar)."""
from __future__ import annotations

import numpy as np


def crossing_rate(q) -> float:
    """Share of rows where raw quantiles are not non-decreasing (before sorting)."""
    q = np.asarray(q, float)
    return float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1)))


def postprocess(q, sun_up=None, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    q = np.sort(np.asarray(q, float), axis=1)       # fixes quantile crossing
    q = np.clip(q, lo, hi)                           # capacity bounds
    if sun_up is not None:
        q = np.where(np.asarray(sun_up, bool)[:, None], q, 0.0)  # no solar power at night
    return q
