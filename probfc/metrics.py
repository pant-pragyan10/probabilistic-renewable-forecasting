"""Probabilistic forecast metrics. Predictions are arrays of shape [n, K] for K quantile levels."""
from __future__ import annotations

import numpy as np


def _prep(y, q, taus):
    y, q, taus = np.asarray(y, float), np.asarray(q, float), np.asarray(taus, float)
    if q.ndim != 2 or q.shape != (len(y), len(taus)):
        raise ValueError(f"q must have shape ({len(y)}, {len(taus)}), got {q.shape}")
    return y, q, taus


def pinball_per_quantile(y, q, taus) -> np.ndarray:
    """Mean pinball (quantile) loss for each quantile level -> shape [K]."""
    y, q, taus = _prep(y, q, taus)
    diff = y[:, None] - q
    return np.mean(np.maximum(taus * diff, (taus - 1.0) * diff), axis=0)


def pinball(y, q, taus) -> float:
    """Pinball loss averaged over samples and quantile levels."""
    return float(pinball_per_quantile(y, q, taus).mean())


def crps(y, q, taus) -> float:
    """CRPS approximated from the quantile grid: 2 x mean pinball loss.
    This is an approximation (finite grid, 5%..95% only); it is used to compare models, not as an exact CRPS."""
    return 2.0 * pinball(y, q, taus)


def calibration(y, q, taus) -> np.ndarray:
    """Observed share of outcomes at or below each predicted quantile -> shape [K]. Ideal: equals taus."""
    y, q, taus = _prep(y, q, taus)
    return np.mean(y[:, None] <= q, axis=0)


def coverage(y, q, taus, lo: float, hi: float) -> float:
    """Share of outcomes inside the central [lo, hi] interval taken from the quantile grid."""
    y, q, taus = _prep(y, q, taus)
    i, j = int(np.argmin(np.abs(taus - lo))), int(np.argmin(np.abs(taus - hi)))
    return float(np.mean((y >= q[:, i]) & (y <= q[:, j])))


def sharpness(q, taus, lo: float, hi: float) -> float:
    """Mean width of the central [lo, hi] interval (narrower is sharper, only meaningful alongside coverage)."""
    q, taus = np.asarray(q, float), np.asarray(taus, float)
    i, j = int(np.argmin(np.abs(taus - lo))), int(np.argmin(np.abs(taus - hi)))
    return float(np.mean(q[:, j] - q[:, i]))


def skill_score(score: float, reference: float) -> float:
    """1 - score/reference. Positive = better than the reference, 0 = same, negative = worse."""
    return float(1.0 - score / reference)


def summarize(y, q, taus) -> dict:
    """The standard row reported for every model."""
    return {
        "pinball": pinball(y, q, taus),
        "crps_approx": crps(y, q, taus),
        "median_mae": float(np.mean(np.abs(np.asarray(y) - np.asarray(q)[:, int(np.argmin(np.abs(np.asarray(taus) - 0.5)))]))),
        "coverage_90": coverage(y, q, taus, 0.05, 0.95),
        "coverage_50": coverage(y, q, taus, 0.25, 0.75),
        "width_90": sharpness(q, taus, 0.05, 0.95),
        "width_50": sharpness(q, taus, 0.25, 0.75),
    }
