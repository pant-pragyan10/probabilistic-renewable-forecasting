"""Helpers shared by the evaluation scripts: load every model's saved predictions on the same rows,
then slice and score them. Nothing here trains anything."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .config import resolve_path
from .data import load_track, sun_up_mask

MODELS = ["climatology", "quantile_regression", "lightgbm", "neural_net"]
LABELS = {"climatology": "Climatology", "quantile_regression": "Quantile Regression",
          "lightgbm": "LightGBM", "neural_net": "Neural network"}


def qcols(taus):
    return [f"q{int(round(t * 100)):02d}" for t in taus]


def load_all(cfg: dict, track: str, tag: str):
    """-> (base frame with timestamp, zone, y[, sun_up], dict model -> array [n, K]). Rows must match across models."""
    taus = cfg["quantiles"]
    base, Q = None, {}
    for m in MODELS:
        d = pd.read_parquet(resolve_path(cfg, "results") / "predictions" / f"{track}_{m}_{tag}.parquet")
        if base is None:
            base = d[["timestamp", "zone", "y"]].copy()
        else:
            same = (d["timestamp"].to_numpy() == base["timestamp"].to_numpy()).all() and \
                   (d["zone"].to_numpy() == base["zone"].to_numpy()).all() and np.allclose(d["y"], base["y"])
            if not same:
                raise ValueError(f"{track}/{m}: rows differ from the other models; rerun the training scripts")
        Q[m] = d[qcols(taus)].to_numpy()
    if track == "solar":
        full = load_track("solar", cfg)
        full["sun_up"] = sun_up_mask(full, cfg)
        base = base.merge(full[["timestamp", "zone", "sun_up"]], on=["timestamp", "zone"], how="left", validate="one_to_one")
        if base["sun_up"].isna().any():
            raise ValueError("could not attach the sun-up mask to every prediction row")
        base["sun_up"] = base["sun_up"].astype(bool)
    return base, Q


def row_pinball(y, q, taus) -> np.ndarray:
    """Pinball loss of each row, averaged over the quantile levels -> shape [n]."""
    y, q, taus = np.asarray(y, float), np.asarray(q, float), np.asarray(taus, float)
    diff = y[:, None] - q
    return np.maximum(taus * diff, (taus - 1.0) * diff).mean(axis=1)


def pinball_by(base: pd.DataFrame, Q: dict, taus, key, mask=None) -> pd.DataFrame:
    """Mean pinball per group (rows = group values, columns = models). `key` is a column name or a Series."""
    mask = np.ones(len(base), bool) if mask is None else np.asarray(mask, bool)
    k = base[key] if isinstance(key, str) else key
    out = {m: pd.Series(row_pinball(base["y"].to_numpy()[mask], Q[m][mask], taus)).groupby(k.to_numpy()[mask]).mean()
           for m in Q}
    return pd.DataFrame(out)


def results_table(base, Q, taus, track: str, tag: str) -> pd.DataFrame:
    subsets = {"all_hours": np.ones(len(base), bool)}
    if track == "solar":
        subsets["daylight"] = base["sun_up"].to_numpy()
    rows = []
    for sname, mask in subsets.items():
        ref = M.pinball(base["y"].to_numpy()[mask], Q["climatology"][mask], taus)
        for m in Q:
            s = M.summarize(base["y"].to_numpy()[mask], Q[m][mask], taus)
            rows.append({"track": track, "eval": tag, "subset": sname, "model": LABELS[m], "n": int(mask.sum()), **s,
                         "skill_vs_climatology": M.skill_score(s["pinball"], ref)})
    return pd.DataFrame(rows)