"""Leakage-safe features. Everything is computed per zone on the hourly timeline.

Rule: a feature for target hour t may use
  * weather FORECASTS for hours t-1, t, t+1 (the NWP forecast for that day is issued before t), and
  * measured power only from t-24 or earlier (day-ahead issue time).
It must never use power at t-23 .. t. tests/test_features.py checks this by perturbing the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import feature_columns, sun_up_mask

LAG_HOURS = (24, 48, 168)
NWP_SMOOTH = {"solar": ["ssrd", "tcc"], "wind": ["u100", "v100"]}


def _calendar(ts: pd.Series) -> pd.DataFrame:
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.dayofyear
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=ts.index,
    )


def build_features(df: pd.DataFrame, track: str, cfg: dict) -> pd.DataFrame:
    """Return a frame aligned with df (same index) holding timestamp, zone, y, sun_up (solar) and features."""
    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)
    out = df[["timestamp", "zone", "y"]].copy()
    raw = feature_columns(track, cfg)
    for c in raw:
        out[c] = df[c]
    out = pd.concat([out, _calendar(df["timestamp"])], axis=1)

    if track == "solar":
        out["sun_up"] = sun_up_mask(df, cfg).to_numpy()
        # clear-sky-index style proxy: how much of the top-of-atmosphere radiation reaches the ground
        out["csi"] = np.where(df["tsr"] > cfg["sun_up_threshold"], df["ssrd"] / df["tsr"].clip(lower=1.0), 0.0)
    else:
        out["ws10"] = np.hypot(df["u10"], df["v10"])
        out["ws100"] = np.hypot(df["u100"], df["v100"])
        out["shear"] = out["ws100"] - out["ws10"]
        ang = np.arctan2(df["u100"], df["v100"])
        out["wd_sin"], out["wd_cos"] = np.sin(ang), np.cos(ang)
        out["ws100_sq"], out["ws100_cu"] = out["ws100"] ** 2, out["ws100"] ** 3

    g = df.groupby("zone")
    for c in NWP_SMOOTH[track]:                       # forecasts of the neighbouring hours (allowed)
        out[f"{c}_prev"] = g[c].shift(1)
        out[f"{c}_next"] = g[c].shift(-1)
    for L in LAG_HOURS:                               # measured power, at least 24 h old
        out[f"y_lag{L}"] = g["y"].shift(L)
    out["y_mean_24h"] = g["y"].transform(lambda s: s.shift(24).rolling(24, min_periods=24).mean())
    return out


def model_columns(table: pd.DataFrame, kind: str, track: str) -> list[str]:
    """Column sets: 'linear' is a compact set for Quantile Regression; 'full' is everything for tree/NN models."""
    meta = {"timestamp", "zone", "y", "sun_up"}
    full = [c for c in table.columns if c not in meta]
    if kind == "full":
        return full
    if track == "solar":
        return ["ssrd", "csi", "tcc", "rh", "t2m", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "y_lag24", "y_mean_24h"]
    return ["ws100", "ws100_sq", "ws100_cu", "shear", "wd_sin", "wd_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "y_lag24", "y_mean_24h"]


def usable_rows(table: pd.DataFrame) -> pd.Series:
    """Rows whose every feature exists (drops the first week of each zone and the last hour)."""
    cols = [c for c in table.columns if c not in ("timestamp", "zone", "y", "sun_up")]
    return table[cols].notna().all(axis=1)
