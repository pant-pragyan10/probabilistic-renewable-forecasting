"""'What-if' forecasting: set a few weather inputs, get the LightGBM quantile forecast for them.

The model needs ~30 inputs. The user sets the main ones; the rest (e.g. top-of-atmosphere radiation, other lags) are filled with the
typical value for that zone, hour of day and time of year, taken from the TRAINING period. Neighbouring hours are assumed to have the same
weather as the chosen hour. This is a demonstration of the input -> output behaviour, not a real forecast for a real date."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import resolve_path


def load_boosters(cfg: dict, track: str):
    import lightgbm as lgb

    folder = resolve_path(cfg, "results") / "models" / f"lgbm_{track}"
    paths = [folder / f"q{int(round(t * 100)):02d}.txt" for t in cfg["quantiles"]]
    if not all(p.exists() for p in paths):
        return None
    return [lgb.Booster(model_file=str(p)) for p in paths]


def typical_rows(train: pd.DataFrame, zone, hour: int, doy: int, window_days: int = 30) -> pd.DataFrame:
    """Training rows of this zone within +-1 hour of the chosen hour and +-window_days of the chosen day of year."""
    t = train[train["zone"] == zone]
    dh = ((t["timestamp"].dt.hour - hour + 12) % 24) - 12
    dd = ((t["timestamp"].dt.dayofyear - doy + 182) % 365) - 182
    sub = t[(dh.abs() <= 1) & (dd.abs() <= window_days)]
    return sub if len(sub) >= 20 else t[dh.abs() <= 1]


def _calendar(row: pd.Series, hour: int, doy: int) -> None:
    row["hour_sin"], row["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    row["doy_sin"], row["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)


def solar_row(typical: pd.DataFrame, feature_cols, cfg, hour, doy, ssrd, cloud_cover, temp_c, humidity, power_yesterday) -> pd.Series:
    row = typical[feature_cols].median()
    row["ssrd"] = row["ssrd_prev"] = row["ssrd_next"] = float(ssrd)
    row["tcc"] = row["tcc_prev"] = row["tcc_next"] = float(cloud_cover)
    row["t2m"], row["rh"], row["y_lag24"] = float(temp_c) + 273.15, float(humidity), float(power_yesterday)
    row["csi"] = float(ssrd) / max(float(row["tsr"]), 1.0) if row["tsr"] > cfg["sun_up_threshold"] else 0.0
    _calendar(row, hour, doy)
    return row


def wind_row(typical: pd.DataFrame, feature_cols, cfg, hour, doy, speed100, direction_deg, power_yesterday) -> pd.Series:
    row = typical[feature_cols].median()
    ratio = float(typical["ws10"].median() / max(typical["ws100"].median(), 0.1))
    ang = np.deg2rad(direction_deg)
    ws10 = float(speed100) * ratio
    row["u100"] = row["u100_prev"] = float(speed100) * np.sin(ang)
    row["v100"] = row["v100_prev"] = float(speed100) * np.cos(ang)
    row["u10"], row["v10"] = ws10 * np.sin(ang), ws10 * np.cos(ang)
    row["ws100"], row["ws10"] = float(speed100), ws10
    row["shear"], row["ws100_sq"], row["ws100_cu"] = row["ws100"] - ws10, row["ws100"] ** 2, row["ws100"] ** 3
    row["wd_sin"], row["wd_cos"] = np.sin(ang), np.cos(ang)
    row["y_lag24"] = float(power_yesterday)
    _calendar(row, hour, doy)
    return row


def predict(boosters, row: pd.Series, zone, zones, sun_up: bool | None = None) -> np.ndarray:
    names = boosters[0].feature_name()
    X = pd.DataFrame([row[[n for n in names if n != "zone"]].to_dict()])
    X["zone"] = pd.Categorical([zone], categories=sorted(zones))
    X = X[names]
    q = np.sort(np.array([b.predict(X)[0] for b in boosters]))
    q = np.clip(q, 0.0, 1.0)
    return np.zeros_like(q) if sun_up is False else q