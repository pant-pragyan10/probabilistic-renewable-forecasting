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


def solar_row(typical: pd.DataFrame, feature_cols, cfg, hour, doy, ssrd, cloud_cover, temp_c, humidity, power_yesterday,
              ssrd_prev=None, ssrd_next=None, cloud_prev=None, cloud_next=None, tsr=None) -> pd.Series:
    """Neighbouring-hour values default to the chosen hour's. `tsr` (top-of-atmosphere net radiation) can be given, e.g. from solar
    geometry for a live location; by default it is the typical value for this zone, hour and season."""
    row = typical[feature_cols].median()
    row["ssrd"] = float(ssrd)
    row["ssrd_prev"] = float(ssrd if ssrd_prev is None else ssrd_prev)
    row["ssrd_next"] = float(ssrd if ssrd_next is None else ssrd_next)
    row["tcc"] = float(cloud_cover)
    row["tcc_prev"] = float(cloud_cover if cloud_prev is None else cloud_prev)
    row["tcc_next"] = float(cloud_cover if cloud_next is None else cloud_next)
    if tsr is not None:
        row["tsr"] = float(tsr)
    row["t2m"], row["rh"], row["y_lag24"] = float(temp_c) + 273.15, float(humidity), float(power_yesterday)
    row["csi"] = float(ssrd) / max(float(row["tsr"]), 1.0) if row["tsr"] > cfg["sun_up_threshold"] else 0.0
    _calendar(row, hour, doy)
    return row


def wind_row(typical: pd.DataFrame, feature_cols, cfg, hour, doy, speed100, direction_deg, power_yesterday,
             prev=None, next=None) -> pd.Series:
    """`prev` / `next` = optional (speed100, direction_deg) of the neighbouring hours (default: same as the chosen hour)."""
    row = typical[feature_cols].median()
    ratio = float(typical["ws10"].median() / max(typical["ws100"].median(), 0.1))
    ang = np.deg2rad(direction_deg)
    ws10 = float(speed100) * ratio
    row["u100"], row["v100"] = float(speed100) * np.sin(ang), float(speed100) * np.cos(ang)
    for suffix, nb in (("_prev", prev), ("_next", next)):
        sp, dg = (speed100, direction_deg) if nb is None else nb
        row["u100" + suffix], row["v100" + suffix] = float(sp) * np.sin(np.deg2rad(dg)), float(sp) * np.cos(np.deg2rad(dg))
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


# ---- all four models --------------------------------------------------------------------------------------------
class NumpyNN:
    """The saved neural network, evaluated with plain numpy (no torch needed in the app). Weights come from
    results/models/nn_<track>.npz, written by scripts/09_export_nn.py."""
    name = "neural_net"

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.columns = [str(c) for c in z["columns"]]
        self.mu, self.sd = z["mu"].astype(np.float32), z["sd"].astype(np.float32)
        self.zone_index = {str(k): int(v) for k, v in zip(z["zone_names"], z["zone_index"])}
        self.nets = []
        for s in range(int(z["n_seeds"])):
            g = lambda k: z[f"s{s}_{k}"].astype(np.float32)  # noqa: E731
            body, i = [], 0
            while f"s{s}_body.{i * 3}.weight" in z.files:
                body.append((g(f"body.{i * 3}.weight"), g(f"body.{i * 3}.bias")))
                i += 1
            self.nets.append((g("emb.weight"), body, g("head.weight"), g("head.bias")))

    def predict_quantiles(self, table: pd.DataFrame) -> np.ndarray:
        X = (table[self.columns].to_numpy(np.float32) - self.mu) / self.sd
        z = table["zone"].astype(str).map(self.zone_index).to_numpy(np.int64)
        outs = []
        for emb, body, Wh, bh in self.nets:
            h = np.concatenate([X, emb[z]], axis=1)
            for W, b in body:
                h = np.maximum(h @ W.T + b, 0.0)
            o = h @ Wh.T + bh
            first, inc = o[:, :1], np.log1p(np.exp(o[:, 1:])) * 0.1
            outs.append(np.concatenate([first, first + np.cumsum(inc, axis=1)], axis=1))
        return np.mean(outs, axis=0)


def export_nn(pt_path, npz_path) -> None:
    """Convert a saved torch checkpoint (NNQuantile.save) into the numpy .npz that NumpyNN reads. Needs torch (run once)."""
    import torch

    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arrays = {"columns": np.array(ck["columns"]), "mu": np.asarray(ck["mu"]), "sd": np.asarray(ck["sd"]),
              "zone_names": np.array([str(k) for k in ck["zones"]]), "zone_index": np.array(list(ck["zones"].values())),
              "n_seeds": np.array(len(ck["states"]))}
    for s, st in enumerate(ck["states"]):
        for k, v in st.items():
            arrays[f"s{s}_{k}"] = v.cpu().numpy()
    np.savez(npz_path, **arrays)


def load_models(cfg: dict, track: str, train: pd.DataFrame, feature_cols: list[str]) -> dict:
    """LightGBM (saved boosters), neural net (saved weights, numpy) and Quantile Regression (refitted here in a few seconds
    from the training period). Missing pieces are left out; the dict maps model name -> object."""
    from .features import model_columns
    from .models import LinearQuantileRegression

    out = {}
    bo = load_boosters(cfg, track)
    if bo is not None:
        out["lightgbm"] = bo
    npz = resolve_path(cfg, "results") / "models" / f"nn_{track}.npz"
    if npz.exists():
        out["neural_net"] = NumpyNN(npz)
    taus = np.asarray(cfg["quantiles"])
    out["quantile_regression"] = LinearQuantileRegression(taus, model_columns(train, "linear", track), daylight_only=(track == "solar")).fit(train)
    return out


def predict_all(models: dict, rows: list[pd.Series], zone, zones, sun_up=None) -> dict:
    """Quantile forecasts [n_rows, 19] from every loaded model for a list of input rows (post-processed: sorted, in [0,1], zero at night)."""
    from .postprocess import postprocess

    table = pd.DataFrame([r.to_dict() for r in rows])
    table["zone"] = zone
    sun = None if sun_up is None else np.asarray(sun_up, bool)
    out = {}
    for name, mdl in models.items():
        if name == "lightgbm":
            names = mdl[0].feature_name()
            X = table[[n for n in names if n != "zone"]].copy()
            X["zone"] = pd.Categorical(table["zone"], categories=sorted(zones))
            raw = np.column_stack([b.predict(X[names]) for b in mdl])
        else:
            raw = mdl.predict_quantiles(table)
        out[name] = postprocess(raw, sun)
    return out