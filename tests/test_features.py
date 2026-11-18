import numpy as np
import pandas as pd
import pytest

from probfc.config import load_config
from probfc.data import feature_columns
from probfc.features import build_features, usable_rows

CFG = load_config()


def synthetic(track, hours=24 * 20, zones=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2013-01-01", periods=hours, freq="h")
    frames = []
    for z in zones:
        d = pd.DataFrame({"timestamp": ts, "zone": z})
        for c in feature_columns(track, CFG):
            d[c] = rng.normal(size=hours)
        d["tsr"] = d["tsr"].abs() * 100 if track == "solar" else None
        d["y"] = rng.uniform(size=hours)
        frames.append(d.drop(columns=[c for c in ["tsr"] if track == "wind"]))
    return pd.concat(frames, ignore_index=True)


FEATS_EXCLUDED = {"timestamp", "zone", "y", "sun_up"}


@pytest.mark.parametrize("track", ["solar", "wind"])
def test_no_leakage_from_recent_power(track):
    """Changing measured power at t-23..t (and later) must not change any feature at t."""
    df = synthetic(track)
    base = build_features(df, track, CFG)
    t_idx = 24 * 10                                   # a target hour inside zone 1's block
    tampered = df.copy()
    zone1 = tampered.index[tampered["zone"] == 1]
    tampered.loc[zone1[t_idx - 23: ], "y"] = 99.0     # corrupt power from t-23 onward
    new = build_features(tampered, track, CFG)
    cols = [c for c in base.columns if c not in FEATS_EXCLUDED]
    a = base[base["zone"] == 1].iloc[t_idx][cols].astype(float)
    b = new[new["zone"] == 1].iloc[t_idx][cols].astype(float)
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("track", ["solar", "wind"])
def test_lags_do_not_cross_zones(track):
    df = synthetic(track)
    f = build_features(df, track, CFG)
    for z, g in f.groupby("zone"):
        assert g["y_lag24"].iloc[:24].isna().all()    # first day of each zone has no lag
        assert g["y_lag24"].iloc[24:].notna().all()


def test_lag_values_are_exact():
    df = synthetic("wind")
    f = build_features(df, "wind", CFG)
    z1 = f[f["zone"] == 1].reset_index(drop=True)
    assert z1.loc[100, "y_lag24"] == z1.loc[76, "y"]
    assert z1.loc[300, "y_lag168"] == z1.loc[132, "y"]
    assert z1.loc[100, "y_mean_24h"] == pytest.approx(z1.loc[53:76, "y"].mean())


def test_usable_rows_drops_warmup_only():
    df = synthetic("solar", hours=24 * 20)
    f = build_features(df, "solar", CFG)
    keep = usable_rows(f)
    per_zone = keep.groupby(f["zone"]).sum()
    assert (per_zone == 24 * 20 - 168 - 1).all()      # 168 h warm-up for the weekly lag, 1 h for the 'next' forecast
