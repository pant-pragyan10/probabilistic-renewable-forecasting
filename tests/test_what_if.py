from pathlib import Path

import numpy as np
import pytest

from probfc.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
need = [ROOT / "results" / "models" / "lgbm_solar" / "q50.txt", ROOT / "results" / "models" / "lgbm_wind" / "q50.txt",
        ROOT / "data" / "raw" / "solar_new.csv", ROOT / "data" / "raw" / "wind.csv"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in need), reason="needs trained LightGBM models and raw data")


def setup(track):
    from probfc.data import load_track
    from probfc.features import build_features, usable_rows
    from probfc.split import split_masks
    from probfc.whatif import load_boosters

    t = build_features(load_track(track, CFG), track, CFG)
    t = t[usable_rows(t)].reset_index(drop=True)
    train = t[split_masks(t["timestamp"], track, CFG)["train"]]
    cols = [c for c in t.columns if c not in ("timestamp", "zone", "y", "sun_up")]
    return train, cols, load_boosters(CFG, track), sorted(t["zone"].unique())


def test_solar_more_sun_means_more_power_and_night_is_zero():
    from probfc.whatif import predict, solar_row, typical_rows

    train, cols, bo, zones = setup("solar")
    doy = 288
    typ = typical_rows(train, 1, 12, doy)
    sunny = predict(bo, solar_row(typ, cols, CFG, 12, doy, 950, 0.05, 25, 40, 0.5), 1, zones)
    dark = predict(bo, solar_row(typ, cols, CFG, 12, doy, 60, 0.95, 15, 90, 0.5), 1, zones)
    assert sunny.shape == (19,) and np.all(np.diff(sunny) >= 0) and 0 <= sunny.min() and sunny.max() <= 1
    assert sunny[9] > dark[9] + 0.2
    night = predict(bo, solar_row(typical_rows(train, 1, 2, doy), cols, CFG, 2, doy, 0, 0.5, 15, 60, 0.0), 1, zones, sun_up=False)
    assert night.max() == 0.0


def test_wind_more_wind_means_more_power():
    from probfc.whatif import predict, typical_rows, wind_row

    train, cols, bo, zones = setup("wind")
    typ = typical_rows(train, 1, 12, 100)
    calm = predict(bo, wind_row(typ, cols, CFG, 12, 100, 2.0, 90, 0.2), 1, zones)
    windy = predict(bo, wind_row(typ, cols, CFG, 12, 100, 12.0, 90, 0.2), 1, zones)
    assert np.all(np.diff(windy) >= 0)
    assert windy[9] > calm[9] + 0.3


def test_predict_all_gives_every_model_and_valid_quantiles():
    from probfc.whatif import load_models, predict_all, solar_row, typical_rows

    train, cols, bo, zones = setup("solar")
    models = load_models(CFG, "solar", train, cols)
    assert {"lightgbm", "quantile_regression"} <= set(models)
    rows = [solar_row(typical_rows(train, 1, h, 288), cols, CFG, h, 288, 700, 0.2, 22, 45, 0.5) for h in (11, 12)]
    out = predict_all(models, rows, 1, zones, sun_up=[True, True])
    for name, q in out.items():
        assert q.shape == (2, 19) and np.all(np.diff(q, axis=1) >= 0) and q.min() >= 0 and q.max() <= 1, name
    night = predict_all(models, rows, 1, zones, sun_up=[False, False])
    assert all(q.max() == 0 for q in night.values())