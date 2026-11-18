import numpy as np
import pandas as pd
import pytest

from probfc import metrics as M
from probfc.models import LightGBMQuantile
from probfc.postprocess import postprocess

TAUS = np.array([0.1, 0.5, 0.9])


def table(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    d = pd.DataFrame({"timestamp": pd.date_range("2013-01-01", periods=n, freq="h"),
                      "zone": rng.integers(1, 3, n), "x": x})
    d["y"] = np.sin(x) + rng.normal(scale=0.1, size=n)      # non-linear signal
    d["sun_up"] = True
    return d


def test_learns_nonlinear_signal_with_calibrated_quantiles():
    d = table()
    tr, va, te = d.iloc[:2000], d.iloc[2000:2500], d.iloc[2500:]
    m = LightGBMQuantile(TAUS, ["x"], {"num_leaves": 8, "min_child_samples": 20, "learning_rate": 0.1}).fit(tr, valid=va, max_trees=200)
    q = postprocess(m.predict_quantiles(te), lo=-5, hi=5)
    assert q.shape == (len(te), 3)
    assert abs(M.coverage(te["y"], q, TAUS, 0.1, 0.9) - 0.8) < 0.1
    assert M.pinball(te["y"], q, TAUS) < 0.06


def test_fixed_n_estimators_is_used_and_deterministic():
    d = table()
    a = LightGBMQuantile(TAUS, ["x"], n_estimators=[20, 30, 40]).fit(d)
    b = LightGBMQuantile(TAUS, ["x"], n_estimators=[20, 30, 40]).fit(d)
    assert a.best_iters_ == [20, 30, 40]
    assert np.allclose(a.predict_quantiles(d[:50]), b.predict_quantiles(d[:50]))


def test_requires_valid_when_learning_trees():
    with pytest.raises(ValueError):
        LightGBMQuantile(TAUS, ["x"]).fit(table())


def test_daylight_only_ignores_night_rows():
    d = table()
    d.loc[::2, "sun_up"] = False
    d.loc[~d["sun_up"], "y"] = 1000.0
    m = LightGBMQuantile(TAUS, ["x"], daylight_only=True, n_estimators=[30, 30, 30]).fit(d)
    assert np.abs(m.predict_quantiles(d[d["sun_up"]])).max() < 5