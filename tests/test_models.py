import numpy as np
import pandas as pd
import pytest

from probfc import metrics as M
from probfc.models import Climatology, LinearQuantileRegression
from probfc.postprocess import postprocess

TAUS = np.array([0.1, 0.5, 0.9])


def table(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2013-01-01", periods=n, freq="h")
    x = rng.normal(size=n)
    d = pd.DataFrame({"timestamp": ts, "zone": 1, "x": x, "y": 0.5 + 0.2 * x + rng.normal(scale=0.05, size=n)})
    d["sun_up"] = True
    return d


def test_climatology_shape_and_unseen_hour_fallback():
    d = table()
    m = Climatology(TAUS).fit(d)
    q = m.predict_quantiles(d)
    assert q.shape == (len(d), 3) and np.all(np.diff(q, axis=1) >= 0)
    other_zone = d.assign(zone=9)
    assert np.allclose(m.predict_quantiles(other_zone[:5]), m.fallback_)


def test_quantile_regression_recovers_a_linear_signal_and_beats_climatology():
    d = table()
    tr, te = d.iloc[:1500], d.iloc[1500:]
    qr = LinearQuantileRegression(TAUS, ["x"]).fit(tr)
    q = postprocess(qr.predict_quantiles(te))
    clim = postprocess(Climatology(TAUS).fit(tr).predict_quantiles(te))
    assert q.shape == (len(te), 3)
    assert M.pinball(te["y"], q, TAUS) < 0.5 * M.pinball(te["y"], clim, TAUS)
    assert abs(M.coverage(te["y"], q, TAUS, 0.1, 0.9) - 0.8) < 0.08


def test_daylight_only_fit_ignores_night_rows():
    d = table()
    d.loc[::2, "sun_up"] = False
    d.loc[~d["sun_up"], "y"] = 1000.0                 # garbage at night must not influence the fit
    qr = LinearQuantileRegression(TAUS, ["x"], daylight_only=True).fit(d)
    pred = qr.predict_quantiles(d[d["sun_up"]])
    assert np.all(np.abs(pred) < 5)
