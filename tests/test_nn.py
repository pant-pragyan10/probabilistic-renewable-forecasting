import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from probfc import metrics as M  # noqa: E402
from probfc.models.nn_qr import NNQuantile  # noqa: E402
from probfc.postprocess import crossing_rate  # noqa: E402

TAUS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])


def table(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    d = pd.DataFrame({"timestamp": pd.date_range("2013-01-01", periods=n, freq="h"), "zone": rng.integers(1, 3, n), "x": x})
    d["y"] = np.sin(x) + rng.normal(scale=0.1, size=n)
    d["sun_up"] = True
    return d


def fast(**kw):
    return NNQuantile(TAUS, ["x"], hidden=32, depth=2, max_epochs=40, patience=8, n_seeds=2, device="cpu", **kw)


def test_quantiles_are_ordered_by_construction_and_learn_the_signal():
    d = table()
    tr, va, te = d.iloc[:2000], d.iloc[2000:2500], d.iloc[2500:]
    m = fast().fit(tr, valid=va)
    q = m.predict_quantiles(te)
    assert q.shape == (len(te), 5)
    assert crossing_rate(q) == 0.0
    assert M.pinball(te["y"], q, TAUS) < 0.08
    assert abs(M.coverage(te["y"], q, TAUS, 0.1, 0.9) - 0.8) < 0.1


def test_same_seed_gives_same_predictions():
    d = table(1200)
    a = fast().fit(d.iloc[:900], valid=d.iloc[900:]).predict_quantiles(d.iloc[:20])
    b = fast().fit(d.iloc[:900], valid=d.iloc[900:]).predict_quantiles(d.iloc[:20])
    assert np.allclose(a, b, atol=1e-5)


def test_fixed_epochs_mode_and_missing_args():
    d = table(800)
    m = fast().fit(d, fixed_epochs=3)
    assert m.best_epochs_ == [3, 3]
    with pytest.raises(ValueError):
        fast().fit(d)


def test_daylight_only_ignores_night_rows():
    d = table(1500)
    d.loc[::2, "sun_up"] = False
    d.loc[~d["sun_up"], "y"] = 1000.0
    m = fast(daylight_only=True).fit(d.iloc[:1000], valid=d.iloc[1000:])
    assert np.abs(m.predict_quantiles(d[d["sun_up"]].iloc[:50])).max() < 5