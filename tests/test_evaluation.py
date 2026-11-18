import numpy as np
import pandas as pd
import pytest

from probfc import metrics as M
from probfc.evaluation import pinball_by, row_pinball

TAUS = np.array([0.1, 0.5, 0.9])


def test_row_pinball_mean_equals_overall_pinball():
    rng = np.random.default_rng(0)
    y = rng.uniform(size=200)
    q = np.sort(rng.uniform(size=(200, 3)), axis=1)
    assert row_pinball(y, q, TAUS).mean() == pytest.approx(M.pinball(y, q, TAUS))


def test_pinball_by_groups_and_respects_mask():
    base = pd.DataFrame({"y": [1.0, 1.0, 0.0, 0.0], "g": ["a", "a", "b", "b"]})
    perfect = np.tile([[1.0, 1.0, 1.0]], (4, 1))             # right for group a, wrong by 1 for group b
    out = pinball_by(base, {"m": perfect}, TAUS, "g")
    assert out.loc["a", "m"] == 0.0 and out.loc["b", "m"] > 0.4
    masked = pinball_by(base, {"m": perfect}, TAUS, "g", mask=[True, True, False, False])
    assert list(masked.index) == ["a"]