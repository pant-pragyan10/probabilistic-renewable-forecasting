import numpy as np
import pytest

from probfc import metrics as M
from probfc.postprocess import crossing_rate, postprocess

TAUS = np.array([0.1, 0.5, 0.9])


def test_pinball_matches_hand_calculation():
    y = np.array([1.0])
    q = np.array([[0.0, 0.5, 2.0]])
    # tau .1: under-forecast by 1 -> .1*1 ; tau .5: under by .5 -> .25 ; tau .9: over by 1 -> .1*1
    assert M.pinball_per_quantile(y, q, TAUS) == pytest.approx([0.1, 0.25, 0.1])
    assert M.pinball(y, q, TAUS) == pytest.approx(0.45 / 3)
    assert M.crps(y, q, TAUS) == pytest.approx(2 * 0.45 / 3)


def test_perfect_forecast_scores_zero():
    y = np.array([0.3, 0.7])
    q = np.repeat(y[:, None], 3, axis=1)
    assert M.pinball(y, q, TAUS) == 0.0


def test_true_quantiles_are_best_in_expectation():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20000)
    true_q = np.tile(np.quantile(y, TAUS), (len(y), 1))
    shifted = true_q + 0.5
    assert M.pinball(y, true_q, TAUS) < M.pinball(y, shifted, TAUS)


def test_calibration_and_coverage_of_calibrated_forecast():
    rng = np.random.default_rng(1)
    y = rng.uniform(size=50000)
    q = np.tile(TAUS, (len(y), 1))             # uniform(0,1): the tau-quantile is tau
    assert M.calibration(y, q, TAUS) == pytest.approx(TAUS, abs=0.01)
    assert M.coverage(y, q, TAUS, 0.1, 0.9) == pytest.approx(0.8, abs=0.01)
    assert M.sharpness(q, TAUS, 0.1, 0.9) == pytest.approx(0.8)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.pinball(np.zeros(3), np.zeros((3, 2)), TAUS)


def test_skill_score_signs():
    assert M.skill_score(0.5, 1.0) == 0.5
    assert M.skill_score(1.0, 1.0) == 0.0
    assert M.skill_score(2.0, 1.0) < 0


def test_postprocess_sorts_clips_and_zeroes_night():
    raw = np.array([[0.5, 0.2, 1.4], [-0.3, 0.4, 0.6]])
    assert crossing_rate(raw) == 0.5
    q = postprocess(raw, sun_up=np.array([True, False]))
    assert q[0].tolist() == [0.2, 0.5, 1.0]
    assert q[1].tolist() == [0.0, 0.0, 0.0]
    assert np.all(np.diff(q, axis=1) >= 0) and q.min() >= 0 and q.max() <= 1
