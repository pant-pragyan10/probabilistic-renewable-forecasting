"""Translate scores into something an operator can use: how wide a safety range must be held to cover ~90% of outcomes?

Comparison (same point forecast = the model's own median, so accuracy of the central guess is identical):
  A) fixed margin: median +/- c, where c is one constant chosen on the first half of the evaluation period
  B) the model's own 5%-95% range (no tuning)
Both are scored on the second half of the period: coverage and average width (fraction of installed capacity).
A third column gives the fixed margin hindsight (margin set to match the model's coverage) as a conservative comparison.
Daylight hours only for solar."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from probfc.config import load_config, resolve_path  # noqa: E402
from probfc.evaluation import LABELS, MODELS, load_all  # noqa: E402


def analyse(track, cfg, tag):
    taus = np.asarray(cfg["quantiles"])
    base, Q = load_all(cfg, track, tag)
    mask = base["sun_up"].to_numpy() if track == "solar" else np.ones(len(base), bool)
    i05, i50, i95 = (int(np.argmin(np.abs(taus - t))) for t in (0.05, 0.50, 0.95))
    ts = base["timestamp"]
    cut = ts[mask].min() + (ts[mask].max() - ts[mask].min()) / 2
    calib, test = mask & (ts <= cut).to_numpy(), mask & (ts > cut).to_numpy()
    y = base["y"].to_numpy()
    rows = []
    for m in MODELS:
        q = Q[m]
        med = q[:, i50]
        err = y - med
        # fixed margin: smallest symmetric margin that covers 90% on the calibration half
        c = float(np.quantile(np.abs(err[calib]), 0.90))
        fixed_cov = float(np.mean(np.abs(err[test]) <= c))
        own_cov = float(np.mean((y[test] >= q[test, i05]) & (y[test] <= q[test, i95])))
        own_w = float(np.mean(q[test, i95] - q[test, i05]))
        # conservative comparison: give the fixed margin hindsight, setting it so it covers exactly as often as the model's own range
        c_eq = float(np.quantile(np.abs(err[test]), own_cov))
        rows.append({"track": track, "model": LABELS[m], "eval": tag, "n_test_half": int(test.sum()),
                     "fixed_margin_coverage": fixed_cov, "fixed_margin_width": 2 * c,
                     "own_range_coverage": own_cov, "own_range_width": own_w,
                     "fixed_margin_width_equal_coverage": 2 * c_eq,
                     "width_change_at_equal_coverage_pct": 100 * (own_w / (2 * c_eq) - 1)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg, tag = load_config(), ("test" if a.final else "val")
    out = pd.concat([analyse(t, cfg, tag) for t in ("solar", "wind")], ignore_index=True)
    out.round(4).to_csv(resolve_path(cfg, "results") / f"usefulness_{tag}.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()