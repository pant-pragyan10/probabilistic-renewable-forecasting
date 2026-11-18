"""Which weather inputs drive the LightGBM forecast? Exact TreeSHAP values for the 10%, 50% and 90% quantile models.

Uses LightGBM's built-in TreeSHAP (predict(pred_contrib=True)), the same algorithm as shap.TreeExplainer, so the
`shap` package is not required. Reads the saved models from results/models/ (made by scripts/04_lightgbm.py)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from probfc import plots  # noqa: E402
from probfc.config import load_config, resolve_path  # noqa: E402
from probfc.data import load_track  # noqa: E402
from probfc.features import build_features, usable_rows  # noqa: E402
from probfc.split import split_masks  # noqa: E402

TAUS = (0.10, 0.50, 0.90)


def shap_for(booster, X):
    contrib = booster.predict(X, pred_contrib=True)          # [n, n_features + 1]; last column is the base value
    return contrib[:, :-1], contrib[:, -1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final", action="store_true", help="explain rows from the test period instead of validation")
    ap.add_argument("--n", type=int, default=4000, help="rows to explain (random sample, fixed seed)")
    a = ap.parse_args()
    cfg, tag = load_config(), ("test" if a.final else "val")
    plots.set_style()
    results = resolve_path(cfg, "results")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    rows = []
    for r, track in enumerate(("solar", "wind")):
        table = build_features(load_track(track, cfg), track, cfg)
        table = table[usable_rows(table)].reset_index(drop=True)
        masks = split_masks(table["timestamp"], track, cfg)
        d = table[masks["test" if a.final else "val"]]
        if track == "solar":
            d = d[d["sun_up"]]
        d = d.sample(min(a.n, len(d)), random_state=cfg["seed"])
        for c, tau in enumerate(TAUS):
            path = results / "models" / f"lgbm_{track}" / f"q{int(round(tau * 100)):02d}.txt"
            b = lgb.Booster(model_file=str(path))
            names = b.feature_name()
            X = d[[n for n in names if n != "zone"]].copy()
            X["zone"] = pd.Categorical(d["zone"], categories=sorted(table["zone"].unique()))
            X = X[names]
            sv, base = shap_for(b, X)
            imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
            for f, v in imp.items():
                rows.append({"track": track, "quantile": tau, "feature": f, "mean_abs_shap": v})
            top = imp.head(10)[::-1]
            ax = axes[r, c]
            ax.barh(top.index, top.values, color=plots.BLUE, height=0.65)
            ax.set_title(f"{track.capitalize()}, {int(tau * 100)}th percentile model")
            ax.set_xlabel("Mean |SHAP| (change in forecast, fraction of capacity)")
            ax.grid(axis="y", visible=False)
    out = pd.DataFrame(rows)
    out.to_csv(results / f"shap_importance_{tag}.csv", index=False)
    fig.suptitle("What moves the forecast: top inputs by average SHAP impact", x=0.01, ha="left", fontsize=11)
    print("figure ->", plots.save(fig, f"shap_importance_{tag}.png", cfg))
    for (track, tau), g in out.groupby(["track", "quantile"]):
        print(track, tau, g.sort_values("mean_abs_shap", ascending=False).head(5)["feature"].tolist())


if __name__ == "__main__":
    main()