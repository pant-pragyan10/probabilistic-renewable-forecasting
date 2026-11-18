"""Fit climatology + Quantile Regression on the training period and score them on VALIDATION.

The test period is only touched with --final (refit on train+val, evaluate test once)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from probfc import metrics as M  # noqa: E402
from probfc.config import load_config, resolve_path  # noqa: E402
from probfc.data import load_track  # noqa: E402
from probfc.features import build_features, model_columns, usable_rows  # noqa: E402
from probfc.models import Climatology, LinearQuantileRegression  # noqa: E402
from probfc.postprocess import crossing_rate, postprocess  # noqa: E402
from probfc.split import split_masks  # noqa: E402


def make_models(track, taus, table):
    return [
        Climatology(taus),
        LinearQuantileRegression(taus, model_columns(table, "linear", track), daylight_only=(track == "solar")),
    ]


def evaluate(track, cfg, final):
    taus = np.asarray(cfg["quantiles"])
    df = load_track(track, cfg)
    table = build_features(df, track, cfg)
    table = table[usable_rows(table)].reset_index(drop=True)
    masks = split_masks(table["timestamp"], track, cfg)
    if final:
        fit_idx, eval_idx, tag = masks["train"] | masks["val"], masks["test"], "test"
    else:
        fit_idx, eval_idx, tag = masks["train"], masks["val"], "val"
    fit_t, eval_t = table[fit_idx], table[eval_idx].reset_index(drop=True)
    sun = eval_t["sun_up"].to_numpy() if track == "solar" else None

    rows, preds_dir = [], resolve_path(cfg, "results") / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for model in make_models(track, taus, table):
        t0 = time.time()
        model.fit(fit_t)
        raw = model.predict_quantiles(eval_t)
        q = postprocess(raw, sun)
        subsets = {"all_hours": np.ones(len(eval_t), bool)}
        if track == "solar":
            subsets["daylight"] = sun
        for name, m in subsets.items():
            r = {"track": track, "model": model.name, "eval": tag, "subset": name, "n": int(m.sum()),
                 "crossing_rate_raw": crossing_rate(raw[sun] if track == "solar" else raw), **M.summarize(eval_t["y"].to_numpy()[m], q[m], taus)}
            rows.append(r)
        out = eval_t[["timestamp", "zone", "y"]].copy()
        out[[f"q{int(round(t * 100)):02d}" for t in taus]] = q
        out.to_parquet(preds_dir / f"{track}_{model.name}_{tag}.parquet", index=False)
        print(f"  {track}/{model.name}: fitted + scored in {time.time() - t0:.1f}s")
    res = pd.DataFrame(rows)
    ref = res[res["model"] == "climatology"].set_index("subset")["pinball"]
    res["skill_vs_climatology"] = [M.skill_score(r.pinball, ref[r.subset]) for r in res.itertuples()]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", nargs="+", default=["solar"], choices=["solar", "wind"])
    ap.add_argument("--final", action="store_true", help="refit on train+val and score the test period")
    a = ap.parse_args()
    cfg = load_config()
    out_dir = resolve_path(cfg, "results")
    for track in a.tracks:
        res = evaluate(track, cfg, a.final)
        path = out_dir / f"metrics_{track}_baselines_{'test' if a.final else 'val'}.csv"
        res.round(5).to_csv(path, index=False)
        show = ["model", "subset", "pinball", "crps_approx", "coverage_90", "coverage_50", "skill_vs_climatology", "crossing_rate_raw"]
        print(res[show].round(4).to_string(index=False))
        print("->", path)


if __name__ == "__main__":
    main()
