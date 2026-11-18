"""Tune LightGBM on VALIDATION, save the chosen settings, and score it on validation.

--final: refit on train+val with the saved settings and score the TEST period (run once, at the end)."""
import argparse
import itertools
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
from probfc.models import LightGBMQuantile  # noqa: E402
from probfc.postprocess import crossing_rate, postprocess  # noqa: E402
from probfc.split import split_masks  # noqa: E402

GRID = {"num_leaves": [15, 31, 63], "min_child_samples": [20, 100], "learning_rate": [0.05]}


def score(model, eval_t, taus, track):
    sun = eval_t["sun_up"].to_numpy() if track == "solar" else None
    raw = model.predict_quantiles(eval_t)
    q = postprocess(raw, sun)
    m = sun if track == "solar" else np.ones(len(eval_t), bool)       # tuning criterion: daylight (solar) / all (wind)
    return raw, q, sun, M.pinball(eval_t["y"].to_numpy()[m], q[m], taus)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", nargs="+", default=["solar"], choices=["solar", "wind"])
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    taus = np.asarray(cfg["quantiles"])
    results = resolve_path(cfg, "results")
    (results / "predictions").mkdir(parents=True, exist_ok=True)

    for track in a.tracks:
        table = build_features(load_track(track, cfg), track, cfg)
        table = table[usable_rows(table)].reset_index(drop=True)
        masks = split_masks(table["timestamp"], track, cfg)
        cols = model_columns(table, "full", track)
        daylight = track == "solar"
        best_path = results / f"lgbm_best_{track}.json"

        if not a.final:
            tr, va = table[masks["train"]], table[masks["val"]].reset_index(drop=True)
            trials = []
            for nl, mcs, lr in itertools.product(GRID["num_leaves"], GRID["min_child_samples"], GRID["learning_rate"]):
                t0 = time.time()
                params = {"num_leaves": nl, "min_child_samples": mcs, "learning_rate": lr}
                mdl = LightGBMQuantile(taus, cols, params, daylight_only=daylight).fit(tr, valid=va)
                *_, s = score(mdl, va, taus, track)
                trials.append({"params": params, "val_pinball": s, "n_estimators": mdl.best_iters_})
                print(f"  {track} {params} -> val pinball {s:.5f} ({time.time() - t0:.0f}s)")
            best = min(trials, key=lambda t: t["val_pinball"])
            best_path.write_text(json.dumps({"best": best, "all_trials": trials}, indent=2))
            mdl = LightGBMQuantile(taus, cols, best["params"], daylight_only=daylight).fit(tr, valid=va)
            eval_t, tag = va, "val"
        else:
            if not best_path.exists():
                sys.exit(f"{best_path} not found: run without --final first")
            best = json.loads(best_path.read_text())["best"]
            mdl = LightGBMQuantile(taus, cols, best["params"], daylight_only=daylight,
                                   n_estimators=[int(n * 1.15) for n in best["n_estimators"]])  # a bit more data -> slightly more trees
            mdl.fit(table[masks["train"] | masks["val"]])
            eval_t, tag = table[masks["test"]].reset_index(drop=True), "test"

        mdl.save(results / "models" / f"lgbm_{track}")
        raw, q, sun, _ = score(mdl, eval_t, taus, track)
        subsets = {"all_hours": np.ones(len(eval_t), bool)}
        if daylight:
            subsets["daylight"] = sun
        rows = [{"track": track, "model": mdl.name, "eval": tag, "subset": k, "n": int(m.sum()),
                 "crossing_rate_raw": crossing_rate(raw[sun] if daylight else raw),
                 **M.summarize(eval_t["y"].to_numpy()[m], q[m], taus)} for k, m in subsets.items()]
        out = eval_t[["timestamp", "zone", "y"]].copy()
        out[[f"q{int(round(t * 100)):02d}" for t in taus]] = q
        out.to_parquet(results / "predictions" / f"{track}_lightgbm_{tag}.parquet", index=False)

        base = pd.read_csv(results / f"metrics_{track}_baselines_{tag}.csv")
        ref = base[base["model"] == "climatology"].set_index("subset")["pinball"]
        qr = base[base["model"] == "quantile_regression"].set_index("subset")["pinball"]
        res = pd.DataFrame(rows)
        res["skill_vs_climatology"] = [M.skill_score(r.pinball, ref[r.subset]) for r in res.itertuples()]
        res["skill_vs_quantile_regression"] = [M.skill_score(r.pinball, qr[r.subset]) for r in res.itertuples()]
        path = results / f"metrics_{track}_lightgbm_{tag}.csv"
        res.round(5).to_csv(path, index=False)
        print(res[["model", "subset", "pinball", "crps_approx", "coverage_90", "coverage_50",
                   "skill_vs_climatology", "skill_vs_quantile_regression"]].round(4).to_string(index=False))
        print("best params:", best["params"], "-> ", path)


if __name__ == "__main__":
    main()