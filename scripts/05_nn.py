"""Train the PyTorch quantile network. Runs on a laptop CPU, Colab or Kaggle (auto-detects GPU/MPS).

Default: small tuning grid on VALIDATION (1 seed each), then train N seeds with the best setting and score validation.
--final: refit on train+val for the saved number of epochs and score the TEST period (run once, at the end).
Outputs are the same files as the other models, so evaluation/plots treat all three models alike.
"""
import argparse
import os
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # tolerate two OpenMP runtimes on macOS

import torch  # noqa: E402,F401  (import torch first, before pandas/numpy load their own libraries)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from probfc import metrics as M  # noqa: E402
from probfc.config import load_config, resolve_path  # noqa: E402
from probfc.data import download_raw, load_track  # noqa: E402
from probfc.features import build_features, model_columns, usable_rows  # noqa: E402
from probfc.models.nn_qr import NNQuantile, pick_device  # noqa: E402
from probfc.postprocess import crossing_rate, postprocess  # noqa: E402
from probfc.split import split_masks  # noqa: E402

GRID = {"hidden": [128, 256], "dropout": [0.1, 0.2], "weight_decay": [1e-4]}


def val_score(model, eval_t, taus, track):
    sun = eval_t["sun_up"].to_numpy() if track == "solar" else None
    raw = model.predict_quantiles(eval_t)
    q = postprocess(raw, sun)
    m = sun if track == "solar" else np.ones(len(eval_t), bool)
    return raw, q, sun, M.pinball(eval_t["y"].to_numpy()[m], q[m], taus)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", nargs="+", default=["solar"], choices=["solar", "wind"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seeds", type=int, default=3, help="networks averaged in the final model")
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"], help="cpu (default, safest); auto = CUDA if available; mps can crash on some Macs")
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works (not for results)")
    a = ap.parse_args()

    cfg = load_config()
    taus = np.asarray(cfg["quantiles"])
    results = resolve_path(cfg, "results")
    (results / "predictions").mkdir(parents=True, exist_ok=True)
    download_raw(cfg)                                   # no-op if the files are already there (Colab/Kaggle: fetches them)
    print("device:", pick_device(a.device))
    tag_suffix = "_quick" if a.quick else ""
    max_ep = 3 if a.quick else a.max_epochs

    for track in a.tracks:
        table = build_features(load_track(track, cfg), track, cfg)
        table = table[usable_rows(table)].reset_index(drop=True)
        masks = split_masks(table["timestamp"], track, cfg)
        cols = model_columns(table, "full", track)
        daylight = track == "solar"
        best_path = results / f"nn_best_{track}.json"
        t0 = time.time()

        if not a.final:
            tr, va = table[masks["train"]], table[masks["val"]].reset_index(drop=True)
            trials = []
            for h, dr, wd in itertools.product(GRID["hidden"], GRID["dropout"], GRID["weight_decay"]):
                hp = {"hidden": h, "dropout": dr, "weight_decay": wd}
                m = NNQuantile(taus, cols, daylight, n_seeds=1, max_epochs=max_ep, device=a.device, **hp).fit(tr, valid=va)
                *_, s = val_score(m, va, taus, track)
                trials.append({"params": hp, "val_pinball": s, "best_epoch": m.best_epochs_[0]})
                print(f"  {track} {hp} -> val pinball {s:.5f}, best epoch {m.best_epochs_[0]} ({time.time() - t0:.0f}s)")
            best = min(trials, key=lambda t: t["val_pinball"])
            model = NNQuantile(taus, cols, daylight, n_seeds=a.seeds, max_epochs=max_ep, device=a.device, **best["params"]).fit(tr, valid=va)
            best["best_epochs_per_seed"] = model.best_epochs_
            if not a.quick:
                best_path.write_text(json.dumps({"best": best, "all_trials": trials}, indent=2))
            eval_t, tag = va, "val"
        else:
            if not best_path.exists():
                sys.exit(f"{best_path} not found: run without --final first")
            best = json.loads(best_path.read_text())["best"]
            epochs = int(round(float(np.mean(best["best_epochs_per_seed"])) * 1.15))
            model = NNQuantile(taus, cols, daylight, n_seeds=a.seeds, max_epochs=max_ep, device=a.device, **best["params"])
            model.fit(table[masks["train"] | masks["val"]], fixed_epochs=epochs)
            eval_t, tag = table[masks["test"]].reset_index(drop=True), "test"

        model.save(results / f"nn_{track}{tag_suffix}.pt")
        pd.DataFrame(model.history_).to_csv(results / f"nn_history_{track}_{tag}{tag_suffix}.csv", index=False)
        raw, q, sun, _ = val_score(model, eval_t, taus, track)
        subsets = {"all_hours": np.ones(len(eval_t), bool)}
        if daylight:
            subsets["daylight"] = sun
        rows = [{"track": track, "model": model.name, "eval": tag, "subset": k, "n": int(m.sum()),
                 "crossing_rate_raw": crossing_rate(raw[sun] if daylight else raw),
                 **M.summarize(eval_t["y"].to_numpy()[m], q[m], taus)} for k, m in subsets.items()]
        out = eval_t[["timestamp", "zone", "y"]].copy()
        out[[f"q{int(round(t * 100)):02d}" for t in taus]] = q
        out.to_parquet(results / "predictions" / f"{track}_neural_net_{tag}{tag_suffix}.parquet", index=False)

        res = pd.DataFrame(rows)
        for ref_name, label in (("climatology", "skill_vs_climatology"), ("quantile_regression", "skill_vs_quantile_regression")):
            f = results / f"metrics_{track}_baselines_{tag}.csv"
            if f.exists():
                base = pd.read_csv(f)
                ref = base[base["model"] == ref_name].set_index("subset")["pinball"]
                res[label] = [M.skill_score(r.pinball, ref[r.subset]) for r in res.itertuples()]
        f = results / f"metrics_{track}_lightgbm_{tag}.csv"
        if f.exists():
            lg = pd.read_csv(f).set_index("subset")["pinball"]
            res["skill_vs_lightgbm"] = [M.skill_score(r.pinball, lg[r.subset]) for r in res.itertuples()]
        path = results / f"metrics_{track}_neural_net_{tag}{tag_suffix}.csv"
        res.round(5).to_csv(path, index=False)
        show = [c for c in ["model", "subset", "pinball", "crps_approx", "coverage_90", "coverage_50", "skill_vs_climatology",
                            "skill_vs_quantile_regression", "skill_vs_lightgbm"] if c in res.columns]
        print(res[show].round(4).to_string(index=False))
        print(f"{track}: done in {(time.time() - t0) / 60:.1f} min -> {path}")


if __name__ == "__main__":
    main()