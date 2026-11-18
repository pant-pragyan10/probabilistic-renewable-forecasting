"""Run the whole pipeline in order, stopping at the first failure.

    python scripts/run_all.py            # data -> validation -> EDA -> models (tuned on validation) -> evaluation -> SHAP
    python scripts/run_all.py --final    # the test-period run: refit on train+val, score the test set ONCE

Do the plain run first and only use --final when you are done changing anything: the test set is for one honest look.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

PLAIN = [
    ("download data", ["scripts/00_get_data.py"]),
    ("validate data", ["scripts/01_validate_data.py"]),
    ("EDA figures", ["scripts/02_eda.py"]),
    ("baselines: climatology + quantile regression", ["scripts/03_baselines.py", "--tracks", "solar", "wind"]),
    ("LightGBM (tuning on validation)", ["scripts/04_lightgbm.py", "--tracks", "solar", "wind"]),
    ("neural network (tuning on validation)", ["scripts/05_nn.py", "--tracks", "solar", "wind"]),
    ("evaluation", ["scripts/06_evaluate.py"]),
    ("SHAP", ["scripts/07_shap.py"]),
    ("margin comparison", ["scripts/08_usefulness.py"]),
]
FINAL = [
    ("baselines on test", ["scripts/03_baselines.py", "--tracks", "solar", "wind", "--final"]),
    ("LightGBM on test", ["scripts/04_lightgbm.py", "--tracks", "solar", "wind", "--final"]),
    ("neural network on test", ["scripts/05_nn.py", "--tracks", "solar", "wind", "--final"]),
    ("evaluation on test", ["scripts/06_evaluate.py", "--final"]),
    ("SHAP on test", ["scripts/07_shap.py", "--final"]),
    ("margin comparison on test", ["scripts/08_usefulness.py", "--final"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="run the test-set steps")
    ap.add_argument("--tests", action="store_true", help="also run pytest first")
    a = ap.parse_args()
    steps = ([("unit tests", ["-m", "pytest", "-q"])] if a.tests else []) + (FINAL if a.final else PLAIN)
    t_all = time.time()
    for i, (name, cmd) in enumerate(steps, 1):
        print(f"\n=== [{i}/{len(steps)}] {name} ===", flush=True)
        t0 = time.time()
        r = subprocess.run([PY, *cmd], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"\nStep '{name}' failed (exit {r.returncode}). Fix it and run again.")
        print(f"--- done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nAll {len(steps)} steps finished in {(time.time() - t_all) / 60:.1f} min.")


if __name__ == "__main__":
    main()