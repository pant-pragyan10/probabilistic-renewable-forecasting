"""Convert the trained neural networks (results/nn_*.pt) into plain numpy weights (results/models/nn_*.npz) for the demo app.

Why: the app then needs no PyTorch at all (and cannot clash with LightGBM on a Mac). Run once after scripts/05_nn.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probfc.config import load_config, resolve_path  # noqa: E402
from probfc.whatif import export_nn  # noqa: E402


def main():
    cfg = load_config()
    res = resolve_path(cfg, "results")
    (res / "models").mkdir(parents=True, exist_ok=True)
    done = 0
    for track in ("solar", "wind"):
        src = res / f"nn_{track}.pt"
        if not src.exists():
            print(f"skip {track}: {src.name} not found (run scripts/05_nn.py first)")
            continue
        dst = res / "models" / f"nn_{track}.npz"
        export_nn(src, dst)
        print(f"{track}: wrote {dst}")
        done += 1
    print("done" if done else "nothing exported")


if __name__ == "__main__":
    main()