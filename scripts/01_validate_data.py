"""Load both tracks, run integrity checks, write cleaned parquet files and a data report.

Exits with a non-zero status if any check fails, so later steps never run on bad data.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from probfc.config import load_config, resolve_path  # noqa: E402
from probfc.data import TRACKS, load_track, sun_up_mask, validate_track  # noqa: E402
from probfc.split import assert_valid_split, split_masks  # noqa: E402


def main() -> int:
    cfg = load_config()
    processed = resolve_path(cfg, "processed")
    results = resolve_path(cfg, "results")
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    report, md, all_ok = {}, ["# Data validation report", ""], True
    for track in TRACKS:
        df = load_track(track, cfg)
        checks = validate_track(df, track, cfg)
        masks = split_masks(df["timestamp"], track, cfg)
        assert_valid_split(df["timestamp"], masks)
        ok = all(c["passed"] for c in checks)
        all_ok &= ok

        stats = {
            "rows": int(len(df)),
            "zones": int(df["zone"].nunique()),
            "period": [str(df["timestamp"].min()), str(df["timestamp"].max())],
            "target_mean": round(float(df["y"].mean()), 4),
            "share_target_exactly_zero": round(float((df["y"] == 0).mean()), 4),
            "split_rows": {k: int(m.sum()) for k, m in masks.items()},
        }
        if track == "solar":
            stats["share_sun_down"] = round(float(1 - sun_up_mask(df, cfg).mean()), 4)
        report[track] = {"passed": ok, "checks": checks, "stats": stats}

        df.to_parquet(processed / f"{track}_clean.parquet", index=False)

        md += [f"## {track} — {'PASS' if ok else 'FAIL'}", "", "| check | result | detail |", "|---|---|---|"]
        md += [f"| {c['check']} | {'pass' if c['passed'] else '**FAIL**'} | {c['detail']} |" for c in checks]
        md += ["", "```", json.dumps(stats, indent=2), "```", ""]
        print(f"[{track}] {'PASS' if ok else 'FAIL'}  rows={stats['rows']:,}  splits={stats['split_rows']}")
        for c in checks:
            if not c["passed"]:
                print(f"   FAILED: {c['check']} -> {c['detail']}")

    (results / "data_report.json").write_text(json.dumps(report, indent=2))
    (results / "data_report.md").write_text("\n".join(md))
    print(f"report -> {results / 'data_report.md'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
