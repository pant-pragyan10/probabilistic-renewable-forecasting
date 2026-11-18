"""Loading and validating the GEFCom2014 solar / wind tracks.

Tidy format returned by ``load_track``: one row per (zone, hour) with columns
``timestamp, zone, <weather features...>, y`` where ``y`` is the target power scaled to [0, 1].
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path

TRACKS = ("solar", "wind")


def raw_path(cfg: dict, track: str) -> Path:
    return resolve_path(cfg, "raw") / cfg["data"][track]["file"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(cfg: dict, tracks=TRACKS, force: bool = False) -> dict[str, Path]:
    """Download the raw CSVs into paths.raw. Works anywhere with internet (laptop, Colab, Kaggle)."""
    out = {}
    for track in tracks:
        dest = raw_path(cfg, track)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            out[track] = dest
            continue
        url = f"{cfg['data']['base_url']}/{cfg['data'][track]['file']}"
        tmp = dest.parent / (dest.name + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "probfc-data-fetch"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(dest)
        out[track] = dest
    return out


def feature_columns(track: str, cfg: dict) -> list[str]:
    """Names of the raw weather-forecast columns for a track, after renaming."""
    return list(cfg["data"][track]["rename"].values())


def load_track(track: str, cfg: dict, path: str | Path | None = None) -> pd.DataFrame:
    """Read a raw CSV and return the tidy frame, sorted by (zone, timestamp)."""
    spec = cfg["data"][track]
    raw = pd.read_csv(path or raw_path(cfg, track))
    raw["TIMESTAMP"] = pd.to_datetime(raw["TIMESTAMP"])

    zone_cols = [f"ZONE_{i}" for i in range(1, spec["n_zones"] + 1)]
    missing = [c for c in zone_cols + [spec["target"]] + list(spec["rename"]) if c not in raw.columns]
    if missing:
        raise ValueError(f"{track}: columns missing from CSV: {missing}")

    onehot = raw[zone_cols].to_numpy()
    if not (onehot.sum(axis=1) == 1).all():
        raise ValueError(f"{track}: zone one-hot columns do not sum to exactly 1 on every row")

    df = raw.drop(columns=zone_cols).rename(
        columns={"TIMESTAMP": "timestamp", spec["target"]: "y", **spec["rename"]}
    )
    df["zone"] = onehot.argmax(axis=1) + 1
    cols = ["timestamp", "zone"] + feature_columns(track, cfg) + ["y"]
    return df[cols].sort_values(["zone", "timestamp"]).reset_index(drop=True)


def sun_up_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Solar only. True when the sun is up, derived from the top-of-atmosphere radiation *forecast*."""
    return df["tsr"] > cfg["sun_up_threshold"]


def validate_track(df: pd.DataFrame, track: str, cfg: dict) -> list[dict]:
    """Run integrity checks. Returns a list of {check, passed, detail} dicts (never raises on bad data)."""
    spec = cfg["data"][track]
    feats = feature_columns(track, cfg)
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})

    need = ["timestamp", "zone", "y"] + feats
    absent = [c for c in need if c not in df.columns]
    add("required columns present", not absent, f"missing: {absent}" if absent else f"{len(need)} columns")
    if absent:
        return checks

    zones = sorted(df["zone"].unique().tolist())
    add("zone ids are 1..n_zones", zones == list(range(1, spec["n_zones"] + 1)), f"found {zones}")

    counts = df.groupby("zone").size()
    add(
        "rows per zone as expected",
        (counts == spec["rows_per_zone"]).all(),
        f"expected {spec['rows_per_zone']}, found {counts.to_dict()}",
    )
    add("no missing values", int(df[need].isna().sum().sum()) == 0, f"{int(df[need].isna().sum().sum())} NaN")

    num = df[feats + ["y"]].to_numpy(dtype=float)
    add("all numeric values finite", bool(np.isfinite(num).all()))

    hourly_ok, unique_ok = True, True
    ranges = set()
    for _, g in df.groupby("zone"):
        ts = g["timestamp"]
        unique_ok &= bool(ts.is_unique)
        hourly_ok &= bool((ts.diff().dropna() == pd.Timedelta("1h")).all())
        ranges.add((ts.min(), ts.max()))
    add("timestamps unique within each zone", unique_ok)
    add("timestamps strictly hourly (no gaps) in each zone", hourly_ok)
    add("all zones cover the same time range", len(ranges) == 1)

    start, end = pd.Timestamp(spec["start"]), pd.Timestamp(spec["end"])
    add(
        "date range as expected",
        df["timestamp"].min() == start and df["timestamp"].max() == end,
        f"found {df['timestamp'].min()} -> {df['timestamp'].max()}",
    )

    ymin, ymax = float(df["y"].min()), float(df["y"].max())
    add("target within [0, 1.01]", ymin >= 0.0 and ymax <= 1.01, f"min={ymin:.4f}, max={ymax:.5f}")
    return checks
