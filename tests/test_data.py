import copy

import numpy as np
import pandas as pd
import pytest

from probfc.config import load_config
from probfc.data import feature_columns, load_track, validate_track


@pytest.fixture()
def small_cfg():
    """The real config shrunk to 2 zones x 48 hours so tests run instantly."""
    cfg = copy.deepcopy(load_config())
    for track in ("solar", "wind"):
        cfg["data"][track].update(n_zones=2, rows_per_zone=48, start="2020-01-01 00:00:00", end="2020-01-02 23:00:00")
    return cfg


def make_df(track, cfg, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01 00:00:00", periods=48, freq="h")
    frames = []
    for z in (1, 2):
        d = pd.DataFrame({"timestamp": ts, "zone": z})
        for c in feature_columns(track, cfg):
            d[c] = rng.normal(size=len(ts))
        d["y"] = rng.uniform(0, 1, size=len(ts))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def failed(checks):
    return {c["check"] for c in checks if not c["passed"]}


@pytest.mark.parametrize("track", ["solar", "wind"])
def test_clean_frame_passes_every_check(track, small_cfg):
    assert failed(validate_track(make_df(track, small_cfg), track, small_cfg)) == set()


def test_missing_hour_is_detected(small_cfg):
    df = make_df("solar", small_cfg)
    df = df.drop(index=10).reset_index(drop=True)
    bad = failed(validate_track(df, "solar", small_cfg))
    assert "timestamps strictly hourly (no gaps) in each zone" in bad
    assert "rows per zone as expected" in bad


def test_nan_is_detected(small_cfg):
    df = make_df("wind", small_cfg)
    df.loc[3, "u10"] = np.nan
    assert "no missing values" in failed(validate_track(df, "wind", small_cfg))


def test_target_out_of_range_is_detected(small_cfg):
    df = make_df("solar", small_cfg)
    df.loc[5, "y"] = 1.5
    assert "target within [0, 1.01]" in failed(validate_track(df, "solar", small_cfg))


def test_duplicate_timestamp_is_detected(small_cfg):
    df = make_df("solar", small_cfg)
    df.loc[7, "timestamp"] = df.loc[6, "timestamp"]
    assert "timestamps unique within each zone" in failed(validate_track(df, "solar", small_cfg))


def test_missing_column_is_reported_not_raised(small_cfg):
    df = make_df("solar", small_cfg).drop(columns=["tsr"])
    assert failed(validate_track(df, "solar", small_cfg)) == {"required columns present"}


def test_load_track_builds_zone_from_onehot_and_renames(tmp_path, small_cfg):
    spec = small_cfg["data"]["wind"]
    spec["n_zones"] = 2
    csv = pd.DataFrame(
        {
            "TIMESTAMP": ["2020-01-01 01:00:00", "2020-01-01 00:00:00", "2020-01-01 00:00:00"],
            "TARGETVAR": [0.2, 0.5, 0.7],
            "U10": [1.0, 2.0, 3.0], "V10": [1.0, 2.0, 3.0], "U100": [1.0, 2.0, 3.0], "V100": [1.0, 2.0, 3.0],
            "ZONE_1": [1, 1, 0], "ZONE_2": [0, 0, 1],
        }
    )
    p = tmp_path / "wind.csv"
    csv.to_csv(p, index=False)
    df = load_track("wind", small_cfg, path=p)
    assert list(df.columns) == ["timestamp", "zone", "u10", "v10", "u100", "v100", "y"]
    assert df["zone"].tolist() == [1, 1, 2]                      # sorted by (zone, timestamp)
    assert df["timestamp"].is_monotonic_increasing is False      # zone 2 restarts the clock
    assert df.loc[0, "y"] == 0.5 and df.loc[1, "y"] == 0.2


def test_load_track_rejects_bad_onehot(tmp_path, small_cfg):
    csv = pd.DataFrame(
        {"TIMESTAMP": ["2020-01-01 00:00:00"], "TARGETVAR": [0.5], "U10": [1.0], "V10": [1.0],
         "U100": [1.0], "V100": [1.0], "ZONE_1": [1], "ZONE_2": [1]}
    )
    p = tmp_path / "wind.csv"
    csv.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_track("wind", small_cfg, path=p)
