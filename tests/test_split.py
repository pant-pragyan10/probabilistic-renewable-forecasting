import numpy as np
import pandas as pd
import pytest

from probfc.config import load_config
from probfc.split import assert_valid_split, split_masks

CFG = load_config()


def hourly(start, end):
    return pd.Series(pd.date_range(start, end, freq="h"))


@pytest.mark.parametrize(
    "track,start,end", [("solar", "2012-04-02 01:00", "2014-07-01 00:00"), ("wind", "2012-01-02 00:00", "2013-12-31 23:00")]
)
def test_real_config_gives_a_valid_chronological_split(track, start, end):
    ts = hourly(start, end)
    masks = split_masks(ts, track, CFG)
    assert_valid_split(ts, masks)
    assert all(m.sum() > 0 for m in masks.values())


def test_boundary_timestamp_belongs_to_the_later_split():
    ts = pd.Series(pd.to_datetime(["2013-06-30 23:00", "2013-07-01 00:00", "2013-12-31 23:00", "2014-01-01 00:00"]))
    m = split_masks(ts, "solar", CFG)
    assert m["train"].tolist() == [True, False, False, False]
    assert m["val"].tolist() == [False, True, True, False]
    assert m["test"].tolist() == [False, False, False, True]


def test_split_works_on_multi_zone_frames_in_any_row_order():
    ts = pd.concat([hourly("2012-04-02 01:00", "2014-07-01 00:00")] * 3, ignore_index=True)
    ts = ts.sample(frac=1.0, random_state=0).reset_index(drop=True)       # shuffled rows
    masks = split_masks(ts, "solar", CFG)
    assert_valid_split(ts, masks)


def test_overlapping_masks_are_rejected():
    ts = hourly("2014-01-01", "2014-01-10")
    masks = {"train": np.ones(len(ts), bool), "val": np.ones(len(ts), bool), "test": np.zeros(len(ts), bool)}
    with pytest.raises(AssertionError):
        assert_valid_split(ts, masks)


def test_test_before_train_is_rejected():
    ts = hourly("2014-01-01", "2014-01-09")
    n = len(ts)
    masks = {  # deliberately reversed order: "train" is the latest block
        "train": np.arange(n) >= 2 * n // 3,
        "val": (np.arange(n) >= n // 3) & (np.arange(n) < 2 * n // 3),
        "test": np.arange(n) < n // 3,
    }
    with pytest.raises(AssertionError):
        assert_valid_split(ts, masks)
