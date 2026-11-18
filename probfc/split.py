"""Chronological train / validation / test split. Never shuffles across time."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_masks(timestamps: pd.Series, track: str, cfg: dict) -> dict[str, np.ndarray]:
    """Boolean masks for train / val / test using half-open boundaries from the config.

    train: ts < train_end;  val: train_end <= ts < val_end;  test: ts >= val_end.
    """
    s = cfg["splits"][track]
    t_train_end, t_val_end = pd.Timestamp(s["train_end"]), pd.Timestamp(s["val_end"])
    ts = pd.to_datetime(timestamps)
    return {
        "train": (ts < t_train_end).to_numpy(),
        "val": ((ts >= t_train_end) & (ts < t_val_end)).to_numpy(),
        "test": (ts >= t_val_end).to_numpy(),
    }


def assert_valid_split(timestamps: pd.Series, masks: dict[str, np.ndarray]) -> None:
    """Raise AssertionError unless the split is non-empty, disjoint, exhaustive and time-ordered."""
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    total = sum(m.sum() for m in masks.values())
    assert total == len(ts), "split masks must cover every row exactly once"
    assert (sum(m.astype(int) for m in masks.values()) == 1).all(), "split masks overlap"
    for name, m in masks.items():
        assert m.sum() > 0, f"{name} split is empty"
    assert ts[masks["train"]].max() < ts[masks["val"]].min(), "train must end before validation starts"
    assert ts[masks["val"]].max() < ts[masks["test"]].min(), "validation must end before test starts"
