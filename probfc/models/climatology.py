"""Reference model: for each zone and hour of day, the empirical quantiles of the training power.
No weather information at all, so every real model should beat it."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Climatology:
    name = "climatology"

    def __init__(self, taus):
        self.taus = np.asarray(taus, float)

    def fit(self, table: pd.DataFrame, **_):
        g = table.groupby(["zone", table["timestamp"].dt.hour])["y"]
        self.table_ = {k: np.quantile(v.to_numpy(), self.taus) for k, v in g}
        self.fallback_ = np.quantile(table["y"].to_numpy(), self.taus)
        return self

    def predict_quantiles(self, table: pd.DataFrame) -> np.ndarray:
        keys = zip(table["zone"].to_numpy(), table["timestamp"].dt.hour.to_numpy())
        return np.vstack([self.table_.get(k, self.fallback_) for k in keys])
