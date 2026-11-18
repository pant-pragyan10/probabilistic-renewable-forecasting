"""LightGBM quantile regression: one gradient-boosted model per quantile, pooled over all zones
(zone is a categorical feature). Solar is fitted on sun-up hours only (night is zero in post-processing)."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


class LightGBMQuantile:
    name = "lightgbm"

    def __init__(self, taus, columns, params=None, daylight_only=False, n_estimators=None, seed=42):
        """n_estimators: None -> learn it per quantile by early stopping on `valid`; or a list/array (one per tau)."""
        self.taus, self.columns, self.daylight_only, self.seed = np.asarray(taus, float), list(columns), daylight_only, seed
        self.params = {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50, "subsample": 0.8,
                       "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0, **(params or {})}
        self.n_estimators = None if n_estimators is None else list(n_estimators)
        self.boosters_, self.best_iters_ = [], []

    def _xy(self, d):
        if self.daylight_only:
            d = d[d["sun_up"]]
        X = d[self.columns].copy()
        X["zone"] = pd.Categorical(d["zone"], categories=self._zones)
        return X, d["y"].to_numpy()

    def fit(self, table: pd.DataFrame, valid: pd.DataFrame | None = None, max_trees: int = 800, patience: int = 50):
        self._zones = sorted(table["zone"].unique())
        X, y = self._xy(table)
        Xv, yv = self._xy(valid) if valid is not None else (None, None)
        self.boosters_, self.best_iters_ = [], []
        for i, tau in enumerate(self.taus):
            p = {"objective": "quantile", "alpha": float(tau), "verbose": -1, "seed": self.seed, "n_jobs": 2, **self.params}
            if self.n_estimators is None:
                if valid is None:
                    raise ValueError("pass `valid` to learn n_estimators, or give n_estimators explicitly")
                b = lgb.train(p, lgb.Dataset(X, y), num_boost_round=max_trees, valid_sets=[lgb.Dataset(Xv, yv)],
                              callbacks=[lgb.early_stopping(patience, verbose=False)])
                self.best_iters_.append(int(b.best_iteration))
            else:
                b = lgb.train(p, lgb.Dataset(X, y), num_boost_round=int(self.n_estimators[i]))
                self.best_iters_.append(int(self.n_estimators[i]))
            self.boosters_.append(b)
        return self

    def predict_quantiles(self, table: pd.DataFrame) -> np.ndarray:
        X = table[self.columns].copy()
        X["zone"] = pd.Categorical(table["zone"], categories=self._zones)
        return np.column_stack([b.predict(X, num_iteration=n) for b, n in zip(self.boosters_, self.best_iters_)])

    def save(self, folder) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for tau, b in zip(self.taus, self.boosters_):
            b.save_model(str(folder / f"q{int(round(tau * 100)):02d}.txt"), num_iteration=None)