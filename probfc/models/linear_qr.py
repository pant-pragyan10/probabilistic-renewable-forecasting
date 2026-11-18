"""Quantile Regression baseline: one linear model per (zone, quantile) using statsmodels QuantReg.
Solar is fitted on sun-up hours only (night is set to zero in post-processing)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


class LinearQuantileRegression:
    name = "quantile_regression"

    def __init__(self, taus, columns, daylight_only: bool = False, max_iter: int = 5000):
        self.taus, self.columns = np.asarray(taus, float), list(columns)
        self.daylight_only, self.max_iter = daylight_only, max_iter

    def _x(self, d, mu, sd):
        return sm.add_constant((d[self.columns].to_numpy(float) - mu) / sd, has_constant="add")

    def fit(self, table: pd.DataFrame, **_):
        self.coef_ = {}
        for z, d in table.groupby("zone"):
            if self.daylight_only:
                d = d[d["sun_up"]]
            mu, sd = d[self.columns].mean().to_numpy(), d[self.columns].std().replace(0, 1).to_numpy()
            X, y = self._x(d, mu, sd), d["y"].to_numpy()
            betas = []
            for tau in self.taus:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    betas.append(sm.QuantReg(y, X).fit(q=tau, max_iter=self.max_iter).params)
            self.coef_[z] = (mu, sd, np.vstack(betas))
        return self

    def predict_quantiles(self, table: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(table), len(self.taus)))
        zones = table["zone"].to_numpy()
        for z, (mu, sd, B) in self.coef_.items():
            idx = np.where(zones == z)[0]
            out[idx] = self._x(table.iloc[idx], mu, sd) @ B.T
        return out
