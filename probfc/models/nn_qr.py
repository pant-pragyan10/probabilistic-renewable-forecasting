"""Neural network quantile regression (PyTorch).

One MLP with a 19-value output head, trained with pinball loss on all quantiles at once.
The head predicts a first value plus positive increments (softplus + cumulative sum), so the
quantiles are ordered by construction. Zone is a learned embedding. Solar is fitted on sun-up hours only.
Several seeds are trained and their quantile predictions averaged.

torch is imported lazily so the rest of the project works without it.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd


def pick_device(prefer=None):
    """CPU by default. This data is small, and Apple's MPS backend can crash (segfault) on some setups.
    Pass 'cuda' (Colab/Kaggle GPU), 'mps' or 'cpu' explicitly, or 'auto' to use CUDA when it is available."""
    import torch

    if prefer in (None, "cpu"):
        return "cpu"
    if prefer == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return prefer


def _build_net(n_in, n_zones, n_q, hidden, depth, dropout, emb_dim=4):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_zones, emb_dim)
            layers, d = [], n_in + emb_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(d, n_q)

        def forward(self, x, z):
            h = self.head(self.body(torch.cat([x, self.emb(z)], dim=1)))
            first, inc = h[:, :1], torch.nn.functional.softplus(h[:, 1:]) * 0.1
            return torch.cat([first, first + torch.cumsum(inc, dim=1)], dim=1)   # non-decreasing in tau

    return Net()


def pinball_torch(pred, y, taus):
    diff = y[:, None] - pred
    return torch_max(taus * diff, (taus - 1.0) * diff).mean()


def torch_max(a, b):
    import torch

    return torch.maximum(a, b)


class NNQuantile:
    name = "neural_net"

    def __init__(self, taus, columns, daylight_only=False, hidden=128, depth=3, dropout=0.1, lr=1e-3,
                 weight_decay=1e-4, batch_size=512, max_epochs=150, patience=12, n_seeds=3, seed=42, device=None):
        self.taus, self.columns, self.daylight_only = np.asarray(taus, float), list(columns), daylight_only
        self.hp = dict(hidden=hidden, depth=depth, dropout=dropout, lr=lr, weight_decay=weight_decay,
                       batch_size=batch_size, max_epochs=max_epochs, patience=patience)
        self.n_seeds, self.seed, self.device = n_seeds, seed, device
        self.nets_, self.best_epochs_, self.history_ = [], [], []

    # -- data ---------------------------------------------------------------
    def _arrays(self, d):
        if self.daylight_only:
            d = d[d["sun_up"]]
        X = (d[self.columns].to_numpy(np.float32) - self.mu_) / self.sd_
        z = d["zone"].map(self.zone_index_).to_numpy(np.int64)
        return X, z, d["y"].to_numpy(np.float32)

    def fit(self, table: pd.DataFrame, valid: pd.DataFrame | None = None, fixed_epochs: int | None = None):
        """With `valid`: early stopping, records best epoch. With `fixed_epochs`: train exactly that long (final refit)."""
        import torch

        if valid is None and fixed_epochs is None:
            raise ValueError("give `valid` (early stopping) or `fixed_epochs`")
        dev = pick_device(self.device)
        src = table[table["sun_up"]] if self.daylight_only else table
        self.mu_ = src[self.columns].mean().to_numpy(np.float32)
        self.sd_ = src[self.columns].std().replace(0, 1).to_numpy(np.float32)
        self.zone_index_ = {z: i for i, z in enumerate(sorted(table["zone"].unique()))}
        X, z, y = self._arrays(table)
        taus_t = torch.tensor(self.taus, dtype=torch.float32, device=dev)
        Xt, zt, yt = (torch.from_numpy(a).to(dev) for a in (X, z, y))
        if valid is not None:
            Xv, zv, yv = (torch.from_numpy(a).to(dev) for a in self._arrays(valid))

        self.nets_, self.best_epochs_, self.history_ = [], [], []
        for s in range(self.n_seeds):
            torch.manual_seed(self.seed + s)
            gen = torch.Generator().manual_seed(self.seed + s)
            net = _build_net(X.shape[1], len(self.zone_index_), len(self.taus), self.hp["hidden"], self.hp["depth"],
                             self.hp["dropout"]).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
            best, best_state, best_ep, bad, hist = np.inf, None, 0, 0, []
            n_ep = fixed_epochs if fixed_epochs is not None else self.hp["max_epochs"]
            for ep in range(1, n_ep + 1):
                net.train()
                perm = torch.randperm(len(Xt), generator=gen).to(dev)
                tot = 0.0
                for i in range(0, len(perm), self.hp["batch_size"]):
                    idx = perm[i:i + self.hp["batch_size"]]
                    loss = pinball_torch(net(Xt[idx], zt[idx]), yt[idx], taus_t)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss.detach()) * len(idx)
                rec = {"seed": s, "epoch": ep, "train_pinball": tot / len(perm)}
                if valid is not None:
                    net.eval()
                    with torch.no_grad():
                        v = float(pinball_torch(net(Xv, zv), yv, taus_t))
                    rec["val_pinball"] = v
                    sched.step(v)
                    if v < best - 1e-6:
                        best, best_state, best_ep, bad = v, copy.deepcopy(net.state_dict()), ep, 0
                    else:
                        bad += 1
                    hist.append(rec)
                    if bad >= self.hp["patience"]:
                        break
                else:
                    hist.append(rec)
            if valid is not None:
                net.load_state_dict(best_state)
                self.best_epochs_.append(best_ep)
            else:
                self.best_epochs_.append(fixed_epochs)
            net.eval()
            self.nets_.append(net.cpu())
            self.history_ += hist
        self._dev = "cpu"
        return self

    def predict_quantiles(self, table: pd.DataFrame) -> np.ndarray:
        import torch

        X = (table[self.columns].to_numpy(np.float32) - self.mu_) / self.sd_
        z = table["zone"].map(self.zone_index_).to_numpy(np.int64)
        Xt, zt = torch.from_numpy(X), torch.from_numpy(z)
        with torch.no_grad():
            preds = [n(Xt, zt).numpy() for n in self.nets_]
        return np.mean(preds, axis=0)                     # average of seeds, per quantile

    def save(self, path) -> None:
        import torch

        torch.save({"states": [n.state_dict() for n in self.nets_], "mu": self.mu_, "sd": self.sd_,
                    "zones": self.zone_index_, "columns": self.columns, "hp": self.hp}, path)