"""Compare all models: results table, reliability diagrams, fan charts, performance by hour / month / zone.
Reads saved predictions only. Use --final after the test-set runs to produce the test versions."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from probfc import metrics as M  # noqa: E402
from probfc import plots  # noqa: E402
from probfc.config import load_config, resolve_path  # noqa: E402
from probfc.evaluation import LABELS, MODELS, load_all, pinball_by, results_table  # noqa: E402

COLORS = {"quantile_regression": plots.BLUE, "lightgbm": plots.ORANGE, "neural_net": plots.AQUA, "climatology": plots.INK_2}
STYLE = {"climatology": dict(ls="--", lw=1.6)}
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def draw(ax, x, df, **kw):
    for m in MODELS:
        ax.plot(x if x is not None else df.index, df[m], color=COLORS[m], label=LABELS[m], **STYLE.get(m, {}), **kw)


def fig_reliability(data, taus, tag, cfg):
    fig, axes = plt.subplots(1, len(data), figsize=(4.6 * len(data), 4.2), layout="constrained", sharey=True)
    for ax, (track, (base, Q, mask)) in zip(np.atleast_1d(axes), data.items()):
        ax.plot([0, 1], [0, 1], color=plots.GRID, lw=2.4, zorder=1)
        for m in MODELS:
            ax.plot(taus, M.calibration(base["y"].to_numpy()[mask], Q[m][mask], taus), color=COLORS[m], label=LABELS[m],
                    marker="o", ms=3.5, **STYLE.get(m, {}))
        ax.set_title(f"{track.capitalize()}{' (daylight hours)' if track == 'solar' else ''}")
        ax.set_xlabel("Nominal quantile level")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    np.atleast_1d(axes)[0].set_ylabel("Share of outcomes below the forecast quantile")
    np.atleast_1d(axes)[-1].legend(loc="lower right")
    fig.suptitle("Reliability: closer to the diagonal means the ranges are honest", x=0.01, ha="left", fontsize=11)
    return plots.save(fig, f"eval_reliability_{tag}.png", cfg)


def fig_fan(track, base, Q, taus, model, tag, cfg, zone=1, days=6):
    ts = base["timestamp"]
    z = base[base["zone"] == zone]
    t0 = z["timestamp"].min() + (z["timestamp"].max() - z["timestamp"].min()) * 0.45
    sel = z[(z["timestamp"] >= t0.normalize()) & (z["timestamp"] < t0.normalize() + pd.Timedelta(days=days))]
    idx, q = sel.index.to_numpy(), Q[model][sel.index.to_numpy()]
    col = lambda t: int(np.argmin(np.abs(np.asarray(taus) - t)))
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    x = sel["timestamp"]
    ax.fill_between(x, q[:, col(0.05)], q[:, col(0.95)], color=plots.SEQ_BLUE[0], linewidth=0, label="90% range")
    ax.fill_between(x, q[:, col(0.25)], q[:, col(0.75)], color=plots.SEQ_BLUE[2], linewidth=0, label="50% range")
    ax.plot(x, q[:, col(0.5)], color=plots.SEQ_BLUE[5], lw=1.6, label="Median forecast")
    ax.plot(x, sel["y"], color=plots.INK, lw=1.4, label="Actual")
    ax.set_ylabel("Power (fraction of capacity)")
    ax.set_title(f"{track.capitalize()}, zone {zone}: {LABELS[model]} forecast range vs. what happened")
    ax.legend(loc="upper right", ncol=4)
    ax.set_ylim(-0.02, 1.15)
    return plots.save(fig, f"eval_fan_{track}_{tag}.png", cfg)


def fig_slices(parts, tag, cfg):
    n = len(parts)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.9), layout="constrained")
    for ax, (title, df, xlabel, ticks) in zip(np.atleast_1d(axes), parts):
        draw(ax, df.index.to_numpy(), df, marker="o", ms=3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ticks is not None:
            ax.set_xticks(list(ticks.keys()))
            ax.set_xticklabels(list(ticks.values()))
    np.atleast_1d(axes)[0].set_ylabel("Mean pinball loss (lower is better)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4)
    return plots.save(fig, f"eval_slices_{tag}.png", cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final", action="store_true", help="use the test-period predictions")
    a = ap.parse_args()
    cfg, tag = load_config(), ("test" if a.final else "val")
    taus = np.asarray(cfg["quantiles"])
    plots.set_style()
    results = resolve_path(cfg, "results")

    data, tables, parts = {}, [], []
    for track in ("solar", "wind"):
        base, Q = load_all(cfg, track, tag)
        mask = base["sun_up"].to_numpy() if track == "solar" else np.ones(len(base), bool)
        data[track] = (base, Q, mask)
        tables.append(results_table(base, Q, taus, track, tag))

        month = base["timestamp"].dt.month
        by_month = pinball_by(base, Q, taus, month, mask)
        by_month.to_csv(results / f"slice_{track}_by_month_{tag}.csv")
        pinball_by(base, Q, taus, "zone", mask).to_csv(results / f"slice_{track}_by_zone_{tag}.csv")
        if track == "solar":
            by_hour = pinball_by(base, Q, taus, base["timestamp"].dt.hour, mask)
            by_hour.to_csv(results / f"slice_solar_by_hour_{tag}.csv")
            parts.append(("Solar by hour of day (daylight)", by_hour, "Hour of day", {h: str(h) for h in range(6, 20, 2)}))
        parts.append((f"{track.capitalize()} by month", by_month, "Month", {m: MONTHS[m - 1] for m in by_month.index}))
        print("figure ->", fig_fan(track, base, Q, taus, "neural_net", tag, cfg))

    table = pd.concat(tables, ignore_index=True)
    table.round(5).to_csv(results / f"results_table_{tag}.csv", index=False)
    show = table[["track", "subset", "model", "pinball", "crps_approx", "coverage_90", "coverage_50", "width_90", "skill_vs_climatology"]]
    try:
        text = show.round(4).to_markdown(index=False)       # needs the optional `tabulate` package
    except ImportError:
        text = show.round(4).to_string(index=False)
    (results / f"results_table_{tag}.md").write_text(text)
    print(show.round(4).to_string(index=False))
    print("figure ->", fig_reliability(data, taus, tag, cfg))
    print("figure ->", fig_slices(parts, tag, cfg))


if __name__ == "__main__":
    main()