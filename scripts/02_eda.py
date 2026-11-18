"""Exploratory figures + a small JSON of headline numbers. Reads the raw CSVs via the loader."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from probfc import plots  # noqa: E402
from probfc.config import load_config, resolve_path  # noqa: E402
from probfc.data import load_track, sun_up_mask  # noqa: E402
from probfc.split import split_masks  # noqa: E402


def fig_solar_diurnal(solar, cfg):
    hod = solar.assign(hour=solar["timestamp"].dt.hour).groupby(["zone", "hour"])["y"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for z, col in zip(hod.columns, plots.SERIES):
        ax.plot(hod.index, hod[z], color=col, label=f"Zone {z}")
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day (timestamp clock)")
    ax.set_ylabel("Mean power (fraction of capacity)")
    ax.set_title("Solar output follows the sun: zero overnight, peak around midday")
    ax.legend(loc="upper right")
    return plots.save(fig, "eda_solar_diurnal.png", cfg)


def fig_solar_vs_radiation(solar, cfg):
    day = solar[solar["sun_up"]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True, sharey=True, layout="constrained")
    hb = None
    for ax, (z, g) in zip(axes, day.groupby("zone")):
        hb = ax.hexbin(g["ssrd"], g["y"], gridsize=40, cmap=plots.SEQ_CMAP, mincnt=1, linewidths=0, bins="log")
        ax.set_title(f"Zone {z}")
        ax.set_xlabel("Forecast surface solar radiation (W/m²)")
    axes[0].set_ylabel("Actual power (fraction of capacity)")
    fig.colorbar(hb, ax=axes.ravel().tolist(), shrink=0.8, label="hours (log scale)")
    fig.suptitle("Solar power vs. forecast radiation (sun-up hours): strong signal, visible scatter", x=0.01, ha="left")
    return plots.save(fig, "eda_solar_power_vs_radiation.png", cfg)


def fig_wind_power_curve(wind, cfg):
    w = wind.assign(ws100=np.hypot(wind["u100"], wind["v100"]))
    bins = np.linspace(0, w["ws100"].quantile(0.995), 26)
    w["bin"] = pd.cut(w["ws100"], bins)
    q = w.groupby("bin", observed=True).agg(
        x=("ws100", "mean"),
        p10=("y", lambda s: s.quantile(0.10)),
        p50=("y", "median"),
        p90=("y", lambda s: s.quantile(0.90)),
        n=("y", "size"),
    )
    q = q[q["n"] >= 800]  # drop sparse high-wind bins whose percentiles are noise
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    hb = ax.hexbin(w["ws100"], w["y"], gridsize=50, cmap=plots.SEQ_CMAP, mincnt=1, linewidths=0, bins="log")
    ax.plot(q["x"], q["p90"], color=plots.ORANGE, lw=1.6, ls="--")
    ax.plot(q["x"], q["p50"], color=plots.ORANGE, lw=2.2)
    ax.plot(q["x"], q["p10"], color=plots.ORANGE, lw=1.6, ls="--")
    for key, text in (("p90", "90th percentile"), ("p50", "median"), ("p10", "10th percentile")):
        ax.annotate(text, (q["x"].iloc[-1], q[key].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=plots.INK_2, va="center", fontsize=8.5)
    ax.set_xlim(0, q["x"].iloc[-1] * 1.22)
    ax.set_xlabel("Forecast wind speed at 100 m (m/s)")
    ax.set_ylabel("Actual power (fraction of capacity)")
    ax.set_title("Same forecast wind speed, very different output: the spread a point forecast hides")
    fig.colorbar(hb, ax=ax, shrink=0.8, label="hours (log scale)")
    return plots.save(fig, "eda_wind_power_curve.png", cfg)


def fig_monthly(solar, wind, cfg):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    ms = solar.assign(m=solar["timestamp"].dt.month).groupby(["zone", "m"])["y"].mean().unstack(0)
    for z, col in zip(ms.columns, plots.SERIES):
        a1.plot(ms.index, ms[z], color=col, label=f"Zone {z}")
    a1.set_title("Solar: mean power by calendar month")
    a1.legend(loc="best")
    mw = wind.assign(m=wind["timestamp"].dt.month).groupby(["zone", "m"])["y"].mean().unstack(0)
    a2.fill_between(mw.index, mw.min(axis=1), mw.max(axis=1), color=plots.SEQ_BLUE[1], alpha=0.6, linewidth=0,
                    label="range across the 10 zones")
    a2.plot(mw.index, mw.mean(axis=1), color=plots.BLUE, label="mean of zones")
    a2.set_title("Wind: mean power by calendar month")
    a2.legend(loc="best")
    for ax in (a1, a2):
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Month")
    a1.set_ylabel("Mean power (fraction of capacity)")
    return plots.save(fig, "eda_monthly_mean.png", cfg)


def fig_splits(frames, cfg):
    fig, ax = plt.subplots(figsize=(9, 2.9), layout="constrained")
    colors = {"Train": plots.BLUE, "Validation": plots.ORANGE, "Test": plots.AQUA}
    for row, (track, df) in enumerate(frames.items()):
        s = cfg["splits"][track]
        t0, t1, t2, t3 = df["timestamp"].min(), pd.Timestamp(s["train_end"]), pd.Timestamp(s["val_end"]), df["timestamp"].max()
        for name, (a, b) in zip(colors, ((t0, t1), (t1, t2), (t2, t3))):
            x0, w = mdates.date2num(a), (b - a).days
            ax.barh(row, w, left=x0, height=0.5, color=colors[name], edgecolor=plots.SURFACE, linewidth=2)
            ax.text(x0 + w / 2, row - 0.3, f"{name}  ({round(w / 30.44)} mo)", ha="center", va="bottom",
                    color=plots.INK_2, fontsize=8.5)
    ax.set_yticks(range(len(frames)))
    ax.set_yticklabels([k.capitalize() for k in frames])
    ax.set_ylim(len(frames) - 0.5, -0.65)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3)
    ax.set_title("Chronological split: models never see the future")
    return plots.save(fig, "eda_splits.png", cfg)


def fig_target_dist(solar, wind, cfg):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    a1.hist(solar.loc[solar["sun_up"], "y"], bins=50, color=plots.BLUE, edgecolor=plots.SURFACE, linewidth=0.5)
    a1.set_title("Solar power, sun-up hours only")
    a2.hist(wind["y"], bins=50, color=plots.BLUE, edgecolor=plots.SURFACE, linewidth=0.5)
    a2.set_title("Wind power, all hours")
    for ax in (a1, a2):
        ax.set_xlabel("Power (fraction of capacity)")
    a1.set_ylabel("Hours")
    return plots.save(fig, "eda_target_distribution.png", cfg)


def main() -> None:
    cfg = load_config()
    plots.set_style()
    solar, wind = load_track("solar", cfg), load_track("wind", cfg)
    solar["sun_up"] = sun_up_mask(solar, cfg)

    results = resolve_path(cfg, "results")
    results.mkdir(parents=True, exist_ok=True)

    feats = [c for c in solar.columns if c not in ("timestamp", "zone", "y", "sun_up")]
    var = solar[feats].describe().T[["min", "mean", "max"]]
    var["corr_with_power"] = [solar[c].corr(solar["y"]) for c in feats]
    var.round(4).to_csv(results / "solar_variable_summary.csv")

    ws100 = np.hypot(wind["u100"], wind["v100"])
    up = solar[solar["sun_up"]]
    summary = {
        "solar": {
            "share_sun_down": round(float(1 - solar["sun_up"].mean()), 4),
            "share_exact_zero_all_hours": round(float((solar["y"] == 0).mean()), 4),
            "share_exact_zero_sun_up": round(float((up["y"] == 0).mean()), 4),
            "share_at_least_0.99_sun_up": round(float((up["y"] >= 0.99).mean()), 4),
            "corr_power_ssrd": round(float(solar["y"].corr(solar["ssrd"])), 4),
            "split_rows": {k: int(m.sum()) for k, m in split_masks(solar["timestamp"], "solar", cfg).items()},
        },
        "wind": {
            "share_exact_zero": round(float((wind["y"] == 0).mean()), 4),
            "share_at_least_0.99": round(float((wind["y"] >= 0.99).mean()), 4),
            "corr_power_ws100": round(float(wind["y"].corr(ws100)), 4),
            "split_rows": {k: int(m.sum()) for k, m in split_masks(wind["timestamp"], "wind", cfg).items()},
        },
    }
    (results / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    paths = [
        fig_solar_diurnal(solar, cfg),
        fig_solar_vs_radiation(solar, cfg),
        fig_wind_power_curve(wind, cfg),
        fig_monthly(solar, wind, cfg),
        fig_splits({"solar": solar, "wind": wind}, cfg),
        fig_target_dist(solar, wind, cfg),
    ]
    print(json.dumps(summary, indent=2))
    for p in paths:
        print("figure ->", p)


if __name__ == "__main__":
    main()
