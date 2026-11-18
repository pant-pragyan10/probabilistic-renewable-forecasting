"""Shared plotting style. Colors follow the validated categorical palette (light mode):
slot 1 blue, slot 2 orange, slot 3 aqua; sequential ramp is the blue ramp (light -> dark)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import resolve_path  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]  # fixed order, never cycled; fold/facet beyond 3 for scatter-type charts
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.titlelocation": "left",
            "axes.titlesize": 11,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK_2,
            "font.size": 9.5,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def figures_dir(cfg: dict) -> Path:
    d = resolve_path(cfg, "results") / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, name: str, cfg: dict) -> Path:
    path = figures_dir(cfg) / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
