"""Shared drawing helpers for paper figures."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from dynisafety.paper_figures.data import DisplayNames, label_level, label_model
from dynisafety.paper_figures.style import INK, LEVEL_COLORS, MUTED, light_ygrid, panel_label


def empty_axes(ax: Axes, message: str = "", *, letter: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.4)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    if message:
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#8A8A8A", fontsize=7)
    if letter:
        panel_label(ax, letter)


def grouped_level_bars(
    ax: Axes,
    *,
    models: Sequence[str],
    levels: Sequence[int],
    value: Callable[[str, int], float | None],
    error: Callable[[str, int], tuple[float, float] | None] | None,
    display: DisplayNames,
    ylabel: str,
    letter: str | None = None,
    ylim: tuple[float, float] = (0.0, 1.02),
    label_rotation: float | None = None,
    annotate: Callable[[str, int], str | None] | None = None,
    null_level: Callable[[str, int], float | None] | None = None,
) -> bool:
    """Grouped bars. ``value`` returning None skips that bar (never draws 0).

    ``annotate`` labels a drawn bar with its own denominator; ``null_level``
    draws that bar's null-distribution reference (a pre-onset false-positive
    rate) as a tick, and returning None leaves it blank rather than 0.
    """
    n_m = len(models)
    n_l = len(levels)
    if n_m == 0 or n_l == 0:
        empty_axes(ax, "", letter=letter)
        return False
    width = 0.78 / n_l
    drawn = False
    for i, level in enumerate(levels):
        xs: list[float] = []
        ys: list[float] = []
        drawn_models: list[str] = []
        yerr: list[list[float]] = [[], []]
        for j, model in enumerate(models):
            val = value(model, int(level))
            if val is None:
                continue
            pos = j + (i - (n_l - 1) / 2.0) * width
            xs.append(pos)
            ys.append(float(val))
            drawn_models.append(str(model))
            if error is not None:
                band = error(model, int(level))
                if band is None:
                    yerr[0].append(0.0)
                    yerr[1].append(0.0)
                else:
                    lo, hi = band
                    yerr[0].append(max(0.0, float(val) - float(lo)))
                    yerr[1].append(max(0.0, float(hi) - float(val)))
        if not xs:
            continue
        drawn = True
        ax.bar(
            xs,
            ys,
            width=width * 0.90,
            color=LEVEL_COLORS.get(int(level), "#333333"),
            label=label_level(display, int(level), short=True),
            edgecolor="none",
            linewidth=0.0,
            zorder=2,
        )
        if annotate is not None:
            for pos, val, model in zip(xs, ys, drawn_models):
                text = annotate(model, int(level))
                if not text:
                    continue
                ax.text(
                    pos,
                    val + 0.02,
                    text,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=4.6,
                    color=MUTED,
                    zorder=4,
                )
        if null_level is not None:
            for pos, model in zip(xs, drawn_models):
                null = null_level(model, int(level))
                if null is None:
                    continue
                ax.plot(
                    [pos - width * 0.45, pos + width * 0.45],
                    [float(null), float(null)],
                    color=MUTED,
                    linewidth=0.6,
                    solid_capstyle="butt",
                    zorder=4,
                )
        if error is not None and any(a + b > 0 for a, b in zip(yerr[0], yerr[1])):
            ax.errorbar(
                xs,
                ys,
                yerr=np.asarray(yerr),
                fmt="none",
                ecolor=INK,
                elinewidth=0.45,
                capsize=1.0,
                capthick=0.45,
                zorder=3,
            )
    ax.set_xticks(range(n_m))
    rotation = (0 if n_m <= 3 else 18) if label_rotation is None else float(label_rotation)
    ax.set_xticklabels(
        [label_model(display, m) for m in models],
        rotation=rotation,
        ha="center" if rotation == 0 else "right",
    )
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    light_ygrid(ax)
    if letter:
        panel_label(ax, letter)
    return drawn


def lookup_rate(df, *, model: str, level: int, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    sub = df[(df["model"].astype(str) == str(model)) & (df["level"].astype(int) == int(level))]
    if sub.empty or sub[col].isna().all():
        return None
    return float(sub[col].astype(float).iloc[0])


def lookup_ci(df, *, model: str, level: int, col: str) -> tuple[float, float] | None:
    lo_c, hi_c = f"{col}_ci_lo", f"{col}_ci_hi"
    if df.empty or lo_c not in df.columns or hi_c not in df.columns:
        return None
    sub = df[(df["model"].astype(str) == str(model)) & (df["level"].astype(int) == int(level))]
    if sub.empty or sub[lo_c].isna().all() or sub[hi_c].isna().all():
        return None
    return float(sub[lo_c].iloc[0]), float(sub[hi_c].iloc[0])


def add_level_legend(fig: Figure, axes: Sequence[Axes], display: DisplayNames, levels: Sequence[int]) -> None:
    handles = [
        Patch(facecolor=LEVEL_COLORS[int(lv)], edgecolor="none", label=label_level(display, int(lv), short=True))
        for lv in levels
    ]
    labels = [label_level(display, int(lv), short=True) for lv in levels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(levels),
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )
    del axes
