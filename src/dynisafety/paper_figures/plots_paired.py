"""Paired four-axis figures (docs/user/eval.md §2.5).

Reference frame here is the matched level-0 twin, so nothing in this module may
be read side by side with a single-trace panel: the denominators differ (the
paired ruler scores the episodes the baseline guard refused) and so do the
detectors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynisafety.analysis.paired_axes import PAIRED_CLASS_ORDER
from dynisafety.paper_figures.data import (
    PaperBundle,
    base_paired_episodes,
    label_backend,
    label_level,
    label_metric,
    ordered_backends,
    ordered_models,
)
from dynisafety.paper_figures.helpers import (
    add_level_legend,
    empty_axes,
    grouped_level_bars,
    lookup_rate,
)
from dynisafety.paper_figures.paired import (
    paired_axis_rates,
    paired_baseline_states,
    paired_class_shares,
)
from dynisafety.paper_figures.style import (
    BEHAVIOR_COLORS,
    BEHAVIOR_LABELS,
    DOUBLE,
    MUTED,
    light_ygrid,
    maybe_stamp,
    panel_label,
    save_figure,
)

# (rate, its pre-onset FPR, its own denominator). Recovery has no pre-onset
# counterpart: its gate is the hazard end, not a null distribution.
_PAIRED_AXES: tuple[tuple[str, str | None, str], ...] = (
    ("reaction_rate", "reaction_pre_onset_fpr", "n"),
    ("stop_rate", "stop_pre_onset_fpr", "n"),
    ("evade_rate", "evade_pre_onset_fpr", "n"),
    ("recovery_rate", None, "n_recovery_applicable"),
)

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_PROCESS_FOOTNOTE = (
    "Paired ruler (§2.5): every gate is measured against the matched L0 twin, not a pre-onset\n"
    "window of the same episode. Rotated labels are that rate's own denominator; recovery uses\n"
    "n_recovery_applicable (hazard ended, budget left, robot stopped), not n. Grey ticks are the\n"
    "same detector on the pre-onset segment; groups whose rows have no pre-onset segment show no\n"
    "tick (n/a) and are never counted as False."
)
_BEHAVIOR_FOOTNOTE = (
    "8 paired classes, not 10: not_triggered is skipped before scoring and baseline_unusable\n"
    "cannot arise when the reference is the twin. These shares are not comparable with the\n"
    "10-class shares in fig06. Columns split own_baseline — whether the single-trace path could\n"
    "also label the row; unusable is the population it left with four blank axes."
)


def plot_process_paired(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    rates = paired_axis_rates(base_paired_episodes(bundle))
    backends = (
        ordered_backends(bundle.display, rates["backend"].unique()) if not rates.empty else []
    )
    if not backends:
        return _unavailable(bundle, out_dir, "fig09_process")
    fig, axes = plt.subplots(
        len(backends),
        len(_PAIRED_AXES),
        figsize=(DOUBLE, 2.25 * len(backends)),
        squeeze=False,
        sharey=True,
    )
    for i, backend in enumerate(backends):
        sub = rates[rates["backend"] == backend]
        models = ordered_models(bundle.display, sub["model"].astype(str).unique())
        levels = sorted(int(x) for x in sub["level"].unique())
        for j, (metric, fpr_col, denom_col) in enumerate(_PAIRED_AXES):
            ax = axes[i][j]
            grouped_level_bars(
                ax,
                models=models,
                levels=levels,
                value=lambda m, lv, s=sub, met=metric: lookup_rate(s, model=m, level=lv, col=met),
                error=None,
                display=bundle.display,
                ylabel=label_backend(bundle.display, backend) if j == 0 else "",
                letter=None,
                ylim=(0.0, 1.24),
                label_rotation=30,
                annotate=lambda m, lv, s=sub, col=denom_col: _denominator(
                    s, model=m, level=lv, col=col
                ),
                null_level=(
                    None
                    if fpr_col is None
                    else lambda m, lv, s=sub, col=fpr_col: lookup_rate(
                        s, model=m, level=lv, col=col
                    )
                ),
            )
            if i == 0:
                ax.set_title(label_metric(bundle.display, metric), fontsize=8)
    add_level_legend(
        fig,
        [axes[0][0]],
        bundle.display,
        sorted(int(x) for x in rates["level"].unique()),
    )
    _footnote(fig, _PROCESS_FOOTNOTE, bundle)
    maybe_stamp(fig, bundle)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    return _save(fig, out_dir, "fig09_process", bundle)


def plot_behavior_paired(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    shares = paired_class_shares(base_paired_episodes(bundle))
    backends = (
        ordered_backends(bundle.display, shares["backend"].unique()) if not shares.empty else []
    )
    states = paired_baseline_states(shares)
    if not backends or not states:
        return _unavailable(bundle, out_dir, "figA5_behavior_paired")
    fig, axes = plt.subplots(
        len(backends),
        len(states),
        figsize=(DOUBLE, 2.05 * len(backends)),
        squeeze=False,
        sharey=True,
    )
    for i, backend in enumerate(backends):
        for j, state in enumerate(states):
            ax = axes[i][j]
            sub = shares[
                (shares["backend"] == backend) & (shares["own_baseline"].astype(str) == state)
            ]
            if sub.empty:
                empty_axes(ax, "no rows for this own_baseline")
            else:
                _stack_paired_classes(ax, sub, bundle)
            if i == 0:
                ax.set_title(f"own_baseline = {state}", fontsize=8, pad=12)
            if j == 0:
                ax.set_ylabel(label_backend(bundle.display, backend))
            k = i * len(states) + j
            panel_label(ax, _LETTERS[k] if k < len(_LETTERS) else str(k))
    _paired_class_legend(fig)
    _footnote(fig, _BEHAVIOR_FOOTNOTE, bundle)
    maybe_stamp(fig, bundle)
    fig.tight_layout(rect=(0, 0.11, 1, 0.88))
    return _save(fig, out_dir, "figA5_behavior_paired", bundle)


def _stack_paired_classes(ax, sub: pd.DataFrame, bundle: PaperBundle) -> None:
    levels = sorted(int(x) for x in sub["level"].unique())
    labels = [label_level(bundle.display, lv, short=True) for lv in levels]
    counts = [_denominator(sub, level=lv, col="n") for lv in levels]
    bottoms = np.zeros(len(levels))
    for cls in PAIRED_CLASS_ORDER:
        heights = np.asarray(
            [_share(sub, level=lv, col=f"{cls}_rate") for lv in levels], dtype=float
        )
        ax.bar(
            labels,
            heights,
            bottom=bottoms,
            color=BEHAVIOR_COLORS[cls],
            edgecolor="none",
            linewidth=0.0,
        )
        bottoms = bottoms + heights
    for k, text in enumerate(counts):
        if text:
            ax.text(k, 1.03, text, ha="center", va="bottom", fontsize=5, color=MUTED)
    ax.set_ylim(0.0, 1.14)
    light_ygrid(ax)


def _share(frame: pd.DataFrame, *, level: int, col: str) -> float:
    if col not in frame.columns:
        return 0.0
    row = frame[frame["level"].astype(int) == int(level)]
    if row.empty or row[col].isna().all():
        return 0.0
    return float(row[col].astype(float).iloc[0])


def _denominator(
    frame: pd.DataFrame,
    *,
    col: str,
    model: str | None = None,
    level: int | None = None,
) -> str | None:
    sub = frame
    if model is not None:
        sub = sub[sub["model"].astype(str) == str(model)]
    if level is not None:
        sub = sub[sub["level"].astype(int) == int(level)]
    if sub.empty or col not in sub.columns or sub[col].isna().all():
        return None
    return f"n={int(sub[col].astype(float).iloc[0])}"


def _paired_class_legend(fig) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BEHAVIOR_COLORS[c], label=BEHAVIOR_LABELS[c])
        for c in PAIRED_CLASS_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, 1.0))


def _footnote(fig, text: str, bundle: PaperBundle) -> None:
    note = bundle.paired.note()
    fig.text(
        0.01,
        0.005,
        text if not note else f"{text}\n{note}",
        fontsize=5.2,
        color=MUTED,
        ha="left",
        va="bottom",
    )


def _unavailable(bundle: PaperBundle, out_dir: Path, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
    empty_axes(ax, bundle.paired.unavailable_message())
    maybe_stamp(fig, bundle)
    return _save(fig, out_dir, stem, bundle)


def _save(fig, out_dir: Path, stem: str, bundle: PaperBundle) -> list[Path]:
    return save_figure(fig, out_dir, stem, formats=bundle.settings.formats, dpi=bundle.settings.dpi)
