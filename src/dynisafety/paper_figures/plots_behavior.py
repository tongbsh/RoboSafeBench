"""Behavior, oracle, process, force, and CFF-decomposition figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynisafety.paper_figures.data import (
    PaperBundle,
    attach_quadrant,
    base_episodes,
    base_runs,
    label_backend,
    label_level,
    label_metric,
    ordered_backends,
    ordered_models,
    weighted_collapse,
)
from dynisafety.paper_figures.helpers import (
    add_level_legend,
    empty_axes,
    grouped_level_bars,
    lookup_rate,
)
from dynisafety.paper_figures.style import (
    BEHAVIOR_COLORS,
    BEHAVIOR_LABELS,
    BEHAVIOR_ORDER,
    DOUBLE,
    LEVEL_COLORS,
    MUTED,
    OKABE,
    light_ygrid,
    maybe_stamp,
    panel_label,
    save_figure,
)

_UNOBSERVED = ("not_triggered", "baseline_unusable")
_PROCESS = ("reaction_rate", "stop_rate", "evade_rate", "recovery_rate")


def plot_behavior(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    episodes = base_episodes(bundle)
    if episodes.empty or "diag_behavior_class" not in episodes.columns:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "fig06_behavior", bundle)
    work = episodes.loc[episodes["diag_behavior_class"].notna()].copy()
    if work.empty:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "fig06_behavior", bundle)
    backends = ordered_backends(bundle.display, work["backend"].unique())
    # One panel per backend; stack by level (models pooled within backend for readability).
    fig, axes = plt.subplots(1, len(backends), figsize=(DOUBLE, 3.1), squeeze=False, sharey=True)
    for i, backend in enumerate(backends):
        ax = axes[0][i]
        sub = work[work["backend"] == backend]
        levels = sorted(int(x) for x in sub["level"].unique())
        _stack_behavior(ax, sub, levels, "level")
        ax.set_title(label_backend(bundle.display, backend), fontsize=8)
        ax.set_xlabel("level")
        if i == 0:
            ax.set_ylabel("class share")
        panel_label(ax, "abcd"[i] if i < 4 else str(i))
    _behavior_legend(fig)
    maybe_stamp(fig, bundle)
    fig.tight_layout(rect=(0, 0.02, 1, 0.84))
    return _save(fig, out_dir, "fig06_behavior", bundle)


def plot_oracle(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    mapping = bundle.settings.oracle_base_map
    episodes = bundle.episodes
    if episodes.empty:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "fig07_oracle", bundle)
    frames: list[pd.DataFrame] = []
    for oracle, base in mapping.items():
        paired = _paired_oracle_rows(
            episodes, base=base, oracle=oracle, confidence=bundle.settings.confidence
        )
        if not paired.empty:
            frames.append(paired)
    if not frames:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "fig07_oracle", bundle)
    paired = pd.concat(frames, ignore_index=True)
    grouped = weighted_collapse(
        paired, ("arm", "level"), confidence=bundle.settings.confidence
    )
    metrics = ("sr", "ssr", "cr")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.55), squeeze=False, sharey=True)
    arms = [a for a in ("base", "oracle") if a in set(grouped["arm"].astype(str))]
    levels = sorted(int(x) for x in grouped["level"].unique() if int(x) > 0) or sorted(
        int(x) for x in grouped["level"].unique()
    )
    colors = {"base": OKABE["blue"], "oracle": OKABE["green"]}
    labels = {"base": "Base VLA", "oracle": "Oracle / shield"}
    for j, metric in enumerate(metrics):
        ax = axes[0][j]
        n_l = len(levels)
        width = 0.36
        for i, arm in enumerate(arms):
            xs, ys = [], []
            for k, level in enumerate(levels):
                row = grouped[(grouped["arm"] == arm) & (grouped["level"].astype(int) == int(level))]
                if row.empty or row[metric].isna().all():
                    continue
                xs.append(k + (i - 0.5) * width)
                ys.append(float(row[metric].iloc[0]))
            if xs:
                ax.bar(xs, ys, width=width * 0.95, color=colors[arm], label=labels[arm])
        ax.set_xticks(range(n_l))
        ax.set_xticklabels([label_level(bundle.display, lv, short=True) for lv in levels])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(label_metric(bundle.display, metric), fontsize=8)
        light_ygrid(ax)
        panel_label(ax, "abc"[j])
        if j == 0:
            ax.set_ylabel("rate")
            ax.legend(loc="best", fontsize=6)
    maybe_stamp(fig, bundle)
    fig.tight_layout()
    return _save(fig, out_dir, "fig07_oracle", bundle)


def plot_process_single_trace(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    """Single-trace four axes — the Limitations control, not the body figure.

    Kept next to the paired fig09 because §9 limitation 1 has to be shown, not
    only asserted: the kinematic reaction gate saturates on L0 controls and
    ``reaction_rate`` collapses to 0 for long action chunks.
    """
    runs = base_runs(bundle)
    if runs.empty:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "figA4_process_single_trace", bundle)
    collapsed = weighted_collapse(
        runs, ("backend", "model", "level"), confidence=bundle.settings.confidence
    )
    # L0 has no trigger denominator — drop it.
    if "level" in collapsed.columns:
        collapsed = collapsed[collapsed["level"].astype(int) > 0]
    backends = ordered_backends(bundle.display, collapsed["backend"].unique() if not collapsed.empty else [])
    if collapsed.empty or not backends:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "figA4_process_single_trace", bundle)
    fig, axes = plt.subplots(
        len(backends),
        len(_PROCESS),
        figsize=(DOUBLE, 2.15 * len(backends)),
        squeeze=False,
        sharey=True,
    )
    for i, backend in enumerate(backends):
        sub = collapsed[collapsed["backend"] == backend]
        models = ordered_models(bundle.display, sub["model"].astype(str).unique())
        levels = sorted(int(x) for x in sub["level"].unique())
        for j, metric in enumerate(_PROCESS):
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
                label_rotation=30,
            )
            if i == 0:
                ax.set_title(label_metric(bundle.display, metric), fontsize=8)
    add_level_legend(
        fig,
        [axes[0][0]],
        bundle.display,
        sorted(int(x) for x in collapsed["level"].unique()),
    )
    fig.text(
        0.01,
        0.005,
        "Single-trace reference frame (§2.1–§2.4): gates are relative to this episode's own\n"
        "pre-onset window, and baseline_unusable episodes are outside every denominator. Not\n"
        "readable against the paired fig09 — different detectors and different denominators.",
        fontsize=5.2,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    maybe_stamp(fig, bundle)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    return _save(fig, out_dir, "figA4_process_single_trace", bundle)


def plot_force_box(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    episodes = base_episodes(bundle)
    if episodes.empty or "peak_force_n" not in episodes.columns:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "figA1_force_box", bundle)
    backends = ordered_backends(bundle.display, episodes["backend"].unique())
    fig, axes = plt.subplots(1, len(backends), figsize=(DOUBLE, 2.7), squeeze=False)
    for i, backend in enumerate(backends):
        ax = axes[0][i]
        sub = episodes[episodes["backend"] == backend]
        levels = sorted(int(x) for x in sub["level"].unique())
        data = [sub.loc[sub["level"].astype(int) == lv, "peak_force_n"].astype(float).to_numpy() for lv in levels]
        if all(len(d) == 0 for d in data):
            empty_axes(ax)
            continue
        bp = ax.boxplot(
            data,
            tick_labels=[label_level(bundle.display, lv, short=True) for lv in levels],
            patch_artist=True,
            medianprops={"color": "#222222", "linewidth": 0.8},
            boxprops={"linewidth": 0.5},
            whiskerprops={"color": "#222222", "linewidth": 0.5},
            capprops={"color": "#222222", "linewidth": 0.5},
            flierprops={"markersize": 2, "markeredgecolor": MUTED, "marker": "."},
        )
        for patch, lv in zip(bp["boxes"], levels):
            patch.set_facecolor(LEVEL_COLORS.get(int(lv), OKABE["gray"]))
            patch.set_edgecolor("#222222")
            patch.set_alpha(0.85)
        ax.set_title(label_backend(bundle.display, backend), fontsize=8)
        ax.set_ylabel("peak contact index")
        light_ygrid(ax)
        panel_label(ax, "abcd"[i] if i < 4 else str(i))
    fig.text(
        0.01,
        0.01,
        "Within-backend only; force_source is not comparable across simulators.",
        fontsize=6,
        color=MUTED,
    )
    maybe_stamp(fig, bundle)
    fig.tight_layout()
    return _save(fig, out_dir, "figA1_force_box", bundle)


def plot_failure_decomp(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    episodes = base_episodes(bundle)
    needed = ("success", "contact_steps", "diag_behavior_class")
    if episodes.empty or any(c not in episodes.columns for c in needed):
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "figA2_failure_decomp", bundle)
    work = attach_quadrant(episodes.loc[episodes["diag_behavior_class"].notna()].copy())
    work = work[~work["diag_behavior_class"].astype(str).isin(_UNOBSERVED)]
    if work.empty:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "figA2_failure_decomp", bundle)
    quads = ("ss", "us", "cff", "cif")
    titles = {"ss": "SS", "us": "US", "cff": "CFF", "cif": "CIF"}
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 2.8), squeeze=False, sharey=True)
    levels = sorted(int(x) for x in work["level"].unique())
    class_order = [c for c in BEHAVIOR_ORDER if c not in _UNOBSERVED]
    for i, quad in enumerate(quads):
        ax = axes[0][i]
        sub = work[work["quadrant"] == quad]
        bottoms = np.zeros(len(levels))
        xs = [str(label_level(bundle.display, lv, short=True)) for lv in levels]
        for cls in class_order:
            heights = [
                float(((sub["level"].astype(int) == lv) & (sub["diag_behavior_class"] == cls)).sum())
                for lv in levels
            ]
            ax.bar(
                xs,
                heights,
                bottom=bottoms,
                color=BEHAVIOR_COLORS[cls],
                edgecolor="none",
                linewidth=0.0,
            )
            bottoms = bottoms + np.asarray(heights)
        ax.set_title(titles[quad], fontsize=8)
        if i == 0:
            ax.set_ylabel("episode count")
        light_ygrid(ax)
        panel_label(ax, "abcd"[i])
    fig.text(
        0.01,
        0.01,
        "CFF is contact-free failure, not Safe Stop.",
        fontsize=6,
        color=MUTED,
    )
    _behavior_legend(fig, observed_only=True)
    maybe_stamp(fig, bundle)
    fig.tight_layout(rect=(0, 0.02, 1, 0.84))
    return _save(fig, out_dir, "figA2_failure_decomp", bundle)


def _stack_behavior(ax, sub: pd.DataFrame, keys: list, key_col: str) -> None:
    labels = [str(k) if key_col != "level" else label_level_short(int(k)) for k in keys]
    bottoms = np.zeros(len(keys))
    totals = [float((sub[key_col] == k).sum()) if key_col != "level" else float((sub[key_col].astype(int) == int(k)).sum()) for k in keys]
    for cls in BEHAVIOR_ORDER:
        if key_col == "level":
            raw = [float(((sub[key_col].astype(int) == int(k)) & (sub["diag_behavior_class"] == cls)).sum()) for k in keys]
        else:
            raw = [float(((sub[key_col] == k) & (sub["diag_behavior_class"] == cls)).sum()) for k in keys]
        heights = [c / t if t > 0 else 0.0 for c, t in zip(raw, totals)]
        ax.bar(
            labels,
            heights,
            bottom=bottoms,
            color=BEHAVIOR_COLORS[cls],
            edgecolor="none",
            linewidth=0.0,
        )
        bottoms = bottoms + np.asarray(heights)
    ax.set_ylim(0.0, 1.02)
    light_ygrid(ax)


def label_level_short(level: int) -> str:
    return f"L{int(level)}"


def _behavior_legend(fig, *, observed_only: bool = False) -> None:
    classes = [c for c in BEHAVIOR_ORDER if (c not in _UNOBSERVED) or not observed_only]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BEHAVIOR_COLORS[c], label=BEHAVIOR_LABELS[c])
        for c in classes
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, 1.02))


def _paired_oracle_rows(
    episodes: pd.DataFrame,
    *,
    base: str,
    oracle: str,
    confidence: float,
) -> pd.DataFrame:
    keys = ["backend", "suite", "task_id", "level", "seed"]
    need = keys + ["model", "success", "contact_steps"]
    if any(c not in episodes.columns for c in need):
        return pd.DataFrame()
    base_ep = episodes[episodes["model"].astype(str) == base]
    ora_ep = episodes[episodes["model"].astype(str) == oracle]
    if base_ep.empty or ora_ep.empty:
        return pd.DataFrame()
    from dynisafety.paper_figures.data import collapse_episodes

    base_cells = collapse_episodes(base_ep, keys, confidence=confidence)
    ora_cells = collapse_episodes(ora_ep, keys, confidence=confidence)
    merged = base_cells.merge(ora_cells, on=keys, how="inner", suffixes=("_base", "_ora"))
    if merged.empty:
        return pd.DataFrame()
    rows = []
    for _, rec in merged.iterrows():
        for arm, suffix in (("base", "_base"), ("oracle", "_ora")):
            rows.append(
                {
                    "backend": rec["backend"],
                    "suite": rec["suite"],
                    "task_id": rec["task_id"],
                    "level": rec["level"],
                    "seed": rec["seed"],
                    "arm": arm,
                    "n_episodes": rec[f"n_episodes{suffix}"],
                    "ss": rec[f"ss{suffix}"],
                    "us": rec[f"us{suffix}"],
                    "cff": rec[f"cff{suffix}"],
                    "cif": rec[f"cif{suffix}"],
                    "sr": rec[f"sr{suffix}"],
                    "cr": rec[f"cr{suffix}"],
                    "ssr": rec[f"ssr{suffix}"],
                }
            )
    return pd.DataFrame(rows)


def _save(fig, out_dir: Path, stem: str, bundle: PaperBundle) -> list[Path]:
    return save_figure(fig, out_dir, stem, formats=bundle.settings.formats, dpi=bundle.settings.dpi)
