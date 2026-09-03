"""Main result figures: rates, joint outcomes, Pareto, heatmaps, curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynisafety.paper_figures.data import (
    PaperBundle,
    base_runs,
    label_backend,
    label_level,
    label_metric,
    label_model,
    label_suite,
    ordered_backends,
    ordered_models,
    weighted_collapse,
)
from dynisafety.paper_figures.helpers import (
    add_level_legend,
    empty_axes,
    grouped_level_bars,
    lookup_ci,
    lookup_rate,
)
from dynisafety.paper_figures.style import (
    DOUBLE,
    HEATMAP_CMAP,
    MUTED,
    QUAD_COLORS,
    QUAD_ORDER,
    light_ygrid,
    maybe_stamp,
    model_color,
    panel_label,
    save_figure,
)


def plot_main_metrics(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    runs = _by_model_level(bundle)
    backends = ordered_backends(bundle.display, runs["backend"].unique() if not runs.empty else [])
    metrics = ("sr", "ssr", "cr")
    if not backends:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "fig03_main_metrics", bundle)
    fig, axes = plt.subplots(
        len(backends),
        3,
        figsize=(DOUBLE, 2.15 * len(backends)),
        squeeze=False,
        sharey=True,
    )
    letters = "abcdefghijklmnopqrstuvwxyz"
    k = 0
    for i, backend in enumerate(backends):
        sub = runs[runs["backend"] == backend]
        models = ordered_models(bundle.display, sub["model"].astype(str).unique())
        levels = sorted(int(x) for x in sub["level"].unique())
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            grouped_level_bars(
                ax,
                models=models,
                levels=levels,
                value=lambda m, lv, s=sub, met=metric: lookup_rate(s, model=m, level=lv, col=met),
                error=lambda m, lv, s=sub, met=metric: lookup_ci(s, model=m, level=lv, col=met),
                display=bundle.display,
                ylabel=label_backend(bundle.display, backend) if j == 0 else "",
                letter=letters[k],
            )
            if i == 0:
                ax.set_title(label_metric(bundle.display, metric), fontsize=8)
            k += 1
    add_level_legend(fig, [axes[0][0]], bundle.display, (0, 1, 2, 3))
    maybe_stamp(fig, bundle)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _save(fig, out_dir, "fig03_main_metrics", bundle)


def plot_joint_stack(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    runs = _by_model_level(bundle)
    backends = ordered_backends(bundle.display, runs["backend"].unique() if not runs.empty else [])
    if not backends:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "fig04_joint_stack", bundle)
    fig, axes = plt.subplots(1, len(backends), figsize=(DOUBLE, 3.0), squeeze=False)
    for i, backend in enumerate(backends):
        ax = axes[0][i]
        sub = runs[runs["backend"] == backend]
        models = ordered_models(bundle.display, sub["model"].astype(str).unique())
        levels = sorted(int(x) for x in sub["level"].unique())
        _draw_joint_clusters(ax, sub, models, levels, bundle)
        ax.set_title(label_backend(bundle.display, backend), fontsize=8)
        panel_label(ax, "abcd"[i] if i < 4 else str(i))
        if i == 0:
            ax.set_ylabel("episode share")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=QUAD_COLORS[q], label=bundle.display.quadrants.get(q, q.upper()))
        for q in QUAD_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    maybe_stamp(fig, bundle)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    return _save(fig, out_dir, "fig04_joint_stack", bundle)


def plot_pareto(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    runs = _by_model_level(bundle)
    backends = ordered_backends(bundle.display, runs["backend"].unique() if not runs.empty else [])
    if not backends:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.6))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "fig05_pareto", bundle)
    fig, axes = plt.subplots(1, len(backends), figsize=(DOUBLE, 2.7), squeeze=False, sharex=True, sharey=True)
    known = ordered_models(
        bundle.display,
        runs["model"].astype(str).unique(),
    )
    for i, backend in enumerate(backends):
        ax = axes[0][i]
        sub = runs[runs["backend"] == backend]
        models = ordered_models(bundle.display, sub["model"].astype(str).unique())
        for model in models:
            rows = sub[sub["model"].astype(str) == model].sort_values("level")
            if rows.empty:
                continue
            xs = rows["sr"].astype(float).tolist()
            ys = rows["ssr"].astype(float).tolist()
            ax.plot(
                xs,
                ys,
                "-o",
                color=model_color(model, known),
                label=label_model(bundle.display, model),
                markersize=4,
            )
            for _, rec in rows.iterrows():
                ax.annotate(
                    label_level(bundle.display, int(rec["level"]), short=True),
                    (float(rec["sr"]), float(rec["ssr"])),
                    fontsize=6,
                    xytext=(3, 3),
                    textcoords="offset points",
                    color=MUTED,
                )
        ax.set_xlabel("SR")
        if i == 0:
            ax.set_ylabel("SSR")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(label_backend(bundle.display, backend), fontsize=8)
        light_ygrid(ax)
        ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.4)
        panel_label(ax, "abcd"[i] if i < 4 else str(i))
        ax.legend(fontsize=6, loc="lower left")
    maybe_stamp(fig, bundle)
    fig.tight_layout()
    return _save(fig, out_dir, "fig05_pareto", bundle)


def plot_heatmap_ssr(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    runs = base_runs(bundle)
    if runs.empty:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "fig08_heatmap_ssr", bundle)
    backends = ordered_backends(bundle.display, runs["backend"].unique())
    # One row per backend; models as columns if few enough.
    model_sets = {
        b: ordered_models(bundle.display, runs.loc[runs["backend"] == b, "model"].astype(str).unique())
        for b in backends
    }
    max_models = max((len(v) for v in model_sets.values()), default=1)
    facet = max_models <= bundle.settings.heatmap_max_model_facets
    n_cols = max_models if facet else 1
    fig, axes = plt.subplots(
        len(backends),
        n_cols,
        figsize=(DOUBLE, 2.3 * len(backends)),
        squeeze=False,
    )
    for i, backend in enumerate(backends):
        sub = runs[runs["backend"] == backend]
        models = model_sets[backend]
        if facet:
            for j in range(n_cols):
                ax = axes[i][j]
                if j >= len(models):
                    ax.set_axis_off()
                    continue
                _ssr_heatmap(
                    ax,
                    weighted_collapse(
                        sub[sub["model"].astype(str) == models[j]],
                        ("suite", "task_id", "level"),
                        confidence=bundle.settings.confidence,
                    ),
                    bundle,
                )
                title = f"{label_backend(bundle.display, backend)} · {label_model(bundle.display, models[j])}"
                ax.set_title(title, fontsize=8)
        else:
            ax = axes[i][0]
            collapsed = weighted_collapse(
                sub, ("suite", "task_id", "level"), confidence=bundle.settings.confidence
            )
            _ssr_heatmap(ax, collapsed, bundle)
            ax.set_title(f"{label_backend(bundle.display, backend)} (models pooled)", fontsize=8)
    maybe_stamp(fig, bundle)
    fig.tight_layout()
    return _save(fig, out_dir, "fig08_heatmap_ssr", bundle)


def plot_level_curves(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    runs = _by_model_level(bundle)
    backends = ordered_backends(bundle.display, runs["backend"].unique() if not runs.empty else [])
    metrics = ("sr", "ssr", "cr")
    if not backends:
        fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
        empty_axes(ax)
        maybe_stamp(fig, bundle)
        return _save(fig, out_dir, "fig10_level_curves", bundle)
    fig, axes = plt.subplots(
        len(backends),
        3,
        figsize=(DOUBLE, 2.1 * len(backends)),
        squeeze=False,
        sharey=True,
    )
    known = ordered_models(bundle.display, runs["model"].astype(str).unique())
    for i, backend in enumerate(backends):
        sub = runs[runs["backend"] == backend]
        models = ordered_models(bundle.display, sub["model"].astype(str).unique())
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            for model in models:
                rows = sub[sub["model"].astype(str) == model].sort_values("level")
                rows = rows[rows[metric].notna()]
                if rows.empty:
                    continue
                ax.plot(
                    rows["level"].astype(int),
                    rows[metric].astype(float),
                    "-o",
                    color=model_color(model, known),
                    label=label_model(bundle.display, model),
                    markersize=4,
                )
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(["L0", "L1", "L2", "L3"])
            ax.set_ylim(-0.02, 1.05)
            light_ygrid(ax)
            if i == 0:
                ax.set_title(label_metric(bundle.display, metric), fontsize=8)
            if j == 0:
                ax.set_ylabel(label_backend(bundle.display, backend))
            if i == 0 and j == 2:
                ax.legend(fontsize=6, loc="best")
    maybe_stamp(fig, bundle)
    fig.tight_layout()
    return _save(fig, out_dir, "fig10_level_curves", bundle)


def _by_model_level(bundle: PaperBundle) -> pd.DataFrame:
    runs = base_runs(bundle)
    if runs.empty:
        return runs
    return weighted_collapse(
        runs,
        ("backend", "model", "level"),
        confidence=bundle.settings.confidence,
    )


def _draw_joint_clusters(ax, sub, models, levels, bundle: PaperBundle) -> None:
    if not models:
        empty_axes(ax)
        return
    n_l = max(len(levels), 1)
    width = 0.8 / n_l
    for i, level in enumerate(levels):
        bottoms = np.zeros(len(models))
        xs = [j + (i - (n_l - 1) / 2.0) * width for j in range(len(models))]
        for q in QUAD_ORDER:
            heights = []
            for model in models:
                row = sub[(sub["model"].astype(str) == model) & (sub["level"].astype(int) == int(level))]
                if row.empty:
                    heights.append(0.0)
                    continue
                n = float(row["n_episodes"].iloc[0])
                heights.append(float(row[q].iloc[0]) / n if n > 0 else 0.0)
            # Skip a fully-missing cluster (all zeros because no row, not true zeros).
            have = [
                not sub[(sub["model"].astype(str) == m) & (sub["level"].astype(int) == int(level))].empty
                for m in models
            ]
            draw_h = [h if ok else 0.0 for h, ok in zip(heights, have)]
            if any(have):
                ax.bar(
                    xs,
                    draw_h,
                    width=width * 0.92,
                    bottom=bottoms,
                    color=QUAD_COLORS[q],
                    edgecolor="none",
                    linewidth=0.0,
                )
                bottoms = bottoms + np.asarray(draw_h)
    tick_x: list[float] = []
    tick_lab: list[str] = []
    for j, _model in enumerate(models):
        for i, level in enumerate(levels):
            tick_x.append(j + (i - (n_l - 1) / 2.0) * width)
            tick_lab.append(label_level(bundle.display, int(level), short=True))
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_lab, fontsize=6)
    ax.tick_params(axis="x", pad=1.5)
    for j, model in enumerate(models):
        ax.text(
            j,
            -0.13,
            label_model(bundle.display, model),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            clip_on=False,
        )
    ax.set_ylim(0.0, 1.02)
    light_ygrid(ax)


def _ssr_heatmap(ax, frame: pd.DataFrame, bundle: PaperBundle) -> None:
    if frame.empty or "ssr" not in frame.columns:
        empty_axes(ax)
        return
    work = frame.copy()
    if "task_label" not in work.columns:
        work["task_label"] = [
            f"{label_suite(bundle.display, str(s))}-{int(t)}"
            for s, t in zip(work["suite"].astype(str), work["task_id"].astype(int))
        ]
    levels = sorted(int(x) for x in work["level"].unique())
    tasks = list(dict.fromkeys(work["task_label"].tolist()))
    grid = np.full((len(tasks), len(levels)), np.nan)
    for rec in work.itertuples(index=False):
        r = tasks.index(rec.task_label)
        c = levels.index(int(rec.level))
        grid[r, c] = float(rec.ssr)
    im = ax.imshow(grid, aspect="auto", cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([label_level(bundle.display, lv, short=True) for lv in levels])
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _save(fig, out_dir: Path, stem: str, bundle: PaperBundle) -> list[Path]:
    return save_figure(fig, out_dir, stem, formats=bundle.settings.formats, dpi=bundle.settings.dpi)
