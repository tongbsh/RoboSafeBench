"""Benchmark-scale, framework, and asset-scaffold figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynisafety.paper_figures.data import PAPER_BACKENDS, PaperBundle, label_backend, label_level
from dynisafety.paper_figures.helpers import empty_axes
from dynisafety.paper_figures.style import (
    DOUBLE,
    INK,
    MUTED,
    OKABE,
    SINGLE,
    draw_booktabs,
    light_ygrid,
    maybe_stamp,
    panel_label,
    save_figure,
)

_TRAJECTORY_PHASES = ("pre", "enter", "closest", "exit")


def plot_scale_overview(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 4.05))
    scale_rows = [
        [
            label_backend(bundle.display, block.backend),
            str(block.n_tasks),
            str(block.n_models),
            f"{block.n_jobs:,}",
            f"{block.n_episodes:,}",
        ]
        for block in bundle.scale.blocks
    ]
    scale_rows.append(
        [
            "Total",
            str(bundle.scale.n_tasks),
            "—",
            f"{bundle.scale.n_jobs:,}",
            f"{bundle.scale.n_episodes:,}",
        ]
    )
    draw_booktabs(
        axes[0],
        ("Backend", "Tasks", "Models", "Jobs", "Episodes"),
        scale_rows,
        col_align=("left", "center", "center", "right", "right"),
        col_widths=(0.28, 0.16, 0.16, 0.18, 0.22),
        highlight_rows=(len(scale_rows) - 1,),
        title="Planned evaluation matrix",
        letter="a",
        footnote="3 backends · 4 levels · 39 tasks · 1,332 jobs · 54,600 episodes",
    )

    peer_rows = [
        [
            str(peer["name"]),
            str(int(peer["simulators"])),
            str(int(peer["tasks"])),
            str(int(peer["dynamic_levels"])),
            str(peer.get("safety_axis", "—")),
        ]
        for peer in bundle.peers
    ]
    peer_rows.append(
        [
            bundle.settings.benchmark_name,
            str(bundle.scale.n_backends),
            str(bundle.scale.n_tasks),
            "4",
            "dynamic",
        ]
    )
    draw_booktabs(
        axes[1],
        ("Benchmark", "Sims", "Tasks", "Dyn.", "Safety"),
        peer_rows,
        col_align=("left", "center", "center", "center", "center"),
        col_widths=(0.38, 0.14, 0.16, 0.14, 0.18),
        highlight_rows=(len(peer_rows) - 1,),
        title="Host and safety benchmarks",
        letter="b",
        footnote="Safety: none / static obstacle / dynamic contact",
    )
    fig.subplots_adjust(hspace=0.22, left=0.03, right=0.97, top=0.96, bottom=0.03)
    maybe_stamp(fig, bundle)
    return save_figure(
        fig,
        out_dir,
        "fig01_scale_overview",
        formats=bundle.settings.formats,
        dpi=bundle.settings.dpi,
    )


def plot_framework(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    """Four-column methods strip: type and rules only, no clip-art boxes."""
    fig, ax = plt.subplots(figsize=(DOUBLE, 1.72))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = (
        ("1", "Simulators", ("LIBERO", "SimplerEnv", "RoboTwin")),
        ("2", "Disturbance", ("L0 Baseline", "L1 Sudden Drop", "L2 Arc-Transit", "L3 Shared Reach")),
        ("3", "Policy", ("VLA", "Oracle SWR", "AEGIS shield")),
        ("4", "Outcomes", ("SSR · SR · CR", "SS / US / CFF / CIF", "9-class diagnostics")),
    )
    n = len(stages)
    width = 1.0 / n
    for i, (num, title, items) in enumerate(stages):
        x = i * width
        if i > 0:
            ax.plot([x, x], [0.08, 0.92], color="#D0D0D0", lw=0.5, solid_capstyle="butt")
        ax.text(x + 0.03, 0.88, num, ha="left", va="top", fontsize=8, color=INK)
        ax.text(x + 0.08, 0.88, title, ha="left", va="top", fontsize=8, color=INK)
        for k, item in enumerate(items):
            ax.text(x + 0.08, 0.68 - 0.16 * k, item, ha="left", va="top", fontsize=7, color=MUTED)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
    return save_figure(
        fig, out_dir, "fig02_framework", formats=bundle.settings.formats, dpi=bundle.settings.dpi
    )


def plot_teaser_montage(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    backends = PAPER_BACKENDS
    levels = (0, 1, 2, 3)
    fig, axes = plt.subplots(len(backends), len(levels), figsize=(DOUBLE, 3.55))
    root = bundle.settings.teaser_dir
    for i, backend in enumerate(backends):
        for j, level in enumerate(levels):
            ax = axes[i][j]
            path = Path(root) / f"{backend}_l{level}.png" if root is not None else None
            _show_or_placeholder(ax, path)
            if i == 0:
                ax.set_title(label_level(bundle.display, level, short=True), fontsize=8, pad=3)
            if j == 0:
                ax.set_ylabel(label_backend(bundle.display, backend), fontsize=8)
    fig.subplots_adjust(wspace=0.06, hspace=0.08, left=0.08, right=0.99, top=0.93, bottom=0.02)
    maybe_stamp(fig, bundle)
    return save_figure(
        fig, out_dir, "figS1_teaser_montage", formats=bundle.settings.formats, dpi=bundle.settings.dpi
    )


def plot_trajectory(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    backends = PAPER_BACKENDS
    levels = (1, 2, 3)
    fig, axes = plt.subplots(
        len(backends) * len(levels),
        len(_TRAJECTORY_PHASES),
        figsize=(DOUBLE, 6.4),
    )
    root = bundle.settings.trajectory_dir
    row = 0
    for backend in backends:
        for level in levels:
            for j, phase in enumerate(_TRAJECTORY_PHASES):
                ax = axes[row][j]
                path = (
                    Path(root) / f"{backend}_l{level}_{phase}.png" if root is not None else None
                )
                _show_or_placeholder(ax, path)
                if row == 0:
                    ax.set_title(phase, fontsize=8, pad=3)
                if j == 0:
                    ax.set_ylabel(
                        f"{label_backend(bundle.display, backend)}  "
                        f"{label_level(bundle.display, level, short=True)}",
                        fontsize=7,
                    )
            row += 1
    fig.subplots_adjust(wspace=0.05, hspace=0.08, left=0.12, right=0.99, top=0.96, bottom=0.01)
    maybe_stamp(fig, bundle)
    return save_figure(
        fig, out_dir, "figS2_trajectory", formats=bundle.settings.formats, dpi=bundle.settings.dpi
    )


def plot_coverage(bundle: PaperBundle, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(SINGLE, 2.35))
    planned = bundle.scale.planned
    if planned.empty:
        empty_axes(ax, "")
    else:
        backends = [b.backend for b in bundle.scale.blocks]
        planned_n = [int((planned["backend"] == b).sum()) for b in backends]
        if bundle.coverage.empty:
            done_n = [0 for _ in backends]
        else:
            done_n = [
                int(bundle.coverage.loc[bundle.coverage["backend"] == b, "completed"].sum())
                for b in backends
            ]
        x = np.arange(len(backends))
        ax.bar(x - 0.16, planned_n, 0.32, color="#C8C8C8", edgecolor="none", label="Planned")
        ax.bar(x + 0.16, done_n, 0.32, color=OKABE["blue"], edgecolor="none", label="Completed")
        ax.set_xticks(list(x))
        ax.set_xticklabels([label_backend(bundle.display, b) for b in backends])
        ax.set_ylabel("Evaluation cells")
        ax.legend(loc="upper right")
        light_ygrid(ax)
        for i, (p, d) in enumerate(zip(planned_n, done_n)):
            pct = 100.0 * d / p if p else 0.0
            ax.text(i, max(p, d) * 1.03, f"{pct:.0f}%", ha="center", fontsize=6, color=MUTED)
        panel_label(ax, "a")
    maybe_stamp(fig, bundle)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.90, bottom=0.16)
    return save_figure(
        fig, out_dir, "figA3_coverage", formats=bundle.settings.formats, dpi=bundle.settings.dpi
    )


def _show_or_placeholder(ax, path: Path | None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("#C8C8C8")
    if path is not None and path.is_file():
        import matplotlib.image as mpimg

        ax.imshow(mpimg.imread(path))
        ax.set_axis_off()
        return
    ax.set_facecolor("#F7F7F7")
