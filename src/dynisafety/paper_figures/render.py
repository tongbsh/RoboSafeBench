"""Orchestrate paper-figure rendering. Does not write into the eval tree."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from dynisafety.paper_figures.catalog import FIGURES, FigureSpec, spec_by_id
from dynisafety.paper_figures.data import PaperBundle
from dynisafety.paper_figures.plots_behavior import (
    plot_behavior,
    plot_failure_decomp,
    plot_force_box,
    plot_oracle,
    plot_process_single_trace,
)
from dynisafety.paper_figures.plots_overview import (
    plot_coverage,
    plot_framework,
    plot_scale_overview,
    plot_teaser_montage,
    plot_trajectory,
)
from dynisafety.paper_figures.plots_paired import (
    plot_behavior_paired,
    plot_process_paired,
)
from dynisafety.paper_figures.plots_results import (
    plot_heatmap_ssr,
    plot_joint_stack,
    plot_level_curves,
    plot_main_metrics,
    plot_pareto,
)
from dynisafety.paper_figures.style import apply_iclr_style

_RENDERERS: dict[str, Callable[[PaperBundle, Path], list[Path]]] = {
    "fig01_scale_overview": plot_scale_overview,
    "fig02_framework": plot_framework,
    "fig03_main_metrics": plot_main_metrics,
    "fig04_joint_stack": plot_joint_stack,
    "fig05_pareto": plot_pareto,
    "fig06_behavior": plot_behavior,
    "fig07_oracle": plot_oracle,
    "fig08_heatmap_ssr": plot_heatmap_ssr,
    "fig09_process": plot_process_paired,
    "fig10_level_curves": plot_level_curves,
    "figA1_force_box": plot_force_box,
    "figA2_failure_decomp": plot_failure_decomp,
    "figA3_coverage": plot_coverage,
    "figA4_process_single_trace": plot_process_single_trace,
    "figA5_behavior_paired": plot_behavior_paired,
    "figS1_teaser_montage": plot_teaser_montage,
    "figS2_trajectory": plot_trajectory,
}


def select_figures(names: Sequence[str] | None) -> tuple[FigureSpec, ...]:
    if names is None:
        return FIGURES
    return tuple(spec_by_id(name) for name in names)


def render_figures(
    bundle: PaperBundle,
    out: Path,
    *,
    figures: Sequence[str] | None = None,
    overview_only: bool = False,
) -> dict[str, list[str]]:
    """Write selected figures under ``out``. Returns stem → written paths."""
    apply_iclr_style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = select_figures(figures)
    if overview_only:
        chosen = tuple(s for s in chosen if not s.needs_results)
    written: dict[str, list[str]] = {}
    for spec in chosen:
        renderer = _RENDERERS[spec.stem]
        paths = renderer(bundle, out)
        written[spec.stem] = [str(p) for p in paths]
    _write_manifest(out, bundle, written)
    return written


def _write_manifest(
    out: Path,
    bundle: PaperBundle,
    written: dict[str, list[str]],
) -> None:
    """Always list the full inventory. ``--figures`` must not wipe the catalog."""
    figures_payload = []
    for spec in FIGURES:
        on_disk = sorted(str(p) for p in out.glob(f"{spec.stem}.*") if p.is_file())
        files = written.get(spec.stem, on_disk)
        figures_payload.append(
            {
                "fig_id": spec.fig_id,
                "stem": spec.stem,
                "section": spec.section,
                "title": spec.title,
                "caption": spec.caption,
                "files": files,
            }
        )
    payload = {
        "benchmark": bundle.settings.benchmark_name,
        "n_planned_jobs": bundle.n_planned_jobs,
        "n_completed_jobs": bundle.n_completed_jobs,
        "n_planned_episodes": bundle.scale.n_episodes,
        "n_planned_tasks": bundle.scale.n_tasks,
        "n_backends": bundle.scale.n_backends,
        "partial": bundle.partial,
        "stamp": bundle.stamp(),
        "blocks": [
            {
                "backend": b.backend,
                "suites": list(b.suites),
                "n_tasks": b.n_tasks,
                "n_models": b.n_models,
                "n_jobs": b.n_jobs,
                "n_episodes": b.n_episodes,
                "episodes_per_task": b.episodes_per_task,
            }
            for b in bundle.scale.blocks
        ],
        "figures": figures_payload,
    }
    (out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {bundle.settings.benchmark_name} paper figures",
        "",
        f"- planned jobs: **{bundle.n_planned_jobs}**",
        f"- completed cells: **{bundle.n_completed_jobs}**",
        f"- planned episodes: **{bundle.scale.n_episodes:,}**",
        f"- planned tasks: **{bundle.scale.n_tasks}**",
    ]
    stamp = bundle.stamp()
    if stamp:
        lines.append(f"- {stamp}")
    lines.extend(
        [
            "",
            "| ID | File | Section | Title |",
            "|---|---|---|---|",
        ]
    )
    for spec in FIGURES:
        lines.append(f"| {spec.fig_id} | `{spec.stem}` | {spec.section} | {spec.title} |")
    lines.append("")
    (out / "catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
