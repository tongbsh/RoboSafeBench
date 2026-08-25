"""Main report plots: level curves, force distributions, joint outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from dynisafety.core.joint_outcomes import classify_joint_outcome
from dynisafety.report.tables import UNAVAILABLE, _aggregate_metric_mean

# Locked quadrant stack order (bottom → top)
_STACK_ORDER = ("ss", "us", "cff", "cif")
_STACK_COLORS = {
    "ss": "#2ca02c",   # green — safe success
    "us": "#ff7f0e",   # orange — unsafe success
    "cff": "#7f7f7f",  # gray — contact-free failure
    "cif": "#d62728",  # red — contact-involved failure
}
_STACK_LABELS = {
    "ss": "SS",
    "us": "US",
    "cff": "CFF",
    "cif": "CIF",
}

# 9-class palette / labels (docs/user/behavior_diagnostics.md §3). Shared by both
# new figures so the CFF split in failure_decomposition.png matches the
# distribution figure.
_BC_ORDER = (
    "collide_unreacted",
    "collide_reacted",
    "evade_success",
    "evade_fail",
    "stop_recover",
    "stop_no_recover",
    "partial_response",
    "no_response",
    "not_triggered",
)
_BC_COLORS = {
    "collide_unreacted": "#8c1515",
    "collide_reacted": "#e6550d",
    "evade_success": "#31a354",
    "evade_fail": "#74c476",
    "stop_recover": "#3182bd",
    "stop_no_recover": "#6baed6",
    "partial_response": "#fdae6b",
    "no_response": "#636363",
    "not_triggered": "#d9d9d9",
}
_BC_LABELS = {
    "collide_unreacted": "Collide (no reaction)",
    "collide_reacted": "Collide (late)",
    "evade_success": "Evade",
    "evade_fail": "Evade→fail",
    "stop_recover": "Stop→recover",
    "stop_no_recover": "Stop→no recover",
    "partial_response": "Partial",
    "no_response": "No response",
    "not_triggered": "Not triggered",
}
_QUADRANTS = ("SS", "US", "CFF", "CIF")
# CFF process split (the reviewer-facing claim): not "Safe Stop".
_CFF_CLASSES = ("stop_no_recover", "evade_fail", "partial_response", "no_response")


def plot_level_curves(
    runs_df: pd.DataFrame,
    out_path: Path,
    *,
    metrics: Sequence[str],
) -> None:
    """Line plot: x=level, y=metric, hue=model (one panel per metric)."""
    if runs_df.empty:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        ax.set_title("no data")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    work = runs_df.copy()
    if "n_episodes" not in work.columns:
        work["n_episodes"] = 1.0

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5), squeeze=False)
    for i, metric in enumerate(metrics):
        ax = axes[0][i]
        if metric not in work.columns:
            ax.set_title(f"{metric} (missing)")
            continue
        for model, grp in work.groupby("model"):
            # Collapse seeds/tasks: weighted mean by level
            levels = []
            vals = []
            for level, g2 in grp.groupby("level"):
                w = g2["n_episodes"].astype(float)
                v = g2[metric]
                mask = v.notna()
                if not bool(mask.any()):
                    continue
                levels.append(int(level))
                vals.append(float((v[mask].astype(float) * w[mask]).sum() / w[mask].sum()))
            if levels:
                order = sorted(range(len(levels)), key=lambda j: levels[j])
                ax.plot(
                    [levels[j] for j in order],
                    [vals[j] for j in order],
                    marker="o",
                    label=str(model),
                )
        ax.set_xlabel("level")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_force_box(
    episodes_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Box plot of peak_force_n by level (hue=model when present)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if episodes_df.empty or "peak_force_n" not in episodes_df.columns:
        ax.set_title("no force data")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    work = episodes_df.copy()
    if "level" not in work.columns:
        raise KeyError("episodes_df must contain 'level'")

    levels = sorted(work["level"].unique())
    if "model" in work.columns and work["model"].nunique() > 1:
        models = sorted(work["model"].astype(str).unique())
        width = 0.8 / max(len(models), 1)
        for i, model in enumerate(models):
            sub = work[work["model"].astype(str) == model]
            data = [sub.loc[sub["level"] == lv, "peak_force_n"].astype(float).values for lv in levels]
            pos = [lv + (i - (len(models) - 1) / 2) * width for lv in levels]
            ax.boxplot(
                data,
                positions=pos,
                widths=width * 0.9,
                patch_artist=True,
                tick_labels=[""] * len(levels),
            )
            ax.plot([], [], label=model, color=f"C{i}")
        ax.set_xticks(list(levels))
        ax.set_xticklabels([str(lv) for lv in levels])
        ax.legend(fontsize=8)
    else:
        data = [
            work.loc[work["level"] == lv, "peak_force_n"].astype(float).values
            for lv in levels
        ]
        ax.boxplot(data, tick_labels=[str(lv) for lv in levels])
    ax.set_xlabel("level")
    ax.set_ylabel("peak_force_n")
    ax.set_title("Force distribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _joint_counts_available(df: pd.DataFrame) -> bool:
    return all(c in df.columns and df[c].notna().any() for c in _STACK_ORDER)


def plot_joint_outcome_stacked(
    runs_df: pd.DataFrame,
    out_path: Path,
    *,
    group_col: str = "model",
    normalize: bool = False,
    title: str | None = None,
) -> None:
    """Stacked bar of SS/US/CFF/CIF counts (or fractions) per group.

    Missing joint columns → empty figure annotated ``unavailable`` (never fill 0).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * max(runs_df[group_col].nunique() if group_col in runs_df.columns else 1, 1)), 4.5))

    if runs_df.empty or group_col not in runs_df.columns or not _joint_counts_available(runs_df):
        ax.set_title(f"SS/US/CFF/CIF stacked ({UNAVAILABLE})")
        ax.text(0.5, 0.5, UNAVAILABLE, ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    gcols = [group_col]
    if "level" in runs_df.columns and group_col != "level":
        # Collapse across other dims but keep one bar per model (sum levels)
        pass
    grouped = _aggregate_metric_mean(runs_df, gcols)
    labels = [str(x) for x in grouped[group_col].tolist()]
    bottoms = [0.0] * len(labels)
    for key in _STACK_ORDER:
        raw = grouped[key].astype(float).tolist()
        if normalize:
            totals = [
                float(sum(grouped.loc[i, list(_STACK_ORDER)].astype(float)))
                for i in grouped.index
            ]
            heights = [v / t if t > 0 else 0.0 for v, t in zip(raw, totals)]
        else:
            heights = raw
        ax.bar(
            labels,
            heights,
            bottom=bottoms,
            color=_STACK_COLORS[key],
            label=_STACK_LABELS[key],
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xlabel(group_col)
    ax.set_ylabel("fraction" if normalize else "episode count")
    ax.set_title(title or ("Joint outcomes (normalized)" if normalize else "Joint outcomes (SS/US/CFF/CIF)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_joint_outcome_stacked_by_level(
    runs_df: pd.DataFrame,
    out_path: Path,
    *,
    model_col: str = "model",
    level_col: str = "level",
    normalize: bool = False,
) -> None:
    """Faceted stacked bars: one panel per model, x=level."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        runs_df.empty
        or model_col not in runs_df.columns
        or level_col not in runs_df.columns
        or not _joint_counts_available(runs_df)
    ):
        fig, ax = plt.subplots()
        ax.set_title(f"joint stack by level ({UNAVAILABLE})")
        ax.text(0.5, 0.5, UNAVAILABLE, ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    models = sorted(runs_df[model_col].astype(str).unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), squeeze=False, sharey=True)
    grouped = _aggregate_metric_mean(runs_df, [model_col, level_col])

    for i, model in enumerate(models):
        ax = axes[0][i]
        sub = grouped[grouped[model_col].astype(str) == model].sort_values(level_col)
        if sub.empty:
            ax.set_title(f"{model} ({UNAVAILABLE})")
            continue
        levels = [str(lv) for lv in sub[level_col].tolist()]
        bottoms = [0.0] * len(levels)
        for key in _STACK_ORDER:
            raw = sub[key].astype(float).tolist()
            if normalize:
                totals = [
                    float(sum(sub.loc[idx, list(_STACK_ORDER)].astype(float)))
                    for idx in sub.index
                ]
                heights = [v / t if t > 0 else 0.0 for v, t in zip(raw, totals)]
            else:
                heights = raw
            ax.bar(
                levels,
                heights,
                bottom=bottoms,
                color=_STACK_COLORS[key],
                label=_STACK_LABELS[key] if i == 0 else None,
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        ax.set_title(str(model), fontsize=9)
        ax.set_xlabel("level")
        if i == 0:
            ax.set_ylabel("fraction" if normalize else "count")
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("Joint outcomes by model × level", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class ParetoSeriesSpec:
    """One series on an SR–SSR or SR–CR Pareto scatter.

    When the requested columns are missing, the series is skipped and noted as
    ``unavailable`` (reserved for Oracle / Shield until those runs exist).
    """

    label: str
    sr_col: str = "sr"
    y_col: str = "ssr"
    filter: Mapping[str, Any] | None = None
    marker: str = "o"


def _weighted_xy_by_group(
    work: pd.DataFrame,
    *,
    group_col: str,
    sr_col: str,
    y_col: str,
) -> list[tuple[str, float, float]]:
    pts: list[tuple[str, float, float]] = []
    for gname, grp in work.groupby(group_col):
        w = (
            grp["n_episodes"].astype(float)
            if "n_episodes" in grp.columns
            else pd.Series(1.0, index=grp.index)
        )
        mask = grp[sr_col].notna() & grp[y_col].notna() & w.notna()
        if not bool(mask.any()):
            continue
        ww = w[mask]
        wsum = float(ww.sum())
        if wsum <= 0:
            continue
        pts.append(
            (
                str(gname),
                float((grp.loc[mask, sr_col].astype(float) * ww).sum() / wsum),
                float((grp.loc[mask, y_col].astype(float) * ww).sum() / wsum),
            )
        )
    return pts


def plot_pareto_sr_vs(
    runs_df: pd.DataFrame,
    out_path: Path,
    *,
    y_metric: str = "ssr",
    series: Sequence[ParetoSeriesSpec] | None = None,
    group_col: str = "model",
) -> None:
    """Scatter Pareto: x=SR, y=SSR (or CR). Oracle series may be unavailable."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    if series is None:
        series = (
            ParetoSeriesSpec(label="Base", sr_col="sr", y_col=y_metric, marker="o"),
            ParetoSeriesSpec(
                label="Oracle",
                sr_col="oracle_sr",
                y_col=f"oracle_{y_metric}" if y_metric in {"ssr", "cr"} else y_metric,
                marker="^",
            ),
        )

    notes: list[str] = []
    plotted = 0
    for spec in series:
        work = runs_df
        if spec.filter:
            mask = pd.Series(True, index=work.index)
            for k, v in spec.filter.items():
                if k in work.columns:
                    mask &= work[k] == v
            work = work.loc[mask]
        sr_col, y_col = spec.sr_col, spec.y_col
        if work.empty or sr_col not in work.columns or y_col not in work.columns:
            notes.append(f"{spec.label}: {UNAVAILABLE}")
            continue
        sub = work.dropna(subset=[sr_col, y_col])
        if sub.empty:
            notes.append(f"{spec.label}: {UNAVAILABLE}")
            continue
        if group_col in sub.columns:
            pts = _weighted_xy_by_group(
                sub, group_col=group_col, sr_col=sr_col, y_col=y_col
            )
            if not pts:
                notes.append(f"{spec.label}: {UNAVAILABLE}")
                continue
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            ax.scatter(xs, ys, marker=spec.marker, label=spec.label, s=60)
            for name, x, y in pts:
                ax.annotate(
                    name,
                    (x, y),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        else:
            ax.scatter(
                sub[sr_col].astype(float),
                sub[y_col].astype(float),
                marker=spec.marker,
                label=spec.label,
                s=60,
            )
        plotted += 1

    ax.set_xlabel("SR")
    ylabel = {"ssr": "SSR", "cr": "CR", "sr": "SR"}.get(y_metric, y_metric)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Pareto: SR vs {ylabel}")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    if notes:
        ax.text(
            0.02,
            0.02,
            "; ".join(notes),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            color="#555555",
        )
    if plotted == 0:
        ax.text(0.5, 0.5, UNAVAILABLE, ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ssr_heatmap(
    runs_df: pd.DataFrame,
    out_path: Path,
    *,
    task_col: str = "task_id",
    level_col: str = "level",
    value_col: str = "ssr",
) -> None:
    """Heatmap of SSR (or other rate) over task × level."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    if (
        runs_df.empty
        or task_col not in runs_df.columns
        or level_col not in runs_df.columns
        or value_col not in runs_df.columns
        or not runs_df[value_col].notna().any()
    ):
        ax.set_title(f"{value_col} heatmap ({UNAVAILABLE})")
        ax.text(0.5, 0.5, UNAVAILABLE, ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    grouped = _aggregate_metric_mean(runs_df, [task_col, level_col])
    pivot = grouped.pivot(index=task_col, columns=level_col, values=value_col)
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("level")
    ax.set_ylabel(task_col)
    ax.set_title(f"{value_col.upper()} heatmap (task × level)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_joint_plots(
    runs_df: pd.DataFrame,
    out_dir: Path,
    *,
    stack_normalize: bool = False,
) -> None:
    """Write joint stacked / Pareto / heatmap under ``out_dir/plots/``."""
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_joint_outcome_stacked(
        runs_df,
        plots_dir / "joint_stack_by_model.png",
        group_col="model",
        normalize=stack_normalize,
    )
    plot_joint_outcome_stacked(
        runs_df,
        plots_dir / "joint_stack_by_model_frac.png",
        group_col="model",
        normalize=True,
    )
    plot_joint_outcome_stacked_by_level(
        runs_df,
        plots_dir / "joint_stack_by_model_level.png",
        normalize=stack_normalize,
    )
    # Base-only SR–SSR Pareto; Oracle series reserved → unavailable annotation
    plot_pareto_sr_vs(
        runs_df,
        plots_dir / "pareto_sr_ssr.png",
        y_metric="ssr",
        series=(
            ParetoSeriesSpec(label="Base", sr_col="sr", y_col="ssr", marker="o"),
            ParetoSeriesSpec(label="Oracle", sr_col="oracle_sr", y_col="oracle_ssr", marker="^"),
        ),
    )
    plot_pareto_sr_vs(
        runs_df,
        plots_dir / "pareto_sr_cr.png",
        y_metric="cr",
        series=(
            ParetoSeriesSpec(label="Base", sr_col="sr", y_col="cr", marker="o"),
            ParetoSeriesSpec(label="Oracle", sr_col="oracle_sr", y_col="oracle_cr", marker="^"),
        ),
    )
    task_col = "task_id" if "task_id" in runs_df.columns else (
        "task_name" if "task_name" in runs_df.columns else None
    )
    if task_col is not None:
        plot_ssr_heatmap(
            runs_df,
            plots_dir / "heatmap_ssr_task_level.png",
            task_col=task_col,
        )


def _behavior_class_available(df: pd.DataFrame) -> bool:
    return "diag_behavior_class" in df.columns and df["diag_behavior_class"].notna().any()


def _unavailable_figure(out_path: Path, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.text(0.5, 0.5, UNAVAILABLE, ha="center", va="center", transform=ax.transAxes)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_behavior_classes(
    episodes_df: pd.DataFrame,
    out_path: Path,
    *,
    level_col: str = "level",
    model_col: str = "model",
) -> None:
    """Stacked 9-class shares by level (L0→L3). Includes ``not_triggered``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        episodes_df.empty
        or level_col not in episodes_df.columns
        or not _behavior_class_available(episodes_df)
    ):
        _unavailable_figure(out_path, f"behavior classes ({UNAVAILABLE})")
        return

    work = episodes_df.loc[episodes_df["diag_behavior_class"].notna()].copy()
    models = (
        sorted(work[model_col].astype(str).unique())
        if model_col in work.columns and work[model_col].nunique() > 1
        else [None]
    )
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False, sharey=True)

    for i, model in enumerate(models):
        ax = axes[0][i]
        sub = work if model is None else work[work[model_col].astype(str) == model]
        levels = sorted(int(lv) for lv in sub[level_col].unique())
        labels = [str(lv) for lv in levels]
        bottoms = [0.0] * len(levels)
        totals = [
            float((sub[level_col] == lv).sum())
            for lv in levels
        ]
        for cls in _BC_ORDER:
            raw = [
                float(((sub[level_col] == lv) & (sub["diag_behavior_class"] == cls)).sum())
                for lv in levels
            ]
            heights = [c / t if t > 0 else 0.0 for c, t in zip(raw, totals)]
            ax.bar(
                labels,
                heights,
                bottom=bottoms,
                color=_BC_COLORS[cls],
                label=_BC_LABELS[cls] if i == 0 else None,
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        ax.set_xlabel("level")
        if i == 0:
            ax.set_ylabel("class share")
        title = "Behavior classes" if model is None else str(model)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            fontsize=7,
            bbox_to_anchor=(0.5, 1.18),
        )
    fig.suptitle("Behavior-class distribution by level (L0→L3)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_failure_decomposition(
    episodes_df: pd.DataFrame,
    out_path: Path,
    *,
    level_col: str = "level",
) -> None:
    """Split SS/US/CFF/CIF by behavior class.

    This is the reviewer-facing figure: CFF is **not** “Safe Stop”. The CFF
    panel is titled to make the split obvious — ``stop_no_recover`` (halted,
    never finished), ``evade_fail`` (cleared contact, task unfinished),
    ``partial_response`` (measurable but incomplete), ``no_response`` (failed
    with an unchanged trajectory). ``not_triggered`` is excluded (it sits
    outside the four-quadrant denominator).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    needed = ("success", "contact_steps", "diag_behavior_class")
    if episodes_df.empty or any(c not in episodes_df.columns for c in needed):
        _unavailable_figure(out_path, f"failure decomposition ({UNAVAILABLE})")
        return
    work = episodes_df.loc[episodes_df["diag_behavior_class"].notna()].copy()
    work = work[work["diag_behavior_class"].astype(str) != "not_triggered"]
    if work.empty:
        _unavailable_figure(out_path, f"failure decomposition ({UNAVAILABLE})")
        return

    work["quadrant"] = [
        classify_joint_outcome(success=bool(success), contact_steps=int(contact_steps))
        for success, contact_steps in zip(
            work["success"].tolist(),
            work["contact_steps"].tolist(),
        )
    ]

    has_level = level_col in work.columns and work[level_col].nunique() > 0
    levels = sorted(int(lv) for lv in work[level_col].unique()) if has_level else [None]
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(3.4 * 4, 4.2),
        squeeze=False,
        sharey=True,
    )
    class_order = [c for c in _BC_ORDER if c != "not_triggered"]

    for i, quad in enumerate(_QUADRANTS):
        ax = axes[0][i]
        sub = work[work["quadrant"] == quad]
        labels = [str(lv) if lv is not None else quad for lv in levels]
        bottoms = [0.0] * len(levels)
        for cls in class_order:
            if has_level:
                heights = [
                    float(((sub[level_col] == lv) & (sub["diag_behavior_class"] == cls)).sum())
                    for lv in levels
                ]
            else:
                heights = [float((sub["diag_behavior_class"] == cls).sum())]
            ax.bar(
                labels,
                heights,
                bottom=bottoms,
                color=_BC_COLORS[cls],
                label=_BC_LABELS[cls] if i == 0 else None,
                edgecolor="white",
                linewidth=0.4,
                hatch="//" if quad == "CFF" and cls in _CFF_CLASSES else None,
            )
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        if quad == "CFF":
            ax.set_title('CFF — not "Safe Stop"', fontsize=9, color="#8c1515")
            ax.set_xlabel("stop-no-rec / evade-fail / no-resp")
            for spine in ax.spines.values():
                spine.set_color("#8c1515")
                spine.set_linewidth(1.6)
        else:
            ax.set_title(quad, fontsize=9)
            ax.set_xlabel("level" if has_level else "")
        if i == 0:
            ax.set_ylabel("episode count")
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            fontsize=7,
            bbox_to_anchor=(0.5, 1.16),
        )
    fig.suptitle(
        "CFF is not Safe Stop: contact-free failures split into "
        "Stop→no recover / Evade→fail / No response",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_all_plots(
    runs_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    out_dir: Path,
    *,
    include_joint: bool = True,
) -> None:
    """Write level curves, force box, and optional joint plots under ``out_dir/plots/``."""
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_level_curves(
        runs_df,
        plots_dir / "level_sr_cr.png",
        metrics=("sr", "cr"),
    )
    # Include SSR on level curves when present (display id stays ssr, not sasr)
    if "ssr" in runs_df.columns and runs_df["ssr"].notna().any():
        plot_level_curves(
            runs_df,
            plots_dir / "level_sr_ssr_cr.png",
            metrics=("sr", "ssr", "cr"),
        )
    plot_force_box(episodes_df, plots_dir / "force_box.png")
    plot_behavior_classes(episodes_df, plots_dir / "behavior_classes.png")
    plot_failure_decomposition(episodes_df, plots_dir / "failure_decomposition.png")
    if include_joint:
        render_joint_plots(runs_df, out_dir)
