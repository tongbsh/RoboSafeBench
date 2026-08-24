"""Write feasibility tables, heatmap, and narrative report (no fabricated paper numbers)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from dynisafety.feasibility.aggregate import CellMetrics
from dynisafety.feasibility.estimate import ResourceEstimate


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table without optional ``tabulate`` dependency."""
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for c in cols:
            val = row[c]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                cells.append("")
            elif isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def narrative_notes() -> list[str]:
    """Expected scientific narrative templates — fill with real numbers after authorized runs."""
    return [
        "Oracle vs Base: CR is expected to decrease under stop-wait-resume (pause VLA "
        "while disturbance is active). Do not invent CR values before authorized runs.",
        "Oracle = dynamic-phase pause; AEGIS = after settle, CBF-QP filter around the "
        "remaining static obstacle (sim_gt). Track oracle_pause_steps and "
        "aegis_intervened_steps separately.",
        "Closing aegis (base / oracle_swr) must not change those regressions: always "
        "emit explicit extra.libero.aegis.enabled=false in eval commands.",
        "L1/L2: after settle a static obstacle remains. If Oracle CR drops but SSR stays "
        "low, interpret as missing active avoidance; oracle_swr_aegis is the first "
        "plug-in layer (not VLA weight changes / SafeVLA).",
        "L3: hand withdraw clears the scene ≈ L0; optional proof cell keeps aegis off.",
        "Never label oracle controller failure as invalid-geometry; use unresolved + reason.",
        "CFF is contact-free failure, not Safe Stop; CIF is contact-involved failure, "
        "not Collision-induced Failure (Prompt 20 LOCKED names).",
    ]


def render_feasibility_markdown(
    *,
    compare_rows: Sequence[Mapping[str, Any]],
    matrix_rows: Sequence[Mapping[str, Any]],
    estimate: ResourceEstimate | None,
    status: str,
) -> str:
    lines: list[str] = [
        "# Feasibility: Base vs Oracle-SWR vs Oracle-SWR+AEGIS",
        "",
        f"**Run status:** `{status}`",
        "",
        "## Narrative (protocol — not fabricated results)",
        "",
    ]
    for note in narrative_notes():
        lines.append(f"- {note}")
    lines.extend(["", "## Base vs Oracle vs AEGIS (joint metrics)", ""])
    if compare_rows:
        df = pd.DataFrame(list(compare_rows))
        cols = [
            c
            for c in (
                "level",
                "task_id",
                "task_name",
                "N_base",
                "N_oracle",
                "N_aegis",
                "Base_SR",
                "Base_CR",
                "Base_SSR",
                "Oracle_SR",
                "Oracle_CR",
                "Oracle_SSR",
                "Oracle_pause_steps_mean",
                "Oracle_policy_infer_steps_mean",
                "Aegis_SR",
                "Aegis_CR",
                "Aegis_SSR",
                "Aegis_pause_steps_mean",
                "Aegis_intervened_steps_mean",
                "delta_CR",
                "delta_SSR",
                "delta_CR_aegis_vs_oracle",
                "delta_SSR_aegis_vs_oracle",
            )
            if c in df.columns
        ]
        lines.append(_df_to_markdown(df[cols]))
    else:
        lines.append("_No comparison rows (awaiting authorized eval or fixture)._")

    lines.extend(["", "## Feasibility class matrix", ""])
    if matrix_rows:
        dfm = pd.DataFrame(list(matrix_rows))
        cols_m = [
            c
            for c in (
                "level",
                "task_id",
                "task_name",
                "feasibility_class",
                "base_cr",
                "oracle_cr",
                "aegis_cr",
                "base_ssr",
                "oracle_ssr",
                "aegis_ssr",
                "oracle_pause_steps_mean",
                "aegis_intervened_steps_mean",
                "feasibility_reason",
            )
            if c in dfm.columns
        ]
        lines.append(_df_to_markdown(dfm[cols_m]))
    else:
        lines.append("_No matrix rows._")

    if estimate is not None:
        lines.extend(
            [
                "",
                "## Resource estimate (dry)",
                "",
                f"- jobs: **{estimate.n_jobs}**",
                f"- episodes: **{estimate.n_episodes_total}**",
                f"- GPU-hours (1×GPU wall): **{estimate.gpu_hours_low}–{estimate.gpu_hours_high}**",
                "",
            ]
        )
        for n in estimate.notes:
            lines.append(f"- {n}")

    lines.extend(["", "## Quadrant reminder", "", "| SS | US |", "|----|----|", "| CFF | CIF |", ""])
    return "\n".join(lines) + "\n"


def plot_feasibility_heatmap(
    matrix_rows: Sequence[Mapping[str, Any]],
    out_path: Path,
) -> Path | None:
    """Categorical heatmap of feasibility_class by task×level."""
    if not matrix_rows:
        return None
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    df = pd.DataFrame(list(matrix_rows))
    if df.empty or "feasibility_class" not in df.columns:
        return None

    class_to_int = {
        "safe-success-feasible": 3,
        "safe-stop-only": 2,
        "unresolved": 1,
        "invalid-geometry": 0,
    }
    df = df.copy()
    df["task_label"] = df.apply(
        lambda r: f"t{int(r['task_id'])}:{str(r.get('task_name', ''))[:24]}", axis=1
    )
    pivot = df.pivot_table(
        index="task_label",
        columns="level",
        values="feasibility_class",
        aggfunc="first",
    )
    data = pivot.map(lambda x: class_to_int.get(str(x), -1) if pd.notna(x) else np.nan)

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * data.shape[1] + 2), max(3, 0.4 * data.shape[0] + 2)))
    im = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=3)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f"L{c}" for c in data.columns])
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(list(data.index))
    ax.set_title("Feasibility class (fixture/authorized runs only)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(
        ["invalid-geometry", "unresolved", "safe-stop-only", "safe-success-feasible"]
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_feasibility_artifacts(
    *,
    out_dir: Path,
    cells: Sequence[CellMetrics],
    compare_rows: Sequence[Mapping[str, Any]],
    matrix_rows: Sequence[Mapping[str, Any]],
    estimate: ResourceEstimate | None,
    status: str,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    cells_path = out_dir / "table_cells.csv"
    pd.DataFrame([c.as_dict() for c in cells]).to_csv(cells_path, index=False)
    paths["cells"] = cells_path

    cmp_path = out_dir / "table_base_vs_oracle.csv"
    pd.DataFrame(list(compare_rows)).to_csv(cmp_path, index=False)
    paths["compare"] = cmp_path

    mat_path = out_dir / "table_feasibility_matrix.csv"
    pd.DataFrame(list(matrix_rows)).to_csv(mat_path, index=False)
    paths["matrix"] = mat_path

    md_path = out_dir / "feasibility_report.md"
    md_path.write_text(
        render_feasibility_markdown(
            compare_rows=compare_rows,
            matrix_rows=matrix_rows,
            estimate=estimate,
            status=status,
        ),
        encoding="utf-8",
    )
    paths["report"] = md_path

    heat = plot_feasibility_heatmap(matrix_rows, out_dir / "plots" / "heatmap_feasibility_class.png")
    if heat is not None:
        paths["heatmap"] = heat

    meta = {
        "status": status,
        "n_cells": len(cells),
        "narrative": narrative_notes(),
        "estimate": None if estimate is None else estimate.as_dict(),
    }
    meta_path = out_dir / "feasibility_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["meta"] = meta_path
    return paths
