"""Rank-stability appendix. Not a headline metric and not part of report generate.

Ranks models by SSR / SR / CR, then correlates those ranks across backends,
seeds, and optional hold-out suites. A single seed or a single model is a
real gap: pairwise correlations are n/a, never a silent 0.0.

Reads episodes (or an existing runs.csv). Does not import backends or
``dynisafety.report``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dynisafety.analysis.load import BoundEpisode, group_key_tuple, is_safety_layer_model
from dynisafety.analysis.render import render_table
from dynisafety.core.joint_outcomes import ssr
from dynisafety.core.metrics import cr, sr

DEFAULT_METRICS: tuple[str, ...] = ("ssr", "sr", "cr")
_LOWER_BETTER: frozenset[str] = frozenset({"cr"})
_HOLDOUT_MARKERS: tuple[str, ...] = ("holdout", "hold_out", "hold-out")
_DESKTOP_BACKENDS: tuple[str, ...] = ("libero", "robotwin", "simplerenv")


def higher_is_better(metric: str) -> bool:
    return metric not in _LOWER_BETTER


def is_holdout_suite(suite: str, extra: Sequence[str] = ()) -> bool:
    if extra and suite in extra:
        return True
    name = str(suite).lower()
    return any(marker in name for marker in _HOLDOUT_MARKERS)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or a.size != b.size:
        return None
    a = a - float(a.mean())
    b = b - float(b.mean())
    sa = float(np.sqrt(np.sum(a * a)))
    sb = float(np.sqrt(np.sum(b * b)))
    if sa < 1e-15 or sb < 1e-15:
        return None
    return float(np.sum(a * b) / (sa * sb))


def average_ranks(values: Sequence[float], *, higher_is_better: bool) -> list[float]:
    """Competition ranks with averages on ties. Rank 1 is best."""
    n = len(values)
    oriented = [(-float(v) if not higher_is_better else float(v)) for v in values]
    order = sorted(range(n), key=lambda i: oriented[i], reverse=True)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and oriented[order[j + 1]] == oriented[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(
    left: Sequence[float],
    right: Sequence[float],
    *,
    higher_is_better: bool,
) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    return _pearson(
        average_ranks(left, higher_is_better=higher_is_better),
        average_ranks(right, higher_is_better=higher_is_better),
    )


def kendall_tau(
    left: Sequence[float],
    right: Sequence[float],
    *,
    higher_is_better: bool,
) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    sign = 1.0 if higher_is_better else -1.0
    a = [sign * float(v) for v in left]
    b = [sign * float(v) for v in right]
    conc = disc = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da = (a[i] > a[j]) - (a[i] < a[j])
            db = (b[i] > b[j]) - (b[i] < b[j])
            if da == 0 or db == 0:
                continue
            if da == db:
                conc += 1
            else:
                disc += 1
    denom = conc + disc
    if denom == 0:
        return None
    return float(conc - disc) / float(denom)


def _metric_value(episodes: Sequence[Any], metric: str) -> float | None:
    if not episodes:
        return None
    if metric == "sr":
        return float(sr(episodes))
    if metric == "cr":
        return float(cr(episodes))
    if metric == "ssr":
        return float(ssr(episodes))
    raise ValueError(f"unsupported rank metric {metric!r}; expected ssr/sr/cr")


def score_cells(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """One row per group × metric from episodes."""
    columns = [*group_cols, "n", "metric", "value"]
    buckets: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for item in bound:
        buckets[group_key_tuple(item.episode, group_cols)].append(item.episode)
    rows: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        eps = buckets[key]
        for metric in metrics:
            rec: dict[str, Any] = {col: val for col, val in zip(group_cols, key, strict=True)}
            rec.update({"n": len(eps), "metric": metric, "value": _metric_value(eps, metric)})
            rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def cells_from_runs_csv(
    path: Path | str,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Collapse a report ``runs.csv`` to (backend, model, level, seed, suite)."""
    frame = pd.read_csv(Path(path))
    needed = {"backend", "model", "level", "n_episodes"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"runs.csv missing columns: {sorted(missing)}")
    if "seed" not in frame.columns:
        frame["seed"] = 0
    if "suite" not in frame.columns:
        frame["suite"] = ""
    group_cols = ["backend", "model", "level", "seed", "suite"]
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_cols, dropna=False)
    for key, chunk in grouped:
        weights = chunk["n_episodes"].to_numpy(dtype=np.float64)
        n = int(np.sum(weights))
        for metric in metrics:
            if metric not in chunk.columns:
                value = None
            else:
                vals = chunk[metric].to_numpy(dtype=np.float64)
                if n <= 0 or not np.isfinite(vals).all() or not np.isfinite(weights).all():
                    value = None
                else:
                    value = float(np.sum(vals * weights) / float(n))
            rec = {col: val for col, val in zip(group_cols, key, strict=True)}
            rec.update({"n": n, "metric": metric, "value": value})
            rows.append(rec)
    columns = [*group_cols, "n", "metric", "value"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _collapse(
    cells: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Re-pool already-scored cells by weighted mean of ``value``."""
    if cells.empty:
        return cells.iloc[0:0].copy()
    rows: list[dict[str, Any]] = []
    grouped = cells.groupby([*group_cols, "metric"], dropna=False)
    for key, chunk in grouped:
        weights = chunk["n"].to_numpy(dtype=np.float64)
        vals = chunk["value"].to_numpy(dtype=np.float64)
        n = int(np.sum(weights))
        if n <= 0 or not np.isfinite(vals).all():
            value = None
        else:
            value = float(np.sum(vals * weights) / float(n))
        rec = {col: val for col, val in zip([*group_cols, "metric"], key, strict=True)}
        rec.update({"n": n, "value": value})
        rows.append(rec)
    return pd.DataFrame(rows)


def rank_models(cells: pd.DataFrame, *, slice_cols: Sequence[str]) -> pd.DataFrame:
    """Add ``rank`` / ``n_models`` within each slice × metric. Ties average."""
    columns = [
        *slice_cols,
        "model",
        "metric",
        "value",
        "n",
        "rank",
        "n_models",
    ]
    if cells.empty or "model" not in cells.columns:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = cells.groupby([*slice_cols, "metric"], dropna=False)
    for key, chunk in grouped:
        usable = chunk.dropna(subset=["value"])
        models = [str(m) for m in usable["model"].tolist()]
        values = [float(v) for v in usable["value"].tolist()]
        ns = [int(v) for v in usable["n"].tolist()]
        metric = str(key[-1])
        if not models:
            continue
        ranks = average_ranks(values, higher_is_better=higher_is_better(metric))
        n_models = len(models)
        for model, value, n_ep, rank in zip(models, values, ns, ranks, strict=True):
            rec = {col: val for col, val in zip([*slice_cols, "metric"], key, strict=True)}
            rec.update(
                {
                    "model": model,
                    "value": value,
                    "n": n_ep,
                    "rank": rank,
                    "n_models": n_models,
                }
            )
            rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _pair_corr(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    metric: str,
) -> dict[str, Any]:
    merged = left.merge(right, on="model", suffixes=("_a", "_b"))
    shared = merged.dropna(subset=["value_a", "value_b"])
    n = int(len(shared))
    better = higher_is_better(metric)
    if n < 2:
        return {
            "n_shared_models": n,
            "spearman": None,
            "kendall": None,
        }
    return {
        "n_shared_models": n,
        "spearman": spearman_rho(
            shared["value_a"].tolist(),
            shared["value_b"].tolist(),
            higher_is_better=better,
        ),
        "kendall": kendall_tau(
            shared["value_a"].tolist(),
            shared["value_b"].tolist(),
            higher_is_better=better,
        ),
    }


def table_rank_backend_pairs(cells: pd.DataFrame) -> pd.DataFrame:
    """Model-rank correlation between backends, per level × metric."""
    columns = [
        "level",
        "metric",
        "backend_a",
        "backend_b",
        "n_shared_models",
        "spearman",
        "kendall",
    ]
    pooled = _collapse(cells, ("backend", "model", "level"))
    if pooled.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (level, metric), chunk in pooled.groupby(["level", "metric"], dropna=False):
        backends = sorted({str(b) for b in chunk["backend"].tolist()})
        by_backend = {
            name: chunk[chunk["backend"] == name][["model", "value"]]
            for name in backends
        }
        for i, a in enumerate(backends):
            for b in backends[i + 1 :]:
                rec = {
                    "level": level,
                    "metric": metric,
                    "backend_a": a,
                    "backend_b": b,
                    **_pair_corr(by_backend[a], by_backend[b], metric=str(metric)),
                }
                rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_rank_seed_pairs(cells: pd.DataFrame) -> pd.DataFrame:
    """Model-rank correlation between seeds, per backend × level × metric."""
    columns = [
        "backend",
        "level",
        "metric",
        "seed_a",
        "seed_b",
        "n_shared_models",
        "spearman",
        "kendall",
    ]
    if "seed" not in cells.columns:
        return pd.DataFrame(columns=columns)
    pooled = _collapse(cells, ("backend", "model", "level", "seed"))
    if pooled.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    keys = ["backend", "level", "metric"]
    for key, chunk in pooled.groupby(keys, dropna=False):
        seeds = sorted({int(s) for s in chunk["seed"].tolist()})
        if len(seeds) < 2:
            continue
        by_seed = {
            seed: chunk[chunk["seed"] == seed][["model", "value"]] for seed in seeds
        }
        for i, a in enumerate(seeds):
            for b in seeds[i + 1 :]:
                rec = {
                    "backend": key[0],
                    "level": key[1],
                    "metric": key[2],
                    "seed_a": a,
                    "seed_b": b,
                    **_pair_corr(by_seed[a], by_seed[b], metric=str(key[2])),
                }
                rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_rank_holdout(
    cells: pd.DataFrame,
    *,
    holdout_suites: Sequence[str] = (),
) -> pd.DataFrame:
    """Core vs hold-out rank correlation. Empty when no hold-out suite exists."""
    columns = [
        "backend",
        "level",
        "metric",
        "n_core",
        "n_holdout",
        "n_shared_models",
        "spearman",
        "kendall",
    ]
    if "suite" not in cells.columns or cells.empty:
        return pd.DataFrame(columns=columns)
    flagged = cells.copy()
    flagged["split"] = [
        "holdout" if is_holdout_suite(str(s), holdout_suites) else "core"
        for s in flagged["suite"].tolist()
    ]
    if "holdout" not in set(flagged["split"]):
        return pd.DataFrame(columns=columns)
    pooled = _collapse(flagged, ("backend", "model", "level", "split"))
    rows: list[dict[str, Any]] = []
    for (backend, level, metric), chunk in pooled.groupby(
        ["backend", "level", "metric"], dropna=False
    ):
        core = chunk[chunk["split"] == "core"][["model", "value"]]
        hold = chunk[chunk["split"] == "holdout"][["model", "value"]]
        rec = {
            "backend": backend,
            "level": level,
            "metric": metric,
            "n_core": int(core["model"].nunique()) if not core.empty else 0,
            "n_holdout": int(hold["model"].nunique()) if not hold.empty else 0,
            **_pair_corr(core, hold, metric=str(metric)),
        }
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_rank_coverage(
    cells: pd.DataFrame,
    *,
    holdout_suites: Sequence[str] = (),
) -> pd.DataFrame:
    """What the current results can and cannot support."""
    if cells.empty:
        n_models = n_backends = n_seeds = n_suites = n_holdout = 0
        n_levels = 0
        desktop = 0
    else:
        n_models = int(cells["model"].nunique()) if "model" in cells.columns else 0
        n_backends = int(cells["backend"].nunique()) if "backend" in cells.columns else 0
        n_seeds = int(cells["seed"].nunique()) if "seed" in cells.columns else 0
        n_suites = int(cells["suite"].nunique()) if "suite" in cells.columns else 0
        n_levels = int(cells["level"].nunique()) if "level" in cells.columns else 0
        suites = [str(s) for s in cells["suite"].tolist()] if "suite" in cells.columns else []
        n_holdout = len({s for s in suites if is_holdout_suite(s, holdout_suites)})
        backends = {str(b) for b in cells["backend"].tolist()} if "backend" in cells.columns else set()
        desktop = len(backends & set(_DESKTOP_BACKENDS))
    notes = []
    if n_models < 2:
        notes.append("fewer than 2 models: all pairwise correlations are n/a")
    if n_backends < 2:
        notes.append("fewer than 2 backends: cross-backend Spearman is n/a")
    if n_seeds < 2:
        notes.append("single seed: seed-pair table stays empty until a second seed exists")
    if n_holdout == 0:
        notes.append("no hold-out suite: core vs hold-out is n/a (not 0.0)")
    return pd.DataFrame(
        [
            {
                "n_cells": int(len(cells)),
                "n_models": n_models,
                "n_backends": n_backends,
                "n_desktop_backends": desktop,
                "n_levels": n_levels,
                "n_seeds": n_seeds,
                "n_suites": n_suites,
                "n_holdout_suites": n_holdout,
                "note": "; ".join(notes) if notes else "enough axes for pairwise ranks",
            }
        ]
    )


_CELL_NOTES: tuple[str, ...] = (
    "Score cells pooled from episodes (or a weighted collapse of runs.csv).",
    "CR is lower-better when ranked. SSR / SR are higher-better.",
)

_PAIR_NOTES: tuple[str, ...] = (
    "Spearman / Kendall on shared models only. n_shared_models < 2 → n/a, not 0.0.",
    "Do not treat a single-seed or single-model run as evidence of rank stability.",
    "Backends do not share a clock; this is rank agreement, not a pooled SSR.",
)

_SEED_NOTES: tuple[str, ...] = (
    "Empty table means the results tree has fewer than 2 seeds — expected for paper_main.",
)

_HOLDOUT_NOTES: tuple[str, ...] = (
    "Hold-out suites are names containing holdout / hold_out, or --holdout-suites.",
    "Empty table means no hold-out suite was present.",
)

_COV_NOTES: tuple[str, ...] = (
    "Coverage is the honest status of the current artifacts, not a metric.",
    "Safety-layer arms ({base}_oracle_*) are dropped from rank unless "
    "--models names them. Comparing a filter to a bare VLA is not a rank.",
)


def _empty_figure(out: Path, message: str) -> None:
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_rank_backend_heatmap(
    pairs: pd.DataFrame,
    out_path: Path | str,
    *,
    metric: str = "ssr",
) -> None:
    """Spearman heatmap per level for one metric. n/a when a pair cannot be scored."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    work = pairs[pairs["metric"] == metric] if not pairs.empty else pairs
    scored = work.dropna(subset=["spearman"]) if not work.empty else work
    if scored is None or scored.empty:
        _empty_figure(out, f"rank Spearman {metric} (n/a — need ≥2 models on ≥2 backends)")
        return
    levels = sorted({int(lv) for lv in scored["level"].tolist()})
    fig, axes = plt.subplots(
        1,
        len(levels),
        figsize=(3.4 * len(levels) + 1.4, 3.6),
        squeeze=False,
    )
    for ax, level in zip(axes[0], levels, strict=True):
        sub = scored[scored["level"] == level]
        names = sorted(
            {str(x) for x in list(sub["backend_a"]) + list(sub["backend_b"])}
        )
        mat = np.full((len(names), len(names)), np.nan, dtype=np.float64)
        idx = {name: i for i, name in enumerate(names)}
        for i in range(len(names)):
            mat[i, i] = 1.0
        for _, row in sub.iterrows():
            i = idx[str(row["backend_a"])]
            j = idx[str(row["backend_b"])]
            val = float(row["spearman"])
            mat[i, j] = val
            mat[j, i] = val
        im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"L{level} {metric}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Model-rank Spearman across backends (n/a cells stay blank)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_rank(
    bound: Sequence[BoundEpisode],
    out_dir: Path | str,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    holdout_suites: Sequence[str] = (),
    runs_csv: Path | str | None = None,
    exclude_safety_layers: bool = True,
) -> dict[str, pd.DataFrame]:
    """Write rank cells, pairwise correlations, coverage, and a heatmap."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if runs_csv is not None:
        cells = cells_from_runs_csv(runs_csv, metrics=metrics)
    else:
        cells = score_cells(
            bound,
            group_cols=("backend", "model", "level", "seed", "suite"),
            metrics=metrics,
        )
    if exclude_safety_layers and not cells.empty and "model" in cells.columns:
        cells = cells.loc[~cells["model"].map(is_safety_layer_model)].copy()
    ranks = rank_models(cells, slice_cols=("backend", "level", "seed", "suite"))
    pairs = table_rank_backend_pairs(cells)
    seeds = table_rank_seed_pairs(cells)
    holdout = table_rank_holdout(cells, holdout_suites=holdout_suites)
    coverage = table_rank_coverage(cells, holdout_suites=holdout_suites)
    tables = {
        "rank_cells": (cells, _CELL_NOTES),
        "table_rank_models": (ranks, _CELL_NOTES),
        "table_rank_backend_pairs": (pairs, _PAIR_NOTES),
        "table_rank_seed_pairs": (seeds, _SEED_NOTES),
        "table_rank_holdout": (holdout, _HOLDOUT_NOTES),
        "table_rank_coverage": (coverage, _COV_NOTES),
    }
    written: dict[str, pd.DataFrame] = {}
    for stem, (frame, notes) in tables.items():
        render_table(frame, out, stem, footnotes=notes)
        written[stem] = frame
    plot_rank_backend_heatmap(pairs, out / "plots" / "rank_spearman_ssr.png", metric="ssr")
    return written
