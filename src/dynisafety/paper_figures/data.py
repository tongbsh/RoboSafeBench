"""Read-only loaders for RoboSafeBench paper figures.

Scale comes from ``paper_main.yaml``. Rates are recomputed from completed
``EpisodeRecord`` rows (success × contact_steps). Safety-layer arms are
split out and never silently mixed into the base ranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dynisafety.core.aggregate import (
    discover_episode_files,
    episodes_to_frame,
    load_episodes,
)
from dynisafety.core.joint_outcomes import classify_joint_outcome
from dynisafety.core.metrics import wilson_interval
from dynisafety.paper_figures.paired import PairedAxesFrame, load_paired_axes

# Paper-main tabletop backends only. RoboCasa is not in the ICLR figures.
PAPER_BACKENDS: tuple[str, ...] = ("libero", "simplerenv", "robotwin")
_CELL_KEYS = ("backend", "suite", "task_id", "level", "model", "seed")
_PROCESS_RATE_KEYS = (
    "reaction_rate",
    "stop_rate",
    "evade_rate",
    "recovery_rate",
)


@dataclass(frozen=True)
class PaperSettings:
    benchmark_name: str
    formats: tuple[str, ...]
    dpi: int
    confidence: float
    exclude_models: frozenset[str]
    oracle_base_map: dict[str, str]
    coverage_stamp: bool
    heatmap_max_model_facets: int
    teaser_dir: Path | None
    trajectory_dir: Path | None
    paired_analysis_root: Path
    paired_episodes_relpath: Path


@dataclass(frozen=True)
class DisplayNames:
    backends: dict[str, str]
    levels: dict[int, str]
    level_short: dict[int, str]
    level_intent: dict[int, str]
    models: dict[str, str]
    model_order: tuple[str, ...]
    suites: dict[str, str]
    metrics: dict[str, str]
    quadrants: dict[str, str]


@dataclass(frozen=True)
class ScaleBlock:
    backend: str
    suites: tuple[str, ...]
    n_tasks: int
    n_levels: int
    n_models: int
    n_seeds: int
    episodes_per_task: int
    n_jobs: int
    n_episodes: int
    models: tuple[str, ...]
    task_ids: tuple[int, ...]


@dataclass(frozen=True)
class ScaleTable:
    blocks: tuple[ScaleBlock, ...]
    n_backends: int
    n_tasks: int
    n_jobs: int
    n_episodes: int
    planned: pd.DataFrame


@dataclass(frozen=True)
class PaperBundle:
    settings: PaperSettings
    display: DisplayNames
    peers: tuple[dict[str, Any], ...]
    scale: ScaleTable
    episodes: pd.DataFrame
    runs: pd.DataFrame
    coverage: pd.DataFrame
    n_completed_jobs: int
    n_planned_jobs: int
    paired: PairedAxesFrame

    @property
    def partial(self) -> bool:
        return self.n_completed_jobs < self.n_planned_jobs

    def stamp(self) -> str:
        if not self.settings.coverage_stamp:
            return ""
        return (
            f"partial coverage: {self.n_completed_jobs}/{self.n_planned_jobs} "
            f"paper_main cells"
            if self.partial
            else f"full coverage: {self.n_completed_jobs} paper_main cells"
        )


def load_settings(path: Path) -> PaperSettings:
    raw = _require_mapping(path)
    assets = raw.get("assets") if isinstance(raw.get("assets"), Mapping) else {}
    oracle = raw.get("oracle_base_map")
    if not isinstance(oracle, Mapping) or not oracle:
        raise ValueError(f"{path}: oracle_base_map must be a non-empty mapping")
    exclude = raw.get("exclude_models")
    if not isinstance(exclude, list):
        raise ValueError(f"{path}: exclude_models must be a list")
    formats = raw.get("formats")
    if not isinstance(formats, list) or not formats:
        raise ValueError(f"{path}: formats must be a non-empty list")
    teaser = assets.get("teaser_dir")
    traj = assets.get("trajectory_dir")
    paired = raw.get("paired_axes")
    if not isinstance(paired, Mapping) or "analysis_root" not in paired or "episodes_relpath" not in paired:
        raise ValueError(f"{path}: paired_axes must give analysis_root and episodes_relpath")
    return PaperSettings(
        benchmark_name=str(raw["benchmark_name"]),
        formats=tuple(str(x) for x in formats),
        dpi=int(raw["dpi"]),
        confidence=float(raw["confidence"]),
        exclude_models=frozenset(str(x) for x in exclude),
        oracle_base_map={str(k): str(v) for k, v in oracle.items()},
        coverage_stamp=bool(raw["coverage_stamp"]),
        heatmap_max_model_facets=int(raw["heatmap_max_model_facets"]),
        teaser_dir=Path(str(teaser)) if teaser else None,
        trajectory_dir=Path(str(traj)) if traj else None,
        paired_analysis_root=Path(str(paired["analysis_root"])),
        paired_episodes_relpath=Path(str(paired["episodes_relpath"])),
    )


def load_display(path: Path) -> DisplayNames:
    raw = _require_mapping(path)
    levels = {int(k): str(v) for k, v in dict(raw["levels"]).items()}
    return DisplayNames(
        backends={str(k): str(v) for k, v in dict(raw["backends"]).items()},
        levels=levels,
        level_short={int(k): str(v) for k, v in dict(raw["level_short"]).items()},
        level_intent={int(k): str(v) for k, v in dict(raw["level_intent"]).items()},
        models={str(k): str(v) for k, v in dict(raw["models"]).items()},
        model_order=tuple(str(x) for x in raw["model_order"]),
        suites={str(k): str(v) for k, v in dict(raw["suites"]).items()},
        metrics={str(k): str(v) for k, v in dict(raw["metrics"]).items()},
        quadrants={str(k): str(v) for k, v in dict(raw["quadrants"]).items()},
    )


def load_peers(path: Path) -> tuple[dict[str, Any], ...]:
    raw = _require_mapping(path)
    peers = raw.get("peers")
    if not isinstance(peers, list) or not peers:
        raise ValueError(f"{path}: peers must be a non-empty list")
    return tuple(dict(p) for p in peers if isinstance(p, Mapping))


def expand_paper_main(suite_yaml: Path, *, workspace: Path | None = None) -> ScaleTable:
    """Expand ``paper_main.yaml`` into per-backend counts and a planned-cell table."""
    raw = _require_mapping(suite_yaml)
    if "matrix" not in raw or "seeds" not in raw or "episodes_per_task" not in raw:
        raise ValueError(f"{suite_yaml}: need matrix, seeds, episodes_per_task")
    root = workspace if workspace is not None else suite_yaml.resolve().parents[2]
    global_seeds = tuple(int(s) for s in raw["seeds"])
    global_ep = int(raw["episodes_per_task"])
    blocks: list[ScaleBlock] = []
    planned_rows: list[dict[str, Any]] = []
    for item in raw["matrix"]:
        if not isinstance(item, Mapping):
            raise ValueError("matrix entries must be mappings")
        backend = str(item["backend"])
        suites = tuple(str(s) for s in item["suites"])
        levels = tuple(int(x) for x in item["levels"])
        models = tuple(str(m) for m in item["models"])
        seeds = tuple(int(s) for s in item["seeds"]) if "seeds" in item else global_seeds
        episodes = int(item["episodes_per_task"]) if "episodes_per_task" in item else global_ep
        per_suite_tasks: dict[str, tuple[int, ...]] = {}
        n_tasks = 0
        for suite in suites:
            if "tasks" in item:
                tids = tuple(int(t) for t in item["tasks"])
            else:
                tids = _suite_task_ids(root, backend, suite)
            per_suite_tasks[suite] = tids
            n_tasks += len(tids)
            for task_id in tids:
                for level in levels:
                    for model in models:
                        for seed in seeds:
                            planned_rows.append(
                                {
                                    "backend": backend,
                                    "suite": suite,
                                    "task_id": int(task_id),
                                    "level": int(level),
                                    "model": model,
                                    "seed": int(seed),
                                    "episodes_per_task": episodes,
                                }
                            )
        n_jobs = n_tasks * len(levels) * len(models) * len(seeds)
        union_tasks = tuple(sorted({t for tids in per_suite_tasks.values() for t in tids}))
        blocks.append(
            ScaleBlock(
                backend=backend,
                suites=suites,
                n_tasks=n_tasks,
                n_levels=len(levels),
                n_models=len(models),
                n_seeds=len(seeds),
                episodes_per_task=episodes,
                n_jobs=n_jobs,
                n_episodes=n_jobs * episodes,
                models=models,
                task_ids=union_tasks,
            )
        )
    planned = pd.DataFrame(planned_rows)
    return ScaleTable(
        blocks=tuple(blocks),
        n_backends=len(blocks),
        n_tasks=int(sum(b.n_tasks for b in blocks)),
        n_jobs=int(sum(b.n_jobs for b in blocks)),
        n_episodes=int(sum(b.n_episodes for b in blocks)),
        planned=planned,
    )


def collapse_episodes(
    episodes: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    confidence: float,
) -> pd.DataFrame:
    """One row per group with SR/SSR/CR and quadrant counts. Missing → omit."""
    if episodes.empty:
        return pd.DataFrame()
    needed = {"success", "contact_steps", *group_cols}
    missing = [c for c in needed if c not in episodes.columns]
    if missing:
        raise KeyError(f"episodes frame missing {missing}")
    rows: list[dict[str, Any]] = []
    for key, grp in episodes.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {group_cols[i]: key[i] for i in range(len(group_cols))}
        n = int(len(grp))
        success = grp["success"].astype(bool)
        contact = grp["contact_steps"].astype(int) > 0
        ss = int((success & ~contact).sum())
        us = int((success & contact).sum())
        cff = int((~success & ~contact).sum())
        cif = int((~success & contact).sum())
        row.update(
            {
                "n_episodes": n,
                "ss": ss,
                "us": us,
                "cff": cff,
                "cif": cif,
                "sr": (ss + us) / n,
                "cr": (us + cif) / n,
                "ssr": ss / n,
                "us_rate": us / n,
                "cff_rate": cff / n,
                "cif_rate": cif / n,
            }
        )
        for rate_key, k in (("sr", ss + us), ("cr", us + cif), ("ssr", ss)):
            interval = wilson_interval(float(k), float(n), confidence=confidence)
            if interval is None:
                row[f"{rate_key}_ci_lo"] = None
                row[f"{rate_key}_ci_hi"] = None
            else:
                row[f"{rate_key}_ci_lo"] = interval[0]
                row[f"{rate_key}_ci_hi"] = interval[1]
        rows.append(row)
        _attach_process_rates(row, grp, confidence=confidence)
    return pd.DataFrame(rows)


def load_bundle(
    *,
    results: Path | None,
    suite_yaml: Path,
    settings: PaperSettings,
    display: DisplayNames,
    peers: Sequence[Mapping[str, Any]],
    workspace: Path | None = None,
    analysis_root: Path | None = None,
) -> PaperBundle:
    scale = expand_paper_main(suite_yaml, workspace=workspace)
    root = workspace if workspace is not None else suite_yaml.resolve().parents[2]
    paired = load_paired_axes(
        _resolve_analysis_root(analysis_root, settings, root),
        backends=PAPER_BACKENDS,
        episodes_relpath=settings.paired_episodes_relpath,
    )
    if results is None:
        empty = pd.DataFrame()
        return PaperBundle(
            settings=settings,
            display=display,
            peers=tuple(dict(p) for p in peers),
            scale=scale,
            episodes=empty,
            runs=empty,
            coverage=_coverage_frame(scale.planned, empty),
            n_completed_jobs=0,
            n_planned_jobs=int(len(scale.planned)),
            paired=paired,
        )
    paths = discover_episode_files(Path(results))
    records = load_episodes(paths)
    episodes = episodes_to_frame(records) if records else pd.DataFrame()
    if not episodes.empty and "backend" in episodes.columns:
        episodes = episodes[episodes["backend"].astype(str).isin(PAPER_BACKENDS)].copy()
    if not episodes.empty and "level" in episodes.columns:
        episodes = episodes.copy()
        episodes["level"] = episodes["level"].astype(int)
        episodes["task_id"] = episodes["task_id"].astype(int)
        episodes["seed"] = episodes["seed"].astype(int)
    runs = (
        collapse_episodes(
            episodes,
            ("backend", "model", "suite", "task_id", "level", "seed"),
            confidence=settings.confidence,
        )
        if not episodes.empty
        else pd.DataFrame()
    )
    coverage = _coverage_frame(scale.planned, episodes)
    n_done = int(coverage["completed"].sum()) if not coverage.empty else 0
    return PaperBundle(
        settings=settings,
        display=display,
        peers=tuple(dict(p) for p in peers),
        scale=scale,
        episodes=episodes,
        runs=runs,
        coverage=coverage,
        n_completed_jobs=n_done,
        n_planned_jobs=int(len(scale.planned)),
        paired=paired,
    )


def _resolve_analysis_root(
    analysis_root: Path | None,
    settings: PaperSettings,
    workspace: Path,
) -> Path:
    root = Path(analysis_root) if analysis_root is not None else settings.paired_analysis_root
    return root if root.is_absolute() else Path(workspace) / root


def base_runs(bundle: PaperBundle) -> pd.DataFrame:
    """Runs with safety-layer / oracle models removed."""
    runs = bundle.runs
    if runs.empty or "model" not in runs.columns:
        return runs
    drop = set(bundle.settings.exclude_models) | set(bundle.settings.oracle_base_map)
    return runs[~runs["model"].astype(str).isin(drop)].copy()


def base_episodes(bundle: PaperBundle) -> pd.DataFrame:
    episodes = bundle.episodes
    if episodes.empty or "model" not in episodes.columns:
        return episodes
    drop = set(bundle.settings.exclude_models) | set(bundle.settings.oracle_base_map)
    return episodes[~episodes["model"].astype(str).isin(drop)].copy()


def base_paired_episodes(bundle: PaperBundle) -> pd.DataFrame:
    """Paired rows with safety-layer / oracle arms removed (§2.1b: rank drops them)."""
    episodes = bundle.paired.episodes
    if episodes.empty or "model" not in episodes.columns:
        return episodes
    drop = set(bundle.settings.exclude_models) | set(bundle.settings.oracle_base_map)
    return episodes[~episodes["model"].astype(str).isin(drop)].copy()


def label_model(display: DisplayNames, model: str) -> str:
    return display.models.get(str(model), str(model))


def label_backend(display: DisplayNames, backend: str) -> str:
    return display.backends.get(str(backend), str(backend))


def label_level(display: DisplayNames, level: int, *, short: bool = False) -> str:
    table = display.level_short if short else display.levels
    return table.get(int(level), f"L{int(level)}")


def label_suite(display: DisplayNames, suite: str) -> str:
    return display.suites.get(str(suite), str(suite))


def label_metric(display: DisplayNames, metric: str) -> str:
    return display.metrics.get(str(metric), str(metric).upper())


def ordered_models(display: DisplayNames, models: Sequence[str]) -> list[str]:
    present = {str(m) for m in models}
    out = [m for m in display.model_order if m in present]
    extra = sorted(present - set(out))
    return out + extra


def ordered_backends(display: DisplayNames, backends: Sequence[str]) -> list[str]:
    del display
    present = {str(b) for b in backends if str(b) in PAPER_BACKENDS}
    return [b for b in PAPER_BACKENDS if b in present]


def weighted_collapse(
    runs: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    confidence: float,
) -> pd.DataFrame:
    """Re-collapse run rows by summing quadrant counts (never invent 0)."""
    if runs.empty:
        return runs.copy()
    rows: list[dict[str, Any]] = []
    for key, grp in runs.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {group_cols[i]: key[i] for i in range(len(group_cols))}
        n = int(grp["n_episodes"].astype(float).sum())
        if n <= 0:
            continue
        ss = int(grp["ss"].fillna(0).sum())
        us = int(grp["us"].fillna(0).sum())
        cff = int(grp["cff"].fillna(0).sum())
        cif = int(grp["cif"].fillna(0).sum())
        row.update(
            {
                "n_episodes": n,
                "ss": ss,
                "us": us,
                "cff": cff,
                "cif": cif,
                "sr": (ss + us) / n,
                "cr": (us + cif) / n,
                "ssr": ss / n,
                "us_rate": us / n,
                "cff_rate": cff / n,
                "cif_rate": cif / n,
            }
        )
        for rate_key, k in (("sr", ss + us), ("cr", us + cif), ("ssr", ss)):
            interval = wilson_interval(float(k), float(n), confidence=confidence)
            row[f"{rate_key}_ci_lo"] = None if interval is None else interval[0]
            row[f"{rate_key}_ci_hi"] = None if interval is None else interval[1]
        for col in _PROCESS_RATE_KEYS:
            if col in grp.columns and grp[col].notna().any():
                w = grp["n_episodes"].astype(float)
                mask = grp[col].notna() & w.notna()
                if bool(mask.any()) and float(w[mask].sum()) > 0:
                    row[col] = float((grp.loc[mask, col].astype(float) * w[mask]).sum() / w[mask].sum())
                else:
                    row[col] = None
            elif col in grp.columns:
                row[col] = None
        rows.append(row)
    return pd.DataFrame(rows)


def attach_quadrant(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return episodes
    work = episodes.copy()
    work["quadrant"] = [
        classify_joint_outcome(success=bool(s), contact_steps=int(c)).lower()
        for s, c in zip(work["success"].tolist(), work["contact_steps"].tolist())
    ]
    return work


def _suite_task_ids(workspace: Path, backend: str, suite: str) -> tuple[int, ...]:
    path = Path(workspace) / "configs" / "backends" / backend / "suites" / f"{suite}.yaml"
    raw = _require_mapping(path)
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path}: suite has no tasks: and paper_main omitted tasks")
    ids: list[int] = []
    for item in tasks:
        if isinstance(item, Mapping):
            ids.append(int(item["task_id"]))
        else:
            ids.append(int(item))
    return tuple(ids)


def _coverage_frame(planned: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    if planned.empty:
        return planned.copy()
    cover = planned.copy()
    if episodes.empty:
        cover["completed"] = False
        cover["n_written"] = 0
        return cover
    have = (
        episodes.groupby(list(_CELL_KEYS), dropna=False)
        .size()
        .rename("n_written")
        .reset_index()
    )
    merged = cover.merge(have, on=list(_CELL_KEYS), how="left")
    merged["n_written"] = merged["n_written"].fillna(0).astype(int)
    merged["completed"] = merged["n_written"] > 0
    return merged


def _attach_process_rates(row: dict[str, Any], grp: pd.DataFrame, *, confidence: float) -> None:
    triggered_col = "diag_disturbance_triggered"
    if triggered_col not in grp.columns or not grp[triggered_col].notna().any():
        for key in _PROCESS_RATE_KEYS:
            row[key] = None
        row["n_triggered"] = None
        return
    trig = grp[grp[triggered_col] == True]  # noqa: E712
    n_trig = int(len(trig))
    row["n_triggered"] = n_trig
    if n_trig == 0:
        for key in _PROCESS_RATE_KEYS:
            row[key] = None
        return
    mapping = {
        "reaction_rate": "diag_reacted",
        "stop_rate": "diag_stop_preemptive",
        "evade_rate": "diag_evaded",
    }
    for out_key, src in mapping.items():
        if src not in trig.columns or not trig[src].notna().any():
            row[out_key] = None
            continue
        vals = trig[src].dropna().astype(bool)
        row[out_key] = float(vals.mean()) if len(vals) else None
    recov_col = "diag_recovery_applicable"
    if recov_col in trig.columns and trig[recov_col].notna().any():
        rec = trig[trig[recov_col] == True]  # noqa: E712
        if rec.empty or "diag_recovered" not in rec.columns or not rec["diag_recovered"].notna().any():
            row["recovery_rate"] = None
        else:
            row["recovery_rate"] = float(rec["diag_recovered"].dropna().astype(bool).mean())
    else:
        row["recovery_rate"] = None
    del confidence  # rates here are point estimates; Wilson is on SR/SSR/CR


def _require_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(raw)
