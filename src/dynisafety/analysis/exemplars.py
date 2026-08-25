"""Pick representative episodes per behavior class and plot EEF XY traces.

Selection is deterministic. Missing traces skip the figure but still appear
in the CSV so a cell is never silently dropped.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dynisafety.analysis.load import (
    BoundEpisode,
    behavior_class_of,
    episode_source_row,
    load_trace,
    resolve_trace_path,
    resolve_video_path,
)
from dynisafety.analysis.render import render_table
from dynisafety.analysis.termination import cff_bucket, classify_termination
from dynisafety.core.joint_outcomes import classify_episode
from dynisafety.diagnostics.schema import BehaviorClass, Trace

_CLASS_ORDER: tuple[str, ...] = tuple(c.value for c in BehaviorClass)
_CELL_COLS: tuple[str, ...] = ("backend", "level", "behavior_class")
_CFF_PREFER_TIMEOUT = frozenset(
    {
        BehaviorClass.STOP_NO_RECOVER.value,
        BehaviorClass.EVADE_FAIL.value,
        BehaviorClass.NO_RESPONSE.value,
        BehaviorClass.PARTIAL_RESPONSE.value,
    }
)


@dataclass(frozen=True)
class ExemplarRow:
    backend: str
    level: int
    behavior_class: str
    quadrant: str
    termination_kind: str
    cff_bucket: str | None
    run_dir: str
    run_id: str
    model: str
    suite: str
    task_id: int
    episode_index: int
    seed: int
    video_path: str | None
    trace_path: str | None
    plot_path: str | None
    t_onset: int | None
    t_hazard_end: int | None
    first_contact_step: int | None
    steps: int
    max_steps: int
    success: bool
    contact_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "level": self.level,
            "behavior_class": self.behavior_class,
            "quadrant": self.quadrant,
            "termination_kind": self.termination_kind,
            "cff_bucket": self.cff_bucket,
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "model": self.model,
            "suite": self.suite,
            "task_id": self.task_id,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "video_path": self.video_path,
            "trace_path": self.trace_path,
            "plot_path": self.plot_path,
            "t_onset": self.t_onset,
            "t_hazard_end": self.t_hazard_end,
            "first_contact_step": self.first_contact_step,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "success": self.success,
            "contact_steps": self.contact_steps,
        }


def _cell_key(bound: BoundEpisode) -> tuple[str, int, str] | None:
    cls = behavior_class_of(bound.episode)
    if cls is None:
        return None
    ep = bound.episode
    return (str(ep.backend), int(ep.level), cls)


def _prefer_timeout(bound: BoundEpisode) -> bool:
    cls = behavior_class_of(bound.episode)
    if cls not in _CFF_PREFER_TIMEOUT:
        return False
    return classify_episode(bound.episode) == "CFF" and classify_termination(
        bound.episode
    ) == "timeout"


def _rank_tuple(bound: BoundEpisode) -> tuple[int, int, int, str, int]:
    """Lower is better. Prefer a real trace, then a video, then CFF∩timeout."""
    has_trace = 0 if resolve_trace_path(bound) is not None else 1
    has_video = 0 if resolve_video_path(bound) is not None else 1
    timeout_rank = 0 if _prefer_timeout(bound) else 1
    return (
        has_trace,
        has_video,
        timeout_rank,
        bound.episode.run_id,
        int(bound.episode.episode_index),
    )


def select_exemplars(
    bound: Sequence[BoundEpisode],
    *,
    per_class: int = 1,
    classes: Sequence[str] | None = None,
    include_not_triggered: bool = True,
) -> list[BoundEpisode]:
    """Pick ``per_class`` episodes per ``(backend, level, behavior_class)``."""
    if per_class < 1:
        raise ValueError(f"per_class must be >= 1, got {per_class}")
    allowed = set(classes) if classes is not None else set(_CLASS_ORDER)
    if not include_not_triggered:
        allowed.discard(BehaviorClass.NOT_TRIGGERED.value)
    unknown = allowed - set(_CLASS_ORDER)
    if unknown:
        raise ValueError(f"unknown behavior_class values: {sorted(unknown)}")

    cells: dict[tuple[str, int, str], list[BoundEpisode]] = defaultdict(list)
    for item in bound:
        key = _cell_key(item)
        if key is None or key[2] not in allowed:
            continue
        cells[key].append(item)

    chosen: list[BoundEpisode] = []
    for key in sorted(cells, key=lambda k: (k[0], k[1], _CLASS_ORDER.index(k[2]))):
        ranked = sorted(cells[key], key=_rank_tuple)
        chosen.extend(ranked[:per_class])
    return chosen


def _diag_int(ep: Any, key: str) -> int | None:
    diag = ep.diagnostics
    if not isinstance(diag, dict) or diag.get(key) is None:
        return None
    return int(diag[key])


def _row_from_bound(
    bound: BoundEpisode,
    *,
    plot_path: str | None,
) -> ExemplarRow:
    ep = bound.episode
    cls = behavior_class_of(ep)
    if cls is None:
        raise ValueError("exemplar is missing behavior_class")
    trace_path = resolve_trace_path(bound)
    video_path = resolve_video_path(bound)
    extra = episode_source_row(bound)
    return ExemplarRow(
        backend=str(extra["backend"]),
        level=int(extra["level"]),
        behavior_class=cls,
        quadrant=classify_episode(ep),
        termination_kind=classify_termination(ep),
        cff_bucket=cff_bucket(ep),
        run_dir=str(extra["run_dir"]),
        run_id=str(extra["run_id"]),
        model=str(extra["model"]),
        suite=str(extra["suite"]),
        task_id=int(extra["task_id"]),
        episode_index=int(extra["episode_index"]),
        seed=int(extra["seed"]),
        video_path=None if video_path is None else str(video_path),
        trace_path=None if trace_path is None else str(trace_path),
        plot_path=plot_path,
        t_onset=_diag_int(ep, "t_onset"),
        t_hazard_end=_diag_int(ep, "t_hazard_end"),
        first_contact_step=(
            None if ep.first_contact_step is None else int(ep.first_contact_step)
        ),
        steps=int(ep.steps),
        max_steps=int(ep.max_steps),
        success=bool(ep.success),
        contact_steps=int(ep.contact_steps),
    )


def _closest_index(times: Sequence[int], target: int | None) -> int | None:
    if target is None or not times:
        return None
    best_i = 0
    best_d = abs(int(times[0]) - int(target))
    for i, t in enumerate(times):
        d = abs(int(t) - int(target))
        if d < best_d:
            best_i = i
            best_d = d
    return best_i


def plot_exemplar_trajectory(
    trace: Trace,
    out_path: Path | str,
    *,
    first_contact_step: int | None = None,
    title: str | None = None,
) -> None:
    """EEF XY path with onset / closest / hazard-end / contact markers."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    xs = [float(step.eef_pos[0]) for step in trace.steps]
    ys = [float(step.eef_pos[1]) for step in trace.steps]
    times = [int(step.t) for step in trace.steps]
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    if xs:
        ax.plot(xs, ys, color="#4c72b0", linewidth=1.4, label="EEF XY")
        ax.scatter(xs[0], ys[0], c="#55a868", s=28, zorder=3, label="start")
        ax.scatter(xs[-1], ys[-1], c="#4c72b0", s=28, zorder=3, label="end")
    header = trace.header
    onset_i = _closest_index(times, header.t_onset)
    if onset_i is not None:
        ax.scatter(
            xs[onset_i],
            ys[onset_i],
            marker="o",
            facecolors="none",
            edgecolors="#dd8452",
            s=70,
            linewidths=1.6,
            zorder=4,
            label="onset",
        )
    end_i = _closest_index(times, header.t_hazard_end)
    if end_i is not None:
        ax.scatter(
            xs[end_i],
            ys[end_i],
            marker="s",
            facecolors="none",
            edgecolors="#937860",
            s=55,
            linewidths=1.4,
            zorder=4,
            label="hazard_end",
        )
    clearances = [
        (i, float(step.d_eef_obstacle))
        for i, step in enumerate(trace.steps)
        if step.d_eef_obstacle is not None
    ]
    if clearances:
        closest_i = min(clearances, key=lambda item: item[1])[0]
        ax.scatter(
            xs[closest_i],
            ys[closest_i],
            marker="*",
            c="#c44e52",
            s=90,
            zorder=5,
            label="min d_eef",
        )
    contact_i = _closest_index(times, first_contact_step)
    if contact_i is not None:
        ax.scatter(
            xs[contact_i],
            ys[contact_i],
            marker="x",
            c="#8c1515",
            s=55,
            linewidths=1.6,
            zorder=5,
            label="first contact",
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"{header.backend} ep{header.episode_index}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_filename(bound: BoundEpisode) -> str:
    ep = bound.episode
    cls = behavior_class_of(ep) or "unknown"
    return (
        f"{ep.backend}_l{int(ep.level)}_{cls}_"
        f"{ep.model}_t{int(ep.task_id)}_ep{int(ep.episode_index):04d}.png"
    )


def write_exemplars(
    bound: Sequence[BoundEpisode],
    out_dir: Path | str,
    *,
    per_class: int = 1,
    classes: Sequence[str] | None = None,
    include_not_triggered: bool = True,
) -> list[ExemplarRow]:
    """Select exemplars, write ``exemplars.csv`` and trajectory PNGs."""
    out = Path(out_dir)
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    chosen = select_exemplars(
        bound,
        per_class=per_class,
        classes=classes,
        include_not_triggered=include_not_triggered,
    )
    rows: list[ExemplarRow] = []
    for item in chosen:
        plot_rel: str | None = None
        trace = load_trace(item)
        if trace is not None:
            dest = plots_dir / _plot_filename(item)
            cls = behavior_class_of(item.episode) or "unknown"
            plot_exemplar_trajectory(
                trace,
                dest,
                first_contact_step=item.episode.first_contact_step,
                title=(
                    f"{item.episode.backend} L{item.episode.level} "
                    f"{cls} ep{item.episode.episode_index}"
                ),
            )
            plot_rel = str(dest.relative_to(out))
        rows.append(_row_from_bound(item, plot_path=plot_rel))

    table = pd.DataFrame([row.to_dict() for row in rows])
    if table.empty:
        table = pd.DataFrame(columns=list(ExemplarRow.__dataclass_fields__))
    render_table(
        table,
        out,
        "exemplars",
        footnotes=(
            "One (or --per-class) episode per (backend, level, behavior_class).",
            "plot_path is empty when traces/ is missing; the row is still kept.",
            "CFF classes prefer a horizon-timeout episode when one exists.",
        ),
    )
    return rows
