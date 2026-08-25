"""Appendix kinematics from stored traces. Not a headline metric.

Path length, XY heading reversals, and step-index LDLJ. Step clocks are
backend-local, so every table is grouped by backend and must not be ranked
across simulators. Missing traces render n/a, never a silent 0.0.

This package does not write into ``dynisafety report generate`` or
``diagnostics.analyze`` — those stay the official 9-class path.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dynisafety.analysis.load import (
    BoundEpisode,
    behavior_class_of,
    episode_source_row,
    group_key_tuple,
    load_trace,
)
from dynisafety.analysis.render import render_table
from dynisafety.analysis.termination import classify_termination
from dynisafety.core.joint_outcomes import classify_episode
from dynisafety.diagnostics.schema import Trace

# Analysis-only gates (not diagnostic YAML). Footnoted on every table.
MIN_SEGMENT_M = 1e-4
MIN_TURN_DEG = 25.0
_LDLJ_MIN_POINTS = 5
_PAPER_GROUP_COLS: tuple[str, ...] = ("backend", "model", "level")
_QUADRANTS: tuple[str, ...] = ("SS", "US", "CFF", "CIF")


@dataclass(frozen=True)
class KinematicsMetrics:
    n_records: int
    stride: int
    control_freq_hz: float | None
    path_length_full_m: float | None
    net_disp_full_m: float | None
    detour_ratio_full: float | None
    n_heading_reversals_full: int | None
    ldlj_steps_full: float | None
    path_length_post_onset_m: float | None
    net_disp_post_onset_m: float | None
    detour_ratio_post_onset: float | None
    n_heading_reversals_post_onset: int | None
    ldlj_steps_post_onset: float | None
    window_onset_t: int | None
    has_trace: bool

    def to_metric_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "stride": self.stride,
            "control_freq_hz": self.control_freq_hz,
            "path_length_full_m": self.path_length_full_m,
            "net_disp_full_m": self.net_disp_full_m,
            "detour_ratio_full": self.detour_ratio_full,
            "n_heading_reversals_full": self.n_heading_reversals_full,
            "ldlj_steps_full": self.ldlj_steps_full,
            "path_length_post_onset_m": self.path_length_post_onset_m,
            "net_disp_post_onset_m": self.net_disp_post_onset_m,
            "detour_ratio_post_onset": self.detour_ratio_post_onset,
            "n_heading_reversals_post_onset": self.n_heading_reversals_post_onset,
            "ldlj_steps_post_onset": self.ldlj_steps_post_onset,
            "window_onset_t": self.window_onset_t,
            "has_trace": self.has_trace,
        }


def _empty_metrics() -> KinematicsMetrics:
    return KinematicsMetrics(
        n_records=0,
        stride=1,
        control_freq_hz=None,
        path_length_full_m=None,
        net_disp_full_m=None,
        detour_ratio_full=None,
        n_heading_reversals_full=None,
        ldlj_steps_full=None,
        path_length_post_onset_m=None,
        net_disp_post_onset_m=None,
        detour_ratio_post_onset=None,
        n_heading_reversals_post_onset=None,
        ldlj_steps_post_onset=None,
        window_onset_t=None,
        has_trace=False,
    )


def _positions_and_times(trace: Trace) -> tuple[np.ndarray, np.ndarray]:
    n = len(trace.steps)
    times = np.empty(n, dtype=np.int64)
    pts = np.empty((n, 3), dtype=np.float64)
    for i, step in enumerate(trace.steps):
        times[i] = int(step.t)
        pts[i, 0] = float(step.eef_pos[0])
        pts[i, 1] = float(step.eef_pos[1])
        pts[i, 2] = float(step.eef_pos[2])
    return pts, times


def path_length_m(pts: np.ndarray) -> float | None:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def net_displacement_m(pts: np.ndarray) -> float | None:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    return float(np.linalg.norm(arr[-1] - arr[0]))


def detour_ratio(path: float | None, net: float | None) -> float | None:
    if path is None or net is None or net < MIN_SEGMENT_M:
        return None
    return float(path / net)


def heading_reversals(
    pts: np.ndarray,
    *,
    min_segment_m: float = MIN_SEGMENT_M,
    min_turn_deg: float = MIN_TURN_DEG,
) -> int | None:
    """Count XY turn-direction flips on segments longer than ``min_segment_m``.

    ``None`` when two directed segments cannot be formed. ``0`` means the
    heading never reversed (a real observation).
    """
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return None
    dxy = np.diff(arr[:, :2], axis=0)
    speeds = np.linalg.norm(dxy, axis=1)
    vecs = dxy[speeds > float(min_segment_m)]
    if vecs.shape[0] < 2:
        return None
    turns: list[int] = []
    gate = float(min_turn_deg)
    for i in range(vecs.shape[0] - 1):
        a = vecs[i]
        b = vecs[i + 1]
        cross = float(a[0] * b[1] - a[1] * b[0])
        dot = float(a[0] * b[0] + a[1] * b[1])
        angle = float(np.degrees(np.arctan2(cross, dot)))
        if abs(angle) >= gate:
            turns.append(1 if angle > 0.0 else -1)
    if len(turns) < 2:
        return 0
    return int(sum(1 for i in range(1, len(turns)) if turns[i] != turns[i - 1]))


def ldlj_steps(pts: np.ndarray) -> float | None:
    """Log dimensionless jerk on the record-index clock (``dt = 1``).

    ``LDLJ = -ln( T^5 / v_peak^2 * sum ||jerk||^2 )``. More negative is
    jerkier. ``None`` when there are too few points, no motion, or the
    integrated jerk is ~0 (constant velocity). Never convert to seconds —
    that would invite a cross-backend ranking.
    """
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < _LDLJ_MIN_POINTS:
        return None
    vel = np.diff(arr, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    v_peak = float(np.max(np.linalg.norm(vel, axis=1)))
    if v_peak < MIN_SEGMENT_M:
        return None
    t_span = float(arr.shape[0] - 1)
    jerk_energy = float(np.sum(np.square(jerk)))
    dj = ((t_span**5) / (v_peak**2)) * jerk_energy
    if not np.isfinite(dj) or dj <= 0.0:
        return None
    return float(-np.log(dj))


def _window_metrics(pts: np.ndarray) -> tuple[float | None, float | None, float | None, int | None, float | None]:
    path = path_length_m(pts)
    net = net_displacement_m(pts)
    return path, net, detour_ratio(path, net), heading_reversals(pts), ldlj_steps(pts)


def _slice_from_onset(pts: np.ndarray, times: np.ndarray, t_onset: int) -> np.ndarray | None:
    mask = times >= int(t_onset)
    if int(np.count_nonzero(mask)) < 2:
        return None
    return pts[mask]


def measure_trace(trace: Trace) -> KinematicsMetrics:
    """Pure geometry on one ``Trace``. Does not import a backend."""
    pts, times = _positions_and_times(trace)
    path, net, detour, revs, ldlj = _window_metrics(pts)
    onset = trace.header.t_onset if trace.header.disturbance_triggered else None
    post = _slice_from_onset(pts, times, int(onset)) if onset is not None else None
    if post is None:
        post_vals = (None, None, None, None, None)
    else:
        post_vals = _window_metrics(post)
    freq = trace.header.control_freq_hz
    return KinematicsMetrics(
        n_records=int(len(trace.steps)),
        stride=int(trace.header.stride),
        control_freq_hz=None if freq is None else float(freq),
        path_length_full_m=path,
        net_disp_full_m=net,
        detour_ratio_full=detour,
        n_heading_reversals_full=revs,
        ldlj_steps_full=ldlj,
        path_length_post_onset_m=post_vals[0],
        net_disp_post_onset_m=post_vals[1],
        detour_ratio_post_onset=post_vals[2],
        n_heading_reversals_post_onset=post_vals[3],
        ldlj_steps_post_onset=post_vals[4],
        window_onset_t=None if onset is None else int(onset),
        has_trace=True,
    )


def collect_kinematics(
    bound: Sequence[BoundEpisode],
    *,
    traces: Sequence[Trace | None] | None = None,
) -> list[dict[str, Any]]:
    """One dict per episode. ``traces`` is a test override; default is disk."""
    if traces is not None and len(traces) != len(bound):
        raise ValueError("traces override must align 1:1 with bound episodes")
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(bound):
        loaded = traces[i] if traces is not None else load_trace(item)
        metrics = measure_trace(loaded) if loaded is not None else _empty_metrics()
        ep = item.episode
        row = episode_source_row(item)
        row.update(metrics.to_metric_dict())
        row["quadrant"] = classify_episode(ep)
        row["termination_kind"] = classify_termination(ep)
        row["behavior_class"] = behavior_class_of(ep)
        row["success"] = bool(ep.success)
        row["contact_steps"] = int(ep.contact_steps)
        rows.append(row)
    return rows


def _median(values: Sequence[Any]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not finite:
        return None
    return float(np.median(np.asarray(finite, dtype=np.float64)))


def _subset(rows: Sequence[dict[str, Any]], **equals: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if all(row.get(k) == v for k, v in equals.items()):
            out.append(row)
    return out


def _median_field(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    return _median([row.get(field) for row in rows if row.get("has_trace")])


def table_kinematics_episodes(
    bound: Sequence[BoundEpisode],
    *,
    traces: Sequence[Trace | None] | None = None,
) -> pd.DataFrame:
    columns = [
        "run_dir",
        "run_id",
        "backend",
        "model",
        "suite",
        "task_id",
        "task_name",
        "level",
        "seed",
        "episode_index",
        "quadrant",
        "termination_kind",
        "behavior_class",
        "success",
        "contact_steps",
        "has_trace",
        "n_records",
        "stride",
        "control_freq_hz",
        "path_length_full_m",
        "net_disp_full_m",
        "detour_ratio_full",
        "n_heading_reversals_full",
        "ldlj_steps_full",
        "path_length_post_onset_m",
        "net_disp_post_onset_m",
        "detour_ratio_post_onset",
        "n_heading_reversals_post_onset",
        "ldlj_steps_post_onset",
        "window_onset_t",
    ]
    rows = collect_kinematics(bound, traces=traces)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_kinematics(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
    traces: Sequence[Trace | None] | None = None,
) -> pd.DataFrame:
    """Per-backend medians. Empty-trace groups keep n/a, never 0.0."""
    columns = [
        *group_cols,
        "n",
        "n_with_trace",
        "median_path_length_full_m",
        "median_detour_ratio_full",
        "median_n_heading_reversals_full",
        "median_ldlj_steps_full",
        "median_path_length_post_onset_m",
        "n_cff_traced",
        "median_path_cff_m",
        "median_detour_cff",
        "n_ss_traced",
        "median_path_ss_m",
        "median_detour_ss",
        "n_timeout_traced",
        "median_path_timeout_m",
    ]
    records = collect_kinematics(bound, traces=traces)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item, row in zip(bound, records, strict=True):
        groups[group_key_tuple(item.episode, group_cols)].append(row)
    out_rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rows = groups[key]
        traced = [r for r in rows if r.get("has_trace")]
        cff = _subset(traced, quadrant="CFF")
        ss = _subset(traced, quadrant="SS")
        timeout = _subset(traced, termination_kind="timeout")
        rec: dict[str, Any] = {col: val for col, val in zip(group_cols, key, strict=True)}
        rec.update(
            {
                "n": len(rows),
                "n_with_trace": len(traced),
                "median_path_length_full_m": _median_field(rows, "path_length_full_m"),
                "median_detour_ratio_full": _median_field(rows, "detour_ratio_full"),
                "median_n_heading_reversals_full": _median_field(
                    rows, "n_heading_reversals_full"
                ),
                "median_ldlj_steps_full": _median_field(rows, "ldlj_steps_full"),
                "median_path_length_post_onset_m": _median_field(
                    rows, "path_length_post_onset_m"
                ),
                "n_cff_traced": len(cff),
                "median_path_cff_m": _median_field(cff, "path_length_full_m"),
                "median_detour_cff": _median_field(cff, "detour_ratio_full"),
                "n_ss_traced": len(ss),
                "median_path_ss_m": _median_field(ss, "path_length_full_m"),
                "median_detour_ss": _median_field(ss, "detour_ratio_full"),
                "n_timeout_traced": len(timeout),
                "median_path_timeout_m": _median_field(timeout, "path_length_full_m"),
            }
        )
        out_rows.append(rec)
    if not out_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(out_rows, columns=columns)


def table_kinematics_by_quadrant(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
    traces: Sequence[Trace | None] | None = None,
) -> pd.DataFrame:
    columns = [
        *group_cols,
        "quadrant",
        "n",
        "n_with_trace",
        "median_path_length_full_m",
        "median_detour_ratio_full",
        "median_n_heading_reversals_full",
        "median_ldlj_steps_full",
    ]
    records = collect_kinematics(bound, traces=traces)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item, row in zip(bound, records, strict=True):
        key = (*group_key_tuple(item.episode, group_cols), row["quadrant"])
        groups[key].append(row)
    out_rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rows = groups[key]
        traced = [r for r in rows if r.get("has_trace")]
        rec: dict[str, Any] = {col: val for col, val in zip(group_cols, key[:-1], strict=True)}
        rec.update(
            {
                "quadrant": key[-1],
                "n": len(rows),
                "n_with_trace": len(traced),
                "median_path_length_full_m": _median_field(rows, "path_length_full_m"),
                "median_detour_ratio_full": _median_field(rows, "detour_ratio_full"),
                "median_n_heading_reversals_full": _median_field(
                    rows, "n_heading_reversals_full"
                ),
                "median_ldlj_steps_full": _median_field(rows, "ldlj_steps_full"),
            }
        )
        out_rows.append(rec)
    if not out_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(out_rows, columns=columns)


_TABLE_NOTES: tuple[str, ...] = (
    "Appendix only. Do not rank path length / LDLJ / reversals across backends.",
    "LDLJ uses the record-index clock (dt=1), not seconds. control_freq_hz is provenance.",
    "Missing traces → n/a. n_with_trace==0 does not become a silent 0.0 median.",
    f"Heading reversal: XY turn ≥ {MIN_TURN_DEG:.0f}° then opposite sign. "
    f"Segments shorter than {MIN_SEGMENT_M} m are ignored.",
    "detour_ratio = path_length / net_displacement; n/a when the net move is ~0.",
    "CFF vs SS medians support “contact-free failure wandered / oscillated”, not a ranking.",
)

_QUAD_NOTES: tuple[str, ...] = (
    "Same backend+level only. SS / US / CFF / CIF use the locked joint-outcome names.",
    "n_with_trace==0 → medians are n/a.",
)


def _empty_figure(out: Path, message: str) -> None:
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_kinematics_path_by_quadrant(
    bound: Sequence[BoundEpisode],
    out_path: Path | str,
    *,
    traces: Sequence[Trace | None] | None = None,
) -> None:
    """Boxplot of full path length by quadrant, one panel per backend."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    records = [r for r in collect_kinematics(bound, traces=traces) if r.get("has_trace")]
    usable = [r for r in records if r.get("path_length_full_m") is not None]
    if not usable:
        _empty_figure(out, "kinematics path length (n/a — no traces)")
        return
    backends = sorted({str(r["backend"]) for r in usable})
    fig, axes = plt.subplots(
        1,
        len(backends),
        figsize=(3.6 * len(backends) + 1.2, 3.8),
        squeeze=False,
    )
    for ax, backend in zip(axes[0], backends, strict=True):
        rows = [r for r in usable if str(r["backend"]) == backend]
        data = []
        labels = []
        for quad in _QUADRANTS:
            vals = [
                float(r["path_length_full_m"])
                for r in rows
                if r.get("quadrant") == quad and r.get("path_length_full_m") is not None
            ]
            if vals:
                data.append(vals)
                labels.append(quad)
        if not data:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_title(backend)
        ax.set_ylabel("path_length_full_m")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("EEF path length by quadrant (within backend)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_kinematics_detour_cff_vs_ss(
    bound: Sequence[BoundEpisode],
    out_path: Path | str,
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
    traces: Sequence[Trace | None] | None = None,
) -> None:
    """Median detour ratio, CFF vs SS, one bar group per backend+level cell."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    table = table_kinematics(bound, group_cols=group_cols, traces=traces)
    if table.empty or int(table["n_with_trace"].sum()) == 0:
        _empty_figure(out, "kinematics detour CFF vs SS (n/a — no traces)")
        return
    labels = [" / ".join(str(row[c]) for c in group_cols) for _, row in table.iterrows()]
    cff = [
        float(v) if v is not None and np.isfinite(v) else np.nan
        for v in table["median_detour_cff"].tolist()
    ]
    ss = [
        float(v) if v is not None and np.isfinite(v) else np.nan
        for v in table["median_detour_ss"].tolist()
    ]
    if not np.isfinite(cff).any() and not np.isfinite(ss).any():
        _empty_figure(out, "kinematics detour CFF vs SS (n/a)")
        return
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * len(labels) + 2.5), 3.8))
    ax.bar(x - width / 2, ss, width, label="SS", color="#55a868", edgecolor="white")
    ax.bar(x + width / 2, cff, width, label="CFF", color="#c44e52", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("median detour_ratio")
    ax.set_title("Detour (path / net) — CFF vs SS, within backend")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_kinematics(
    bound: Sequence[BoundEpisode],
    out_dir: Path | str,
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
    traces: Sequence[Trace | None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Write appendix tables and plots under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "table_kinematics": (
            table_kinematics(bound, group_cols=group_cols, traces=traces),
            _TABLE_NOTES,
        ),
        "table_kinematics_by_quadrant": (
            table_kinematics_by_quadrant(bound, group_cols=group_cols, traces=traces),
            _QUAD_NOTES,
        ),
        "kinematics_episodes": (table_kinematics_episodes(bound, traces=traces), ()),
    }
    written: dict[str, pd.DataFrame] = {}
    for stem, (frame, notes) in tables.items():
        render_table(frame, out, stem, footnotes=notes)
        written[stem] = frame
    plot_kinematics_path_by_quadrant(
        bound, out / "plots" / "kinematics_path_by_quadrant.png", traces=traces
    )
    plot_kinematics_detour_cff_vs_ss(
        bound,
        out / "plots" / "kinematics_detour_cff_vs_ss.png",
        group_cols=group_cols,
        traces=traces,
    )
    return written
