#!/usr/bin/env python3
"""Aggregate DyniSafety-RoboTwin summaries into CSV tables.

Scans summary files produced under::

    dynamic_RoboTwin/outputs/dynisafety/<task_config>/<task>/<model>/level*/_dynisafety_summary.txt

By default (average-only) writes:

1. ``summary_model_x_task_L{level}.csv`` — per level, rows ``(metric, model)``,
   single column ``mean`` (mean over tasks).
2. ``summary_model_x_level.csv`` — rows ``(metric, model)``, columns ``L0..Ln``
   plus ``mean`` (mean across levels).

With ``--per-task``, also writes ``summary_per_task_level.csv`` and per-level
CSVs with one column per task plus ``mean``.

When the same ``(model, task_config, task, level)`` bucket appears multiple
times, the newest summary file by mtime wins automatically.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"

RE_LEVEL_DIR = re.compile(r"^level(\d+)$")
RE_DYNAMIC_LEVEL = re.compile(r"^dynamic_level=(\d+)$")
RE_EPISODES = re.compile(r"^episodes=(\d+)$")
RE_SUCCESS = re.compile(r"^success=(\d+)/(\d+)\s+\(([\d.]+)%\)$")
RE_COLLISION_EPISODES = re.compile(r"^collision_episodes=(\d+)/(\d+)\s+\(([\d.]+)%\)$")
RE_TOPPLE_EPISODES = re.compile(r"^topple_episodes=(\d+)/(\d+)(?:\s+\(([\d.]+)%\))?$")
RE_MAX_TILT = re.compile(rf"^max_tilt_deg=({FLOAT_RE})$")
RE_TOPPLE_THRESHOLD = re.compile(rf"^topple_threshold_deg=({FLOAT_RE})$")
RE_TOTAL_COLLISION_STEPS = re.compile(r"^total_collision_steps=(\d+)$")
RE_AVG_MAX_FORCE = re.compile(rf"^avg_max_collision_force_N=({FLOAT_RE})$")
RE_AVG_MAX_IMPULSE = re.compile(rf"^avg_max_collision_impulse_Ns=({FLOAT_RE})$")
RE_PEAK_FORCE = re.compile(rf"^peak_collision_force_N=({FLOAT_RE})$")
RE_PEAK_IMPULSE = re.compile(rf"^peak_collision_impulse_Ns=({FLOAT_RE})$")
RE_EPISODE_ROW = re.compile(
    rf"^episode=(?P<episode>\d+)\s+"
    rf"seed=(?P<seed>\S+)\s+"
    rf"success=(?P<success>True|False)\s+"
    rf"collision_steps=(?P<collision_steps>\d+)\s+"
    rf"collision_events=(?P<collision_events>\d+)\s+"
    rf"first_collision_step=\S+\s+"
    rf"max_collision_force_N=(?P<max_force>{FLOAT_RE})\s+"
    rf"max_collision_impulse_Ns=(?P<max_impulse>{FLOAT_RE})\b"
)
RE_EPISODE_TOPPLE = re.compile(
    rf"\bobstacle_toppled=(?P<toppled>True|False)\s+"
    rf"max_tilt_deg=(?P<max_tilt>{FLOAT_RE})\s+"
    rf"topple_threshold_deg=(?P<threshold>{FLOAT_RE})\b"
)

TASK_METRICS: Tuple[str, ...] = (
    "SR",
    "CR",
    "ColSteps/Ep",
    "Topple",
    "MaxTilt(deg)",
    "FPeakMean(N)",
    "FPeakMax(N)",
    "AvgMaxForce(N)",
    "AvgMaxImpulse(Ns)",
    "PeakForce(N)",
    "PeakImpulse(Ns)",
)
OVERVIEW_METRICS: Tuple[str, ...] = TASK_METRICS


@dataclass
class SummaryRecord:
    """Parsed metrics from one RoboTwin DyniSafety summary file."""

    path: Path
    task_config: str
    task: str
    model: str
    dyn_level: int
    episodes: int
    successes: int
    collision_episodes: int
    topple_episodes: int
    total_collision_steps: int
    max_tilt_deg: float
    topple_threshold_deg: float
    avg_max_collision_force_n: float
    avg_max_collision_impulse_ns: float
    peak_collision_force_n: float
    peak_collision_impulse_ns: float
    mtime: float

    @property
    def sr_pct(self) -> float:
        return 100.0 * self.successes / self.episodes if self.episodes else 0.0

    @property
    def cr_pct(self) -> float:
        return (
            100.0 * self.collision_episodes / self.episodes if self.episodes else 0.0
        )

    @property
    def topple_pct(self) -> float:
        return 100.0 * self.topple_episodes / self.episodes if self.episodes else 0.0

    @property
    def collision_steps_per_episode(self) -> float:
        return self.total_collision_steps / self.episodes if self.episodes else 0.0


@dataclass
class Aggregate:
    """Arithmetic-mean aggregate across multiple task-level records."""

    n_records: int = 0
    episodes: int = 0
    successes: int = 0
    collision_episodes: int = 0
    topple_episodes: int = 0
    total_collision_steps: int = 0
    sr_pcts: List[float] = field(default_factory=list)
    cr_pcts: List[float] = field(default_factory=list)
    topple_pcts: List[float] = field(default_factory=list)
    collision_steps_per_ep: List[float] = field(default_factory=list)
    max_tilt_deg: List[float] = field(default_factory=list)
    avg_max_force_n: List[float] = field(default_factory=list)
    avg_max_impulse_ns: List[float] = field(default_factory=list)
    peak_force_n: List[float] = field(default_factory=list)
    peak_impulse_ns: List[float] = field(default_factory=list)

    def add(self, rec: SummaryRecord) -> None:
        self.n_records += 1
        self.episodes += rec.episodes
        self.successes += rec.successes
        self.collision_episodes += rec.collision_episodes
        self.topple_episodes += rec.topple_episodes
        self.total_collision_steps += rec.total_collision_steps
        self.sr_pcts.append(rec.sr_pct)
        self.cr_pcts.append(rec.cr_pct)
        self.topple_pcts.append(rec.topple_pct)
        self.collision_steps_per_ep.append(rec.collision_steps_per_episode)
        self.max_tilt_deg.append(rec.max_tilt_deg)
        self.avg_max_force_n.append(rec.avg_max_collision_force_n)
        self.avg_max_impulse_ns.append(rec.avg_max_collision_impulse_ns)
        self.peak_force_n.append(rec.peak_collision_force_n)
        self.peak_impulse_ns.append(rec.peak_collision_impulse_ns)

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def metric(self, name: str) -> Optional[float]:
        return {
            "SR": self._mean(self.sr_pcts),
            "CR": self._mean(self.cr_pcts),
            "ColSteps/Ep": self._mean(self.collision_steps_per_ep),
            "Topple": self._mean(self.topple_pcts),
            "MaxTilt(deg)": self._mean(self.max_tilt_deg),
            "FPeakMean(N)": self._mean(self.avg_max_force_n),
            "FPeakMax(N)": self._mean(self.peak_force_n),
            "AvgMaxForce(N)": self._mean(self.avg_max_force_n),
            "AvgMaxImpulse(Ns)": self._mean(self.avg_max_impulse_ns),
            "PeakForce(N)": self._mean(self.peak_force_n),
            "PeakImpulse(Ns)": self._mean(self.peak_impulse_ns),
        }[name]


def parse_summary_file(path: Path, outputs_root: Path) -> Optional[SummaryRecord]:
    """Parse one RoboTwin summary file."""
    try:
        rel_path = path.relative_to(outputs_root)
    except ValueError:
        print(f"[warn] summary file is outside outputs root, skipping: {path}")
        return None

    parts = rel_path.parts
    if len(parts) != 5:
        print(f"[warn] unexpected summary path layout, skipping: {rel_path}")
        return None

    task_config, task, model, level_dir, filename = parts
    if filename != "_dynisafety_summary.txt":
        print(f"[warn] unexpected summary filename, skipping: {rel_path}")
        return None

    level_match = RE_LEVEL_DIR.match(level_dir)
    if level_match is None:
        print(f"[warn] unexpected level directory, skipping: {rel_path}")
        return None
    level_from_path = int(level_match.group(1))

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[warn] cannot read {path}: {exc}")
        return None

    header_level: Optional[int] = None
    header_episodes: Optional[int] = None
    header_successes: Optional[int] = None
    header_collision_episodes: Optional[int] = None
    header_topple_episodes: Optional[int] = None
    header_max_tilt_deg: Optional[float] = None
    header_topple_threshold_deg: Optional[float] = None
    header_total_collision_steps: Optional[int] = None
    header_avg_max_force: Optional[float] = None
    header_avg_max_impulse: Optional[float] = None
    header_peak_force: Optional[float] = None
    header_peak_impulse: Optional[float] = None

    per_episode_count = 0
    per_episode_successes = 0
    per_episode_collision_episodes = 0
    per_episode_topple_episodes = 0
    per_episode_total_collision_steps = 0
    per_episode_max_tilts: List[float] = []
    per_episode_forces: List[float] = []
    per_episode_impulses: List[float] = []

    for line in text.splitlines():
        if header_level is None and (match := RE_DYNAMIC_LEVEL.match(line)):
            header_level = int(match.group(1))
            continue
        if header_episodes is None and (match := RE_EPISODES.match(line)):
            header_episodes = int(match.group(1))
            continue
        if header_successes is None and (match := RE_SUCCESS.match(line)):
            header_successes = int(match.group(1))
            if header_episodes is None:
                header_episodes = int(match.group(2))
            continue
        if header_collision_episodes is None and (
            match := RE_COLLISION_EPISODES.match(line)
        ):
            header_collision_episodes = int(match.group(1))
            if header_episodes is None:
                header_episodes = int(match.group(2))
            continue
        if header_topple_episodes is None and (
            match := RE_TOPPLE_EPISODES.match(line)
        ):
            header_topple_episodes = int(match.group(1))
            if header_episodes is None:
                header_episodes = int(match.group(2))
            continue
        if header_max_tilt_deg is None and (match := RE_MAX_TILT.match(line)):
            header_max_tilt_deg = float(match.group(1))
            continue
        if header_topple_threshold_deg is None and (
            match := RE_TOPPLE_THRESHOLD.match(line)
        ):
            header_topple_threshold_deg = float(match.group(1))
            continue
        if header_total_collision_steps is None and (
            match := RE_TOTAL_COLLISION_STEPS.match(line)
        ):
            header_total_collision_steps = int(match.group(1))
            continue
        if header_avg_max_force is None and (match := RE_AVG_MAX_FORCE.match(line)):
            header_avg_max_force = float(match.group(1))
            continue
        if header_avg_max_impulse is None and (
            match := RE_AVG_MAX_IMPULSE.match(line)
        ):
            header_avg_max_impulse = float(match.group(1))
            continue
        if header_peak_force is None and (match := RE_PEAK_FORCE.match(line)):
            header_peak_force = float(match.group(1))
            continue
        if header_peak_impulse is None and (match := RE_PEAK_IMPULSE.match(line)):
            header_peak_impulse = float(match.group(1))
            continue
        if match := RE_EPISODE_ROW.match(line):
            per_episode_count += 1
            success = match.group("success") == "True"
            collision_steps = int(match.group("collision_steps"))
            max_force = float(match.group("max_force"))
            max_impulse = float(match.group("max_impulse"))

            if success:
                per_episode_successes += 1
            per_episode_total_collision_steps += collision_steps
            if collision_steps > 0:
                per_episode_collision_episodes += 1
                per_episode_forces.append(max_force)
                per_episode_impulses.append(max_impulse)
            if topple_match := RE_EPISODE_TOPPLE.search(line):
                if topple_match.group("toppled") == "True":
                    per_episode_topple_episodes += 1
                per_episode_max_tilts.append(float(topple_match.group("max_tilt")))
                if header_topple_threshold_deg is None:
                    header_topple_threshold_deg = float(topple_match.group("threshold"))

    if header_level is not None and header_level != level_from_path:
        print(
            f"[warn] level mismatch in {rel_path}: path=level{level_from_path}, "
            f"header={header_level}; using header value"
        )
    dyn_level = header_level if header_level is not None else level_from_path

    if per_episode_count > 0:
        episodes = per_episode_count
        successes = per_episode_successes
        collision_episodes = per_episode_collision_episodes
        topple_episodes = per_episode_topple_episodes
        total_collision_steps = per_episode_total_collision_steps
        max_tilt_deg = max(per_episode_max_tilts) if per_episode_max_tilts else (
            header_max_tilt_deg or 0.0
        )
        avg_max_force = (
            sum(per_episode_forces) / len(per_episode_forces) if per_episode_forces else 0.0
        )
        avg_max_impulse = (
            sum(per_episode_impulses) / len(per_episode_impulses)
            if per_episode_impulses
            else 0.0
        )
        peak_force = max(per_episode_forces) if per_episode_forces else 0.0
        peak_impulse = max(per_episode_impulses) if per_episode_impulses else 0.0
    else:
        missing_header = any(
            value is None
            for value in (
                header_episodes,
                header_successes,
                header_collision_episodes,
                header_total_collision_steps,
                header_avg_max_force,
                header_avg_max_impulse,
                header_peak_force,
                header_peak_impulse,
            )
        )
        if missing_header:
            print(f"[warn] incomplete summary file, skipping: {rel_path}")
            return None
        episodes = header_episodes or 0
        successes = header_successes or 0
        collision_episodes = header_collision_episodes or 0
        topple_episodes = header_topple_episodes or 0
        total_collision_steps = header_total_collision_steps or 0
        max_tilt_deg = header_max_tilt_deg or 0.0
        avg_max_force = header_avg_max_force or 0.0
        avg_max_impulse = header_avg_max_impulse or 0.0
        peak_force = header_peak_force or 0.0
        peak_impulse = header_peak_impulse or 0.0

    return SummaryRecord(
        path=path,
        task_config=task_config,
        task=task,
        model=model,
        dyn_level=dyn_level,
        episodes=episodes,
        successes=successes,
        collision_episodes=collision_episodes,
        topple_episodes=topple_episodes,
        total_collision_steps=total_collision_steps,
        max_tilt_deg=max_tilt_deg,
        topple_threshold_deg=header_topple_threshold_deg or 30.0,
        avg_max_collision_force_n=avg_max_force,
        avg_max_collision_impulse_ns=avg_max_impulse,
        peak_collision_force_n=peak_force,
        peak_collision_impulse_ns=peak_impulse,
        mtime=path.stat().st_mtime,
    )


def discover_summary_files(outputs_root: Path) -> List[Path]:
    """Locate every RoboTwin summary file under outputs_root."""
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"outputs root not found: {outputs_root}")
    return sorted(outputs_root.rglob("_dynisafety_summary.txt"))


def dedupe_keep_latest(records: List[SummaryRecord]) -> List[SummaryRecord]:
    """Keep only the newest file per (model, task_config, task, level)."""
    latest: Dict[Tuple[str, str, str, int], SummaryRecord] = {}
    for rec in records:
        key = (rec.model, rec.task_config, rec.task, rec.dyn_level)
        prev = latest.get(key)
        if prev is None or rec.mtime > prev.mtime:
            latest[key] = rec
    return list(latest.values())


def filter_records(
    records: List[SummaryRecord],
    models: Optional[Iterable[str]],
    task_configs: Optional[Iterable[str]],
    tasks: Optional[Iterable[str]],
    levels: Optional[Iterable[int]],
) -> List[SummaryRecord]:
    """Apply optional filters."""
    model_set = set(models) if models else None
    task_config_set = set(task_configs) if task_configs else None
    task_set = set(tasks) if tasks else None
    level_set = set(levels) if levels else None

    filtered: List[SummaryRecord] = []
    for rec in records:
        if model_set is not None and rec.model not in model_set:
            continue
        if task_config_set is not None and rec.task_config not in task_config_set:
            continue
        if task_set is not None and rec.task not in task_set:
            continue
        if level_set is not None and rec.dyn_level not in level_set:
            continue
        filtered.append(rec)
    return filtered


def _round_metric(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.2f}"


def _build_task_label_map(
    records: List[SummaryRecord],
) -> Dict[Tuple[str, str], str]:
    task_config_names = {rec.task_config for rec in records}
    task_to_configs: Dict[str, set[str]] = defaultdict(set)
    for rec in records:
        task_to_configs[rec.task].add(rec.task_config)

    needs_config_prefix = len(task_config_names) > 1 or any(
        len(configs) > 1 for configs in task_to_configs.values()
    )

    labels: Dict[Tuple[str, str], str] = {}
    for rec in records:
        key = (rec.task_config, rec.task)
        if key in labels:
            continue
        labels[key] = (
            f"{rec.task_config}/{rec.task}" if needs_config_prefix else rec.task
        )
    return labels


def write_long_csv(records: List[SummaryRecord], out_path: Path) -> None:
    """Write one row per (model, task_config, task, level)."""
    rows = []
    for rec in sorted(
        records, key=lambda x: (x.model, x.task_config, x.task, x.dyn_level)
    ):
        rows.append(
            {
                "model": rec.model,
                "task_config": rec.task_config,
                "task": rec.task,
                "dynamic_level": rec.dyn_level,
                "episodes": rec.episodes,
                "successes": rec.successes,
                "collision_episodes": rec.collision_episodes,
                "topple_episodes": rec.topple_episodes,
                "total_collision_steps": rec.total_collision_steps,
                "success_rate_pct": round(rec.sr_pct, 2),
                "collision_rate_pct": round(rec.cr_pct, 2),
                "topple_rate_pct": round(rec.topple_pct, 2),
                "collision_steps_per_episode": round(
                    rec.collision_steps_per_episode, 2
                ),
                "max_tilt_deg": round(rec.max_tilt_deg, 1),
                "topple_threshold_deg": round(rec.topple_threshold_deg, 1),
                "f_peak_mean_N": round(rec.avg_max_collision_force_n, 2),
                "f_peak_max_N": round(rec.peak_collision_force_n, 2),
                "avg_max_collision_force_N": round(rec.avg_max_collision_force_n, 2),
                "avg_max_collision_impulse_Ns": round(
                    rec.avg_max_collision_impulse_ns, 4
                ),
                "peak_collision_force_N": round(rec.peak_collision_force_n, 2),
                "peak_collision_impulse_Ns": round(rec.peak_collision_impulse_ns, 4),
                "summary_file": str(rec.path),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "task_config",
        "task",
        "dynamic_level",
        "episodes",
        "successes",
        "collision_episodes",
        "topple_episodes",
        "total_collision_steps",
        "success_rate_pct",
        "collision_rate_pct",
        "topple_rate_pct",
        "collision_steps_per_episode",
        "max_tilt_deg",
        "topple_threshold_deg",
        "f_peak_mean_N",
        "f_peak_max_N",
        "avg_max_collision_force_N",
        "avg_max_collision_impulse_Ns",
        "peak_collision_force_N",
        "peak_collision_impulse_Ns",
        "summary_file",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {out_path} ({len(rows)} rows)")


def write_pivot_csvs(
    records: List[SummaryRecord], out_dir: Path, *, per_task: bool
) -> None:
    """Write per-level pivots and a model-vs-level overview."""
    labels = _build_task_label_map(records)
    by_model_task_level: Dict[Tuple[str, str, int], Aggregate] = defaultdict(Aggregate)
    by_model_level: Dict[Tuple[str, int], Aggregate] = defaultdict(Aggregate)

    for rec in records:
        task_label = labels[(rec.task_config, rec.task)]
        by_model_task_level[(rec.model, task_label, rec.dyn_level)].add(rec)
        by_model_level[(rec.model, rec.dyn_level)].add(rec)

    models = sorted({rec.model for rec in records})
    tasks = sorted({labels[(rec.task_config, rec.task)] for rec in records})
    levels = sorted({rec.dyn_level for rec in records})

    out_dir.mkdir(parents=True, exist_ok=True)

    for level in levels:
        out_path = out_dir / f"summary_model_x_task_L{level}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if per_task:
                writer.writerow(["metric", "model"] + tasks + ["mean"])
            else:
                writer.writerow(["metric", "model", "mean"])
            for metric in TASK_METRICS:
                for model in models:
                    values: List[float] = []
                    for task in tasks:
                        agg = by_model_task_level.get((model, task, level))
                        value = agg.metric(metric) if agg is not None else None
                        if value is not None:
                            values.append(value)
                    mean_val = sum(values) / len(values) if values else None
                    if per_task:
                        row: List[object] = [metric, model]
                        for task in tasks:
                            agg = by_model_task_level.get((model, task, level))
                            v = agg.metric(metric) if agg is not None else None
                            row.append(_round_metric(v))
                        row.append(_round_metric(mean_val))
                        writer.writerow(row)
                    else:
                        writer.writerow([metric, model, _round_metric(mean_val)])
        mode = "per-task + mean" if per_task else "mean over tasks only"
        print(f"[ok] wrote {out_path} ({mode}; metrics: {', '.join(TASK_METRICS)})")

    out_path = out_dir / "summary_model_x_level.csv"
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "model"] + [f"L{lv}" for lv in levels] + ["mean"])
        for metric in OVERVIEW_METRICS:
            for model in models:
                row: List[object] = [metric, model]
                values: List[float] = []
                for level in levels:
                    agg = by_model_level.get((model, level))
                    value = agg.metric(metric) if agg is not None else None
                    row.append(_round_metric(value))
                    if value is not None:
                        values.append(value)
                row.append(_round_metric(sum(values) / len(values) if values else None))
                writer.writerow(row)
    print(f"[ok] wrote {out_path} (metrics: {', '.join(OVERVIEW_METRICS)})")


def print_console_summary(records: List[SummaryRecord], *, per_task: bool) -> None:
    """Pretty-print a quick terminal summary."""
    if not records:
        print("[warn] no records parsed")
        return

    if per_task:
        labels = _build_task_label_map(records)
        model_w = max(14, max(len(rec.model) for rec in records) + 2)
        task_w = max(
            18, max(len(labels[(rec.task_config, rec.task)]) for rec in records) + 2
        )
        width = model_w + task_w + 3 + 8 + 10 + 10 + 14 + 12 + 12
        print("\n" + "=" * width)
        print(
            f"{'model':<{model_w}}"
            f"{'task':<{task_w}}"
            f"{'L':>3}"
            f"{'ep':>8}"
            f"{'SR%':>10}"
            f"{'CR%':>10}"
            f"{'ColStep/Ep':>14}"
            f"{'AvgF(N)':>12}"
            f"{'PeakF(N)':>12}"
        )
        print("-" * width)
        for rec in sorted(
            records, key=lambda x: (x.model, x.task_config, x.task, x.dyn_level)
        ):
            task_label = labels[(rec.task_config, rec.task)]
            print(
                f"{rec.model:<{model_w}}"
                f"{task_label:<{task_w}}"
                f"{rec.dyn_level:>3}"
                f"{rec.episodes:>8}"
                f"{rec.sr_pct:>10.2f}"
                f"{rec.cr_pct:>10.2f}"
                f"{rec.collision_steps_per_episode:>14.2f}"
                f"{rec.avg_max_collision_force_n:>12.2f}"
                f"{rec.peak_collision_force_n:>12.2f}"
            )
        print("=" * width)
        return

    by_model_level: Dict[Tuple[str, int], Aggregate] = defaultdict(Aggregate)
    for rec in records:
        by_model_level[(rec.model, rec.dyn_level)].add(rec)

    models = sorted({rec.model for rec in records})
    levels = sorted({rec.dyn_level for rec in records})
    model_w = max(14, max(len(m) for m in models) + 2)
    width = model_w + 3 + 8 + 10 + 10 + 14 + 12 + 12
    print("\n" + "=" * width)
    print("(values are mean over tasks)")
    print(
        f"{'model':<{model_w}}"
        f"{'L':>3}"
        f"{'ep*':>8}"
        f"{'SR%':>10}"
        f"{'CR%':>10}"
        f"{'ColStep/Ep':>14}"
        f"{'AvgF(N)':>12}"
        f"{'PeakF(N)':>12}"
    )
    print("-" * width)
    for model in models:
        for level in levels:
            agg = by_model_level.get((model, level))
            if agg is None or agg.n_records == 0:
                continue
            mean_ep = agg.episodes / agg.n_records
            print(
                f"{model:<{model_w}}"
                f"{level:>3}"
                f"{mean_ep:>8.1f}"
                f"{agg.metric('SR') or 0.0:>10.2f}"
                f"{agg.metric('CR') or 0.0:>10.2f}"
                f"{agg.metric('ColSteps/Ep') or 0.0:>14.2f}"
                f"{agg.metric('FPeakMean(N)') or 0.0:>12.2f}"
                f"{agg.metric('FPeakMax(N)') or 0.0:>12.2f}"
            )
    print("=" * width)
    print("*ep = mean episodes per task bucket")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=script_dir.parent / "outputs" / "dynisafety",
        help="Root directory containing <task_config>/<task>/<model>/level*/ summaries.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict to these model names.",
    )
    parser.add_argument(
        "--task-configs",
        nargs="*",
        default=None,
        help="Restrict to these task_config names, e.g. demo_clean.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Restrict to these task names.",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=None,
        help="Restrict to these dynamic levels, e.g. --levels 0 1 2 3.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write CSV outputs (default: <outputs-root>/_aggregated).",
    )
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="Emit per-task columns and summary_per_task_level.csv; default is avg-only.",
    )
    args = parser.parse_args()

    outputs_root = args.outputs_root.resolve()
    out_dir = (args.out_dir or (outputs_root / "_aggregated")).resolve()

    files = discover_summary_files(outputs_root)
    if not files:
        print(f"[error] no _dynisafety_summary.txt files found under {outputs_root}")
        return
    print(f"[info] found {len(files)} summary files under {outputs_root}")

    records: List[SummaryRecord] = []
    for file_path in files:
        rec = parse_summary_file(file_path, outputs_root)
        if rec is not None:
            records.append(rec)
    print(f"[info] parsed {len(records)}/{len(files)} summaries successfully")

    records = dedupe_keep_latest(records)
    print(
        "[info] "
        f"{len(records)} unique (model, task_config, task, level) buckets after dedup"
    )

    records = filter_records(
        records=records,
        models=args.models,
        task_configs=args.task_configs,
        tasks=args.tasks,
        levels=args.levels,
    )
    if not records:
        print("[warn] no records remain after applying filters")
        return

    print_console_summary(records, per_task=args.per_task)
    if args.per_task:
        write_long_csv(records, out_dir / "summary_per_task_level.csv")
    else:
        print("[info] avg-only mode: skipping summary_per_task_level.csv (use --per-task for long CSV)")
    write_pivot_csvs(records, out_dir, per_task=args.per_task)


if __name__ == "__main__":
    main()
