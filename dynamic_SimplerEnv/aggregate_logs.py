#!/usr/bin/env python3
"""Aggregate DyniSafety-SimplerEnv evaluation logs into CSV summaries.

This script is the SimplerEnv version of the old LIBERO aggregator.

It scans ``dynamic_SimplerEnv/outputs/<model>/logs/EVAL-dynisafety-*.txt``
and parses the bench-layout logs produced by ``main_inference.py``. In the
SimplerEnv setup, each log file corresponds to exactly one
``(model, bench_task, dynamic_level)`` run. If the same bucket is evaluated
multiple times, the newest log file by modification time wins automatically.

When logs include the summary line
``Avg max collision force (N), excl. force-anomaly episodes: … (k/n)``,
that value is used for ``f_peak_mean_N`` / pivot metric ``FPeakMean(N)``
(instead of the non-excluded line or per-episode fallback).

Outputs:
1. ``summary_per_task_level.csv``
   Long-format table with one row per ``(model, task, level)``.
2. ``summary_model_x_task_L{level}.csv``
   Wide tables for each dynamic level. Rows are ``(metric, model)`` and
   columns are the tasks plus ``mean``.
3. ``summary_model_x_level.csv``
   Wide overview table. Rows are ``(metric, model)`` and columns are
   ``L0..L3`` plus ``mean``.
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

RE_POLICY_MODEL = re.compile(r"^\s*policy_model=(.+)$")
RE_BENCH_TASK = re.compile(r"^\s*bench_task=(.+)$")
RE_HEADER_LEVEL = re.compile(r"^\s*dynamic_level=(\d+)$")
RE_SUMMARY_LEVEL = re.compile(r"^\s*Dynamic level\s*:\s*(\d+)$")
RE_TOTAL_EPISODES = re.compile(r"^\s*Total episodes\s*:\s*(\d+)$")
RE_SUCCESS = re.compile(r"^\s*Success rate\s*:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)$")
RE_COLLISION_EPISODES = re.compile(
    r"^\s*Collision episodes\s*:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)$"
)
RE_TOPPLE_EPISODES = re.compile(
    r"^\s*Topple episodes\s*:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)$"
)
RE_MAX_TILT = re.compile(rf"^\s*Max tilt deg\s*:\s*({FLOAT_RE})$")
RE_TOPPLE_THRESHOLD = re.compile(rf"^\s*Topple threshold deg\s*:\s*({FLOAT_RE})$")
RE_AVG_MAX_FORCE = re.compile(rf"^\s*Avg max collision force \(N\)\s*:\s*({FLOAT_RE})$")
# Prefer this line when present (matches bench summary after anomaly filtering).
RE_AVG_MAX_FORCE_EXCL_ANOMALY = re.compile(
    rf"^\s*Avg max collision force \(N\),\s*excl\.\s*force-anomaly episodes:\s*"
    rf"({FLOAT_RE})\s*\((\d+)/(\d+)\)\s*$"
)
RE_PEAK_FORCE = re.compile(rf"^\s*Peak collision force \(N\)\s*:\s*({FLOAT_RE})$")
RE_AVG_MAX_IMPULSE = re.compile(rf"^\s*Avg max collision impulse \(Ns\)\s*:\s*({FLOAT_RE})$")
RE_PEAK_IMPULSE = re.compile(rf"^\s*Peak collision impulse \(Ns\)\s*:\s*({FLOAT_RE})$")
RE_TOTAL_COLLISION_STEPS = re.compile(r"^\s*Total collision steps\s*:\s*(\d+)$")
RE_FORCE_ANOMALY_EPISODES = re.compile(r"^\s*Force anomaly episodes\s*:\s*(\d+)/(\d+)$")
RE_FORCE_ANOMALY_STEPS = re.compile(r"^\s*Force anomaly steps\s*:\s*(\d+)$")
RE_PRE_PHYSICS_COLLISION_PREVENTS = re.compile(
    r"^\s*Pre-physics collision prevents\s*:\s*(\d+)$"
)
RE_EPISODE_LINE = re.compile(
    r"^episode=(\d+)\s+success=(True|False)\s+collision_steps=(\d+)\b"
)
RE_EPISODE_EXTRA = re.compile(
    rf"\bobstacle_toppled=(?P<toppled>True|False)\s+"
    rf"max_tilt_deg=(?P<max_tilt>{FLOAT_RE})\s+"
    rf"topple_threshold_deg=(?P<threshold>{FLOAT_RE})\s+"
    rf"max_collision_force_N=(?P<force>{FLOAT_RE})\s+"
    rf"max_collision_impulse_Ns=(?P<impulse>{FLOAT_RE})\b"
)
RE_EPISODE_SANITY = re.compile(
    r"\bforce_anomaly=(?P<anomaly>True|False)\s+"
    r"force_anomaly_steps=(?P<steps>\d+)\s+"
    r"pre_physics_collision_prevents=(?P<prevents>\d+)\b"
)

TASK_METRICS: Tuple[str, ...] = (
    "SR",
    "CR",
    "ColSteps/Ep",
    "Topple",
    "MaxTilt(deg)",
    "FPeakMean(N)",
    "FPeakMax(N)",
)
OVERVIEW_METRICS: Tuple[str, ...] = TASK_METRICS


@dataclass
class LogRecord:
    """Parsed metrics from a single SimplerEnv bench log."""

    path: Path
    model: str
    bench_task: str
    dyn_level: int
    episodes: int
    successes: int
    collision_episodes: int
    topple_episodes: int
    total_collision_steps: int
    max_tilt_deg: float
    topple_threshold_deg: float
    avg_max_collision_force_n: float
    peak_collision_force_n: float
    avg_max_collision_impulse_ns: float
    peak_collision_impulse_ns: float
    force_anomaly_episodes: int
    force_anomaly_steps: int
    pre_physics_collision_prevents: int
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

    @property
    def collision_steps_per_collision_episode(self) -> Optional[float]:
        if not self.collision_episodes:
            return None
        return self.total_collision_steps / self.collision_episodes


@dataclass
class Aggregate:
    """Arithmetic-mean aggregate across multiple LogRecord rows."""

    n_records: int = 0
    episodes: int = 0
    successes: int = 0
    collision_episodes: int = 0
    topple_episodes: int = 0
    total_collision_steps: int = 0
    sr_pcts: List[float] = field(default_factory=list)
    cr_pcts: List[float] = field(default_factory=list)
    topple_pcts: List[float] = field(default_factory=list)
    col_steps_per_ep: List[float] = field(default_factory=list)
    max_tilts: List[float] = field(default_factory=list)
    avg_max_forces: List[float] = field(default_factory=list)
    peak_forces: List[float] = field(default_factory=list)

    def add(self, rec: LogRecord) -> None:
        self.n_records += 1
        self.episodes += rec.episodes
        self.successes += rec.successes
        self.collision_episodes += rec.collision_episodes
        self.topple_episodes += rec.topple_episodes
        self.total_collision_steps += rec.total_collision_steps
        self.sr_pcts.append(rec.sr_pct)
        self.cr_pcts.append(rec.cr_pct)
        self.topple_pcts.append(rec.topple_pct)
        self.col_steps_per_ep.append(rec.collision_steps_per_episode)
        self.max_tilts.append(rec.max_tilt_deg)
        self.avg_max_forces.append(rec.avg_max_collision_force_n)
        self.peak_forces.append(rec.peak_collision_force_n)

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def metric(self, name: str) -> Optional[float]:
        return {
            "SR": self._mean(self.sr_pcts),
            "CR": self._mean(self.cr_pcts),
            "ColSteps/Ep": self._mean(self.col_steps_per_ep),
            "Topple": self._mean(self.topple_pcts),
            "MaxTilt(deg)": self._mean(self.max_tilts),
            "FPeakMean(N)": self._mean(self.avg_max_forces),
            "FPeakMax(N)": self._mean(self.peak_forces),
        }[name]


def parse_log_file(path: Path) -> Optional[LogRecord]:
    """Parse one SimplerEnv bench log file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[warn] cannot read {path}: {exc}")
        return None

    model: Optional[str] = None
    bench_task: Optional[str] = None
    dyn_level: Optional[int] = None
    episodes: Optional[int] = None
    successes: Optional[int] = None
    collision_episodes: Optional[int] = None
    topple_episodes: Optional[int] = None
    total_collision_steps: Optional[int] = None
    max_tilt_deg: Optional[float] = None
    topple_threshold_deg: Optional[float] = None
    avg_max_force: Optional[float] = None
    avg_max_force_excl_anomaly: Optional[float] = None
    peak_force: Optional[float] = None
    avg_max_impulse: Optional[float] = None
    peak_impulse: Optional[float] = None
    force_anomaly_episodes: Optional[int] = None
    force_anomaly_steps: Optional[int] = None
    pre_physics_collision_prevents: Optional[int] = None

    fallback_episodes = 0
    fallback_successes = 0
    fallback_collision_episodes = 0
    fallback_topple_episodes = 0
    fallback_total_collision_steps = 0
    fallback_force_anomaly_episodes = 0
    fallback_force_anomaly_steps = 0
    fallback_pre_physics_collision_prevents = 0
    fallback_max_tilts: List[float] = []
    fallback_forces: List[float] = []
    fallback_impulses: List[float] = []
    has_episode_rows = False
    has_summary = False

    for line in text.splitlines():
        if model is None and (match := RE_POLICY_MODEL.match(line)):
            model = match.group(1).strip()
            continue
        if bench_task is None and (match := RE_BENCH_TASK.match(line)):
            bench_task = match.group(1).strip()
            continue
        if dyn_level is None and (match := RE_HEADER_LEVEL.match(line)):
            dyn_level = int(match.group(1))
            continue
        if dyn_level is None and (match := RE_SUMMARY_LEVEL.match(line)):
            dyn_level = int(match.group(1))
            has_summary = True
            continue
        if match := RE_TOTAL_EPISODES.match(line):
            episodes = int(match.group(1))
            has_summary = True
            continue
        if match := RE_SUCCESS.match(line):
            successes = int(match.group(1))
            episodes = episodes or int(match.group(2))
            has_summary = True
            continue
        if match := RE_COLLISION_EPISODES.match(line):
            collision_episodes = int(match.group(1))
            episodes = episodes or int(match.group(2))
            has_summary = True
            continue
        if match := RE_TOPPLE_EPISODES.match(line):
            topple_episodes = int(match.group(1))
            episodes = episodes or int(match.group(2))
            has_summary = True
            continue
        if match := RE_MAX_TILT.match(line):
            max_tilt_deg = float(match.group(1))
            has_summary = True
            continue
        if match := RE_TOPPLE_THRESHOLD.match(line):
            topple_threshold_deg = float(match.group(1))
            has_summary = True
            continue
        if match := RE_AVG_MAX_FORCE.match(line):
            avg_max_force = float(match.group(1))
            has_summary = True
            continue
        if match := RE_AVG_MAX_FORCE_EXCL_ANOMALY.match(line):
            avg_max_force_excl_anomaly = float(match.group(1))
            has_summary = True
            continue
        if match := RE_PEAK_FORCE.match(line):
            peak_force = float(match.group(1))
            has_summary = True
            continue
        if match := RE_AVG_MAX_IMPULSE.match(line):
            avg_max_impulse = float(match.group(1))
            has_summary = True
            continue
        if match := RE_PEAK_IMPULSE.match(line):
            peak_impulse = float(match.group(1))
            has_summary = True
            continue
        if match := RE_TOTAL_COLLISION_STEPS.match(line):
            total_collision_steps = int(match.group(1))
            has_summary = True
            continue
        if match := RE_FORCE_ANOMALY_EPISODES.match(line):
            force_anomaly_episodes = int(match.group(1))
            has_summary = True
            continue
        if match := RE_FORCE_ANOMALY_STEPS.match(line):
            force_anomaly_steps = int(match.group(1))
            has_summary = True
            continue
        if match := RE_PRE_PHYSICS_COLLISION_PREVENTS.match(line):
            pre_physics_collision_prevents = int(match.group(1))
            has_summary = True
            continue
        if match := RE_EPISODE_LINE.match(line):
            has_episode_rows = True
            fallback_episodes += 1
            success = match.group(2) == "True"
            col_steps = int(match.group(3))
            if success:
                fallback_successes += 1
            if col_steps > 0:
                fallback_collision_episodes += 1
            fallback_total_collision_steps += col_steps
            if extra := RE_EPISODE_EXTRA.search(line):
                if extra.group("toppled") == "True":
                    fallback_topple_episodes += 1
                fallback_max_tilts.append(float(extra.group("max_tilt")))
                fallback_forces.append(float(extra.group("force")))
                fallback_impulses.append(float(extra.group("impulse")))
                if topple_threshold_deg is None:
                    topple_threshold_deg = float(extra.group("threshold"))
            if sanity := RE_EPISODE_SANITY.search(line):
                if sanity.group("anomaly") == "True":
                    fallback_force_anomaly_episodes += 1
                fallback_force_anomaly_steps += int(sanity.group("steps"))
                fallback_pre_physics_collision_prevents += int(sanity.group("prevents"))

    if model is None:
        model = path.parent.parent.name
    if bench_task is None or dyn_level is None:
        print(f"[warn] missing bench_task/dynamic_level, skipping: {path.name}")
        return None

    if episodes is None and has_episode_rows:
        episodes = fallback_episodes
    if successes is None and has_episode_rows:
        successes = fallback_successes
    if collision_episodes is None and has_episode_rows:
        collision_episodes = fallback_collision_episodes
    if topple_episodes is None and has_episode_rows:
        topple_episodes = fallback_topple_episodes
    if total_collision_steps is None and has_episode_rows:
        total_collision_steps = fallback_total_collision_steps
    if max_tilt_deg is None and fallback_max_tilts:
        max_tilt_deg = max(fallback_max_tilts)
    if avg_max_force_excl_anomaly is not None:
        avg_max_force = avg_max_force_excl_anomaly
    elif avg_max_force is None and fallback_forces:
        avg_max_force = sum(fallback_forces) / len(fallback_forces)
    if peak_force is None and fallback_forces:
        peak_force = max(fallback_forces)
    if avg_max_impulse is None and fallback_impulses:
        avg_max_impulse = sum(fallback_impulses) / len(fallback_impulses)
    if peak_impulse is None and fallback_impulses:
        peak_impulse = max(fallback_impulses)
    if force_anomaly_episodes is None:
        force_anomaly_episodes = fallback_force_anomaly_episodes
    if force_anomaly_steps is None:
        force_anomaly_steps = fallback_force_anomaly_steps
    if pre_physics_collision_prevents is None:
        pre_physics_collision_prevents = fallback_pre_physics_collision_prevents

    if not has_summary and not has_episode_rows:
        print(f"[warn] no usable content, skipping: {path.name}")
        return None

    if episodes is None or successes is None or collision_episodes is None:
        print(f"[warn] incomplete metrics, skipping: {path.name}")
        return None

    return LogRecord(
        path=path,
        model=model,
        bench_task=bench_task,
        dyn_level=dyn_level,
        episodes=episodes,
        successes=successes,
        collision_episodes=collision_episodes,
        topple_episodes=topple_episodes or 0,
        total_collision_steps=total_collision_steps or 0,
        max_tilt_deg=max_tilt_deg or 0.0,
        topple_threshold_deg=topple_threshold_deg or 30.0,
        avg_max_collision_force_n=avg_max_force or 0.0,
        peak_collision_force_n=peak_force or 0.0,
        avg_max_collision_impulse_ns=avg_max_impulse or 0.0,
        peak_collision_impulse_ns=peak_impulse or 0.0,
        force_anomaly_episodes=force_anomaly_episodes or 0,
        force_anomaly_steps=force_anomaly_steps or 0,
        pre_physics_collision_prevents=pre_physics_collision_prevents or 0,
        mtime=path.stat().st_mtime,
    )


def discover_log_files(outputs_root: Path, models: Optional[Iterable[str]]) -> List[Path]:
    """Locate every EVAL-dynisafety log under outputs_root/<model>/logs."""
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"outputs root not found: {outputs_root}")

    if models:
        model_dirs = [outputs_root / model for model in models]
    else:
        model_dirs = [
            p for p in outputs_root.iterdir() if p.is_dir() and not p.name.startswith("_")
        ]

    files: List[Path] = []
    for model_dir in model_dirs:
        log_dir = model_dir / "logs"
        if not log_dir.is_dir():
            print(f"[info] no logs/ under {model_dir}, skipping")
            continue
        files.extend(sorted(log_dir.glob("EVAL-dynisafety-*.txt")))
    return files


def dedupe_keep_latest(records: List[LogRecord]) -> List[LogRecord]:
    """Keep only the newest file per (model, task, level)."""
    latest: Dict[Tuple[str, str, int], LogRecord] = {}
    for rec in records:
        key = (rec.model, rec.bench_task, rec.dyn_level)
        prev = latest.get(key)
        if prev is None or rec.mtime > prev.mtime:
            latest[key] = rec
    return list(latest.values())


def filter_records(
    records: List[LogRecord],
    tasks: Optional[Iterable[str]],
    levels: Optional[Iterable[int]],
) -> List[LogRecord]:
    """Apply optional task/level filters."""
    task_set = set(tasks) if tasks else None
    level_set = set(levels) if levels else None
    filtered: List[LogRecord] = []
    for rec in records:
        if task_set is not None and rec.bench_task not in task_set:
            continue
        if level_set is not None and rec.dyn_level not in level_set:
            continue
        filtered.append(rec)
    return filtered


def _round_metric(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.2f}"


def write_long_csv(records: List[LogRecord], out_path: Path) -> None:
    """Write one row per (model, task, level)."""
    rows = []
    for rec in sorted(records, key=lambda x: (x.model, x.bench_task, x.dyn_level)):
        rows.append(
            {
                "model": rec.model,
                "task": rec.bench_task,
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
                "avg_max_collision_impulse_Ns": round(rec.avg_max_collision_impulse_ns, 4),
                "peak_collision_impulse_Ns": round(rec.peak_collision_impulse_ns, 4),
                "force_anomaly_episodes": rec.force_anomaly_episodes,
                "force_anomaly_steps": rec.force_anomaly_steps,
                "pre_physics_collision_prevents": rec.pre_physics_collision_prevents,
                "collision_steps_per_collision_episode": (
                    ""
                    if rec.collision_steps_per_collision_episode is None
                    else round(rec.collision_steps_per_collision_episode, 2)
                ),
                "log_file": str(rec.path),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
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
        "avg_max_collision_impulse_Ns",
        "peak_collision_impulse_Ns",
        "force_anomaly_episodes",
        "force_anomaly_steps",
        "pre_physics_collision_prevents",
        "collision_steps_per_collision_episode",
        "log_file",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {out_path} ({len(rows)} rows)")


def write_pivot_csvs(records: List[LogRecord], out_dir: Path) -> None:
    """Write per-level task pivots and a model-vs-level overview."""
    by_model_task_level: Dict[Tuple[str, str, int], Aggregate] = defaultdict(Aggregate)
    by_model_level: Dict[Tuple[str, int], Aggregate] = defaultdict(Aggregate)

    for rec in records:
        by_model_task_level[(rec.model, rec.bench_task, rec.dyn_level)].add(rec)
        by_model_level[(rec.model, rec.dyn_level)].add(rec)

    models = sorted({rec.model for rec in records})
    tasks = sorted({rec.bench_task for rec in records})
    levels = sorted({rec.dyn_level for rec in records})

    out_dir.mkdir(parents=True, exist_ok=True)

    for level in levels:
        out_path = out_dir / f"summary_model_x_task_L{level}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["metric", "model"] + tasks + ["mean"])
            for metric in TASK_METRICS:
                for model in models:
                    row: List[object] = [metric, model]
                    values: List[float] = []
                    for task in tasks:
                        agg = by_model_task_level.get((model, task, level))
                        value = agg.metric(metric) if agg is not None else None
                        row.append(_round_metric(value))
                        if value is not None:
                            values.append(value)
                    row.append(_round_metric(sum(values) / len(values) if values else None))
                    writer.writerow(row)
        print(f"[ok] wrote {out_path} (metrics: {', '.join(TASK_METRICS)})")

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


def print_console_summary(records: List[LogRecord]) -> None:
    """Pretty-print a quick terminal summary."""
    if not records:
        print("[warn] no records parsed")
        return

    model_w = max(14, max(len(rec.model) for rec in records) + 2)
    task_w = max(16, max(len(rec.bench_task) for rec in records) + 2)
    width = model_w + task_w + 3 + 8 + 10 + 10 + 15
    print("\n" + "=" * width)
    print(
        f"{'model':<{model_w}}"
        f"{'task':<{task_w}}"
        f"{'L':>3}"
        f"{'ep':>8}"
        f"{'SR%':>10}"
        f"{'CR%':>10}"
        f"{'ColStep/Ep':>15}"
    )
    print("-" * width)
    for rec in sorted(records, key=lambda x: (x.model, x.bench_task, x.dyn_level)):
        print(
            f"{rec.model:<{model_w}}"
            f"{rec.bench_task:<{task_w}}"
            f"{rec.dyn_level:>3}"
            f"{rec.episodes:>8}"
            f"{rec.sr_pct:>10.2f}"
            f"{rec.cr_pct:>10.2f}"
            f"{rec.collision_steps_per_episode:>15.2f}"
        )
    print("=" * width)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=script_dir / "outputs",
        help="Root directory containing <model>/logs subfolders.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict to these model dirs (default: scan all model subdirs).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task filter, e.g. bridge_carrot bridge_stack bridge_spoon.",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=None,
        help="Optional level filter, e.g. --levels 0 1 2 3.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write CSV outputs (default: <outputs-root>/_aggregated).",
    )
    args = parser.parse_args()

    outputs_root = args.outputs_root.resolve()
    out_dir = (args.out_dir or (outputs_root / "_aggregated")).resolve()

    files = discover_log_files(outputs_root, args.models)
    if not files:
        print(f"[error] no EVAL-dynisafety-*.txt files found under {outputs_root}")
        return
    print(f"[info] found {len(files)} log files under {outputs_root}")

    records: List[LogRecord] = []
    for file_path in files:
        rec = parse_log_file(file_path)
        if rec is not None:
            records.append(rec)
    print(f"[info] parsed {len(records)}/{len(files)} logs successfully")

    records = dedupe_keep_latest(records)
    print(f"[info] {len(records)} unique (model, task, level) buckets after dedup")

    records = filter_records(records, args.tasks, args.levels)
    if not records:
        print("[warn] no records remain after applying filters")
        return

    print_console_summary(records)
    write_long_csv(records, out_dir / "summary_per_task_level.csv")
    write_pivot_csvs(records, out_dir)


if __name__ == "__main__":
    main()
