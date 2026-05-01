#!/usr/bin/env python3
"""Aggregate DyniSafety-LIBERO evaluation logs into CSV summaries.

Scans ``dynisafety_libero/outputs/<model>/logs/*.txt`` produced by
``run_dynisafety_eval.py`` and emits two CSV views:

1. ``summary_per_suite_level.csv``  — long format, one row per
   (model, suite, dynamic_level) aggregated over all task_id files
   inside that bucket. Aggregation is **episode-weighted**.
2. ``summary_model_x_level_<metric>.csv`` — wide format, rows = model,
   columns = dynamic level, values aggregated across all suites for the
   chosen core metric (SR / CR / SASR / Topple / SafetyScore).

The script is robust to repeated runs of the same (model, suite, level,
task_id): the **latest log file by mtime** wins so reruns automatically
override stale data.

Usage examples
--------------
# Scan every model under dynisafety_libero/outputs/
python dynisafety_libero/scripts/aggregate_logs.py

# Restrict to a specific model directory
python dynisafety_libero/scripts/aggregate_logs.py --models pi0

# Multiple specific models, custom outputs root
python dynisafety_libero/scripts/aggregate_logs.py \
    --models pi0 xvla \
    --outputs-root dynisafety_libero/outputs \
    --out-dir dynisafety_libero/outputs/_aggregated
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Regex patterns for the summary block at the bottom of every log file
# ---------------------------------------------------------------------------
RE_TASK_SUITE = re.compile(r"^Task suite:\s*(\S+)")
RE_SAFETY_LEVEL = re.compile(r"^Safety level:\s*(\S+)")
RE_DYN_LEVEL = re.compile(r"^Dynamic level:\s*(\d+)")
RE_TASK_ID = re.compile(r"task_id\s*=\s*(\d+)")
RE_TASK_NAME = re.compile(r"^Task:\s*(.+)$")

RE_TOTAL_EPS = re.compile(r"^Total episodes\s*:\s*(\d+)")
RE_TOTAL_SUCC = re.compile(r"^Total successes\s*:\s*(\d+)")
RE_SR = re.compile(r"^Success Rate \(SR\)\s*:\s*([\d.]+)%")
# Lines like "Collision Rate (CR)  : 98.0%  (49/50)"
RE_CR = re.compile(r"^Collision Rate \(CR\)\s*:\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
RE_ALL_COLL = re.compile(r"^All-obstacle coll\.\s*:\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
RE_TOPPLE = re.compile(r"^Topple rate.*?:\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
RE_SASR = re.compile(r"^SASR\s*:\s*([\d.]+)%")
RE_SAFETY_SCORE = re.compile(r"^Safety Score \(1-CR\)\s*:\s*([\d.]+)")
# "Peak force (N)       : mean=20.3 ± 44.5, max=162.8, median=1.4"  (only present at L3)
RE_PEAK_FORCE = re.compile(r"^Peak force \(N\)\s*:\s*mean=([\d.]+)")

# Filename like EVAL-dynisafety-<model>-<suite>-<date>-<time>.txt
RE_FILENAME = re.compile(
    r"^EVAL-dynisafety-(?P<model>[^-]+)-(?P<suite>[^-]+(?:_[^-]+)?)-"
    r"(?P<date>\d{4}_\d{2}_\d{2})-(?P<time>\d{2}_\d{2}_\d{2})\.txt$"
)

# Per-level metric whitelists (matches the columns we actually want in tables).
LEVEL_METRICS: Dict[int, Tuple[str, ...]] = {
    0: ("SR", "CR", "Topple"),
    1: ("SR", "CR", "Topple"),
    2: ("SR", "CR", "Topple"),
    3: ("SR", "CR", "PeakForce"),
}
# Metrics shown in the level-overview file (rows = (metric, model), cols = L0..L3).
OVERVIEW_METRICS: Tuple[str, ...] = ("SR", "CR", "Topple", "PeakForce")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class FileRecord:
    """Parsed metadata + summary numbers from a single log file."""

    path: Path
    model: str
    suite: str
    safety_level: str
    dyn_level: int
    task_id: Optional[int]
    task_name: str
    episodes: int
    successes: int
    collisions: int
    all_obs_collisions: int
    topples: int
    sasr_pct: float  # SASR percentage as printed; we re-derive count = pct/100 * episodes
    safety_score: float  # 1 - CR (already in 0..1 form)
    peak_force_mean: Optional[float]  # mean of Peak force (N); None when log doesn't report it
    mtime: float


@dataclass
class Aggregate:
    """Per-task arithmetic-mean aggregate.

    Each parsed log file contributes ONE data point per metric (the
    per-task rate from its summary block). Aggregate metrics are then
    the simple arithmetic mean over those data points (i.e. ``sum / n_tasks``).

    For the standard 4-task × 50-episode layout this is mathematically
    identical to episode-weighted aggregation. The behaviours diverge
    only when task counts are uneven (e.g. only 2/4 tasks finished),
    in which case the arithmetic mean correctly normalises by the
    number of tasks that actually contributed (``sum / 2``) instead of
    by the heterogeneous episode total.

    We still keep raw episode/success/collision counts so downstream
    code can audit or recompute episode-weighted variants if needed.
    """

    n_files: int = 0
    episodes: int = 0
    successes: int = 0
    collisions: int = 0
    all_obs_collisions: int = 0
    topples: int = 0
    task_ids: List[int] = field(default_factory=list)

    # Per-task data points (one entry per FileRecord) for arithmetic mean.
    sr_pcts: List[float] = field(default_factory=list)
    cr_pcts: List[float] = field(default_factory=list)
    all_obs_cr_pcts: List[float] = field(default_factory=list)
    topple_pcts: List[float] = field(default_factory=list)
    sasr_pcts: List[float] = field(default_factory=list)
    safety_scores: List[float] = field(default_factory=list)
    peak_forces: List[float] = field(default_factory=list)  # only L3 tasks contribute

    def add(self, rec: FileRecord) -> None:
        self.n_files += 1
        self.episodes += rec.episodes
        self.successes += rec.successes
        self.collisions += rec.collisions
        self.all_obs_collisions += rec.all_obs_collisions
        self.topples += rec.topples
        if rec.task_id is not None:
            self.task_ids.append(rec.task_id)

        # Per-task rates derived from the file's own counts (not from the
        # printed string) so we don't lose precision to the 1-decimal-place
        # rounding used in the log summary.
        eps = rec.episodes
        if eps > 0:
            self.sr_pcts.append(100.0 * rec.successes / eps)
            self.cr_pcts.append(100.0 * rec.collisions / eps)
            self.all_obs_cr_pcts.append(100.0 * rec.all_obs_collisions / eps)
            self.topple_pcts.append(100.0 * rec.topples / eps)
            self.safety_scores.append(1.0 - rec.collisions / eps)
        # SASR is only available as a printed percentage (no raw count),
        # so we use the printed value directly as the per-task data point.
        self.sasr_pcts.append(rec.sasr_pct)

        if rec.peak_force_mean is not None:
            self.peak_forces.append(rec.peak_force_mean)

    @staticmethod
    def _mean(xs: List[float]) -> Optional[float]:
        return (sum(xs) / len(xs)) if xs else None

    # --- Derived metrics (per-task arithmetic mean) ---------------------
    @property
    def sr_pct(self) -> float:
        return self._mean(self.sr_pcts) or 0.0

    @property
    def cr_pct(self) -> float:
        return self._mean(self.cr_pcts) or 0.0

    @property
    def all_obs_cr_pct(self) -> float:
        return self._mean(self.all_obs_cr_pcts) or 0.0

    @property
    def topple_pct(self) -> float:
        return self._mean(self.topple_pcts) or 0.0

    @property
    def sasr_pct(self) -> float:
        return self._mean(self.sasr_pcts) or 0.0

    @property
    def safety_score(self) -> float:
        v = self._mean(self.safety_scores)
        return 1.0 if v is None else v

    @property
    def peak_force_mean(self) -> Optional[float]:
        return self._mean(self.peak_forces)

    def metric(self, name: str) -> Optional[float]:
        if name == "PeakForce":
            return self.peak_force_mean
        return {
            "SR": self.sr_pct,
            "CR": self.cr_pct,
            "SASR": self.sasr_pct,
            "Topple": self.topple_pct,
            "SafetyScore": self.safety_score,
        }[name]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_log_file(path: Path) -> Optional[FileRecord]:
    """Parse a single evaluation log and return a FileRecord, or None on failure."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[warn] cannot read {path}: {exc}")
        return None

    suite = safety_level = None
    dyn_level: Optional[int] = None
    task_id: Optional[int] = None
    task_name = ""
    episodes = successes = collisions = all_obs_coll = topples = 0
    sasr_pct = 0.0
    safety_score = 1.0
    peak_force_mean: Optional[float] = None
    has_summary = False

    for line in text.splitlines():
        if suite is None and (m := RE_TASK_SUITE.match(line)):
            suite = m.group(1)
            continue
        if safety_level is None and (m := RE_SAFETY_LEVEL.match(line)):
            safety_level = m.group(1)
            continue
        if dyn_level is None and (m := RE_DYN_LEVEL.match(line)):
            dyn_level = int(m.group(1))
            continue
        if task_id is None and (m := RE_TASK_ID.search(line)):
            task_id = int(m.group(1))
            continue
        if not task_name and (m := RE_TASK_NAME.match(line)):
            task_name = m.group(1).strip()
            continue
        if (m := RE_TOTAL_EPS.match(line)):
            episodes = int(m.group(1)); has_summary = True; continue
        if (m := RE_TOTAL_SUCC.match(line)):
            successes = int(m.group(1)); continue
        if (m := RE_CR.match(line)):
            collisions = int(m.group(2)); episodes = max(episodes, int(m.group(3))); continue
        if (m := RE_ALL_COLL.match(line)):
            all_obs_coll = int(m.group(2)); continue
        if (m := RE_TOPPLE.match(line)):
            topples = int(m.group(2)); continue
        if (m := RE_SASR.match(line)):
            sasr_pct = float(m.group(1)); continue
        if (m := RE_SAFETY_SCORE.match(line)):
            safety_score = float(m.group(1)); continue
        if (m := RE_PEAK_FORCE.match(line)):
            peak_force_mean = float(m.group(1)); continue

    if not has_summary:
        # Likely an incomplete / crashed run — skip but warn
        print(f"[warn] no summary block, skipping: {path.name}")
        return None
    if suite is None or dyn_level is None:
        print(f"[warn] missing suite/dyn_level, skipping: {path.name}")
        return None

    # Model name — prefer the model directory name (e.g. openvla_oft_rlinf),
    # since the in-filename slug may be a shortened alias (e.g. openvla_oft).
    model = path.parent.parent.name

    return FileRecord(
        path=path,
        model=model,
        suite=suite,
        safety_level=safety_level or "?",
        dyn_level=dyn_level,
        task_id=task_id,
        task_name=task_name,
        episodes=episodes,
        successes=successes,
        collisions=collisions,
        all_obs_collisions=all_obs_coll,
        topples=topples,
        sasr_pct=sasr_pct,
        safety_score=safety_score,
        peak_force_mean=peak_force_mean,
        mtime=path.stat().st_mtime,
    )


def discover_log_files(outputs_root: Path, models: Optional[Iterable[str]]) -> List[Path]:
    """Locate every EVAL-*.txt under outputs_root/<model>/logs."""
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"outputs root not found: {outputs_root}")

    if models:
        model_dirs = [outputs_root / m for m in models]
    else:
        model_dirs = [p for p in outputs_root.iterdir() if p.is_dir() and not p.name.startswith("_")]

    files: List[Path] = []
    for mdir in model_dirs:
        log_dir = mdir / "logs"
        if not log_dir.is_dir():
            print(f"[info] no logs/ under {mdir}, skipping")
            continue
        files.extend(sorted(log_dir.glob("EVAL-*.txt")))
    return files


def dedupe_keep_latest(records: List[FileRecord]) -> List[FileRecord]:
    """Keep only the newest file per (model, suite, dyn_level, task_id)."""
    by_key: Dict[Tuple[str, str, int, Optional[int]], FileRecord] = {}
    for r in records:
        key = (r.model, r.suite, r.dyn_level, r.task_id)
        prev = by_key.get(key)
        if prev is None or r.mtime > prev.mtime:
            by_key[key] = r
    return list(by_key.values())


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _round_metric(metric: str, v: float) -> float:
    """Per-metric rounding rule (PeakForce in N → 2 decimals; others → 2)."""
    if metric == "SafetyScore":
        return round(v, 4)
    return round(v, 2)


def write_long_csv(records: List[FileRecord], out_path: Path) -> None:
    """One row per (model, suite, dyn_level), episode-weighted across task_ids.
    Carries every parsed field so downstream pivots can be recomputed cheaply."""
    grouped: Dict[Tuple[str, str, int], Aggregate] = defaultdict(Aggregate)
    safety_levels: Dict[Tuple[str, str, int], str] = {}
    for r in records:
        key = (r.model, r.suite, r.dyn_level)
        grouped[key].add(r)
        safety_levels.setdefault(key, r.safety_level)

    rows = []
    for (model, suite, dyn_level), agg in sorted(grouped.items()):
        pf = agg.peak_force_mean
        rows.append({
            "model": model,
            "suite": suite,
            "safety_level": safety_levels[(model, suite, dyn_level)],
            "dynamic_level": dyn_level,
            "n_tasks": agg.n_files,
            "task_ids": ",".join(str(t) for t in sorted(set(agg.task_ids))),
            "episodes": agg.episodes,
            "successes": agg.successes,
            "collisions": agg.collisions,
            "all_obstacle_collisions": agg.all_obs_collisions,
            "topples": agg.topples,
            "SR(%)": round(agg.sr_pct, 2),
            "CR(%)": round(agg.cr_pct, 2),
            "AllObsCR(%)": round(agg.all_obs_cr_pct, 2),
            "Topple(%)": round(agg.topple_pct, 2),
            "SASR(%)": round(agg.sasr_pct, 2),
            "SafetyScore": round(agg.safety_score, 4),
            "PeakForce(N)": "" if pf is None else round(pf, 2),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "model", "suite", "safety_level", "dynamic_level"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {out_path}  ({len(rows)} rows)")


def write_pivot_csvs(records: List[FileRecord], out_dir: Path) -> None:
    """Write two flavors of pivot CSVs, metric whitelists driven by LEVEL_METRICS
    and OVERVIEW_METRICS:

    1. ``summary_model_x_suite_L{level}.csv`` — one file per dynamic_level.
       Rows are ``(metric, model)``; columns are the suites + ``mean``.
       The set of metrics shown is whatever ``LEVEL_METRICS[level]`` says
       (e.g. L0/L1/L2 → SR/CR/Topple, L3 → SR/CR/PeakForce).
    2. ``summary_model_x_level.csv`` — single overview file. Rows are
       ``(metric, model)``; columns are L0/L1/L2/L3 + ``mean``. Metrics
       come from ``OVERVIEW_METRICS``. Cells where the metric isn't
       defined for that level (e.g. PeakForce on L0–L2) are left blank.
    """
    # --- groupings ------------------------------------------------------
    by_model_suite_level: Dict[Tuple[str, str, int], Aggregate] = defaultdict(Aggregate)
    by_model_level: Dict[Tuple[str, int], Aggregate] = defaultdict(Aggregate)
    for r in records:
        by_model_suite_level[(r.model, r.suite, r.dyn_level)].add(r)
        by_model_level[(r.model, r.dyn_level)].add(r)

    models = sorted({m for (m, _, _) in by_model_suite_level})
    suites = sorted({s for (_, s, _) in by_model_suite_level})
    levels = sorted({lv for (_, lv) in by_model_level})

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- one file per level, only the metrics requested for that level --
    for lv in levels:
        metrics = LEVEL_METRICS.get(lv, ("SR", "CR", "Topple"))
        out_path = out_dir / f"summary_model_x_suite_L{lv}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "model"] + suites + ["mean"])
            for metric in metrics:
                for m in models:
                    row: List[object] = [metric, m]
                    vals: List[float] = []
                    for s in suites:
                        agg = by_model_suite_level.get((m, s, lv))
                        v = agg.metric(metric) if (agg and agg.episodes > 0) else None
                        if v is None:
                            row.append("")
                        else:
                            row.append(_round_metric(metric, v))
                            vals.append(v)
                    row.append(_round_metric(metric, sum(vals) / len(vals)) if vals else "")
                    writer.writerow(row)
        print(f"[ok] wrote {out_path}  (metrics: {', '.join(metrics)})")

    # --- single overview file: union of OVERVIEW_METRICS × levels -------
    out_path = out_dir / "summary_model_x_level.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "model"] + [f"L{lv}" for lv in levels] + ["mean"])
        for metric in OVERVIEW_METRICS:
            for m in models:
                row: List[object] = [metric, m]
                vals: List[float] = []
                for lv in levels:
                    agg = by_model_level.get((m, lv))
                    v = agg.metric(metric) if (agg and agg.episodes > 0) else None
                    if v is None:
                        row.append("")
                    else:
                        row.append(_round_metric(metric, v))
                        vals.append(v)
                row.append(_round_metric(metric, sum(vals) / len(vals)) if vals else "")
                writer.writerow(row)
    print(f"[ok] wrote {out_path}  (metrics: {', '.join(OVERVIEW_METRICS)})")


def print_console_summary(records: List[FileRecord]) -> None:
    """Pretty-print a quick overview to the terminal."""
    if not records:
        print("[warn] no records parsed.")
        return

    # By (model, suite, dyn_level)
    grouped: Dict[Tuple[str, str, int], Aggregate] = defaultdict(Aggregate)
    for r in records:
        grouped[(r.model, r.suite, r.dyn_level)].add(r)

    model_w = max(14, max(len(r.model) for r in records) + 2)
    width = model_w + 22 + 3 + 7 + 6 + 7 * 5
    print("\n" + "=" * width)
    print(f"{'model':<{model_w}}{'suite':<22}{'L':>3} {'tasks':>6} {'ep':>5} "
          f"{'SR%':>7}{'CR%':>7}{'SASR%':>7}{'Top%':>7}{'1-CR':>7}")
    print("-" * width)
    for (model, suite, lv), agg in sorted(grouped.items()):
        print(f"{model:<{model_w}}{suite:<22}{lv:>3} {agg.n_files:>6} {agg.episodes:>5} "
              f"{agg.sr_pct:>7.2f}{agg.cr_pct:>7.2f}{agg.sasr_pct:>7.2f}"
              f"{agg.topple_pct:>7.2f}{agg.safety_score:>7.3f}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("dynisafety_libero/outputs"),
        help="Root directory containing <model>/logs subfolders.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict to these model dirs (default: scan all subdirs of outputs-root).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write CSV outputs (default: <outputs-root>/_aggregated).",
    )
    args = parser.parse_args()

    outputs_root: Path = args.outputs_root.resolve()
    out_dir: Path = (args.out_dir or (outputs_root / "_aggregated")).resolve()

    files = discover_log_files(outputs_root, args.models)
    if not files:
        print(f"[error] no EVAL-*.txt files found under {outputs_root}")
        return
    print(f"[info] found {len(files)} log files under {outputs_root}")

    records: List[FileRecord] = []
    for fp in files:
        rec = parse_log_file(fp)
        if rec is not None:
            records.append(rec)
    print(f"[info] parsed {len(records)}/{len(files)} successfully")

    records = dedupe_keep_latest(records)
    print(f"[info] {len(records)} unique (model, suite, level, task_id) records after dedup")

    print_console_summary(records)
    write_long_csv(records, out_dir / "summary_per_suite_level.csv")
    write_pivot_csvs(records, out_dir)


if __name__ == "__main__":
    main()
