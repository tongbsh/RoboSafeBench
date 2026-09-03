"""Offline rewrite of ``EpisodeRecord.diagnostics`` from saved traces.

Definition changes (closed-loop ``reacted``, 9-class labels) must not require
re-stepping a simulator. This module loops every completed run under a
results root, re-runs ``analyze_trace``, and atomically replaces
``episodes.jsonl``. ``traces/`` stay byte-identical.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dynisafety.config.schema import DiagnosticsConfig
from dynisafety.core.aggregate import discover_episode_files, load_episodes
from dynisafety.core.episode import EpisodeRecord
from dynisafety.diagnostics.analyze import analyze_trace
from dynisafety.diagnostics.calibrate import header_trace_relpath
from dynisafety.diagnostics.reader import iter_traces
from dynisafety.diagnostics.schema import Trace

_SWEEP_DIRNAME = "_sweep"


class RecomputeError(ValueError):
    """User-facing recompute failure (missing root, corrupt JSONL, …)."""


@dataclass(frozen=True)
class RecomputeReport:
    runs: int
    episodes: int
    rewritten: int
    unchanged: int
    skipped_no_trace: int
    skipped_incomplete: int
    dry_run: bool


def discover_recompute_files(
    results_root: Path | str,
    *,
    include_incomplete: bool = False,
) -> list[Path]:
    """``episodes.jsonl`` paths the recompute loop will visit."""
    root = Path(results_root)
    if not root.is_dir():
        raise RecomputeError(f"results root not found: {root}")
    if include_incomplete:
        return sorted(
            path
            for path in root.rglob("episodes.jsonl")
            if _SWEEP_DIRNAME not in path.parts
        )
    return discover_episode_files(root)


def recompute_results(
    results_root: Path | str,
    diagnostics: DiagnosticsConfig,
    *,
    dry_run: bool = False,
    include_incomplete: bool = False,
) -> RecomputeReport:
    """Loop completed runs and rewrite ``diagnostics`` from traces.

    Outcome fields (success / contact / force) are left as stored. Only
    ``EpisodeRecord.diagnostics`` is replaced. Missing traces skip that
    episode instead of inventing a blank 9-class label.
    """
    paths = discover_recompute_files(
        results_root, include_incomplete=include_incomplete
    )
    skipped_incomplete = 0
    if include_incomplete:
        completed = {p.resolve() for p in discover_episode_files(Path(results_root))}
        skipped_incomplete = sum(1 for p in paths if p.resolve() not in completed)

    runs = 0
    episodes = 0
    rewritten = 0
    unchanged = 0
    skipped_no_trace = 0
    for path in paths:
        stats = recompute_run(path.parent, diagnostics, dry_run=dry_run)
        runs += 1
        episodes += stats["episodes"]
        rewritten += stats["rewritten"]
        unchanged += stats["unchanged"]
        skipped_no_trace += stats["skipped_no_trace"]
    return RecomputeReport(
        runs=runs,
        episodes=episodes,
        rewritten=rewritten,
        unchanged=unchanged,
        skipped_no_trace=skipped_no_trace,
        skipped_incomplete=skipped_incomplete,
        dry_run=bool(dry_run),
    )


def recompute_run(
    run_dir: Path | str,
    diagnostics: DiagnosticsConfig,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Rewrite one run's ``episodes.jsonl``. Does not touch ``traces/``."""
    root = Path(run_dir)
    jsonl = root / "episodes.jsonl"
    if not jsonl.is_file():
        raise RecomputeError(f"episodes.jsonl not found: {jsonl}")
    records = load_episodes((jsonl,))
    traces = _traces_by_index(root)
    updated: list[EpisodeRecord] = []
    rewritten = 0
    unchanged = 0
    skipped_no_trace = 0
    for record in records:
        idx = int(record.episode_index)
        trace = traces.get(idx)
        if trace is None:
            updated.append(record)
            skipped_no_trace += 1
            continue
        fresh = _analyze_record(record, trace, diagnostics)
        if record.diagnostics == fresh:
            updated.append(record)
            unchanged += 1
            continue
        updated.append(replace(record, diagnostics=fresh))
        rewritten += 1
    if not dry_run:
        _write_episodes_atomic(jsonl, updated)
    return {
        "episodes": len(records),
        "rewritten": rewritten,
        "unchanged": unchanged,
        "skipped_no_trace": skipped_no_trace,
    }


def _traces_by_index(run_dir: Path) -> dict[int, Trace]:
    out: dict[int, Trace] = {}
    for trace in iter_traces(run_dir, strict=True):
        out[int(trace.header.episode_index)] = trace
    return out


def _analyze_record(
    record: EpisodeRecord,
    trace: Trace,
    diagnostics: DiagnosticsConfig,
) -> dict[str, Any]:
    stored = record.diagnostics if isinstance(record.diagnostics, dict) else {}
    trace_path = stored.get("trace_path")
    if not trace_path:
        trace_path = header_trace_relpath(trace.header)
    diag = analyze_trace(
        trace,
        diagnostics,
        success=bool(record.success),
        success_step=record.success_step,
        first_contact_step=record.first_contact_step,
        contact_steps=int(record.contact_steps),
        trace_path=str(trace_path),
    )
    return diag.to_json_dict()


def _write_episodes_atomic(path: Path, records: list[EpisodeRecord]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record.to_json_dict(), ensure_ascii=False, separators=(",", ":")))
            fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def register_diagnostics_parser(subparsers: Any) -> None:
    """Attach ``diagnostics recompute`` under the top-level CLI."""
    parser = subparsers.add_parser(
        "diagnostics",
        help="Offline behavior-diagnostics tools (no simulator)",
    )
    sub = parser.add_subparsers(dest="diagnostics_command", required=True)
    rec = sub.add_parser(
        "recompute",
        help="Rewrite diagnostics from traces (does not step the simulator)",
    )
    rec.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Results root containing run dirs with episodes.jsonl and traces/",
    )
    rec.add_argument(
        "--base-yaml",
        type=Path,
        required=True,
        help="YAML with a complete diagnostics: block (typically configs/base.yaml)",
    )
    rec.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and count diffs; do not write episodes.jsonl",
    )
    rec.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also visit runs whose summary.status is not completed",
    )


def dispatch_diagnostics(args: Any) -> int:
    if str(args.diagnostics_command) != "recompute":
        print(f"dynisafety diagnostics {args.diagnostics_command}: not implemented", flush=True)
        return 2
    from dynisafety.diagnostics.calibrate import CalibrationError, load_baseline_diagnostics

    try:
        cfg = load_baseline_diagnostics(Path(args.base_yaml))
        report = recompute_results(
            Path(args.results),
            cfg,
            dry_run=bool(args.dry_run),
            include_incomplete=bool(args.include_incomplete),
        )
    except (CalibrationError, RecomputeError) as exc:
        print(f"diagnostics recompute failed: {exc}", flush=True)
        return 2
    mode = "dry-run" if report.dry_run else "wrote"
    print(
        f"diagnostics recompute ({mode}): "
        f"runs={report.runs} episodes={report.episodes} "
        f"rewritten={report.rewritten} unchanged={report.unchanged} "
        f"skipped_no_trace={report.skipped_no_trace} "
        f"skipped_incomplete={report.skipped_incomplete}",
        flush=True,
    )
    return 0


__all__ = [
    "RecomputeError",
    "RecomputeReport",
    "discover_recompute_files",
    "dispatch_diagnostics",
    "recompute_results",
    "recompute_run",
    "register_diagnostics_parser",
]
