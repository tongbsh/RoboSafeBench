"""``dynisafety sweep`` — multi-job matrix scheduler (frozen flags)."""

from __future__ import annotations

import argparse
import sys

from dynisafety.config.resolve import find_workspace_root
from dynisafety.runner.scheduler import (
    SchedulerConfig,
    SweepScheduler,
    detect_gpu_ids,
    estimate_totals,
    expand_jobs,
    load_sweep_suite,
    parse_only_filter,
    resolve_suite_path,
)


def register_sweep_parser(subparsers: argparse._SubParsersAction) -> None:
    """Frozen flags + ``--retry-failed`` (S2 确认 D2)."""
    p = subparsers.add_parser("sweep", help="Run a declarative sweep suite")
    p.add_argument(
        "--suite",
        required=True,
        help="Sweep suite name under configs/suites/ or path to YAML",
    )
    p.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids (omit → runtime discovery)",
    )
    p.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=None,
        help="Global cap on concurrent jobs per GPU (omit → model YAML only)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs with completed summary.json",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only (re)run jobs listed in results/_sweep/<suite>/failures.jsonl",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Filter jobs, e.g. model=starvla,level=3 or level=1|2 (OR within a key)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job plan and totals; do not launch",
    )
    p.set_defaults(sweep_handler=dispatch_sweep)


def _parse_gpus(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("--gpus is empty")
    return tuple(int(p) for p in parts)


def run_sweep(
    *,
    suite: str,
    gpus: str | None,
    jobs_per_gpu: int | None,
    resume: bool,
    retry_failed: bool,
    only: str | None,
    dry_run: bool,
) -> int:
    import os

    workspace = find_workspace_root()
    configs_root = workspace / "configs"
    try:
        suite_path = resolve_suite_path(suite, configs_root=configs_root)
        sweep_suite = load_sweep_suite(suite_path)
        only_filter = parse_only_filter(only)
        gpu_ids = detect_gpu_ids(_parse_gpus(gpus))
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"[sweep] ERROR: {exc}", file=sys.stderr)
        return 2

    jobs = expand_jobs(sweep_suite, configs_root=configs_root)
    root = workspace / "results"
    sweep_dir = root / "_sweep" / sweep_suite.name

    # Operational timeout (not a sim hyperparameter); env override only.
    health_raw = os.environ.get("DYNISAFETY_SERVER_WAIT_TIMEOUT")
    health_timeout_s = float(health_raw) if health_raw else 600.0

    cfg = SchedulerConfig(
        gpu_ids=gpu_ids,
        jobs_per_gpu_cap=jobs_per_gpu,
        resume=resume,
        retry_failed=retry_failed,
        dry_run=dry_run,
        results_root=root,
        sweep_dir=sweep_dir,
        configs_root=configs_root,
        health_timeout_s=health_timeout_s,
        only=only_filter,
    )
    scheduler = SweepScheduler(cfg, jobs)
    if dry_run:
        full = estimate_totals(jobs)
        print(f"[sweep] suite              = {sweep_suite.name}")
        print(f"[sweep] suite_path         = {suite_path}")
        print(f"[sweep] expanded_n_jobs    = {full['n_jobs']}")
        print(f"[sweep] expanded_episodes  = {full['n_episodes_total']}")
    return scheduler.run()


def dispatch_sweep(args: argparse.Namespace) -> int:
    return run_sweep(
        suite=args.suite,
        gpus=args.gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        resume=bool(args.resume),
        retry_failed=bool(args.retry_failed),
        only=args.only,
        dry_run=bool(args.dry_run),
    )
