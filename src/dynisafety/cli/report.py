"""``dynisafety report`` / ``report validate`` — aggregate JSONL into paper artifacts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from dynisafety.config.schema import MetricsConfig, ReactivityConfig
from dynisafety.core.aggregate import (
    AggregateOptions,
    CoverageReport,
    aggregate_groups,
    discover_episode_files,
    episodes_to_frame,
    filter_episodes_for_grid,
    load_coverage_grid,
    load_episodes,
    validate_coverage,
    write_episodes_parquet,
    write_runs_csv,
)
from dynisafety.core.episode import EpisodeRecord
from dynisafety.report.plots import render_all_plots, render_joint_plots
from dynisafety.report.tables import (
    UNAVAILABLE,
    check_joint_table_consistency,
    render_all_tables,
    render_joint_tables,
)

_DEFAULT_GROUP_KEYS = (
    "backend",
    "model",
    "suite",
    "task_id",
    "level",
    "seed",
    "config_hash",
)


@dataclass(frozen=True)
class ReportSettings:
    """Loaded from YAML; no Python defaults — all fields required in the file."""

    bootstrap_samples: int
    bootstrap_seed: int
    confidence: float
    max_error_rate: float


def load_metrics_config(path: Path) -> MetricsConfig:
    """Load ``MetricsConfig`` from a YAML file (full base.yaml or metrics-only)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"metrics YAML must be a mapping: {path}")
    section = raw["metrics"] if "metrics" in raw else raw
    if not isinstance(section, Mapping):
        raise ValueError(f"metrics section must be a mapping: {path}")
    react_raw = section["reactivity"]
    if not isinstance(react_raw, Mapping):
        raise ValueError("metrics.reactivity must be a mapping")
    reactivity = ReactivityConfig(
        window_size=int(react_raw["window_size"]),
        sensitivity=float(react_raw["sensitivity"]),
        freeze_threshold=float(react_raw["freeze_threshold"]),
        freeze_min_steps=int(react_raw["freeze_min_steps"]),
    )
    thresholds = dict(section["injury_thresholds_n"])
    # YAML may load .inf as float('inf')
    return MetricsConfig(
        injury_thresholds_n={str(k): float(v) for k, v in thresholds.items()},
        sasr_penalty=dict(section["sasr_penalty"]),
        reactivity=reactivity,
    )


def load_report_settings(path: Path) -> ReportSettings:
    """Load report CI/coverage settings from YAML (top-level or ``report:`` section)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"report settings YAML must be a mapping: {path}")
    section = raw["report"] if "report" in raw else raw
    if not isinstance(section, Mapping):
        raise ValueError(f"report section must be a mapping: {path}")
    required = (
        "bootstrap_samples",
        "bootstrap_seed",
        "confidence",
        "max_error_rate",
    )
    missing = [k for k in required if k not in section]
    if missing:
        raise KeyError(f"report settings missing keys: {missing}")
    return ReportSettings(
        bootstrap_samples=int(section["bootstrap_samples"]),
        bootstrap_seed=int(section["bootstrap_seed"]),
        confidence=float(section["confidence"]),
        max_error_rate=float(section["max_error_rate"]),
    )


@dataclass(frozen=True)
class JointReportSettings:
    """Joint-outcome report options loaded from YAML (all keys required)."""

    group_by: tuple[str, ...]
    stack_normalize: bool
    include_pareto: bool
    include_heatmap: bool
    unavailable_token: str


def load_joint_report_settings(path: Path) -> JointReportSettings:
    """Load joint report settings (top-level or ``joint_outcomes:`` section)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"joint settings YAML must be a mapping: {path}")
    section = raw["joint_outcomes"] if "joint_outcomes" in raw else raw
    if not isinstance(section, Mapping):
        raise ValueError(f"joint_outcomes section must be a mapping: {path}")
    required = (
        "group_by",
        "stack_normalize",
        "include_pareto",
        "include_heatmap",
        "unavailable_token",
    )
    missing = [k for k in required if k not in section]
    if missing:
        raise KeyError(f"joint settings missing keys: {missing}")
    group_by = section["group_by"]
    if not isinstance(group_by, (list, tuple)) or not group_by:
        raise ValueError("joint_outcomes.group_by must be a non-empty list")
    token = str(section["unavailable_token"])
    if token != UNAVAILABLE:
        # Allow only the locked token so tables never invent alternate spellings silently.
        raise ValueError(
            f"joint_outcomes.unavailable_token must be {UNAVAILABLE!r}, got {token!r}"
        )
    return JointReportSettings(
        group_by=tuple(str(x) for x in group_by),
        stack_normalize=bool(section["stack_normalize"]),
        include_pareto=bool(section["include_pareto"]),
        include_heatmap=bool(section["include_heatmap"]),
        unavailable_token=token,
    )


def _print_coverage_issues(report: CoverageReport, stream=sys.stderr) -> None:
    print(
        f"coverage incomplete: {len(report.issues)} issue(s)",
        file=stream,
    )
    for issue in report.issues:
        print(
            f"  [{issue.kind}] {issue.key}: {issue.message} "
            f"(expected={issue.expected!r}, actual={issue.actual!r})",
            file=stream,
        )


def run_coverage(
    *,
    results: Path,
    out: Path,
    grid: Path,
    settings: ReportSettings,
    expected_config_hash: str | None,
) -> tuple[list[EpisodeRecord], CoverageReport]:
    """Load episodes, slice to this grid, validate, write ``coverage.csv``.

    The slice is what ``generate`` aggregates: a shared ``results/`` tree can
    hold every paper backend without each grid's ``table_main`` absorbing the
    others. ``backend`` / ``suite`` on the grid (or inferred from the filename)
    are part of the cell key.
    """
    paths = discover_episode_files(results)
    expected = load_coverage_grid(grid)
    episodes = filter_episodes_for_grid(load_episodes(paths), expected)

    coverage = validate_coverage(
        pd.DataFrame(),
        episodes,
        expected,
        max_error_rate=settings.max_error_rate,
        expected_config_hash=expected_config_hash,
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    coverage.rows.to_csv(out / "coverage.csv", index=False)
    return episodes, coverage


def generate_report(
    *,
    results: Path,
    out: Path,
    grid: Path,
    metrics_cfg: MetricsConfig,
    settings: ReportSettings,
    allow_incomplete: bool,
    expected_config_hash: str | None,
    write_plots: bool,
    write_parquet: bool,
) -> int:
    """Full report pipeline. Returns 0 on success, ≠0 if refused or incomplete validate.

    When coverage is incomplete and ``allow_incomplete`` is False, writes only
    ``coverage.csv`` and refuses tables/runs/plots.

    Successful generate also writes ``table_behavior.{csv,md,tex}`` and
    ``plots/behavior_classes.png`` / ``plots/failure_decomposition.png``.
    """
    episodes, coverage = run_coverage(
        results=results,
        out=out,
        grid=grid,
        settings=settings,
        expected_config_hash=expected_config_hash,
    )
    if not coverage.complete and not allow_incomplete:
        _print_coverage_issues(coverage)
        print(
            "refusing to emit tables (pass --allow-incomplete to override)",
            file=sys.stderr,
        )
        return 1

    options = AggregateOptions(
        group_keys=_DEFAULT_GROUP_KEYS,
        ci_method="bootstrap",
        bootstrap_samples=settings.bootstrap_samples,
        bootstrap_seed=settings.bootstrap_seed,
        confidence=settings.confidence,
    )
    runs_df = aggregate_groups(episodes, metrics_cfg, options)
    write_runs_csv(runs_df, Path(out) / "runs.csv")
    episodes_df = episodes_to_frame(episodes)
    if write_parquet:
        try:
            write_episodes_parquet(episodes_df, Path(out) / "episodes.parquet")
        except ImportError as exc:
            print(f"warning: skip episodes.parquet: {exc}", file=sys.stderr)

    backends = sorted({ep.backend for ep in episodes})
    render_all_tables(runs_df, Path(out), backends_present=backends)
    if write_plots:
        render_all_plots(runs_df, episodes_df, Path(out))
    return 0


def run_report_joint(args: argparse.Namespace) -> int:
    """Regenerate joint-outcome tables/plots from ``runs.csv`` or ``results/``.

    ``--results`` must also pass ``--grid``. The grid slices the tree the same
    way ``report generate`` does, so a shared model id (``starvla``) is not
    pooled across backends or LIBERO suites.
    """
    joint_cfg = load_joint_report_settings(Path(args.joint_yaml))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    oracle_df = None
    if args.oracle_runs is not None:
        oracle_path = Path(args.oracle_runs)
        if oracle_path.exists():
            oracle_df = pd.read_csv(oracle_path)
        else:
            print(
                f"warning: oracle runs not found ({oracle_path}); "
                f"Oracle columns → {UNAVAILABLE}",
                file=sys.stderr,
            )

    if args.runs is not None:
        runs_df = pd.read_csv(Path(args.runs))
    elif args.results is not None:
        if args.metrics_yaml is None or args.report_yaml is None:
            print(
                "dynisafety report joint --results requires --metrics-yaml and --report-yaml",
                file=sys.stderr,
            )
            return 2
        grid = getattr(args, "grid", None)
        if grid is None:
            print(
                "dynisafety report joint --results requires --grid "
                "(same coverage grid as report generate; slices backend/suite "
                "so shared model ids such as starvla are not pooled)",
                file=sys.stderr,
            )
            return 2
        metrics_cfg = load_metrics_config(Path(args.metrics_yaml))
        settings = load_report_settings(Path(args.report_yaml))
        expected = load_coverage_grid(Path(grid))
        paths = discover_episode_files(Path(args.results))
        episodes = filter_episodes_for_grid(load_episodes(paths), expected)
        options = AggregateOptions(
            group_keys=_DEFAULT_GROUP_KEYS,
            ci_method="bootstrap",
            bootstrap_samples=settings.bootstrap_samples,
            bootstrap_seed=settings.bootstrap_seed,
            confidence=settings.confidence,
        )
        runs_df = aggregate_groups(episodes, metrics_cfg, options)
        write_runs_csv(runs_df, out / "runs.csv")
    else:
        print("dynisafety report joint: require --runs or --results", file=sys.stderr)
        return 2

    issues = check_joint_table_consistency(runs_df)
    if issues:
        print(f"joint consistency warnings ({len(issues)}):", file=sys.stderr)
        for msg in issues[:20]:
            print(f"  {msg}", file=sys.stderr)

    gcols = [c for c in joint_cfg.group_by if c in runs_df.columns]
    if not gcols:
        gcols = [c for c in ("model", "level") if c in runs_df.columns]
    render_joint_tables(
        runs_df,
        out,
        group_cols=gcols or ("model",),
        oracle_runs_df=oracle_df,
    )
    if not args.no_plots:
        render_joint_plots(
            runs_df,
            out,
            stack_normalize=joint_cfg.stack_normalize,
        )
        plots_dir = out / "plots"
        if not joint_cfg.include_pareto:
            for name in ("pareto_sr_ssr.png", "pareto_sr_cr.png"):
                p = plots_dir / name
                if p.exists():
                    p.unlink()
        if not joint_cfg.include_heatmap:
            p = plots_dir / "heatmap_ssr_task_level.png"
            if p.exists():
                p.unlink()
    print(f"joint report → {out}")
    return 0


def run_report_generate(args: argparse.Namespace) -> int:
    metrics_cfg = load_metrics_config(Path(args.metrics_yaml))
    settings = load_report_settings(Path(args.report_yaml))
    return generate_report(
        results=Path(args.results),
        out=Path(args.out),
        grid=Path(args.grid),
        metrics_cfg=metrics_cfg,
        settings=settings,
        allow_incomplete=bool(args.allow_incomplete),
        expected_config_hash=args.expected_config_hash,
        write_plots=not bool(args.no_plots),
        write_parquet=not bool(args.no_parquet),
    )


def run_report_validate(args: argparse.Namespace) -> int:
    settings = load_report_settings(Path(args.report_yaml))
    _episodes, coverage = run_coverage(
        results=Path(args.results),
        out=Path(args.out),
        grid=Path(args.grid),
        settings=settings,
        expected_config_hash=args.expected_config_hash,
    )
    if coverage.complete:
        print(f"coverage OK → {Path(args.out) / 'coverage.csv'}")
        return 0
    _print_coverage_issues(coverage)
    return 1


def register_report_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``report`` with ``generate`` / ``validate`` subcommands."""
    report = subparsers.add_parser("report", help="Aggregate JSONL results into reports")
    report_sub = report.add_subparsers(dest="report_command", required=True)

    def _add_common(p: argparse.ArgumentParser) -> None:
        # Paths and settings are required; no hyperparameter default=.
        p.add_argument("--results", type=Path, required=True, help="Results root with episodes.jsonl")
        p.add_argument("--out", type=Path, required=True, help="Output reports/ directory")
        p.add_argument("--grid", type=Path, required=True, help="Coverage grid YAML")
        p.add_argument(
            "--report-yaml",
            type=Path,
            required=True,
            help="YAML with bootstrap_samples/bootstrap_seed/confidence/max_error_rate",
        )
        p.add_argument(
            "--expected-config-hash",
            default=None,
            help="Optional expected config_hash for coverage check",
        )

    p_gen = report_sub.add_parser("generate", help="Write runs/tables/plots (coverage-gated)")
    _add_common(p_gen)
    p_gen.add_argument(
        "--metrics-yaml",
        type=Path,
        required=True,
        help="YAML containing metrics.* (e.g. configs/base.yaml)",
    )
    p_gen.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Emit tables even when coverage is incomplete",
    )
    p_gen.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot rendering",
    )
    p_gen.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip episodes.parquet",
    )
    p_gen.set_defaults(report_handler=run_report_generate)

    p_val = report_sub.add_parser("validate", help="Write coverage.csv and exit on gaps")
    _add_common(p_val)
    p_val.set_defaults(report_handler=run_report_validate)

    p_joint = report_sub.add_parser(
        "joint",
        help="Regenerate SS/US/CFF/CIF tables and plots from runs.csv or results/",
    )
    src = p_joint.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--runs",
        type=Path,
        help="Existing runs.csv (with ss/us/cff/cif/ssr when available)",
    )
    src.add_argument(
        "--results",
        type=Path,
        help="Results root with episodes.jsonl (re-aggregate then plot)",
    )
    p_joint.add_argument("--out", type=Path, required=True, help="Output directory")
    p_joint.add_argument(
        "--joint-yaml",
        type=Path,
        required=True,
        help="YAML with joint_outcomes.* (e.g. configs/report/joint_outcomes.yaml)",
    )
    p_joint.add_argument(
        "--report-yaml",
        type=Path,
        default=None,
        help="Required with --results: bootstrap/coverage settings",
    )
    p_joint.add_argument(
        "--metrics-yaml",
        type=Path,
        default=None,
        help="Required with --results: metrics YAML (e.g. configs/base.yaml)",
    )
    p_joint.add_argument(
        "--grid",
        type=Path,
        default=None,
        help="Required with --results: coverage grid (backend/suite slice)",
    )
    p_joint.add_argument(
        "--oracle-runs",
        type=Path,
        default=None,
        help="Optional Oracle runs.csv; if omitted Oracle columns are unavailable",
    )
    p_joint.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot rendering",
    )
    p_joint.set_defaults(report_handler=run_report_joint)


def dispatch_report(args: argparse.Namespace) -> int:
    handler = getattr(args, "report_handler", None)
    if handler is None:
        print("dynisafety report: missing subcommand", file=sys.stderr)
        return 2
    return int(handler(args))
