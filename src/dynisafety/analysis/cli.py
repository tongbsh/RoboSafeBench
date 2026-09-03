"""CLI for the offline paper-analysis package.

Does not participate in ``dynisafety report generate``. Writes only under
``--out``. The results tree is opened read-only.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dynisafety.analysis.counterfactual import write_counterfactual
from dynisafety.analysis.diversity import write_diversity
from dynisafety.analysis.exemplars import write_exemplars
from dynisafety.analysis.keyfindings import write_key_findings
from dynisafety.analysis.kinematics import write_kinematics
from dynisafety.analysis.load import fingerprint_results, load_bound_episodes
from dynisafety.analysis.paired_axes import write_paired_axes
from dynisafety.analysis.rank import DEFAULT_METRICS, write_rank
from dynisafety.analysis.termination import plot_cff_termination, write_termination_tables
from dynisafety.diagnostics.schema import BehaviorClass

_PAPER_GROUP = ("backend", "model", "level")


def _parse_group_by(raw: str) -> tuple[str, ...]:
    cols = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not cols:
        raise ValueError("--group-by must list at least one EpisodeRecord field")
    return cols


def _parse_metrics(raw: str) -> tuple[str, ...]:
    cols = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = set(DEFAULT_METRICS)
    unknown = [c for c in cols if c not in allowed]
    if unknown:
        raise ValueError(f"unknown rank metrics: {unknown}; expected {sorted(allowed)}")
    if not cols:
        raise ValueError("--metrics must list at least one of ssr,sr,cr")
    return cols


def _parse_model_list(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--models / --exclude-models given but empty")
    return values


def _model_filter(args: argparse.Namespace) -> dict[str, tuple[str, ...] | None]:
    return {
        "models": _parse_model_list(getattr(args, "models", None)),
        "exclude_models": _parse_model_list(getattr(args, "exclude_models", None)),
    }


def _parse_holdout(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_classes(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    known = {c.value for c in BehaviorClass}
    unknown = [v for v in values if v not in known]
    if unknown:
        raise ValueError(f"unknown behavior_class values: {unknown}; expected {sorted(known)}")
    return values


def add_subcommands(parser: argparse.ArgumentParser) -> None:
    """Attach paper-analysis subcommands onto ``parser``."""
    sub = parser.add_subparsers(dest="analysis_command", required=True)

    def add_io(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--results",
            type=Path,
            required=True,
            help="Results root containing run dirs with episodes.jsonl",
        )
        p.add_argument(
            "--out",
            type=Path,
            required=True,
            help="Output directory (created if missing). Never inside a run dir.",
        )
        # The results tree is keyed by suite/level, not by model, so every arm
        # that wrote into it is pooled unless filtered — including safety-layer
        # arms, which must not be ranked against the base policies.
        p.add_argument(
            "--models",
            default=None,
            help="Comma-separated allowlist of EpisodeRecord.model (default: all)",
        )
        p.add_argument(
            "--exclude-models",
            default=None,
            help="Comma-separated denylist, e.g. safety-layer arms",
        )

    p_term = sub.add_parser(
        "termination",
        help="Write table_termination and a CFF timeout stacked bar",
    )
    add_io(p_term)
    p_term.add_argument(
        "--group-by",
        default=",".join(_PAPER_GROUP),
        help="Comma-separated EpisodeRecord fields (default: backend,model,level)",
    )

    p_ex = sub.add_parser(
        "exemplars",
        help="Select one episode per (backend, level, class) and plot EEF XY",
    )
    add_io(p_ex)
    p_ex.add_argument("--per-class", type=int, default=1)
    p_ex.add_argument(
        "--classes",
        default=None,
        help="Optional comma-separated 9-class subset",
    )
    p_ex.add_argument(
        "--skip-not-triggered",
        action="store_true",
        help="Drop not_triggered cells (default: keep them)",
    )

    p_div = sub.add_parser(
        "diversity",
        help="Read extra.spawn_plan / extra.l3_plan (n/a when keys are absent)",
    )
    add_io(p_div)
    p_div.add_argument(
        "--group-by",
        default=",".join(_PAPER_GROUP),
        help="Comma-separated EpisodeRecord fields (default: backend,model,level)",
    )

    p_kin = sub.add_parser(
        "kinematics",
        help="Appendix EEF path / reversals / LDLJ (within-backend only)",
    )
    add_io(p_kin)
    p_kin.add_argument(
        "--group-by",
        default=",".join(_PAPER_GROUP),
        help="Comma-separated EpisodeRecord fields (default: backend,model,level)",
    )

    p_rank = sub.add_parser(
        "rank",
        help="Model-rank Spearman across backends / seeds / hold-out (n/a if 1 seed)",
    )
    add_io(p_rank)
    p_rank.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics (default: ssr,sr,cr)",
    )
    p_rank.add_argument(
        "--holdout-suites",
        default=None,
        help="Suite names treated as hold-out (default: name contains holdout)",
    )
    p_rank.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="Optional existing runs.csv; otherwise scores episodes under --results",
    )

    p_cf = sub.add_parser(
        "counterfactual",
        help="Reaction vs the matched level-0 control (needs a paired L0 run)",
    )
    add_io(p_cf)
    p_cf.add_argument(
        "--base-yaml",
        type=Path,
        required=True,
        help="YAML with a complete diagnostics: block (typically configs/base.yaml)",
    )
    p_cf.add_argument(
        "--group-by",
        default=",".join(_PAPER_GROUP),
        help=f"Comma-separated EpisodeRecord fields (default: {','.join(_PAPER_GROUP)})",
    )

    p_paired = sub.add_parser(
        "paired-axes",
        help=(
            "Four axes scored against the matched level-0 twin, so hazard "
            "episodes with no pre-onset window (t_onset=0) get a label too"
        ),
    )
    add_io(p_paired)
    p_paired.add_argument(
        "--base-yaml",
        type=Path,
        required=True,
        help="YAML with a complete diagnostics: block (typically configs/base.yaml)",
    )
    p_paired.add_argument(
        "--group-by",
        default=",".join(_PAPER_GROUP),
        help=f"Comma-separated EpisodeRecord fields (default: {','.join(_PAPER_GROUP)})",
    )

    p_kf = sub.add_parser(
        "key-findings",
        help="Cross-backend key findings (coverage, outcomes, CFF split, warnings)",
    )
    add_io(p_kf)
    p_kf.add_argument(
        "--base-yaml",
        type=Path,
        required=True,
        help="YAML with a complete diagnostics: block (typically configs/base.yaml)",
    )
    p_kf.add_argument(
        "--group-by",
        default=",".join(_PAPER_GROUP),
        help=f"Comma-separated EpisodeRecord fields (default: {','.join(_PAPER_GROUP)})",
    )
    p_kf.add_argument(
        "--analysis-root",
        type=Path,
        default=None,
        help=(
            "Where to read already-written paired_axes / termination CSVs. "
            "Default: this command's --out (sibling stages and <backend>/ subdirs)."
        ),
    )

    p_all = sub.add_parser(
        "all",
        help=(
            "Run termination + exemplars + diversity + kinematics + rank, "
            "plus counterfactual, paired-axes and key-findings when --base-yaml is given"
        ),
    )
    add_io(p_all)
    p_all.add_argument(
        "--base-yaml",
        type=Path,
        default=None,
        help=(
            "YAML with a complete diagnostics: block. Omit to skip the "
            "counterfactual stage rather than run it on implicit defaults."
        ),
    )
    p_all.add_argument(
        "--group-by",
        default=",".join(_PAPER_GROUP),
        help="Comma-separated EpisodeRecord fields (default: backend,model,level)",
    )
    p_all.add_argument("--per-class", type=int, default=1)
    p_all.add_argument("--classes", default=None)
    p_all.add_argument("--skip-not-triggered", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynisafety analysis",
        description=(
            "Read-only paper analysis: timeout, exemplars, diversity, "
            "kinematics, and rank stability. Does not modify results/ or backends."
        ),
    )
    add_subcommands(parser)
    return parser


def _refuse_out_inside_results(results: Path, out: Path) -> None:
    try:
        out.resolve().relative_to(results.resolve())
    except ValueError:
        return
    raise ValueError(
        f"--out {out} resolves inside --results {results}. "
        "Write analysis artifacts to a separate directory so the results "
        "tree stays byte-identical."
    )


def run_termination(
    results: Path,
    out: Path,
    *,
    group_by: Sequence[str],
    model_filter: dict[str, Any],
) -> int:
    before = fingerprint_results(results)
    bound = load_bound_episodes(results, **model_filter)
    write_termination_tables(bound, out, group_cols=group_by)
    plot_cff_termination(bound, Path(out) / "plots" / "cff_termination.png")
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.termination mutated the results tree")
    print(f"wrote table_termination + plots/cff_termination.png under {out}")
    print(f"episodes: {len(bound)}")
    return 0


def run_counterfactual(
    results: Path,
    out: Path,
    *,
    base_yaml: Path,
    group_by: Sequence[str],
    model_filter: dict[str, Any],
) -> int:
    from dynisafety.diagnostics.calibrate import load_baseline_diagnostics

    cfg = load_baseline_diagnostics(base_yaml)
    before = fingerprint_results(results)
    bound = load_bound_episodes(results, **model_filter)
    written = write_counterfactual(
        bound,
        out,
        cfg.counterfactual,
        min_latency_chunks=float(cfg.reaction.min_latency_chunks),
        stop_speed_frac=float(cfg.stop.speed_frac),
        stop_speed_floor=float(cfg.stop.speed_floor_m_per_step),
        baseline_window=int(cfg.reaction.window_size),
        group_cols=group_by,
    )
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.counterfactual mutated the results tree")
    coverage = written["table_counterfactual_coverage"].iloc[0]
    print(f"wrote counterfactual tables + plots under {out}")
    print(
        f"hazard episodes: {int(coverage['n_hazard_episodes'])}; "
        f"paired: {int(coverage['n_paired'])}; scored: {int(coverage['n_scored'])}; "
        f"borrowed-null: {int(coverage.get('n_scored_borrowed_null', 0))}"
    )
    return 0


def run_paired_axes(
    results: Path,
    out: Path,
    *,
    base_yaml: Path,
    group_by: Sequence[str],
    model_filter: dict[str, Any],
) -> int:
    from dynisafety.diagnostics.calibrate import load_baseline_diagnostics

    cfg = load_baseline_diagnostics(base_yaml)
    before = fingerprint_results(results)
    bound = load_bound_episodes(results, **model_filter)
    written = write_paired_axes(bound, out, cfg, group_cols=group_by)
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.paired_axes mutated the results tree")
    coverage = written["table_paired_coverage"].iloc[0]
    print(f"wrote paired four-axis tables + plots under {out}")
    print(
        f"hazard episodes: {int(coverage['n_hazard_episodes'])}; "
        f"scored: {int(coverage['n_scored'])}; "
        f"recovered-from-baseline_unusable: {int(coverage['n_scored_own_unusable'])}; "
        f"no pre-onset window: {int(coverage['n_scored_no_pre_onset'])}"
    )
    return 0


def run_key_findings(
    results: Path,
    out: Path,
    *,
    base_yaml: Path,
    group_by: Sequence[str],
    model_filter: dict[str, Any],
    analysis_root: Path | None,
) -> int:
    from dynisafety.diagnostics.calibrate import load_baseline_diagnostics

    cfg = load_baseline_diagnostics(base_yaml)
    before = fingerprint_results(results)
    bound = load_bound_episodes(results, **model_filter)
    written = write_key_findings(
        bound,
        out,
        cfg,
        group_cols=group_by,
        analysis_root=analysis_root if analysis_root is not None else out.parent,
        results_root=results,
    )
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.key_findings mutated the results tree")
    warnings = written["table_warnings"]
    n_warn = 0 if warnings.empty else int(len(warnings))
    print(f"wrote key_findings.md + tables under {out}")
    print(f"episodes: {len(bound)}; warning rows: {n_warn}")
    return 0


def run_rank(
    results: Path,
    out: Path,
    *,
    metrics: Sequence[str],
    holdout_suites: Sequence[str],
    runs_csv: Path | None,
    model_filter: dict[str, Any],
) -> int:
    before = fingerprint_results(results)
    bound = load_bound_episodes(results, **model_filter)
    written = write_rank(
        bound,
        out,
        metrics=metrics,
        holdout_suites=holdout_suites,
        runs_csv=runs_csv,
        exclude_safety_layers=model_filter.get("models") is None,
    )
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.rank mutated the results tree")
    n_models = int(written["table_rank_coverage"].iloc[0]["n_models"])
    print(f"wrote rank-stability tables + plots under {out}")
    print(f"episodes: {len(bound)}; models: {n_models}")
    return 0


def run_kinematics(
    results: Path,
    out: Path,
    *,
    group_by: Sequence[str],
    model_filter: dict[str, Any],
) -> int:
    before = fingerprint_results(results)
    bound = load_bound_episodes(results, **model_filter)
    written = write_kinematics(bound, out, group_cols=group_by)
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.kinematics mutated the results tree")
    episodes = written["kinematics_episodes"]
    n_traced = int(episodes["has_trace"].sum()) if len(episodes) else 0
    print(f"wrote kinematics tables + plots under {out}")
    print(f"episodes: {len(bound)}; with_trace: {n_traced}")
    return 0


def run_diversity(
    results: Path,
    out: Path,
    *,
    group_by: Sequence[str],
    model_filter: dict[str, Any],
) -> int:
    before = fingerprint_results(results)
    bound = load_bound_episodes(results, **model_filter)
    written = write_diversity(bound, out, group_cols=group_by)
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.diversity mutated the results tree")
    n_planned = int(len(written["diversity_episodes"]))
    print(f"wrote diversity tables + plots under {out}")
    print(f"episodes: {len(bound)}; planned: {n_planned}")
    return 0


def run_exemplars(
    results: Path,
    out: Path,
    *,
    per_class: int,
    classes: Sequence[str] | None,
    include_not_triggered: bool,
    model_filter: dict[str, Any],
) -> int:
    before = fingerprint_results(results)
    bound = load_bound_episodes(results, **model_filter)
    rows = write_exemplars(
        bound,
        out,
        per_class=per_class,
        classes=classes,
        include_not_triggered=include_not_triggered,
    )
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.exemplars mutated the results tree")
    print(f"wrote exemplars.csv and {sum(1 for r in rows if r.plot_path)} plots under {out}")
    print(f"selected: {len(rows)}")
    return 0


def dispatch(args: argparse.Namespace) -> int:
    results = Path(args.results)
    out = Path(args.out)
    _refuse_out_inside_results(results, out)
    cmd = str(args.analysis_command)
    model_filter = _model_filter(args)
    # Every stage owns ``out/<stage>/``, for ``all`` and for a single stage
    # alike. Letting one stage write straight into ``--out`` meant rerunning it
    # after ``all`` scattered a second copy of its tables at the top level,
    # where nothing later overwrites or removes them.
    if cmd == "termination":
        return run_termination(results, out / cmd, group_by=_parse_group_by(args.group_by), model_filter=model_filter)
    if cmd == "exemplars":
        return run_exemplars(
            results,
            out / cmd,
            per_class=int(args.per_class),
            classes=_parse_classes(args.classes),
            include_not_triggered=not bool(args.skip_not_triggered),
            model_filter=model_filter,
        )
    if cmd == "diversity":
        return run_diversity(results, out / cmd, group_by=_parse_group_by(args.group_by), model_filter=model_filter)
    if cmd == "kinematics":
        return run_kinematics(results, out / cmd, group_by=_parse_group_by(args.group_by), model_filter=model_filter)
    if cmd == "counterfactual":
        return run_counterfactual(
            results,
            out / cmd,
            base_yaml=Path(args.base_yaml),
            group_by=_parse_group_by(args.group_by),
            model_filter=model_filter,
        )
    if cmd == "paired-axes":
        return run_paired_axes(
            results,
            out / "paired_axes",
            base_yaml=Path(args.base_yaml),
            group_by=_parse_group_by(args.group_by),
            model_filter=model_filter,
        )
    if cmd == "key-findings":
        analysis_root = Path(args.analysis_root) if getattr(args, "analysis_root", None) else out
        return run_key_findings(
            results,
            out / "key_findings",
            base_yaml=Path(args.base_yaml),
            group_by=_parse_group_by(args.group_by),
            model_filter=model_filter,
            analysis_root=analysis_root,
        )
    if cmd == "rank":
        return run_rank(
            results,
            out / cmd,
            metrics=_parse_metrics(args.metrics),
            holdout_suites=_parse_holdout(args.holdout_suites),
            runs_csv=Path(args.runs) if args.runs is not None else None,
            model_filter=model_filter,
        )
    if cmd == "all":
        group_by = _parse_group_by(args.group_by)
        run_termination(results, out / "termination", group_by=group_by, model_filter=model_filter)
        run_exemplars(
            results,
            out / "exemplars",
            per_class=int(args.per_class),
            classes=_parse_classes(args.classes),
            include_not_triggered=not bool(args.skip_not_triggered),
            model_filter=model_filter,
        )
        run_diversity(results, out / "diversity", group_by=group_by, model_filter=model_filter)
        run_kinematics(results, out / "kinematics", group_by=group_by, model_filter=model_filter)
        if getattr(args, "base_yaml", None) is not None:
            run_counterfactual(
                results,
                out / "counterfactual",
                base_yaml=Path(args.base_yaml),
                group_by=group_by,
                model_filter=model_filter,
            )
            run_paired_axes(
                results,
                out / "paired_axes",
                base_yaml=Path(args.base_yaml),
                group_by=group_by,
                model_filter=model_filter,
            )
            run_key_findings(
                results,
                out / "key_findings",
                base_yaml=Path(args.base_yaml),
                group_by=group_by,
                model_filter=model_filter,
                analysis_root=out,
            )
        return run_rank(
            results,
            out / "rank",
            metrics=DEFAULT_METRICS,
            holdout_suites=(),
            runs_csv=None,
            model_filter=model_filter,
        )
    raise ValueError(f"unknown analysis command {cmd!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except Exception as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 1


def register_analysis_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``analysis`` under the top-level ``dynisafety`` parser."""
    parser = subparsers.add_parser(
        "analysis",
        help="Read-only paper analysis (does not change report)",
    )
    add_subcommands(parser)
