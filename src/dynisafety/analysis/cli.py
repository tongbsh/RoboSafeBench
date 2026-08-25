"""CLI for the offline paper-analysis package.

Does not participate in ``dynisafety report generate``. Writes only under
``--out``. The results tree is opened read-only.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dynisafety.analysis.diversity import write_diversity
from dynisafety.analysis.exemplars import write_exemplars
from dynisafety.analysis.kinematics import write_kinematics
from dynisafety.analysis.load import fingerprint_results, load_bound_episodes
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

    p_all = sub.add_parser(
        "all",
        help="Run termination + exemplars + diversity + kinematics + rank",
    )
    add_io(p_all)
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


def run_termination(results: Path, out: Path, *, group_by: Sequence[str]) -> int:
    before = fingerprint_results(results)
    bound = load_bound_episodes(results)
    write_termination_tables(bound, out, group_cols=group_by)
    plot_cff_termination(bound, Path(out) / "plots" / "cff_termination.png")
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.termination mutated the results tree")
    print(f"wrote table_termination + plots/cff_termination.png under {out}")
    print(f"episodes: {len(bound)}")
    return 0


def run_rank(
    results: Path,
    out: Path,
    *,
    metrics: Sequence[str],
    holdout_suites: Sequence[str],
    runs_csv: Path | None,
) -> int:
    before = fingerprint_results(results)
    bound = load_bound_episodes(results)
    written = write_rank(
        bound,
        out,
        metrics=metrics,
        holdout_suites=holdout_suites,
        runs_csv=runs_csv,
    )
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.rank mutated the results tree")
    n_models = int(written["table_rank_coverage"].iloc[0]["n_models"])
    print(f"wrote rank-stability tables + plots under {out}")
    print(f"episodes: {len(bound)}; models: {n_models}")
    return 0


def run_kinematics(results: Path, out: Path, *, group_by: Sequence[str]) -> int:
    before = fingerprint_results(results)
    bound = load_bound_episodes(results)
    written = write_kinematics(bound, out, group_cols=group_by)
    after = fingerprint_results(results)
    if before != after:
        raise RuntimeError("analysis.kinematics mutated the results tree")
    episodes = written["kinematics_episodes"]
    n_traced = int(episodes["has_trace"].sum()) if len(episodes) else 0
    print(f"wrote kinematics tables + plots under {out}")
    print(f"episodes: {len(bound)}; with_trace: {n_traced}")
    return 0


def run_diversity(results: Path, out: Path, *, group_by: Sequence[str]) -> int:
    before = fingerprint_results(results)
    bound = load_bound_episodes(results)
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
) -> int:
    before = fingerprint_results(results)
    bound = load_bound_episodes(results)
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
    if cmd == "termination":
        return run_termination(results, out, group_by=_parse_group_by(args.group_by))
    if cmd == "exemplars":
        return run_exemplars(
            results,
            out,
            per_class=int(args.per_class),
            classes=_parse_classes(args.classes),
            include_not_triggered=not bool(args.skip_not_triggered),
        )
    if cmd == "diversity":
        return run_diversity(results, out, group_by=_parse_group_by(args.group_by))
    if cmd == "kinematics":
        return run_kinematics(results, out, group_by=_parse_group_by(args.group_by))
    if cmd == "rank":
        return run_rank(
            results,
            out,
            metrics=_parse_metrics(args.metrics),
            holdout_suites=_parse_holdout(args.holdout_suites),
            runs_csv=Path(args.runs) if args.runs is not None else None,
        )
    if cmd == "all":
        term_out = out / "termination"
        ex_out = out / "exemplars"
        div_out = out / "diversity"
        kin_out = out / "kinematics"
        rank_out = out / "rank"
        group_by = _parse_group_by(args.group_by)
        run_termination(results, term_out, group_by=group_by)
        run_exemplars(
            results,
            ex_out,
            per_class=int(args.per_class),
            classes=_parse_classes(args.classes),
            include_not_triggered=not bool(args.skip_not_triggered),
        )
        run_diversity(results, div_out, group_by=group_by)
        run_kinematics(results, kin_out, group_by=group_by)
        return run_rank(
            results,
            rank_out,
            metrics=DEFAULT_METRICS,
            holdout_suites=(),
            runs_csv=None,
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
