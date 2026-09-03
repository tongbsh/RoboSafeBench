"""CLI for ICLR paper figures. Does not participate in ``report generate``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dynisafety.paper_figures.catalog import FIGURES
from dynisafety.paper_figures.data import load_bundle, load_display, load_peers, load_settings
from dynisafety.paper_figures.render import render_figures


def add_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results", type=Path, default=None, help="Eval root with episodes.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output directory (not inside --results)")
    parser.add_argument("--suite-yaml", type=Path, required=True, help="Usually configs/suites/paper_main.yaml")
    parser.add_argument("--settings-yaml", type=Path, required=True)
    parser.add_argument("--peers-yaml", type=Path, required=True)
    parser.add_argument("--display-yaml", type=Path, required=True)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=None,
        help=(
            "Root of the stage-03 analysis tree holding "
            "<backend>/paired_axes/paired_axes_episodes.csv "
            "(default: paired_axes.analysis_root from --settings-yaml). "
            "Run scripts/eval/03_analysis.sh paired-axes first; missing CSVs render n/a."
        ),
    )
    parser.add_argument(
        "--figures",
        default=None,
        help="Comma-separated fig ids or stems (default: all)",
    )
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help="Only figures that do not need eval JSONL (scale + framework + montages)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynisafety paper-figures",
        description=(
            "ICLR paper figures for RoboSafeBench. Reads eval JSONL read-only. "
            "Writes PDF/PNG under --out. Does not change report generate."
        ),
    )
    add_parser(parser)
    return parser


def _refuse_out_inside_results(results: Path | None, out: Path) -> None:
    if results is None:
        return
    try:
        out.resolve().relative_to(results.resolve())
    except ValueError:
        return
    raise ValueError(
        f"--out {out} resolves inside --results {results}. "
        "Write paper figures to a separate directory."
    )


def _parse_figures(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--figures given but empty")
    return values


def dispatch(args: argparse.Namespace) -> int:
    results = Path(args.results) if args.results is not None else None
    out = Path(args.out)
    _refuse_out_inside_results(results, out)
    if results is None and not bool(args.overview_only):
        print("paper-figures: --results is required unless --overview-only", file=sys.stderr)
        return 2
    if results is not None and not results.is_dir():
        print(f"paper-figures: results root not found: {results}", file=sys.stderr)
        return 2
    settings = load_settings(Path(args.settings_yaml))
    display = load_display(Path(args.display_yaml))
    peers = load_peers(Path(args.peers_yaml))
    bundle = load_bundle(
        results=results,
        suite_yaml=Path(args.suite_yaml),
        settings=settings,
        display=display,
        peers=peers,
        analysis_root=Path(args.analysis_root) if args.analysis_root is not None else None,
    )
    written = render_figures(
        bundle,
        out,
        figures=_parse_figures(args.figures),
        overview_only=bool(args.overview_only),
    )
    print(f"wrote {len(written)} figure groups under {out}")
    print(
        f"planned jobs: {bundle.n_planned_jobs}; "
        f"completed cells: {bundle.n_completed_jobs}; "
        f"{bundle.stamp()}"
    )
    print(
        f"paired axes: {len(bundle.paired.episodes)} episode rows "
        f"from {len(bundle.paired.sources)} backend CSVs"
    )
    for note in bundle.paired.missing:
        print(f"paired axes n/a (rendered as n/a, not 0): {note}", file=sys.stderr)
    print(f"catalog: {out / 'catalog.md'}")
    return 0


def register_paper_figures_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "paper-figures",
        help="ICLR paper figures (RoboSafeBench; does not change report)",
    )
    add_parser(parser)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except Exception as exc:
        print(f"paper-figures failed: {exc}", file=sys.stderr)
        return 1


# Keep FIGURES imported so ``dynisafety paper-figures --help`` stays next to the catalog.
_ = FIGURES
