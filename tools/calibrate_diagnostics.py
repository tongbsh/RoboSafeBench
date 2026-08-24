#!/usr/bin/env python3
"""P5 offline calibration for behavior diagnostics.

Reads traces already written by a sweep. Recomputes ``analyze_trace`` in
memory under a threshold grid. Never launches a simulator and never writes
back into the results tree.

    python tools/calibrate_diagnostics.py --results <results 根> --out <输出目录> \
        [--target-fpr 0.05] [--n-virtual-onsets 5] [--seed 7]

Outputs: ``fpr_curve.csv``, ``sensitivity.csv``, ``baseline_health.csv``,
``recommendation.md``, ``plots/fpr_curve.png``, ``plots/sensitivity.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dynisafety.diagnostics.calibrate import (  # noqa: E402
    CalibrationError,
    run_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate behavior-diagnostics thresholds from saved traces",
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Results root containing run dirs with traces/ and episodes.jsonl",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory for CSV / markdown / plots (created if missing)",
    )
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--n-virtual-onsets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--base-yaml",
        type=Path,
        default=None,
        help="Override configs/base.yaml (must contain a complete diagnostics: block)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_calibration(
            results=args.results,
            out=args.out,
            target_fpr=float(args.target_fpr),
            n_virtual_onsets=int(args.n_virtual_onsets),
            seed=int(args.seed),
            base_yaml=args.base_yaml,
        )
    except CalibrationError as exc:
        print(f"[calibrate] ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"[calibrate] L0={result.n_l0} L1–L3={result.n_l1_l3} onset_pool={result.n_onset_pool}")
    print(f"[calibrate] traces unmodified: {result.traces_unmodified}")
    print(f"[calibrate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
