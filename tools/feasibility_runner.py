#!/usr/bin/env python3
"""Feasibility runner: Base vs Oracle-SWR vs Oracle-SWR+AEGIS (Prompt 14 / A2).

Default mode is dry-run / analyze-fixtures. Formal GPU eval requires
``--authorize-gpu`` and is otherwise BLOCKED_UNTIL_AUTHORIZED.

Usage:
  python tools/feasibility_runner.py --matrix configs/feasibility/libero_openvla_oft_small.yaml --dry-run
  python tools/feasibility_runner.py --matrix configs/feasibility/libero_openvla_oft_aegis_small.yaml \\
      --dry-run --out artifacts/feasibility/aegis_dry_run
  python tools/feasibility_runner.py --analyze tests/fixtures/feasibility/episodes_fixture.jsonl \\
      --matrix configs/feasibility/libero_openvla_oft_small.yaml \\
      --out artifacts/feasibility/fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from dynisafety.feasibility.aggregate import (  # noqa: E402
    aggregate_episodes_to_cells,
    build_base_vs_oracle_table,
    build_feasibility_matrix,
    load_episode_dicts_from_jsonl,
)
from dynisafety.feasibility.estimate import (  # noqa: E402
    estimate_matrix_resources,
    expand_matrix_jobs,
    job_to_eval_command,
    load_feasibility_matrix,
    thresholds_from_matrix,
)
from dynisafety.feasibility.report import write_feasibility_artifacts  # noqa: E402


def _cmd_dry_run(matrix_path: Path, out_dir: Path) -> int:
    data = load_feasibility_matrix(matrix_path)
    jobs = expand_matrix_jobs(data)
    est_cfg = data.get("estimate") or {}
    estimate = estimate_matrix_resources(
        jobs,
        seconds_per_episode_low=float(est_cfg.get("seconds_per_episode_low", 45)),
        seconds_per_episode_high=float(est_cfg.get("seconds_per_episode_high", 120)),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "BLOCKED_UNTIL_AUTHORIZED",
        "matrix": str(matrix_path),
        "estimate": estimate.as_dict(),
        "jobs": [
            {
                "backend": j.backend,
                "model": j.model,
                "suite": j.suite,
                "task_id": j.task_id,
                "level": j.level,
                "condition": j.condition,
                "episodes": j.episodes,
                "seed": j.seed,
                "command": job_to_eval_command(j),
            }
            for j in jobs
        ],
        "narrative": [
            "Oracle = dynamic-phase pause; AEGIS = settle-time static-obstacle filter.",
            "Oracle CR expected to drop vs Base (do not invent numbers).",
            "Track oracle_pause_steps + aegis_intervened_steps; aegis off must not alter base/oracle.",
            "L1/L2 residual low SSR after settle → static obstacle / needs avoidance layer.",
            "L3 optional proof cell keeps aegis off after clear-scene withdraw.",
        ],
    }
    plan_path = out_dir / "dry_run_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    cmds_path = out_dir / "eval_commands.sh"
    cmds_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "# BLOCKED_UNTIL_AUTHORIZED — do not run without GPU authorization.",
                "set -euo pipefail",
                "",
                *[job_to_eval_command(j) for j in jobs],
                "",
            ]
        ),
        encoding="utf-8",
    )
    # Empty artifact report shell for consistency
    write_feasibility_artifacts(
        out_dir=out_dir,
        cells=[],
        compare_rows=[],
        matrix_rows=[],
        estimate=estimate,
        status="BLOCKED_UNTIL_AUTHORIZED",
    )
    print(yaml.safe_dump(
        {
            "status": "BLOCKED_UNTIL_AUTHORIZED",
            "n_jobs": estimate.n_jobs,
            "n_episodes_total": estimate.n_episodes_total,
            "gpu_hours": f"{estimate.gpu_hours_low}-{estimate.gpu_hours_high}",
            "plan": str(plan_path),
            "commands": str(cmds_path),
        },
        sort_keys=False,
    ))
    return 0


def _cmd_analyze(matrix_path: Path, episode_paths: list[Path], out_dir: Path) -> int:
    data = load_feasibility_matrix(matrix_path)
    thresholds = thresholds_from_matrix(data)
    episodes = load_episode_dicts_from_jsonl(episode_paths)
    cells = aggregate_episodes_to_cells(episodes, thresholds=thresholds)
    compare = build_base_vs_oracle_table(cells)
    matrix_rows = build_feasibility_matrix(cells, thresholds=thresholds)
    jobs = expand_matrix_jobs(data)
    est_cfg = data.get("estimate") or {}
    estimate = estimate_matrix_resources(
        jobs,
        seconds_per_episode_low=float(est_cfg.get("seconds_per_episode_low", 45)),
        seconds_per_episode_high=float(est_cfg.get("seconds_per_episode_high", 120)),
    )
    paths = write_feasibility_artifacts(
        out_dir=out_dir,
        cells=cells,
        compare_rows=compare,
        matrix_rows=matrix_rows,
        estimate=estimate,
        status="fixture_or_authorized_results",
    )
    print(yaml.safe_dump({k: str(v) for k, v in paths.items()}, sort_keys=False))
    return 0


def _cmd_run_authorized(matrix_path: Path, out_dir: Path) -> int:
    """Print reminder — actual eval still delegated to dynisafety eval commands."""
    print(
        "BLOCKED_UNTIL_AUTHORIZED: refusing to launch GPU eval from this tool.\n"
        "Re-run with --dry-run to emit commands, then execute under authorization:\n"
        f"  bash {out_dir / 'eval_commands.sh'}",
        file=sys.stderr,
    )
    return _cmd_dry_run(matrix_path, out_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--matrix",
        type=Path,
        default=REPO / "configs/feasibility/libero_openvla_oft_aegis_small.yaml",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "artifacts/feasibility/dry_run",
    )
    ap.add_argument("--dry-run", action="store_true", help="Expand jobs + estimate only")
    ap.add_argument(
        "--analyze",
        type=Path,
        nargs="+",
        default=None,
        help="Episode JSONL path(s) (fixture or real results)",
    )
    ap.add_argument(
        "--authorize-gpu",
        action="store_true",
        help="Explicit GPU authorization flag (still does not auto-launch long jobs)",
    )
    args = ap.parse_args(argv)

    if args.analyze:
        return _cmd_analyze(args.matrix, list(args.analyze), args.out)
    if args.dry_run or not args.authorize_gpu:
        if not args.dry_run and not args.authorize_gpu:
            print(
                "# defaulting to --dry-run (pass --authorize-gpu only after human approval)",
                file=sys.stderr,
            )
        return _cmd_dry_run(args.matrix, args.out)
    return _cmd_run_authorized(args.matrix, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
