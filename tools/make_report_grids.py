#!/usr/bin/env python3
"""Generate coverage grids for ``dynisafety report`` from a sweep suite.

``report generate --grid`` needs an explicit expected (model × task × level ×
seed × n_episodes) grid. Deriving it from the sweep suite keeps a single source
of truth: the same YAML that drives the runs also defines what "complete" means.

Writes one grid per suite matrix entry, since backends differ in tasks, models
and episode counts.

Usage:
    python tools/make_report_grids.py                      # from paper_main
    python tools/make_report_grids.py --suite smoke
    python tools/make_report_grids.py --print              # stdout, do not write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from dynisafety.backends.registry import populate_backends  # noqa: E402
from dynisafety.core.registry import Registry  # noqa: E402

SUITES = REPO / "configs" / "suites"
OUT_DIR = REPO / "configs" / "report" / "grids"


def backend_registry() -> Registry:
    reg = Registry("backend")
    populate_backends(reg)
    return reg


def tasks_for(reg: Registry, backend: str, suite: str) -> list[dict]:
    specs = reg.get(backend)().list_tasks(suite)
    return [{"task_id": int(t.task_id), "task_name": str(t.task_name)} for t in specs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="paper_main", help="sweep suite under configs/suites/")
    ap.add_argument("--print", dest="to_stdout", action="store_true")
    args = ap.parse_args()

    suite_path = SUITES / f"{args.suite}.yaml"
    if not suite_path.exists():
        print(f"no such suite: {suite_path}", file=sys.stderr)
        return 2
    spec = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    default_eps = int(spec["episodes_per_task"])
    default_seeds = [int(s) for s in spec["seeds"]]

    reg = backend_registry()
    written: list[Path] = []

    for entry in spec.get("matrix") or []:
        backend = str(entry["backend"])
        models = [str(m) for m in entry["models"]]
        levels = [int(x) for x in entry["levels"]]
        seeds = [int(s) for s in entry.get("seeds", default_seeds)]
        n_eps = int(entry.get("episodes_per_task", default_eps))

        for suite in [str(s) for s in entry["suites"]]:
            explicit = entry.get("tasks")
            all_tasks = tasks_for(reg, backend, suite)
            if explicit is not None:
                keep = {int(t) for t in explicit}
                tasks = [t for t in all_tasks if t["task_id"] in keep]
                found = {t["task_id"] for t in tasks}
                if missing := keep - found:
                    print(
                        f"warning: {backend}/{suite} has no task_id {sorted(missing)}",
                        file=sys.stderr,
                    )
            else:
                tasks = all_tasks

            grid = {
                "models": models,
                "tasks": tasks,
                "levels": levels,
                "seeds": seeds,
                "n_episodes": n_eps,
            }
            header = (
                f"# Coverage grid for `dynisafety report` — GENERATED, do not hand-edit.\n"
                f"# Source: configs/suites/{args.suite}.yaml (backend={backend}, suite={suite})\n"
                f"# Regenerate: python tools/make_report_grids.py --suite {args.suite}\n"
                f"# Expected rows: {len(models)} models x {len(tasks)} tasks "
                f"x {len(levels)} levels x {len(seeds)} seeds = "
                f"{len(models) * len(tasks) * len(levels) * len(seeds)} runs "
                f"x {n_eps} episodes\n"
            )
            body = yaml.safe_dump(grid, sort_keys=False, allow_unicode=True)
            if args.to_stdout:
                print(f"--- {backend}_{suite}.yaml")
                print(header + body)
                continue
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUT_DIR / f"{backend}_{suite}.yaml"
            path.write_text(header + body, encoding="utf-8")
            written.append(path)

    for path in written:
        print(f"wrote {path.relative_to(REPO)}")
    if written:
        print(f"\n{len(written)} grid(s). Use with:")
        print(
            "  dynisafety report generate --results results/ --out reports/ \\\n"
            f"      --grid configs/report/grids/{written[0].name} \\\n"
            "      --report-yaml configs/report/report_settings.yaml \\\n"
            "      --metrics-yaml configs/base.yaml"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
