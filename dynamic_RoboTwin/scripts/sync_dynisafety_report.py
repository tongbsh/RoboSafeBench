#!/usr/bin/env python3
"""Copy only _dynisafety_summary.txt into outputs/dynaSafty_report for reporting.

- L1/L2 from outputs/dynisafety/<task_config>/...
- L0/L3 from outputs/dynisafety_no_topple/<task_config>/...

Preserves layout required by aggregate_logs.py:
  <outputs_root>/<task_config>/<task>/<model>/levelN/_dynisafety_summary.txt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SUMMARY_NAME = "_dynisafety_summary.txt"


def sync_task_config(
    dst_root: Path,
    src_dynisafety: Path,
    src_no_topple: Path,
    task_config: str,
) -> tuple[int, list[tuple[str, str, int]]]:
    """Copy summaries for one task_config name. Returns (copied_count, missing list)."""
    copied = 0
    missing: list[tuple[str, str, int]] = []

    # L1, L2 from dynisafety
    base_a = src_dynisafety / task_config
    if base_a.is_dir():
        for level in (1, 2):
            level_glob = f"level{level}"
            for summary in base_a.rglob(f"{level_glob}/{SUMMARY_NAME}"):
                rel = summary.relative_to(base_a)
                # rel = task/model/levelN/_dynisafety_summary.txt
                parts = rel.parts
                if len(parts) != 4 or parts[-1] != SUMMARY_NAME:
                    continue
                dst = dst_root / task_config / Path(*parts[:-1]) / SUMMARY_NAME
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary, dst)
                copied += 1
    else:
        print(f"[warn] missing source dir: {base_a}")

    # L0, L3 from no_topple
    base_b = src_no_topple / task_config
    if base_b.is_dir():
        for level in (0, 3):
            level_glob = f"level{level}"
            for summary in base_b.rglob(f"{level_glob}/{SUMMARY_NAME}"):
                rel = summary.relative_to(base_b)
                parts = rel.parts
                if len(parts) != 4 or parts[-1] != SUMMARY_NAME:
                    continue
                dst = dst_root / task_config / Path(*parts[:-1]) / SUMMARY_NAME
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary, dst)
                copied += 1
    else:
        print(f"[warn] missing source dir: {base_b}")

    # Report missing (task, model, level) for expected grid — optional scan
    for base, levels in [(base_a, (1, 2)), (base_b, (0, 3))]:
        if not base.is_dir():
            continue
        tasks = [d for d in base.iterdir() if d.is_dir()]
        for task_dir in tasks:
            task = task_dir.name
            for model_dir in [d for d in task_dir.iterdir() if d.is_dir()]:
                model = model_dir.name
                for lev in levels:
                    p = model_dir / f"level{lev}" / SUMMARY_NAME
                    if not p.is_file():
                        missing.append((task, model, lev))

    return copied, missing


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_scripts = script_dir
    outputs = repo_scripts.parent / "outputs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dst",
        type=Path,
        default=outputs / "dynaSafty_report",
        help="Destination outputs root (default: .../outputs/dynaSafty_report)",
    )
    parser.add_argument(
        "--src-dynisafety",
        type=Path,
        default=outputs / "dynisafety",
        help="Root containing dynisafety task_config dirs",
    )
    parser.add_argument(
        "--src-no-topple",
        type=Path,
        default=outputs / "dynisafety_no_topple",
        help="Root containing dynisafety_no_topple task_config dirs",
    )
    parser.add_argument(
        "--task-configs",
        nargs="*",
        default=None,
        help="Task config folder names under each src (default: all common dirs)",
    )
    args = parser.parse_args()

    dst_root = args.dst.resolve()
    src_a = args.src_dynisafety.resolve()
    src_b = args.src_no_topple.resolve()

    if args.task_configs:
        task_configs = args.task_configs
    else:
        a = {p.name for p in src_a.iterdir() if p.is_dir() and not p.name.startswith("_")}
        b = {p.name for p in src_b.iterdir() if p.is_dir() and not p.name.startswith("_")}
        task_configs = sorted(a & b)
        if not task_configs:
            print("[error] no shared task_config dirs between sources")
            return

    total = 0
    for tc in task_configs:
        n, miss = sync_task_config(dst_root, src_a, src_b, tc)
        total += n
        print(f"[info] task_config={tc!r}: copied {n} summary file(s)")
        if miss:
            sample = miss[:20]
            print(f"[warn] task_config={tc!r}: {len(miss)} missing summary cells (showing up to 20):")
            for t, m, lev in sample:
                print(f"  - {t}/{m}/level{lev}")

    print(f"[info] total copied: {total} -> {dst_root}")


if __name__ == "__main__":
    main()
