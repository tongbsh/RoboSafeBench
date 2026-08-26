#!/usr/bin/env python3
"""Scan X-VLA walkthrough runs: metrics, disturbance extras, videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from dynisafety.config.paths import load_paths
from dynisafety.config.resolve import find_workspace_root
from dynisafety.runner.scheduler import (
    JobSpec,
    expand_jobs,
    is_job_complete,
    load_sweep_suite,
    resolve_suite_path,
)

_HARD = "hard"
_WARN = "warn"


def _parse_args() -> argparse.Namespace:
    root = find_workspace_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite",
        default=None,
        help="Sweep suite name or YAML path (default: debug_xvla_dynamic / suite.used.yaml)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Expect paper episode counts (debug_xvla_dynamic_full)",
    )
    p.add_argument("--results", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _resolve_suite_arg(root: Path, raw: str | None, *, full: bool) -> Path:
    if raw:
        return resolve_suite_path(raw, configs_root=root / "configs")
    name = "debug_xvla_dynamic_full" if full else "debug_xvla_dynamic"
    used = load_paths(workspace=root).sweep_dir(name) / "suite.used.yaml"
    if used.is_file():
        return used
    legacy = root / "results" / "_sweep" / name / "suite.used.yaml"
    if legacy.is_file():
        return legacy
    return resolve_suite_path(name if full else "debug_xvla_dynamic", configs_root=root / "configs")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _latest_run_dir(results_root: Path, job: JobSpec) -> Path | None:
    prefix = f"{job.backend}-{job.model}-{job.suite}-t{job.task_id}-l{job.level}-s{job.seed}-"
    search: list[Path] = []
    try:
        search.append(
            load_paths()
            .with_out(results_root)
            .eval_parent(backend=job.backend, suite=job.suite, level=int(job.level))
        )
    except (OSError, ValueError, KeyError):
        pass
    search.append(results_root / f"l{int(job.level)}")
    cands: list[Path] = []
    for level_dir in search:
        if not level_dir.is_dir():
            continue
        cands.extend(
            path
            for path in level_dir.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        )
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _disturbance_flags(extra: dict[str, Any]) -> dict[str, bool]:
    spawn = extra.get("spawn_plan")
    l3 = extra.get("l3_plan")
    rt = extra.get("robotwin_disturbance")
    return {
        "active_obstacle": extra.get("active_obstacle") not in (None, "", []),
        "spawn_plan": isinstance(spawn, dict) and bool(spawn),
        "l3_plan": isinstance(l3, dict) and bool(l3),
        "robotwin_disturbance": isinstance(rt, dict) and bool(rt),
        "l1_onset": bool(
            extra.get("l1_workspace_onset")
            or extra.get("l1_home_onset")
            or extra.get("l1_home_inside")
        ),
    }


def _disturbance_warning(level: int, flags: dict[str, bool]) -> str | None:
    if level == 1:
        if not (
            flags["active_obstacle"]
            or flags["l1_onset"]
            or flags["robotwin_disturbance"]
            or flags["spawn_plan"]
        ):
            return "L1 未见投放/onset/disturbance 痕迹（机器人可能没进工作区，或动态没挂上）"
        return None
    if level == 2 and not flags["spawn_plan"]:
        return "L2 缺少 extra.spawn_plan"
    if level == 3 and not flags["l3_plan"]:
        return "L3 缺少 extra.l3_plan"
    return None


def _video_ok(rec: dict[str, Any]) -> tuple[bool, str]:
    raw = rec.get("video_path")
    if not raw:
        return False, "video_path 为空"
    path = Path(str(raw))
    if path.is_file() and path.stat().st_size > 0:
        return True, str(path)
    return False, f"视频缺失或空文件: {path}"


def main() -> int:
    args = _parse_args()
    root = find_workspace_root()
    paths = load_paths(workspace=root)
    env_results = os.environ.get("DYNI_RESULTS", "").strip()
    results_root = (
        Path(args.results)
        if args.results
        else Path(env_results)
        if env_results
        else paths.out
    )
    out_dir = (
        Path(args.out)
        if args.out
        else paths.logs_dir("debug", "xvla_walkthrough")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_path = _resolve_suite_arg(root, args.suite, full=bool(args.full))
    suite = load_sweep_suite(suite_path)
    jobs = expand_jobs(suite, configs_root=root / "configs")

    anomalies: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    videos: list[str] = []
    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {
            "n": 0,
            "success": 0,
            "contact": 0,
            "error": 0,
            "no_video": 0,
            "dyn_warn": 0,
            "steps": 0.0,
        }
    )

    missing_jobs = 0
    incomplete_jobs = 0

    for job in jobs:
        run_dir = _latest_run_dir(results_root, job)
        complete = is_job_complete(results_root, job)
        if run_dir is None:
            missing_jobs += 1
            anomalies.append(
                {
                    "severity": _HARD,
                    "job_key": job.job_key,
                    "backend": job.backend,
                    "suite": job.suite,
                    "task_id": job.task_id,
                    "level": job.level,
                    "model": job.model,
                    "msg": "没有对应 run 目录",
                }
            )
            continue
        if not complete:
            incomplete_jobs += 1
            anomalies.append(
                {
                    "severity": _HARD,
                    "job_key": job.job_key,
                    "run_dir": str(run_dir),
                    "backend": job.backend,
                    "suite": job.suite,
                    "task_id": job.task_id,
                    "level": job.level,
                    "model": job.model,
                    "msg": "summary 未 completed 或 episode 数不够",
                }
            )

        records = _read_jsonl(run_dir / "episodes.jsonl")
        if not records:
            anomalies.append(
                {
                    "severity": _HARD,
                    "job_key": job.job_key,
                    "run_dir": str(run_dir),
                    "backend": job.backend,
                    "msg": "episodes.jsonl 为空",
                }
            )
            continue

        for rec in records:
            extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
            flags = _disturbance_flags(extra)
            dyn_msg = _disturbance_warning(int(rec.get("level", job.level)), flags)
            video_ok, video_note = _video_ok(rec)
            term = str(rec.get("termination") or "")
            steps = int(rec.get("steps") or 0)
            success = bool(rec.get("success"))
            contact_steps = int(rec.get("contact_steps") or 0)
            err = extra.get("error")

            hard_msgs: list[str] = []
            if term == "error" or err:
                hard_msgs.append(f"termination/error: {term or err}")
            if steps <= 0:
                hard_msgs.append("steps=0")
            if not video_ok:
                hard_msgs.append(video_note)

            key = (str(rec.get("backend", job.backend)), int(rec.get("level", job.level)))
            g = grouped[key]
            g["n"] += 1
            g["success"] += int(success)
            g["contact"] += int(contact_steps > 0)
            g["error"] += int(bool(hard_msgs and ("termination" in hard_msgs[0] or term == "error")))
            g["no_video"] += int(not video_ok)
            g["dyn_warn"] += int(dyn_msg is not None)
            g["steps"] += float(steps)

            if video_ok:
                videos.append(video_note)

            row = {
                "backend": rec.get("backend", job.backend),
                "suite": rec.get("suite", job.suite),
                "task_id": rec.get("task_id", job.task_id),
                "task_name": rec.get("task_name", job.task_name),
                "level": rec.get("level", job.level),
                "model": rec.get("model", job.model),
                "episode_index": rec.get("episode_index"),
                "success": success,
                "termination": term,
                "steps": steps,
                "contact_steps": contact_steps,
                "contact_events": rec.get("contact_events"),
                "peak_force_n": rec.get("peak_force_n"),
                "toppled": rec.get("toppled"),
                "video_ok": video_ok,
                "video_path": rec.get("video_path"),
                "dyn_ok": dyn_msg is None,
                "dyn_note": dyn_msg or "",
                "run_dir": str(run_dir),
            }
            episode_rows.append(row)

            ident = {
                "job_key": job.job_key,
                "run_dir": str(run_dir),
                "backend": row["backend"],
                "suite": row["suite"],
                "task_id": row["task_id"],
                "level": row["level"],
                "episode_index": row["episode_index"],
            }
            for msg in hard_msgs:
                anomalies.append({**ident, "severity": _HARD, "msg": msg})
            if dyn_msg:
                anomalies.append({**ident, "severity": _WARN, "msg": dyn_msg})

    csv_path = out_dir / "episodes.csv"
    if episode_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(episode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(episode_rows)

    videos_path = out_dir / "videos.txt"
    videos_path.write_text("\n".join(videos) + ("\n" if videos else ""), encoding="utf-8")

    anom_path = out_dir / "anomalies.jsonl"
    with anom_path.open("w", encoding="utf-8") as fp:
        for row in anomalies:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    hard_n = sum(1 for a in anomalies if a["severity"] == _HARD)
    warn_n = sum(1 for a in anomalies if a["severity"] == _WARN)

    lines: list[str] = [
        f"# X-VLA walkthrough check",
        "",
        f"- suite: `{suite.name}`  (`{suite_path}`)",
        f"- expected jobs: **{len(jobs)}**",
        f"- missing run dirs: **{missing_jobs}**",
        f"- incomplete jobs: **{incomplete_jobs}**",
        f"- episodes scanned: **{len(episode_rows)}**",
        f"- videos found: **{len(videos)}**",
        f"- hard anomalies: **{hard_n}**",
        f"- dynamic warnings: **{warn_n}**",
        "",
        "## backend × level",
        "",
        "| backend | level | n | SR | contact_rate | mean_steps | errors | no_video | dyn_warn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend, level in sorted(grouped):
        g = grouped[(backend, level)]
        n = max(int(g["n"]), 1)
        lines.append(
            "| {backend} | {level} | {n} | {sr:.2f} | {cr:.2f} | {ms:.1f} | {er} | {nv} | {dw} |".format(
                backend=backend,
                level=level,
                n=int(g["n"]),
                sr=g["success"] / n,
                cr=g["contact"] / n,
                ms=g["steps"] / n,
                er=int(g["error"]),
                nv=int(g["no_video"]),
                dw=int(g["dyn_warn"]),
            )
        )

    lines += [
        "",
        "## 怎么看",
        "",
        "- hard：`termination=error`、`steps=0`、缺视频、job 没写完 → 流程或接线有问题。",
        "- dyn_warn：L2 应有 `spawn_plan`，L3 应有 `l3_plan`；L1 没 traces 常见于机器人没进投放区。",
        "- SR 低不一定异常（安全扰动下任务失败是预期）；**缺视频 / error / 动态字段全空**才优先查。",
        f"- 明细：`{csv_path}`",
        f"- 视频清单：`{videos_path}`",
        f"- 异常：`{anom_path}`",
    ]
    md_path = out_dir / "check.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[ok] wrote {md_path}")
    return 1 if hard_n else 0


if __name__ == "__main__":
    raise SystemExit(main())
