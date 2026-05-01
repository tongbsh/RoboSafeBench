#!/usr/bin/env python3
"""Compliance scan: ``pruned_init`` vs BDDL declared regions.

For every (suite, task, episode) we:

1. Parse the BDDL file's ``(:init (On <obj> <fixture>_<region>))`` declarations
   to learn which named region each object is supposed to start in, and parse
   the ``(:regions ... :ranges ((xmin ymin xmax ymax)))`` blocks to learn the
   xy bounds of each region.
2. Load the suite's pruned_init, pad it for the hand DoFs, apply it inside
   the actual ``OffScreenRenderEnv`` used by the runner, and read every task
   object's actual world-frame xy directly from MuJoCo.
3. Flag any object whose actual xy falls **outside** its declared region's
   axis-aligned bounding box (with a small tolerance for floating-point /
   collision settle effects).

Outputs:
  * Per-task summary printed to stdout
  * CSV at ``dynisafety_libero/outputs/_aggregated/pruned_init_compliance.csv``
    (one row per <suite, task, episode, object>) so you can grep / pivot

Usage::

    conda activate openvla-oft  # or any env with libero importable via the project bootstrap
    python dynisafety_libero/scripts/diagnose_pruned_init.py
    python dynisafety_libero/scripts/diagnose_pruned_init.py --suites safelibero_object
    python dynisafety_libero/scripts/diagnose_pruned_init.py --episodes 5 --tolerance 0.01
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project bootstrap — mirror the path setup done by run_dynisafety_eval.py
# so we get the dyniSafety copy of libero (with safelibero_* benchmarks).
# ---------------------------------------------------------------------------
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent  # …/dynamic_safty_bench
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "simulator_env" / "dyniSafety_libero"))
os.environ.setdefault(
    "LIBERO_DATA_ROOT",
    str(_PROJ_ROOT / "simulator_env" / "dyniSafety_libero" / "libero" / "libero"),
)

import numpy as np  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from dynisafety_libero.env_utils import pad_init_state_for_hand  # noqa: E402

ALL_SUITES = (
    "safelibero_spatial",
    "safelibero_object",
    "safelibero_goal",
    "safelibero_long",
)

# Anything matching these substrings is a moving disturbance (not part of the
# task scene we evaluate at L0) — skip during compliance check.
OBSTACLE_NAME_PATTERNS = ("obstacle", "hand_pusher", "box_base", "box_small_base")


# ---------------------------------------------------------------------------
# BDDL parser (minimal, regex-based — sufficient for the structure used by
# safelibero BDDLs).
# ---------------------------------------------------------------------------
# Matches a region declaration header ``(<name>_region`` and a ``:ranges`` block
# that contains a 4-tuple of floats. We use paren-balancing for the surrounding
# region body so we don't trip over the ``(:target floor)`` sub-expression.
_RE_REGION_HEADER = re.compile(r"\(\s*([A-Za-z_][\w]*_region)\b")
_RE_RANGES_TUPLE = re.compile(
    r":ranges\s*\(\s*\(\s*([\-\d.eE+\s]+?)\s*\)",
)
_RE_ON_LINE = re.compile(r"\(\s*On\s+([\w]+)\s+([\w]+)\s*\)")


def _extract_balanced_block(text: str, opening: str) -> Optional[str]:
    """Return the *contents* (no surrounding parens) of the first
    ``(opening ...)`` s-expression, using paren counting."""
    idx = text.find(opening)
    if idx < 0:
        return None
    start = idx + len(opening)
    depth = 1
    for i in range(start, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


@dataclass
class RegionBox:
    """Axis-aligned xy bounding box declared in a BDDL ``:regions`` block."""

    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, x: float, y: float, tol: float = 0.0) -> bool:
        return (
            self.x_min - tol <= x <= self.x_max + tol
            and self.y_min - tol <= y <= self.y_max + tol
        )

    def __str__(self) -> str:
        return (
            f"x∈({self.x_min:+.3f}, {self.x_max:+.3f}) "
            f"y∈({self.y_min:+.3f}, {self.y_max:+.3f})"
        )


def parse_bddl(bddl_path: str) -> Tuple[Dict[str, RegionBox], Dict[str, str], set]:
    """Return ``(regions, obj_to_region_basename, obj_of_interest)``."""
    text = Path(bddl_path).read_text(encoding="utf-8")

    # --- regions: walk every (NAME_region ... ) sexpr via paren counting ---
    regions: Dict[str, RegionBox] = {}
    for hm in _RE_REGION_HEADER.finditer(text):
        name = hm.group(1)
        # Find the matching ``)`` for this region's opening ``(``.
        i = hm.start()
        depth = 0
        end = -1
        for j in range(i, len(text)):
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end < 0:
            continue
        body = text[i:end + 1]
        rm = _RE_RANGES_TUPLE.search(body)
        if rm is None:
            continue
        nums = [float(tok) for tok in rm.group(1).split()]
        if len(nums) != 4:
            continue
        x_min, y_min, x_max, y_max = nums
        regions[name] = RegionBox(
            name=name,
            x_min=min(x_min, x_max), x_max=max(x_min, x_max),
            y_min=min(y_min, y_max), y_max=max(y_min, y_max),
        )

    obj_to_region: Dict[str, str] = {}
    init_block = _extract_balanced_block(text, "(:init")
    if init_block is not None:
        for m in _RE_ON_LINE.finditer(init_block):
            obj, fixture_region = m.group(1), m.group(2)
            # ``floor_target_object_region`` → strip leading fixture token.
            # We try the longest matching region name first.
            for region_name in sorted(regions.keys(), key=len, reverse=True):
                if fixture_region.endswith(region_name):
                    obj_to_region[obj] = region_name
                    break

    # --- objects of interest (the ones the policy must actually manipulate) --
    obj_of_interest: set = set()
    ooi_block = _extract_balanced_block(text, "(:obj_of_interest")
    if ooi_block is not None:
        for tok in ooi_block.split():
            tok = tok.strip()
            if tok and tok.replace("_", "").replace("0", "").replace("1", "").replace("2", "").isalpha() or tok.endswith("_1"):
                obj_of_interest.add(tok)

    return regions, obj_to_region, obj_of_interest


# ---------------------------------------------------------------------------
# Sim helpers
# ---------------------------------------------------------------------------
def get_object_xy_world(env, obj_name: str) -> Optional[Tuple[float, float, float]]:
    """Return ``(x, y, z)`` of *obj_name*'s free joint qpos in world frame."""
    obj = env.env.objects_dict.get(obj_name)
    if obj is None:
        return None
    sim = env.env.sim
    try:
        joint_name = obj.joints[-1]
        qpos_addr = sim.model.jnt_qposadr[sim.model.joint_name2id(joint_name)]
        return tuple(float(v) for v in sim.data.qpos[qpos_addr:qpos_addr + 3])
    except Exception:
        return None


def is_obstacle(name: str) -> bool:
    nl = name.lower()
    return any(p in nl for p in OBSTACLE_NAME_PATTERNS)


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    suite: str
    task_id: int
    task_name: str
    episode: int
    object: str
    region: str
    declared_box: str
    actual_xyz: Tuple[float, float, float]
    inside: bool
    delta_xy: float            # |actual - region center|
    is_obj_of_interest: bool   # True ⇒ a real bug (target object misplaced)
    is_hidden: bool            # True ⇒ teleported far away (|x|>5 or |y|>5)


def scan_suite(
    suite_name: str,
    n_episodes: int,
    tol: float,
    safety_level: str,
    verbose: bool,
) -> List[CheckResult]:
    bdict = benchmark.get_benchmark_dict()
    suite_cls = bdict.get(suite_name)
    if suite_cls is None:
        print(f"[warn] suite '{suite_name}' not registered, skipping.")
        return []

    suite = suite_cls(safety_level=safety_level)
    bddl_root = get_libero_path("bddl_files")
    results: List[CheckResult] = []

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        bddl_path = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        regions, obj_to_region, obj_of_interest = parse_bddl(bddl_path)

        try:
            init_states = suite.get_task_init_states(task_id)
        except Exception as exc:
            print(f"[warn] {suite_name} task {task_id}: cannot load init states: {exc}")
            continue

        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=64, camera_widths=64)
        env.seed(0)

        n_eps_avail = len(init_states)
        n_eps_run = min(n_episodes, n_eps_avail)

        if verbose:
            print(f"\n--- {suite_name} task {task_id}: {task.name}")
            print(f"    bddl    : {os.path.basename(bddl_path)}")
            print(f"    episodes      : {n_eps_run}/{n_eps_avail}")
            print(f"    obj_of_interest: {sorted(obj_of_interest)}")
            print(f"    obj→region: {obj_to_region}")

        for ep in range(n_eps_run):
            try:
                env.reset()
                padded = pad_init_state_for_hand(init_states[ep], env)
                env.set_init_state(padded)
            except Exception as exc:
                print(f"[warn] {suite_name} task {task_id} ep {ep}: set_init_state failed: {exc}")
                continue

            for obj_name, region_name in obj_to_region.items():
                if is_obstacle(obj_name):
                    continue
                box = regions.get(region_name)
                if box is None:
                    continue
                xyz = get_object_xy_world(env, obj_name)
                if xyz is None:
                    continue
                inside = box.contains(xyz[0], xyz[1], tol=tol)
                cx = 0.5 * (box.x_min + box.x_max)
                cy = 0.5 * (box.y_min + box.y_max)
                d = float(np.hypot(xyz[0] - cx, xyz[1] - cy))
                hidden = abs(xyz[0]) > 5.0 or abs(xyz[1]) > 5.0
                results.append(CheckResult(
                    suite=suite_name, task_id=task_id, task_name=task.name,
                    episode=ep, object=obj_name, region=region_name,
                    declared_box=str(box), actual_xyz=xyz, inside=inside, delta_xy=d,
                    is_obj_of_interest=(obj_name in obj_of_interest),
                    is_hidden=hidden,
                ))

        env.close()

    return results


def summarise(results: List[CheckResult]) -> None:
    """Print a per-(suite, task) compliance summary.

    We separate three error tiers per task:
      - target_OOB: any obj_of_interest landed outside its declared region
                    AND was not teleported far away (real placement bug)
      - hidden_OOB: object teleported to (~10, 10) etc. — by design for some
                    distractors, listed only as info
      - other_OOB: any non-target object that's also out of region
    """
    by_task: Dict[Tuple[str, int, str], List[CheckResult]] = {}
    for r in results:
        by_task.setdefault((r.suite, r.task_id, r.task_name), []).append(r)

    print("\n" + "=" * 110)
    print(
        f"{'suite':<22}{'tid':>3}  {'name':<55}"
        f"{'eps':>4}{'tgt_BAD':>9}{'oth_OOB':>9}{'hidden':>8}"
    )
    print("-" * 110)
    for (suite, tid, name), recs in sorted(by_task.items()):
        eps = len({r.episode for r in recs})
        target_bad = sum(1 for r in recs if r.is_obj_of_interest and not r.inside and not r.is_hidden)
        target_total = sum(1 for r in recs if r.is_obj_of_interest)
        other_bad = sum(1 for r in recs if (not r.is_obj_of_interest) and (not r.inside) and (not r.is_hidden))
        hidden = sum(1 for r in recs if r.is_hidden)
        flag = "  ❌" if target_bad > 0 else ("  ⚠" if other_bad > 0 else "  ✓")
        tgt_str = f"{target_bad}/{target_total}" if target_total else "n/a"
        print(
            f"{suite:<22}{tid:>3}  {name[:55]:<55}"
            f"{eps:>4}{tgt_str:>9}{other_bad:>9}{hidden:>8}{flag}"
        )
    print("=" * 110)
    print("Legend: tgt_BAD = obj_of_interest misplaced (REAL bug)")
    print("        oth_OOB = other named object OOB (might still distract policy)")
    print("        hidden  = object teleported far away (often by design)")


def write_csv(results: List[CheckResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "suite", "task_id", "task_name", "episode", "object",
            "is_obj_of_interest", "is_hidden", "region", "declared_box",
            "actual_x", "actual_y", "actual_z",
            "inside_region", "delta_xy_to_center",
        ])
        for r in results:
            w.writerow([
                r.suite, r.task_id, r.task_name, r.episode, r.object,
                r.is_obj_of_interest, r.is_hidden, r.region, r.declared_box,
                round(r.actual_xyz[0], 4),
                round(r.actual_xyz[1], 4),
                round(r.actual_xyz[2], 4),
                r.inside, round(r.delta_xy, 4),
            ])
    print(f"\n[ok] wrote per-row CSV: {out_path}  ({len(results)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="*", default=list(ALL_SUITES),
                    help="Suite names to scan (default: all 4 safelibero suites).")
    ap.add_argument("--episodes", type=int, default=3,
                    help="How many episodes per task to sample (default: 3).")
    ap.add_argument("--tolerance", type=float, default=0.005,
                    help="xy tolerance in metres added to each region's bounds (default: 5mm).")
    ap.add_argument("--safety-level", default="I", choices=["I", "II"],
                    help="Safety level for init state files (default: I).")
    ap.add_argument("--out-csv", type=Path,
                    default=_PROJ_ROOT / "dynisafety_libero" / "outputs" / "_aggregated" / "pruned_init_compliance.csv")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-task verbose dump (still prints summary table).")
    args = ap.parse_args()

    all_results: List[CheckResult] = []
    for suite in args.suites:
        all_results.extend(scan_suite(
            suite_name=suite,
            n_episodes=args.episodes,
            tol=args.tolerance,
            safety_level=args.safety_level,
            verbose=not args.quiet,
        ))

    if not all_results:
        print("[error] no results collected.")
        return

    summarise(all_results)
    write_csv(all_results, args.out_csv)

    # Real bugs first: obj_of_interest that's misplaced and NOT teleported away.
    real_bugs = [r for r in all_results
                 if r.is_obj_of_interest and not r.inside and not r.is_hidden]
    real_bugs.sort(key=lambda r: -r.delta_xy)
    print(f"\nREAL bugs (obj_of_interest misplaced, not teleported): {len(real_bugs)}")
    print("-" * 110)
    for r in real_bugs[:25]:
        print(
            f"  {r.suite:<22}t{r.task_id} ep{r.episode:<2} "
            f"{r.object:<22} actual=({r.actual_xyz[0]:+.3f}, {r.actual_xyz[1]:+.3f}) "
            f"region={r.region}  declared={r.declared_box}  Δ={r.delta_xy:.3f}m"
        )


if __name__ == "__main__":
    main()
