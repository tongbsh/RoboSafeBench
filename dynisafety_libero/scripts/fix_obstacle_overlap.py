#!/usr/bin/env python3
"""Fix L1 obstacle / task-object overlap caused by the earlier pruned_init fix.

Background
----------
After ``fix_pruned_init.py`` restored task objects to their training-distribution
positions, several obstacles' saved init positions ended up <6cm from the new
task object location. When the L1 SuddenDropController spawns those obstacles
(it drops them onto their saved init xy), they collide directly with the task
object and bounce it away.

What this script does
---------------------
For each (suite, task) where any obstacle is closer than ``--min-dist`` to a
task object, recompute that obstacle's xy as::

    new_obstacle_xy = task_xy + unit(eef_xy - task_xy) * push_dist

i.e. push the obstacle along the task-object → EEF direction by ``--push-dist``
(default 0.10m). This keeps the obstacle on the policy's likely approach path
(so L1 still presents a real disturbance) while guaranteeing physical clearance.

Only the offending obstacle's 7-qpos slot is rewritten in the safelibero
pruned_init; every other DoF (other obstacles, hand_pusher, distractor objects,
robot, task object) stays byte-identical. A ``.bak2`` backup is written next to
each modified pruned_init.

Usage::

    python dynisafety_libero/scripts/fix_obstacle_overlap.py            # dry-run
    python dynisafety_libero/scripts/fix_obstacle_overlap.py --apply    # write
    python dynisafety_libero/scripts/fix_obstacle_overlap.py --apply \\
        --min-dist 0.06 --push-dist 0.10
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "simulator_env" / "dyniSafety_libero"))
os.environ.setdefault(
    "LIBERO_DATA_ROOT",
    str(_PROJ_ROOT / "simulator_env" / "dyniSafety_libero" / "libero" / "libero"),
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from dynisafety_libero.env_utils import pad_init_state_for_hand  # noqa: E402
from dynisafety_libero.scripts.diagnose_pruned_init import parse_bddl, is_obstacle  # noqa: E402

ALL_SUITES = ("safelibero_spatial", "safelibero_object", "safelibero_goal", "safelibero_long")


def _slot_for(env, name: str, hand_qpos_addr: int) -> Optional[int]:
    sim = env.env.sim
    obj = env.env.objects_dict.get(name)
    if obj is None:
        return None
    try:
        jid = sim.model.joint_name2id(obj.joints[-1])
        addr = sim.model.jnt_qposadr[jid]
    except Exception:
        return None
    if hand_qpos_addr >= 0 and addr > hand_qpos_addr:
        addr -= 7
    return 1 + addr


def _eef_xy_from_state(env, init_state_padded: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Apply the init state and read the robot end-effector world xy/z."""
    try:
        env.reset()
        env.set_init_state(init_state_padded)
    except Exception:
        return None
    sim = env.env.sim
    try:
        eef_id = sim.model.site_name2id(env.env.robots[0].robot_model.eef_name)
    except Exception:
        # robosuite versions where the helper differs
        try:
            eef_id = env.env.robots[0].eef_site_id
        except Exception:
            return None
    pos = sim.data.site_xpos[eef_id].copy()
    return float(pos[0]), float(pos[1]), float(pos[2])


def scan_and_fix(
    suite_name: str,
    apply: bool,
    min_dist: float,
    push_dist: float,
) -> Dict[str, int]:
    bdict = benchmark.get_benchmark_dict()
    suite = bdict[suite_name](safety_level="I")
    counts = {"checked": 0, "moved": 0, "kept": 0, "no_target": 0}

    for tid in range(suite.n_tasks):
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), suite_name, task.bddl_file)
        regions, obj_to_region, obj_of_interest = parse_bddl(bddl)
        target_objs = [n for n in obj_of_interest if not is_obstacle(n)]
        if not target_objs:
            counts["no_target"] += 1
            continue

        init_path = os.path.join(
            get_libero_path("init_states"), suite_name,
            task.init_states_file.replace(".pruned_init", "_level_I.pruned_init"),
        )
        init = torch.load(init_path)

        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
        env.seed(0)
        sim = env.env.sim
        hand_addr = -1
        h = env.env.objects_dict.get("hand_pusher_1")
        if h is not None:
            try:
                jid = sim.model.joint_name2id(h.joints[-1])
                hand_addr = sim.model.jnt_qposadr[jid]
            except Exception:
                pass

        # Resolve EEF xy by actually applying ep0's init state (it's identical
        # across episodes for the robot DoFs, so we only do this once).
        try:
            padded0 = pad_init_state_for_hand(init[0], env)
            eef_xy = _eef_xy_from_state(env, padded0)
        except Exception as exc:
            print(f"[warn] {suite_name} t{tid}: EEF resolve failed: {exc}", flush=True)
            env.close()
            continue
        if eef_xy is None:
            print(f"[warn] {suite_name} t{tid}: cannot read EEF site_xpos", flush=True)
            env.close()
            continue

        # Now pull every object's saved xy (raw bytes, episode 0)
        slot_map: Dict[str, int] = {}
        for n in env.env.objects_dict.keys():
            s = _slot_for(env, n, hand_addr)
            if s is not None and s + 7 <= init.shape[1]:
                slot_map[n] = s
        env.close()

        # Active (visible) obstacles only — skip those teleported to (10,10)
        target_xy = {n: (float(init[0, slot_map[n]]), float(init[0, slot_map[n] + 1]))
                     for n in target_objs if n in slot_map}
        active_obstacles = {
            n: (float(init[0, s]), float(init[0, s + 1]))
            for n, s in slot_map.items()
            if is_obstacle(n) and "hand" not in n.lower()
            and "box_base" not in n.lower() and "box_small_base" not in n.lower()
            and abs(init[0, s]) < 5 and abs(init[0, s + 1]) < 5
        }

        any_change_in_task = False
        modified = init.copy()
        for obs_name, (ox, oy) in active_obstacles.items():
            for tname, (tx, ty) in target_xy.items():
                d = float(np.hypot(ox - tx, oy - ty))
                counts["checked"] += 1
                if d >= min_dist:
                    counts["kept"] += 1
                    continue
                # COMPUTE new obstacle xy: from task_obj along (eef - task) by push_dist
                vec = np.array([eef_xy[0] - tx, eef_xy[1] - ty])
                vec_norm = float(np.linalg.norm(vec))
                if vec_norm < 1e-6:
                    direction = np.array([0.0, 1.0])  # arbitrary fallback
                else:
                    direction = vec / vec_norm
                new_x = tx + direction[0] * push_dist
                new_y = ty + direction[1] * push_dist
                old_xy_str = f"({ox:+.3f}, {oy:+.3f})"
                new_xy_str = f"({new_x:+.3f}, {new_y:+.3f})"
                eef_str = f"({eef_xy[0]:+.3f}, {eef_xy[1]:+.3f})"
                print(
                    f"  >>> {suite_name} t{tid}  "
                    f"target={tname} target_xy=({tx:+.3f}, {ty:+.3f})  eef_xy={eef_str}",
                    flush=True,
                )
                print(
                    f"      obstacle={obs_name}: {old_xy_str} -> {new_xy_str}  "
                    f"old_dist={d:.3f}m  new_dist={push_dist:.3f}m",
                    flush=True,
                )

                # Apply to ALL episodes (obstacle xy is identical across eps)
                s = slot_map[obs_name]
                modified[:, s] = new_x
                modified[:, s + 1] = new_y
                # Keep z and quat unchanged.
                counts["moved"] += 1
                any_change_in_task = True

        if any_change_in_task:
            if apply:
                bak = init_path + ".bak2"
                if not os.path.exists(bak):
                    shutil.copy2(init_path, bak)
                    print(f"      [bak2] {bak}", flush=True)
                torch.save(modified, init_path)
                print(f"      [save] {init_path}", flush=True)
            else:
                print("      [dry-run] would save above changes; rerun with --apply", flush=True)

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="*", default=list(ALL_SUITES))
    ap.add_argument("--apply", action="store_true",
                    help="Write modified pruned_init files (default: dry-run).")
    ap.add_argument("--min-dist", type=float, default=0.06,
                    help="Treat obstacle/task pairs closer than this as overlap (m).")
    ap.add_argument("--push-dist", type=float, default=0.10,
                    help="Where to relocate the offending obstacle: along task→EEF, this far (m).")
    args = ap.parse_args()

    print(f"[mode] {'APPLY' if args.apply else 'DRY-RUN (no writes)'}", flush=True)
    print(f"[suites] {args.suites}", flush=True)
    print(f"[thresholds] min_dist={args.min_dist}m, push_dist={args.push_dist}m", flush=True)

    grand = {"checked": 0, "moved": 0, "kept": 0, "no_target": 0}
    for s in args.suites:
        c = scan_and_fix(s, apply=args.apply,
                         min_dist=args.min_dist, push_dist=args.push_dist)
        for k, v in c.items():
            grand[k] += v

    print()
    print("=" * 72)
    print(
        f"GRAND  checked_pairs={grand['checked']}  "
        f"kept={grand['kept']}  moved={grand['moved']}  "
        f"no_target_skip={grand['no_target']}",
        flush=True,
    )
    print(f"  mode: {'APPLY' if args.apply else 'DRY-RUN'}", flush=True)
    print("=" * 72)


if __name__ == "__main__":
    main()
