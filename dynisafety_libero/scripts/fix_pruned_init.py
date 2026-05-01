#!/usr/bin/env python3
"""Fix safelibero ``pruned_init`` files where ``obj_of_interest`` ended up
out-of-region.

Diagnostic showed: the safelibero pruned_init reuses every other object's
xy verbatim from the original LIBERO init, **but the target object** was
displaced to a position outside the BDDL ``target_object_region`` (and the
training data distribution). Models therefore can't find the target.

Fix strategy: for every (suite, task, episode) where the target object is
out-of-region, copy that object's xy (and full quaternion) from the
original LIBERO pruned_init into the safelibero state, leave every other
DoF untouched, and rewrite the pruned_init file.

Outputs:
  * Backups: ``<file>.bak`` next to each modified pruned_init
  * Modified ``*_level_I.pruned_init`` in place

Usage::

    conda activate openvla-oft
    # Dry-run (default): show what would change, no files written
    python dynisafety_libero/scripts/fix_pruned_init.py
    # Apply fixes
    python dynisafety_libero/scripts/fix_pruned_init.py --apply
    # Limit to one suite
    python dynisafety_libero/scripts/fix_pruned_init.py --apply --suites safelibero_object
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

# ---------------------------------------------------------------------------
# Project bootstrap (mirrors run_dynisafety_eval.py)
# ---------------------------------------------------------------------------
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
from dynisafety_libero.scripts.diagnose_pruned_init import (  # noqa: E402
    parse_bddl,
    is_obstacle,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ORIG_LIBERO_ROOT = _PROJ_ROOT / "simulator_env" / "LIBERO" / "libero" / "libero"
SAFE_LIBERO_ROOT = _PROJ_ROOT / "simulator_env" / "dyniSafety_libero" / "libero" / "libero"

# safelibero suite name → original LIBERO problem folder name
ORIG_SUITE_FOLDER = {
    "safelibero_spatial": "libero_spatial",
    "safelibero_object":  "libero_object",
    "safelibero_goal":    "libero_goal",
    "safelibero_long":    "libero_10",  # LIBERO-100 long-horizon split
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_object_qpos(env, obj_name: str) -> Optional[np.ndarray]:
    """Return the 7-element free-joint qpos (x,y,z, qw,qx,qy,qz) for an object."""
    obj = env.env.objects_dict.get(obj_name)
    if obj is None:
        return None
    sim = env.env.sim
    try:
        joint_name = obj.joints[-1]
        qpos_addr = sim.model.jnt_qposadr[sim.model.joint_name2id(joint_name)]
        return np.array(sim.data.qpos[qpos_addr:qpos_addr + 7], copy=True)
    except Exception:
        return None


def set_object_qpos(env, obj_name: str, qpos7: np.ndarray) -> bool:
    obj = env.env.objects_dict.get(obj_name)
    if obj is None:
        return False
    sim = env.env.sim
    try:
        joint_name = obj.joints[-1]
        qpos_addr = sim.model.jnt_qposadr[sim.model.joint_name2id(joint_name)]
        sim.data.qpos[qpos_addr:qpos_addr + 7] = qpos7
        # zero out velocities for this object
        dof_addr = sim.model.jnt_dofadr[sim.model.joint_name2id(joint_name)]
        sim.data.qvel[dof_addr:dof_addr + 6] = 0.0
        return True
    except Exception:
        return False


def get_object_region(obj_to_region, regions, obj):
    return regions.get(obj_to_region.get(obj, ""))


def in_region(box, x, y, tol=0.005):
    return box and box.contains(x, y, tol=tol)


def find_orig_bddl_init(task_name: str, suite_name: str) -> Optional[Tuple[Path, Path]]:
    """Locate the matching original-LIBERO BDDL + pruned_init pair, if any."""
    folder = ORIG_SUITE_FOLDER.get(suite_name)
    if folder is None:
        return None
    bddl_dir = ORIG_LIBERO_ROOT / "bddl_files" / folder
    init_dir = ORIG_LIBERO_ROOT / "init_files" / folder

    candidates = [
        f"{task_name}.bddl",                              # plain
        f"KITCHEN_SCENE10_{task_name}.bddl",
    ]
    bddl_path = None
    for c in candidates:
        p = bddl_dir / c
        if p.exists():
            bddl_path = p
            break
    if bddl_path is None:
        for p in bddl_dir.glob(f"*{task_name}.bddl"):
            bddl_path = p
            break
    if bddl_path is None:
        return None

    init_path = init_dir / (bddl_path.stem + ".pruned_init")
    if not init_path.exists():
        return None
    return bddl_path, init_path


# ---------------------------------------------------------------------------
# Main per-task fix routine
# ---------------------------------------------------------------------------
def _build_qpos_addr_map(env, obj_names: List[str]) -> Dict[str, int]:
    """Return ``{obj_name: qpos_addr_in_env_model}`` for every obj that exists."""
    sim = env.env.sim
    out: Dict[str, int] = {}
    for n in obj_names:
        obj = env.env.objects_dict.get(n)
        if obj is None:
            continue
        try:
            jid = sim.model.joint_name2id(obj.joints[-1])
            out[n] = sim.model.jnt_qposadr[jid]
        except Exception:
            pass
    return out


def fix_one_task(
    suite_name: str,
    task_id: int,
    apply: bool,
    tol: float,
    settle_steps: int,
    max_episodes: Optional[int] = None,
) -> Dict[str, int]:
    """Fast in-place patcher.

    1. Build *one* safe env and *one* orig env per task; never reset/set_init
       inside the per-episode loop.
    2. Compute each task object's ``qpos_addr`` once in each model.
    3. The saved flat state has layout ``[time(1), qpos(nq_safe-7), qvel]``
       (the hand_pusher block is stripped). For object qpos addresses *before*
       the hand block, the saved-state slot equals ``1 + addr_in_safe_model``;
       for addresses *after* it, subtract the hand's 7-qpos width.
    4. For every (episode, object_of_interest), check whether the safe state's
       saved xy is inside the BDDL region; if not, copy the orig state's
       7-qpos slot into the safe state's slot. Pure numpy — no MuJoCo step.
    """
    bdict = benchmark.get_benchmark_dict()
    suite = bdict[suite_name](safety_level="I")
    task = suite.get_task(task_id)

    bddl_safe = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_safe_path = os.path.join(
        get_libero_path("init_states"), task.problem_folder,
        task.init_states_file.replace(".pruned_init", "_level_I.pruned_init"),
    )
    regions, obj_to_region, obj_of_interest = parse_bddl(bddl_safe)

    pair = find_orig_bddl_init(task.name, suite_name)
    if pair is None:
        print(f"  [skip] {suite_name} t{task_id} {task.name}: no original LIBERO match", flush=True)
        return {"checked": 0, "fixed": 0, "kept": 0, "no_orig_obj": 0, "no_target_objs": 0}
    bddl_orig, init_orig_path = pair

    init_safe = torch.load(init_safe_path)
    init_orig = torch.load(init_orig_path)

    n_eps_full = init_safe.shape[0]
    n_eps = n_eps_full if max_episodes is None else min(max_episodes, n_eps_full)
    print(f"\n  >>> {suite_name} t{task_id}  {task.name}", flush=True)
    if n_eps < n_eps_full:
        print(f"      [limit] processing {n_eps}/{n_eps_full} episodes", flush=True)
    print(f"      safe  init: {os.path.basename(init_safe_path)}  shape={tuple(init_safe.shape)}", flush=True)
    print(f"      orig  init: {init_orig_path.name}  shape={tuple(init_orig.shape)}", flush=True)
    print(f"      obj_of_interest: {sorted(obj_of_interest)}", flush=True)

    counts = {"checked": 0, "fixed": 0, "kept": 0, "no_orig_obj": 0, "no_target_objs": 0}
    target_objs = [n for n in obj_of_interest if not is_obstacle(n)]
    if not target_objs:
        counts["no_target_objs"] = 1
        return counts

    # Build envs once. We need them only to look up qpos addresses + hand offset.
    env_safe = OffScreenRenderEnv(bddl_file_name=bddl_safe, camera_heights=64, camera_widths=64)
    env_safe.seed(0)
    env_orig = OffScreenRenderEnv(bddl_file_name=str(bddl_orig), camera_heights=64, camera_widths=64)
    env_orig.seed(0)

    addr_safe = _build_qpos_addr_map(env_safe, target_objs)
    addr_orig = _build_qpos_addr_map(env_orig, target_objs)

    # Hand DoF block in safelibero (we strip it when persisting).
    hand_qpos_addr = -1
    hand = env_safe.env.objects_dict.get("hand_pusher_1")
    if hand is not None:
        try:
            jid = env_safe.env.sim.model.joint_name2id(hand.joints[-1])
            hand_qpos_addr = env_safe.env.sim.model.jnt_qposadr[jid]
        except Exception:
            pass

    env_safe.close()
    env_orig.close()

    def safe_slot(addr: int) -> int:
        """Map a safelibero qpos address into the saved flat-state index."""
        # saved layout starts with 1 timestep value then qpos (hand stripped)
        if hand_qpos_addr >= 0 and addr > hand_qpos_addr:
            addr -= 7
        return 1 + addr

    def orig_slot(addr: int) -> int:
        """Original LIBERO has no hand, no obstacle DoFs."""
        return 1 + addr

    safe_init_modified = init_safe.copy() if isinstance(init_safe, np.ndarray) else init_safe.clone()

    for ep in range(n_eps):
        flat_safe = safe_init_modified[ep]
        flat_orig = init_orig[ep]

        for obj_name in target_objs:
            if obj_name not in addr_safe:
                continue
            box = get_object_region(obj_to_region, regions, obj_name)
            if box is None:
                continue
            counts["checked"] += 1

            s_idx = safe_slot(addr_safe[obj_name])
            if s_idx + 7 > flat_safe.shape[0]:
                continue
            cur_xy = (float(flat_safe[s_idx]), float(flat_safe[s_idx + 1]))
            if box.contains(cur_xy[0], cur_xy[1], tol=tol):
                counts["kept"] += 1
                continue

            if obj_name not in addr_orig:
                counts["no_orig_obj"] += 1
                continue
            o_idx = orig_slot(addr_orig[obj_name])
            if o_idx + 7 > flat_orig.shape[0]:
                counts["no_orig_obj"] += 1
                continue

            # Copy original 7-tuple (xyz + quat); preserve safe z to be safe
            # against fixture-height differences (rare but cheap).
            new_qpos = np.array(flat_orig[o_idx:o_idx + 7], copy=True)
            new_qpos[2] = flat_safe[s_idx + 2]

            flat_safe[s_idx:s_idx + 7] = new_qpos
            counts["fixed"] += 1
            if ep < 1:
                print(
                    f"      [ep{ep}] fix {obj_name}: "
                    f"({cur_xy[0]:+.3f}, {cur_xy[1]:+.3f}) -> "
                    f"({new_qpos[0]:+.3f}, {new_qpos[1]:+.3f})  "
                    f"region={box}",
                    flush=True,
                )

        safe_init_modified[ep] = flat_safe

    if apply and counts["fixed"] > 0:
        backup = init_safe_path + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(init_safe_path, backup)
            print(f"      [bak] backed up to {backup}", flush=True)
        torch.save(safe_init_modified, init_safe_path)
        print(f"      [save] wrote {init_safe_path}", flush=True)
    elif counts["fixed"] > 0:
        print(
            f"      [dry-run] would patch {counts['fixed']} object-episodes "
            f"(kept={counts['kept']}). Rerun with --apply to write.",
            flush=True,
        )
    else:
        print("      [ok] nothing to fix", flush=True)

    return counts


def strip_hand_from_state(flat_state: np.ndarray, env) -> np.ndarray:
    """Remove the 7 qpos + 6 qvel of hand_pusher_1 from a flattened sim state.

    The pruned_init format does NOT include the hand (it predates the safety
    extension); ``pad_init_state_for_hand`` adds it back when loading. We
    therefore mirror that operation in reverse before persisting.
    """
    inner = env.env
    sim = inner.sim
    hand = inner.objects_dict.get("hand_pusher_1")
    if hand is None:
        return flat_state.copy()
    nq = sim.model.nq
    nv = sim.model.nv
    # flat layout is [time(1), qpos(nq), qvel(nv)]
    # If your robosuite version omits the time prefix this still works because
    # we only strip relative slots, but we still need to know the total layout.
    expected_full = 1 + nq + nv
    if flat_state.shape[0] != expected_full:
        # Some robosuite versions don't prefix time; handle that gracefully.
        offset = 0 if flat_state.shape[0] == nq + nv else None
        if offset is None:
            return flat_state.copy()
    else:
        offset = 1

    jid = sim.model.joint_name2id(hand.joints[-1])
    qpos_addr = sim.model.jnt_qposadr[jid]
    dof_addr = sim.model.jnt_dofadr[jid]

    qpos = flat_state[offset:offset + nq]
    qvel = flat_state[offset + nq:offset + nq + nv]

    new_qpos = np.delete(qpos, np.s_[qpos_addr:qpos_addr + 7])
    new_qvel = np.delete(qvel, np.s_[dof_addr:dof_addr + 6])

    if offset == 1:
        return np.concatenate([flat_state[:1], new_qpos, new_qvel])
    return np.concatenate([new_qpos, new_qvel])


def _patch_objects_in_flat(
    arr,
    ep: int,
    env,
    objects_qpos: List[Tuple[str, Optional[np.ndarray]]],
) -> None:
    """Fallback patcher: write only the listed objects' qpos slots into the
    saved flat state, leaving everything else untouched. We compute the
    *stripped-format* qpos addresses by removing the hand range first."""
    inner = env.env
    sim = inner.sim
    hand = inner.objects_dict.get("hand_pusher_1")
    hand_qpos_addr = -1
    if hand is not None:
        jid = sim.model.joint_name2id(hand.joints[-1])
        hand_qpos_addr = sim.model.jnt_qposadr[jid]

    flat = arr[ep]
    # offset of qpos within the saved flat: 1 (timestep) on most versions
    qpos_offset = 1
    nq = sim.model.nq
    for obj_name, qpos7 in objects_qpos:
        if qpos7 is None:
            continue
        obj = inner.objects_dict.get(obj_name)
        if obj is None:
            continue
        try:
            jid = sim.model.joint_name2id(obj.joints[-1])
            full_addr = sim.model.jnt_qposadr[jid]
        except Exception:
            continue
        # Map full-model qpos addr → stripped (pre-hand) addr
        stripped_addr = full_addr if (hand_qpos_addr < 0 or full_addr < hand_qpos_addr) else (full_addr - 7)
        slot_start = qpos_offset + stripped_addr
        if slot_start + 7 > flat.shape[0]:
            continue
        flat[slot_start:slot_start + 7] = qpos7
    arr[ep] = flat


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="*", default=list(ORIG_SUITE_FOLDER.keys()))
    ap.add_argument("--apply", action="store_true",
                    help="Write modified pruned_init files (default: dry-run).")
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--settle-steps", type=int, default=0,
                    help="Optional sim.forward() iterations to settle contacts after patch.")
    ap.add_argument("--episodes", type=int, default=None,
                    help="Limit episodes per task (default: all 50). Useful for quick dry-runs.")
    args = ap.parse_args()

    print(f"[mode] {'APPLY (writing files + .bak)' if args.apply else 'DRY-RUN (no writes)'}")
    print(f"[suites] {args.suites}")
    grand = {"checked": 0, "fixed": 0, "kept": 0, "no_orig_obj": 0, "no_target_objs": 0}

    for suite_name in args.suites:
        bdict = benchmark.get_benchmark_dict()
        suite_cls = bdict.get(suite_name)
        if suite_cls is None:
            print(f"\n[skip] suite '{suite_name}' not registered.")
            continue
        suite = suite_cls(safety_level="I")
        for task_id in range(suite.n_tasks):
            counts = fix_one_task(
                suite_name, task_id, apply=args.apply,
                tol=args.tolerance, settle_steps=args.settle_steps,
                max_episodes=args.episodes,
            )
            for k, v in counts.items():
                grand[k] += v

    print("\n" + "=" * 70, flush=True)
    print(
        f"GRAND TOTAL  checked={grand['checked']} kept={grand['kept']} "
        f"fixed={grand['fixed']} no_orig_obj={grand['no_orig_obj']} "
        f"no_target_objs={grand['no_target_objs']}",
        flush=True,
    )
    print(f"  mode: {'APPLY' if args.apply else 'DRY-RUN (no files written)'}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
