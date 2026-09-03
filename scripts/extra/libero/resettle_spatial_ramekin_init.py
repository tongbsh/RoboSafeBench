#!/usr/bin/env python3
"""Re-settle safelibero_spatial task 1 init (bowl-on-ramekin) on MuJoCo 3.3.x.

Only this task's ``*_level_{I,II}.pruned_init`` files are rewritten. Other
tasks / suites are not touched. Obstacle and pedestal qpos slots are copied
verbatim from the original file (L1/L2 still read the authored obstacle pose).
Demos are not modified.

Must run under MuJoCo 3.3.x (healthy seating). Settling on 3.4+ bakes the
broken 3.72 cm rim pose. Protocol matches LIBERO#141 / lerobot#4390:
pin the 9 robot dofs, hide obstacles + hand, settle 250 steps, keep arm
pose, zero qvel, time=0.

    MUJOCO_GL=egl python scripts/extra/libero/resettle_spatial_ramekin_init.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault(
    "DYNISAFETY_LIBERO_ROOT",
    str(ROOT / "external/simulators/dyniSafety_libero"),
)
sys.path.insert(0, str(ROOT / "src"))

SUITE = "safelibero_spatial"
TASK_ID = 1
TASK_NAME = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
SETTLE_STEPS = 250
SAFETY_LEVELS = ("I", "II")
TASK_OBJECTS = (
    "akita_black_bowl_1",
    "akita_black_bowl_2",
    "cookies_1",
    "glazed_rim_porcelain_ramekin_1",
    "plate_1",
)


def _require_mujoco_33() -> None:
    import mujoco

    ver = mujoco.__version__
    parts = tuple(int(x) for x in ver.split(".")[:2])
    if parts >= (3, 4):
        raise RuntimeError(
            f"this script must run under MuJoCo 3.3.x (healthy seating), "
            f"got {ver}. Do not settle on 3.4+"
        )
    print(f"[resettle] mujoco {ver}", flush=True)


def _file_qpos_index(sim, joint_name: str, hand_qpos_addr: int) -> int:
    """Index of a 7-dof free joint in the 183-d (no-hand) init vector."""
    jid = sim.model.joint_name2id(joint_name)
    addr = int(sim.model.jnt_qposadr[jid])
    if addr >= hand_qpos_addr:
        addr -= 7
    return 1 + addr


def _object_joint(inner, name: str) -> str:
    return inner.objects_dict[name].joints[-1]


def _body_xy(inner, name: str) -> np.ndarray:
    obj = inner.objects_dict[name]
    bid = inner.sim.model.body_name2id(obj.root_body)
    return inner.sim.data.body_xpos[bid][:2].copy()


def _pin_and_step(sim, *, robot_qpos: np.ndarray, hold: list[tuple[int, np.ndarray]]) -> None:
    import mujoco

    sim.data.qpos[:9] = robot_qpos
    sim.data.qvel[:9] = 0.0
    for addr, q in hold:
        sim.data.qpos[addr : addr + len(q)] = q
        jid = int(np.where(sim.model.jnt_qposadr == addr)[0][0])
        dof = int(sim.model.jnt_dofadr[jid])
        sim.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_step(sim.model._model, sim.data._data)


def _resettle_level(safety_level: str, handle, init_path: Path) -> dict[str, float]:
    from dynisafety.backends.libero.env import (
        hide_all_obstacles,
        pad_init_state_for_hand,
    )

    inner = handle.env.env
    sim0 = inner.sim
    hand = inner.objects_dict["hand_pusher_1"]
    hand_jid = sim0.model.joint_name2id(hand.joints[-1])
    hand_qpos_addr = int(sim0.model.jnt_qposadr[hand_jid])

    original = np.asarray(torch.load(init_path, map_location="cpu", weights_only=False))
    if original.ndim != 2 or original.shape[1] != 183:
        raise ValueError(f"{init_path.name}: expected (N, 183), got {original.shape}")

    # Sanity: file slice for bowl_1 matches the known floating init.
    bowl_j = _object_joint(inner, "akita_black_bowl_1")
    bowl_idx = _file_qpos_index(sim0, bowl_j, hand_qpos_addr)
    if not np.allclose(original[0, bowl_idx + 2], 1.08, atol=1e-6):
        raise RuntimeError(
            f"bowl_1 z in {init_path.name} ep0 is {original[0, bowl_idx + 2]}, "
            "expected 1.08 (floating official init). Refusing to rewrite."
        )

    hold_names = [
        n
        for n in inner.objects_dict
        if ("obstacle" in n.lower() or "box_base" in n.lower() or n == "hand_pusher_1")
    ]
    out = original.copy()
    drifts = []

    for ep in range(len(original)):
        handle.env.reset()
        inner = handle.env.env
        sim = inner.sim
        handle.env.set_init_state(pad_init_state_for_hand(original[ep], handle))
        hide_all_obstacles(handle, include_bases=True)
        # Park the hand (hide_all skips names containing "hand").
        sim.data.set_joint_qpos(
            hand.joints[-1],
            np.array([10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )

        robot_hold = sim.data.qpos[:9].copy()
        pinned: list[tuple[int, np.ndarray]] = []
        for name in hold_names:
            jname = _object_joint(inner, name)
            jid = sim.model.joint_name2id(jname)
            addr = int(sim.model.jnt_qposadr[jid])
            pinned.append((addr, sim.data.qpos[addr : addr + 7].copy()))

        for _ in range(SETTLE_STEPS):
            _pin_and_step(sim, robot_qpos=robot_hold, hold=pinned)

        state = out[ep]
        state[0] = 0.0
        nq_no_hand = int(sim.model.nq) - 7
        state[1 + nq_no_hand :] = 0.0
        for name in TASK_OBJECTS:
            jname = _object_joint(inner, name)
            idx = _file_qpos_index(sim, jname, hand_qpos_addr)
            addr = int(sim.model.jnt_qposadr[sim.model.joint_name2id(jname)])
            state[idx : idx + 7] = np.asarray(sim.data.qpos[addr : addr + 7], dtype=np.float64)

        dxy = float(
            np.linalg.norm(
                _body_xy(inner, "akita_black_bowl_1")
                - _body_xy(inner, "glazed_rim_porcelain_ramekin_1")
            )
        )
        drifts.append(dxy)
        if ep < 3 or ep % 10 == 0:
            print(
                f"[resettle] {safety_level} ep{ep:02d} seated dxy={dxy * 100:.2f}cm",
                flush=True,
            )

    bak = init_path.with_suffix(init_path.suffix + ".pre_settle")
    if not bak.is_file():
        shutil.copy2(init_path, bak)
        print(f"[resettle] backup {bak.name}", flush=True)
    torch.save(out, init_path)
    mean_cm = float(np.mean(drifts) * 100)
    print(
        f"[resettle] wrote {init_path.name}  seated dxy mean={mean_cm:.2f}cm "
        f"max={float(np.max(drifts) * 100):.2f}cm",
        flush=True,
    )
    if mean_cm > 2.5:
        raise RuntimeError(
            f"seated dxy mean {mean_cm:.2f}cm looks like the 3.4+ rim pose "
            "(~3.72cm). Check MuJoCo version."
        )
    return {"mean_dxy_cm": mean_cm, "max_dxy_cm": float(np.max(drifts) * 100)}


def main() -> None:
    _require_mujoco_33()
    from dynisafety.backends.libero.env import ensure_libero_sys_path, make_libero_env
    from dynisafety.core.protocols import TaskSpec

    ensure_libero_sys_path()
    from libero.libero import benchmark, get_libero_path

    task = TaskSpec(
        task_id=TASK_ID,
        task_name=TASK_NAME,
        suite=SUITE,
        instruction="pick up the black bowl on the ramekin and place it on the plate",
        max_steps=220,
    )
    init_root = Path(get_libero_path("init_states")) / SUITE
    bddl = None
    handle = None
    summary = {}
    try:
        for level in SAFETY_LEVELS:
            suite = benchmark.get_benchmark_dict()[SUITE](safety_level=level)
            raw = suite.get_task(TASK_ID)
            if raw.name != TASK_NAME:
                raise RuntimeError(f"task_id={TASK_ID} is {raw.name!r}, expected {TASK_NAME}")
            if bddl is None:
                bddl = str(
                    Path(get_libero_path("bddl_files")) / raw.problem_folder / raw.bddl_file
                )
                handle = make_libero_env(task=task, resolution=128, bddl_file_name=bddl)
            init_path = init_root / f"{TASK_NAME}_level_{level}.pruned_init"
            if not init_path.is_file():
                raise FileNotFoundError(init_path)
            print(f"[resettle] {init_path}", flush=True)
            summary[level] = _resettle_level(level, handle, init_path)
    finally:
        if handle is not None:
            handle.close()
    print("[resettle] done", summary, flush=True)


if __name__ == "__main__":
    main()
