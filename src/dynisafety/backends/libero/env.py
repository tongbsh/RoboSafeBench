"""LIBERO / SafeLIBERO environment helpers (migrated from ``env_utils.py``).

Backend-only adaptation: env lifecycle, images, video, visibility, rescale.
No metric aggregation and no hyperparameter defaults (iron law 1).
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.core.protocols import TaskSpec

# Scene asset name for the SafeLIBERO hand body (not a tunable hyperparameter).
_HAND_OBJECT_NAME = "hand_pusher_1"

# Wait-phase no-op matching LIBERO/robosuite training convention (not YAML-tunable).
WAIT_PHASE_ACTION: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

# Ablation proxy RGBA used when hand.geometry == "box" (visual-only; physics unchanged).
_BOX_PROXY_RGBA = (0.87, 0.72, 0.60, 0.85)


def resolve_libero_root(
    *,
    workspace: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the SafeLIBERO / dyniSafety_libero checkout root.

    Priority: ``DYNISAFETY_LIBERO_ROOT`` → legacy ``DYNISAFETY_SIMENV`` → first
    tree under workspace that contains a ``libero/`` package
    (``third_party/dyniSafety_libero`` then
    ``external/simulators/dyniSafety_libero``).

    Needed because the fork is not reliably pip-installed: the shipped editable
    install registers a finder with an empty MAPPING, so a bare ``import
    libero`` fails. See ``ensure_libero_sys_path``.
    """
    env_map = env if env is not None else os.environ
    for var in ("DYNISAFETY_LIBERO_ROOT", "DYNISAFETY_SIMENV"):
        if var in env_map:
            root = Path(env_map[var])
            if root.is_dir():
                return root.resolve()
            raise FileNotFoundError(f"{var} is not a directory: {root}")

    if workspace is None:
        workspace = Path(__file__).resolve().parents[4]
    ws = workspace.resolve()
    candidates = (
        ws / "third_party" / "dyniSafety_libero",
        ws / "external" / "simulators" / "dyniSafety_libero",
    )
    for cand in candidates:
        if (cand / "libero").is_dir():
            return cand
    raise FileNotFoundError(
        "SafeLIBERO tree not found: set DYNISAFETY_LIBERO_ROOT, or run "
        f"scripts/install/install_libero.sh (tried {candidates[0]} and {candidates[1]})"
    )


def ensure_libero_sys_path(
    *,
    workspace: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Put the SafeLIBERO tree on ``sys.path`` so ``import libero`` resolves.

    Mirrors ``ensure_robotwin_sys_path``. Idempotent. Returns the root used.
    """
    root = resolve_libero_root(workspace=workspace, env=env)
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root


@dataclass
class LiberoEnvHandle:
    """Concrete ``EnvHandle`` for SafeLIBERO ``OffScreenRenderEnv``."""

    env: Any
    task_description: str
    suite: str
    task_id: int
    task_name: str

    def close(self) -> None:
        env = self.env
        close = getattr(env, "close", None)
        if callable(close):
            close()


def make_libero_env(
    *,
    task: TaskSpec,
    resolution: int,
    bddl_file_name: str,
) -> LiberoEnvHandle:
    """Create an OffScreenRenderEnv for *task*.

    ``resolution`` and ``bddl_file_name`` must be supplied by the caller from
    config / benchmark (no defaults here).
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution!r}")
    if not bddl_file_name:
        raise ValueError("bddl_file_name must be a non-empty path string")

    ensure_libero_sys_path()
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_name,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return LiberoEnvHandle(
        env=env,
        task_description=task.instruction,
        suite=task.suite,
        task_id=task.task_id,
        task_name=task.task_name,
    )


def get_primary_image(obs: Mapping[str, Any]) -> np.ndarray:
    """Third-person camera, rotated 180° (training preprocessing)."""
    img = obs["agentview_image"]
    return img[::-1, ::-1]


def get_wrist_image(obs: Mapping[str, Any]) -> np.ndarray:
    """Wrist camera, rotated 180° (training preprocessing)."""
    img = obs["robot0_eye_in_hand_image"]
    return img[::-1, ::-1]


def dummy_action() -> list[float]:
    """No-op action for the stabilisation / wait phase."""
    return list(WAIT_PHASE_ACTION)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle."""
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0

    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float64)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def pad_init_state_for_hand(
    init_state: np.ndarray,
    handle: LiberoEnvHandle,
) -> np.ndarray:
    """Pad a saved init_state for the hand body's extra DOFs."""
    env = handle.env
    inner = env.env
    sim = inner.sim
    expected_len = 1 + sim.model.nq + sim.model.nv

    if len(init_state) == expected_len:
        return init_state

    if _HAND_OBJECT_NAME not in inner.objects_dict:
        return np.asarray(init_state, dtype=np.float64)

    obj = inner.objects_dict[_HAND_OBJECT_NAME]
    jid = sim.model.joint_name2id(obj.joints[-1])
    qpos_addr = sim.model.jnt_qposadr[jid]
    dof_addr = sim.model.jnt_dofadr[jid]

    nq_orig = sim.model.nq - 7
    hand_qpos = np.array([10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    hand_qvel = np.zeros(6)

    state = np.asarray(init_state, dtype=np.float64)
    padded_qpos = np.insert(state[: 1 + nq_orig], 1 + qpos_addr, hand_qpos)
    padded_qvel = np.insert(state[1 + nq_orig :], dof_addr, hand_qvel)
    return np.concatenate([padded_qpos, padded_qvel])


def rescale_obstacle_inplace(
    handle: LiberoEnvHandle, obstacle_name: str, scale: float
) -> dict[str, Any]:
    """Scale one obstacle's geoms/meshes in-memory (this env instance only)."""
    import mujoco

    env = handle.env
    inner = env.env
    sim = inner.sim
    if scale == 1.0:
        return {"n_geoms": 0, "n_meshes": 0, "body_ids": []}
    obj = inner.objects_dict.get(obstacle_name)
    if obj is None:
        return {"n_geoms": 0, "n_meshes": 0, "body_ids": []}

    try:
        root_body_id = int(sim.model.body_name2id(obj.root_body))
    except Exception:
        return {"n_geoms": 0, "n_meshes": 0, "body_ids": []}

    body_ids = {root_body_id}
    nbody = int(sim.model.nbody)
    grew = True
    while grew:
        grew = False
        for child in range(nbody):
            parent = int(sim.model.body_parentid[child])
            if parent in body_ids and child not in body_ids:
                body_ids.add(child)
                grew = True

    geom_ids = [
        gid for gid in range(int(sim.model.ngeom)) if int(sim.model.geom_bodyid[gid]) in body_ids
    ]
    for gid in geom_ids:
        sim.model.geom_pos[gid] *= scale
        sim.model.geom_size[gid] *= scale

    mesh_geom_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    mesh_ids_used: set[int] = set()
    for gid in geom_ids:
        if int(sim.model.geom_type[gid]) == mesh_geom_type:
            mid = int(sim.model.geom_dataid[gid])
            if mid >= 0:
                mesh_ids_used.add(mid)

    n_meshes_scaled = 0
    for mid in mesh_ids_used:
        used_externally = False
        for other_gid in range(int(sim.model.ngeom)):
            if int(sim.model.geom_type[other_gid]) != mesh_geom_type:
                continue
            if int(sim.model.geom_dataid[other_gid]) != mid:
                continue
            if int(sim.model.geom_bodyid[other_gid]) not in body_ids:
                used_externally = True
                break
        if used_externally:
            continue
        start = int(sim.model.mesh_vertadr[mid])
        nverts = int(sim.model.mesh_vertnum[mid])
        sim.model.mesh_vert[start : start + nverts] *= scale
        n_meshes_scaled += 1

    mujoco.mj_forward(sim.model._model, sim.data._data)
    return {
        "n_geoms": len(geom_ids),
        "n_meshes": n_meshes_scaled,
        "body_ids": sorted(body_ids),
    }


def rescale_obstacles(handle: LiberoEnvHandle, rescale: Mapping[str, float]) -> None:
    """Apply per-obstacle scales from config ``disturbance.obstacle.rescale``."""
    for name, scale in rescale.items():
        factor = float(scale)
        if factor <= 0.0:
            raise ValueError(f"disturbance.obstacle.rescale[{name!r}] must be > 0, got {scale!r}")
        info = rescale_obstacle_inplace(handle, str(name), factor)
        print(
            f"[rescale] {name} x{factor:g} geoms={info['n_geoms']} meshes={info['n_meshes']}",
            flush=True,
        )


def hide_all_obstacles(handle: LiberoEnvHandle, *, include_bases: bool) -> None:
    """Teleport obstacles (and optionally pedestals) off-screen."""
    import mujoco

    env = handle.env
    inner = env.env
    sim = inner.sim
    for obj_name, obj in inner.objects_dict.items():
        nl = obj_name.lower()
        is_obstacle = "obstacle" in nl
        is_base = "box_base" in nl or "box_small_base" in nl
        if not is_obstacle and not (include_bases and is_base):
            continue
        if "hand" in nl:
            continue
        try:
            joint_name = obj.joints[-1]
            far_pos = np.array([10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            sim.data.set_joint_qpos(joint_name, far_pos)
            jid = sim.model.joint_name2id(joint_name)
            dof = sim.model.jnt_dofadr[jid]
            sim.data.qvel[dof : dof + 6] = 0.0
        except Exception:
            pass
    mujoco.mj_forward(sim.model._model, sim.data._data)


def _is_base_name(name: str) -> bool:
    nl = name.lower()
    return ("box_base" in nl or "box_small_base" in nl) and "hand" not in nl


def _is_obstacle_name(name: str) -> bool:
    nl = name.lower()
    return "obstacle" in nl and "hand" not in nl


def _is_visible_xy(pos: Any) -> bool:
    return abs(float(pos[0])) < 5.0 and abs(float(pos[1])) < 5.0


def _root_body_xy(sim: Any, obj: Any) -> np.ndarray:
    bid = sim.model.body_name2id(obj.root_body)
    return sim.data.body_xpos[bid][:2].copy()


def hide_overlapping_bases(
    handle: LiberoEnvHandle,
    *,
    overlap_radius: float,
    exclude_names: Sequence[str] = (),
    include_obstacles_in_check: bool = False,
    return_details: bool = False,
) -> list[Any]:
    """Hide pedestals that clip with non-obstacle scene objects."""
    import mujoco

    env = handle.env
    inner = env.env
    sim = inner.sim
    exclude = {str(n) for n in exclude_names}

    bases: list[tuple[str, Any, np.ndarray]] = []
    normals: list[tuple[str, np.ndarray]] = []
    for obj_name, obj in inner.objects_dict.items():
        if obj_name in exclude:
            continue
        if "hand" in obj_name.lower():
            continue
        try:
            xy = _root_body_xy(sim, obj)
        except Exception:
            continue
        if not _is_visible_xy(xy):
            continue
        if _is_base_name(obj_name):
            bases.append((obj_name, obj, xy))
            continue
        if _is_obstacle_name(obj_name) and not include_obstacles_in_check:
            continue
        normals.append((obj_name, xy))

    hidden: list[str] = []
    details: list[tuple[Any, ...]] = []
    r = float(overlap_radius)
    for name, obj, base_xy in bases:
        closest: tuple[str, float, np.ndarray] | None = None
        for other_name, other_xy in normals:
            d = float(np.linalg.norm(base_xy - other_xy))
            if closest is None or d < closest[1]:
                closest = (other_name, d, other_xy)
        if closest is None or closest[1] >= r:
            continue
        try:
            joint_name = obj.joints[-1]
            far = np.array([10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            sim.data.set_joint_qpos(joint_name, far)
            jid = sim.model.joint_name2id(joint_name)
            dof = sim.model.jnt_dofadr[jid]
            sim.data.qvel[dof : dof + 6] = 0.0
        except Exception:
            continue
        hidden.append(name)
        details.append((name, closest[0], closest[1], base_xy.copy(), closest[2].copy()))

    if hidden:
        mujoco.mj_forward(sim.model._model, sim.data._data)
    return details if return_details else hidden


def _as_optional_vec3(raw: Sequence[float] | None, *, name: str) -> np.ndarray | None:
    if raw is None:
        return None
    vals = [float(x) for x in raw]
    if not vals:
        return None
    if len(vals) != 3:
        raise ValueError(f"{name} must be length 3 or empty, got {raw!r}")
    return np.asarray(vals, dtype=np.float64)


def apply_hand_collision_box(
    handle: LiberoEnvHandle,
    *,
    half_size: Sequence[float] | None = None,
    collision_offset: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Override the hand collision box from YAML (this env instance only).

    Empty / omitted vectors keep the XML defaults. Both the contact box and
    the visual-only proxy are updated so ``hand.geometry: box`` stays aligned.
    """
    import mujoco

    hs = _as_optional_vec3(half_size, name="hand.half_size")
    off = _as_optional_vec3(collision_offset, name="hand.collision_offset")
    if hs is not None and np.any(hs <= 0.0):
        raise ValueError(f"hand.half_size must be > 0, got {half_size!r}")
    if hs is None and off is None:
        return {"n_geoms": 0}

    env = handle.env
    inner = env.env
    sim = inner.sim
    model = sim.model
    obj = inner.objects_dict.get(_HAND_OBJECT_NAME)
    if obj is None:
        return {"n_geoms": 0}

    try:
        root_body_id = int(model.body_name2id(obj.root_body))
    except Exception:
        return {"n_geoms": 0}

    body_ids: set[int] = {root_body_id}
    nbody = int(model.nbody)
    grew = True
    while grew:
        grew = False
        for child in range(nbody):
            parent = int(model.body_parentid[child])
            if parent in body_ids and child not in body_ids:
                body_ids.add(child)
                grew = True

    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    n_geoms = 0
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) not in body_ids:
            continue
        if int(model.geom_type[gid]) != box_type:
            continue
        if hs is not None:
            model.geom_size[gid] = hs
        if off is not None:
            model.geom_pos[gid] = off
        n_geoms += 1

    if n_geoms:
        mujoco.mj_forward(sim.model._model, sim.data._data)
    return {"n_geoms": n_geoms}


def configure_hand_mesh_visibility(handle: LiberoEnvHandle, *, geometry: str) -> None:
    """Show full hand mesh (``mesh``) or box proxy (``box``)."""
    import mujoco

    if geometry not in ("box", "mesh"):
        raise ValueError(f"hand geometry must be 'box' or 'mesh', got {geometry!r}")
    show_mesh = geometry == "mesh"

    env = handle.env
    inner = env.env
    sim = inner.sim
    model = sim.model

    obj = inner.objects_dict.get(_HAND_OBJECT_NAME)
    if obj is None:
        return

    try:
        root_body_id = int(model.body_name2id(obj.root_body))
    except Exception:
        return

    body_ids: set[int] = {root_body_id}
    nbody = int(model.nbody)
    grew = True
    while grew:
        grew = False
        for child in range(nbody):
            parent = int(model.body_parentid[child])
            if parent in body_ids and child not in body_ids:
                body_ids.add(child)
                grew = True

    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    box_rgba = np.asarray(_BOX_PROXY_RGBA, dtype=np.float32)

    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) not in body_ids:
            continue
        gtype = int(model.geom_type[gid])
        if gtype == mesh_type:
            model.geom_group[gid] = 1 if show_mesh else 0
        elif gtype == box_type:
            is_collision_box = bool(model.geom_contype[gid] or model.geom_conaffinity[gid])
            if is_collision_box:
                model.geom_group[gid] = 0
                model.geom_rgba[gid, 3] = 0.0
            elif show_mesh:
                model.geom_group[gid] = 0
                model.geom_rgba[gid, 3] = 0.0
            else:
                model.geom_group[gid] = 1
                model.geom_rgba[gid] = box_rgba

    mujoco.mj_forward(sim.model._model, sim.data._data)


def build_video_path(
    rollout_dir: Path,
    *,
    episode_idx: int,
    success: bool,
    task_description: str,
) -> Path:
    """Build an MP4 path; timestamp is taken at call time (D5=A)."""
    stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    processed = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    name = f"{stamp}--dynisafety--episode={episode_idx}--success={success}--task={processed}.mp4"
    return Path(rollout_dir) / name


def save_episode_video(
    frames: Sequence[np.ndarray],
    *,
    path: Path,
) -> None:
    """Write rollout frames to *path* as MP4 (fps=30)."""
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=30)
    try:
        for img in frames:
            writer.append_data(img)
    finally:
        writer.close()
