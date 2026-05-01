"""Environment helpers for DyniSafety-LIBERO.

Covers environment creation, observation pre-processing, video saving,
quaternion conversion, init-state padding, and obstacle visibility toggling.
"""

import math
import os
import time

import imageio
import mujoco
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Computed once at import time (same as the original robot_utils values).
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def get_dynisafety_env(task, model_family, resolution=256):
    """Initializes and returns the DyniSafety-LIBERO environment."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def get_dynisafety_dummy_action(model_family: str):
    """No-op action used during the stabilisation phase."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_dynisafety_image(obs):
    """Third-person camera image, rotated 180° to match training preprocessing."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    return img


def get_dynisafety_wrist_image(obs):
    """Wrist camera image, rotated 180° to match training preprocessing."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]
    return img


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def save_rollout_video(rollout_images, idx, success, task_description,
                       log_file=None, rollout_dir=None):
    """Saves an MP4 replay of an episode."""
    if rollout_dir is None:
        rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--dynisafety--episode={idx}"
        f"--success={success}--task={processed_task_description}.mp4"
    )
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


# ---------------------------------------------------------------------------
# Quaternion → axis-angle
# ---------------------------------------------------------------------------

def quat2axisangle(quat):
    """Converts quaternion (x,y,z,w) to axis-angle."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ---------------------------------------------------------------------------
# Init-state padding (hand body adds 7 qpos + 6 qvel)
# ---------------------------------------------------------------------------

def pad_init_state_for_hand(init_state, env, hand_name="hand_pusher_1"):
    """Pad a saved init_state to account for the hand body's extra DOFs."""
    inner = env.env
    sim = inner.sim
    expected_len = 1 + sim.model.nq + sim.model.nv

    if len(init_state) == expected_len:
        return init_state

    if hand_name not in inner.objects_dict:
        return np.array(init_state, dtype=np.float64)

    obj = inner.objects_dict[hand_name]
    jid = sim.model.joint_name2id(obj.joints[-1])
    qpos_addr = sim.model.jnt_qposadr[jid]
    dof_addr = sim.model.jnt_dofadr[jid]

    nq_orig = sim.model.nq - 7
    nv_orig = sim.model.nv - 6  # noqa: F841

    hand_qpos = np.array([10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    hand_qvel = np.zeros(6)

    state = np.array(init_state, dtype=np.float64)
    padded_qpos = np.insert(state[:1 + nq_orig], 1 + qpos_addr, hand_qpos)
    padded_qvel = np.insert(state[1 + nq_orig:], dof_addr, hand_qvel)
    return np.concatenate([padded_qpos, padded_qvel])


# ---------------------------------------------------------------------------
# Obstacle visibility
# ---------------------------------------------------------------------------

def rescale_obstacle_inplace(env, obstacle_name: str, scale: float):
    """Scale a single obstacle's collision geoms and visual mesh by ``scale``.

    Modifies MuJoCo's in-memory model arrays so the change is local to
    the currently-loaded env instance — i.e. it only affects the task
    whose env this is.  The original asset XML and any other tasks that
    share the same obstacle type are untouched.

    What gets scaled:
      * ``model.geom_pos[gid]`` and ``model.geom_size[gid]`` for every
        geom under the obstacle's body subtree (collision boxes,
        spheres, cylinders, the visual-mesh wrapper geom — all of them).
      * ``model.mesh_vert`` of every mesh asset that's *exclusively*
        used by this obstacle's geoms (so we don't accidentally shrink
        another obstacle that happens to share the same mesh asset —
        unlikely in SafeLIBERO but cheap to check).

    The body's free-joint qpos is **not** modified, so the body origin
    stays at its current world position.  After scaling the obstacle's
    bottom geometry sits ~``(1-scale) * original_bottom_offset`` above
    its support; physics' first ``mj_step`` settles that small gap
    naturally (typically < 1 cm for a 0.9× scale).

    Args:
        env: DyniSafety wrapper (``env.env`` is the inner robosuite env).
        obstacle_name: key in ``env.env.objects_dict`` (e.g.
            ``"milk_obstacle_1"``).
        scale: positive scalar (1.0 = no-op).

    Returns:
        Dict with diagnostic counts: ``{"n_geoms": int, "n_meshes": int,
        "body_ids": list[int]}``.  Returns zero counts if the obstacle
        is not in the scene or scale is 1.0.
    """
    inner = env.env
    sim = inner.sim
    if scale == 1.0:
        return {"n_geoms": 0, "n_meshes": 0, "body_ids": []}
    obj = inner.objects_dict.get(obstacle_name)
    if obj is None:
        return {"n_geoms": 0, "n_meshes": 0, "body_ids": []}

    # 1. Collect all bodies under the obstacle's root (BFS).
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

    # 2. Scale geom-local pos and size.
    geom_ids = [
        gid for gid in range(int(sim.model.ngeom))
        if int(sim.model.geom_bodyid[gid]) in body_ids
    ]
    for gid in geom_ids:
        sim.model.geom_pos[gid] *= scale
        sim.model.geom_size[gid] *= scale

    # 3. Scale mesh vertices for any mesh asset exclusively used by
    #    these geoms (so we don't inadvertently shrink other props
    #    that happen to share the asset).
    mesh_geom_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    mesh_ids_used = set()
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
        sim.model.mesh_vert[start:start + nverts] *= scale
        n_meshes_scaled += 1

    mujoco.mj_forward(sim.model._model, sim.data._data)
    return {
        "n_geoms": len(geom_ids),
        "n_meshes": n_meshes_scaled,
        "body_ids": sorted(body_ids),
    }


def hide_all_obstacles(env, include_bases: bool = False):
    """Teleport every obstacle (and optionally pedestal) far off-screen.

    By default this function only hides ``obstacle``-suffixed entries.
    Decorative ``box_base`` / ``box_small_base`` pedestals are deliberately
    left alone so that callers can apply the smarter ``hide_overlapping_bases``
    policy (keep pedestals when they don't visibly clip with other props).

    Set ``include_bases=True`` for the legacy "hide everything" behaviour.
    """
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
            sim.data.qvel[dof:dof + 6] = 0.0
        except Exception:
            pass
    mujoco.mj_forward(sim.model._model, sim.data._data)


# ---------------------------------------------------------------------------
# Pedestal / base helpers
# ---------------------------------------------------------------------------

def _is_base_name(name: str) -> bool:
    """Whether a SafeLIBERO ``objects_dict`` entry is a decorative pedestal."""
    nl = name.lower()
    return ("box_base" in nl or "box_small_base" in nl) and "hand" not in nl


def _is_obstacle_name(name: str) -> bool:
    """Whether a SafeLIBERO ``objects_dict`` entry is a dynamic-style obstacle."""
    nl = name.lower()
    return "obstacle" in nl and "hand" not in nl


def _is_visible_xy(pos):
    """SafeLIBERO hides inactive props at |x|>=5 or |y|>=5."""
    return abs(float(pos[0])) < 5.0 and abs(float(pos[1])) < 5.0


def _root_body_xy(sim, obj):
    """World XY of an object's root body (cheap, no qpos lookup needed)."""
    bid = sim.model.body_name2id(obj.root_body)
    return sim.data.body_xpos[bid][:2].copy()


def hide_overlapping_bases(env, overlap_radius: float = 0.10,
                           exclude_names=(),
                           include_obstacles_in_check: bool = False,
                           return_details: bool = False):
    """Hide ``box_base`` / ``box_small_base`` pedestals that visibly clip with
    *normal* (non-obstacle, non-base) scene objects.

    SafeLIBERO scenes intentionally co-locate a pedestal with one of its
    obstacles (e.g. ``box_small_base_1`` exactly under
    ``wine_bottle_obstacle_1``) so that the obstacle visually rests on
    top of the pedestal.  The pedestal sits at floor Z and the obstacle
    above it, so even when their XY footprints coincide the rendered
    image looks correct — that is *useful* clutter for safety evaluation
    and must be preserved.

    The visually broken case is when ``pruned_init`` places the pedestal
    at the same XY as a *normal* (i.e. task-relevant) object such as
    ``chocolate_pudding_1`` — there's no Z separation by design, and the
    pedestal ends up clipping into the task target.  Hide only that case.

    Args:
        env: DyniSafety / SafeLIBERO env wrapper (``env.env`` is the inner
             ``OffScreenRenderEnv``).
        overlap_radius: XY distance (metres) below which two objects are
                        considered overlapping.  Defaults to 10 cm — large
                        enough to catch the typical "pedestal directly
                        under target" pattern without spuriously hiding
                        pedestals that just happen to sit near another
                        prop.
        exclude_names: Iterable of object names that should NOT be
                       considered when checking for overlap (e.g. the
                       chosen dynamic obstacle, which will soon be flown
                       in to its start position and is not at its
                       ``pruned_init`` XY anyway).
        include_obstacles_in_check: If *True*, also count obstacle props
            (names containing ``obstacle``) as potential overlap
            partners.  By default obstacles are excluded so that the
            "obstacle resting on its pedestal" SafeLIBERO layout is
            preserved.
        return_details: If *True*, return a list of
            ``(base_name, conflict_name, distance, base_xy, conflict_xy)``
            tuples describing each hide decision.  Otherwise (default),
            only return the list of hidden pedestal names — preserves
            backward-compatibility with existing callers.

    Returns:
        Either a list of hidden pedestal names, or — when
        ``return_details=True`` — a list of
        ``(base_name, conflict_name, distance, base_xy, conflict_xy)``
        tuples.
    """
    inner = env.env
    sim = inner.sim
    exclude = {str(n) for n in exclude_names}

    bases = []
    normals = []
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
            # Skip — obstacles are *meant* to sit on pedestals.
            continue
        normals.append((obj_name, xy))

    hidden = []
    details = []
    r = float(overlap_radius)
    for name, obj, base_xy in bases:
        # Pick the closest normal-object conflict so the diagnostic
        # message names the true culprit, not just the first match.
        closest = None
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
            sim.data.qvel[dof:dof + 6] = 0.0
        except Exception:
            continue
        hidden.append(name)
        details.append(
            (name, closest[0], closest[1], base_xy.copy(), closest[2].copy())
        )

    if hidden:
        mujoco.mj_forward(sim.model._model, sim.data._data)
    return details if return_details else hidden
