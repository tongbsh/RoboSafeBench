"""RoboCasa NavigateKitchen env helpers (Window C L0 + Window D L1 merge).

No metric aggregation. L2/L3 disturbance controllers stay out of this module.
No Python defaults for camera keys — callers must pass YAML values.
"""

from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.core.protocols import TaskSpec

# ---------------------------------------------------------------------------
# 12-D layout (identity with robosuite HybridMobileBase for PandaOmron)
# ---------------------------------------------------------------------------
# Confirmed from source (do NOT silently swap):
# 1) Kitchen.__init__ sets body_part_ordering =
#    ["right", "right_gripper", "base", "torso"]
#    (third_party/robocasa/.../kitchen.py) and HybridMobileBase appends
#    one extra dim (all_action[-1] = base_mode).
# 2) robocasa.utils.env_utils.convert_action (gym dict view of the same 12-D):
#    [0:3] EE pos, [3:6] EE rot, [6:7] gripper,
#    [7:11] base_motion = base(3)+torso(1), [11:12] control_mode.
# 3) gym PandaOmronKeyConverter.unmap_action:
#    robot0_right = ee pos||rot (6), robot0_right_gripper (1),
#    robot0_base = base_motion[:3], robot0_torso = base_motion[3:4],
#    robot0_base_mode = last dim.
# Therefore hybrid_mobile_12d numpy == robosuite env.step vector. Identity.
HYBRID_MOBILE_12D_DIM = 12
HYBRID_MOBILE_12D_SLICES: dict[str, tuple[int, int]] = {
    "right": (0, 6),
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "right_gripper": (6, 7),
    "gripper_close": (6, 7),
    "base": (7, 10),
    "torso": (10, 11),
    "base_motion": (7, 11),
    "base_mode": (11, 12),
    "control_mode": (11, 12),
}
EXPECTED_BODY_PART_ORDERING: tuple[str, ...] = (
    "right",
    "right_gripper",
    "base",
    "torso",
)

# NavigateKitchen.valid_* class pools (kitchen_navigate.py). Pin by class.
NAV_VALID_SRC_CLASSES: frozenset[str] = frozenset(
    {
        "CoffeeMachine",
        "Toaster",
        "ToasterOven",
        "Stove",
        "Stovetop",
        "SingleCabinet",
        "HingeCabinet",
        "OpenCabinet",
        "Drawer",
        "Microwave",
        "Sink",
        "Hood",
        "Oven",
        "FridgeFrenchDoor",
        "FridgeSideBySide",
        "FridgeBottomFreezer",
        "Dishwasher",
        "ElectricKettle",
        "StandMixer",
    }
)
NAV_VALID_TARGET_CLASSES: frozenset[str] = frozenset(
    {
        "CoffeeMachine",
        "Toaster",
        "ToasterOven",
        "Stove",
        "Stovetop",
        "OpenCabinet",
        "Microwave",
        "Sink",
        "Hood",
        "Oven",
        "FridgeFrenchDoor",
        "FridgeSideBySide",
        "FridgeBottomFreezer",
        "Dishwasher",
        "ElectricKettle",
        "StandMixer",
    }
)
NAV_MIN_SRC_TARGET_DIST = 1.0
NAV_FRIDGE_CLASSES: tuple[str, ...] = (
    "FridgeFrenchDoor",
    "FridgeSideBySide",
    "FridgeBottomFreezer",
)
HIDDEN_OBSTACLE_BODY = "dynisafety_l12_obstacle"
HIDDEN_L3_BODY = "dynisafety_l3_human"
HIDDEN_OBSTACLE_POS = (0.0, 0.0, -10.0)
HIDDEN_OBSTACLE_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
_WINDOW_C_FIX = "FIX：完成窗口 C"
_WINDOW_DE_FIX = "FIX：窗口 D/E"
_WINDOW_E_FIX = "FIX：窗口 E"

# Wait-phase no-op: 12-D zeros, base_mode=-1 (arm mode). Not a 7-D pad.
WAIT_PHASE_ACTION_12D: list[float] = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0,
]


def _require_str(mapping: Mapping[str, Any], key: str, *, where: str) -> str:
    if key not in mapping:
        raise ValueError(f"{where} missing {key!r}; set it in YAML ({_WINDOW_C_FIX})")
    val = mapping[key]
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{where}.{key} must be a non-empty str, got {val!r} ({_WINDOW_C_FIX})")
    return val.strip()


def _require_int(mapping: Mapping[str, Any], key: str, *, where: str) -> int:
    if key not in mapping:
        raise ValueError(f"{where} missing {key!r}; set it in YAML ({_WINDOW_C_FIX})")
    val = mapping[key]
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{where}.{key} must be int, got {val!r} ({_WINDOW_C_FIX})")
    return int(val)


def names_of_class(class_by_name: Mapping[str, str], class_name: str) -> list[str]:
    """Return fixture instance names whose class equals *class_name*."""
    return [name for name, cls in class_by_name.items() if cls == class_name]


def resolve_pinned_target_name(
    class_by_name: Mapping[str, str],
    target_class: str,
) -> str:
    """Pin destination to exactly one instance of *target_class*.

    Layout may vary; the destination fixture *family* must not.
    """
    if target_class not in NAV_VALID_TARGET_CLASSES:
        raise ValueError(
            f"target_fixture class {target_class!r} is not a NavigateKitchen "
            f"valid target (no PickPlace; no class 'Fridge'). {_WINDOW_C_FIX}"
        )
    names = names_of_class(class_by_name, target_class)
    if len(names) == 1:
        return names[0]
    if target_class in NAV_FRIDGE_CLASSES:
        for alt in NAV_FRIDGE_CLASSES:
            if alt == target_class:
                continue
            alt_names = names_of_class(class_by_name, alt)
            if len(alt_names) == 1:
                return alt_names[0]
    raise ValueError(
        f"target_fixture class {target_class!r} must appear exactly once "
        f"in the scene, got {len(names)}: {names}. {_WINDOW_C_FIX}"
    )


def candidate_src_names(
    class_by_name: Mapping[str, str],
    *,
    target_name: str,
    pairwise_dist: Mapping[tuple[str, str], float],
    min_dist: float = NAV_MIN_SRC_TARGET_DIST,
) -> list[str]:
    """Valid NavigateKitchen src fixtures far enough from the pinned target."""
    out: list[str] = []
    for name, cls in class_by_name.items():
        if name == target_name:
            continue
        if cls not in NAV_VALID_SRC_CLASSES:
            continue
        dist = pairwise_dist.get((name, target_name))
        if dist is None:
            dist = pairwise_dist.get((target_name, name))
        if dist is None:
            continue
        if float(dist) <= float(min_dist):
            continue
        out.append(name)
    return out


def convert_action_layout_dict(action12: np.ndarray) -> dict[str, np.ndarray]:
    """Mirror ``robocasa.utils.env_utils.convert_action`` (no import)."""
    action = np.asarray(action12, dtype=np.float64).reshape(HYBRID_MOBILE_12D_DIM)
    return {
        "action.end_effector_position": action[0:3],
        "action.end_effector_rotation": action[3:6],
        "action.gripper_close": action[6:7],
        "action.base_motion": action[7:11],
        "action.control_mode": action[11:12],
    }


def to_robosuite_action(action12: np.ndarray) -> np.ndarray:
    """Identity: hybrid_mobile_12d is the robosuite HybridMobileBase vector.

    Raises if the vector is not 12-D. Never pads 7-D with zeros.
    """
    vec = np.asarray(action12, dtype=np.float64).reshape(-1)
    if vec.shape != (HYBRID_MOBILE_12D_DIM,):
        raise ValueError(
            f"hybrid_mobile_12d action must have shape ({HYBRID_MOBILE_12D_DIM},), "
            f"got {vec.shape}. 7-D pad-with-zeros is forbidden. {_WINDOW_C_FIX}"
        )
    return vec.copy()


def dummy_action_12d() -> np.ndarray:
    return np.asarray(WAIT_PHASE_ACTION_12D, dtype=np.float64)


@dataclass
class RoboCasaEnvHandle:
    """Concrete ``EnvHandle`` for a robosuite ``NavigateKitchen`` env."""

    env: Any
    task_description: str
    suite: str
    task_id: int
    task_name: str
    camera: str
    target_fixture_class: str
    extra: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        env = self.env
        close = getattr(env, "close", None)
        if callable(close):
            close()


def resolve_robocasa_root(
    *,
    sim_root: str,
    workspace: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the RoboCasa checkout. *sim_root* comes from YAML (no default)."""
    del env  # reserved; YAML sim_root is the contract
    root = Path(sim_root)
    if not root.is_absolute() and workspace is not None:
        root = workspace / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"extra.robocasa.sim_root is not a directory: {root} ({_WINDOW_C_FIX})"
        )
    if not (root / "robocasa").is_dir():
        raise FileNotFoundError(f"sim_root {root} has no robocasa/ package ({_WINDOW_C_FIX})")
    return root


def ensure_robocasa_sys_path(
    *,
    sim_root: str,
    workspace: Path | None = None,
) -> Path:
    """Put third_party/robocasa (and sibling robosuite) on ``sys.path``."""
    root = resolve_robocasa_root(sim_root=sim_root, workspace=workspace)
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    sibling = root.parent / "robosuite"
    if sibling.is_dir():
        sib = str(sibling.resolve())
        if sib not in sys.path:
            sys.path.insert(0, sib)
    return root


def get_primary_image(obs: Mapping[str, Any], *, camera: str) -> np.ndarray:
    """Image for YAML camera key. Vertical flip matches robocasa gym wrapper."""
    key = f"{camera}_image"
    if key not in obs:
        raise KeyError(
            f"observation missing {key!r}; extra.robocasa.camera={camera!r} "
            f"must be in create_env camera_names. {_WINDOW_C_FIX}"
        )
    img = np.asarray(obs[key])
    return np.copy(img[::-1, :, :])


def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Match Xiaomi ``eval_robocasa365/entry.py`` (xyzw → axis-angle)."""
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def robocasa365_state_14d(obs: Mapping[str, Any]) -> np.ndarray:
    """EE-first 14D state used by Xiaomi-Robotics-1 RoboCasa365.

    Keys are robosuite PandaOmron names; layout matches
    ``PandaOmronKeyConverter.map_obs`` + Xiaomi ``observation_to_state``.
    """
    required = (
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_gripper_qpos",
        "robot0_base_pos",
        "robot0_base_quat",
    )
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(
            f"robocasa365 14D state missing {missing}; have {sorted(obs)}. {_WINDOW_C_FIX}"
        )
    state = np.concatenate(
        [
            np.asarray(obs["robot0_base_to_eef_pos"], dtype=np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(obs["robot0_base_to_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_base_pos"], dtype=np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(obs["robot0_base_quat"]),
        ]
    ).astype(np.float32)
    if state.shape != (14,):
        raise ValueError(f"Expected a 14D RoboCasa365 state, got {state.shape}")
    return state


def read_base_xy_yaw(env: Any) -> tuple[np.ndarray, float]:
    """XY of ``mobilebase0_base`` plus yaw (rad) from body xmat."""
    import robosuite.utils.transform_utils as T

    robot_id = env.sim.model.body_name2id("mobilebase0_base")
    xy = np.asarray(env.sim.data.body_xpos[robot_id][:2], dtype=np.float64).copy()
    ori = T.mat2euler(np.asarray(env.sim.data.body_xmat[robot_id]).reshape((3, 3)))
    return xy, float(ori[2])


def read_nav_dest_xy(env: Any) -> np.ndarray:
    """NavigateKitchen completion XY (``target_pos`` after fixture pin)."""
    target = getattr(env, "target_pos", None)
    if target is None:
        raise RuntimeError(
            f"env.target_pos missing; pin the NavigateKitchen destination first ({_WINDOW_DE_FIX})"
        )
    arr = np.asarray(target, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise RuntimeError(f"env.target_pos has no XY, got {target!r} ({_WINDOW_DE_FIX})")
    return arr[:2].copy()


def parse_obstacle_rgba(raw: Any, *, where: str = "disturbance.obstacle.color") -> list[float]:
    """RGB or RGBA in [0, 1] from YAML (obstacle color or L3 mesh)."""
    if raw is None:
        raise ValueError(f"{where} is required in YAML ({_WINDOW_C_FIX})")
    vals = [float(x) for x in raw]
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        raise ValueError(f"{where} must be RGB or RGBA, got {list(raw)!r} ({_WINDOW_C_FIX})")
    if any(c < 0.0 or c > 1.0 for c in vals):
        raise ValueError(f"{where} channels must be in [0, 1], got {vals!r}")
    return vals


def read_base_joint_qpos(env: Any) -> dict[str, float]:
    """Named mobile-base joint qpos (proof the chassis is not frozen)."""
    out: dict[str, float] = {}
    model = env.sim.model
    names = list(getattr(model, "joint_names", []) or [])
    for name in names:
        if "mobile" not in str(name).lower() and "base" not in str(name).lower():
            continue
        try:
            addr = model.get_joint_qpos_addr(name)
        except Exception:
            continue
        if isinstance(addr, tuple):
            start, end = int(addr[0]), int(addr[1])
            q = np.asarray(env.sim.data.qpos[start:end], dtype=np.float64)
            out[str(name)] = float(q[0]) if q.size else 0.0
        else:
            out[str(name)] = float(env.sim.data.qpos[int(addr)])
    return out


def action_split_indexes(env: Any) -> dict[str, tuple[int, int]]:
    robot = env.robots[0]
    cc = robot.composite_controller
    raw = dict(cc._action_split_indexes)
    dim = int(np.asarray(cc.action_limits[0]).shape[0])
    raw["base_mode"] = (dim - 1, dim)
    return {str(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


def assert_hybrid_mobile_12d_layout(env: Any) -> dict[str, tuple[int, int]]:
    """Fail loudly if PandaOmron split indexes are not the documented 12-D.

    Never remaps / swaps dims to 'make it work'.
    """
    robot = env.robots[0]
    cc = robot.composite_controller
    low, _high = cc.action_limits
    dim = int(np.asarray(low).shape[0])
    if dim != HYBRID_MOBILE_12D_DIM:
        raise RuntimeError(
            f"PandaOmron HybridMobileBase action_dim={dim}, expected "
            f"{HYBRID_MOBILE_12D_DIM}. {_WINDOW_C_FIX}"
        )
    splits = {str(k): (int(v[0]), int(v[1])) for k, v in cc._action_split_indexes.items()}
    expected = {
        "right": (0, 6),
        "right_gripper": (6, 7),
        "base": (7, 10),
        "torso": (10, 11),
    }
    if splits != expected:
        raise RuntimeError(
            f"HybridMobileBase _action_split_indexes={splits} != {expected}. "
            f"Refusing to silently swap dims. {_WINDOW_C_FIX}"
        )
    return action_split_indexes(env)


def box_density_from_mass(half_size: Sequence[float], mass: float) -> float:
    """density = mass / (8 * hx * hy * hz). YAML mass is the contract."""
    hx, hy, hz = (float(half_size[0]), float(half_size[1]), float(half_size[2]))
    volume = 8.0 * hx * hy * hz
    if volume <= 0:
        raise ValueError(f"obstacle volume must be > 0, got half_size={half_size!r}")
    mass_f = float(mass)
    if mass_f <= 0:
        raise ValueError(f"disturbance.obstacle.mass must be > 0, got {mass_f}")
    return mass_f / volume


def _apply_body_rgba(env: Any, body_name: str, rgba: Sequence[float]) -> None:
    model = env.sim.model
    raw = getattr(model, "_model", model)
    if not hasattr(raw, "geom_rgba"):
        return
    body_id = int(model.body_name2id(str(body_name)))
    r, g, b, a = (float(x) for x in rgba)
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        raw.geom_rgba[gid, 0] = r
        raw.geom_rgba[gid, 1] = g
        raw.geom_rgba[gid, 2] = b
        raw.geom_rgba[gid, 3] = a


def _attach_l12_visuals(box: Any, env: Any, meta: dict[str, Any]) -> None:
    visuals = getattr(env, "_dynisafety_l12_visuals", None)
    if not visuals:
        return
    from dynisafety.backends.robocasa.visuals import L12_AUTHOR_HALF_XY, attach_visual_specs

    hs = [float(x) for x in meta["half_size"]]
    xy_scale = min(hs[0], hs[1]) / L12_AUTHOR_HALF_XY
    mapping = attach_visual_specs(
        box, visuals, half_z=hs[2], xy_scale=xy_scale, visible_names=()
    )
    meta["visual_geoms"] = mapping
    meta["visual_rgba"] = {spec.name: list(spec.rgba) for spec in visuals}
    meta["visual_kind"] = "l12_pool"


def merge_l12_floor_obstacle(
    env: Any,
    *,
    half_size: Sequence[float],
    mass: float,
    rgba: Sequence[float],
) -> dict[str, Any]:
    """Merge the L1/L2 shared free-joint floor box into ``env.model``.

    Idempotent across Kitchen._load_model retries.
    """
    color = parse_obstacle_rgba(rgba)
    visuals = getattr(env, "_dynisafety_l12_visuals", None)
    box_rgba = [0.0, 0.0, 0.0, 0.0] if visuals else color
    existing = getattr(env, "objects", None)
    if isinstance(existing, dict) and HIDDEN_OBSTACLE_BODY in existing:
        obj = existing[HIDDEN_OBSTACLE_BODY]
        meta = {
            "object_name": HIDDEN_OBSTACLE_BODY,
            "body_name": str(obj.root_body),
            "joint_name": str(obj.joints[0]),
            "half_size": [float(x) for x in half_size],
            "mass": float(mass),
            "rgba": color,
            "merge": "model.merge_objects",
        }
        _attach_l12_visuals(obj, env, meta)
        env._dynisafety_l12 = meta
        if not visuals:
            try:
                _apply_body_rgba(env, str(obj.root_body), color)
            except Exception:
                pass
        return meta

    from robosuite.models.objects.primitive.box import BoxObject
    from robosuite.utils.mjcf_utils import array_to_string

    hs = [float(x) for x in half_size]
    if len(hs) != 3:
        raise ValueError(f"disturbance.obstacle.half_size must be length 3, got {hs!r}")
    density = box_density_from_mass(hs, mass)
    box = BoxObject(
        name=HIDDEN_OBSTACLE_BODY,
        size=hs,
        density=density,
        rgba=box_rgba,
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=True,
    )
    box.get_obj().set("pos", array_to_string(list(HIDDEN_OBSTACLE_POS)))
    meta = {
        "object_name": str(box.name),
        "body_name": str(box.root_body),
        "joint_name": str(box.joints[0]),
        "half_size": hs,
        "mass": float(mass),
        "density": float(density),
        "rgba": color,
        "merge": "DyniSafetyNavigateKitchen._load_model + model.merge_objects",
    }
    _attach_l12_visuals(box, env, meta)
    if not hasattr(env, "objects") or env.objects is None:
        env.objects = {}
    env.objects[box.name] = box
    env.model.merge_objects([box])
    if not hasattr(env, "object_placements") or env.object_placements is None:
        env.object_placements = {}
    env.object_placements[box.name] = (
        np.array(HIDDEN_OBSTACLE_POS, dtype=np.float64),
        np.array(HIDDEN_OBSTACLE_QUAT_WXYZ, dtype=np.float64),
        box,
    )
    env._dynisafety_l12 = meta
    return meta


def _l3_mesh_meta(
    *,
    half_size: Sequence[float],
    mass: float,
    mesh_path: Path,
    mesh_rgba: Sequence[float],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "half_size": [float(x) for x in half_size],
        "mass": float(mass),
        "kind": "pedestrian_proxy",
        "visual": "clothed_standing_mesh",
        "mesh_path": str(mesh_path),
        "mesh_rgba": [float(x) for x in mesh_rgba],
    }
    if extra:
        meta.update(dict(extra))
    return meta


def _attach_l3_parts(box: Any, env: Any, meta: dict[str, Any], half_z: float) -> None:
    from dynisafety.backends.robocasa.visuals import attach_visual_specs

    parts = getattr(env, "_dynisafety_l3_visuals", None)
    if not parts:
        raise ValueError("L3 clothed visual parts missing on env")
    mapping = attach_visual_specs(box, parts, half_z=half_z)
    meta["visual_geoms"] = mapping
    meta["visual_parts"] = [spec.name for spec in parts]


def merge_l3_pedestrian_proxy(
    env: Any,
    *,
    half_size: Sequence[float],
    mass: float,
    mesh_path: str | Path,
    mesh_rgba: Sequence[float],
) -> dict[str, Any]:
    """Merge the L3 floor pedestrian: box collision + clothed visual parts.

    Idempotent across Kitchen._load_model retries. Missing mesh is an error
    (no silent fallback to the blue box).
    """
    hs = [float(x) for x in half_size]
    if len(hs) != 3:
        raise ValueError(f"extra.robocasa.l3.half_size must be length 3, got {hs!r}")
    color = parse_obstacle_rgba(mesh_rgba, where="extra.robocasa.l3.mesh_rgba")
    resolved_mesh = Path(mesh_path).resolve()
    if not resolved_mesh.is_file():
        raise FileNotFoundError(f"L3 pedestrian mesh missing: {resolved_mesh}")

    existing = getattr(env, "objects", None)
    if isinstance(existing, dict) and HIDDEN_L3_BODY in existing:
        obj = existing[HIDDEN_L3_BODY]
        meta = _l3_mesh_meta(
            half_size=hs,
            mass=mass,
            mesh_path=resolved_mesh,
            mesh_rgba=color,
            extra={
                "object_name": HIDDEN_L3_BODY,
                "body_name": str(obj.root_body),
                "joint_name": str(obj.joints[0]),
                "merge": "model.merge_objects",
            },
        )
        _attach_l3_parts(obj, env, meta, hs[2])
        env._dynisafety_l3 = meta
        return meta

    from robosuite.models.objects.primitive.box import BoxObject
    from robosuite.utils.mjcf_utils import array_to_string

    density = box_density_from_mass(hs, mass)
    box = BoxObject(
        name=HIDDEN_L3_BODY,
        size=hs,
        density=density,
        rgba=[0.0, 0.0, 0.0, 0.0],
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=True,
    )
    box.get_obj().set("pos", array_to_string(list(HIDDEN_OBSTACLE_POS)))
    meta = _l3_mesh_meta(
        half_size=hs,
        mass=mass,
        mesh_path=resolved_mesh,
        mesh_rgba=color,
        extra={
            "object_name": str(box.name),
            "body_name": str(box.root_body),
            "joint_name": str(box.joints[0]),
            "density": float(density),
            "merge": "DyniSafetyNavigateKitchen._load_model + model.merge_objects",
        },
    )
    _attach_l3_parts(box, env, meta, hs[2])
    if not hasattr(env, "objects") or env.objects is None:
        env.objects = {}
    env.objects[box.name] = box
    env.model.merge_objects([box])
    if not hasattr(env, "object_placements") or env.object_placements is None:
        env.object_placements = {}
    env.object_placements[box.name] = (
        np.array(HIDDEN_OBSTACLE_POS, dtype=np.float64),
        np.array(HIDDEN_OBSTACLE_QUAT_WXYZ, dtype=np.float64),
        box,
    )
    env._dynisafety_l3 = meta
    return meta


def _public_l12_meta(meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """JSON-safe merge metadata (drop per-geom contact templates)."""
    if not isinstance(meta, Mapping):
        return None
    out: dict[str, Any] = {}
    for key, val in meta.items():
        if key == "geom_contact":
            continue
        out[str(key)] = val
    return out


def install_l12_navigate_kitchen_wrapper(
    *,
    half_size: Sequence[float],
    mass: float,
    rgba: Sequence[float],
    l3_half_size: Sequence[float] | None = None,
    l3_mass: float | None = None,
    l3_mesh_path: str | Path | None = None,
    l3_mesh_rgba: Sequence[float] | None = None,
    l12_visuals: Sequence[Any] | None = None,
    l3_visuals: Sequence[Any] | None = None,
) -> type:
    """Register a NavigateKitchen subclass that merges the free-joint floor box.

    YAML ``env_class`` stays ``NavigateKitchen``; we overwrite REGISTERED_ENVS.
    Optional L3 pedestrian proxy is merged only when *l3_half_size* is given
    (None means skip — not a geometry default).
    """
    from robocasa.environments.kitchen.atomic.kitchen_navigate import NavigateKitchen
    from robosuite.environments.base import REGISTERED_ENVS

    hs = tuple(float(x) for x in half_size)
    mass_f = float(mass)
    color = tuple(parse_obstacle_rgba(rgba))
    l3_hs = None if l3_half_size is None else tuple(float(x) for x in l3_half_size)
    l3_mass_f = None if l3_mass is None else float(l3_mass)
    l3_mesh = None if l3_mesh_path is None else str(l3_mesh_path)
    l3_mesh_color = (
        None if l3_mesh_rgba is None else tuple(parse_obstacle_rgba(l3_mesh_rgba, where="extra.robocasa.l3.mesh_rgba"))
    )
    l12_vis = tuple(l12_visuals) if l12_visuals else ()
    l3_vis = tuple(l3_visuals) if l3_visuals else ()
    current = REGISTERED_ENVS.get("NavigateKitchen")
    if getattr(current, "_dynisafety_l12_wrapper", False):
        current._obstacle_half_size = hs
        current._obstacle_mass = mass_f
        current._obstacle_rgba = color
        current._l3_half_size = l3_hs
        current._l3_mass = l3_mass_f
        current._l3_mesh_path = l3_mesh
        current._l3_mesh_rgba = l3_mesh_color
        current._l12_visuals = l12_vis
        current._l3_visuals = l3_vis
        return current

    class DyniSafetyNavigateKitchen(NavigateKitchen):
        _dynisafety_l12_wrapper = True
        _obstacle_half_size = hs
        _obstacle_mass = mass_f
        _obstacle_rgba = color
        _l3_half_size = l3_hs
        _l3_mass = l3_mass_f
        _l3_mesh_path = l3_mesh
        _l3_mesh_rgba = l3_mesh_color
        _l12_visuals = l12_vis
        _l3_visuals = l3_vis

        def _load_model(self, attempt_num: int = 1) -> None:
            super()._load_model(attempt_num=attempt_num)
            self._dynisafety_l12_visuals = type(self)._l12_visuals
            self._dynisafety_l3_visuals = type(self)._l3_visuals
            merge_l12_floor_obstacle(
                self,
                half_size=type(self)._obstacle_half_size,
                mass=type(self)._obstacle_mass,
                rgba=type(self)._obstacle_rgba,
            )
            l3_size = type(self)._l3_half_size
            l3_m = type(self)._l3_mass
            if l3_size is not None and l3_m is not None:
                mesh_path = type(self)._l3_mesh_path
                mesh_rgba = type(self)._l3_mesh_rgba
                if not mesh_path or mesh_rgba is None:
                    raise ValueError(
                        "extra.robocasa.l3.mesh_path and mesh_rgba "
                        f"are required for L3 ({_WINDOW_E_FIX})"
                    )
                merge_l3_pedestrian_proxy(
                    self,
                    half_size=l3_size,
                    mass=l3_m,
                    mesh_path=mesh_path,
                    mesh_rgba=mesh_rgba,
                )

    REGISTERED_ENVS["NavigateKitchen"] = DyniSafetyNavigateKitchen
    return DyniSafetyNavigateKitchen


def inject_hidden_obstacle_xml(
    xml_str: str,
    *,
    half_size: Sequence[float],
    mass: float,
) -> str:
    """Window C mocap embed. Unused after Window D free-joint merge."""
    if HIDDEN_OBSTACLE_BODY in xml_str:
        return xml_str
    if len(half_size) != 3:
        raise ValueError(f"disturbance.obstacle.half_size must be length 3, got {half_size!r}")
    sx, sy, sz = (float(half_size[0]), float(half_size[1]), float(half_size[2]))
    mass_f = float(mass)
    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF missing worldbody ({_WINDOW_C_FIX})")
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": HIDDEN_OBSTACLE_BODY,
            "pos": "0 0 -10",
            "mocap": "true",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{HIDDEN_OBSTACLE_BODY}_geom",
            "type": "box",
            "size": f"{sx} {sy} {sz}",
            "rgba": "0 0 0 0",
            "contype": "0",
            "conaffinity": "0",
            "group": "3",
            "mass": f"{mass_f}",
        },
    )
    return ET.tostring(root, encoding="utf8").decode("utf8")


def install_hidden_obstacle_xml_processor(
    env: Any,
    *,
    half_size: Sequence[float],
    mass: float,
) -> None:
    """Patch ``_xml_processors`` so every hard_reset re-embeds the hidden actor."""
    original = env.edit_model_xml
    orig_func = getattr(original, "__func__", original)

    def _wrapped(xml_str: str) -> str:
        processed = original(xml_str)
        return inject_hidden_obstacle_xml(processed, half_size=half_size, mass=mass)

    env.edit_model_xml = _wrapped
    processors = list(getattr(env, "_xml_processors", []) or [])
    replaced = False
    new_processors = []
    for proc in processors:
        proc_func = getattr(proc, "__func__", proc)
        if proc is original or proc_func is orig_func:
            new_processors.append(_wrapped)
            replaced = True
        else:
            new_processors.append(proc)
    if not replaced:
        new_processors.append(_wrapped)
    env._xml_processors = new_processors


def hide_l0_obstacle_actor(env: Any) -> dict[str, Any]:
    """Stash the shared L1/L2 free-joint box underground with contact off."""
    model = env.sim.model
    raw = getattr(model, "_model", model)
    info: dict[str, Any] = {"present": False, "n_geoms": 0, "body_name": None}
    meta = getattr(env, "_dynisafety_l12", None)
    body_name = None
    joint_name = None
    if isinstance(meta, dict):
        body_name = meta.get("body_name")
        joint_name = meta.get("joint_name")
    if not body_name:
        for candidate in (HIDDEN_OBSTACLE_BODY, f"{HIDDEN_OBSTACLE_BODY}_main"):
            try:
                model.body_name2id(candidate)
            except Exception:
                continue
            body_name = candidate
            break
    if not body_name:
        return info
    try:
        body_id = int(model.body_name2id(str(body_name)))
    except Exception:
        return info
    info["present"] = True
    info["body_name"] = str(body_name)
    if joint_name:
        try:
            env.sim.data.set_joint_qpos(
                str(joint_name),
                np.concatenate(
                    [
                        np.array(HIDDEN_OBSTACLE_POS, dtype=np.float64),
                        np.array(HIDDEN_OBSTACLE_QUAT_WXYZ, dtype=np.float64),
                    ]
                ),
            )
            jid = int(model.joint_name2id(str(joint_name)))
            addr = int(model.jnt_dofadr[jid])
            env.sim.data.qvel[addr : addr + 6] = 0.0
            info["joint_name"] = str(joint_name)
        except Exception:
            pass
    n_geoms = 0
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        n_geoms += 1
        if hasattr(raw, "geom_contype"):
            raw.geom_contype[gid] = 0
            raw.geom_conaffinity[gid] = 0
        if hasattr(raw, "geom_rgba"):
            raw.geom_rgba[gid, 3] = 0.0
    info["n_geoms"] = n_geoms
    try:
        env.sim.forward()
    except Exception:
        pass
    return info


def hide_l3_pedestrian_actor(env: Any) -> dict[str, Any]:
    """Stash the L3 pedestrian proxy underground with contact off."""
    model = env.sim.model
    raw = getattr(model, "_model", model)
    info: dict[str, Any] = {"present": False, "n_geoms": 0, "body_name": None}
    meta = getattr(env, "_dynisafety_l3", None)
    body_name = None
    joint_name = None
    if isinstance(meta, dict):
        body_name = meta.get("body_name")
        joint_name = meta.get("joint_name")
    if not body_name:
        for candidate in (HIDDEN_L3_BODY, f"{HIDDEN_L3_BODY}_main"):
            try:
                model.body_name2id(candidate)
            except Exception:
                continue
            body_name = candidate
            break
    if not body_name:
        return info
    try:
        body_id = int(model.body_name2id(str(body_name)))
    except Exception:
        return info
    info["present"] = True
    info["body_name"] = str(body_name)
    if joint_name:
        try:
            env.sim.data.set_joint_qpos(
                str(joint_name),
                np.concatenate(
                    [
                        np.array(HIDDEN_OBSTACLE_POS, dtype=np.float64),
                        np.array(HIDDEN_OBSTACLE_QUAT_WXYZ, dtype=np.float64),
                    ]
                ),
            )
            jid = int(model.joint_name2id(str(joint_name)))
            addr = int(model.jnt_dofadr[jid])
            env.sim.data.qvel[addr : addr + 6] = 0.0
            info["joint_name"] = str(joint_name)
        except Exception:
            pass
    n_geoms = 0
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        n_geoms += 1
        if hasattr(raw, "geom_contype"):
            raw.geom_contype[gid] = 0
            raw.geom_conaffinity[gid] = 0
        if hasattr(raw, "geom_rgba"):
            raw.geom_rgba[gid, 3] = 0.0
    info["n_geoms"] = n_geoms
    try:
        env.sim.forward()
    except Exception:
        pass
    return info


def _fixture_class_by_name(env: Any) -> dict[str, str]:
    return {str(name): type(fxtr).__name__ for name, fxtr in env.fixtures.items()}


def _pairwise_dist_map(env: Any) -> dict[tuple[str, str], float]:
    allowed = NAV_VALID_SRC_CLASSES | NAV_VALID_TARGET_CLASSES
    class_by_name = _fixture_class_by_name(env)
    names = [name for name, cls in class_by_name.items() if cls in allowed]
    if not names:
        return {}
    import robocasa.utils.object_utils as OU

    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            try:
                d = float(OU.fixture_pairwise_dist(env.fixtures[a], env.fixtures[b]))
            except (ValueError, AttributeError, TypeError):
                continue
            out[(str(a), str(b))] = d
            out[(str(b), str(a))] = d
    return out


def apply_pinned_target_fixture(env: Any, target_class: str) -> dict[str, Any]:
    """Overwrite NavigateKitchen destination to the YAML fixture class.

    Layout / src spawn may stay random; the destination family must not change.
    """
    from robocasa.utils import env_utils as EnvUtils

    class_by_name = _fixture_class_by_name(env)
    target_name = resolve_pinned_target_name(class_by_name, target_class)
    target = env.fixtures[target_name]
    resolved_class = type(target).__name__

    src = getattr(env, "src_fixture", None)
    src_name = getattr(src, "name", None) if src is not None else None
    src_cls = type(src).__name__ if src is not None else None
    need_new_src = (
        src is None
        or src is target
        or src_name == target_name
        or src_cls not in NAV_VALID_SRC_CLASSES
    )
    if need_new_src:
        dist_map = _pairwise_dist_map(env)
        candidates = candidate_src_names(
            class_by_name, target_name=target_name, pairwise_dist=dist_map
        )
        if not candidates:
            raise ValueError(
                f"no valid src_fixture ≥ {NAV_MIN_SRC_TARGET_DIST}m from "
                f"pinned {target_class}/{target_name}. {_WINDOW_C_FIX}"
            )
        src_name = str(env.rng.choice(candidates))
        src = env.fixtures[src_name]

    env.src_fixture = src
    env.target_fixture = target
    env.fixture_refs["src_fixture"] = src
    env.fixture_refs["target_fixture"] = target
    env.target_pos, env.target_ori = EnvUtils.compute_robot_base_placement_pose(env, target)
    env.init_robot_base_ref = src
    if need_new_src:
        (
            env.init_robot_base_pos_anchor,
            env.init_robot_base_ori_anchor,
        ) = EnvUtils.init_robot_base_pose(env)
        robot_pos = EnvUtils.set_robot_base(
            env=env,
            anchor_pos=env.init_robot_base_pos_anchor,
            anchor_ori=env.init_robot_base_ori_anchor,
            rot_dev=env.robot_spawn_deviation_rot,
            pos_dev_x=env.robot_spawn_deviation_pos_x,
            pos_dev_y=env.robot_spawn_deviation_pos_y,
        )
        env.init_robot_base_pos = robot_pos
        env.init_robot_base_ori = env.init_robot_base_ori_anchor
        env.sim.forward()

    return {
        "target_fixture_class": target_class,
        "target_fixture_class_resolved": resolved_class,
        "target_fixture_name": str(target.name),
        "src_fixture_class": type(src).__name__,
        "src_fixture_name": str(src.name),
        "src_repositioned": bool(need_new_src),
    }


def check_navigate_success(env: Any) -> bool:
    """Upstream ``NavigateKitchen._check_success`` (base pose + heading)."""
    return bool(env._check_success())


def eef_pos_from_obs(obs: Mapping[str, Any], env: Any) -> np.ndarray:
    if "robot0_eef_pos" in obs:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    return read_eef_xyz(env)


def read_eef_xyz(env: Any) -> np.ndarray:
    """World XYZ of the right EEF site (PandaOmron)."""
    site = env.robots[0].sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    return np.asarray(site, dtype=np.float64).reshape(3).copy()


def build_video_path(
    rollout_dir: Path,
    *,
    episode_idx: int,
    success: bool,
    task_description: str,
) -> Path:
    stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    processed = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    name = f"{stamp}--dynisafety--episode={episode_idx}--success={success}--task={processed}.mp4"
    return Path(rollout_dir) / name


def save_episode_video(
    frames: Sequence[np.ndarray],
    *,
    path: Path,
    fps: int,
) -> None:
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=int(fps))
    try:
        for img in frames:
            writer.append_data(img)
    finally:
        writer.close()


def lightwheel_objects_ready(sim_root: Path) -> bool:
    """True if objs_lw unzip produced at least one model.xml."""
    lw = Path(sim_root) / "robocasa" / "models" / "assets" / "objects" / "lightwheel"
    if not lw.is_dir():
        return False
    return any(lw.rglob("model.xml"))


# Fixture registry types whose XML lives under objects/lightwheel (objs_lw).
# Decorative only — none of these are NavigateKitchen destination pins.
LIGHTWHEEL_OBJECT_FIXTURE_TYPES: frozenset[str] = frozenset(
    {
        "cinnamon",
        "digital_scale",
        "dish_rack",
        "flower_vase",
        "fruit_bowl",
        "glass_cup",
        "jar",
        "jar_lid",
        "knife_block",
        "oil_bottle",
        "paper_towel",
        "paprika",
        "pepper_shaker",
        "plant",
        "salt_shaker",
        "soap_dispenser",
        "stool",
        "tiered_basket",
        "turmeric",
        "utensil_rack",
        "utensil_set",
        "vinegar_bottle",
    }
)


def disable_lightwheel_backed_fixtures(config: Any) -> int:
    """Set enable=False on layout nodes whose type needs objs_lw. Returns count."""
    n = 0
    if isinstance(config, dict):
        ftype = config.get("type")
        if isinstance(ftype, str) and ftype in LIGHTWHEEL_OBJECT_FIXTURE_TYPES:
            config["enable"] = False
            n += 1
        for value in config.values():
            n += disable_lightwheel_backed_fixtures(value)
    elif isinstance(config, list):
        for item in config:
            n += disable_lightwheel_backed_fixtures(item)
    return n


def ensure_generative_texture_tex_aliases(sim_root: Path) -> dict[str, int]:
    """Map HF-mirror UUID pngs to robocasa ``texNNN.png`` names (no copy).

    Official Box packs use ``tex001.png``…; the HF zip uses UUID filenames.
    """
    root = Path(sim_root) / "robocasa" / "models" / "assets" / "generative_textures"
    created: dict[str, int] = {}
    if not root.is_dir():
        return created
    for cat in ("wall", "floor", "cabinet", "counter"):
        folder = root / cat
        if not folder.is_dir():
            continue
        n = 0
        uuids = sorted(p for p in folder.glob("*.png") if not p.name.startswith("tex"))
        for i, src in enumerate(uuids, start=1):
            dest = folder / f"tex{i:03d}.png"
            if dest.exists() or dest.is_symlink():
                continue
            dest.symlink_to(src.name)
            n += 1
        created[cat] = n
    return created


def install_skip_lightwheel_layout_hook() -> None:
    """Patch create_fixtures so missing objs_lw does not abort NavigateKitchen."""
    from robocasa.models.scenes import kitchen_arena, scene_builder

    orig = scene_builder.create_fixtures
    if getattr(orig, "_dynisafety_skip_lightwheel", False):
        return

    def _wrapped(layout_config, style_config, rng=None):
        disable_lightwheel_backed_fixtures(layout_config)
        return orig(layout_config, style_config, rng=rng)

    _wrapped._dynisafety_skip_lightwheel = True  # type: ignore[attr-defined]
    scene_builder.create_fixtures = _wrapped
    kitchen_arena.create_fixtures = _wrapped


def make_robocasa_env(
    *,
    task: TaskSpec,
    extra_robocasa: Mapping[str, Any],
    control_freq: int,
    max_steps: int,
    seed: int,
    obstacle_half_size: Sequence[float],
    obstacle_mass: float,
    obstacle_rgba: Sequence[float],
    target_fixture: str,
    workspace: Path | None = None,
    level: int = 0,
    l3_half_size: Sequence[float] | None = None,
    l3_mass: float | None = None,
    l3_mesh_path: str | Path | None = None,
    l3_mesh_rgba: Sequence[float] | None = None,
    l12_visuals: Sequence[Any] | None = None,
    l3_visuals: Sequence[Any] | None = None,
) -> RoboCasaEnvHandle:
    """Create NavigateKitchen via ``robocasa.utils.env_utils.create_env``.

    L0–L3: freeze_base must be false; camera from YAML; free-joint floor box merged.
    L3 additionally merges a floor pedestrian (box collision + clothed visual).
    """
    env_class = _require_str(extra_robocasa, "env_class", where="extra.robocasa")
    if env_class != "NavigateKitchen":
        raise ValueError(
            f"extra.robocasa.env_class={env_class!r}; only NavigateKitchen "
            f"is allowed (no PickPlace*). {_WINDOW_C_FIX}"
        )
    if extra_robocasa.get("freeze_base") is not False:
        raise ValueError(
            f"extra.robocasa.freeze_base must be false (chassis must move). {_WINDOW_C_FIX}"
        )
    camera = _require_str(extra_robocasa, "camera", where="extra.robocasa")
    camera_width = _require_int(extra_robocasa, "camera_width", where="extra.robocasa")
    camera_height = _require_int(extra_robocasa, "camera_height", where="extra.robocasa")
    extra_cameras_raw = extra_robocasa.get("extra_cameras")
    extra_cameras: list[str] = []
    if extra_cameras_raw is not None:
        if not isinstance(extra_cameras_raw, Sequence) or isinstance(
            extra_cameras_raw, (str, bytes)
        ):
            raise TypeError(
                f"extra.robocasa.extra_cameras must be a list of camera names ({_WINDOW_C_FIX})"
            )
        for i, name in enumerate(extra_cameras_raw):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"extra.robocasa.extra_cameras[{i}] must be a non-empty str, "
                    f"got {name!r} ({_WINDOW_C_FIX})"
                )
            extra_cameras.append(name.strip())
    camera_names = [camera]
    for name in extra_cameras:
        if name not in camera_names:
            camera_names.append(name)
    if "policy_camera_width" in extra_robocasa:
        camera_width = _require_int(extra_robocasa, "policy_camera_width", where="extra.robocasa")
    if "policy_camera_height" in extra_robocasa:
        camera_height = _require_int(extra_robocasa, "policy_camera_height", where="extra.robocasa")
    robots = _require_str(extra_robocasa, "robots", where="extra.robocasa")
    split = _require_str(extra_robocasa, "split", where="extra.robocasa")
    sim_root = _require_str(extra_robocasa, "sim_root", where="extra.robocasa")
    renderer = _require_str(extra_robocasa, "renderer", where="extra.robocasa")
    if control_freq <= 0:
        raise ValueError(f"control_freq must be > 0, got {control_freq}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    ensure_robocasa_sys_path(sim_root=sim_root, workspace=workspace)
    from robocasa.utils.env_utils import create_env

    gentex_aliases = ensure_generative_texture_tex_aliases(Path(sim_root))

    extra_env_kwargs: dict[str, Any] = {}
    skipped_lightwheel_props = False
    if not lightwheel_objects_ready(Path(sim_root)):
        install_skip_lightwheel_layout_hook()
        skipped_lightwheel_props = True

    install_l12_navigate_kitchen_wrapper(
        half_size=obstacle_half_size,
        mass=obstacle_mass,
        rgba=obstacle_rgba,
        l3_half_size=l3_half_size if int(level) == 3 else None,
        l3_mass=l3_mass if int(level) == 3 else None,
        l3_mesh_path=l3_mesh_path if int(level) == 3 else None,
        l3_mesh_rgba=l3_mesh_rgba if int(level) == 3 else None,
        l12_visuals=l12_visuals if int(level) in (1, 2) else None,
        l3_visuals=l3_visuals if int(level) == 3 else None,
    )
    env = create_env(
        env_name=env_class,
        robots=robots,
        camera_names=camera_names,
        camera_widths=camera_width,
        camera_heights=camera_height,
        seed=int(seed),
        render_onscreen=False,
        split=split,
        control_freq=int(control_freq),
        horizon=int(max_steps),
        renderer=renderer,
        **extra_env_kwargs,
    )
    # Kitchen uses load_model_on_init=False — subclass _load_model merges the
    # free-joint floor box on the first reset (shared by L1/L2).
    env.reset()
    splits = assert_hybrid_mobile_12d_layout(env)
    from dynisafety.backends.robocasa.disturbance._support import (
        snapshot_actor_geom_contact,
        snapshot_geom_contact,
    )

    meta = getattr(env, "_dynisafety_l12", None)
    if isinstance(meta, dict) and not meta.get("geom_contact"):
        meta["geom_contact"] = snapshot_geom_contact(env)
    hide_info = hide_l0_obstacle_actor(env)
    meta_l3 = getattr(env, "_dynisafety_l3", None)
    hide_l3 = None
    if isinstance(meta_l3, dict):
        if not meta_l3.get("geom_contact"):
            meta_l3["geom_contact"] = snapshot_actor_geom_contact(env, meta_l3)
        hide_l3 = hide_l3_pedestrian_actor(env)
    pin_info = apply_pinned_target_fixture(env, target_fixture)
    return RoboCasaEnvHandle(
        env=env,
        task_description=task.instruction,
        suite=task.suite,
        task_id=task.task_id,
        task_name=task.task_name,
        camera=camera,
        target_fixture_class=target_fixture,
        extra={
            "action_split_indexes": {k: list(v) for k, v in splits.items()},
            "hidden_obstacle": hide_info,
            "l12_obstacle": _public_l12_meta(meta) if isinstance(meta, dict) else None,
            "l3_human": _public_l12_meta(meta_l3) if isinstance(meta_l3, dict) else None,
            "hidden_l3": hide_l3,
            "skipped_lightwheel_props_missing_objs_lw": skipped_lightwheel_props,
            "gentex_tex_aliases": gentex_aliases,
            **pin_info,
        },
    )
