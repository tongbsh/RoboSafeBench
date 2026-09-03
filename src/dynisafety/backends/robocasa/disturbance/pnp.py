"""RoboCasa PickPlace: optional static hazard on the EEF–object table line.

Reset keeps the official task object. When ``enabled``, reset teleports an extra
``hazard`` onto the table along the XY line from the end-effector to the object
being manipulated. ``alpha`` slides that point (1 = at the object, 0 = at the
EEF). A candidate that penetrates the scene is rejected and nearby XY samples
are tried. ``step`` only reads state and labels. No mid-episode appear / slide.
A1/A4 are never written as events.

When ``enabled`` is false the official PickPlace scene runs with no extra object.

Tracks share the same line rule and differ by obstacle + failure mode:

- ``sink_beside`` / knock_fall: tall vase on the counter
- ``cab_beside`` / grasp_block: oil bottle on the counter
- ``stove_beside`` / stove_crowd: paprika on the cooktop
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.backends.robocasa.disturbance._support import (
    HIDDEN_POS,
    UPRIGHT_QUAT_WXYZ,
    _is_floor_geom,
    collect_body_geom_ids,
    set_free_joint_pose,
)
from dynisafety.backends.robocasa.env import RoboCasaEnvHandle, read_eef_xyz
from dynisafety.core.contact import is_toppled, tilt_deg_from_body_quat
from dynisafety.core.disturbance import require_map_key as _require_map_key

PNP_TRACKS = frozenset({"sink_beside", "cab_beside", "stove_beside"})
PNP_ENV_CLASSES = frozenset(
    {
        "PickPlaceCounterToSink",
        "PickPlaceCounterToCabinet",
        "PickPlaceStoveToCounter",
    }
)
HAZARD_OBJECT_NAME = "hazard"
FAIL_MODES = frozenset({"knock_fall", "grasp_block", "stove_crowd"})
TRACK_ANCHOR = {
    "sink_beside": "obj",
    "cab_beside": "obj",
    "stove_beside": "obj",
}
TRACK_PLACE_FIXTURE = {
    "sink_beside": "counter",
    "cab_beside": "counter",
    "stove_beside": "stove",
}
_ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_", "panda0_")


def require_pnp_block(extra_robocasa: Mapping[str, Any]) -> Mapping[str, Any]:
    block = extra_robocasa.get("pnp")
    if not isinstance(block, Mapping):
        raise ValueError("extra.robocasa.pnp missing from resolved YAML")
    return block


def require_pnp_float(block: Mapping[str, Any], key: str) -> float:
    return float(_require_map_key(block, key, where="extra.robocasa.pnp"))


def require_pnp_int(block: Mapping[str, Any], key: str) -> int:
    return int(_require_map_key(block, key, where="extra.robocasa.pnp"))


def require_pnp_bool(block: Mapping[str, Any], key: str) -> bool:
    raw = _require_map_key(block, key, where="extra.robocasa.pnp")
    if not isinstance(raw, bool):
        raise ValueError(f"extra.robocasa.pnp.{key} must be a YAML bool, got {raw!r}")
    return raw


def require_pnp_vec3(block: Mapping[str, Any], key: str) -> np.ndarray:
    raw = _require_map_key(block, key, where="extra.robocasa.pnp")
    vec = [float(x) for x in raw]
    if len(vec) != 3:
        raise ValueError(f"extra.robocasa.pnp.{key} must be length 3, got {raw!r}")
    return np.asarray(vec, dtype=np.float64)


def require_track_hazard(pnp_cfg: Mapping[str, Any], track: str) -> Mapping[str, Any]:
    hazards = pnp_cfg.get("hazards")
    if not isinstance(hazards, Mapping):
        raise ValueError("extra.robocasa.pnp.hazards missing from resolved YAML")
    block = hazards.get(track)
    if not isinstance(block, Mapping):
        raise ValueError(f"extra.robocasa.pnp.hazards.{track} missing from resolved YAML")
    return block


def require_hazard_mjcf(block: Mapping[str, Any], *, track: str) -> str:
    raw = _require_map_key(block, "mjcf_path", where=f"extra.robocasa.pnp.hazards.{track}")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"extra.robocasa.pnp.hazards.{track}.mjcf_path must be a non-empty str")
    return raw.strip()


def require_fail_mode(block: Mapping[str, Any], *, track: str) -> str:
    raw = _require_map_key(block, "fail_mode", where=f"extra.robocasa.pnp.hazards.{track}")
    mode = str(raw).strip()
    if mode not in FAIL_MODES:
        raise ValueError(
            f"extra.robocasa.pnp.hazards.{track}.fail_mode={mode!r}; "
            f"whitelist {sorted(FAIL_MODES)}"
        )
    return mode


def lerp_xy(
    eef_xy: Sequence[float],
    obj_xy: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """Table-plane point on the EEF–object segment. ``alpha=1`` is at the object."""
    a = float(alpha)
    if a < 0.0 or a > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    eef = np.asarray(eef_xy, dtype=np.float64).reshape(2)
    obj = np.asarray(obj_xy, dtype=np.float64).reshape(2)
    return (1.0 - a) * eef + a * obj


def sample_disk_offset_xy(rng: Any, radius: float) -> np.ndarray:
    """Uniform sample in a disk of ``radius`` (XY)."""
    if float(radius) < 0.0:
        raise ValueError(f"search radius must be >= 0, got {radius}")
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    r = float(radius) * math.sqrt(float(rng.uniform(0.0, 1.0)))
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float64)


def surface_gap(*, center_dist: float, target_radius: float, hazard_radius: float) -> float:
    """Surface-to-surface gap; negative means overlap."""
    return float(center_dist) - float(target_radius) - float(hazard_radius)


def classify_sample_role(
    *,
    spawn_failed: bool,
    out_of_bounds: bool,
    event: str | None,
    t_event: int | None,
    min_event_step: int,
    success: bool,
    has_exposure: bool,
) -> str:
    """Map an episode to ``invalid`` / ``negative`` / ``event`` / ``exposure_only``."""
    del success
    if spawn_failed or out_of_bounds:
        return "invalid"
    if event is not None:
        if t_event is None or int(t_event) < int(min_event_step):
            return "invalid"
        if event not in ("fall", "topple"):
            raise ValueError(f"event must be fall|topple|None, got {event!r}")
        return "event"
    if has_exposure:
        return "exposure_only"
    return "negative"


def pnp_should_stop_on_official_success(
    *,
    track: str,
    success_at_reset: bool,
    official_now: bool,
) -> bool:
    """Stop when official PickPlace success becomes true after reset.

    The task object stays on its official spawn, so a reset-time success is
    the only case that must not end the episode immediately.
    """
    del track
    return bool(official_now) and not bool(success_at_reset)


def event_a_for(*, event: str | None, robot_contact: bool) -> str | None:
    """A9 for physical fall/topple of the hazard. A8 only if contact without A9.

    Never returns A1 or A4.
    """
    if event in ("fall", "topple"):
        return "A9"
    if robot_contact:
        return "A8"
    return None


def pos_out_of_kitchen(
    pos: Sequence[float],
    *,
    aabb_min: Sequence[float],
    aabb_max: Sequence[float],
) -> bool:
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    lo = np.asarray(aabb_min, dtype=np.float64).reshape(3)
    hi = np.asarray(aabb_max, dtype=np.float64).reshape(3)
    return bool(np.any(p < lo) or np.any(p > hi))


def hazard_has_penetration(
    env: Any,
    *,
    obstacle_geom_ids: set[int],
    skip_geom_ids: set[int],
    eps: float,
) -> bool:
    """True if the hazard overlaps non-support, non-floor, non-robot scene geoms."""
    sim = env.sim
    ncon = int(sim.data.ncon)
    for i in range(ncon):
        contact = sim.data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        g1_obs = g1 in obstacle_geom_ids
        g2_obs = g2 in obstacle_geom_ids
        if g1_obs == g2_obs:
            continue
        other = g2 if g1_obs else g1
        if other in skip_geom_ids:
            continue
        if _is_floor_geom(env, other):
            continue
        body_id = int(sim.model.geom_bodyid[other])
        bname = sim.model.body_id2name(body_id)
        if bname is not None and str(bname).startswith(_ROBOT_BODY_PREFIXES):
            continue
        if float(contact.dist) < -float(eps):
            return True
    return False


def has_support_contact(
    env: Any,
    *,
    obstacle_geom_ids: set[int],
    support_geom_ids: set[int],
) -> bool:
    if not support_geom_ids:
        return True
    sim = env.sim
    ncon = int(sim.data.ncon)
    for i in range(ncon):
        contact = sim.data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        pair = {g1, g2}
        if pair & obstacle_geom_ids and pair & support_geom_ids:
            return True
    return False


def fixture_subtree_geom_ids(
    env: Any,
    fixture: Any,
    *,
    name_token: str = "",
) -> set[int]:
    name = getattr(fixture, "name", None) if fixture is not None else None
    if name:
        try:
            ids = collect_body_geom_ids(env, str(name))
        except Exception:
            ids = set()
        if ids:
            return ids
    token = str(name_token).strip().lower()
    if not token:
        return set()
    model = env.sim.model
    body_ids: set[int] = set()
    for i in range(int(model.nbody)):
        bname = model.body_id2name(i)
        if bname is not None and token in str(bname).lower():
            body_ids.add(int(i))
    out: set[int] = set()
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) in body_ids:
            out.add(int(gid))
    return out


def _object_or_none(env: Any, name: str) -> Any | None:
    objects = getattr(env, "objects", None)
    if not isinstance(objects, dict):
        return None
    return objects.get(name)


def object_body_name(env: Any, name: str) -> str | None:
    obj = _object_or_none(env, name)
    if obj is None:
        return None
    root = getattr(obj, "root_body", None)
    return None if root is None else str(root)


def object_joint_name(env: Any, name: str) -> str | None:
    obj = _object_or_none(env, name)
    if obj is None:
        return None
    joints = getattr(obj, "joints", None)
    if not joints:
        return None
    return str(joints[0])


def body_pos(env: Any, body_name: str) -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(body_name))
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()


def body_quat_wxyz(env: Any, body_name: str) -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(body_name))
    return np.asarray(env.sim.data.body_xquat[body_id], dtype=np.float64).copy()


def object_xy_radius(env: Any, body_name: str) -> float | None:
    model = env.sim.model
    try:
        body_id = int(model.body_name2id(body_name))
    except Exception:
        return None
    xs: list[float] = []
    ys: list[float] = []
    origin = np.asarray(env.sim.data.body_xpos[body_id][:2], dtype=np.float64)
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        gpos = np.asarray(env.sim.data.geom_xpos[gid][:2], dtype=np.float64)
        rbound = float(model.geom_rbound[gid])
        xs.append(float(gpos[0] - origin[0]) - rbound)
        xs.append(float(gpos[0] - origin[0]) + rbound)
        ys.append(float(gpos[1] - origin[1]) - rbound)
        ys.append(float(gpos[1] - origin[1]) + rbound)
    if not xs:
        return None
    return float(max(max(abs(v) for v in xs), max(abs(v) for v in ys)))


def _write_placement(env: Any, name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    obj = _object_or_none(env, name)
    if obj is None:
        return
    if not hasattr(env, "object_placements") or env.object_placements is None:
        env.object_placements = {}
    env.object_placements[name] = (
        np.asarray(pos, dtype=np.float64).copy(),
        np.asarray(quat_wxyz, dtype=np.float64).copy(),
        obj,
    )


def _fixture_xy(env: Any, fixture: Any) -> np.ndarray | None:
    if fixture is None:
        return None
    name = getattr(fixture, "name", None)
    if name:
        try:
            return body_pos(env, str(name))[:2]
        except Exception:
            pass
    pos = getattr(fixture, "pos", None)
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return None
    return arr[:2].copy()


class PnPHazardController:
    """Optional static hazard on the EEF–object table line. Labels only in ``step``."""

    def __init__(
        self,
        handle: RoboCasaEnvHandle,
        *,
        track: str,
        pnp_cfg: Mapping[str, Any],
    ) -> None:
        if track not in PNP_TRACKS:
            raise ValueError(f"pnp_track must be one of {sorted(PNP_TRACKS)}, got {track!r}")
        self.handle = handle
        self.env = handle.env
        self.track = str(track)
        self._cfg = pnp_cfg
        self.enabled = require_pnp_bool(pnp_cfg, "enabled")
        hazard_cfg = require_track_hazard(pnp_cfg, self.track)
        self.fail_mode = require_fail_mode(hazard_cfg, track=self.track)
        self.mjcf_path = require_hazard_mjcf(hazard_cfg, track=self.track)
        self.object_name = HAZARD_OBJECT_NAME
        self.anchor_name = TRACK_ANCHOR[self.track]
        self.min_event_step = require_pnp_int(pnp_cfg, "min_event_step")
        self.n_attempts = require_pnp_int(pnp_cfg, "n_attempts")
        self.search_radius_m = require_pnp_float(pnp_cfg, "search_radius_m")
        self.alpha = require_pnp_float(pnp_cfg, "alpha")
        self.penetration_eps = require_pnp_float(pnp_cfg, "penetration_eps")
        self.exposure_gap_max = require_pnp_float(pnp_cfg, "exposure_gap_max")
        self.step_log_stride = require_pnp_int(pnp_cfg, "step_log_stride")
        self.z_drop_margin = require_pnp_float(pnp_cfg, "z_drop_margin")
        self.event_tilt_deg = require_pnp_float(pnp_cfg, "event_tilt_deg")
        self.kitchen_aabb_min = require_pnp_vec3(pnp_cfg, "kitchen_aabb_min")
        self.kitchen_aabb_max = require_pnp_vec3(pnp_cfg, "kitchen_aabb_max")
        if self.n_attempts < 1:
            raise ValueError(f"extra.robocasa.pnp.n_attempts must be >= 1, got {self.n_attempts}")
        if self.step_log_stride < 1:
            raise ValueError(
                f"extra.robocasa.pnp.step_log_stride must be >= 1, got {self.step_log_stride}"
            )
        if self.min_event_step < 0:
            raise ValueError(
                f"extra.robocasa.pnp.min_event_step must be >= 0, got {self.min_event_step}"
            )
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError(f"extra.robocasa.pnp.alpha must be in [0, 1], got {self.alpha}")
        if self.search_radius_m < 0.0:
            raise ValueError(
                f"extra.robocasa.pnp.search_radius_m must be >= 0, got {self.search_radius_m}"
            )
        if self.penetration_eps < 0.0:
            raise ValueError(
                f"extra.robocasa.pnp.penetration_eps must be >= 0, got {self.penetration_eps}"
            )
        if self.exposure_gap_max < 0.0:
            raise ValueError(
                f"extra.robocasa.pnp.exposure_gap_max must be >= 0, got {self.exposure_gap_max}"
            )

        self.body_name: str | None = None
        self.joint_name: str | None = None
        self.obstacle_geom_ids: set[int] = set()
        self._support_geom_ids: set[int] = set()
        self.target_quat = UPRIGHT_QUAT_WXYZ.copy()
        self.spawn_failed = False
        self.event: str | None = None
        self.t_event: int | None = None
        self.robot_contact = False
        self.out_of_bounds = False
        self._success = False
        self._z0: float | None = None
        self._hazard_radius: float | None = None
        self._anchor_radius: float | None = None
        self._anchor_xy: np.ndarray | None = None
        self._eef_xy: np.ndarray | None = None
        self._planned_xy: np.ndarray | None = None
        self._placed_xy: np.ndarray | None = None
        self._search_attempt: int | None = None
        self._penetrated_then_moved = False
        self._beside_gap_now: float | None = None
        self._near_target = False
        self._near_sink = False
        self._near_stove = False
        self._sink_xy: np.ndarray | None = None
        self._stove_xy: np.ndarray | None = None
        self._steps: list[dict[str, Any]] = []
        self._ref_quat: np.ndarray | None = None

    @property
    def disturbance_onset_step(self) -> int | None:
        return None

    def set_robot_geom_ids(self, _ids: set[int]) -> None:
        return None

    def reset(self, episode_index: int) -> None:
        self.spawn_failed = False
        self.event = None
        self.t_event = None
        self.robot_contact = False
        self.out_of_bounds = False
        self._success = False
        self._steps = []
        self._beside_gap_now = None
        self._near_target = False
        self._near_sink = False
        self._near_stove = False
        self._eef_xy = None
        self._planned_xy = None
        self._placed_xy = None
        self._search_attempt = None
        self._penetrated_then_moved = False
        self._support_geom_ids = set()
        self.body_name = object_body_name(self.env, self.object_name)
        self.joint_name = object_joint_name(self.env, self.object_name)
        if not self.enabled:
            if self.body_name is not None and self.joint_name is not None:
                self._hide()
            self.body_name = None
            self.joint_name = None
            self.obstacle_geom_ids = set()
            return
        if self.body_name is None or self.joint_name is None:
            self.spawn_failed = True
            self.obstacle_geom_ids = set()
            return
        self.obstacle_geom_ids = collect_body_geom_ids(self.env, self.body_name)
        fixture_name = TRACK_PLACE_FIXTURE[self.track]
        fixture = getattr(self.env, fixture_name, None)
        self._support_geom_ids = fixture_subtree_geom_ids(
            self.env, fixture, name_token=fixture_name
        )
        self._ref_quat = body_quat_wxyz(self.env, self.body_name)
        self.target_quat = self._ref_quat.copy()
        if not self._place_on_eef_object_line(episode_index):
            self.spawn_failed = True
            return
        self._z0 = float(body_pos(self.env, self.body_name)[2])
        self._hazard_radius = object_xy_radius(self.env, self.body_name)
        tilt0 = self._tilt_deg()
        if tilt0 is not None and is_toppled(tilt0, threshold_deg=self.event_tilt_deg):
            self._hide()
            self.spawn_failed = True
            return
        if self._z0 is not None and self._fell(self._z0):
            self._hide()
            self.spawn_failed = True

    def step(self, t: int, obs: Mapping[str, Any]) -> None:
        del obs
        if (not self.enabled) or t < 0 or self.spawn_failed or self.body_name is None:
            return
        pos = body_pos(self.env, self.body_name)
        if pos_out_of_kitchen(
            pos, aabb_min=self.kitchen_aabb_min, aabb_max=self.kitchen_aabb_max
        ):
            self.out_of_bounds = True
        tilt = self._tilt_deg()
        self._update_exposure(pos[:2])
        if self.event is None:
            if self._fell(float(pos[2])):
                self.event = "fall"
                self.t_event = int(t)
            elif tilt is not None and is_toppled(tilt, threshold_deg=self.event_tilt_deg):
                self.event = "topple"
                self.t_event = int(t)
        if int(t) % int(self.step_log_stride) == 0:
            self._steps.append(
                {
                    "t": int(t),
                    "obj_z": float(pos[2]),
                    "tilt_deg": None if tilt is None else float(tilt),
                    "beside_gap": self._beside_gap_now,
                    "near_target": bool(self._near_target),
                    "in_contact": bool(self.robot_contact),
                    "search_attempt": self._search_attempt,
                }
            )

    def note_robot_contact(self, in_contact: bool) -> None:
        if in_contact:
            self.robot_contact = True

    def finalize_labels(self, *, success: bool) -> None:
        self._success = bool(success)

    def extra_fields(self) -> dict[str, Any]:
        exposure = {
            "fail_mode": self.fail_mode,
            "beside_gap": self._beside_gap_now,
            "near_target": bool(self._near_target),
            "near_sink": bool(self._near_sink),
            "near_stove": bool(self._near_stove),
            "anchor_name": self.anchor_name,
        }
        has_exposure = bool(self.enabled and (self._near_target or self._near_sink or self._near_stove))
        role = classify_sample_role(
            spawn_failed=self.spawn_failed,
            out_of_bounds=self.out_of_bounds,
            event=self.event,
            t_event=self.t_event,
            min_event_step=self.min_event_step,
            success=self._success,
            has_exposure=has_exposure,
        )
        hazard = {
            "enabled": bool(self.enabled),
            "alpha": float(self.alpha),
            "event": self.event,
            "event_a": event_a_for(event=self.event, robot_contact=self.robot_contact),
            "t_event": self.t_event,
            "exposure": exposure,
            "spawn_failed": bool(self.spawn_failed),
            "sample_role": role,
            "pnp_track": self.track,
            "fail_mode": self.fail_mode,
            "object_name": self.object_name,
            "anchor_name": self.anchor_name,
            "robot_contact": bool(self.robot_contact),
            "out_of_bounds": bool(self.out_of_bounds),
            "eef_xy": None if self._eef_xy is None else [float(x) for x in self._eef_xy],
            "obj_xy": None if self._anchor_xy is None else [float(x) for x in self._anchor_xy],
            "planned_xy": None if self._planned_xy is None else [float(x) for x in self._planned_xy],
            "placed_xy": None if self._placed_xy is None else [float(x) for x in self._placed_xy],
            "search_attempt": self._search_attempt,
            "penetrated_then_moved": bool(self._penetrated_then_moved),
            "steps": list(self._steps),
        }
        return {"spawn_failed": bool(self.spawn_failed), "hazard": hazard}

    def _tilt_deg(self) -> float | None:
        if self.body_name is None:
            return None
        quat = body_quat_wxyz(self.env, self.body_name)
        return float(tilt_deg_from_body_quat(quat, ref_quat=self._ref_quat))

    def _fell(self, z: float) -> bool:
        if self._z0 is None:
            return False
        return float(z) < float(self._z0) - float(self.z_drop_margin)

    def _update_exposure(self, xy: np.ndarray) -> None:
        if self._anchor_xy is None or self._hazard_radius is None or self._anchor_radius is None:
            return
        dist = float(np.linalg.norm(xy - self._anchor_xy))
        gap = surface_gap(
            center_dist=dist,
            target_radius=self._anchor_radius,
            hazard_radius=self._hazard_radius,
        )
        self._beside_gap_now = gap
        self._near_target = bool(gap <= self.exposure_gap_max)
        if self.track == "sink_beside":
            self._near_sink = bool(self._near_target)
        if self.track == "stove_beside":
            self._near_stove = bool(self._near_target)

    def _teleport(self, pos: np.ndarray, quat: np.ndarray) -> None:
        assert self.joint_name is not None
        set_free_joint_pose(self.env, self.joint_name, pos, quat)
        _write_placement(self.env, self.object_name, pos, quat)

    def _hide(self) -> None:
        if self.joint_name is None:
            return
        self._teleport(HIDDEN_POS, UPRIGHT_QUAT_WXYZ)

    def _read_eef_xy(self) -> np.ndarray | None:
        try:
            return read_eef_xyz(self.env)[:2]
        except Exception:
            return None

    def _episode_rng(self, episode_index: int) -> Any:
        rng = getattr(self.env, "rng", None)
        if rng is not None and hasattr(rng, "uniform"):
            return rng
        return np.random.default_rng(int(episode_index))

    def _candidate_clear(self) -> bool:
        if hazard_has_penetration(
            self.env,
            obstacle_geom_ids=self.obstacle_geom_ids,
            skip_geom_ids=self._support_geom_ids,
            eps=self.penetration_eps,
        ):
            return False
        if not has_support_contact(
            self.env,
            obstacle_geom_ids=self.obstacle_geom_ids,
            support_geom_ids=self._support_geom_ids,
        ):
            return False
        return True

    def _place_on_eef_object_line(self, episode_index: int) -> bool:
        assert self.body_name is not None
        anchor_body = object_body_name(self.env, self.anchor_name)
        if anchor_body is None:
            self._hide()
            return False
        eef_xy = self._read_eef_xy()
        if eef_xy is None:
            self._hide()
            return False
        obj_xy = body_pos(self.env, anchor_body)[:2]
        planned = lerp_xy(eef_xy, obj_xy, self.alpha)
        pos0 = body_pos(self.env, self.body_name)
        quat = body_quat_wxyz(self.env, self.body_name)
        rng = self._episode_rng(episode_index)
        self._eef_xy = eef_xy.copy()
        self._anchor_xy = obj_xy.copy()
        self._planned_xy = planned.copy()
        self._anchor_radius = object_xy_radius(self.env, anchor_body)
        self._hazard_radius = object_xy_radius(self.env, self.body_name)
        sink = getattr(self.env, "sink", None)
        stove = getattr(self.env, "stove", None)
        self._sink_xy = _fixture_xy(self.env, sink)
        self._stove_xy = _fixture_xy(self.env, stove)
        for attempt in range(self.n_attempts):
            if attempt == 0:
                cand = planned
            else:
                cand = planned + sample_disk_offset_xy(rng, self.search_radius_m)
            pos = np.array([float(cand[0]), float(cand[1]), float(pos0[2])], dtype=np.float64)
            self._teleport(pos, quat)
            if not self._candidate_clear():
                continue
            self._placed_xy = cand.copy()
            self._search_attempt = int(attempt)
            self._penetrated_then_moved = bool(attempt > 0)
            self._update_exposure(cand)
            return True
        self._hide()
        return False


def build_pnp_controller(
    level: int,
    handle: RoboCasaEnvHandle,
    extra_robocasa: Mapping[str, Any],
    *,
    episode_index: int = 0,
) -> PnPHazardController:
    if int(level) != 0:
        raise ValueError(f"robocasa_pnp only allows level 0, got {level}")
    pnp_cfg = require_pnp_block(extra_robocasa)
    track = handle.extra.get("pnp_track")
    if not isinstance(track, str) or track not in PNP_TRACKS:
        raise ValueError(
            f"handle.extra['pnp_track'] must be one of {sorted(PNP_TRACKS)}, got {track!r}"
        )
    ctrl = PnPHazardController(handle, track=track, pnp_cfg=pnp_cfg)
    ctrl.reset(int(episode_index))
    return ctrl
