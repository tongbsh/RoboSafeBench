"""LIBERO god-view yield-to-home (privileged feasibility arm).

Latch ``engage_lead_steps`` before L2/L3 onset (obstacle already in scene /
hand about to enter), stop calling the VLA, and snap arm joints toward the
episode-start qpos at ``max_joint_step_rad`` per control tick. Hold at home
through the live motion window; resume only after arrived (or timeout
outside the keep-out) **and** the phase is settled/withdrawn. Timeout does
**not** hand the VLA back while the EEF is still inside the keep-out. On
resume the VLA task clock / env horizon is zeroed; the disturbance wall
clock is **not** rewound. Default-off; does not wrap Oracle SWR. AEGIS may
be paired as a **post-resume** filter (parked L2 residual); the backend
must not call it while this session is taking over.

Numeric knobs live in YAML (iron law 1). Default safe pose is the snapshot
taken after ``set_init_state`` (home EEF + joints).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dynisafety.backends.libero.env import quat2axisangle
from dynisafety.backends.libero.safety.phases import LiberoOraclePhase
from dynisafety.config.schema import RunConfig

_REQUIRED_KEYS = (
    "enabled",
    "levels",
    "pos_tol_m",
    "ori_tol_rad",
    "max_step_m",
    "max_step_rad",
    "kp_pos",
    "kp_ori",
    "retreat_max_steps",
    "corridor_radius_m",
    "lateral_offset_m",
    "lift_m",
    "default_gripper_hold",
    "reset_task_clock",
    "engage_lead_steps",
    "max_joint_step_rad",
    "joint_tol_rad",
)

_FILTERABLE = frozenset({"delta_ee_7d", "abs_ee_7d"})


class YieldHomeState(str, Enum):
    PASS_THROUGH = "pass_through"
    YIELD = "yield"
    HOLD_SAFE = "hold_safe"


def yield_home_enabled_from_extra(extra_libero: Mapping[str, Any] | None) -> bool:
    """Require explicit ``extra.libero.yield_home.enabled`` (absent → false)."""
    if not extra_libero:
        return False
    block = extra_libero.get("yield_home")
    if not isinstance(block, Mapping):
        return False
    return bool(block.get("enabled", False))


def _require(block: Mapping[str, Any], key: str) -> Any:
    if key not in block:
        raise ValueError(f"extra.libero.yield_home.{key} is required when enabled")
    return block[key]


def _require_bool(block: Mapping[str, Any], key: str) -> bool:
    value = _require(block, key)
    if not isinstance(value, bool):
        raise ValueError(f"extra.libero.yield_home.{key} must be bool, got {value!r}")
    return bool(value)


def _require_float(block: Mapping[str, Any], key: str, *, min_value: float | None = None) -> float:
    value = _require(block, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"extra.libero.yield_home.{key} must be a number, got {value!r}")
    out = float(value)
    if min_value is not None and out < min_value:
        raise ValueError(f"extra.libero.yield_home.{key} must be >= {min_value}, got {out}")
    return out


def _require_int(block: Mapping[str, Any], key: str, *, min_value: int) -> int:
    value = _require(block, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"extra.libero.yield_home.{key} must be int, got {value!r}")
    if int(value) < min_value:
        raise ValueError(f"extra.libero.yield_home.{key} must be >= {min_value}, got {value}")
    return int(value)


def _require_levels(block: Mapping[str, Any]) -> frozenset[int]:
    raw = _require(block, "levels")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("extra.libero.yield_home.levels must be a non-empty list of ints")
    out: set[int] = set()
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or int(item) < 0:
            raise ValueError(f"extra.libero.yield_home.levels entries must be int >= 0, got {item!r}")
        out.add(int(item))
    return frozenset(out)


def segment_hits_sphere(
    p0: np.ndarray,
    p1: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> bool:
    """True if the segment ``p0→p1`` comes within ``radius`` of ``center``."""
    a = np.asarray(p0, dtype=np.float64).reshape(3)
    b = np.asarray(p1, dtype=np.float64).reshape(3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    r = float(radius)
    if r < 0.0:
        raise ValueError(f"radius must be >= 0, got {r}")
    delta = b - a
    f = a - c
    aa = float(np.dot(delta, delta))
    if aa < 1e-16:
        return float(np.dot(f, f)) <= r * r
    t = float(np.clip(-np.dot(f, delta) / aa, 0.0, 1.0))
    closest = a + t * delta
    return float(np.linalg.norm(closest - c)) <= r


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    cap = float(max_norm)
    n = float(np.linalg.norm(arr))
    if cap <= 0.0 or n <= cap or n < 1e-12:
        return arr
    return arr * (cap / n)


def _as_xyz(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3].copy()


def _l3_dwell(controller: Any) -> np.ndarray | None:
    dwell = _as_xyz(getattr(controller, "_dwell_pos", None))
    if dwell is not None:
        return dwell
    plan = getattr(controller, "l3_plan", None)
    if isinstance(plan, Mapping):
        return _as_xyz(plan.get("dwell"))
    return None


def _l2_slide_path(controller: Any) -> np.ndarray | None:
    """Planned L2 polyline (start→land). Fallback: start_pos / final_pos."""
    path = getattr(controller, "_planned_path", None)
    if path is not None:
        arr = np.asarray(path, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 3:
            return arr[:, :3].copy()
    start = _as_xyz(getattr(controller, "start_pos", None))
    land = _as_xyz(getattr(controller, "final_pos", None))
    if start is not None and land is not None:
        return np.stack([start, land], axis=0)
    return None


def _hazard_geometry(controller: Any | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Keep-out sphere center (L3 dwell / L2 landing) and optional L2 polyline."""
    if controller is None:
        return None, None
    dwell = _l3_dwell(controller)
    if dwell is not None:
        return dwell, None
    path = _l2_slide_path(controller)
    if path is not None:
        return None, path
    return _as_xyz(getattr(controller, "final_pos", None)), None


def segment_hits_segment(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    radius: float,
) -> bool:
    """True if two 3D segments come within ``radius``."""
    return segment_segment_distance(a0, a1, b0, b1) <= float(radius)


def segment_segment_distance(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> float:
    """Minimum distance between two 3D segments (sampled; keep-out is centimetres)."""
    p1 = np.asarray(a0, dtype=np.float64).reshape(3)
    q1 = np.asarray(a1, dtype=np.float64).reshape(3)
    p2 = np.asarray(b0, dtype=np.float64).reshape(3)
    q2 = np.asarray(b1, dtype=np.float64).reshape(3)
    d1 = q1 - p1
    d2 = q2 - p2
    len2 = float(np.dot(d2, d2))
    samples = 16
    best = float("inf")
    for i in range(samples + 1):
        s = i / samples
        p = p1 + s * d1
        if len2 < 1e-16:
            dist = float(np.linalg.norm(p - p2))
        else:
            t = float(np.clip(np.dot(p - p2, d2) / len2, 0.0, 1.0))
            dist = float(np.linalg.norm(p - (p2 + t * d2)))
        if dist < best:
            best = dist
    return best


def polyline_hits_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    path: np.ndarray,
    radius: float,
) -> tuple[bool, np.ndarray | None]:
    """If ``p0→p1`` comes within ``radius`` of the polyline, return the nearest path point."""
    line0 = np.asarray(p0, dtype=np.float64).reshape(3)
    line1 = np.asarray(p1, dtype=np.float64).reshape(3)
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
        raise ValueError(f"hazard path must be (N, 3) with N>=2, got {pts.shape}")
    r = float(radius)
    best_d = float("inf")
    best_pt: np.ndarray | None = None
    for i in range(pts.shape[0] - 1):
        a = pts[i, :3]
        b = pts[i + 1, :3]
        dist = segment_segment_distance(line0, line1, a, b)
        if dist <= r and dist < best_d:
            best_d = dist
            mid = 0.5 * (a + b)
            best_pt = mid.copy()
    if best_pt is None:
        return False, None
    return True, best_pt


def point_in_keepout(
    pos: np.ndarray,
    *,
    dwell_pos: np.ndarray | None,
    hazard_path: np.ndarray | None,
    radius: float,
) -> bool:
    """True if ``pos`` is inside the L3 dwell sphere or L2 slide tube."""
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    r = float(radius)
    if r < 0.0:
        raise ValueError(f"radius must be >= 0, got {r}")
    if dwell_pos is not None:
        c = np.asarray(dwell_pos, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(p - c)) <= r:
            return True
    if hazard_path is not None:
        pts = np.asarray(hazard_path, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
            raise ValueError(f"hazard path must be (N, 3) with N>=2, got {pts.shape}")
        for i in range(pts.shape[0] - 1):
            if segment_segment_distance(p, p, pts[i, :3], pts[i + 1, :3]) <= r:
                return True
    return False


def retract_clearance_z(*, current_z: float, home_z: float, lift_m: float) -> float:
    """Lift plane: ``home_z + lift_m``, or current z if already above it.

    Do not add ``lift_m`` on top of the live EEF: LIBERO ready pose is already
    ~1.18 m, and ``current_z + 0.28`` is past the Panda workspace so the
    retract leg never finishes.
    """
    clearance = float(home_z) + float(lift_m)
    return max(float(current_z), clearance)


def disturbance_onset_step(controller: Any | None) -> int | None:
    """L2 ``move_start_step`` / L3 ``_entry_start_step`` (task-clock onset)."""
    if controller is None:
        return None
    for attr in ("move_start_step", "_entry_start_step", "_move_start_step"):
        value = getattr(controller, attr, None)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def is_imminent(
    *,
    controller: Any | None,
    task_step: int | None,
    lead_steps: int,
) -> bool:
    """True once ``task_step`` is within ``lead_steps`` of disturbance onset."""
    if task_step is None or int(lead_steps) < 0:
        return False
    start = disturbance_onset_step(controller)
    if start is None:
        return False
    return int(task_step) >= int(start) - int(lead_steps)


def clip_joint_step(
    current: np.ndarray,
    target: np.ndarray,
    max_step_rad: float,
) -> np.ndarray:
    """Per-joint clip toward ``target`` (fast coordinated snap)."""
    cur = np.asarray(current, dtype=np.float64).reshape(-1)
    tgt = np.asarray(target, dtype=np.float64).reshape(-1)
    if cur.size != tgt.size:
        raise ValueError(f"joint size mismatch: current={cur.size} target={tgt.size}")
    cap = float(max_step_rad)
    if cap < 0.0:
        raise ValueError(f"max_step_rad must be >= 0, got {cap}")
    return cur + np.clip(tgt - cur, -cap, cap)


def apply_arm_qpos(env: Any, qpos: Sequence[float]) -> None:
    """Write Panda arm qpos through robosuite ``set_robot_joint_positions``."""
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    current = env
    seen: set[int] = set()
    for _ in range(8):
        if current is None:
            break
        ident = id(current)
        if ident in seen:
            break
        seen.add(ident)
        robots = getattr(current, "robots", None)
        if robots:
            robot = robots[0]
            setter = getattr(robot, "set_robot_joint_positions", None)
            if callable(setter):
                setter(q)
                return
        current = getattr(current, "env", None)
    raise RuntimeError("apply_arm_qpos: no robot.set_robot_joint_positions on env")


def refresh_libero_obs(env: Any) -> Mapping[str, Any]:
    """Re-read observations after a privileged qpos write."""
    current = env
    seen: set[int] = set()
    for _ in range(8):
        if current is None:
            break
        ident = id(current)
        if ident in seen:
            break
        seen.add(ident)
        getter = getattr(current, "_get_observations", None)
        if callable(getter):
            obs = getter()
            if isinstance(obs, Mapping):
                return obs
        current = getattr(current, "env", None)
    raise RuntimeError("refresh_libero_obs: no _get_observations on env")


def reset_task_horizon_clock(env: Any) -> tuple[str, ...]:
    """Zero wrapper ``timestep`` counters without touching qpos / disturbance.

    robosuite/LIBERO ``done`` on horizon uses this counter. Returning the arm
    to the start EEF must not inherit the pre-yield elapsed steps.
    """
    reset: list[str] = []
    seen: set[int] = set()
    current = env
    for _ in range(8):
        if current is None:
            break
        ident = id(current)
        if ident in seen:
            break
        seen.add(ident)
        if hasattr(current, "timestep"):
            current.timestep = 0
            reset.append("timestep")
        elapsed = getattr(current, "_elapsed_steps", None)
        if isinstance(elapsed, int) and not isinstance(elapsed, bool):
            current._elapsed_steps = 0
            reset.append("_elapsed_steps")
        current = getattr(current, "env", None)
    return tuple(reset)


def extend_sim_step_cap(
    *,
    current_t: int,
    max_steps: int,
    current_cap: int,
    horizon: int | None,
) -> int:
    """Give a full ``max_steps`` of loop room after a yield resume."""
    desired = int(current_t) + int(max_steps)
    cap = max(int(current_cap), desired)
    if horizon is None:
        return cap
    return min(cap, int(current_t) + max(int(horizon) - 1, int(max_steps)))


def _eef_pos_quat_grip(obs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    g = float(grip[0]) if grip.size else 0.0
    return pos, quat, g


def _joint_pos(obs: Mapping[str, Any]) -> list[float] | None:
    raw = obs.get("robot0_joint_pos")
    if raw is None:
        raw = obs.get("robot0_joint_positions")
    if raw is None:
        return None
    return [float(v) for v in np.asarray(raw, dtype=np.float64).reshape(-1)]


def choose_safe_target(
    *,
    home_pos: np.ndarray,
    current_pos: np.ndarray,
    dwell_pos: np.ndarray | None,
    corridor_radius_m: float,
    lateral_offset_m: float,
    lift_m: float,
    hazard_path: np.ndarray | None = None,
) -> tuple[np.ndarray, str, bool]:
    """Default: episode-start EEF. Lateral+lift if home path hits dwell or L2 slide."""
    home = np.asarray(home_pos, dtype=np.float64).reshape(3)
    current = np.asarray(current_pos, dtype=np.float64).reshape(3)
    hit_center: np.ndarray | None = None
    if dwell_pos is not None:
        dwell = np.asarray(dwell_pos, dtype=np.float64).reshape(3)
        if segment_hits_sphere(current, home, dwell, corridor_radius_m):
            hit_center = dwell
    if hazard_path is not None:
        hit, pt = polyline_hits_segment(current, home, hazard_path, corridor_radius_m)
        if hit and pt is not None:
            hit_center = pt
    if hit_center is None:
        return home.copy(), "home", False
    away = home[:2] - hit_center[:2]
    n = float(np.linalg.norm(away))
    if n < 1e-8:
        delta = home[:2] - current[:2]
        if float(np.linalg.norm(delta)) < 1e-8:
            away = np.array([1.0, 0.0], dtype=np.float64)
            n = 1.0
        else:
            away = np.array([-delta[1], delta[0]], dtype=np.float64)
            n = float(np.linalg.norm(away))
    unit = away / n
    target = home.copy()
    target[0] += float(lateral_offset_m) * float(unit[0])
    target[1] += float(lateral_offset_m) * float(unit[1])
    target[2] = float(home[2]) + float(lift_m)
    return target, "lateral_lift", True


@dataclass
class YieldHomeConfig:
    enabled: bool
    levels: frozenset[int]
    pos_tol_m: float
    ori_tol_rad: float
    max_step_m: float
    max_step_rad: float
    kp_pos: float
    kp_ori: float
    retreat_max_steps: int
    corridor_radius_m: float
    lateral_offset_m: float
    lift_m: float
    default_gripper_hold: float
    reset_task_clock: bool
    engage_lead_steps: int
    max_joint_step_rad: float
    joint_tol_rad: float

    @classmethod
    def from_extra(cls, extra_libero: Mapping[str, Any]) -> YieldHomeConfig:
        block = extra_libero.get("yield_home")
        if not isinstance(block, Mapping):
            raise TypeError("extra.libero.yield_home must be a mapping when enabled")
        missing = [key for key in _REQUIRED_KEYS if key not in block]
        if missing:
            raise ValueError(f"extra.libero.yield_home missing keys: {missing}")
        return cls(
            enabled=bool(_require(block, "enabled")),
            levels=_require_levels(block),
            pos_tol_m=_require_float(block, "pos_tol_m", min_value=0.0),
            ori_tol_rad=_require_float(block, "ori_tol_rad", min_value=0.0),
            max_step_m=_require_float(block, "max_step_m", min_value=0.0),
            max_step_rad=_require_float(block, "max_step_rad", min_value=0.0),
            kp_pos=_require_float(block, "kp_pos", min_value=0.0),
            kp_ori=_require_float(block, "kp_ori", min_value=0.0),
            retreat_max_steps=_require_int(block, "retreat_max_steps", min_value=1),
            corridor_radius_m=_require_float(block, "corridor_radius_m", min_value=0.0),
            lateral_offset_m=_require_float(block, "lateral_offset_m", min_value=0.0),
            lift_m=_require_float(block, "lift_m", min_value=0.0),
            default_gripper_hold=_require_float(block, "default_gripper_hold"),
            reset_task_clock=_require_bool(block, "reset_task_clock"),
            engage_lead_steps=_require_int(block, "engage_lead_steps", min_value=0),
            max_joint_step_rad=_require_float(block, "max_joint_step_rad", min_value=0.0),
            joint_tol_rad=_require_float(block, "joint_tol_rad", min_value=0.0),
        )


@dataclass
class LiberoYieldHomeSession:
    """Per-episode bookkeeping; when disabled every hook is a no-op."""

    enabled: bool
    cfg: YieldHomeConfig | None
    level: int
    home_pos: np.ndarray | None = None
    home_aa: np.ndarray | None = None
    home_joints: list[float] | None = None
    retract_pos: np.ndarray | None = None
    target_pos: np.ndarray | None = None
    target_kind: str | None = None
    motion_leg: str | None = None
    crossed_corridor: bool = False
    state: YieldHomeState = YieldHomeState.PASS_THROUGH
    yield_steps: int = 0
    hold_steps: int = 0
    arrived: bool = False
    timed_out: bool = False
    resume_occurred: bool = False
    clock_reset: bool = False
    clock_reset_wall_step: int | None = None
    _was_taking: bool = False
    _engaged: bool = False
    _in_corridor: bool = False
    _keepout_dwell: np.ndarray | None = None
    _keepout_path: np.ndarray | None = None
    last_pos_err_m: float | None = None
    last_ori_err_rad: float | None = None
    last_joint_err_rad: float | None = None
    pending_qpos: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        *,
        cfg: RunConfig,
        extra_libero: Mapping[str, Any],
        oracle_enabled: bool,
        aegis_enabled: bool,
    ) -> LiberoYieldHomeSession:
        level = int(cfg.level)
        enabled = yield_home_enabled_from_extra(extra_libero)
        if not enabled:
            return cls(enabled=False, cfg=None, level=level)
        if oracle_enabled:
            raise ValueError(
                "extra.libero.yield_home.enabled cannot combine with extra.libero.oracle.enabled; "
                "this arm replaces stop-wait, it does not wrap it"
            )
        # ``aegis_enabled`` is accepted: CBF runs only after resume (backend +
        # AEGIS phase gate). Keep the argument so callers stay explicit.
        # Fail closed if someone enabled the flag without going through from_extra keys.
        parsed = YieldHomeConfig.from_extra(extra_libero)
        if not parsed.enabled:
            return cls(enabled=False, cfg=None, level=level)
        return cls(enabled=True, cfg=parsed, level=level)

    def capture_home_if_needed(self, obs: Mapping[str, Any]) -> None:
        if not self.enabled or self.home_pos is not None:
            return
        pos, quat, _grip = _eef_pos_quat_grip(obs)
        self.home_pos = pos.copy()
        self.home_aa = quat2axisangle(quat)
        self.home_joints = _joint_pos(obs)

    def applies_to_level(self) -> bool:
        if not self.enabled or self.cfg is None:
            return False
        return int(self.level) in self.cfg.levels

    def allows_aegis_filter(self) -> bool:
        """AEGIS may filter VLA actions only after yield has handed control back.

        When yield does not apply (disabled / other level), AEGIS is unrestricted
        so the existing oracle-off static对照 path is unchanged.
        """
        if not self.applies_to_level():
            return True
        return bool(self.resume_occurred)

    def should_takeover(
        self,
        phase: LiberoOraclePhase | None,
        *,
        controller: Any | None = None,
        task_step: int | None = None,
    ) -> bool:
        """Latch before onset; hold until parked and the motion window is over.

        Engage on ``entering_or_active`` or when ``task_step`` is within
        ``engage_lead_steps`` of L2/L3 onset. After engage, do **not** resume
        during remaining pre-move INACTIVE — wait for settled/withdrawn so
        the arm stays home while the body transits.
        """
        if not self.applies_to_level() or self.cfg is None:
            return False
        live = phase is LiberoOraclePhase.ENTERING_OR_ACTIVE
        imminent = is_imminent(
            controller=controller,
            task_step=task_step,
            lead_steps=self.cfg.engage_lead_steps,
        )
        if live or imminent:
            self._engaged = True
        if not self._engaged:
            return False
        parked = bool(self.arrived or (self.timed_out and not self._in_corridor))
        hazard_over = phase in (
            LiberoOraclePhase.SETTLED_OR_DONE,
            LiberoOraclePhase.WITHDRAWN,
        )
        if parked and hazard_over:
            return False
        return True

    def consume_resume_clear(self, *, taking: bool) -> bool:
        """True on the first non-takeover step after a yield/hold (flush VLA chunks)."""
        clear = self._was_taking and not taking
        if clear:
            self.resume_occurred = True
            self.state = YieldHomeState.PASS_THROUGH
        self._was_taking = bool(taking)
        return bool(clear)

    def should_reset_task_clock(self) -> bool:
        return bool(self.enabled and self.cfg is not None and self.cfg.reset_task_clock)

    def note_clock_reset(self, *, wall_task_step: int) -> None:
        self.clock_reset = True
        self.clock_reset_wall_step = int(wall_task_step)

    def command(
        self,
        obs: Mapping[str, Any],
        *,
        controller: Any | None,
        action_space: str,
        last_gripper: float | None,
    ) -> np.ndarray:
        if not self.enabled or self.cfg is None:
            raise RuntimeError("command() called on a disabled yield-home session")
        space = str(action_space)
        if space not in _FILTERABLE:
            raise ValueError(
                f"yield-home only emits {_FILTERABLE}, got action_space={space!r}"
            )
        self.capture_home_if_needed(obs)
        if self.home_pos is None or self.home_aa is None:
            raise RuntimeError("yield-home home pose was not captured")
        if self.home_joints is None:
            raise RuntimeError("yield-home home joints were not captured")
        pos, quat, _g = _eef_pos_quat_grip(obs)
        aa = quat2axisangle(quat)
        q_now = _joint_pos(obs)
        if q_now is None:
            raise RuntimeError("yield-home needs robot0_joint_pos on observations")
        self._plan_targets(pos, controller)
        self._in_corridor = point_in_keepout(
            pos,
            dwell_pos=self._keepout_dwell,
            hazard_path=self._keepout_path,
            radius=self.cfg.corridor_radius_m,
        )

        err_final = (
            np.zeros(3, dtype=np.float64)
            if self.target_pos is None
            else self.target_pos - pos
        )
        err_ori = self.home_aa - aa
        self.last_pos_err_m = float(np.linalg.norm(err_final))
        self.last_ori_err_rad = float(np.linalg.norm(err_ori))
        q_home = np.asarray(self.home_joints, dtype=np.float64).reshape(-1)
        q_cur = np.asarray(q_now, dtype=np.float64).reshape(-1)
        self.last_joint_err_rad = float(np.max(np.abs(q_home - q_cur)))
        self.arrived = self.last_joint_err_rad <= float(self.cfg.joint_tol_rad)
        self.motion_leg = "home" if self.arrived else "joints"

        at_cap = self.yield_steps >= self.cfg.retreat_max_steps
        if at_cap and not self.arrived:
            self.timed_out = True
        hold = self.arrived or (at_cap and not self._in_corridor)
        gripper = (
            float(last_gripper)
            if last_gripper is not None
            else float(self.cfg.default_gripper_hold)
        )
        if hold:
            self.state = YieldHomeState.HOLD_SAFE
            self.hold_steps += 1
            self.pending_qpos = q_home.copy()
            delta = np.zeros(6, dtype=np.float64)
        else:
            self.state = YieldHomeState.YIELD
            self.yield_steps += 1
            self.pending_qpos = clip_joint_step(
                q_cur, q_home, self.cfg.max_joint_step_rad
            )
            delta = np.zeros(6, dtype=np.float64)
        if space == "abs_ee_7d":
            out = np.concatenate([pos + delta[:3], aa + delta[3:], [gripper]])
        else:
            out = np.concatenate([delta, [gripper]])
        return np.asarray(out, dtype=np.float64)

    def _plan_targets(self, pos: np.ndarray, controller: Any | None) -> None:
        if self.target_pos is not None or self.cfg is None or self.home_pos is None:
            return
        dwell, path = _hazard_geometry(controller)
        self._keepout_dwell = dwell
        self._keepout_path = path
        retract = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        retract[2] = retract_clearance_z(
            current_z=float(pos[2]),
            home_z=float(self.home_pos[2]),
            lift_m=float(self.cfg.lift_m),
        )
        self.retract_pos = retract
        # Keep-out is evaluated on the post-lift segment so a table-height
        # slide does not force a detour that still cuts the corridor on the
        # way up.
        target, kind, crossed = choose_safe_target(
            home_pos=self.home_pos,
            current_pos=retract,
            dwell_pos=dwell,
            corridor_radius_m=self.cfg.corridor_radius_m,
            lateral_offset_m=self.cfg.lateral_offset_m,
            lift_m=self.cfg.lift_m,
            hazard_path=path,
        )
        self.target_pos = target
        self.target_kind = kind
        self.crossed_corridor = bool(crossed)

    def episode_extra(self) -> dict[str, Any]:
        if not self.enabled:
            return {"yield_home_enabled": False}
        return {
            "yield_home_enabled": True,
            "yield_home_state": self.state.value,
            "yield_home_target_kind": self.target_kind,
            "yield_home_motion_leg": self.motion_leg,
            "yield_home_in_corridor": bool(self._in_corridor),
            "yield_home_crossed_corridor": bool(self.crossed_corridor),
            "yield_home_yield_steps": int(self.yield_steps),
            "yield_home_hold_steps": int(self.hold_steps),
            "yield_home_arrived": bool(self.arrived),
            "yield_home_timeout": bool(self.timed_out),
            "yield_home_resume_occurred": bool(self.resume_occurred),
            "yield_home_clock_reset": bool(self.clock_reset),
            "yield_home_clock_reset_wall_step": self.clock_reset_wall_step,
            "yield_home_home_eef": None
            if self.home_pos is None
            else [float(v) for v in self.home_pos],
            "yield_home_home_joints": None
            if self.home_joints is None
            else list(self.home_joints),
            "yield_home_last_pos_err_m": self.last_pos_err_m,
            "yield_home_last_ori_err_rad": self.last_ori_err_rad,
            "yield_home_last_joint_err_rad": self.last_joint_err_rad,
        }
