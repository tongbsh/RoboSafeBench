"""HazardHost contract (docs/design/01_integration_rules.md §3) and its data types.

Interfaces only: nothing here simulates, steps or judges anything. Conventions
for every quantity that crosses this boundary: SI units (m, s, rad, kg, N·s),
world frame with z up, quaternions ``wxyz``, ``env`` is the environment index
(always 0 for single-env benchmarks).

Required methods (``HazardHost``) are the Tier-A set of §3.1. Optional
capabilities (§3.2) live in ``OptionalHazardHost`` and are looked up with
``getattr``; a missing optional method degrades the recorded metric (e.g.
``separation_kind=surrogate``) but never silently: the BD must say so (R15).
"""
import dataclasses
import enum
import math
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence,
                    runtime_checkable)

import numpy as np

from ._records import (UNKNOWN, Record, SpecError, check_nonempty_str, check_vec, require)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle with spec/
    from .spec.benchmark_descriptor import BenchmarkDescriptor
    from .spec.hazard_spec import AgentSpec


class Tier(str, enum.Enum):
    """Benchmark capability tier (01 §2). A: full contact semantics + severity;
    B: kinematic hazards only (near-miss metrics, no severity); C: out of scope."""
    A = "A"
    B = "B"
    C = "C"


class CapabilityError(RuntimeError):
    """A declared/required capability is missing. Raised at start-up, never
    swallowed and never replaced by a default (rule R15)."""


class PartnerClass(str, enum.Enum):
    """What a hazard contact partner is (paradigm §6 robot-side definition)."""
    ROBOT_LINK = "robot_link"
    HELD_OBJECT = "held_object"
    SCENE = "scene"
    HAZARD = "hazard"


Action = Any  # benchmark-native action; opaque to the core


def _vec(name: str, value: Any, length: int) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise SpecError("%s: cannot convert %r to a float vector" % (name, value)) from None
    if arr.shape != (length,):
        raise SpecError("%s: expected shape (%d,), got %s" % (name, length, arr.shape))
    if not np.all(np.isfinite(arr)):
        raise SpecError("%s: contains non-finite values %s" % (name, arr))
    return arr


def _finite(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SpecError("%s: cannot convert %r to float" % (name, value)) from None
    if not math.isfinite(number):
        raise SpecError("%s: non-finite value %r" % (name, number))
    return number


@dataclasses.dataclass(eq=False)
class Pose:
    """Position (m) and unit quaternion ``wxyz`` in the world frame.
    Shapes are enforced; unit norm is NOT (tolerances belong to conformance)."""
    pos: Any
    quat: Any

    def __post_init__(self) -> None:
        self.pos = _vec("Pose.pos", self.pos, 3)
        self.quat = _vec("Pose.quat", self.quat, 4)


@dataclasses.dataclass(eq=False)
class HazardState:
    """World-frame state of a hazard body's centre of mass (D10 e).

    During the kinematic phase it is the pose just set with zero velocities;
    after ``release_to_physics`` it is what the engine reports. Shapes and
    finiteness are enforced like ``Pose``; unit norm is NOT."""
    pos: Any
    quat: Any
    lin_vel_mps: Any
    ang_vel_radps: Any

    def __post_init__(self) -> None:
        self.pos = _vec("HazardState.pos", self.pos, 3)
        self.quat = _vec("HazardState.quat", self.quat, 4)
        self.lin_vel_mps = _vec("HazardState.lin_vel_mps", self.lin_vel_mps, 3)
        self.ang_vel_radps = _vec("HazardState.ang_vel_radps", self.ang_vel_radps, 3)


@dataclasses.dataclass(frozen=True)
class HazardHandle:
    """Opaque reference to one hazard body. ``native`` carries whatever the
    adapter needs (body id, actor, prim path…) and is ignored by eq/hash."""
    env: int
    key: str
    native: Any = dataclasses.field(default=None, compare=False, repr=False)


@dataclasses.dataclass(eq=False)
class RobotState:
    """Robot-side state at one instant.

    ``collision_body_pos`` is the (N, 3) list of robot collision-body positions
    (all arm/gripper/base bodies per the BD's ``robot_contact_bodies``).
    ``collision_body_names`` / ``collision_body_lin_vel_mps`` are OPTIONAL
    extensions (not in 01 §3.1): names parallel to the positions and the
    (N, 3) linear velocities of those bodies, which the core needs to complete
    the relative normal velocity (see report T01, interface decisions).
    ``held_object_bodies`` (optional, D10 e) lists every collision body name of
    the held object when it has more than one; contacts with them classify as
    ``held_object``. When absent the runtime uses ``[held_object]``.
    ``collision_body_quat`` (optional, D15 a) is the (N, 4) ``wxyz`` world
    orientation of each collision body, parallel to ``collision_body_pos``;
    finite with non-zero norm (unit norm is NOT enforced, like ``Pose``). It is
    required when the host offers ``robot_collision_points`` (the core then
    raises instead of falling back, see :func:`robot_geometry_offered`).
    """
    eef_pose: Pose
    eef_lin_vel_mps: Any
    collision_body_pos: Any
    gripper_opening_m: Optional[float] = None
    base_pose: Optional[Pose] = None
    held_object: Optional[str] = None
    collision_body_names: Optional[List[str]] = None
    collision_body_lin_vel_mps: Any = None
    held_object_bodies: Optional[List[str]] = None
    collision_body_quat: Any = None

    def __post_init__(self) -> None:
        self.eef_lin_vel_mps = _vec("RobotState.eef_lin_vel_mps", self.eef_lin_vel_mps, 3)
        pos = np.asarray(self.collision_body_pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3 or not np.all(np.isfinite(pos)):
            raise SpecError("RobotState.collision_body_pos: expected finite (N, 3), got shape %s"
                            % (pos.shape,))
        self.collision_body_pos = pos
        if self.gripper_opening_m is not None:
            self.gripper_opening_m = _finite("RobotState.gripper_opening_m", self.gripper_opening_m)
        if self.collision_body_names is not None:
            if len(self.collision_body_names) != pos.shape[0]:
                raise SpecError("RobotState.collision_body_names: %d names for %d positions"
                                % (len(self.collision_body_names), pos.shape[0]))
        if self.collision_body_lin_vel_mps is not None:
            vel = np.asarray(self.collision_body_lin_vel_mps, dtype=np.float64)
            if vel.shape != pos.shape or not np.all(np.isfinite(vel)):
                raise SpecError("RobotState.collision_body_lin_vel_mps: expected finite shape %s, got %s"
                                % (pos.shape, vel.shape))
            self.collision_body_lin_vel_mps = vel
        if self.held_object_bodies is not None:
            require(isinstance(self.held_object_bodies, (list, tuple)),
                    "RobotState.held_object_bodies: expected a list of names, got %r"
                    % (self.held_object_bodies,))
            self.held_object_bodies = list(self.held_object_bodies)
            for i, name in enumerate(self.held_object_bodies):
                check_nonempty_str("RobotState.held_object_bodies[%d]" % i, name)
        if self.collision_body_quat is not None:
            try:
                quat = np.asarray(self.collision_body_quat, dtype=np.float64)
            except (TypeError, ValueError):
                raise SpecError("RobotState.collision_body_quat: cannot convert %r to a float array"
                                % (self.collision_body_quat,)) from None
            if quat.shape != (pos.shape[0], 4) or not np.all(np.isfinite(quat)):
                raise SpecError("RobotState.collision_body_quat: expected finite shape (%d, 4) (wxyz), got %s"
                                % (pos.shape[0], quat.shape))
            if not np.all(np.linalg.norm(quat, axis=1) > 0):
                raise SpecError("RobotState.collision_body_quat: contains a zero quaternion")
            self.collision_body_quat = quat


@dataclasses.dataclass(eq=False)
class ContactObs:
    """One contact seen by a hazard body since the previous ``hazard_contacts`` call.

    ``normal`` is the world-frame direction in which the partner pushes the
    hazard (points from partner toward hazard). ``impulse_ns`` and
    ``rel_normal_speed_mps`` are what the engine reports, if anything; the core
    recomputes the relative normal speed from kinematic data (01 §3.1).
    """
    partner_name: str
    partner_class: PartnerClass
    position: Any
    normal: Any
    time_s: float
    impulse_ns: Optional[float] = None
    rel_normal_speed_mps: Optional[float] = None

    def __post_init__(self) -> None:
        check_nonempty_str("ContactObs.partner_name", self.partner_name)
        if not isinstance(self.partner_class, PartnerClass):
            try:
                self.partner_class = PartnerClass(self.partner_class)
            except ValueError:
                raise SpecError("ContactObs.partner_class: %r is not one of %s"
                                % (self.partner_class, [c.value for c in PartnerClass])) from None
        self.position = _vec("ContactObs.position", self.position, 3)
        self.normal = _vec("ContactObs.normal", self.normal, 3)
        self.time_s = _finite("ContactObs.time_s", self.time_s)
        if self.impulse_ns is not None:
            self.impulse_ns = _finite("ContactObs.impulse_ns", self.impulse_ns)
        if self.rel_normal_speed_mps is not None:
            self.rel_normal_speed_mps = _finite("ContactObs.rel_normal_speed_mps",
                                                self.rel_normal_speed_mps)


@dataclasses.dataclass(eq=False)
class TimeState:
    """Simulator clock. All three values in seconds."""
    sim_time_s: float
    physics_dt_s: float
    control_dt_s: float

    def __post_init__(self) -> None:
        self.sim_time_s = _finite("TimeState.sim_time_s", self.sim_time_s)
        self.physics_dt_s = _finite("TimeState.physics_dt_s", self.physics_dt_s)
        self.control_dt_s = _finite("TimeState.control_dt_s", self.control_dt_s)
        require(self.physics_dt_s > 0, "TimeState.physics_dt_s must be > 0")
        require(self.control_dt_s > 0, "TimeState.control_dt_s must be > 0")


@dataclasses.dataclass
class Surface(Record):
    """An upward-facing support surface (table, shelf, counter top…) as a
    world-frame axis-aligned box ``[min_m, max_m]``; the top height is
    ``max_m[2]``. Rotated surfaces are not representable (known limitation).
    ``source`` records where it came from (e.g. a scene YAML, or probe estimate)."""
    name: str
    min_m: List[float]
    max_m: List[float]
    source: str

    def validate(self) -> None:
        check_nonempty_str("name", self.name)
        check_nonempty_str("source", self.source)
        check_vec("min_m", self.min_m, 3)
        check_vec("max_m", self.max_m, 3)
        for axis in range(3):
            require(self.min_m[axis] <= self.max_m[axis],
                    "min_m[%d]=%r exceeds max_m[%d]=%r" % (axis, self.min_m[axis], axis, self.max_m[axis]))


# --------------------------------------------------------------------------
# protocols
# --------------------------------------------------------------------------

@runtime_checkable
class HazardHost(Protocol):
    """Engine primitives an adapter must provide (Tier A, 01 §3.1).

    ``isinstance(x, HazardHost)`` only checks that the methods exist; signatures
    are checked by the conformance suite (T05). Tier-B adapters cannot honour
    ``release_to_physics`` / ``hazard_contacts`` semantics and must say so in
    their BD instead of faking them.
    """

    def describe(self) -> "BenchmarkDescriptor":
        """Return the BD; the runtime cross-checks its timebase etc. (C1)."""
        ...

    def alloc_hazard(self, env: int, agent: "AgentSpec") -> HazardHandle:
        """Create or claim from a pre-placed pool one hazard body shaped by
        ``agent``. Must be callable before the episode starts."""
        ...

    def free_hazard(self, h: HazardHandle) -> None:
        """Return/destroy the body and restore the parked state."""
        ...

    def set_hazard_pose(self, h: HazardHandle, pos: Sequence[float], quat: Sequence[float]) -> None:
        """Kinematic teleport. The body is kinematic afterwards; no residual velocity."""
        ...

    def set_hazard_visible(self, h: HazardHandle, visible: bool) -> None:
        """Park/unpark. A parked body must not interact with anything and must
        be outside every camera view."""
        ...

    def release_to_physics(self, h: HazardHandle, lin_vel: Sequence[float],
                           ang_vel: Sequence[float]) -> None:
        """Become dynamic with exactly the given velocities (no floor applied)."""
        ...

    def robot_state(self, env: int) -> RobotState:
        ...

    def object_poses(self, env: int) -> Dict[str, Pose]:
        """Named task objects and fixtures (support surfaces go via ``support_surfaces``)."""
        ...

    def hazard_contacts(self, h: HazardHandle) -> List[ContactObs]:
        """Contacts since the previous call."""
        ...

    def hazard_state(self, h: HazardHandle) -> HazardState:
        """World-frame pose and velocities of the hazard's centre of mass (D10 e)."""
        ...

    def time_state(self, env: int) -> TimeState:
        ...

    def task_success(self, env: int) -> bool:
        """Read-only forward of the benchmark's own success test (R5)."""
        ...

    def hold_action(self, env: int) -> Action:
        """Benchmark-native hold / no-op action (T_stop, Oracle gate)."""
        ...

    def install_hooks(self, runtime: Any) -> None:
        """Attach the runtime's reset / pre-step / post-step / episode-end calls
        to the benchmark's reset and *finest* physics step. ``runtime`` is typed
        ``Any`` until T04 defines ``HazardRuntime`` (see report T01)."""
        ...


@runtime_checkable
class OptionalHazardHost(Protocol):
    """Optional capabilities (01 §3.2). Never required; look up with ``getattr``."""

    def min_distance(self, h: HazardHandle) -> float:
        """True-geometry minimum distance from the hazard to the robot (m)."""
        ...

    def snapshot(self, env: int) -> Any:
        ...

    def restore(self, env: int, snap: Any) -> None:
        ...

    def support_surfaces(self, env: int) -> List[Surface]:
        ...

    def render_check(self, h: HazardHandle) -> bool:
        """Whether the hazard appears in the policy cameras."""
        ...

    def robot_collision_points(self, env: int) -> Dict[str, np.ndarray]:
        """Surface sample points (M, 3) of every robot collision body (held objects excluded), in that body's own
        frame, constant within an episode, at most BD ``robot_point_spacing_m`` apart; keys are names of
        ``RobotState.collision_body_names``. A host offering it must fill ``RobotState.collision_body_quat``
        (D15 a)."""
        ...


@runtime_checkable
class EpisodeDriver(Protocol):
    """Episode driving an adapter offers to the conformance suite, gates and
    T_stop calibration (D11 a). Declared apart from ``HazardHost``: it is not
    part of ``REQUIRED_METHODS``; check it with ``require_methods(x, DRIVER_METHODS)``.

    ``reset_episode`` fires the installed reset hook (``HazardRuntime.begin_episode``),
    so a hazard instance must be armed on the runtime before it is called.
    ``step`` advances one control step and fires the per-physics-step hooks.
    """

    def reset_episode(self, env: int, task_id: str, layout_id: str, seed: int) -> None:
        ...

    def step(self, env: int, action: Action) -> None:
        ...

    def policy_observation(self, env: int) -> Dict[str, Any]:
        """Exactly what the policy would receive (R5: no hazard state in vectors)."""
        ...


DRIVER_METHODS = ("reset_episode", "step", "policy_observation")


REQUIRED_METHODS = (
    "describe", "alloc_hazard", "free_hazard", "set_hazard_pose", "set_hazard_visible",
    "release_to_physics", "robot_state", "object_poses", "hazard_contacts", "hazard_state",
    "time_state", "task_success", "hold_action", "install_hooks",
)
OPTIONAL_METHODS = ("min_distance", "snapshot", "restore", "support_surfaces", "render_check",
                    "robot_collision_points")


def missing_methods(host: Any, names: Sequence[str] = REQUIRED_METHODS) -> List[str]:
    """Names in ``names`` that ``host`` does not provide as callables."""
    return [n for n in names if not callable(getattr(host, n, None))]


def require_methods(host: Any, names: Sequence[str] = REQUIRED_METHODS) -> None:
    """Fail loudly at start-up if the host lacks required methods (R15). ``names``
    defaults to the HazardHost set; pass ``DRIVER_METHODS`` for the episode driver."""
    missing = missing_methods(host, names)
    if missing:
        raise CapabilityError("%s is missing required method(s): %s"
                              % (type(host).__name__, ", ".join(missing)))


def robot_geometry_offered(host: Any, bd: Any) -> bool:
    """Whether ``host`` provides robot geometry (D15 a): ``robot_collision_points`` is callable and the BD gives a
    numeric ``robot_point_spacing_m``. The two must agree; a host that offers the method under a BD saying
    ``unknown`` (or the reverse) is a :class:`CapabilityError`, never a silent fallback to the scalar standoff."""
    method = callable(getattr(host, "robot_collision_points", None))
    spacing = getattr(bd, "robot_point_spacing_m", UNKNOWN)
    declared = not (isinstance(spacing, str) and spacing == UNKNOWN)
    if method != declared:
        raise CapabilityError(
            "%s %s robot_collision_points but its BD %r says robot_point_spacing_m = %r; the two must agree "
            "(D15 a)" % (type(host).__name__, "offers" if method else "does not offer",
                         getattr(bd, "name", "?"), spacing))
    return method


def require_body_quats(robot: RobotState, host: Any) -> np.ndarray:
    """``robot.collision_body_quat``, which a host offering ``robot_collision_points`` must report (D15 a)."""
    if robot.collision_body_quat is None or robot.collision_body_names is None:
        raise CapabilityError("%s offers robot_collision_points but robot_state() reports no collision_body_quat "
                              "(or no collision_body_names); the core needs body orientations to place the "
                              "geometry (D15 a)" % (type(host).__name__,))
    return robot.collision_body_quat
