"""FakeHost: a pure-numpy world implementing ``HazardHost``, the optional capabilities and ``EpisodeDriver``.

It exists to test ``HazardRuntime``, the conformance suite (T05) and the pipeline without a simulator.
Nothing in it is a measured or default value: every number comes from an explicit :class:`FakeWorldConfig`.

World (one independent copy per environment index):

* **robot**: an end effector at ``eef`` carrying collision spheres at fixed offsets (``robot_bodies``: a
  hand, links, a camera...) and optionally a held object made of spheres (``held_bodies``). The robot is
  kinematic and infinitely heavy. A :class:`FakeAction` moves the EEF for one physics step: ``follow`` the
  task's nominal path (seconds from the reset), ``hold`` still, or a constant ``velocity``.
* **scene**: an infinite table plane at ``table_height_m`` (reported as a support surface with the
  configured extent) and static task-object spheres. The first time a dynamic hazard hits a task object, its
  orientation is multiplied by the object's configured ``on_hit_rotation_wxyz`` (topple / yaw tests).
* **hazards**: spheres only (other shapes raise ``CapabilityError``). A parked hazard sits at
  ``park_pos_m``, is not rendered and touches nothing. A kinematic hazard stays where it was posed; overlaps
  are checked at the end of every physics step. A dynamic hazard flies ballistically under
  ``gravity_mps2`` (exact per step) with **continuous collision detection**: the first time of impact with
  the moving robot spheres, the task objects and the table is solved exactly (polynomial roots), the
  velocity is reflected with ``restitution`` against the infinitely heavy partner, and the flight continues
  for the rest of the step. A hazard that would land again within one physics step comes to rest: on the
  table (gravity balanced, one table contact per step) or, when gravity presses it into a robot or object
  sphere, on that sphere, carried rigidly with it (one contact per step, no further collision checks) until
  it is posed, hidden or released again. Without CCD only the end of each step is checked.

Contacts are ``ContactObs`` with the absolute time of impact, the partner name, the partner class from the
BD's ``robot_contact_bodies`` (the held bodies are always ``held_object``) and the normal from the partner
toward the hazard. The partner name of the table is ``table``; hazards are named by their handle key.

Mutations (:data:`MUTATIONS`), one injected defect each, for the conformance suite (T05):

``tunneling``                 no CCD for dynamic hazards while the BD still declares ``hazard_ccd: true``
``residual_velocity``         ``set_hazard_pose`` leaves the teleport velocity (displacement / physics step) on
                              the body, visible in ``hazard_state`` and added to the next release
``state_leak``                the hazard position is appended to the policy state vector
``wrong_dt``                  ``describe()`` declares twice the real physics step
``no_rollback``               after a robot-side overlap was reported, ``set_hazard_pose`` is ignored until the
                              next physics step
``park_interacts``            a parked hazard still collides and is still rendered (at the parking pose)
``oracle_depends_on_hazard``  ``task_success`` is false while any hazard is visible
``bad_standoff``              ``nominal_hit_instance`` aims the hazard centroid at the EEF reference point,
                              ignoring the standoff ``r_hazard + eef_contact_radius_m`` (D1)
``hook_dependent_release``    ``release_to_physics`` ignores the given velocity and uses the last two posed
                              positions divided by the physics step
``post_impact_vn``            ``hazard_contacts`` returns the contacts one hook late
"""
import copy
import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .._numeric import finite_float, non_negative_float, positive_float, positive_int, unit_vec3, vec3
from ..host import (OPTIONAL_METHODS, CapabilityError, ContactObs, HazardHandle, HazardState, PartnerClass, Pose,
                    RobotState, Surface, TimeState)
from ..kinematics.hit import FAMILY_FALL, build_hit_path
from ..kinematics.path import SampledPath
from ..kinematics.subdivide import hook_times
from ..semantics.events import classify_partner
from ..spec.benchmark_descriptor import BenchmarkDescriptor
from ..spec.hazard_spec import AgentSpec
from ..spec.instance import HazardInstance

MUTATIONS = ("tunneling", "residual_velocity", "state_leak", "wrong_dt", "no_rollback", "park_interacts",
             "oracle_depends_on_hazard", "bad_standoff", "hook_dependent_release", "post_impact_vn")
ACTION_MODES = ("follow", "hold", "velocity")
TABLE_NAME = "table"

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
_ROBOT_SIDE = (PartnerClass.ROBOT_LINK, PartnerClass.HELD_OBJECT)


def _quat(name: str, value: Any) -> Tuple[float, float, float, float]:
    arr = np.array(value, dtype=np.float64)
    if arr.shape != (4,) or not np.all(np.isfinite(arr)) or not np.linalg.norm(arr) > 0:
        raise ValueError("%s must be a finite non-zero wxyz quaternion, got %r" % (name, value))
    return tuple(float(v) for v in arr)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _xy(name: str, value: Any) -> Tuple[float, float]:
    arr = np.array(value, dtype=np.float64)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError("%s must be a finite (x, y) pair, got %r" % (name, value))
    return float(arr[0]), float(arr[1])


# ------------------------------------------------------------------------------------------ configuration

@dataclasses.dataclass(frozen=True)
class FakeSphere:
    """A collision sphere. For robot and held bodies ``center_m`` is the offset from the EEF; for task
    objects it is the world position."""
    name: str
    center_m: Tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("FakeSphere.name must be a non-empty string, got %r" % (self.name,))
        if self.name == TABLE_NAME:
            raise ValueError("FakeSphere.name %r is reserved for the table" % TABLE_NAME)
        object.__setattr__(self, "center_m", tuple(vec3("%s.center_m" % self.name, self.center_m)))
        object.__setattr__(self, "radius_m", positive_float("%s.radius_m" % self.name, self.radius_m))


@dataclasses.dataclass(frozen=True)
class FakeTaskObject:
    """A static task object; ``on_hit_rotation_wxyz`` (or ``None``) is applied the first time a dynamic
    hazard hits it."""
    body: FakeSphere
    on_hit_rotation_wxyz: Optional[Tuple[float, float, float, float]]

    def __post_init__(self) -> None:
        if not isinstance(self.body, FakeSphere):
            raise ValueError("FakeTaskObject.body must be a FakeSphere")
        if self.on_hit_rotation_wxyz is not None:
            object.__setattr__(self, "on_hit_rotation_wxyz",
                               _quat("%s.on_hit_rotation_wxyz" % self.body.name, self.on_hit_rotation_wxyz))


@dataclasses.dataclass(frozen=True)
class FakeTask:
    """Nominal EEF path (piecewise linear, seconds from the reset), the success goal and the layouts
    (layout id -> task objects)."""
    path_times_s: Tuple[float, ...]
    path_pos_m: Tuple[Tuple[float, float, float], ...]
    goal_pos_m: Tuple[float, float, float]
    success_radius_m: float
    layouts: Mapping[str, Tuple[FakeTaskObject, ...]]

    def __post_init__(self) -> None:
        path = SampledPath(self.path_times_s, self.path_pos_m, None)  # validates
        object.__setattr__(self, "path_times_s", tuple(float(t) for t in path.times_s))
        object.__setattr__(self, "path_pos_m", tuple(tuple(float(v) for v in p) for p in path.points))
        object.__setattr__(self, "goal_pos_m", tuple(vec3("goal_pos_m", self.goal_pos_m)))
        object.__setattr__(self, "success_radius_m", non_negative_float("success_radius_m", self.success_radius_m))
        layouts = {}
        for layout_id, objects in dict(self.layouts).items():
            objects = tuple(objects)
            names = [o.body.name for o in objects]
            if len(set(names)) != len(names):
                raise ValueError("layout %r has duplicate object names %r" % (layout_id, names))
            layouts[layout_id] = objects
        if not layouts:
            raise ValueError("FakeTask needs at least one layout")
        object.__setattr__(self, "layouts", layouts)


@dataclasses.dataclass(frozen=True)
class FakeWorldConfig:
    """Everything that defines a FakeHost world; there are no defaults."""
    num_envs: int
    physics_dt_s: float
    substeps_per_action: int
    gravity_mps2: float
    restitution: float
    table_height_m: float
    table_min_xy_m: Tuple[float, float]
    table_max_xy_m: Tuple[float, float]
    park_pos_m: Tuple[float, float, float]
    robot_bodies: Tuple[FakeSphere, ...]
    held_object: Optional[str]
    held_bodies: Tuple[FakeSphere, ...]
    report_held_object_bodies: bool
    robot_include: Tuple[str, ...]
    robot_exclude: Tuple[str, ...]
    eef_contact_radius_m: float
    image_px: int
    image_min_xy_m: Tuple[float, float]
    image_max_xy_m: Tuple[float, float]
    max_contacts_per_step: int
    tasks: Mapping[str, FakeTask]

    def __post_init__(self) -> None:
        set_ = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731
        set_("num_envs", positive_int("num_envs", self.num_envs))
        set_("physics_dt_s", positive_float("physics_dt_s", self.physics_dt_s))
        set_("substeps_per_action", positive_int("substeps_per_action", self.substeps_per_action))
        set_("gravity_mps2", positive_float("gravity_mps2", self.gravity_mps2))
        set_("restitution", non_negative_float("restitution", self.restitution))
        set_("table_height_m", finite_float("table_height_m", self.table_height_m))
        for name in ("table_min_xy_m", "table_max_xy_m", "image_min_xy_m", "image_max_xy_m"):
            set_(name, _xy(name, getattr(self, name)))
        set_("park_pos_m", tuple(vec3("park_pos_m", self.park_pos_m)))
        set_("robot_bodies", tuple(self.robot_bodies))
        set_("held_bodies", tuple(self.held_bodies))
        if not self.robot_bodies:
            raise ValueError("robot_bodies must not be empty")
        if (self.held_object is None) != (not self.held_bodies):
            raise ValueError("held_object and held_bodies must be both set or both empty")
        names = [b.name for b in self.robot_bodies + self.held_bodies]
        if len(set(names)) != len(names):
            raise ValueError("robot and held body names must be unique, got %r" % (names,))
        set_("robot_include", tuple(self.robot_include))
        set_("robot_exclude", tuple(self.robot_exclude))
        set_("eef_contact_radius_m", non_negative_float("eef_contact_radius_m", self.eef_contact_radius_m))
        set_("image_px", positive_int("image_px", self.image_px))
        set_("max_contacts_per_step", positive_int("max_contacts_per_step", self.max_contacts_per_step))
        if not self.tasks:
            raise ValueError("tasks must not be empty")
        set_("tasks", dict(self.tasks))


@dataclasses.dataclass(frozen=True)
class FakeAction:
    """``follow`` the nominal path, ``hold`` still, or move at ``velocity_mps`` for each physics step."""
    mode: str
    velocity_mps: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        if self.mode not in ACTION_MODES:
            raise ValueError("FakeAction.mode %r is not one of %s" % (self.mode, list(ACTION_MODES)))
        if (self.mode == "velocity") != (self.velocity_mps is not None):
            raise ValueError("velocity_mps is required for mode 'velocity' and only for it")
        if self.velocity_mps is not None:
            object.__setattr__(self, "velocity_mps", tuple(vec3("velocity_mps", self.velocity_mps)))

    @classmethod
    def follow(cls) -> "FakeAction":
        return cls("follow")

    @classmethod
    def hold(cls) -> "FakeAction":
        return cls("hold")

    @classmethod
    def velocity(cls, velocity_mps: Sequence[float]) -> "FakeAction":
        return cls("velocity", tuple(velocity_mps))


# ------------------------------------------------------------------------------------------- world state

class _Hazard:
    def __init__(self, handle: HazardHandle, radius: float, park: np.ndarray) -> None:
        self.handle = handle
        self.radius = radius
        self.pos = park.copy()
        self.quat = np.array(_IDENTITY_QUAT)
        self.vel = np.zeros(3)
        self.ang = np.zeros(3)
        self.dynamic = False
        self.visible = False
        self.resting = False
        self.carrier = None  # type: Optional[Tuple[str, np.ndarray, float]]
        self.set_history = []  # type: List[np.ndarray]
        self.pending = []  # type: List[ContactObs]
        self.delayed = []  # type: List[ContactObs]
        self.overlap_reported = False


class _Object:
    def __init__(self, spec: FakeTaskObject) -> None:
        self.name = spec.body.name
        self.center = np.array(spec.body.center_m)
        self.radius = spec.body.radius_m
        self.quat = np.array(_IDENTITY_QUAT)
        self.rotation = None if spec.on_hit_rotation_wxyz is None else np.array(spec.on_hit_rotation_wxyz)
        self.hit = False


class _World:
    def __init__(self) -> None:
        self.steps = 0
        self.reset_steps = 0
        self.task_id = None  # type: Optional[str]
        self.layout_id = None  # type: Optional[str]
        self.seed = None  # type: Optional[int]
        self.eef = None  # type: Optional[np.ndarray]
        self.eef_vel = np.zeros(3)
        self.objects = {}  # type: Dict[str, _Object]
        self.hazards = {}  # type: Dict[str, _Hazard]


def _first_entry(poly: Any, horizon: float) -> Optional[float]:
    """Earliest ``s`` in ``[0, horizon]`` at which the gap polynomial ``poly`` (highest power first; negative
    means overlapping) is entered while decreasing. ``0`` when it already overlaps or touches and is moving
    further in. Grazing (double) roots are not contacts."""
    coeffs = np.trim_zeros(np.asarray(poly, dtype=np.float64), "f")
    if coeffs.size < 2:
        return None
    if coeffs[-1] <= 0:
        for k in range(coeffs.size - 2, -1, -1):  # derivatives at 0, lowest order first
            if coeffs[k] != 0:
                if coeffs[k] < 0:
                    return 0.0
                break
    deriv = np.polyder(coeffs)
    roots = np.roots(coeffs)
    real = np.sort(roots[roots.imag == 0].real)
    for i, root in enumerate(real):
        if root > horizon:
            break
        if np.polyval(deriv, root) >= 0:
            continue
        if root >= 0:
            return float(root)
        following = real[i + 1] if i + 1 < real.size else np.inf
        if following >= 0 and np.polyval(deriv, 0.0) < 0:
            return 0.0
    return None


def _sphere_entry(pos: np.ndarray, vel: np.ndarray, half_acc: np.ndarray, center: np.ndarray,
                  center_vel: np.ndarray, reach: float, horizon: float) -> Optional[float]:
    # gap(s) = |d(s)|^2 - reach^2 with d(s) = half_acc s^2 + (vel - center_vel) s + (pos - center)
    b = vel - center_vel
    c = pos - center
    poly = [half_acc @ half_acc, 2.0 * (half_acc @ b), b @ b + 2.0 * (half_acc @ c), 2.0 * (b @ c),
            c @ c - reach * reach]
    return _first_entry(poly, horizon)


# ---------------------------------------------------------------------------------------------- the host

class FakeHost:
    """See the module docstring. ``disable`` names optional methods to remove (degradation tests);
    ``ccd=False`` turns continuous collision detection off honestly (the BD then says ``hazard_ccd: false``);
    ``report_body_velocities=False`` leaves ``RobotState.collision_body_lin_vel_mps`` empty."""

    def __init__(self, config: FakeWorldConfig, mutations: Sequence[str] = (), disable: Sequence[str] = (),
                 ccd: bool = True, report_body_velocities: bool = True) -> None:
        if not isinstance(config, FakeWorldConfig):
            raise TypeError("config must be a FakeWorldConfig, got %r" % (type(config).__name__,))
        unknown = sorted(set(mutations) - set(MUTATIONS))
        if unknown:
            raise ValueError("unknown mutation(s) %s (known: %s)" % (unknown, list(MUTATIONS)))
        not_optional = sorted(set(disable) - set(OPTIONAL_METHODS))
        if not_optional:
            raise ValueError("only optional methods can be disabled, not %s" % not_optional)
        self.config = config
        self.mutations = frozenset(mutations)
        self.ccd = bool(ccd)
        self.report_body_velocities = bool(report_body_velocities)
        self._enabled = {name: name not in disable for name in OPTIONAL_METHODS}
        for name in disable:
            setattr(self, name, None)
        self._paths = {task_id: SampledPath(task.path_times_s, task.path_pos_m, None)
                       for task_id, task in config.tasks.items()}
        self._worlds = [_World() for _ in range(config.num_envs)]
        self._runtime = None  # type: Any
        self._allocated = 0

    # ------------------------------------------------------------------------------------ HazardHost

    def describe(self) -> BenchmarkDescriptor:
        cfg = self.config
        declared_dt = cfg.physics_dt_s * (2.0 if "wrong_dt" in self.mutations else 1.0)
        data = {
            "name": "fakehost", "bd_version": "1.0",
            "benchmark": {"repo": "hazardlib.conformance.fakehost", "commits": {"fakehost": "builtin"},
                          "license": "n/a"},
            "python": {"required": "3.8", "env_name": "any"},
            "engine": {"name": "fakehost-numpy", "version": np.__version__},
            "timebase": {"physics_dt_s": declared_dt, "control_dt_s": declared_dt * cfg.substeps_per_action,
                         "substeps_per_action": cfg.substeps_per_action},
            "gravity_mps2": cfg.gravity_mps2,
            "hazard_ccd": self.ccd or "tunneling" in self.mutations,
            "parallel": {"num_envs": cfg.num_envs, "batched": False},
            "loop_owner": "user_script",
            "hook_point": "FakeHost.step runs pre_step / post_step around every physics step",
            "spawn_mode": "runtime", "kinematic_pose_set": True, "release_with_velocity": True,
            "contact_tier": "A", "distance": "geometric" if self._enabled["min_distance"] else "surrogate",
            "robot_contact_bodies": {"include": list(cfg.robot_include), "exclude": list(cfg.robot_exclude)},
            "eef_contact_radius_m": cfg.eef_contact_radius_m,
            "state_access": {"eef": True, "gripper": False, "base": False, "objects": True, "fixtures": True,
                             "support_surfaces": self._enabled["support_surfaces"]},
            "task_entity_source": "oracle_config",
            "success_oracle": {"kind": "eef_goal_distance", "readonly": True},
            "snapshot_restore": self._enabled["snapshot"] and self._enabled["restore"],
            "determinism": {"same_seed_repro": "exact"},
            "invalid_sample_mapping": {"setup_invalid": "none", "sim_unstable": "none"},
            "observation_isolation": {"hazard_visible_in_rgb": True, "hazard_in_state_vectors": False},
            "policy_interface": {"kind": "fakehost_action", "notes": "FakeAction: follow / hold / velocity"},
        }
        return BenchmarkDescriptor.from_dict(data, strict=True)

    def alloc_hazard(self, env: int, agent: AgentSpec) -> HazardHandle:
        world = self._world(env)
        radius = self.sphere_radius(agent)
        self._allocated += 1
        handle = HazardHandle(env=env, key="hazard_%d_%d" % (env, self._allocated))
        world.hazards[handle.key] = _Hazard(handle, radius, np.array(self.config.park_pos_m))
        return handle

    def free_hazard(self, h: HazardHandle) -> None:
        self._hazard(h)
        del self._world(h.env).hazards[h.key]

    def set_hazard_pose(self, h: HazardHandle, pos: Sequence[float], quat: Sequence[float]) -> None:
        hz = self._hazard(h)
        if "no_rollback" in self.mutations and hz.overlap_reported:
            return
        pos = vec3("pos", pos)
        quat = np.array(_quat("quat", quat))
        residual = np.zeros(3)
        if "residual_velocity" in self.mutations:
            residual = (pos - hz.pos) / self.config.physics_dt_s
        hz.pos, hz.quat, hz.vel, hz.ang = pos, quat, residual, np.zeros(3)
        hz.dynamic = False
        hz.resting, hz.carrier = False, None
        hz.set_history = (hz.set_history + [pos.copy()])[-2:]

    def set_hazard_visible(self, h: HazardHandle, visible: bool) -> None:
        hz = self._hazard(h)
        hz.visible = bool(visible)
        if not hz.visible:
            hz.pos = np.array(self.config.park_pos_m)
            hz.vel, hz.ang = np.zeros(3), np.zeros(3)
            hz.dynamic = False
            hz.resting, hz.carrier = False, None
            hz.set_history = []

    def release_to_physics(self, h: HazardHandle, lin_vel: Sequence[float], ang_vel: Sequence[float]) -> None:
        hz = self._hazard(h)
        lin = vec3("lin_vel", lin_vel)
        if "hook_dependent_release" in self.mutations:
            history = hz.set_history
            lin = (history[1] - history[0]) / self.config.physics_dt_s if len(history) == 2 else np.zeros(3)
        elif "residual_velocity" in self.mutations:
            lin = lin + hz.vel
        hz.vel = lin
        hz.ang = vec3("ang_vel", ang_vel)
        hz.dynamic = True
        hz.resting, hz.carrier = False, None

    def robot_state(self, env: int) -> RobotState:
        cfg = self.config
        world = self._ready_world(env)
        bodies = cfg.robot_bodies + cfg.held_bodies
        velocities = None
        if self.report_body_velocities:
            velocities = [world.eef_vel.copy() for _ in bodies]
        held_names = None
        if cfg.report_held_object_bodies and cfg.held_object is not None:
            held_names = [b.name for b in cfg.held_bodies]
        return RobotState(eef_pose=Pose(world.eef.copy(), _IDENTITY_QUAT), eef_lin_vel_mps=world.eef_vel.copy(),
                          collision_body_pos=[world.eef + np.array(b.center_m) for b in bodies],
                          held_object=cfg.held_object, collision_body_names=[b.name for b in bodies],
                          collision_body_lin_vel_mps=velocities, held_object_bodies=held_names)

    def hazard_state(self, h: HazardHandle) -> HazardState:
        hz = self._hazard(h)
        return HazardState(pos=hz.pos.copy(), quat=hz.quat.copy(), lin_vel_mps=hz.vel.copy(),
                           ang_vel_radps=hz.ang.copy())

    def object_poses(self, env: int) -> Dict[str, Pose]:
        world = self._ready_world(env)
        return {name: Pose(obj.center.copy(), obj.quat.copy()) for name, obj in world.objects.items()}

    def hazard_contacts(self, h: HazardHandle) -> List[ContactObs]:
        hz = self._hazard(h)
        if "post_impact_vn" in self.mutations:
            out, hz.delayed = hz.delayed, hz.pending
        else:
            out = hz.pending
        hz.pending = []
        if not hz.dynamic and any(c.partner_class in _ROBOT_SIDE for c in out):
            hz.overlap_reported = True
        return out

    def time_state(self, env: int) -> TimeState:
        cfg = self.config
        return TimeState(sim_time_s=self._world(env).steps * cfg.physics_dt_s, physics_dt_s=cfg.physics_dt_s,
                         control_dt_s=cfg.physics_dt_s * cfg.substeps_per_action)

    def task_success(self, env: int) -> bool:
        world = self._ready_world(env)
        task = self.config.tasks[world.task_id]
        reached = float(np.linalg.norm(world.eef - np.array(task.goal_pos_m))) <= task.success_radius_m
        if "oracle_depends_on_hazard" in self.mutations and any(hz.visible for hz in world.hazards.values()):
            return False
        return reached

    def hold_action(self, env: int) -> FakeAction:
        self._world(env)
        return FakeAction.hold()

    def install_hooks(self, runtime: Any) -> None:
        self._runtime = runtime

    # ------------------------------------------------------------------------------ optional methods

    def min_distance(self, h: HazardHandle) -> float:
        hz = self._hazard(h)
        world = self._ready_world(h.env)
        gaps = [float(np.linalg.norm(hz.pos - center)) - hz.radius - radius
                for name, center, radius in self._spheres(world.eef)
                if self._classify(name) in _ROBOT_SIDE]
        return max(0.0, min(gaps))

    def snapshot(self, env: int) -> Any:
        return copy.deepcopy(self._world(env))

    def restore(self, env: int, snap: Any) -> None:
        if not isinstance(snap, _World):
            raise ValueError("restore needs a snapshot taken by this host")
        self._world(env)
        self._worlds[env] = copy.deepcopy(snap)

    def support_surfaces(self, env: int) -> List[Surface]:
        cfg = self.config
        self._world(env)
        z = cfg.table_height_m
        return [Surface(name=TABLE_NAME, min_m=[cfg.table_min_xy_m[0], cfg.table_min_xy_m[1], z],
                        max_m=[cfg.table_max_xy_m[0], cfg.table_max_xy_m[1], z], source="fakehost_config")]

    def render_check(self, h: HazardHandle) -> bool:
        hz = self._hazard(h)
        if not (hz.visible or "park_interacts" in self.mutations):
            return False
        lo, hi = self.config.image_min_xy_m, self.config.image_max_xy_m
        return all(lo[i] - hz.radius <= hz.pos[i] <= hi[i] + hz.radius for i in range(2))

    # ---------------------------------------------------------------------------------- EpisodeDriver

    def reset_episode(self, env: int, task_id: str, layout_id: str, seed: int) -> None:
        """Reset the world (see :meth:`reset_world`) and fire the reset hook ``runtime.begin_episode``."""
        self.reset_world(env, task_id, layout_id, seed)
        if self._runtime is not None:
            self._runtime.begin_episode(env)

    def step(self, env: int, action: FakeAction) -> None:
        """One control step: ``substeps_per_action`` physics steps, each between the two step hooks."""
        for _ in range(self.config.substeps_per_action):
            if self._runtime is not None:
                self._runtime.pre_step(env)
            self.advance_physics(env, action)
            if self._runtime is not None:
                self._runtime.post_step(env)

    def policy_observation(self, env: int) -> Dict[str, Any]:
        """``rgb``: a top-down ``image_px`` square rendering of the visible bodies; ``state``: EEF position and
        velocity, then each task object's position and orientation (sorted by name)."""
        world = self._ready_world(env)
        parts = [world.eef, world.eef_vel]
        for name in sorted(world.objects):
            parts += [world.objects[name].center, world.objects[name].quat]
        if "state_leak" in self.mutations:
            hazards = list(world.hazards.values())
            parts.append(hazards[0].pos if hazards else np.zeros(3))
        return {"rgb": self._render(world), "state": np.concatenate(parts).astype(np.float64)}

    # ------------------------------------------------------------------------------ test-side helpers

    def reset_world(self, env: int, task_id: str, layout_id: str, seed: int) -> None:
        """Put the EEF at the start of the task's nominal path, lay out the task objects; no hook. The world is
        deterministic, so ``seed`` is only recorded. The clock keeps running across episodes."""
        world = self._world(env)
        if task_id not in self.config.tasks:
            raise ValueError("unknown task_id %r (known: %s)" % (task_id, sorted(self.config.tasks)))
        task = self.config.tasks[task_id]
        if layout_id not in task.layouts:
            raise ValueError("task %r has no layout %r (known: %s)" % (task_id, layout_id, sorted(task.layouts)))
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an int, got %r" % (seed,))
        world.task_id, world.layout_id, world.seed = task_id, layout_id, int(seed)
        world.reset_steps = world.steps
        world.eef = self._paths[task_id].pos(0.0)
        world.eef_vel = np.zeros(3)
        world.objects = {o.body.name: _Object(o) for o in task.layouts[layout_id]}

    def advance_physics(self, env: int, action: FakeAction) -> None:
        """One physics step without hooks."""
        if not isinstance(action, FakeAction):
            raise ValueError("action must be a FakeAction, got %r" % (type(action).__name__,))
        cfg = self.config
        dt = cfg.physics_dt_s
        world = self._ready_world(env)
        eef0 = world.eef.copy()
        if action.mode == "follow":
            eef1 = self._paths[world.task_id].pos((world.steps + 1 - world.reset_steps) * dt)
        elif action.mode == "hold":
            eef1 = eef0.copy()
        else:
            eef1 = eef0 + np.array(action.velocity_mps) * dt
        robot_vel = (eef1 - eef0) / dt
        t_start = world.steps * dt
        t_end = (world.steps + 1) * dt
        world.eef, world.eef_vel = eef1, robot_vel
        for hz in list(world.hazards.values()):
            hz.overlap_reported = False
            if not (hz.visible or "park_interacts" in self.mutations):
                continue
            if hz.dynamic:
                hz.pending.extend(self._fly(world, hz, eef0, robot_vel, t_start, t_end))
            else:
                hz.pending.extend(self._overlaps(world, hz, t_end))
        world.steps += 1

    def nominal_eef_path(self, task_id: str) -> SampledPath:
        if task_id not in self._paths:
            raise ValueError("unknown task_id %r" % (task_id,))
        return self._paths[task_id]

    @staticmethod
    def sphere_radius(agent: AgentSpec) -> float:
        if not isinstance(agent, AgentSpec):
            raise TypeError("agent must be an AgentSpec, got %r" % (type(agent).__name__,))
        extents = list(agent.half_extents_m)
        if agent.shape != "sphere" or len(set(extents)) != 1:
            raise CapabilityError("FakeHost builds spheres only (shape 'sphere', equal half extents); got shape "
                                  "%r with half_extents_m %r" % (agent.shape, extents))
        return float(extents[0])

    def nominal_hit_instance(self, task_id: str, instance_id: str, spec_id: str, family: str, agent: AgentSpec,
                             t_hit_s: float, tau_s: float, params: Mapping[str, Any],
                             keyframe_dt_s: float) -> HazardInstance:
        """A nominal-collision instance on the task's nominal EEF path (D1). The centroid target is the EEF
        reference point plus ``standoff = r_hazard + eef_contact_radius_m`` along the approach normal
        (``+z`` for a fall, ``-direction`` otherwise); ``bad_standoff`` drops the standoff. Keyframes sample
        the solved path every ``keyframe_dt_s``; orientation stays the identity."""
        nominal = self.nominal_eef_path(task_id)
        radius = self.sphere_radius(agent)
        if family == FAMILY_FALL:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = -unit_vec3("params.direction", params["direction"])
        standoff = 0.0 if "bad_standoff" in self.mutations else radius + self.config.eef_contact_radius_m
        hit_point = nominal.pos(t_hit_s) + standoff * normal
        plan = build_hit_path(family, nominal, t_hit_s, hit_point, tau_s, params)
        times = hook_times(plan.path.t_start_s, plan.path.t_end_s, keyframe_dt_s)
        data = {
            "instance_id": instance_id, "spec_id": spec_id, "family": family, "agent": agent.to_dict(),
            "keyframes": {"time_s": times.tolist(), "pos": plan.path.pos(times).tolist(),
                          "quat": [list(_IDENTITY_QUAT)] * times.size},
            "nominal_hit_point": plan.nominal_hit_point.tolist(), "nominal_hit_time_s": plan.nominal_hit_time_s,
            "onset_time_s": plan.onset_time_s, "release_time_s": plan.release_time_s,
            "release_velocity": {"lin_mps": plan.release_velocity.tolist(), "ang_radps": [0.0, 0.0, 0.0]},
            "certificate": None,
            "provenance": {"builder": "FakeHost.nominal_hit_instance", "task_id": task_id, "standoff_m": standoff},
        }
        return HazardInstance.from_dict(data)

    # ---------------------------------------------------------------------------------------- internals

    def _world(self, env: int) -> _World:
        if isinstance(env, bool) or not isinstance(env, (int, np.integer)) or not 0 <= env < len(self._worlds):
            raise ValueError("env must be an index in [0, %d), got %r" % (len(self._worlds), env))
        return self._worlds[env]

    def _ready_world(self, env: int) -> _World:
        world = self._world(env)
        if world.eef is None:
            raise RuntimeError("env %d has not been reset yet" % env)
        return world

    def _hazard(self, h: HazardHandle) -> _Hazard:
        if not isinstance(h, HazardHandle):
            raise TypeError("expected a HazardHandle, got %r" % (type(h).__name__,))
        hazards = self._world(h.env).hazards
        if h.key not in hazards:
            raise ValueError("hazard %r is not allocated in env %d" % (h.key, h.env))
        return hazards[h.key]

    def _classify(self, name: str) -> PartnerClass:
        cfg = self.config
        return classify_partner(name, cfg.robot_include, cfg.robot_exclude, [b.name for b in cfg.held_bodies], [])

    def _spheres(self, eef: np.ndarray) -> List[Tuple[str, np.ndarray, float]]:
        bodies = self.config.robot_bodies + self.config.held_bodies
        return [(b.name, eef + np.array(b.center_m), b.radius_m) for b in bodies]

    def _obs(self, hz: _Hazard, name: str, position: Any, normal: Any, t: float) -> ContactObs:
        partner = PartnerClass.SCENE if name == TABLE_NAME else self._classify(name)
        return ContactObs(partner_name=name, partner_class=partner, position=position, normal=normal, time_s=t)

    def _overlaps(self, world: _World, hz: _Hazard, t: float) -> List[ContactObs]:
        out = []
        partners = self._spheres(world.eef) + [(o.name, o.center, o.radius) for o in world.objects.values()]
        for name, center, radius in partners:
            offset = hz.pos - center
            distance = float(np.linalg.norm(offset))
            if distance < hz.radius + radius:
                normal = offset / distance if distance > 0 else np.array([0.0, 0.0, 1.0])
                out.append(self._obs(hz, name, center + normal * radius, normal, t))
        if hz.pos[2] - hz.radius < self.config.table_height_m:
            out.append(self._obs(hz, TABLE_NAME, [hz.pos[0], hz.pos[1], self.config.table_height_m],
                                 [0.0, 0.0, 1.0], t))
        return out

    def _fly(self, world: _World, hz: _Hazard, eef0: np.ndarray, robot_vel: np.ndarray, t_start: float,
             t_end: float) -> List[ContactObs]:
        cfg = self.config
        dt = t_end - t_start
        ccd = self.ccd and "tunneling" not in self.mutations
        out = []  # type: List[ContactObs]
        if hz.carrier is not None:
            return [self._carry(world, hz, robot_vel, t_end)]
        elapsed = 0.0
        while True:
            remaining = dt - elapsed
            acc = np.zeros(3) if hz.resting else np.array([0.0, 0.0, -cfg.gravity_mps2])
            hit = self._first_hit(world, hz, acc, eef0 + robot_vel * elapsed, robot_vel, remaining) \
                if ccd and remaining > 0 else None
            if hit is None:
                self._advance(hz, acc, remaining)
                if not ccd:
                    out.extend(self._end_of_step_hits(world, hz, eef0 + robot_vel * dt, robot_vel, t_end))
                break
            s, name, center, partner_vel, radius, obj = hit
            self._advance(hz, acc, s)
            elapsed += s
            out.append(self._respond(hz, name, center, radius, partner_vel, obj, t_start + elapsed))
            if hz.carrier is not None:
                self._carry(world, hz, robot_vel, t_end)
                break
            if len(out) > cfg.max_contacts_per_step:
                raise RuntimeError("FakeHost: more than %d contacts of %s within one physics step"
                                   % (cfg.max_contacts_per_step, hz.handle.key))
        if hz.resting:
            out.append(self._obs(hz, TABLE_NAME, [hz.pos[0], hz.pos[1], cfg.table_height_m], [0.0, 0.0, 1.0],
                                 t_end))
        return out

    def _first_hit(self, world: _World, hz: _Hazard, acc: np.ndarray, eef: np.ndarray, robot_vel: np.ndarray,
                   horizon: float) -> Optional[Tuple[float, str, np.ndarray, np.ndarray, float, Optional[_Object]]]:
        half_acc = 0.5 * acc
        candidates = [(name, center, robot_vel, radius, None) for name, center, radius in self._spheres(eef)]
        candidates += [(o.name, o.center, np.zeros(3), o.radius, o) for o in world.objects.values()]
        best = None  # type: Any
        for name, center, partner_vel, radius, obj in candidates:
            s = _sphere_entry(hz.pos, hz.vel, half_acc, center, partner_vel, hz.radius + radius, horizon)
            if s is not None and (best is None or s < best[0]):
                best = (s, name, center + partner_vel * s, partner_vel, radius, obj)
        if not hz.resting:
            gap = hz.pos[2] - hz.radius - self.config.table_height_m
            s = _first_entry([half_acc[2], hz.vel[2], gap], horizon)
            if s is not None and (best is None or s < best[0]):
                best = (s, TABLE_NAME, None, np.zeros(3), None, None)
        return best

    def _end_of_step_hits(self, world: _World, hz: _Hazard, eef: np.ndarray, robot_vel: np.ndarray,
                          t: float) -> List[ContactObs]:
        out = []
        candidates = [(name, center, robot_vel, radius, None) for name, center, radius in self._spheres(eef)]
        candidates += [(o.name, o.center, np.zeros(3), o.radius, o) for o in world.objects.values()]
        for name, center, partner_vel, radius, obj in candidates:
            offset = hz.pos - center
            if float(np.linalg.norm(offset)) < hz.radius + radius and (hz.vel - partner_vel) @ offset < 0:
                out.append(self._respond(hz, name, center, radius, partner_vel, obj, t))
        if hz.pos[2] - hz.radius < self.config.table_height_m and hz.vel[2] < 0:
            out.append(self._respond(hz, TABLE_NAME, None, None, np.zeros(3), None, t))
        return out

    def _carry(self, world: _World, hz: _Hazard, robot_vel: np.ndarray, t: float) -> ContactObs:
        """Move a carried hazard rigidly with its carrier to the end of the step; one contact per step."""
        name, offset, radius = hz.carrier
        if name in world.objects:
            center, vel = world.objects[name].center, np.zeros(3)
        else:
            center, vel = dict((n, c) for n, c, _ in self._spheres(world.eef))[name], robot_vel
        hz.pos, hz.vel = center + offset, vel.copy()
        normal = offset / float(np.linalg.norm(offset))
        return self._obs(hz, name, center + normal * radius, normal, t)

    @staticmethod
    def _advance(hz: _Hazard, acc: np.ndarray, s: float) -> None:
        hz.pos = hz.pos + hz.vel * s + 0.5 * acc * s * s
        hz.vel = hz.vel + acc * s

    def _respond(self, hz: _Hazard, name: str, center: Optional[np.ndarray], radius: Optional[float],
                 partner_vel: np.ndarray, obj: Optional[_Object], t: float) -> ContactObs:
        cfg = self.config
        if name == TABLE_NAME:
            normal = np.array([0.0, 0.0, 1.0])
            hz.pos[2] = cfg.table_height_m + hz.radius
            position = [hz.pos[0], hz.pos[1], cfg.table_height_m]
        else:
            offset = hz.pos - center
            normal = offset / float(np.linalg.norm(offset))
            position = center + normal * radius
        closing = float((hz.vel - partner_vel) @ normal)
        if closing < 0:
            hz.vel = hz.vel - (1.0 + cfg.restitution) * closing * normal
        separating = float((hz.vel - partner_vel) @ normal)
        if name == TABLE_NAME:
            if hz.vel[2] <= cfg.gravity_mps2 * cfg.physics_dt_s:  # would land again within one physics step
                hz.vel[2] = 0.0
                hz.resting = True
        elif normal[2] > 0 and separating <= cfg.gravity_mps2 * normal[2] * cfg.physics_dt_s:
            hz.vel = partner_vel.copy()
            hz.resting, hz.carrier = False, (name, hz.pos - center, float(radius))
        elif hz.resting and hz.vel[2] > 0:
            hz.resting = False
        if obj is not None and obj.rotation is not None and not obj.hit:
            obj.quat = _quat_mul(obj.rotation, obj.quat)
            obj.hit = True
        return self._obs(hz, name, position, normal, t)

    def _render(self, world: _World) -> np.ndarray:
        cfg = self.config
        px = cfg.image_px
        (x0, y0), (x1, y1) = cfg.image_min_xy_m, cfg.image_max_xy_m
        xs = x0 + (np.arange(px) + 0.5) * (x1 - x0) / px
        ys = y0 + (np.arange(px) + 0.5) * (y1 - y0) / px
        grid_x, grid_y = np.meshgrid(xs, ys)
        image = np.zeros((px, px, 3), dtype=np.uint8)

        def disc(center: np.ndarray, radius: float, color: Tuple[int, int, int]) -> None:
            mask = (grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2 <= radius * radius
            image[mask] = color

        for obj in world.objects.values():
            disc(obj.center, obj.radius, (0, 160, 0))
        for name, center, radius in self._spheres(world.eef):
            disc(center, radius, (128, 128, 128))
        for hz in world.hazards.values():
            if hz.visible or "park_interacts" in self.mutations:
                disc(hz.pos, hz.radius, (220, 0, 0))
        return image
