"""HazardRuntime: the core side of the hook contract (01 §3.3; rules R1, R6-R11, R14; decisions D10, D11).

Adapters only forward hooks: ``begin_episode`` from the benchmark's reset, ``pre_step`` / ``post_step``
around every *finest* physics step and ``end_episode`` when the episode ends (``install_hooks``). The
runtime does the rest: it follows the instance keyframes in seconds, re-checks R1, rolls a kinematic hazard
back before releasing it (R8), turns contacts into events with T03's semantics and returns one
``HazardEventLog`` per episode. It never imports a simulator and owns no threshold: every number comes from
``RuntimeConfig`` (``configs/hazard/runtime.yaml`` and ``semantics.yaml``).

Episode binding (D11 a): ``arm(env, instance, ...)`` registers the next episode; the host's reset hook then
calls ``begin_episode(env)``, which fails loudly if nothing was armed. ``instance=None`` is a hazard-free
control episode: ``control_id`` becomes the log's ``instance_id`` and every hazard field is ``None``.

Clock. Instance times are seconds from the episode start, i.e. the host's ``sim_time_s`` when
``begin_episode`` ran. A hook covers one physics step ``[t0, t1 = t0 + physics_dt_s]``: ``pre_step`` sees
the state at ``t0``, ``post_step`` the state at ``t1``. Placement and the onset (``T``) happen in the
``pre_step`` whose ``t0`` is the hook start nearest to ``T`` (``t0 >= T - physics_dt_s / 2``), so they are off
by at most half a physics step whatever the float rounding of the clock. A kinematic hazard is posed at
``p(t1)`` in ``pre_step``: the engine checks overlaps there during the step and ``post_step`` reads them.
The nominal kinematic release at ``T`` happens in the ``pre_step`` of the step whose end ``t1`` is nearest
to ``T``: the hazard leaves its last kinematic pose ``p(t0)`` as a dynamic body and arrives at the release
pose at ``t1``. Its last kinematic pose is therefore never the nominal touching pose of a nominal-hit
instance (D1), which an engine would report as an overlap (contact margins, float rounding).

Families (D11 b, dispatched on ``instance.family`` only):

* ``intrusion.fall``: posed at ``p(onset)``, unparked and released with velocity 0 in one hook (D2). No
  kinematic phase, hence no overlap check and no rollback; every later contact is a dynamic contact.
* ``intrusion.slide`` / ``intrusion.reach``: unparked at the first keyframe and moved kinematically. The
  nominal release at ``release_time_s`` uses ``instance.release_velocity``. A robot-side overlap before
  that is recorded as ``kinematic_overlap`` and rolled back (R8): the hazard returns to the last pose that
  had no overlap, at time ``t_rb``, and is released with the backward difference of the keyframe path over
  ``release_velocity_window_s`` ending at ``t_rb`` (R7, D4; no floor, independent of the hook step). If
  there is no such pose, or the window does not fit before ``t_rb``, the instance is ``setup_invalid`` and
  the hazard is parked again.
* Follow-through (D19), only for an instance with ``exit_keyframes``: the playback path is ``keyframes``
  followed by ``exit_keyframes`` and the nominal release happens at the same step as above **only if the
  robot is engaged**: the latest sampled hazard-to-robot separation (``host.min_distance``, the true
  geometric distance, sampled by the previous ``post_step``) is ``<= max_step_ratio`` x the hazard's smallest
  half extent, the R1 per-hook displacement limit. The decision is taken once, at that step. If the robot is
  not engaged the hazard stays ``KINEMATIC`` along the exit path (overlap detection and the rollback release
  keep working; the rollback velocity window may straddle the junction) and, when the exit path ends, is
  parked (``set_hazard_visible(False)``) in phase ``EXITED``: ``pre_step`` does nothing for it and
  ``post_step`` only drains ``hazard_contacts``. Such an instance needs ``bd.distance == "geometric"``
  (``CapabilityError`` at ``arm``). An instance without ``exit_keyframes`` takes the old path unchanged.
* anything else raises ``CapabilityError`` at ``arm``.

Contacts. The runtime re-classifies every contact with ``classify_partner`` and the BD's
``robot_contact_bodies`` (held-object names from ``RobotState.held_object_bodies``, else
``[held_object]``); a partner the host reports as ``hazard`` stays ``hazard``. After release, every contact
becomes a ``dynamic`` event; the official-contact quantities use robot-side partners only and
``first_scene_contact_t_s`` records the first ``scene`` contact (D10 a). Every post-release hook, including
those without contacts, is handed to ``summarize_contacts``. ``v_rel_normal_mps`` uses the snapshot of the
previous hook (D10 c): the hazard's centre-of-mass velocity and the contacted collision body's velocity,
matched by name in ``collision_body_names``; if that velocity is unavailable it is ``None``, never 0. The
snapshot is refreshed right after every release.

R1 (D11 d) is a check: between two consecutive ``post_step`` reads of ``hazard_state``, a displacement
above ``max_step_ratio`` x the agent's smallest half extent adds one to ``r1_violation_count``. The
runtime's own teleports (placement, rollback) are not counted; it never inserts steps. When the BD says
``hazard_ccd: false`` a non-zero count also raises :class:`R1ViolationWarning` at ``end_episode``.

Topple (D10 h): every name in ``task_entities`` gets its reference orientation at the nominal onset (the
paired instance's onset for a control, ``control_onset_s``); ``max_tilt_deg`` is the largest up-axis tilt
seen afterwards and ``toppled`` compares it with ``tilt_threshold_deg``. ``task_entities=None`` leaves both
``None``.

Instability (D9 d, D10 b): the host calls ``report_instability``; the first report of an episode is
classified by T03's ``classify_invalid`` with the window ``[onset, nominal_hit_time + attribution_window_s]``.
A control episode has no hazard, so its instability is ``sim_unstable``.

Not computed here: ``exposure_low`` (T06, from the probe trajectories), ``engine_peak_force_n`` and
``force_source`` (auxiliary engine quantities, adapter-specific); they stay ``None``.
"""
import dataclasses
import math
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._numeric import finite_float, non_negative_float, positive_float, vec3
from ._records import is_unknown
from .config import ConfigError, load_config
from .host import (CapabilityError, ContactObs, HazardHandle, HazardState, PartnerClass, RobotState, Tier,
                   require_methods)
from .kinematics.hit import FAMILY_FALL, FAMILY_REACH, FAMILY_SLIDE
from .kinematics.path import SampledPath
from .semantics import (ContactEvent, ContactStep, EventKind, HazardEventLog, InvalidAttribution,
                        InvalidClass, analytic_impulse_ns, classify_invalid, classify_partner,
                        closing_normal_speed, compute_proximity, is_toppled, summarize_contacts,
                        tilt_deg_up_axis)
from .spec.benchmark_descriptor import BenchmarkDescriptor
from .spec.instance import HazardInstance
from .spec.manifest import Manifest

KINEMATIC_FAMILIES = (FAMILY_SLIDE, FAMILY_REACH)
SUPPORTED_FAMILIES = (FAMILY_FALL,) + KINEMATIC_FAMILIES

#: Keys read from ``runtime.yaml`` (top level) and ``semantics.yaml`` (section, key).
RUNTIME_KEYS = ("max_step_ratio", "release_velocity_window_s")
SEMANTICS_KEYS = (("contact", "restitution_e"), ("contact", "gap_tolerance_s"), ("clamp", "t_q_s"),
                  ("clamp", "opposing_normals_tol_rad"), ("toppling", "tilt_threshold_deg"),
                  ("stability", "hazard_induced_max_distance_m"), ("stability", "attribution_window_s"))

PARKED = "parked"
KINEMATIC = "kinematic"
RELEASED = "released"
SETUP_FAILED = "setup_failed"
EXITED = "exited"

_ROBOT_SIDE = (PartnerClass.ROBOT_LINK, PartnerClass.HELD_OBJECT)


class R1ViolationWarning(RuntimeWarning):
    """A hazard moved more than ``max_step_ratio`` x its smallest half extent in one hook on a host without
    continuous collision detection for hazards (R1, D11 d). Gate G0 should have rejected the instance."""


# --------------------------------------------------------------------------------------------- settings

@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    """Every number the runtime uses. Build it with :meth:`load` / :meth:`from_configs` from the YAML
    files, or directly with literal values in tests. There are no defaults."""
    max_step_ratio: float
    release_velocity_window_s: float
    restitution_e: float
    gap_tolerance_s: float
    t_q_s: float
    opposing_normals_tol_rad: float
    tilt_threshold_deg: float
    hazard_induced_max_distance_m: float
    attribution_window_s: float

    def __post_init__(self) -> None:
        for name in ("max_step_ratio", "release_velocity_window_s", "t_q_s", "opposing_normals_tol_rad"):
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))
        for name in ("restitution_e", "gap_tolerance_s", "tilt_threshold_deg", "hazard_induced_max_distance_m",
                     "attribution_window_s"):
            object.__setattr__(self, name, non_negative_float(name, getattr(self, name)))
        if self.opposing_normals_tol_rad > math.pi:
            raise ValueError("opposing_normals_tol_rad must be <= pi, got %r" % (self.opposing_normals_tol_rad,))

    @classmethod
    def from_configs(cls, runtime: Mapping[str, Any], semantics: Mapping[str, Any]) -> "RuntimeConfig":
        """From the parsed ``runtime.yaml`` and ``semantics.yaml``. Every value the runtime needs must be
        resolved; ``PLACEHOLDER`` / ``unknown`` / a missing key is a :class:`ConfigError` naming all of them."""
        values = {}  # type: Dict[str, Any]
        problems = []  # type: List[str]
        sources = [("runtime", runtime, (key,)) for key in RUNTIME_KEYS]
        sources += [("semantics", semantics, path) for path in SEMANTICS_KEYS]
        for file_name, data, path in sources:
            current = data  # type: Any
            for part in path:
                current = current.get(part) if isinstance(current, Mapping) else None
            dotted = "%s.%s" % (file_name, ".".join(path))
            if current is None:
                problems.append("%s is missing" % dotted)
            elif isinstance(current, str):
                problems.append("%s=%s" % (dotted, current))
            else:
                values[path[-1]] = current
        if problems:
            raise ConfigError("RuntimeConfig: %d unresolved value(s): %s" % (len(problems), "; ".join(problems)))
        try:
            return cls(**values)
        except ValueError as exc:
            raise ConfigError("RuntimeConfig: %s" % exc) from None

    @classmethod
    def load(cls, config_dir: Optional[Any] = None) -> "RuntimeConfig":
        return cls.from_configs(load_config("runtime", config_dir=config_dir),
                                load_config("semantics", config_dir=config_dir))


@dataclasses.dataclass(frozen=True)
class Provenance:
    """Copied verbatim into every log (R14, D11 c). Without a manifest (development) pass no provenance:
    the four log fields are then ``None``."""
    bd_hash: Optional[str]
    adapter_version: Optional[str]
    manifest_hash: Optional[str]
    benchmark_commits: Optional[Dict[str, str]]

    def __post_init__(self) -> None:
        HazardEventLog.new("provenance", **self.log_fields())  # same validation as the log fields

    def log_fields(self) -> Dict[str, Any]:
        commits = None if self.benchmark_commits is None else dict(self.benchmark_commits)
        return {"bd_hash": self.bd_hash, "adapter_version": self.adapter_version,
                "manifest_hash": self.manifest_hash, "benchmark_commits": commits}

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> "Provenance":
        return cls(bd_hash=manifest.bd_hash, adapter_version=manifest.adapter_version,
                   manifest_hash=manifest.hash(), benchmark_commits=dict(manifest.benchmark_commits))


# ------------------------------------------------------------------------------------------ episode state

@dataclasses.dataclass
class _Arm:
    instance: Optional[HazardInstance]
    task_entities: Optional[List[str]]
    control_id: Optional[str]
    control_onset_s: Optional[float]


class _Episode:
    """Mutable state of one running (or just finished) episode of one environment."""

    def __init__(self, env: int, arm: _Arm, t_start: float, handle: Optional[HazardHandle]) -> None:
        self.env = env
        self.instance = arm.instance
        self.control_id = arm.control_id
        self.task_entities = arm.task_entities
        self.onset_s = arm.control_onset_s if arm.instance is None else arm.instance.onset_time_s
        self.t_start = t_start
        self.handle = handle
        self.path = None  # type: Optional[SampledPath]
        self.follow_through = False  # True once the nominal release step passed with the robot not engaged
        self.exit_t = None  # type: Optional[float]
        if arm.instance is not None:
            frames = arm.instance.keyframes
            times, pos, quat = list(frames.time_s), list(frames.pos), list(frames.quat)
            exit_ = arm.instance.exit_keyframes
            if exit_ is not None:  # the junction sample (validated equal) is counted once
                times += list(exit_.time_s[1:])
                pos += list(exit_.pos[1:])
                quat += list(exit_.quat[1:])
            self.path = SampledPath(times, pos, quat)
        self.phase = PARKED
        self.onset_t = None  # type: Optional[float]
        self.released_t = None  # type: Optional[float]
        self.release_info = None  # type: Optional[Dict[str, Any]]
        self.last_free_t = None  # type: Optional[float]
        self.prev_pos = None  # type: Optional[np.ndarray]
        self.r1_violations = 0
        self.snapshot = None  # type: Optional[Tuple[HazardState, RobotState]]
        self.events = []  # type: List[ContactEvent]
        self.steps = []  # type: List[ContactStep]
        self.first_official = None  # type: Optional[ContactObs]
        self.first_scene_t = None  # type: Optional[float]
        self.v_normal = None  # type: Optional[float]
        self.impulse = None  # type: Optional[float]
        self.trace_t = []  # type: List[float]
        self.trace_hazard = []  # type: List[np.ndarray]
        self.trace_eef = []  # type: List[np.ndarray]
        self.trace_distance = []  # type: List[float]
        self.tilt_ref = None  # type: Optional[Dict[str, np.ndarray]]
        self.max_tilt = None  # type: Optional[float]
        self.setup_failed = False
        self.setup_reason = None  # type: Optional[str]
        self.instability = None  # type: Optional[Dict[str, Any]]
        self.log = None  # type: Optional[HazardEventLog]

    @property
    def log_id(self) -> str:
        return self.control_id if self.instance is None else self.instance.instance_id

    def pose(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.path.pos(t), self.path.quat(t)


def _env_index(env: Any) -> int:
    if isinstance(env, (bool, np.bool_)) or not isinstance(env, (int, np.integer)) or env < 0:
        raise ValueError("env must be an int >= 0, got %r" % (env,))
    return int(env)


def _entity_list(task_entities: Optional[Sequence[str]]) -> Optional[List[str]]:
    if task_entities is None:
        return None
    if isinstance(task_entities, str):
        raise ValueError("task_entities must be a list of names, not the string %r" % (task_entities,))
    names = list(task_entities)
    if not names:
        raise ValueError("task_entities must be None (not monitored) or a non-empty list of names")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("task_entities contains an invalid name %r" % (name,))
    if len(set(names)) != len(names):
        raise ValueError("task_entities contains duplicates: %r" % (names,))
    return names


# ----------------------------------------------------------------------------------------------- runtime

class HazardRuntime:
    """See the module docstring. One runtime serves every environment index of one host; the episodes of
    different indices are independent.

    One host, one live runtime (01 §3.3 item 16, D13 c): the constructor calls ``host.install_hooks(self)``, so a
    runtime constructed later on the same host takes the hooks over and the earlier one is void from then on
    (using it records wrong quantities). There is deliberately no guard against a second runtime: the
    conformance suite takes the hooks over probe by probe. Drivers, the gates and the T_stop protocol therefore
    hold one session (with its runtime) per host instead of the host itself."""

    def __init__(self, host: Any, cfg: RuntimeConfig, provenance: Optional[Provenance] = None) -> None:
        require_methods(host)
        if not isinstance(cfg, RuntimeConfig):
            raise TypeError("cfg must be a RuntimeConfig, got %r" % (type(cfg).__name__,))
        if provenance is not None and not isinstance(provenance, Provenance):
            raise TypeError("provenance must be a Provenance or None, got %r" % (type(provenance).__name__,))
        bd = host.describe()
        if not isinstance(bd, BenchmarkDescriptor):
            raise CapabilityError("%s.describe() must return a BenchmarkDescriptor, got %r"
                                  % (type(host).__name__, type(bd).__name__))
        tier = bd.tier()
        if tier is not Tier.A:
            raise CapabilityError("BD %r is Tier %s; HazardRuntime implements the Tier-A release semantics only"
                                  % (bd.name, tier.value))
        bodies = bd.robot_contact_bodies
        needed = {"robot_contact_bodies.include": None if is_unknown(bodies) else bodies.include,
                  "robot_contact_bodies.exclude": None if is_unknown(bodies) else bodies.exclude,
                  "hazard_ccd": bd.hazard_ccd, "distance": bd.distance}
        unresolved = sorted(k for k, v in needed.items() if v is None or is_unknown(v))
        if unresolved:
            raise CapabilityError("BD %r leaves %s unknown; the runtime needs them (run the probes first)"
                                  % (bd.name, ", ".join(unresolved)))
        self._geometric = bd.distance == "geometric"
        if self._geometric and not callable(getattr(host, "min_distance", None)):
            raise CapabilityError("BD %r declares distance=geometric but %s has no min_distance"
                                  % (bd.name, type(host).__name__))
        self.host = host
        self.cfg = cfg
        self.bd = bd
        self.provenance = provenance
        self._include = list(bodies.include)
        self._exclude = list(bodies.exclude)
        self._armed = {}  # type: Dict[int, _Arm]
        self._running = {}  # type: Dict[int, _Episode]
        self._finished = {}  # type: Dict[int, _Episode]
        host.install_hooks(self)

    # ---------------------------------------------------------------------------------- binding (D11 a)

    def arm(self, env: int, instance: Optional[HazardInstance], task_entities: Optional[Sequence[str]] = None,
            control_id: Optional[str] = None, control_onset_s: Optional[float] = None) -> None:
        """Register the next episode of ``env``. ``control_onset_s`` (control episodes only) is the paired
        instance's nominal onset, the reference time of the topple measurement (D10 h); it is required when a
        control episode monitors ``task_entities``."""
        env = _env_index(env)
        if env in self._armed:
            raise RuntimeError("env %d is already armed; begin_episode consumes one arm() per episode" % env)
        entities = _entity_list(task_entities)
        if instance is None:
            if not isinstance(control_id, str) or not control_id:
                raise ValueError("a control episode (instance=None) needs a non-empty control_id")
            if control_onset_s is not None:
                control_onset_s = non_negative_float("control_onset_s", control_onset_s)
            elif entities is not None:
                raise ValueError("a control episode that monitors task_entities needs control_onset_s "
                                 "(the paired instance's nominal onset, D10 h)")
        else:
            if not isinstance(instance, HazardInstance):
                raise TypeError("instance must be a HazardInstance or None, got %r" % (type(instance).__name__,))
            if control_id is not None or control_onset_s is not None:
                raise ValueError("control_id / control_onset_s are for control episodes (instance=None) only")
            self._check_instance(instance)
        self._armed[env] = _Arm(instance, entities, control_id, control_onset_s)

    def _check_instance(self, inst: HazardInstance) -> None:
        if inst.family not in SUPPORTED_FAMILIES:
            raise CapabilityError("instance %r: family %r is not supported by the runtime (supported: %s)"
                                  % (inst.instance_id, inst.family, ", ".join(SUPPORTED_FAMILIES)))
        where = "instance %r (%s)" % (inst.instance_id, inst.family)
        if inst.exit_keyframes is not None and not self._geometric:
            raise CapabilityError("%s has exit_keyframes: the follow-through decision needs a true geometric "
                                  "distance, but BD %r declares distance=%r" % (where, self.bd.name, self.bd.distance))
        if inst.release_time_s is None:
            raise CapabilityError("%s has no release; the Tier-A runtime releases every hazard" % where)
        if inst.onset_time_s < 0:
            raise ValueError("%s: onset_time_s=%r is before the episode start" % (where, inst.onset_time_s))
        times = inst.keyframes.time_s
        if len(times) < 2:
            raise ValueError("%s needs at least 2 keyframes, got %d" % (where, len(times)))
        onset, release = inst.onset_time_s, inst.release_time_s
        if inst.family == FAMILY_FALL:
            # D2: released at rest at the onset (both are copied from the same HitPlan value).
            if release != onset:
                raise ValueError("%s: release_time_s=%r must equal onset_time_s=%r" % (where, release, onset))
            velocity = inst.release_velocity
            if any(v != 0.0 for v in list(velocity.lin_mps) + list(velocity.ang_radps)):
                raise ValueError("%s: the release velocity must be zero (D2), got %r / %r"
                                 % (where, velocity.lin_mps, velocity.ang_radps))
            if not times[0] <= onset <= times[-1]:
                raise ValueError("%s: keyframes [%r, %r] do not cover the onset %r"
                                 % (where, times[0], times[-1], onset))
            return
        if times[0] > onset:
            raise ValueError("%s: the first keyframe %r is after the onset %r" % (where, times[0], onset))
        if not onset <= release <= times[-1]:
            raise ValueError("%s: release_time_s=%r must lie in [onset %r, last keyframe %r]"
                             % (where, release, onset, times[-1]))

    def begin_episode(self, env: int) -> None:
        """Reset hook: start the episode registered by :meth:`arm`; the hazard is allocated and parked."""
        env = _env_index(env)
        if env in self._running:
            raise RuntimeError("env %d: begin_episode while an episode is running; call end_episode first" % env)
        arm = self._armed.pop(env, None)
        if arm is None:
            raise RuntimeError("env %d: begin_episode without arm(); arm the next instance (or a control) "
                               "before the reset" % env)
        if arm.task_entities is not None:
            poses = self.host.object_poses(env)
            missing = [n for n in arm.task_entities if n not in poses]
            if missing:
                raise ValueError("env %d: task_entities %s are not in object_poses (known: %s)"
                                 % (env, missing, sorted(poses)))
        t_start = self.host.time_state(env).sim_time_s
        handle = None
        if arm.instance is not None:
            handle = self.host.alloc_hazard(env, arm.instance.agent)
            self.host.set_hazard_visible(handle, False)
        self._finished.pop(env, None)
        self._running[env] = _Episode(env, arm, t_start, handle)

    # -------------------------------------------------------------------------------------------- hooks

    def pre_step(self, env: int) -> None:
        ep = self._running.get(_env_index(env))
        if ep is None:
            return
        state = self.host.time_state(ep.env)
        t0 = state.sim_time_s - ep.t_start
        half = 0.5 * state.physics_dt_s
        t1 = t0 + state.physics_dt_s
        if ep.task_entities is not None and ep.tilt_ref is None and ep.onset_s is not None \
                and t0 >= ep.onset_s - half:
            poses = self.host.object_poses(ep.env)
            ep.tilt_ref = {name: np.array(self._entity_pose(poses, name).quat) for name in ep.task_entities}
            ep.max_tilt = 0.0
        inst = ep.instance
        if inst is None:
            return
        if ep.onset_t is None and t0 >= inst.onset_time_s - half:
            ep.onset_t = t0
        h = ep.handle
        if ep.phase == PARKED:
            if inst.family == FAMILY_FALL:
                if t0 >= inst.onset_time_s - half:
                    pos, quat = ep.pose(inst.onset_time_s)
                    self.host.set_hazard_pose(h, pos, quat)
                    self.host.set_hazard_visible(h, True)
                    velocity = inst.release_velocity
                    self._release(ep, t0, velocity.lin_mps, velocity.ang_radps, "onset", None)
            elif t0 >= ep.path.t_start_s - half:
                pos, quat = ep.pose(t1)
                self.host.set_hazard_pose(h, pos, quat)
                self.host.set_hazard_visible(h, True)
                ep.phase = KINEMATIC
                ep.prev_pos = None
        elif ep.phase == KINEMATIC:
            if inst.exit_keyframes is None:
                if t1 >= inst.release_time_s - half:
                    velocity = inst.release_velocity
                    self._release(ep, t0, velocity.lin_mps, velocity.ang_radps, "nominal", None)
                else:
                    pos, quat = ep.pose(t1)
                    self.host.set_hazard_pose(h, pos, quat)
            else:
                self._follow_through_step(ep, t0, t1, half)

    def post_step(self, env: int) -> None:
        ep = self._running.get(_env_index(env))
        if ep is None:
            return
        t1 = self.host.time_state(ep.env).sim_time_s - ep.t_start
        if ep.tilt_ref is not None:
            poses = self.host.object_poses(ep.env)
            for name, ref in ep.tilt_ref.items():
                tilt = tilt_deg_up_axis(ref, self._entity_pose(poses, name).quat)
                ep.max_tilt = max(ep.max_tilt, tilt)
        if ep.instance is None:
            return
        h = ep.handle
        if ep.phase in (PARKED, SETUP_FAILED, EXITED):
            self.host.hazard_contacts(h)  # drained: a parked hazard must not interact (checked by C2)
            return
        hazard = self.host.hazard_state(h)
        robot = self.host.robot_state(ep.env)
        contacts = self._classify(ep, self.host.hazard_contacts(h), robot)
        self._check_r1(ep, hazard)
        self._sample_proximity(ep, t1, hazard, robot)
        if ep.phase == KINEMATIC:
            overlaps = [c for c in contacts if c.partner_class in _ROBOT_SIDE]
            if not overlaps:
                ep.last_free_t = min(t1, ep.path.t_end_s)
                return
            for obs in overlaps:
                ep.events.append(ContactEvent.from_obs(obs, EventKind.KINEMATIC_OVERLAP, None))
            self._roll_back(ep, t1)
            return
        ep.steps.append(ContactStep(t1, contacts))
        for obs in sorted(contacts, key=lambda c: c.time_s):
            v_normal = None
            if obs.partner_class in _ROBOT_SIDE and ep.first_official is None:
                ep.first_official = obs
                v_normal = self._v_normal(ep, obs)
                ep.v_normal = v_normal
                if v_normal is not None:
                    ep.impulse = analytic_impulse_ns(ep.instance.agent.mass_kg, v_normal, self.cfg.restitution_e)
            elif obs.partner_class is PartnerClass.SCENE and ep.first_scene_t is None:
                ep.first_scene_t = obs.time_s
            ep.events.append(ContactEvent.from_obs(obs, EventKind.DYNAMIC, v_normal))
        ep.snapshot = (hazard, robot)

    def end_episode(self, env: int) -> HazardEventLog:
        """Episode-end hook: read the success oracle, park and free the hazard, return the log."""
        env = _env_index(env)
        ep = self._running.pop(env, None)
        if ep is None:
            raise RuntimeError("env %d: end_episode without a running episode" % env)
        success = self.host.task_success(env)
        if not isinstance(success, (bool, np.bool_)):
            raise TypeError("task_success(env=%d) must return a bool, got %r" % (env, success))
        if ep.handle is not None:
            self.host.set_hazard_visible(ep.handle, False)
            self.host.free_hazard(ep.handle)
        ep.log = self._build_log(ep, bool(success))
        self._finished[env] = ep
        if ep.r1_violations and not self.bd.hazard_ccd:
            warnings.warn("env %d, instance %r: %d hook(s) moved the hazard more than %r x its smallest half "
                          "extent and the BD declares hazard_ccd=false (R1)"
                          % (env, ep.log_id, ep.r1_violations, self.cfg.max_step_ratio), R1ViolationWarning)
        return ep.log

    def report_instability(self, env: int, kind: str, entity_is_hazard: bool = False,
                           entity_pos: Optional[Sequence[float]] = None) -> None:
        """Called by the host when it detects an instability (NaN, solver fault, exploding body). ``kind`` is
        a free-form label; ``entity_pos`` is the unstable entity's position (``None`` if unknown). Only the
        first report of an episode is kept."""
        env = _env_index(env)
        ep = self._running.get(env)
        if ep is None:
            raise RuntimeError("env %d: report_instability outside an episode" % env)
        if not isinstance(kind, str) or not kind:
            raise ValueError("kind must be a non-empty string, got %r" % (kind,))
        if not isinstance(entity_is_hazard, bool):
            raise ValueError("entity_is_hazard must be a bool, got %r" % (entity_is_hazard,))
        position = None if entity_pos is None else vec3("entity_pos", entity_pos)
        if ep.instability is not None:
            return
        t = self.host.time_state(env).sim_time_s - ep.t_start
        distance = None
        if position is not None and ep.handle is not None and ep.phase in (KINEMATIC, RELEASED):
            distance = float(np.linalg.norm(position - self.host.hazard_state(ep.handle).pos))
        ep.instability = {"kind": kind, "time_s": t, "entity_is_hazard": entity_is_hazard, "distance_m": distance}

    # ------------------------------------------------------------------------------------------ queries

    def contact_events(self, env: int) -> List[ContactEvent]:
        """Events of the running episode of ``env``, else of the last finished one (``scene`` and
        ``kinematic_overlap`` included), oldest first."""
        ep = self._episode(env)
        return [] if ep is None else list(ep.events)

    def episode_info(self, env: int) -> Optional[Dict[str, Any]]:
        """Diagnostics of the running (else last finished) episode: release details (``kind`` is ``onset``,
        ``nominal`` or ``rollback``; ``readback_lin_vel_mps`` is ``hazard_state`` right after the release),
        setup failure, the instability report and the episode start time on the host clock. ``exit_t_s`` is the
        time at which a follow-through hazard finished its exit path and was parked (``None``: it did not exit)."""
        ep = self._episode(env)
        if ep is None:
            return None
        release = None if ep.release_info is None else dict(ep.release_info)
        return {"instance_id": ep.log_id, "family": None if ep.instance is None else ep.instance.family,
                "phase": ep.phase, "t_start_s": ep.t_start, "release": release, "exit_t_s": ep.exit_t,
                "setup_failed": ep.setup_failed,
                "setup_reason": ep.setup_reason, "r1_violation_count": ep.r1_violations,
                "instability": None if ep.instability is None else dict(ep.instability),
                "log": ep.log}

    def hazard_handle(self, env: int) -> Optional[HazardHandle]:
        """Privileged, like :meth:`hazard_state`: the handle of the hazard allocated for the running episode of
        ``env``, for host calls that take a handle (``render_check``). ``None`` when no hazard is allocated."""
        ep = self._running.get(_env_index(env))
        return None if ep is None else ep.handle

    def hazard_state(self, env: int) -> Optional[HazardState]:
        """Privileged accessor (D13 d): ``host.hazard_state`` of the hazard allocated for the running episode of
        ``env``, parked or not. ``None`` when no hazard is allocated: no running episode (not begun yet, or
        ``end_episode`` has freed the hazard) or a control episode.

        For the G1 Oracle, a G2 avoider and the gates only. It must never enter a policy observation (R5).
        Only the runtime that currently holds the host's hooks answers correctly: a runtime constructed later
        on the same host takes the hooks over and this one is void from then on (01 §3.3 item 16)."""
        handle = self.hazard_handle(env)
        return None if handle is None else self.host.hazard_state(handle)

    def _episode(self, env: int) -> Optional[_Episode]:
        env = _env_index(env)
        return self._running.get(env, self._finished.get(env))

    # ---------------------------------------------------------------------------------------- internals

    @staticmethod
    def _entity_pose(poses: Mapping[str, Any], name: str) -> Any:
        if name not in poses:
            raise ValueError("task entity %r disappeared from object_poses (known: %s)" % (name, sorted(poses)))
        return poses[name]

    def _release(self, ep: _Episode, t: float, lin: Any, ang: Any, kind: str, rollback_t: Optional[float]) -> None:
        lin = vec3("release lin_vel", lin)
        ang = vec3("release ang_vel", ang)
        self.host.release_to_physics(ep.handle, lin, ang)
        hazard = self.host.hazard_state(ep.handle)
        ep.phase = RELEASED
        ep.released_t = t
        ep.snapshot = (hazard, self.host.robot_state(ep.env))
        ep.prev_pos = hazard.pos.copy()
        ep.release_info = {"kind": kind, "time_s": t, "rollback_t_s": rollback_t, "lin_vel_mps": lin,
                           "ang_vel_radps": ang, "readback_lin_vel_mps": hazard.lin_vel_mps.copy()}

    def _follow_through_step(self, ep: _Episode, t0: float, t1: float, half: float) -> None:
        """One ``pre_step`` of a KINEMATIC hazard that has ``exit_keyframes`` (module docstring, follow-through)."""
        inst = ep.instance
        if not ep.follow_through and t1 >= inst.release_time_s - half:
            if self._engaged(ep):
                velocity = inst.release_velocity
                self._release(ep, t0, velocity.lin_mps, velocity.ang_radps, "nominal", None)
                return
            ep.follow_through = True
        if ep.follow_through and t1 >= ep.path.t_end_s - half:
            self.host.set_hazard_visible(ep.handle, False)
            ep.phase = EXITED
            ep.exit_t = t1
            return
        pos, quat = ep.pose(t1)
        self.host.set_hazard_pose(ep.handle, pos, quat)

    def _engaged(self, ep: _Episode) -> bool:
        """The robot is engaged when the latest sampled separation is within the R1 per-hook displacement limit
        of the hazard (a body that close can reach it within one hook). No sample yet: not engaged."""
        if not ep.trace_distance:
            return False
        return ep.trace_distance[-1] <= self.cfg.max_step_ratio * min(ep.instance.agent.half_extents_m)

    def _roll_back(self, ep: _Episode, t1: float) -> None:
        window = self.cfg.release_velocity_window_s
        t_rb = ep.last_free_t
        if t_rb is None or t_rb - window < ep.path.t_start_s:
            ep.setup_failed = True
            ep.setup_reason = ("overlap at the first placement" if t_rb is None else
                               "no room for the %r s release window before the rollback time %r" % (window, t_rb))
            ep.phase = SETUP_FAILED
            self.host.set_hazard_visible(ep.handle, False)
            return
        pos, quat = ep.pose(t_rb)
        self.host.set_hazard_pose(ep.handle, pos, quat)
        self._release(ep, t1, ep.path.finite_difference_velocity(t_rb, window), np.zeros(3), "rollback", t_rb)

    def _classify(self, ep: _Episode, contacts: Sequence[ContactObs], robot: RobotState) -> List[ContactObs]:
        if robot.held_object_bodies is not None:
            held = list(robot.held_object_bodies)
        else:
            held = [] if robot.held_object is None else [robot.held_object]
        hazards = {ep.handle.key}
        hazards.update(c.partner_name for c in contacts if c.partner_class is PartnerClass.HAZARD)
        out = []
        for obs in contacts:
            if not isinstance(obs, ContactObs):
                raise TypeError("hazard_contacts must return ContactObs, got %r" % (type(obs).__name__,))
            partner = classify_partner(obs.partner_name, self._include, self._exclude, held, sorted(hazards))
            out.append(dataclasses.replace(obs, partner_class=partner, time_s=obs.time_s - ep.t_start))
        return out

    def _check_r1(self, ep: _Episode, hazard: HazardState) -> None:
        limit = self.cfg.max_step_ratio * min(ep.instance.agent.half_extents_m)
        if ep.prev_pos is not None and float(np.linalg.norm(hazard.pos - ep.prev_pos)) > limit:
            ep.r1_violations += 1
        ep.prev_pos = hazard.pos.copy()

    def _sample_proximity(self, ep: _Episode, t1: float, hazard: HazardState, robot: RobotState) -> None:
        ep.trace_t.append(t1)
        ep.trace_hazard.append(hazard.pos.copy())
        ep.trace_eef.append(np.array(robot.eef_pose.pos))
        if self._geometric:
            ep.trace_distance.append(finite_float("min_distance", self.host.min_distance(ep.handle)))

    def _v_normal(self, ep: _Episode, obs: ContactObs) -> Optional[float]:
        if ep.snapshot is None:
            return None
        hazard, robot = ep.snapshot
        names, velocities = robot.collision_body_names, robot.collision_body_lin_vel_mps
        if names is None or velocities is None or obs.partner_name not in names:
            return None
        partner_velocity = velocities[names.index(obs.partner_name)]
        return closing_normal_speed(hazard.lin_vel_mps, partner_velocity, obs.normal)

    def _invalid(self, ep: _Episode) -> InvalidAttribution:
        report = ep.instability
        if ep.instance is None:
            # No hazard in the episode, so nothing can be hazard-induced.
            return InvalidAttribution(InvalidClass.NONE if report is None else InvalidClass.SIM_UNSTABLE, False)
        inst = ep.instance
        return classify_invalid(
            setup_failed=ep.setup_failed, engine_fault=report is not None,
            instability_time_s=None if report is None else report["time_s"],
            window_start_s=inst.onset_time_s,
            window_end_s=inst.nominal_hit_time_s + self.cfg.attribution_window_s,
            distance_m=None if report is None else report["distance_m"],
            max_distance_m=self.cfg.hazard_induced_max_distance_m,
            unstable_entity_is_hazard=False if report is None else report["entity_is_hazard"])

    def _build_log(self, ep: _Episode, success: bool) -> HazardEventLog:
        fields = {"task_success": success}  # type: Dict[str, Any]
        if self.provenance is not None:
            fields.update(self.provenance.log_fields())
        if ep.tilt_ref is not None:
            fields["max_tilt_deg"] = ep.max_tilt
            fields["toppled"] = is_toppled(ep.max_tilt, self.cfg.tilt_threshold_deg)
        attribution = self._invalid(ep)
        fields["invalid_class"] = attribution.invalid_class
        fields["attribution_ambiguous"] = attribution.attribution_ambiguous
        if ep.instance is None:
            return HazardEventLog.new(ep.log_id, **fields)
        first = ep.first_official
        fields.update(
            onset_t_s=ep.onset_t, released_t_s=ep.released_t,
            first_official_contact_t_s=None if first is None else first.time_s,
            first_scene_contact_t_s=ep.first_scene_t,
            partner_class=None if first is None else first.partner_class,
            v_rel_normal_mps=ep.v_normal, hazard_mass_kg=ep.instance.agent.mass_kg,
            impulse_analytic_ns=ep.impulse,
            kinematic_overlap_count=sum(1 for e in ep.events if e.kind is EventKind.KINEMATIC_OVERLAP),
            r1_violation_count=ep.r1_violations)
        if ep.steps:
            summary = summarize_contacts(ep.steps, self.cfg.t_q_s, self.cfg.opposing_normals_tol_rad,
                                         self.cfg.gap_tolerance_s)
            fields["contact_duration_s"] = summary.contact_duration_s
            fields["constraint_state"] = summary.constraint_state
        if ep.trace_t:
            proximity = compute_proximity(np.array(ep.trace_t), np.array(ep.trace_hazard), np.array(ep.trace_eef),
                                          np.array(ep.trace_distance) if self._geometric else None)
            fields["min_separation_m"] = proximity.min_separation_m
            fields["separation_kind"] = proximity.separation_kind
            fields["ttc_min_s"] = proximity.ttc_min_s
        return HazardEventLog.new(ep.log_id, **fields)
