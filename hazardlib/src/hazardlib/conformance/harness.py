"""Probe harness of the conformance suite (T05). Generic: it uses nothing but the ``HazardHost`` /
``EpisodeDriver`` contract (01 §3) and a real :class:`~hazardlib.runtime.HazardRuntime`.

* :class:`RecordingHost` wraps the host handed to the runtime. It forwards every call and records what the checks
  need: each ``set_hazard_pose`` / ``release_to_physics`` together with an immediate ``hazard_state`` read-back
  (01 §3.1: a teleport leaves no residual velocity; D12 e 8: the release velocity is readable at once), every
  ``hazard_contacts`` result with the time and whether the hazard was parked, every ``hazard_state`` the runtime
  reads, and while parked also the parked pose and ``render_check``.
* :class:`HookTap` sits between the host and the runtime's hooks (``install_hooks`` receives the tap). It records
  the host clock at every hook, the scene as ``begin_episode`` saw it and the EEF state after every physics step.
* :class:`Probe` owns one recording host and one runtime and drives episodes (``arm`` -> ``reset_episode`` ->
  ``step`` -> ``end_episode``), sampling ``policy_observation`` and ``task_success`` after the reset and after
  every control step.
* Probe instances: synthetic ``HazardInstance`` objects (spec id ``conformance.probe``) whose motion the checks
  choose freely (a hazard parked, floating, driven along a line, released toward a body, dropped), and
  :func:`build_nominal_hit_instance`, the D1 builder of C13.
"""
import copy
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._numeric import positive_float, unit_vec3, vec3
from ..host import CapabilityError, ContactObs, HazardHandle, HazardState, RobotState
from ..kinematics.hit import FAMILY_FALL, FAMILY_REACH, FAMILY_SLIDE, build_hit_path
from ..kinematics.path import Path
from ..kinematics.subdivide import hook_times
from ..runtime import HazardRuntime, RuntimeConfig
from ..semantics import ContactEvent, HazardEventLog
from ..spec.hazard_spec import AgentSpec
from ..spec.instance import HazardInstance
from .target import EpisodeRef

PROBE_SPEC_ID = "conformance.probe"
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


class ObservationFormatError(ValueError):
    """``policy_observation`` broke the 01 §3.1a format (images ``H x W x 3 uint8``, vectors 1-D numeric)."""


# ------------------------------------------------------------------------------------------------ records

@dataclasses.dataclass
class PoseRecord:
    env: int
    key: str
    t_s: float
    pos: np.ndarray
    quat: np.ndarray
    readback: HazardState


@dataclasses.dataclass
class ReleaseRecord:
    env: int
    key: str
    t_s: float
    lin_vel: np.ndarray
    ang_vel: np.ndarray
    readback: HazardState


@dataclasses.dataclass
class ContactRecord:
    env: int
    key: str
    t_s: float
    parked: bool
    contacts: List[ContactObs]


@dataclasses.dataclass
class StateRecord:
    env: int
    key: str
    t_s: float
    state: HazardState


@dataclasses.dataclass
class HookMark:
    env: int
    kind: str  # "begin" | "pre" | "post"
    sim_time_s: float
    physics_dt_s: float
    control_dt_s: float


# ----------------------------------------------------------------------------------------- tap and proxy

class HookTap:
    """Forwards the host's hook calls to the runtime and records the host clock and the EEF at each of them.
    Any other attribute (``report_instability`` ...) is the runtime's."""

    def __init__(self, runtime: Any, host: Any) -> None:
        self._runtime = runtime
        self._host = host
        self.marks = []  # type: List[HookMark]
        self.begin_scene = {}  # type: Dict[int, Dict[str, Any]]
        self.robot = {}  # type: Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("_runtime", "_host"):
            raise AttributeError(name)
        return getattr(self._runtime, name)

    def _mark(self, env: int, kind: str) -> float:
        state = self._host.time_state(env)
        self.marks.append(HookMark(env, kind, state.sim_time_s, state.physics_dt_s, state.control_dt_s))
        return state.sim_time_s

    def begin_episode(self, env: int) -> None:
        self._runtime.begin_episode(env)
        t = self._mark(env, "begin")
        robot = self._host.robot_state(env)
        poses = self._host.object_poses(env)
        self.begin_scene[env] = {"t_s": t, "eef": np.array(robot.eef_pose.pos),
                                 "objects": {k: (np.array(p.pos), np.array(p.quat)) for k, p in poses.items()}}
        self.robot[env] = []

    def pre_step(self, env: int) -> None:
        self._mark(env, "pre")
        self._runtime.pre_step(env)

    def post_step(self, env: int) -> None:
        self._runtime.post_step(env)
        t = self._mark(env, "post")
        robot = self._host.robot_state(env)
        self.robot.setdefault(env, []).append((t, np.array(robot.eef_pose.pos), np.array(robot.eef_lin_vel_mps)))

    def end_episode(self, env: int) -> HazardEventLog:
        return self._runtime.end_episode(env)


class RecordingHost:
    """The host as the runtime sees it during a probe (see the module docstring). Optional methods are not
    defined here, so a host without them stays without them."""

    def __init__(self, host: Any) -> None:
        self._host = host
        self.tap = None  # type: Optional[HookTap]
        self.handles = []  # type: List[HazardHandle]
        self.visible = {}  # type: Dict[str, bool]
        self.poses = []  # type: List[PoseRecord]
        self.releases = []  # type: List[ReleaseRecord]
        self.contacts = []  # type: List[ContactRecord]
        self.states = []  # type: List[StateRecord]
        self.parked_states = []  # type: List[StateRecord]
        self.parked_render = []  # type: List[Tuple[int, str, float, bool]]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name == "_host":
            raise AttributeError(name)
        return getattr(self._host, name)

    def _now(self, env: int) -> float:
        return self._host.time_state(env).sim_time_s

    def install_hooks(self, runtime: Any) -> None:
        self.tap = HookTap(runtime, self._host)
        self._host.install_hooks(self.tap)

    def alloc_hazard(self, env: int, agent: AgentSpec) -> HazardHandle:
        handle = self._host.alloc_hazard(env, agent)
        self.handles.append(handle)
        self.visible[handle.key] = False
        return handle

    def set_hazard_pose(self, h: HazardHandle, pos: Sequence[float], quat: Sequence[float]) -> None:
        self._host.set_hazard_pose(h, pos, quat)
        self.poses.append(PoseRecord(h.env, h.key, self._now(h.env), np.array(pos, dtype=float),
                                     np.array(quat, dtype=float), self._host.hazard_state(h)))

    def set_hazard_visible(self, h: HazardHandle, visible: bool) -> None:
        self._host.set_hazard_visible(h, visible)
        self.visible[h.key] = bool(visible)
        render_check = getattr(self._host, "render_check", None)
        if not visible and callable(render_check):
            self.parked_render.append((h.env, h.key, self._now(h.env), bool(render_check(h))))

    def release_to_physics(self, h: HazardHandle, lin_vel: Sequence[float], ang_vel: Sequence[float]) -> None:
        self._host.release_to_physics(h, lin_vel, ang_vel)
        self.releases.append(ReleaseRecord(h.env, h.key, self._now(h.env), np.array(lin_vel, dtype=float),
                                           np.array(ang_vel, dtype=float), self._host.hazard_state(h)))

    def hazard_contacts(self, h: HazardHandle) -> List[ContactObs]:
        out = list(self._host.hazard_contacts(h))
        parked = not self.visible.get(h.key, False)
        t = self._now(h.env)
        self.contacts.append(ContactRecord(h.env, h.key, t, parked, out))
        if parked:
            self.parked_states.append(StateRecord(h.env, h.key, t, self._host.hazard_state(h)))
        return out

    def hazard_state(self, h: HazardHandle) -> HazardState:
        state = self._host.hazard_state(h)
        self.states.append(StateRecord(h.env, h.key, self._now(h.env), state))
        return state


# ----------------------------------------------------------------------------------------------- episodes

@dataclasses.dataclass
class Sample:
    """What the policy saw and what the oracle said, ``t_s`` seconds after the reset."""
    t_s: float
    obs: Dict[str, Any]
    success: bool


@dataclasses.dataclass
class EpisodeRun:
    env: int
    instance: Optional[HazardInstance]
    ref: EpisodeRef
    t_start: float
    control_dt: float
    handle: Optional[HazardHandle]
    samples: List[Sample]
    begin_scene: Dict[str, Any]
    robot: List[Tuple[float, np.ndarray, np.ndarray]]
    log: Optional[HazardEventLog] = None
    events: Optional[List[ContactEvent]] = None
    info: Optional[Dict[str, Any]] = None


def _copy_obs(obs: Any) -> Dict[str, Any]:
    if not isinstance(obs, dict):
        raise ObservationFormatError("policy_observation must return a dict, got %r" % (type(obs).__name__,))
    return {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else copy.deepcopy(v) for k, v in obs.items()}


class Probe:
    """One recording host and one runtime around ``host`` (a fresh runtime per probe; the host's hooks now point
    at it)."""

    def __init__(self, host: Any, runtime_cfg: RuntimeConfig) -> None:
        self.host = host
        self.rec = RecordingHost(host)
        self.runtime = HazardRuntime(self.rec, runtime_cfg)
        if self.rec.tap is None:
            raise CapabilityError("HazardRuntime did not call install_hooks on the host")
        self.tap = self.rec.tap
        self._open = {}  # type: Dict[int, EpisodeRun]

    # -- step by step (several environments interleaved) --------------------------------------------------

    def start(self, env: int, instance: Optional[HazardInstance], ref: EpisodeRef, control_id: Optional[str] = None,
              task_entities: Optional[Sequence[str]] = None, control_onset_s: Optional[float] = None) -> EpisodeRun:
        if env in self._open:
            raise RuntimeError("env %d already has an open probe episode" % env)
        self.runtime.arm(env, instance, task_entities=task_entities, control_id=control_id,
                         control_onset_s=control_onset_s)
        n_handles = len(self.rec.handles)
        self.tap.begin_scene.pop(env, None)
        self.host.reset_episode(env, ref.task_id, ref.layout_id, ref.seed)
        if env not in self.tap.begin_scene:
            raise CapabilityError("reset_episode did not fire the reset hook (runtime.begin_episode)")
        handle = None
        if instance is not None:
            new = [h for h in self.rec.handles[n_handles:] if h.env == env]
            handle = new[-1] if new else None
        scene = self.tap.begin_scene[env]
        run = EpisodeRun(env=env, instance=instance, ref=ref, t_start=scene["t_s"],
                         control_dt=self.host.time_state(env).control_dt_s, handle=handle, samples=[],
                         begin_scene=scene, robot=self.tap.robot[env])
        self._open[env] = run
        self._sample(run)
        return run

    def step(self, env: int, action: Any) -> None:
        run = self._open[env]
        self.host.step(env, action)
        self._sample(run)

    def finish(self, env: int) -> EpisodeRun:
        run = self._open.pop(env)
        run.log = self.runtime.end_episode(env)
        run.events = self.runtime.contact_events(env)
        run.info = self.runtime.episode_info(env)
        return run

    def abort(self) -> None:
        """End every open episode after an error, so the host is left with its hazards parked and freed."""
        for env in list(self._open):
            self._open.pop(env)
            try:
                self.runtime.end_episode(env)
            except Exception:  # noqa: BLE001 - best-effort cleanup after a failure that is already reported
                pass

    def _sample(self, run: EpisodeRun) -> None:
        t = self.host.time_state(run.env).sim_time_s - run.t_start
        success = self.host.task_success(run.env)
        if not isinstance(success, (bool, np.bool_)):
            raise TypeError("task_success(env=%d) must return a bool, got %r" % (run.env, success))
        run.samples.append(Sample(t, _copy_obs(self.host.policy_observation(run.env)), bool(success)))

    # -- whole episode --------------------------------------------------------------------------------------

    def run(self, env: int, instance: Optional[HazardInstance], ref: EpisodeRef, duration_s: float,
            actions: Callable[[float], Any], control_id: Optional[str] = None,
            on_step: Optional[Callable[[EpisodeRun, int], None]] = None) -> EpisodeRun:
        """``actions(t)`` gives the action of the control step starting at episode time ``t``; ``on_step(run, k)``
        runs after control step ``k``."""
        duration = positive_float("duration_s", duration_s)
        try:
            run = self.start(env, instance, ref, control_id=control_id)
            steps = max(1, int(round(duration / run.control_dt)))
            for k in range(steps):
                self.step(env, actions(k * run.control_dt))
                if on_step is not None:
                    on_step(run, k)
            return self.finish(env)
        except BaseException:
            self.abort()
            raise

    def hold(self, env: int) -> Callable[[float], Any]:
        return lambda t: self.host.hold_action(env)

    # -- record queries ---------------------------------------------------------------------------------

    def robot_trace(self, run: EpisodeRun) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(t, eef_pos, eef_vel)`` after every physics step of ``run``, times relative to its reset."""
        rows = run.robot
        if not rows:
            return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
        return (np.array([r[0] for r in rows]) - run.t_start, np.array([r[1] for r in rows]),
                np.array([r[2] for r in rows]))

    def records(self, items: Sequence[Any], run: EpisodeRun) -> List[Any]:
        """The records of ``items`` (poses, releases, contacts, states ...) that belong to ``run``'s hazard."""
        if run.handle is None:
            return []
        return [r for r in items if r.env == run.env and r.key == run.handle.key]


# ------------------------------------------------------------------------------------ observation compare

def split_observation(obs: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """``(images, vectors)``; anything else violates 01 §3.1a."""
    images, vectors = {}, {}
    for key, value in obs.items():
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8:
            images[key] = arr
        elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
            vectors[key] = arr
        else:
            raise ObservationFormatError("policy_observation[%r] is neither an H x W x 3 uint8 image nor a 1-D "
                                         "numeric vector (shape %s, dtype %s)" % (key, arr.shape, arr.dtype))
    return images, vectors


@dataclasses.dataclass
class ObsDiff:
    """Difference of two observations. ``mismatch`` lists keys missing on one side or with different shapes;
    ``vector_max_abs`` / ``image_max_abs`` are the largest absolute element differences and ``image_pixels`` the
    largest number of differing pixels in one image."""
    mismatch: List[str]
    vector_max_abs: float
    image_max_abs: float
    image_pixels: int

    @property
    def identical(self) -> bool:
        return not self.mismatch and self.vector_max_abs == 0.0 and self.image_max_abs == 0.0


def observation_diff(a: Dict[str, Any], b: Dict[str, Any]) -> ObsDiff:
    images_a, vectors_a = split_observation(a)
    images_b, vectors_b = split_observation(b)
    mismatch = sorted(set(images_a) ^ set(images_b)) + sorted(set(vectors_a) ^ set(vectors_b))
    vector_max = image_max = 0.0
    pixels = 0
    for key in sorted(set(vectors_a) & set(vectors_b)):
        va, vb = vectors_a[key].astype(np.float64), vectors_b[key].astype(np.float64)
        if va.shape != vb.shape:
            mismatch.append("%s: shape %s vs %s" % (key, va.shape, vb.shape))
            continue
        if va.size:
            if not (np.all(np.isfinite(va)) and np.all(np.isfinite(vb))):
                mismatch.append("%s: non-finite values" % key)
                continue
            vector_max = max(vector_max, float(np.max(np.abs(va - vb))))
    for key in sorted(set(images_a) & set(images_b)):
        ia, ib = images_a[key].astype(np.int64), images_b[key].astype(np.int64)
        if ia.shape != ib.shape:
            mismatch.append("%s: shape %s vs %s" % (key, ia.shape, ib.shape))
            continue
        delta = np.abs(ia - ib)
        if delta.size:
            image_max = max(image_max, float(np.max(delta)))
            pixels = max(pixels, int(np.count_nonzero(np.any(delta > 0, axis=2))))
    return ObsDiff(mismatch, vector_max, image_max, pixels)


# ------------------------------------------------------------------------------------------------ geometry

def support_radius(agent: AgentSpec, direction: Sequence[float]) -> float:
    """Extent of the probe body along ``direction`` from its centre (identity orientation): the radius of a
    sphere, ``sum |d_i| h_i`` of an axis-aligned box. Other shapes are not probe bodies."""
    d = np.abs(unit_vec3("direction", direction))
    extents = [float(h) for h in agent.half_extents_m]
    if agent.shape == "sphere":
        if len(set(extents)) != 1:
            raise CapabilityError("a sphere probe agent needs equal half extents, got %r" % (extents,))
        return extents[0]
    if agent.shape == "box":
        if len(extents) != 3:
            raise CapabilityError("a box probe agent needs 3 half extents, got %r" % (extents,))
        return float(d @ np.array(extents))
    raise CapabilityError("probe agents must be spheres or boxes, got shape %r" % (agent.shape,))


def body_position(robot: RobotState, name: str) -> np.ndarray:
    names = robot.collision_body_names
    if names is None or name not in names:
        raise CapabilityError("robot_state() does not report a collision body named %r (names: %r)" % (name, names))
    return np.array(robot.collision_body_pos[list(names).index(name)])


# ----------------------------------------------------------------------------------------- probe instances

def _instance(instance_id: str, family: str, agent: AgentSpec, times: Sequence[float], pos: Any, quat: Any,
              onset_s: float, release_s: float, release_lin: Sequence[float], hit_time_s: float,
              hit_point: Sequence[float], provenance: Dict[str, Any]) -> HazardInstance:
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    return HazardInstance.from_dict({
        "instance_id": instance_id, "spec_id": PROBE_SPEC_ID, "family": family, "agent": agent.to_dict(),
        "keyframes": {"time_s": [float(t) for t in times], "pos": pos.tolist(), "quat": quat.tolist()},
        "nominal_hit_point": vec3("hit_point", hit_point).tolist(), "nominal_hit_time_s": float(hit_time_s),
        "onset_time_s": float(onset_s), "release_time_s": float(release_s),
        "release_velocity": {"lin_mps": vec3("release_lin", release_lin).tolist(), "ang_radps": [0.0, 0.0, 0.0]},
        "certificate": None, "provenance": dict(provenance, builder="hazardlib.conformance.harness"),
    })


def held_instance(instance_id: str, agent: AgentSpec, point: Sequence[float], onset_s: float,
                  release_s: float) -> HazardInstance:
    """A kinematic hazard that appears at ``point`` at ``onset_s`` and stays there; it would be released at rest at
    ``release_s`` (end the episode before that to keep it kinematic)."""
    point = vec3("point", point)
    return _instance(instance_id, FAMILY_SLIDE, agent, [onset_s, release_s], [point, point], [IDENTITY_QUAT] * 2,
                     onset_s, release_s, np.zeros(3), release_s, point, {"probe": "held"})


def strike_instance(instance_id: str, agent: AgentSpec, start: Sequence[float], velocity: Sequence[float],
                    onset_s: float, release_s: float, hit_time_s: float, hit_point: Sequence[float]) -> HazardInstance:
    """Kinematic and still at ``start`` from ``onset_s``; released with ``velocity`` at ``release_s`` (it leaves
    ``start`` one physics step before ``release_s``, D12 b). ``hit_time_s`` / ``hit_point`` are the expected
    first contact (attribution window only)."""
    start = vec3("start", start)
    return _instance(instance_id, FAMILY_SLIDE, agent, [onset_s, release_s], [start, start], [IDENTITY_QUAT] * 2,
                     onset_s, release_s, velocity, hit_time_s, hit_point, {"probe": "strike"})


def path_instance(instance_id: str, agent: AgentSpec, start: Sequence[float], velocity: Sequence[float],
                  onset_s: float, release_s: float, release_velocity: Sequence[float], keyframe_dt_s: float,
                  quat_at: Optional[Callable[[float], Sequence[float]]] = None) -> HazardInstance:
    """Kinematic from ``start`` along ``velocity`` (constant) from ``onset_s``, keyframes every ``keyframe_dt_s``,
    released with ``release_velocity`` at ``release_s``. ``quat_at(u)`` (``u`` in [0, 1] along the path) sets the
    keyframe orientations; identity when ``None``."""
    start, velocity = vec3("start", start), vec3("velocity", velocity)
    times = hook_times(onset_s, release_s, keyframe_dt_s)
    pos = start[None, :] + (times - onset_s)[:, None] * velocity[None, :]
    span = release_s - onset_s
    quats = [IDENTITY_QUAT if quat_at is None else tuple(quat_at((t - onset_s) / span)) for t in times]
    return _instance(instance_id, FAMILY_SLIDE, agent, times, pos, quats, onset_s, release_s, release_velocity,
                     release_s, pos[-1], {"probe": "path"})


def drop_instance(instance_id: str, agent: AgentSpec, point: Sequence[float], onset_s: float,
                  fall_time_s: float) -> HazardInstance:
    """A fall released at rest from ``point`` at ``onset_s``; ``fall_time_s`` only places the nominal hit."""
    point = vec3("point", point)
    end = onset_s + positive_float("fall_time_s", fall_time_s)
    return _instance(instance_id, FAMILY_FALL, agent, [onset_s, end], [point, point], [IDENTITY_QUAT] * 2,
                     onset_s, onset_s, np.zeros(3), end, point, {"probe": "drop"})


def build_nominal_hit_instance(instance_id: str, nominal_eef_path: Path, family: str, agent: AgentSpec,
                               t_hit_s: float, tau_s: float, params: Dict[str, Any], eef_contact_radius_m: float,
                               keyframe_dt_s: float) -> HazardInstance:
    """The suite's D1 builder: the centroid target is the nominal EEF position at ``t_hit_s`` plus
    ``standoff = r_hazard(n) + eef_contact_radius_m`` along the approach normal ``n`` (``+z`` for a fall,
    ``-params["direction"]`` for a slide / reach); ``build_hit_path`` solves the path, sampled every
    ``keyframe_dt_s``."""
    if family == FAMILY_FALL:
        normal = np.array([0.0, 0.0, 1.0])
    elif family in (FAMILY_SLIDE, FAMILY_REACH):
        normal = -unit_vec3("params.direction", params["direction"])
    else:
        raise CapabilityError("C13 has no nominal-hit construction for family %r" % (family,))
    standoff = support_radius(agent, normal) + float(eef_contact_radius_m)
    hit_point = nominal_eef_path.pos(t_hit_s) + standoff * normal
    plan = build_hit_path(family, nominal_eef_path, t_hit_s, hit_point, tau_s, params)
    times = hook_times(plan.path.t_start_s, plan.path.t_end_s, keyframe_dt_s)
    return HazardInstance.from_dict({
        "instance_id": instance_id, "spec_id": PROBE_SPEC_ID, "family": family, "agent": agent.to_dict(),
        "keyframes": {"time_s": times.tolist(), "pos": plan.path.pos(times).tolist(),
                      "quat": [list(IDENTITY_QUAT)] * times.size},
        "nominal_hit_point": plan.nominal_hit_point.tolist(), "nominal_hit_time_s": plan.nominal_hit_time_s,
        "onset_time_s": plan.onset_time_s, "release_time_s": plan.release_time_s,
        "release_velocity": {"lin_mps": plan.release_velocity.tolist(), "ang_radps": [0.0, 0.0, 0.0]},
        "certificate": None,
        "provenance": {"builder": "hazardlib.conformance.harness.build_nominal_hit_instance",
                       "standoff_m": standoff},
    })


def quat_angle_rad(a: Sequence[float], b: Sequence[float]) -> float:
    """Rotation angle between two (not necessarily unit) wxyz quaternions, accurate near zero."""
    qa = np.asarray(a, dtype=float)
    qb = np.asarray(b, dtype=float)
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    w = abs(float(qa @ qb))
    # relative rotation qa^-1 qb: its vector part has norm sin(angle / 2)
    w1, v1 = qa[0], -qa[1:]
    w2, v2 = qb[0], qb[1:]
    vec = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return float(2.0 * np.arctan2(np.linalg.norm(vec), w))
