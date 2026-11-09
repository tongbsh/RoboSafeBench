"""Episode driving shared by probe recording, the gates and the T_stop protocol (D11 a).

One :class:`EpisodeSession` per host: it builds the ``HazardRuntime`` (which installs the hooks on the host;
a session created later on the same host takes them over and the earlier one is void, 01 §3.3 item 16, so use
one session per host at a time) and drives episodes through the ``EpisodeDriver`` methods only: ``arm`` ->
``reset_episode`` -> ``step`` ... -> ``end_episode``. The reference policy reads ``policy_observation`` and
nothing else (R5).

Privileged controllers (the G1 Oracle, a G2 avoider) read the hazard through the runtime's accessor
``session.runtime.hazard_state(env)`` (D13 d); the policy never sees it.

``ReferencePolicy`` is the abstraction of the probe sources and of the G1/G2 reference: ``kind`` is one of
:data:`REFERENCE_KINDS`, ``name`` identifies it, ``under_test`` must be True for a policy under test (R4:
such a policy may not generate its own instances). ``reset`` is called after every ``reset_episode``;
``act(observation)`` returns the benchmark-native action for one control step and is **not** called while a
controller holds the robot (so a replayed demonstration pauses and resumes).
"""
import dataclasses
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from .host import DRIVER_METHODS, PartnerClass, require_body_quats, require_methods, robot_geometry_offered
from .runtime import KINEMATIC, RELEASED, HazardRuntime, Provenance, RuntimeConfig
from .semantics import ContactEvent, EventKind, HazardEventLog, classify_partner
from .spec.instance import HazardInstance

REFERENCE_KINDS = ("demo_replay", "scripted", "external_policy")
ACTIVE_PHASES = (KINEMATIC, RELEASED)
_ROBOT_SIDE = (PartnerClass.ROBOT_LINK, PartnerClass.HELD_OBJECT)


@runtime_checkable
class ReferencePolicy(Protocol):
    kind: str
    name: str
    under_test: bool

    def reset(self, task_id: str, layout_id: str, seed: int) -> None:
        ...

    def act(self, observation: Dict[str, Any]) -> Any:
        ...


def describe_policy(policy: Any) -> Dict[str, Any]:
    """``{"kind", "name", "under_test"}`` of a reference policy, validated."""
    for attr in ("kind", "name", "under_test", "reset", "act"):
        if not hasattr(policy, attr):
            raise TypeError("%s is not a ReferencePolicy: missing %r" % (type(policy).__name__, attr))
    if policy.kind not in REFERENCE_KINDS:
        raise ValueError("policy kind %r is not one of %s" % (policy.kind, list(REFERENCE_KINDS)))
    if not isinstance(policy.name, str) or not policy.name:
        raise ValueError("policy name must be a non-empty string, got %r" % (policy.name,))
    if not isinstance(policy.under_test, bool):
        raise ValueError("policy under_test must be a bool, got %r" % (policy.under_test,))
    return {"kind": policy.kind, "name": policy.name, "under_test": policy.under_test}


class PlainController:
    """The policy acts every control step (no reaction to the hazard)."""

    def reset(self) -> None:
        pass

    def decide(self, session: "EpisodeSession", env: int, t_s: float, observation: Dict[str, Any],
               policy: Any, control_dt_s: float) -> Any:
        return policy.act(observation)

    def summary(self) -> Dict[str, Any]:
        return {"controller": "plain"}


@dataclasses.dataclass
class EpisodeResult:
    """What one driven episode produced. ``times_s`` / ``eef_pos`` / ``eef_speed_mps`` hold the state right
    after the reset and after every control step (seconds from the episode start). ``success_t_s`` is the
    first sample at which ``task_success`` was true (``None``: never). ``official_contact_count`` counts
    ``dynamic`` events with a robot-side partner (D12 f). ``first_visible_t_s`` is the first sample at which
    ``render_check`` saw the moving hazard (only with ``measure_visibility``; ``None`` otherwise or never)."""
    log: HazardEventLog
    events: List[ContactEvent]
    info: Dict[str, Any]
    times_s: np.ndarray
    eef_pos: np.ndarray
    eef_speed_mps: np.ndarray
    success_t_s: Optional[float]
    official_contact_count: int
    first_visible_t_s: Optional[float]
    render_check_available: bool
    controller: Dict[str, Any]
    trace_steps: List[Dict[str, Any]]

    @property
    def success(self) -> bool:
        return self.success_t_s is not None


def _official(events: List[ContactEvent]) -> List[ContactEvent]:
    return [e for e in events if e.kind is EventKind.DYNAMIC and e.partner_class in _ROBOT_SIDE]


class EpisodeSession:
    def __init__(self, host: Any, runtime_cfg: RuntimeConfig, provenance: Optional[Provenance] = None) -> None:
        require_methods(host)
        require_methods(host, DRIVER_METHODS)
        self.host = host
        self.runtime = HazardRuntime(host, runtime_cfg, provenance)
        self.bd = self.runtime.bd

    # ------------------------------------------------------------------------------------------- queries

    def phase(self, env: int) -> Optional[str]:
        info = self.runtime.episode_info(env)
        return None if info is None else info["phase"]

    def geometry_offered(self) -> bool:
        """Whether the host offers robot geometry (``robot_collision_points`` and a numeric BD
        ``robot_point_spacing_m``, D15 a); a mismatch between the two is a ``CapabilityError``."""
        return robot_geometry_offered(self.host, self.runtime.bd)

    def robot_geometry(self, env: int) -> Optional[Dict[str, Any]]:
        """The trace header's ``robot_geometry`` (D15 a): ``{"spacing_m", "points"}`` with the body-frame samples
        of the bodies the BD rules classify as ``robot_link``, or ``None`` when the host offers no geometry."""
        if not self.geometry_offered():
            return None
        rules = self.runtime.bd.robot_contact_bodies
        points = self.host.robot_collision_points(env)
        own = {}
        for name in sorted(points):
            if classify_partner(name, rules.include, rules.exclude, [], []) is PartnerClass.ROBOT_LINK:
                own[name] = np.asarray(points[name], dtype=np.float64).tolist()
        return {"spacing_m": float(self.runtime.bd.robot_point_spacing_m), "points": own}

    def _robot_bodies(self, robot: Any) -> Dict[str, Any]:
        """Trace fields of the collision bodies (D15 b): ``robot_bodies`` holds the robot's own bodies only
        (``robot_link`` per the BD rules, never the held object or its ``held_object_bodies``),
        ``robot_body_quats`` their orientations when the host reports ``collision_body_quat`` (a host with
        robot geometry must), ``held_object_bodies_pos`` the held object's bodies (``None`` when nothing is
        held)."""
        out = {"robot_bodies": None, "robot_body_quats": None,
               "held_object_bodies_pos": None}  # type: Dict[str, Any]
        if self.geometry_offered():
            require_body_quats(robot, self.host)
        if robot.collision_body_names is None:
            return out
        held = list(robot.held_object_bodies) if robot.held_object_bodies is not None else (
            [] if robot.held_object is None else [robot.held_object])
        rules = self.runtime.bd.robot_contact_bodies
        quats = robot.collision_body_quat
        own, own_quats, held_pos = {}, {}, {}  # type: Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]
        for i, (name, pos) in enumerate(zip(robot.collision_body_names, robot.collision_body_pos)):
            kind = classify_partner(name, rules.include, rules.exclude, held, [])
            if kind is PartnerClass.ROBOT_LINK:
                own[name] = [float(v) for v in pos]
                if quats is not None:
                    own_quats[name] = [float(v) for v in quats[i]]
            elif kind is PartnerClass.HELD_OBJECT:
                held_pos[name] = [float(v) for v in pos]
        out["robot_bodies"] = own
        out["robot_body_quats"] = own_quats if quats is not None else None
        out["held_object_bodies_pos"] = held_pos if robot.held_object is not None else None
        return out

    def _trace_step(self, env: int, t: float, robot: Any, in_contact: bool, phase: Optional[str]) -> Dict[str, Any]:
        poses = self.host.object_poses(env)
        bodies = self._robot_bodies(robot)
        return {
            "t_s": float(t),
            "eef_pos": [float(v) for v in robot.eef_pose.pos],
            "eef_quat": [float(v) for v in robot.eef_pose.quat],
            "eef_lin_vel_mps": [float(v) for v in robot.eef_lin_vel_mps],
            "gripper": robot.gripper_opening_m,
            "held_object": robot.held_object,
            "robot_bodies": bodies["robot_bodies"],
            "robot_body_quats": bodies["robot_body_quats"],
            "held_object_bodies_pos": bodies["held_object_bodies_pos"],
            "object_poses": {name: {"pos": [float(v) for v in pose.pos], "quat": [float(v) for v in pose.quat]}
                             for name, pose in sorted(poses.items())},
            "in_contact": bool(in_contact),
            "phase": phase,
        }

    # ------------------------------------------------------------------------------------------ driving

    def run_episode(self, env: int, instance: Optional[HazardInstance], policy: Any, task_id: str,
                    layout_id: str, seed: int, horizon_s: float, controller: Any = None,
                    control_id: Optional[str] = None, record_trace: bool = False, stop_on_success: bool = False,
                    measure_visibility: bool = False) -> EpisodeResult:
        """``arm`` -> ``reset_episode`` -> control steps until ``horizon_s`` (or the first success with
        ``stop_on_success``, or when ``controller.finished`` becomes true) -> ``end_episode``."""
        controller = controller if controller is not None else PlainController()
        describe_policy(policy)
        if instance is None:
            self.runtime.arm(env, None, control_id=control_id)
        else:
            self.runtime.arm(env, instance)
        self.host.reset_episode(env, task_id, layout_id, seed)
        policy.reset(task_id, layout_id, seed)
        controller.reset()
        clock = self.host.time_state(env)
        t0, control_dt = clock.sim_time_s, clock.control_dt_s
        n_steps = int(round(horizon_s / control_dt))
        if n_steps < 1:
            raise ValueError("horizon_s=%r is shorter than one control step (%r s)" % (horizon_s, control_dt))
        render = getattr(self.host, "render_check", None)
        render_ok = callable(render)
        times, eef, speeds, steps = [], [], [], []  # type: List[float], List[Any], List[float], List[Dict[str, Any]]
        success_t = None  # type: Optional[float]
        first_visible = None  # type: Optional[float]
        seen = 0

        def sample(in_contact: bool) -> float:
            t = self.host.time_state(env).sim_time_s - t0
            robot = self.host.robot_state(env)
            times.append(t)
            eef.append(np.array(robot.eef_pose.pos, dtype=np.float64))
            speeds.append(float(np.linalg.norm(robot.eef_lin_vel_mps)))
            if record_trace:
                steps.append(self._trace_step(env, t, robot, in_contact,
                                              None if instance is None else self.phase(env)))
            return t

        t = sample(False)
        for _ in range(n_steps):
            observation = self.host.policy_observation(env)
            action = controller.decide(self, env, t, observation, policy, control_dt)
            self.host.step(env, action)
            events = self.runtime.contact_events(env)
            fresh, seen = events[seen:], len(events)
            t = sample(bool(_official(fresh)))
            if measure_visibility and render_ok and first_visible is None:
                handle = self.runtime.hazard_handle(env)
                if handle is not None and self.phase(env) in ACTIVE_PHASES and render(handle):
                    first_visible = t
            if success_t is None and self.host.task_success(env):
                success_t = t
            if stop_on_success and success_t is not None:
                break
            if getattr(controller, "finished", False):
                break
        log = self.runtime.end_episode(env)
        info = dict(self.runtime.episode_info(env))
        info.pop("log", None)
        events = self.runtime.contact_events(env)
        return EpisodeResult(log=log, events=events, info=info, times_s=np.array(times),
                             eef_pos=np.array(eef).reshape(-1, 3), eef_speed_mps=np.array(speeds),
                             success_t_s=success_t, official_contact_count=len(_official(events)),
                             first_visible_t_s=first_visible, render_check_available=render_ok,
                             controller=controller.summary(), trace_steps=steps)
