"""What the conformance suite runs against: a host plus the scene facts the probes need, and the registry that
resolves ``--host <name>`` to such a target (T05, 01 §5).

A :class:`ConformanceTarget` bundles

* ``host``: an object implementing ``HazardHost`` and ``EpisodeDriver`` (01 §3.1, §3.1a);
* ``scenario``: a :class:`Scenario`, the declarative facts about the benchmark scene that a generic probe cannot
  guess (where free space is, which robot body may be struck from which side, where a dropped body lands, which
  task / layout / seed to reset to). It holds no thresholds: every tolerance and probe magnitude lives in
  ``configs/hazard/conformance.yaml``;
* ``motion_action``: the repeatable scripted robot motion of ``scenario.motion`` (demo replay or scripted expert),
  a function of the episode time at the start of a control step. C8, C10 and C13 replay it;
* optional code hooks: ``nominal_eef_path`` (the motion's known EEF path, when the host has one; otherwise C13
  records it by replay), ``build_hit_instance`` (the instance builder the pipeline uses on this host; C13 runs its
  instances in addition to the suite's own D1 builder) and ``inject_engine_fault`` (forces an engine fault that the
  host must detect and report through ``report_instability``; C11).

Adapter packages register a factory at import time: ``register_host("calvin", make_target)``. ``resolve_target``
imports the modules it is given, then ``hazardhosts.<name>`` if the name is still unknown, and calls the factory.
"""
import dataclasses
import importlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._numeric import finite_float, positive_float, unit_vec3, vec3
from ..host import DRIVER_METHODS, CapabilityError, require_methods
from ..kinematics.path import Path
from ..spec.hazard_spec import AgentSpec


@dataclasses.dataclass(frozen=True)
class EpisodeRef:
    """A reset target: ``EpisodeDriver.reset_episode(env, task_id, layout_id, seed)``."""
    task_id: str
    layout_id: str
    seed: int

    def __post_init__(self) -> None:
        for name in ("task_id", "layout_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError("EpisodeRef.%s must be a non-empty string, got %r" % (name, getattr(self, name)))
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("EpisodeRef.seed must be an int, got %r" % (self.seed,))


@dataclasses.dataclass(frozen=True)
class StrikeTarget:
    """A collision body the probes may hit while the robot holds still. ``body`` is its name in
    ``RobotState.collision_body_names``; a hazard travelling along ``direction`` through the body's reported
    position meets nothing else, before or after the body, and hits it head-on (contact normal close to
    ``-direction``). ``radius_m`` is the body's extent along that line, measured from its reported position."""
    body: str
    direction: Tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.body, str) or not self.body:
            raise ValueError("StrikeTarget.body must be a non-empty string, got %r" % (self.body,))
        object.__setattr__(self, "direction", tuple(unit_vec3("StrikeTarget.direction", self.direction)))
        object.__setattr__(self, "radius_m", positive_float("StrikeTarget.radius_m", self.radius_m))


@dataclasses.dataclass(frozen=True)
class Scenario:
    """Scene facts of one benchmark for the probes (see the module docstring).

    ``still``: an episode in which the robot executes ``host.hold_action`` throughout; its task should be one whose
    success oracle is true while the robot holds still, otherwise C7 cannot observe an oracle that reacts to the
    hazard (the check records that it had no power). ``motion``: the episode replayed with ``motion_action``.
    ``probe_agent``: the hazard body of every probe (sphere or axis-aligned box).
    ``free_point_m`` / ``free_direction``: during ``still`` the probe hazard can sit at the point, and travel
    ``probe.free_travel_m`` from it along the direction, without touching anything; the point is inside a policy
    camera's view. ``strike``: a robot link to hit (C3, C4, C6). ``held_strike``: a held-object body to hit, or
    ``None`` when the benchmark cannot hold an object in ``still`` (C4 records it as not constructible).
    ``drop_point_m``: a body released at rest there falls freely onto the top of a support surface at height
    ``drop_surface_height_m`` (C4, C5). ``motion_hit_time_s`` / ``motion_hit_direction``: where on the moving
    part of ``motion`` C13 places its nominal hits, and the slide / reach approach direction (perpendicular to
    the EEF motion keeps the first contact at the nominal one). ``motion_hold_time_s``: when C10 issues the hold
    during the motion. ``motion_duration_s``: how long the motion is replayed.
    """
    still: EpisodeRef
    motion: EpisodeRef
    probe_agent: AgentSpec
    free_point_m: Tuple[float, float, float]
    free_direction: Tuple[float, float, float]
    strike: StrikeTarget
    held_strike: Optional[StrikeTarget]
    drop_point_m: Tuple[float, float, float]
    drop_surface_height_m: float
    motion_hit_time_s: float
    motion_hit_direction: Tuple[float, float, float]
    motion_hold_time_s: float
    motion_duration_s: float

    def __post_init__(self) -> None:
        for name in ("still", "motion"):
            if not isinstance(getattr(self, name), EpisodeRef):
                raise ValueError("Scenario.%s must be an EpisodeRef" % name)
        if not isinstance(self.probe_agent, AgentSpec):
            raise ValueError("Scenario.probe_agent must be an AgentSpec")
        if not isinstance(self.strike, StrikeTarget):
            raise ValueError("Scenario.strike must be a StrikeTarget")
        if self.held_strike is not None and not isinstance(self.held_strike, StrikeTarget):
            raise ValueError("Scenario.held_strike must be a StrikeTarget or None")
        set_ = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731
        set_("free_point_m", tuple(vec3("free_point_m", self.free_point_m)))
        set_("free_direction", tuple(unit_vec3("free_direction", self.free_direction)))
        set_("drop_point_m", tuple(vec3("drop_point_m", self.drop_point_m)))
        set_("drop_surface_height_m", finite_float("drop_surface_height_m", self.drop_surface_height_m))
        set_("motion_hit_direction", tuple(unit_vec3("motion_hit_direction", self.motion_hit_direction)))
        for name in ("motion_hit_time_s", "motion_hold_time_s", "motion_duration_s"):
            set_(name, positive_float(name, getattr(self, name)))
        if not self.motion_hit_time_s < self.motion_duration_s:
            raise ValueError("motion_hit_time_s=%r must be before motion_duration_s=%r"
                             % (self.motion_hit_time_s, self.motion_duration_s))
        if not self.motion_hold_time_s < self.motion_duration_s:
            raise ValueError("motion_hold_time_s=%r must be before motion_duration_s=%r"
                             % (self.motion_hold_time_s, self.motion_duration_s))
        if self.drop_point_m[2] <= self.drop_surface_height_m:
            raise ValueError("drop_point_m must be above drop_surface_height_m")


@dataclasses.dataclass
class ConformanceTarget:
    """A host under test (see the module docstring). ``build_hit_instance`` takes the keyword arguments
    ``instance_id, family, agent, t_hit_s, tau_s, params, keyframe_dt_s`` and returns a ``HazardInstance`` hitting
    the nominal EEF path of ``scenario.motion`` at ``t_hit_s``. ``inject_engine_fault(env, hazard)`` forces an
    engine fault, during a running episode, in a scene body (``hazard=None``) or in the given hazard body."""
    name: str
    host: Any
    adapter_version: str
    scenario: Scenario
    motion_action: Callable[[float], Any]
    nominal_eef_path: Optional[Path] = None
    build_hit_instance: Optional[Callable[..., Any]] = None
    inject_engine_fault: Optional[Callable[[int, Any], None]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ConformanceTarget.name must be a non-empty string")
        if not isinstance(self.adapter_version, str) or not self.adapter_version:
            raise ValueError("ConformanceTarget.adapter_version must be a non-empty string")
        if not isinstance(self.scenario, Scenario):
            raise ValueError("ConformanceTarget.scenario must be a Scenario")
        if not callable(self.motion_action):
            raise ValueError("ConformanceTarget.motion_action must be callable")
        if self.nominal_eef_path is not None and not isinstance(self.nominal_eef_path, Path):
            raise ValueError("ConformanceTarget.nominal_eef_path must be a hazardlib Path or None")
        require_methods(self.host)
        require_methods(self.host, DRIVER_METHODS)


# ------------------------------------------------------------------------------------------------ registry

_REGISTRY = {}  # type: Dict[str, Callable[[], ConformanceTarget]]


def register_host(name: str, factory: Callable[[], ConformanceTarget]) -> None:
    """Register ``factory`` (no arguments, returns a :class:`ConformanceTarget`) under ``name``. Registering a
    different factory under a taken name is an error; re-registering the same one is a no-op (module reloads)."""
    if not isinstance(name, str) or not name:
        raise ValueError("host name must be a non-empty string, got %r" % (name,))
    if not callable(factory):
        raise ValueError("factory for %r must be callable" % name)
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not factory:
        raise ValueError("host %r is already registered by %r" % (name, existing))
    _REGISTRY[name] = factory


def registered_hosts() -> List[str]:
    return sorted(_REGISTRY)


def resolve_target(name: str, modules: Sequence[str] = ()) -> ConformanceTarget:
    """Import ``modules`` (each may register hosts), then, if ``name`` is still unknown, ``hazardhosts.<name>``;
    call the registered factory. An unknown name is a :class:`CapabilityError` listing the known ones."""
    for module in modules:
        importlib.import_module(module)
    if name not in _REGISTRY:
        try:
            importlib.import_module("hazardhosts." + name)
        except ImportError as exc:
            raise CapabilityError("no conformance host %r registered (known: %s); importing hazardhosts.%s "
                                  "failed: %s" % (name, ", ".join(registered_hosts()), name, exc)) from None
    if name not in _REGISTRY:
        raise CapabilityError("no conformance host %r registered (known: %s); hazardhosts.%s did not call "
                              "register_host" % (name, ", ".join(registered_hosts()), name))
    target = _REGISTRY[name]()
    if not isinstance(target, ConformanceTarget):
        raise CapabilityError("the factory of host %r returned %r, not a ConformanceTarget"
                              % (name, type(target).__name__))
    return target
