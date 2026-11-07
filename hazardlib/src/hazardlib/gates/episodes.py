"""Gates G1 / G2 and the episode diagnostics, driven through ``EpisodeDriver`` (paradigm §8; D5, D11 a).

* **Baseline** (G1 precondition): the reference policy alone, ``g1.repeats`` hazard-free control episodes;
  usable iff the success fraction is >= ``g1.baseline_success_required``.
* **G1** (``run_g1``): the reference policy wrapped in the Oracle stop-wait-resume controller
  (:class:`OracleController`), ``g1.repeats`` episodes; passed iff every episode has at most
  ``g1.max_official_contacts`` official contact events and, if required, succeeds.
* **G2** (``run_g2``): the reference policy's action filtered by a :class:`PrivilegedAvoider`, which sees the
  ground-truth hazard state; same criterion with the ``g2`` settings. Without an implementation G2 is not
  evaluated and the label becomes ``uncertified`` (no avoidance algorithm is provided here).
* **Nominal hit** (``run_nominal_hit``): the reference policy without any reaction; records whether and when
  the first official contact happened relative to ``t_hit`` and the visible lead ``tau_vis = t_hit -
  t_first_visible`` from ``render_check`` (``None`` with a note when the host has none; record only, D5).

Episodes last ``max(nominal success time, t_hit) + horizon_extra_s`` seconds; the k-th repetition uses the
environment seed ``seed + k``. Success means ``task_success`` was true at some control step.
"""
import dataclasses
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from ..driving import EpisodeResult, EpisodeSession
from ..runtime import KINEMATIC, RELEASED
from ..spec.instance import HazardInstance
from ..spec.task_context import TaskContext
from .settings import EpisodeGateSettings, GateSettings


@runtime_checkable
class PrivilegedAvoider(Protocol):
    """G2 controller with ground-truth hazard state (00 §8). ``hazard_state`` is ``None`` while no hazard is
    moving. Returns the benchmark-native action to execute instead of ``action``."""

    def filter_action(self, action: Any, robot_state: Any, hazard_state: Any) -> Any:
        ...


def nominal_success_time_s(task_context: TaskContext) -> float:
    return float(task_context.phases[-1].t_end_s)


def _horizon(task_context: TaskContext, instance: Optional[HazardInstance], extra_s: float) -> float:
    end = nominal_success_time_s(task_context)
    if instance is not None:
        end = max(end, instance.nominal_hit_time_s)
    return end + extra_s


class OracleController:
    """Stop-wait-resume: hold (``hold_action``, the policy is not called) while the hazard is kinematic, or
    released and still moving inside the workspace; resume once it has been slower than ``resume_speed_mps``
    or outside the workspace for ``settle_s``. The hazard is read through the runtime's privileged accessor
    ``hazard_state(env)`` (D13 d)."""

    def __init__(self, resume_speed_mps: float, settle_s: float, workspace_lo: Any, workspace_hi: Any) -> None:
        self.resume_speed_mps = resume_speed_mps
        self.settle_s = settle_s
        self.lo = np.array(workspace_lo, dtype=np.float64)
        self.hi = np.array(workspace_hi, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.quiet_s = 0.0
        self.paused_s = 0.0
        self.pauses = 0
        self._holding = False

    def _moving(self, session: EpisodeSession, env: int) -> bool:
        state = session.runtime.hazard_state(env)
        inside = bool(np.all(np.logical_and(state.pos >= self.lo, state.pos <= self.hi)))
        return inside and float(np.linalg.norm(state.lin_vel_mps)) >= self.resume_speed_mps

    def decide(self, session: EpisodeSession, env: int, t_s: float, observation: Dict[str, Any], policy: Any,
               control_dt_s: float) -> Any:
        phase = session.phase(env)
        hold = False
        if phase == KINEMATIC:
            self.quiet_s, hold = 0.0, True
        elif phase == RELEASED:
            if self._moving(session, env):
                self.quiet_s, hold = 0.0, True
            else:
                self.quiet_s += control_dt_s
                hold = self.quiet_s < self.settle_s
        if hold:
            if not self._holding:
                self.pauses += 1
            self._holding = True
            self.paused_s += control_dt_s
            return session.host.hold_action(env)
        self._holding = False
        return policy.act(observation)

    def summary(self) -> Dict[str, Any]:
        return {"controller": "oracle_stop_wait_resume", "paused_s": self.paused_s, "pauses": self.pauses}


class AvoiderController:
    def __init__(self, avoider: Any) -> None:
        self.avoider = avoider

    def reset(self) -> None:
        pass

    def decide(self, session: EpisodeSession, env: int, t_s: float, observation: Dict[str, Any], policy: Any,
               control_dt_s: float) -> Any:
        action = policy.act(observation)
        hazard = session.runtime.hazard_state(env) if session.phase(env) in (KINEMATIC, RELEASED) else None
        return self.avoider.filter_action(action, session.host.robot_state(env), hazard)

    def summary(self) -> Dict[str, Any]:
        return {"controller": "privileged_avoider", "avoider": type(self.avoider).__name__}


def _episode_summary(result: EpisodeResult, seed: int) -> Dict[str, Any]:
    return {"seed": seed, "success": result.success, "success_t_s": result.success_t_s,
            "official_contacts": result.official_contact_count,
            "first_official_contact_t_s": result.log.first_official_contact_t_s,
            "setup_failed": bool(result.info.get("setup_failed")), "controller": result.controller}


def _passes(result: EpisodeResult, settings: EpisodeGateSettings) -> bool:
    if result.official_contact_count > settings.max_official_contacts:
        return False
    return result.success or not settings.require_task_success


@dataclasses.dataclass
class BaselineResult:
    runs: int
    successes: int
    success_fraction: float
    usable: bool
    episodes: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def run_baseline(session: EpisodeSession, policy: Any, task_context: TaskContext, settings: GateSettings,
                 seed: int, env: int) -> BaselineResult:
    horizon = _horizon(task_context, None, settings.g1.horizon_extra_s)
    episodes = []
    for k in range(settings.g1.repeats):
        control_id = "baseline:%s:%s:%d" % (task_context.task_id, task_context.layout_id, seed + k)
        result = session.run_episode(env, None, policy, task_context.task_id, task_context.layout_id, seed + k,
                                     horizon, control_id=control_id, stop_on_success=True)
        episodes.append(_episode_summary(result, seed + k))
    successes = sum(1 for e in episodes if e["success"])
    fraction = successes / float(len(episodes))
    return BaselineResult(runs=len(episodes), successes=successes, success_fraction=fraction,
                          usable=fraction >= settings.baseline_success_required, episodes=episodes)


@dataclasses.dataclass
class GateRun:
    """Outcome of G1 or G2. ``passed`` is ``None`` when the gate was not evaluated."""
    passed: Optional[bool]
    episodes: List[Dict[str, Any]]
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _instance_task(instance: HazardInstance, task_context: TaskContext) -> None:
    prov = instance.provenance
    if (prov.get("task_id"), prov.get("layout_id")) != (task_context.task_id, task_context.layout_id):
        raise ValueError("instance %r belongs to (%r, %r), not to the TaskContext (%r, %r)"
                         % (instance.instance_id, prov.get("task_id"), prov.get("layout_id"),
                            task_context.task_id, task_context.layout_id))


def _run_gate(session: EpisodeSession, instance: HazardInstance, policy: Any, task_context: TaskContext,
              episode_settings: EpisodeGateSettings, controller: Any, seed: int, env: int) -> GateRun:
    _instance_task(instance, task_context)
    horizon = _horizon(task_context, instance, episode_settings.horizon_extra_s)
    episodes, passed = [], True
    for k in range(episode_settings.repeats):
        result = session.run_episode(env, instance, policy, task_context.task_id, task_context.layout_id, seed + k,
                                     horizon, controller=controller)
        episodes.append(_episode_summary(result, seed + k))
        passed = passed and _passes(result, episode_settings)
    return GateRun(passed=passed, episodes=episodes, note="")


def run_g1(session: EpisodeSession, instance: HazardInstance, reference_policy: Any, task_context: TaskContext,
           settings: GateSettings, seed: int, env: int) -> GateRun:
    """Oracle stop-wait-resume episodes. The baseline is checked separately (:func:`run_baseline`)."""
    ws = task_context.workspace_bounds
    margin = settings.g0.workspace_margin_m
    oracle = OracleController(settings.resume_hazard_speed_mps, settings.resume_settle_s,
                              np.array(ws.min_m) - margin, np.array(ws.max_m) + margin)
    return _run_gate(session, instance, reference_policy, task_context, settings.g1, oracle, seed, env)


def run_g2(session: EpisodeSession, instance: HazardInstance, reference_policy: Any, avoider: Optional[Any],
           task_context: TaskContext, settings: GateSettings, seed: int, env: int) -> GateRun:
    if avoider is None:
        return GateRun(passed=None, episodes=[], note="no PrivilegedAvoider implementation: G2 not evaluated")
    if not callable(getattr(avoider, "filter_action", None)):
        raise TypeError("avoider must implement filter_action(action, robot_state, hazard_state)")
    return _run_gate(session, instance, reference_policy, task_context, settings.g2, AvoiderController(avoider),
                     seed, env)


def run_nominal_hit(session: EpisodeSession, instance: HazardInstance, reference_policy: Any,
                    task_context: TaskContext, settings: GateSettings, seed: int, env: int) -> Dict[str, Any]:
    _instance_task(instance, task_context)
    horizon = _horizon(task_context, instance, settings.g1.horizon_extra_s)
    result = session.run_episode(env, instance, reference_policy, task_context.task_id, task_context.layout_id,
                                 seed, horizon, measure_visibility=True)
    first = result.log.first_official_contact_t_s
    t_hit = instance.nominal_hit_time_s
    if not result.render_check_available:
        tau_vis, note = None, "tau_vis not measured: the host has no render_check"
    elif result.first_visible_t_s is None or result.first_visible_t_s > t_hit:
        tau_vis, note = 0.0, "hazard not visible to render_check before t_hit"
    else:
        tau_vis, note = t_hit - result.first_visible_t_s, "render_check at the control rate"
    return {"seed": seed, "contact": first is not None, "first_official_contact_t_s": first,
            "hit_time_error_s": None if first is None else first - t_hit, "success": result.success,
            "first_visible_t_s": result.first_visible_t_s, "tau_vis_s": tau_vis, "tau_vis_note": note}
