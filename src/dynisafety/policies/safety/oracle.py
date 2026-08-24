"""Oracle Stop-Wait-Resume: privileged pause of VLA inference (Prompt 10).

Product semantics
-----------------
* Input is god-view ``SafetyObservation`` (phase / active / motion_complete), not RGB.
* While disturbance is active, **do not** call the underlying ``Policy.act()``.
* Emit hold / zero-motion actions and preserve gripper.
* Pause control steps increment ``oracle_pause_steps`` only; VLA calls increment
  ``policy_infer_steps`` only.
* When motion ends (L1/L2 settle done or L3 withdraw done) → RESUME.
* State machine: PASS_THROUGH → STOP → WAIT → RESUME (no RETREAT).
* Missing observation while enabled → fail-closed hold (never silent pass-through).
* ``enabled=false`` (YAML default) → bit-identical to the wrapped Policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dynisafety.policies.base import ActionChunk, EpisodeContext, ModelConfig, Observation, Policy
from dynisafety.policies.safety.action_adapter import HoldActionFactory, ZeroMotionHoldFactory
from dynisafety.policies.safety.config import OracleConfig, load_oracle_config
from dynisafety.policies.safety.types import (
    InterventionType,
    OracleMode,
    OracleState,
    OracleStepStats,
    SafetyInterventionResult,
    SafetyObservation,
)


def _phase_norm(phase: str) -> str:
    return phase.strip().lower()


def should_pause(obs: SafetyObservation, cfg: OracleConfig) -> bool:
    """Whether the oracle should keep VLA inference paused.

    Primary signal: ``disturbance_active`` and not ``motion_complete``.
    Phase sets and optional geometry thresholds reinforce the decision.
    """
    if obs.motion_complete:
        return False
    if obs.disturbance_active:
        return True
    phase = _phase_norm(obs.disturbance_phase)
    if phase in cfg.active_phases:
        return True
    if (
        obs.min_distance_m is not None
        and obs.min_distance_m <= cfg.distance_stop_m
    ):
        return True
    if (
        obs.closing_speed_mps is not None
        and obs.closing_speed_mps >= cfg.closing_speed_stop_mps
        and obs.min_distance_m is not None
        and obs.min_distance_m <= cfg.distance_resume_m
    ):
        return True
    return False


def should_resume(obs: SafetyObservation, cfg: OracleConfig) -> bool:
    """Resume when settle / withdraw finished (god-view)."""
    if obs.motion_complete:
        return True
    if obs.disturbance_active:
        return False
    phase = _phase_norm(obs.disturbance_phase)
    if phase in cfg.resume_phases:
        return True
    if (
        obs.min_distance_m is not None
        and obs.min_distance_m >= cfg.distance_resume_m
        and phase not in cfg.active_phases
    ):
        return True
    return False


def extract_safety_observation(
    obs: Observation,
    *,
    extra_key: str,
) -> SafetyObservation | None:
    """Pull ``SafetyObservation`` from ``Observation.extra`` if present."""
    raw = obs.extra.get(extra_key) if isinstance(obs.extra, dict) else None
    if raw is None:
        return None
    if isinstance(raw, SafetyObservation):
        return raw
    if isinstance(raw, dict):
        return SafetyObservation(
            step=int(raw["step"]) if "step" in raw else int(obs.step),
            disturbance_phase=str(raw["disturbance_phase"]),
            disturbance_active=bool(raw["disturbance_active"]),
            motion_complete=bool(raw["motion_complete"]),
            disturbance_visible=bool(raw.get("disturbance_visible", False)),
            min_distance_m=(
                float(raw["min_distance_m"])
                if raw.get("min_distance_m") is not None
                else None
            ),
            closing_speed_mps=(
                float(raw["closing_speed_mps"])
                if raw.get("closing_speed_mps") is not None
                else None
            ),
            timestamp_s=(
                float(raw["timestamp_s"]) if raw.get("timestamp_s") is not None else None
            ),
            level=int(raw["level"]) if raw.get("level") is not None else None,
            source=str(raw.get("source", "god_view")),
            extra=dict(raw.get("extra", {})),
        )
    raise TypeError(
        f"Observation.extra[{extra_key!r}] must be SafetyObservation or dict, "
        f"got {type(raw)!r}"
    )


@dataclass
class OracleStopWaitResume:
    """Pause-VLA state machine. Does not wrap a Policy by itself."""

    config: OracleConfig
    hold_factory: HoldActionFactory
    state: OracleState = OracleState.PASS_THROUGH
    stats: OracleStepStats = field(default_factory=OracleStepStats)
    last_nominal: ActionChunk | None = None
    last_result: SafetyInterventionResult | None = None
    _action_space: str | None = None

    @classmethod
    def from_config(
        cls,
        config: OracleConfig,
        *,
        hold_factory: HoldActionFactory | None = None,
    ) -> OracleStopWaitResume:
        factory = hold_factory or ZeroMotionHoldFactory(config.stop_action)
        return cls(config=config, hold_factory=factory)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path | None = None,
        *,
        hold_factory: HoldActionFactory | None = None,
        enabled_override: bool | None = None,
    ) -> OracleStopWaitResume:
        cfg = load_oracle_config(path)
        if enabled_override is not None:
            cfg = OracleConfig(
                enabled=bool(enabled_override),
                mode=cfg.mode,
                observation_extra_key=cfg.observation_extra_key,
                fail_closed_on_missing_observation=cfg.fail_closed_on_missing_observation,
                active_phases=cfg.active_phases,
                resume_phases=cfg.resume_phases,
                distance_stop_m=cfg.distance_stop_m,
                distance_resume_m=cfg.distance_resume_m,
                closing_speed_stop_mps=cfg.closing_speed_stop_mps,
                stop_action=cfg.stop_action,
                accounting=cfg.accounting,
            )
        return cls.from_config(cfg, hold_factory=hold_factory)

    def reset(self, *, action_space: str | None = None) -> None:
        self.state = OracleState.PASS_THROUGH
        self.stats = OracleStepStats()
        self.last_nominal = None
        self.last_result = None
        if action_space is not None:
            self._action_space = action_space

    def set_action_space(self, space: str) -> None:
        self._action_space = space

    def _hold(self, obs: Observation) -> ActionChunk:
        if self._action_space is None:
            raise RuntimeError(
                "OracleStopWaitResume action_space unset; call reset(action_space=...) "
                "or set_action_space() before intervene()"
            )
        return self.hold_factory.hold_action(
            space=self._action_space,
            last_action=self.last_nominal,
            obs=obs,
        )

    def _record(
        self,
        *,
        executed: ActionChunk,
        nominal: ActionChunk | None,
        intervention_type: InterventionType,
        reason: str,
        state_before: OracleState,
        state_after: OracleState,
        policy_inferred: bool,
        oracle_paused: bool,
    ) -> SafetyInterventionResult:
        if policy_inferred:
            self.stats.policy_infer_steps += 1
        if oracle_paused:
            self.stats.oracle_pause_steps += 1
        result = SafetyInterventionResult(
            executed_action=executed,
            nominal_action=nominal,
            intervention_type=intervention_type,
            reason=reason,
            state_before=state_before,
            state_after=state_after,
            thresholds=self.config.thresholds_dict(),
            policy_inferred=policy_inferred,
            oracle_paused=oracle_paused,
            policy_infer_steps=self.stats.policy_infer_steps,
            oracle_pause_steps=self.stats.oracle_pause_steps,
        )
        self.last_result = result
        self.state = state_after
        return result

    def intervene(
        self,
        obs: Observation,
        *,
        safety_obs: SafetyObservation | None,
        policy_act: Any,
    ) -> SafetyInterventionResult:
        """Decide one step.

        ``policy_act`` is a zero-arg callable that invokes the underlying
        ``Policy.act(obs)``. It must **not** be called on pause / fail-closed steps.
        """
        cfg = self.config
        before = self.state

        # Disabled: always infer; bit-identical path (caller still wraps).
        if not cfg.enabled:
            nominal = policy_act()
            self.last_nominal = nominal
            return self._record(
                executed=nominal,
                nominal=nominal,
                intervention_type=InterventionType.NONE,
                reason="oracle_disabled",
                state_before=before,
                state_after=OracleState.PASS_THROUGH,
                policy_inferred=True,
                oracle_paused=False,
            )

        # Fail-closed: missing god-view observation → hold, never pass-through.
        if safety_obs is None:
            if not cfg.fail_closed_on_missing_observation:
                raise RuntimeError(
                    "missing SafetyObservation and fail_closed_on_missing_observation=false "
                    "is unsupported (silent pass-through is forbidden)"
                )
            hold = self._hold(obs)
            after = OracleState.WAIT if before != OracleState.PASS_THROUGH else OracleState.STOP
            return self._record(
                executed=hold,
                nominal=None,
                intervention_type=InterventionType.FAIL_CLOSED,
                reason="missing_safety_observation",
                state_before=before,
                state_after=after,
                policy_inferred=False,
                oracle_paused=True,
            )

        # State transitions.
        if before == OracleState.PASS_THROUGH:
            if should_pause(safety_obs, cfg):
                hold = self._hold(obs)
                return self._record(
                    executed=hold,
                    nominal=None,
                    intervention_type=InterventionType.STOP,
                    reason=f"disturbance_active phase={safety_obs.disturbance_phase}",
                    state_before=before,
                    state_after=OracleState.STOP,
                    policy_inferred=False,
                    oracle_paused=True,
                )
            nominal = policy_act()
            self.last_nominal = nominal
            return self._record(
                executed=nominal,
                nominal=nominal,
                intervention_type=InterventionType.NONE,
                reason="pass_through",
                state_before=before,
                state_after=OracleState.PASS_THROUGH,
                policy_inferred=True,
                oracle_paused=False,
            )

        if before == OracleState.STOP:
            # First pause tick already emitted STOP; subsequent ticks are WAIT.
            hold = self._hold(obs)
            if cfg.mode == OracleMode.STOP_ONLY:
                return self._record(
                    executed=hold,
                    nominal=None,
                    intervention_type=InterventionType.WAIT,
                    reason="stop_only_hold",
                    state_before=before,
                    state_after=OracleState.WAIT,
                    policy_inferred=False,
                    oracle_paused=True,
                )
            if should_resume(safety_obs, cfg):
                # Extremely short hazard: STOP then immediate resume next decide.
                # Still do not infer on this transition tick; RESUME→infer next.
                return self._record(
                    executed=hold,
                    nominal=None,
                    intervention_type=InterventionType.RESUME,
                    reason=f"motion_complete phase={safety_obs.disturbance_phase}",
                    state_before=before,
                    state_after=OracleState.RESUME,
                    policy_inferred=False,
                    oracle_paused=True,
                )
            return self._record(
                executed=hold,
                nominal=None,
                intervention_type=InterventionType.WAIT,
                reason="waiting_for_motion_complete",
                state_before=before,
                state_after=OracleState.WAIT,
                policy_inferred=False,
                oracle_paused=True,
            )

        if before == OracleState.WAIT:
            hold = self._hold(obs)
            if cfg.mode == OracleMode.STOP_ONLY:
                return self._record(
                    executed=hold,
                    nominal=None,
                    intervention_type=InterventionType.WAIT,
                    reason="stop_only_hold",
                    state_before=before,
                    state_after=OracleState.WAIT,
                    policy_inferred=False,
                    oracle_paused=True,
                )
            if should_resume(safety_obs, cfg):
                return self._record(
                    executed=hold,
                    nominal=None,
                    intervention_type=InterventionType.RESUME,
                    reason=f"motion_complete phase={safety_obs.disturbance_phase}",
                    state_before=before,
                    state_after=OracleState.RESUME,
                    policy_inferred=False,
                    oracle_paused=True,
                )
            if should_pause(safety_obs, cfg):
                return self._record(
                    executed=hold,
                    nominal=None,
                    intervention_type=InterventionType.WAIT,
                    reason="still_active",
                    state_before=before,
                    state_after=OracleState.WAIT,
                    policy_inferred=False,
                    oracle_paused=True,
                )
            # Not active and not explicitly resume — stay fail-safe in WAIT.
            return self._record(
                executed=hold,
                nominal=None,
                intervention_type=InterventionType.WAIT,
                reason="wait_until_resume_signal",
                state_before=before,
                state_after=OracleState.WAIT,
                policy_inferred=False,
                oracle_paused=True,
            )

        if before == OracleState.RESUME:
            # One-shot: restore VLA inference this step.
            if should_pause(safety_obs, cfg):
                hold = self._hold(obs)
                return self._record(
                    executed=hold,
                    nominal=None,
                    intervention_type=InterventionType.STOP,
                    reason="reentered_active_during_resume",
                    state_before=before,
                    state_after=OracleState.STOP,
                    policy_inferred=False,
                    oracle_paused=True,
                )
            nominal = policy_act()
            self.last_nominal = nominal
            return self._record(
                executed=nominal,
                nominal=nominal,
                intervention_type=InterventionType.RESUME,
                reason="resumed_policy_inference",
                state_before=before,
                state_after=OracleState.PASS_THROUGH,
                policy_inferred=True,
                oracle_paused=False,
            )

        raise RuntimeError(f"unknown oracle state {before!r}")


class OraclePolicyWrapper:
    """Policy wrapper that pauses underlying ``act`` during disturbance.

    When ``config.enabled`` is false, ``act`` is a direct pass-through
    (bit-identical to the wrapped policy for the same Observation).
    """

    def __init__(
        self,
        policy: Policy,
        oracle: OracleStopWaitResume,
        *,
        action_space: str,
    ) -> None:
        self._policy = policy
        self._oracle = oracle
        self._action_space = action_space
        self._oracle.set_action_space(action_space)

    @property
    def wrapped(self) -> Policy:
        return self._policy

    @property
    def oracle(self) -> OracleStopWaitResume:
        return self._oracle

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Policy:
        raise TypeError(
            "OraclePolicyWrapper cannot be constructed via from_config; "
            "wrap an existing Policy with wrap_policy()"
        )

    def reset(self, ctx: EpisodeContext) -> None:
        self._policy.reset(ctx)
        self._oracle.reset(action_space=self._action_space)

    def act(self, obs: Observation) -> ActionChunk:
        if not self._oracle.config.enabled:
            # Bit-identical path: no safety extraction, no extra branching side effects.
            return self._policy.act(obs)

        safety = extract_safety_observation(
            obs, extra_key=self._oracle.config.observation_extra_key
        )

        def _call_policy() -> ActionChunk:
            return self._policy.act(obs)

        result = self._oracle.intervene(obs, safety_obs=safety, policy_act=_call_policy)
        return result.executed_action

    def close(self) -> None:
        self._policy.close()

    def stats_dict(self) -> dict[str, int]:
        return self._oracle.stats.as_dict()


def wrap_policy(
    policy: Policy,
    *,
    action_space: str,
    config: OracleConfig | None = None,
    config_path: str | Path | None = None,
    enabled: bool | None = None,
    hold_factory: HoldActionFactory | None = None,
) -> OraclePolicyWrapper:
    """Wrap *policy* with the pause-VLA oracle.

    Defaults load ``configs/safety/oracle_stop_wait_resume.yaml`` (``enabled: false``).
    """
    if config is not None:
        oracle = OracleStopWaitResume.from_config(config, hold_factory=hold_factory)
        if enabled is not None and enabled != config.enabled:
            oracle = OracleStopWaitResume.from_config(
                OracleConfig(
                    enabled=bool(enabled),
                    mode=config.mode,
                    observation_extra_key=config.observation_extra_key,
                    fail_closed_on_missing_observation=config.fail_closed_on_missing_observation,
                    active_phases=config.active_phases,
                    resume_phases=config.resume_phases,
                    distance_stop_m=config.distance_stop_m,
                    distance_resume_m=config.distance_resume_m,
                    closing_speed_stop_mps=config.closing_speed_stop_mps,
                    stop_action=config.stop_action,
                    accounting=config.accounting,
                ),
                hold_factory=hold_factory,
            )
    else:
        oracle = OracleStopWaitResume.from_yaml(
            config_path, hold_factory=hold_factory, enabled_override=enabled
        )
    return OraclePolicyWrapper(policy, oracle, action_space=action_space)
