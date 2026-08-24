"""LIBERO rollout hooks for pause-VLA Oracle (explicit YAML enable)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dynisafety.backends.libero.safety.compat import import_safety_api
from dynisafety.backends.libero.safety.observation import (
    LEVEL_SEMANTICS,
    build_safety_observation,
    settle_steps_for_level,
)
from dynisafety.backends.libero.safety.phases import LiberoOraclePhase, probe_controller_phase
from dynisafety.config.schema import RunConfig
from dynisafety.policies.base import Observation, Policy

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_ORACLE_YAML = _REPO_ROOT / "configs" / "safety" / "oracle_stop_wait_resume.yaml"
_LIBERO_ORACLE_YAML = (
    _REPO_ROOT / "configs" / "backends" / "libero" / "oracle" / "stop_wait_resume.yaml"
)


def oracle_enabled_from_extra(
    extra_libero: Mapping[str, Any] | None,
    *,
    level: int,
) -> bool:
    """L0 always bypass. Otherwise require explicit ``extra.libero.oracle.enabled``."""
    if level <= 0:
        return False
    if not extra_libero:
        return False
    block = extra_libero.get("oracle")
    if not isinstance(block, Mapping):
        return False
    return bool(block.get("enabled", False))


def _god_view_robot_paused(observation: Observation) -> bool:
    """True while the oracle is holding the arm (active disturbance, not complete).

    Missing SafetyObservation is fail-closed hold, same as ``should_clear_action_queue``.
    """
    extra = observation.extra if isinstance(observation.extra, dict) else None
    safety = extra.get("safety") if extra is not None else None
    if safety is None:
        return True
    active = bool(getattr(safety, "disturbance_active", False))
    complete = bool(getattr(safety, "motion_complete", False))
    return active and not complete


def _oracle_config_path(extra_libero: Mapping[str, Any]) -> Path | None:
    block = extra_libero.get("oracle")
    if not isinstance(block, Mapping):
        return None
    raw = block.get("config_path")
    if raw is None or raw == "":
        if _LIBERO_ORACLE_YAML.is_file():
            return _LIBERO_ORACLE_YAML
        if _DEFAULT_ORACLE_YAML.is_file():
            return _DEFAULT_ORACLE_YAML
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def maybe_wrap_policy(
    policy: Policy,
    *,
    action_space: str,
    enabled: bool,
    extra_libero: Mapping[str, Any],
) -> Policy:
    """Wrap *policy* when oracle explicitly enabled; else return unchanged."""
    if not enabled:
        return policy
    api = import_safety_api()
    wrap_policy = api["wrap_policy"]
    load_oracle_config = api["load_oracle_config"]
    if wrap_policy is None or load_oracle_config is None:
        raise RuntimeError(
            "LIBERO oracle enabled but dynisafety.policies.safety is unavailable. "
            "Merge Prompt 10 (policies/safety) or disable extra.libero.oracle.enabled."
        )
    cfg_path = _oracle_config_path(extra_libero)
    cfg = load_oracle_config(cfg_path)
    # Force enabled=True regardless of shared YAML default (explicit libero switch).
    return wrap_policy(
        policy,
        action_space=action_space,
        config=cfg,
        enabled=True,
    )


@dataclass
class LiberoOracleSession:
    """Per-episode bookkeeping for oracle phase injection + EpisodeRecord.extra."""

    enabled: bool
    level: int
    settle_steps: int
    policy: Policy
    state_seq: list[str] = field(default_factory=list)
    libero_phase_seq: list[str] = field(default_factory=list)
    resume_occurred: bool = False
    _was_pausing: bool = False

    @classmethod
    def create(
        cls,
        *,
        policy: Policy,
        cfg: RunConfig,
        extra_libero: Mapping[str, Any],
        action_space: str,
    ) -> LiberoOracleSession:
        level = int(cfg.level)
        enabled = oracle_enabled_from_extra(extra_libero, level=level)
        wrapped = maybe_wrap_policy(
            policy,
            action_space=action_space,
            enabled=enabled,
            extra_libero=extra_libero,
        )
        return cls(
            enabled=enabled,
            level=level,
            settle_steps=settle_steps_for_level(cfg, level),
            policy=wrapped,
        )

    @property
    def active_policy(self) -> Policy:
        return self.policy

    def inject_observation(
        self,
        observation: Observation,
        *,
        controller: Any | None,
        task_step: int,
    ) -> Observation:
        """Attach SafetyObservation when enabled; L0 / disabled leave obs unchanged."""
        if not self.enabled:
            return observation
        probe = probe_controller_phase(
            controller,
            level=self.level,
            task_step=task_step,
            settle_steps=self.settle_steps,
        )
        self.libero_phase_seq.append(probe.phase.value)
        safety = build_safety_observation(
            controller=controller,
            level=self.level,
            task_step=task_step,
            settle_steps=self.settle_steps,
            probe=probe,
        )
        extra = dict(observation.extra) if isinstance(observation.extra, dict) else {}
        extra["safety"] = safety
        observation.extra = extra
        return observation

    def should_clear_action_queue(self, observation: Observation) -> bool:
        """Clear open-loop VLA chunks when entering / during pause."""
        if not self.enabled:
            return False
        pausing = _god_view_robot_paused(observation)
        clear = pausing or self._was_pausing
        self._was_pausing = pausing
        return clear

    def exempts_vla_budget(self, observation: Observation) -> bool:
        """L2/L3 god-view hold does not consume ``runtime.max_steps``.

        Controller / physics clocks still advance. L0, L1, and oracle-off
        keep the old accounting (every task step counts).
        """
        if not self.enabled or self.level not in (2, 3):
            return False
        return _god_view_robot_paused(observation)

    def note_after_act(self) -> None:
        if not self.enabled:
            return
        oracle = getattr(self.policy, "oracle", None)
        if oracle is None:
            return
        result = getattr(oracle, "last_result", None)
        if result is None:
            return
        state_after = getattr(result, "state_after", None)
        if state_after is not None:
            value = state_after.value if hasattr(state_after, "value") else str(state_after)
            self.state_seq.append(str(value))
        itype = getattr(result, "intervention_type", None)
        ivalue = itype.value if hasattr(itype, "value") else str(itype)
        if ivalue == "resume" or (
            state_after is not None and (getattr(state_after, "value", state_after) == "resume")
        ):
            self.resume_occurred = True

    def stats(self) -> dict[str, int]:
        if not self.enabled:
            return {"policy_infer_steps": 0, "oracle_pause_steps": 0}
        fn = getattr(self.policy, "stats_dict", None)
        if callable(fn):
            return dict(fn())
        oracle = getattr(self.policy, "oracle", None)
        if oracle is not None and hasattr(oracle, "stats"):
            return oracle.stats.as_dict()
        return {"policy_infer_steps": 0, "oracle_pause_steps": 0}

    def episode_extra(self) -> dict[str, Any]:
        """Fields for ``EpisodeRecord.extra`` (schema stays on frozen EpisodeRecord)."""
        stats = self.stats()
        summary = _summarize_states(self.state_seq)
        out: dict[str, Any] = {
            "oracle_enabled": bool(self.enabled),
            "oracle_level_semantics": LEVEL_SEMANTICS.get(self.level, ""),
            "oracle_state_seq": list(self.state_seq),
            "oracle_state_summary": summary,
            "libero_oracle_phase_seq": list(self.libero_phase_seq),
            "oracle_pause_steps": int(stats.get("oracle_pause_steps", 0)),
            "policy_infer_steps": int(stats.get("policy_infer_steps", 0)),
            "oracle_resume_occurred": bool(self.resume_occurred),
        }
        if self.level == 2:
            out["oracle_note"] = (
                "L2 is Arc-Transit object intrusion — not a human hand "
                f"(hand semantics belong to L3). phase={LiberoOraclePhase.SETTLED_OR_DONE.value} "
                "means static obstacle remains after settle."
            )
        return out


def _summarize_states(states: list[str]) -> str:
    if not states:
        return ""
    parts: list[str] = []
    prev = states[0]
    count = 1
    for s in states[1:]:
        if s == prev:
            count += 1
        else:
            parts.append(f"{prev}×{count}")
            prev = s
            count = 1
    parts.append(f"{prev}×{count}")
    return "→".join(parts)
