"""Unit tests for pause-VLA Oracle Stop-Wait-Resume (Prompt 10, CPU_ONLY)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from dynisafety.policies.base import ActionChunk, EpisodeContext, Observation
from dynisafety.policies.safety import (
    DEFAULT_ORACLE_CONFIG_PATH,
    InterventionType,
    OracleMode,
    OraclePolicyWrapper,
    OracleState,
    OracleStopWaitResume,
    SafetyObservation,
    load_oracle_config,
    wrap_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "configs" / "safety" / "oracle_stop_wait_resume.yaml"


def _obs(step: int = 0, *, safety: SafetyObservation | dict | None = None) -> Observation:
    extra: dict[str, Any] = {}
    if safety is not None:
        extra["safety"] = safety
    return Observation(
        images={"primary": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=None,
        instruction="test",
        step=step,
        extra=extra,
    )


def _safety(
    *,
    step: int = 0,
    phase: str = "idle",
    active: bool = False,
    complete: bool = False,
    visible: bool = False,
    dist: float | None = None,
    speed: float | None = None,
) -> SafetyObservation:
    return SafetyObservation(
        step=step,
        disturbance_phase=phase,
        disturbance_active=active,
        motion_complete=complete,
        disturbance_visible=visible,
        min_distance_m=dist,
        closing_speed_mps=speed,
    )


def _ctx() -> EpisodeContext:
    return EpisodeContext(
        instruction="test",
        task_id=0,
        task_name="t",
        episode_index=0,
        seed=0,
        level=1,
        backend="libero",
        suite="s",
        extra={},
    )


def _mock_policy(*, action: np.ndarray | None = None) -> MagicMock:
    policy = MagicMock()
    vec = (
        np.asarray(action, dtype=np.float64)
        if action is not None
        else np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.9], dtype=np.float64)
    )
    chunk = ActionChunk(actions=vec.reshape(1, -1), space="delta_ee_7d")
    policy.act.return_value = chunk
    return policy


def _enabled_oracle(*, mode: str = "stop_wait_resume") -> OracleStopWaitResume:
    # Reconstruct with enabled=True (YAML default is false).
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    raw["oracle"]["enabled"] = True
    raw["oracle"]["mode"] = mode
    from dynisafety.policies.safety.config import oracle_config_from_mapping

    return OracleStopWaitResume.from_config(oracle_config_from_mapping(raw))


# --- config -----------------------------------------------------------------


def test_default_yaml_exists_and_disabled() -> None:
    assert DEFAULT_ORACLE_CONFIG_PATH.is_file() or YAML_PATH.is_file()
    cfg = load_oracle_config(YAML_PATH)
    assert cfg.enabled is False
    assert cfg.mode is OracleMode.STOP_WAIT_RESUME
    assert cfg.fail_closed_on_missing_observation is True
    assert cfg.distance_stop_m > 0
    assert "enter" in cfg.active_phases
    assert "done" in cfg.resume_phases


def test_yaml_has_no_python_side_defaults_for_thresholds() -> None:
    """Iron law 1: thresholds must appear in YAML."""
    text = YAML_PATH.read_text(encoding="utf-8")
    for key in (
        "distance_stop_m",
        "distance_resume_m",
        "closing_speed_stop_mps",
        "default_gripper_hold",
        "active_phases",
        "resume_phases",
    ):
        assert key in text


# --- state transitions ------------------------------------------------------


def test_pass_through_calls_policy_when_idle() -> None:
    oracle = _enabled_oracle()
    oracle.reset(action_space="delta_ee_7d")
    calls: list[int] = []

    def policy_act() -> ActionChunk:
        calls.append(1)
        return ActionChunk(
            actions=np.array([[0.1, 0, 0, 0, 0, 0, 0.5]], dtype=np.float64),
            space="delta_ee_7d",
        )

    result = oracle.intervene(
        _obs(0),
        safety_obs=_safety(phase="idle", active=False, complete=False),
        policy_act=policy_act,
    )
    assert len(calls) == 1
    assert result.policy_inferred is True
    assert result.oracle_paused is False
    assert result.state_after is OracleState.PASS_THROUGH
    assert result.intervention_type is InterventionType.NONE
    assert oracle.stats.policy_infer_steps == 1
    assert oracle.stats.oracle_pause_steps == 0


def test_active_triggers_stop_without_policy_act() -> None:
    oracle = _enabled_oracle()
    oracle.reset(action_space="delta_ee_7d")
    calls: list[int] = []

    def policy_act() -> ActionChunk:
        calls.append(1)
        raise AssertionError("policy.act must not run during STOP")

    # Seed last gripper via a prior pass-through.
    oracle.intervene(
        _obs(0),
        safety_obs=_safety(phase="idle"),
        policy_act=lambda: ActionChunk(
            actions=np.array([[0.2, 0, 0, 0, 0, 0, 0.7]], dtype=np.float64),
            space="delta_ee_7d",
        ),
    )
    result = oracle.intervene(
        _obs(1),
        safety_obs=_safety(step=1, phase="enter", active=True),
        policy_act=policy_act,
    )
    assert calls == []
    assert result.state_after is OracleState.STOP
    assert result.intervention_type is InterventionType.STOP
    assert result.policy_inferred is False
    assert result.oracle_paused is True
    assert result.executed_action.actions.shape == (1, 7)
    # Motion zeroed; gripper held from last nominal.
    np.testing.assert_allclose(result.executed_action.actions[0, :6], 0.0)
    assert result.executed_action.actions[0, 6] == pytest.approx(0.7)
    assert oracle.stats.oracle_pause_steps == 1
    assert oracle.stats.policy_infer_steps == 1  # only the idle step


def test_wait_does_not_call_underlying_act() -> None:
    oracle = _enabled_oracle()
    oracle.reset(action_space="delta_ee_7d")
    mock_act = MagicMock(
        return_value=ActionChunk(
            actions=np.zeros((1, 7), dtype=np.float64), space="delta_ee_7d"
        )
    )

    # Enter STOP then WAIT.
    oracle.intervene(
        _obs(0),
        safety_obs=_safety(phase="approach", active=True),
        policy_act=mock_act,
    )
    assert mock_act.call_count == 0
    assert oracle.state is OracleState.STOP

    r2 = oracle.intervene(
        _obs(1),
        safety_obs=_safety(step=1, phase="dwell", active=True),
        policy_act=mock_act,
    )
    assert mock_act.call_count == 0
    assert r2.state_after is OracleState.WAIT
    assert r2.intervention_type is InterventionType.WAIT
    assert oracle.stats.oracle_pause_steps == 2
    assert oracle.stats.policy_infer_steps == 0


def test_resume_after_motion_complete_restores_inference() -> None:
    oracle = _enabled_oracle()
    oracle.reset(action_space="delta_ee_7d")
    calls: list[str] = []

    def policy_act() -> ActionChunk:
        calls.append("act")
        return ActionChunk(
            actions=np.array([[0.3, 0, 0, 0, 0, 0, -1.0]], dtype=np.float64),
            space="delta_ee_7d",
        )

    # STOP
    oracle.intervene(
        _obs(0),
        safety_obs=_safety(phase="drop", active=True),
        policy_act=policy_act,
    )
    # WAIT
    oracle.intervene(
        _obs(1),
        safety_obs=_safety(step=1, phase="settle", active=True),
        policy_act=policy_act,
    )
    # motion complete → RESUME (still no infer this tick)
    r_resume = oracle.intervene(
        _obs(2),
        safety_obs=_safety(step=2, phase="done", active=False, complete=True),
        policy_act=policy_act,
    )
    assert r_resume.state_after is OracleState.RESUME
    assert r_resume.policy_inferred is False
    assert calls == []

    # Next tick resumes inference
    r_pt = oracle.intervene(
        _obs(3),
        safety_obs=_safety(step=3, phase="done", active=False, complete=True),
        policy_act=policy_act,
    )
    assert calls == ["act"]
    assert r_pt.state_after is OracleState.PASS_THROUGH
    assert r_pt.policy_inferred is True
    assert r_pt.oracle_paused is False
    assert oracle.stats.policy_infer_steps == 1
    assert oracle.stats.oracle_pause_steps == 3


def test_stop_only_never_resumes() -> None:
    oracle = _enabled_oracle(mode="stop_only")
    oracle.reset(action_space="delta_ee_7d")
    calls: list[int] = []

    def policy_act() -> ActionChunk:
        calls.append(1)
        return ActionChunk(
            actions=np.zeros((1, 7), dtype=np.float64), space="delta_ee_7d"
        )

    oracle.intervene(
        _obs(0),
        safety_obs=_safety(phase="enter", active=True),
        policy_act=policy_act,
    )
    for step in range(1, 4):
        r = oracle.intervene(
            _obs(step),
            safety_obs=_safety(step=step, phase="done", active=False, complete=True),
            policy_act=policy_act,
        )
        assert r.policy_inferred is False
        assert r.state_after is OracleState.WAIT
    assert calls == []
    assert oracle.stats.policy_infer_steps == 0
    assert oracle.stats.oracle_pause_steps == 4


def test_l3_withdraw_complete_signal() -> None:
    """Resume when withdraw finishes (motion_complete), not mid-withdraw."""
    oracle = _enabled_oracle()
    oracle.reset(action_space="delta_ee_7d")
    mock = MagicMock(
        return_value=ActionChunk(
            actions=np.zeros((1, 7), dtype=np.float64), space="delta_ee_7d"
        )
    )
    oracle.intervene(
        _obs(0),
        safety_obs=_safety(phase="withdraw", active=True),
        policy_act=mock,
    )
    r_wait = oracle.intervene(
        _obs(1),
        safety_obs=_safety(step=1, phase="withdraw", active=True, complete=False),
        policy_act=mock,
    )
    assert r_wait.state_after is OracleState.WAIT
    assert mock.call_count == 0
    r_res = oracle.intervene(
        _obs(2),
        safety_obs=_safety(step=2, phase="done", active=False, complete=True),
        policy_act=mock,
    )
    assert r_res.intervention_type is InterventionType.RESUME


# --- fail-closed / disabled -------------------------------------------------


def test_missing_observation_fail_closed_no_policy_act() -> None:
    oracle = _enabled_oracle()
    oracle.reset(action_space="delta_ee_7d")
    mock = MagicMock()
    result = oracle.intervene(_obs(0), safety_obs=None, policy_act=mock)
    mock.assert_not_called()
    assert result.intervention_type is InterventionType.FAIL_CLOSED
    assert result.oracle_paused is True
    assert result.policy_inferred is False
    assert result.state_after is OracleState.STOP


def test_disabled_wrapper_is_bit_identical() -> None:
    policy = _mock_policy(
        action=np.array([0.11, -0.2, 0.05, 0.01, 0.0, 0.0, 0.8], dtype=np.float64)
    )
    wrapped = wrap_policy(policy, action_space="delta_ee_7d", config_path=YAML_PATH)
    assert wrapped.oracle.config.enabled is False

    obs = _obs(3)  # no safety key
    out = wrapped.act(obs)
    policy.act.assert_called_once_with(obs)
    direct = policy.act.return_value
    np.testing.assert_array_equal(out.actions, direct.actions)
    assert out.space == direct.space


def test_disabled_ignores_active_disturbance() -> None:
    policy = _mock_policy()
    wrapped = wrap_policy(policy, action_space="delta_ee_7d", config_path=YAML_PATH)
    wrapped.reset(_ctx())
    obs = _obs(1, safety=_safety(phase="enter", active=True))
    wrapped.act(obs)
    policy.act.assert_called_once()


def test_enabled_wrapper_skips_policy_during_wait() -> None:
    policy = _mock_policy()
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    raw["oracle"]["enabled"] = True
    from dynisafety.policies.safety.config import oracle_config_from_mapping

    cfg = oracle_config_from_mapping(raw)
    wrapped = OraclePolicyWrapper(
        policy,
        OracleStopWaitResume.from_config(cfg),
        action_space="delta_ee_7d",
    )
    wrapped.reset(_ctx())

    wrapped.act(_obs(0, safety=_safety(phase="enter", active=True)))
    wrapped.act(_obs(1, safety=_safety(step=1, phase="dwell", active=True)))
    policy.act.assert_not_called()
    assert wrapped.stats_dict()["oracle_pause_steps"] == 2
    assert wrapped.stats_dict()["policy_infer_steps"] == 0


def test_hold_uses_default_gripper_when_no_prior_action() -> None:
    oracle = _enabled_oracle()
    oracle.reset(action_space="delta_ee_7d")
    cfg = load_oracle_config(YAML_PATH)

    def _boom() -> ActionChunk:
        raise AssertionError("policy.act must not run")

    result = oracle.intervene(
        _obs(0),
        safety_obs=_safety(phase="enter", active=True),
        policy_act=_boom,
    )
    assert result.executed_action.actions[0, 6] == pytest.approx(
        cfg.stop_action.default_gripper_hold
    )


def test_safety_observation_rejects_rgb_source() -> None:
    with pytest.raises(ValueError, match="RGB"):
        SafetyObservation(
            step=0,
            disturbance_phase="enter",
            disturbance_active=True,
            motion_complete=False,
            source="rgb_camera",
        )
