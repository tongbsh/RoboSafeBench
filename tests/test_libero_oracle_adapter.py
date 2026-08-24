"""LIBERO Oracle adapter tests (Prompt 11) — CPU_ONLY; no robotwin/simplerenv."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from dynisafety.backends.libero.disturbance._support import Phase as L3Phase
from dynisafety.backends.libero.safety.observation import build_safety_observation
from dynisafety.backends.libero.safety.phases import (
    LEVEL_SEMANTICS,
    LiberoOraclePhase,
    probe_controller_phase,
)
from dynisafety.backends.libero.safety.runtime import (
    LiberoOracleSession,
    oracle_enabled_from_extra,
)
from dynisafety.policies.base import ActionChunk, EpisodeContext, Observation
from dynisafety.policies.safety import wrap_policy


def _obs(step: int = 0) -> Observation:
    return Observation(
        images={"primary": np.zeros((4, 4, 3), dtype=np.uint8)},
        state=None,
        instruction="test",
        step=step,
        extra={},
    )


def _ctx(level: int = 1) -> EpisodeContext:
    return EpisodeContext(
        instruction="t",
        task_id=0,
        task_name="t",
        episode_index=0,
        seed=0,
        level=level,
        backend="libero",
        suite="safelibero_spatial",
        extra={},
    )


# --- phase probes -----------------------------------------------------------


def test_l0_and_none_controller_inactive() -> None:
    r = probe_controller_phase(None, level=0, task_step=0)
    assert r.phase is LiberoOraclePhase.INACTIVE
    assert r.motion_complete is True


def test_l1_pre_drop_inactive_then_active_then_settled() -> None:
    ctrl = SimpleNamespace(_dropped=False, drop_step=None)
    assert (
        probe_controller_phase(ctrl, level=1, task_step=0, settle_steps=0).phase
        is LiberoOraclePhase.INACTIVE
    )
    ctrl = SimpleNamespace(_dropped=True, drop_step=5)
    active = probe_controller_phase(ctrl, level=1, task_step=5, settle_steps=0)
    assert active.phase is LiberoOraclePhase.ENTERING_OR_ACTIVE
    assert active.disturbance_active is True
    settled = probe_controller_phase(ctrl, level=1, task_step=6, settle_steps=0)
    assert settled.phase is LiberoOraclePhase.SETTLED_OR_DONE
    assert settled.motion_complete is True
    assert settled.disturbance_visible is True  # static obstacle remains


def test_l2_move_settle_static_not_hand() -> None:
    ctrl = SimpleNamespace(
        move_start_step=10,
        move_end_step=30,
        _settle_end_step=45,
        released=False,
    )
    assert (
        probe_controller_phase(ctrl, level=2, task_step=0).phase
        is LiberoOraclePhase.INACTIVE
    )
    moving = probe_controller_phase(ctrl, level=2, task_step=15)
    assert moving.phase is LiberoOraclePhase.ENTERING_OR_ACTIVE
    assert "hand" not in moving.level_semantics.lower() or "NOT a human hand" in moving.level_semantics
    settling = probe_controller_phase(ctrl, level=2, task_step=35)
    assert settling.phase is LiberoOraclePhase.ENTERING_OR_ACTIVE
    assert settling.raw_controller_phase == "settling"
    done = probe_controller_phase(ctrl, level=2, task_step=45)
    assert done.phase is LiberoOraclePhase.SETTLED_OR_DONE
    assert "NOT a human hand" in LEVEL_SEMANTICS[2]


def test_l3_withdraw_then_withdrawn() -> None:
    ctrl = SimpleNamespace(_phase=L3Phase.HIDDEN)
    assert (
        probe_controller_phase(ctrl, level=3, task_step=0).phase
        is LiberoOraclePhase.INACTIVE
    )
    ctrl._phase = L3Phase.APPROACH
    active = probe_controller_phase(ctrl, level=3, task_step=20)
    assert active.phase is LiberoOraclePhase.ENTERING_OR_ACTIVE
    ctrl._phase = L3Phase.WITHDRAW
    still = probe_controller_phase(ctrl, level=3, task_step=40)
    assert still.disturbance_active is True
    ctrl._phase = L3Phase.DONE
    done = probe_controller_phase(ctrl, level=3, task_step=80)
    assert done.phase is LiberoOraclePhase.WITHDRAWN
    assert done.motion_complete is True


def test_safety_observation_maps_phases() -> None:
    ctrl = SimpleNamespace(
        move_start_step=10,
        move_end_step=30,
        _settle_end_step=45,
        released=False,
    )
    safety = build_safety_observation(
        controller=ctrl, level=2, task_step=15, settle_steps=15
    )
    assert safety.disturbance_active is True
    assert safety.motion_complete is False
    assert safety.source == "god_view"
    assert safety.extra["libero_oracle_phase"] == LiberoOraclePhase.ENTERING_OR_ACTIVE.value


# --- enable / L0 bypass -----------------------------------------------------


def test_oracle_disabled_by_default_and_l0_bypass() -> None:
    assert oracle_enabled_from_extra({}, level=1) is False
    assert oracle_enabled_from_extra({"oracle": {"enabled": True}}, level=0) is False
    assert oracle_enabled_from_extra({"oracle": {"enabled": True}}, level=2) is True
    assert oracle_enabled_from_extra({"oracle": {"enabled": False}}, level=2) is False


def test_wait_period_no_vla_forward_with_libero_phases() -> None:
    """WAIT must not call underlying Policy.act (openvla_oft stand-in mock)."""
    base = MagicMock()
    base.act.return_value = ActionChunk(
        actions=np.array([[0.1, 0, 0, 0, 0, 0, 0.5]], dtype=np.float64),
        space="delta_ee_7d",
    )
    wrapped = wrap_policy(base, action_space="delta_ee_7d", enabled=True)
    wrapped.reset(_ctx(level=2))

    session = LiberoOracleSession(
        enabled=True,
        level=2,
        settle_steps=15,
        policy=wrapped,
    )
    ctrl = SimpleNamespace(
        move_start_step=10,
        move_end_step=30,
        _settle_end_step=45,
        released=False,
    )

    # Pre-move: should infer
    o0 = session.inject_observation(_obs(0), controller=ctrl, task_step=0)
    session.active_policy.act(o0)
    session.note_after_act()
    assert base.act.call_count == 1

    # Active window: no further VLA forwards
    for step in (10, 20, 30, 40):
        o = session.inject_observation(_obs(step), controller=ctrl, task_step=step)
        assert session.should_clear_action_queue(o) is True
        session.active_policy.act(o)
        session.note_after_act()
    assert base.act.call_count == 1

    # Settled → resume tick (still may not infer) then pass-through infers
    o_done = session.inject_observation(_obs(45), controller=ctrl, task_step=45)
    session.active_policy.act(o_done)
    session.note_after_act()
    o_done2 = session.inject_observation(_obs(46), controller=ctrl, task_step=46)
    session.active_policy.act(o_done2)
    session.note_after_act()

    extra = session.episode_extra()
    assert base.act.call_count >= 2
    assert extra["oracle_pause_steps"] >= 1
    assert extra["policy_infer_steps"] >= 1
    assert "oracle_state_summary" in extra
    assert extra["oracle_resume_occurred"] is True
    assert "not a human hand" in extra["oracle_note"].lower()

def test_session_create_l0_does_not_wrap() -> None:
    from dynisafety.config.schema import MISSING

    base = MagicMock()
    cfg = MagicMock()
    cfg.level = 0
    cfg.disturbance = MISSING
    session = LiberoOracleSession.create(
        policy=base,
        cfg=cfg,
        extra_libero={"oracle": {"enabled": True}},
        action_space="delta_ee_7d",
    )
    assert session.enabled is False
    assert session.active_policy is base


def _paused_obs(*, active: bool, complete: bool) -> Observation:
    obs = _obs()
    obs.extra = {
        "safety": SimpleNamespace(
            disturbance_active=active,
            motion_complete=complete,
        )
    }
    return obs


def test_exempts_vla_budget_only_l2_l3_god_view_pause() -> None:
    session = LiberoOracleSession(
        enabled=True,
        level=2,
        settle_steps=15,
        policy=MagicMock(),
    )
    paused = _paused_obs(active=True, complete=False)
    idle = _paused_obs(active=False, complete=False)
    done = _paused_obs(active=False, complete=True)

    assert session.exempts_vla_budget(paused) is True
    assert session.exempts_vla_budget(idle) is False
    assert session.exempts_vla_budget(done) is False

    session.level = 3
    assert session.exempts_vla_budget(paused) is True
    assert session.exempts_vla_budget(done) is False

    session.level = 1
    assert session.exempts_vla_budget(paused) is False

    session.level = 0
    assert session.exempts_vla_budget(paused) is False

    session.level = 3
    session.enabled = False
    assert session.exempts_vla_budget(paused) is False


def test_exempts_vla_budget_fail_closed_missing_safety() -> None:
    session = LiberoOracleSession(
        enabled=True,
        level=2,
        settle_steps=15,
        policy=MagicMock(),
    )
    assert session.exempts_vla_budget(_obs()) is True
    session.level = 1
    assert session.exempts_vla_budget(_obs()) is False


def test_level_semantics_forbid_hand_as_l2() -> None:
    text = LEVEL_SEMANTICS[2].lower()
    assert "not a human hand" in text
    assert "hand" in LEVEL_SEMANTICS[3].lower()
