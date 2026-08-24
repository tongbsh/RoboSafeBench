"""Fake-controller tests for diagnostics phase probes (no simulator)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynisafety.diagnostics.probes import KNOWN_BACKENDS, make_probe
from dynisafety.diagnostics.schema import DisturbancePhase, DisturbanceState

P = DisturbancePhase
BACKENDS = ("libero", "simplerenv", "robotwin", "robocasa")
PROBES_DIR = Path(__file__).resolve().parents[1] / "src" / "dynisafety" / "diagnostics" / "probes"


def _probe(backend: str, ctrl: object, level: int):
    probe = make_probe(backend, ctrl, level=level)
    assert probe is not None
    return probe


# ---------------------------------------------------------------------------
# make_probe factory
# ---------------------------------------------------------------------------


def test_make_probe_l0_returns_none() -> None:
    assert make_probe("libero", SimpleNamespace(), level=0) is None


def test_make_probe_none_controller_returns_none() -> None:
    assert make_probe("libero", None, level=1) is None
    assert make_probe("simplerenv", None, level=2) is None


def test_make_probe_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown diagnostics backend"):
        make_probe("nope", SimpleNamespace(), level=1)


def test_make_probe_unsupported_level_raises() -> None:
    with pytest.raises(ValueError, match="unsupported disturbance level"):
        make_probe("libero", SimpleNamespace(), level=4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_make_probe_known_backends_l1(backend: str) -> None:
    probe = make_probe(backend, SimpleNamespace(), level=1)
    assert probe is not None
    assert isinstance(probe.snapshot(0), DisturbanceState)


# ---------------------------------------------------------------------------
# Missing attributes: never raise, conservative values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("level, persists", [(1, True), (2, True), (3, False)])
def test_empty_controller_is_conservative(
    backend: str, level: int, persists: bool
) -> None:
    state = _probe(backend, SimpleNamespace(), level).snapshot(0)
    assert state.phase is P.INACTIVE
    assert state.obstacle_pos is None
    assert state.onset_step is None
    assert state.hazard_end_step is None
    assert state.hazard_persists is persists


def test_raising_attribute_does_not_crash_rollout() -> None:
    class Boom:
        @property
        def move_start_step(self) -> int:
            raise RuntimeError("controller exploded")

    state = _probe("libero", Boom(), 2).snapshot(10)
    assert state.phase is P.INACTIVE
    assert state.hazard_end_step is None


# ---------------------------------------------------------------------------
# hazard_persists is a level property, not a phase property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_hazard_persists_by_level(backend: str) -> None:
    assert _probe(backend, SimpleNamespace(), 1).snapshot(0).hazard_persists is True
    assert _probe(backend, SimpleNamespace(), 2).snapshot(0).hazard_persists is True
    assert _probe(backend, SimpleNamespace(), 3).snapshot(0).hazard_persists is False


# ---------------------------------------------------------------------------
# LIBERO
# ---------------------------------------------------------------------------


def test_libero_l1_inactive_active_settling_cleared_and_lock() -> None:
    ctrl = SimpleNamespace(
        _dropped=False,
        drop_step=None,
        settle_steps=5,
        disturbance_onset_step=None,
    )
    probe = _probe("libero", ctrl, 1)
    assert probe.snapshot(3).phase is P.INACTIVE

    ctrl._dropped = True
    ctrl.drop_step = 10
    ctrl.disturbance_onset_step = 10
    assert probe.snapshot(10).phase is P.ACTIVE
    assert probe.snapshot(10).onset_step == 10
    assert probe.snapshot(10).hazard_end_step is None
    assert probe.snapshot(12).phase is P.SETTLING
    cleared = probe.snapshot(15)
    assert cleared.phase is P.CLEARED
    assert cleared.hazard_end_step == 15
    assert cleared.hazard_persists is True
    assert probe.snapshot(40).hazard_end_step == 15


def test_libero_l2_half_path_then_settle_then_cleared() -> None:
    ctrl = SimpleNamespace(
        move_start_step=10,
        move_end_step=20,
        _settle_end_step=25,
        released=False,
        blocked=False,
        disturbance_onset_step=10,
    )
    probe = _probe("libero", ctrl, 2)
    assert probe.snapshot(5).phase is P.INACTIVE
    assert probe.snapshot(10).phase is P.APPROACHING
    assert probe.snapshot(14).phase is P.APPROACHING
    assert probe.snapshot(15).phase is P.ACTIVE
    assert probe.snapshot(19).phase is P.ACTIVE
    assert probe.snapshot(20).phase is P.SETTLING
    cleared = probe.snapshot(25)
    assert cleared.phase is P.CLEARED
    assert cleared.hazard_end_step == 25
    assert cleared.onset_step == 10
    assert probe.snapshot(80).hazard_end_step == 25


def test_libero_l2_blocked_uses_block_task_step() -> None:
    ctrl = SimpleNamespace(
        move_start_step=10,
        move_end_step=20,
        _settle_end_step=25,
        released=False,
        blocked=True,
        _block_task_step=13,
        disturbance_onset_step=10,
    )
    probe = _probe("libero", ctrl, 2)
    state = probe.snapshot(18)
    assert state.phase is P.CLEARED
    assert state.hazard_end_step == 13
    assert probe.snapshot(99).hazard_end_step == 13


def test_libero_l3_phase_vocabulary() -> None:
    ctrl = SimpleNamespace(_phase="hidden", _entry_start_step=8, disturbance_onset_step=8)
    probe = _probe("libero", ctrl, 3)
    assert probe.snapshot(0).phase is P.INACTIVE
    ctrl._phase = "wait"
    assert probe.snapshot(4).phase is P.INACTIVE
    ctrl._phase = "enter"
    assert probe.snapshot(8).phase is P.APPROACHING
    ctrl._phase = "approach"
    assert probe.snapshot(12).phase is P.APPROACHING
    ctrl._phase = "dwell"
    assert probe.snapshot(20).phase is P.ACTIVE
    ctrl._phase = "flinch"
    assert probe.snapshot(21).phase is P.ACTIVE
    ctrl._phase = "withdraw"
    assert probe.snapshot(30).phase is P.SETTLING  # still on-field
    ctrl._phase = "exit"
    assert probe.snapshot(40).phase is P.SETTLING  # only DONE is CLEARED
    ctrl._phase = "done"
    done = probe.snapshot(50)
    assert done.phase is P.CLEARED
    assert done.hazard_persists is False
    assert done.hazard_end_step == 50
    assert done.onset_step == 8
    assert probe.snapshot(90).hazard_end_step == 50


# ---------------------------------------------------------------------------
# SimplerEnv
# ---------------------------------------------------------------------------


def test_simplerenv_l1_one_tick_then_cleared() -> None:
    ctrl = SimpleNamespace(_dropped=False, drop_step=None, disturbance_onset_step=None)
    probe = _probe("simplerenv", ctrl, 1)
    assert probe.snapshot(2).phase is P.INACTIVE
    ctrl._dropped = True
    ctrl.drop_step = 7
    ctrl.disturbance_onset_step = 7
    assert probe.snapshot(7).phase is P.ACTIVE
    cleared = probe.snapshot(8)
    assert cleared.phase is P.CLEARED
    assert cleared.hazard_end_step == 8
    assert cleared.hazard_persists is True
    assert probe.snapshot(20).hazard_end_step == 8


def test_simplerenv_l2_uses_private_released() -> None:
    ctrl = SimpleNamespace(
        move_start_step=0,
        move_end_step=10,
        _settle_end_step=12,
        _released=False,
        blocked=False,
        disturbance_onset_step=0,
    )
    probe = _probe("simplerenv", ctrl, 2)
    assert probe.snapshot(0).phase is P.APPROACHING
    assert probe.snapshot(6).phase is P.ACTIVE
    assert probe.snapshot(10).phase is P.SETTLING
    ctrl._released = True
    cleared = probe.snapshot(12)
    assert cleared.phase is P.CLEARED
    assert cleared.hazard_end_step == 12
    assert probe.snapshot(30).hazard_end_step == 12


def test_simplerenv_l3_public_phase_property() -> None:
    ctrl = SimpleNamespace(phase="hidden", _entry_start_step=3, disturbance_onset_step=3)
    probe = _probe("simplerenv", ctrl, 3)
    assert probe.snapshot(0).phase is P.INACTIVE
    ctrl.phase = "enter"
    assert probe.snapshot(3).phase is P.APPROACHING
    ctrl.phase = "dwell"
    assert probe.snapshot(10).phase is P.ACTIVE
    ctrl.phase = "post_hit_hold"
    assert probe.snapshot(11).phase is P.SETTLING  # still on-field
    ctrl.phase = "flinch_exit"
    assert probe.snapshot(12).phase is P.SETTLING  # only DONE is CLEARED
    ctrl.phase = "done"
    done = probe.snapshot(20)
    assert done.phase is P.CLEARED
    assert done.hazard_persists is False
    assert done.hazard_end_step == 20
    assert probe.snapshot(40).hazard_end_step == 20


# ---------------------------------------------------------------------------
# RoboTwin
# ---------------------------------------------------------------------------


def test_robotwin_l1_placed_step_and_release_clears_private_onset() -> None:
    ctrl = SimpleNamespace(_dropped=False, _placed_sim_step=None, _xy_latched=False)
    probe = _probe("robotwin", ctrl, 1)
    assert probe.snapshot(4).phase is P.INACTIVE
    ctrl._xy_latched = True
    assert probe.snapshot(5).phase is P.INACTIVE
    ctrl._dropped = True
    ctrl._placed_sim_step = 9
    assert probe.snapshot(9).phase is P.ACTIVE
    assert probe.snapshot(9).onset_step == 9
    cleared = probe.snapshot(10)
    assert cleared.phase is P.CLEARED
    assert cleared.hazard_end_step == 10
    # Real controller nulls _placed_sim_step after physics release; onset stays locked.
    ctrl._placed_sim_step = None
    later = probe.snapshot(40)
    assert later.phase is P.CLEARED
    assert later.onset_step == 9
    assert later.hazard_end_step == 10


def test_robotwin_l2_inclusive_move_end_and_hold_end() -> None:
    ctrl = SimpleNamespace(
        move_start_step=1,
        move_end_step=11,
        hold_end_step=14,
        _released=False,
        _frozen=False,
        release_reason=None,
    )
    probe = _probe("robotwin", ctrl, 2)
    assert probe.snapshot(0).phase is P.INACTIVE
    assert probe.snapshot(1).phase is P.APPROACHING
    assert probe.snapshot(11).phase is P.ACTIVE  # end inclusive
    assert probe.snapshot(12).phase is P.SETTLING
    ctrl._released = True
    ctrl.release_reason = "settle"
    cleared = probe.snapshot(14)
    assert cleared.phase is P.CLEARED
    assert cleared.hazard_end_step == 14
    assert cleared.onset_step == 1
    assert probe.snapshot(99).hazard_end_step == 14


def test_robotwin_l3_public_phase_no_withdraw() -> None:
    ctrl = SimpleNamespace(phase="hidden", phase_start=0, move_start_step=6)
    probe = _probe("robotwin", ctrl, 3)
    assert probe.snapshot(2).phase is P.INACTIVE
    ctrl.phase = "enter"
    ctrl.phase_start = 6
    assert probe.snapshot(6).phase is P.APPROACHING
    ctrl.phase = "dwell"
    assert probe.snapshot(20).phase is P.ACTIVE
    ctrl.phase = "exit"
    assert probe.snapshot(30).phase is P.SETTLING  # still on-field; only DONE is CLEARED
    ctrl.phase = "done"
    done = probe.snapshot(40)
    assert done.phase is P.CLEARED
    assert done.hazard_persists is False
    assert done.hazard_end_step == 40
    assert done.onset_step == 6
    assert probe.snapshot(70).hazard_end_step == 40


def test_robotwin_unwraps_manager() -> None:
    inner = SimpleNamespace(
        _dropped=True, _placed_sim_step=4, disturbance_onset_step=None
    )
    manager = SimpleNamespace(controller=inner, dynamic_level=1)
    probe = _probe("robotwin", manager, 1)
    state = probe.snapshot(4)
    assert state.phase is P.ACTIVE
    assert state.onset_step == 4


# ---------------------------------------------------------------------------
# RoboCasa (floor nav, not tabletop)
# ---------------------------------------------------------------------------


def test_robocasa_l1_appear_at_rest_is_immediately_cleared() -> None:
    ctrl = SimpleNamespace(
        _dropped=False,
        drop_step=None,
        blocked=False,
        spawn_failed=False,
        disturbance_onset_step=None,
    )
    probe = _probe("robocasa", ctrl, 1)
    assert probe.snapshot(3).phase is P.INACTIVE
    ctrl.spawn_failed = True
    ctrl.blocked = True
    assert probe.snapshot(4).phase is P.INACTIVE
    ctrl.spawn_failed = False
    ctrl.blocked = False
    ctrl._dropped = True
    ctrl.drop_step = 12
    ctrl.disturbance_onset_step = 12
    cleared = probe.snapshot(12)
    assert cleared.phase is P.CLEARED
    assert cleared.hazard_end_step == 12
    assert cleared.hazard_persists is True
    assert probe.snapshot(30).hazard_end_step == 12


def test_robocasa_l2_pingpong_does_not_use_move_start_as_onset() -> None:
    ctrl = SimpleNamespace(
        _spawned=True,
        onset_step=None,
        disturbance_onset_step=None,
        released=False,
        blocked=False,
        move_start_step=0,
        move_end_step=8,
        _one_way_steps=8,
    )
    probe = _probe("robocasa", ctrl, 2)
    parked = probe.snapshot(5)
    assert parked.phase is P.INACTIVE
    assert parked.onset_step is None

    ctrl.onset_step = 20
    ctrl.disturbance_onset_step = 20
    assert probe.snapshot(21).phase is P.APPROACHING
    assert probe.snapshot(25).phase is P.ACTIVE
    assert probe.snapshot(25).onset_step == 20
    assert probe.snapshot(25).hazard_end_step is None
    ctrl.released = True
    cleared = probe.snapshot(40)
    assert cleared.phase is P.CLEARED
    assert cleared.hazard_end_step == 40
    assert probe.snapshot(80).hazard_end_step == 40


def test_robocasa_l2_clean_episode_hazard_end_stays_none() -> None:
    """Intentional: L2 ping-pongs for the rest of the episode. Do not "fix" this.

    A clean episode never enters CLEARED, so ``hazard_end_step`` stays None
    and Recovery is n/a. Tables must render n/a, never 0.0.
    """
    ctrl = SimpleNamespace(
        _spawned=True,
        onset_step=20,
        disturbance_onset_step=20,
        released=False,
        blocked=False,
        move_start_step=0,
        move_end_step=8,
        _one_way_steps=8,
    )
    probe = _probe("robocasa", ctrl, 2)
    for t in (20, 21, 24, 28, 40, 80, 200):
        state = probe.snapshot(t)
        assert state.hazard_end_step is None
        assert state.phase is not P.CLEARED
    assert probe.snapshot(20).phase is P.APPROACHING
    assert probe.snapshot(28).phase is P.ACTIVE


def test_robocasa_l3_floor_human() -> None:
    ctrl = SimpleNamespace(_phase="hidden", onset_step=None, disturbance_onset_step=None)
    probe = _probe("robocasa", ctrl, 3)
    assert probe.snapshot(0).phase is P.INACTIVE
    ctrl.onset_step = 15
    ctrl.disturbance_onset_step = 15
    ctrl._phase = "enter"
    assert probe.snapshot(15).phase is P.APPROACHING
    ctrl._phase = "approach"
    assert probe.snapshot(20).phase is P.APPROACHING
    ctrl._phase = "dwell"
    assert probe.snapshot(30).phase is P.ACTIVE
    ctrl._phase = "withdraw"
    assert probe.snapshot(40).phase is P.SETTLING  # still on-field; only DONE is CLEARED
    ctrl._phase = "done"
    done = probe.snapshot(50)
    assert done.phase is P.CLEARED
    assert done.hazard_persists is False
    assert done.hazard_end_step == 50
    assert done.onset_step == 15
    assert probe.snapshot(70).hazard_end_step == 50


# ---------------------------------------------------------------------------
# Obstacle pose duck-typing + L3 CLEARED hides the body
# ---------------------------------------------------------------------------


def test_obstacle_pos_from_actor_pose() -> None:
    class _Pose:
        def __init__(self) -> None:
            self.p = (0.1, 0.2, 0.3)

    class _Actor:
        def get_pose(self) -> _Pose:
            return _Pose()

    ctrl = SimpleNamespace(
        _dropped=True,
        drop_step=2,
        disturbance_onset_step=2,
        actor=_Actor(),
    )
    state = _probe("libero", ctrl, 1).snapshot(2)
    assert state.phase is P.ACTIVE
    assert state.obstacle_pos == (0.1, 0.2, 0.3)


def test_l3_cleared_does_not_report_obstacle_pos() -> None:
    class _Pose:
        p = (1.0, 2.0, 3.0)

    class _Actor:
        def get_pose(self) -> _Pose:
            return _Pose()

    ctrl = SimpleNamespace(phase="done", move_start_step=1, actor=_Actor())
    state = _probe("robotwin", ctrl, 3).snapshot(10)
    assert state.phase is P.CLEARED
    assert state.obstacle_pos is None


# ---------------------------------------------------------------------------
# First-seen onset latch (shared SnapshotProbe; all backends / levels)
# ---------------------------------------------------------------------------


def _l1_onset_ctrl(*, step: int) -> SimpleNamespace:
    return SimpleNamespace(
        _dropped=True,
        drop_step=step,
        _placed_sim_step=step,
        disturbance_onset_step=step,
        onset_step=step,
    )


def _clear_onset_fields(ctrl: SimpleNamespace) -> None:
    ctrl.drop_step = None
    ctrl._placed_sim_step = None
    ctrl.disturbance_onset_step = None
    ctrl.onset_step = None
    if hasattr(ctrl, "move_start_step"):
        ctrl.move_start_step = None
    if hasattr(ctrl, "_entry_start_step"):
        ctrl._entry_start_step = None


@pytest.mark.parametrize("backend", BACKENDS)
def test_onset_stays_latched_after_controller_clears(backend: str) -> None:
    """First non-None onset is locked even if the controller later nulls it."""
    ctrl = _l1_onset_ctrl(step=7)
    probe = _probe(backend, ctrl, 1)
    assert probe.snapshot(7).onset_step == 7
    _clear_onset_fields(ctrl)
    assert probe.snapshot(20).onset_step == 7


@pytest.mark.parametrize("backend", BACKENDS)
def test_l2_onset_stays_latched_after_controller_clears(backend: str) -> None:
    ctrl = SimpleNamespace(
        move_start_step=10,
        move_end_step=20,
        _settle_end_step=25,
        hold_end_step=25,
        _one_way_steps=8,
        _spawned=True,
        released=False,
        _released=False,
        blocked=False,
        disturbance_onset_step=10,
        onset_step=10,
    )
    probe = _probe(backend, ctrl, 2)
    assert probe.snapshot(12).onset_step == 10
    _clear_onset_fields(ctrl)
    assert probe.snapshot(30).onset_step == 10


@pytest.mark.parametrize("backend", BACKENDS)
def test_l3_onset_stays_latched_after_controller_clears(backend: str) -> None:
    ctrl = SimpleNamespace(
        phase="dwell",
        _phase="dwell",
        disturbance_onset_step=8,
        onset_step=8,
        _entry_start_step=8,
        move_start_step=8,
    )
    probe = _probe(backend, ctrl, 3)
    assert probe.snapshot(20).onset_step == 8
    _clear_onset_fields(ctrl)
    assert probe.snapshot(40).onset_step == 8


@pytest.mark.parametrize("backend", BACKENDS)
def test_new_probe_instance_does_not_inherit_onset_latch(backend: str) -> None:
    """Each episode must call make_probe again; a new instance has no latch."""
    ctrl = _l1_onset_ctrl(step=4)
    first = _probe(backend, ctrl, 1)
    assert first.snapshot(4).onset_step == 4
    _clear_onset_fields(ctrl)
    nxt = _probe(backend, ctrl, 1)
    assert nxt.snapshot(0).onset_step is None


# ---------------------------------------------------------------------------
# Probe package must not import any backend (simulators are exclusive)
# ---------------------------------------------------------------------------


def test_probe_modules_do_not_import_backends() -> None:
    banned = ("dynisafety.backends", "mujoco", "sapien", "robosuite", "robocasa")
    for path in sorted(PROBES_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for mod in modules:
                for prefix in banned:
                    assert not mod.startswith(prefix), f"{path.name} imports {mod}"


def test_known_backends_tuple_matches_factory() -> None:
    assert KNOWN_BACKENDS == BACKENDS
