"""Kinematic phase, rollback release (R7, R8, D4) and family dispatch (D11 b)."""
import dataclasses

import numpy as np
import pytest

import runtime_world as W
from hazardlib.host import CapabilityError
from hazardlib.kinematics import SampledPath
from hazardlib.semantics import EventKind, InvalidClass

VEL_TOL = W.TOL["c5_release_gravity.release_velocity_abs_tol_mps"]
DTS = (0.001, 0.002, 0.004)       # three hook steps with the same 0.02 s control step
T_CONTACT = 0.4005                # the hand first touches the kinematic hazard here: free at 0.400 on all grids


def _path(inst):
    frames = inst.keyframes
    return SampledPath(frames.time_s, frames.pos, None)


def _intercept(inst):
    """Hand velocity along -x that makes the hand touch the kinematic hazard at T_CONTACT."""
    x_hazard = _path(inst).pos(T_CONTACT)[0]
    speed = -(W.HAZARD_R + W.HAND_R + x_hazard) / T_CONTACT
    assert speed > 0
    return W.away((-speed, 0.0, 0.0))


def _world(dt, mutations=()):
    return W.make(mutations=mutations, physics_dt_s=dt, substeps_per_action=int(round(0.02 / dt)))


def _rollback_release(dt, mutations=(), family="reach"):
    host, runtime = _world(dt, mutations)
    reference, _ = _world(0.002)
    inst = (W.reach_instance if family == "reach" else W.slide_instance)(reference)
    log = W.run_episode(host, runtime, inst, 1.0, action=_intercept(inst))
    return inst, log, runtime.episode_info(0)["release"], runtime.contact_events(0)


@pytest.mark.parametrize("family", ["slide", "reach"])
def test_overlap_is_recorded_and_rolled_back(family):
    inst, log, release, events = _rollback_release(0.002, family=family)
    overlaps = [e for e in events if e.kind is EventKind.KINEMATIC_OVERLAP]
    assert overlaps and all(e.partner_name == "robot0_hand" for e in overlaps)
    assert log.kinematic_overlap_count == len(overlaps)
    assert release["kind"] == "rollback" and release["rollback_t_s"] == pytest.approx(0.4)
    expected = _path(inst).finite_difference_velocity(release["rollback_t_s"], W.WINDOW_S)
    np.testing.assert_allclose(release["readback_lin_vel_mps"], expected, rtol=0, atol=VEL_TOL)
    dynamic = [e for e in events if e.kind is EventKind.DYNAMIC]
    assert dynamic[0].time_s > overlaps[-1].time_s - 1e-12
    assert log.first_official_contact_t_s == dynamic[0].time_s
    assert log.released_t_s == pytest.approx(overlaps[0].time_s)


def test_rollback_velocity_does_not_depend_on_the_hook_step():
    velocities = []
    for dt in DTS:
        inst, log, release, _ = _rollback_release(dt)
        assert release["kind"] == "rollback" and release["rollback_t_s"] == pytest.approx(0.4, abs=1e-12)
        velocities.append(release["readback_lin_vel_mps"])
    expected = _path(inst).finite_difference_velocity(0.4, W.WINDOW_S)
    for velocity in velocities:
        np.testing.assert_allclose(velocity, expected, rtol=0, atol=VEL_TOL)
    # the reach is decelerating here, so the window matters: a one-step difference would differ
    one_step = _path(inst).finite_difference_velocity(0.4, 0.002)
    assert np.max(np.abs(one_step - expected)) > 1e3 * VEL_TOL


def test_hook_dependent_release_is_caught():
    for dt in DTS:
        inst, _, release, _ = _rollback_release(dt, mutations=("hook_dependent_release",))
        expected = _path(inst).finite_difference_velocity(release["rollback_t_s"], W.WINDOW_S)
        assert np.max(np.abs(release["readback_lin_vel_mps"] - expected)) > 0.1
        np.testing.assert_allclose(release["lin_vel_mps"], expected, rtol=0, atol=VEL_TOL)  # the runtime asked right


def test_nominal_release_uses_the_instance_velocity_on_every_hook_step():
    host, _ = _world(0.002)
    inst = W.reach_instance(host)
    for dt in DTS:
        for mutations, same in (((), True), (("hook_dependent_release",), False)):
            host, runtime = _world(dt, mutations)
            W.run_episode(host, runtime, inst, 1.0)
            release = runtime.episode_info(0)["release"]
            assert release["kind"] == "nominal"
            close = np.allclose(release["readback_lin_vel_mps"], inst.release_velocity.lin_mps, rtol=0, atol=VEL_TOL)
            assert close is same, (dt, mutations)


def test_residual_teleport_velocity_shows_in_the_readback():
    host, runtime = W.make(mutations=("residual_velocity",))
    inst = W.slide_instance(host)
    W.run_episode(host, runtime, inst, 1.0)
    release = runtime.episode_info(0)["release"]
    np.testing.assert_allclose(release["lin_vel_mps"], inst.release_velocity.lin_mps, atol=VEL_TOL)
    assert np.max(np.abs(release["readback_lin_vel_mps"] - release["lin_vel_mps"])) > 0.1


# -------------------------------------------------------------------------------------- setup_invalid

def test_overlap_at_the_first_placement_is_setup_invalid():
    host, runtime = W.make()
    inst = W.slide_instance(host)
    np.testing.assert_allclose(_path(inst).pos(_path(inst).t_start_s), W.SLIDE_START, atol=1e-12)
    log = W.run_episode(host, runtime, inst, 1.0, task="blocker")
    info = runtime.episode_info(0)
    assert info["setup_failed"] and info["setup_reason"] == "overlap at the first placement"
    assert info["phase"] == "setup_failed" and info["release"] is None
    assert log.invalid_class is InvalidClass.SETUP_INVALID
    assert log.released_t_s is None and log.first_official_contact_t_s is None
    assert log.kinematic_overlap_count >= 1


def test_no_room_for_the_release_window_is_setup_invalid():
    host, runtime = W.make(cfg=W.runtime_config(release_velocity_window_s=0.5))
    inst = W.reach_instance(host)
    log = W.run_episode(host, runtime, inst, 1.0, action=_intercept(inst))
    info = runtime.episode_info(0)
    assert info["setup_failed"] and "release window" in info["setup_reason"]
    assert log.invalid_class is InvalidClass.SETUP_INVALID and log.released_t_s is None
    parked_after = [e for e in runtime.contact_events(0) if e.kind is EventKind.DYNAMIC]
    assert parked_after == []


# ---------------------------------------------------------------------------------- family dispatch

def test_reach_with_release_equal_to_onset_still_rolls_back():
    host, runtime = W.make()
    base = W.reach_instance(host)
    inst = dataclasses.replace(base, onset_time_s=base.release_time_s)
    assert inst.family == "intrusion.reach" and inst.onset_time_s == inst.release_time_s
    log = W.run_episode(host, runtime, inst, 1.0, action=_intercept(base))
    release = runtime.episode_info(0)["release"]
    assert release["kind"] == "rollback" and log.kinematic_overlap_count > 0
    plain_host, plain_runtime = W.make()
    W.run_episode(plain_host, plain_runtime, inst, 1.0)
    assert plain_runtime.episode_info(0)["release"]["kind"] == "nominal"


def test_fall_has_no_kinematic_phase_even_when_it_starts_inside_the_robot():
    host, runtime = W.make()
    hand = np.array(W.EEF_START)
    inst = W.fall_onto(hand + np.array([0.0, 0.0, -0.3]), t_hit=0.6, tau=0.1)  # starts inside the hand
    log = W.run_episode(host, runtime, inst, 1.0)
    assert log.kinematic_overlap_count == 0
    assert runtime.episode_info(0)["release"]["kind"] == "onset"
    assert log.invalid_class is InvalidClass.NONE


def test_unknown_family_is_a_capability_error():
    host, runtime = W.make()
    inst = dataclasses.replace(W.slide_instance(host), family="intrusion.swing")
    with pytest.raises(CapabilityError, match="intrusion.swing"):
        runtime.arm(0, inst)


@pytest.mark.parametrize("change, message", [
    (dict(release_time_s=None), "no release"),
    (dict(onset_time_s=-0.1), "before the episode start"),
    (dict(release_time_s=9.0), "release_time_s"),
])
def test_kinematic_instances_are_checked_at_arm(change, message):
    host, runtime = W.make()
    with pytest.raises((ValueError, CapabilityError), match=message):
        runtime.arm(0, dataclasses.replace(W.slide_instance(host), **change))


def test_fall_instances_are_checked_at_arm():
    host, runtime = W.make()
    fall = W.fall_instance(host)
    with pytest.raises(ValueError, match="must equal onset_time_s"):
        runtime.arm(0, dataclasses.replace(fall, release_time_s=fall.onset_time_s + 0.01))
    moving = dataclasses.replace(fall, release_velocity=W.slide_instance(host).release_velocity)
    with pytest.raises(ValueError, match="must be zero"):
        runtime.arm(0, moving)
