"""Follow-through of a kinematic hazard that is not released (decision D19): playback of ``keyframes`` +
``exit_keyframes``, the engagement rule at the nominal release, the EXITED phase, rollback during the
follow-through and the capability check. Pure-Python FakeHost, no simulator."""
import dataclasses

import numpy as np
import pytest

import runtime_world as W
from hazardlib.conformance import FakeAction, FakeTask
from hazardlib.host import CapabilityError
from hazardlib.kinematics import SampledPath
from hazardlib.runtime import EXITED, KINEMATIC, PARKED, RELEASED
from hazardlib.semantics import EventKind
from hazardlib.spec import HazardInstance

DTS = (0.001, 0.002, 0.004)       # three hook steps with the same 0.02 s control step
EXIT_S = 0.6                      # length of the "pass" exit path built by with_exit
LAYOUTS = W.tasks()["static"].layouts
MODES = ("pass", "retract")


def _task(times, positions):
    return FakeTask(times, positions, positions[-1], 0.01, LAYOUTS)


def _world(dt=0.002, extra_tasks=None, **kwargs):
    tasks = dict(W.tasks())
    tasks.update(extra_tasks or {})
    return W.make(physics_dt_s=dt, substeps_per_action=int(round(0.02 / dt)), tasks=tasks, **kwargs)


def _instance(host, family):
    return W.slide_instance(host) if family == "slide" else W.reach_instance(host)


def with_exit(inst, mode, exit_s=EXIT_S):
    """``inst`` with an exit path, built here without the builder: the keyframes are cut at the release time;
    ``pass`` continues at the release velocity for ``exit_s``; ``retract`` follows the rest of the analytic path
    (a reach with overshoot) and then retraces the whole path backwards to its start."""
    frames = inst.keyframes
    times, pos = np.array(frames.time_s), np.array(frames.pos)
    release = inst.release_time_s
    cut = SampledPath(times, pos, None).pos(release)
    keep = times < release - 1e-9
    key_t, key_p = np.append(times[keep], release), np.vstack([pos[keep], cut])
    if mode == "pass":
        exit_t = release + 0.01 * np.arange(int(round(exit_s / 0.01)) + 1)
        exit_p = cut + np.outer(exit_t - release, np.array(inst.release_velocity.lin_mps))
    else:
        beyond = times > release + 1e-9
        fwd_t, fwd_p = np.concatenate([[release], times[beyond]]), np.vstack([cut[None, :], pos[beyond]])
        turn = fwd_t[-1]
        back_t = turn + (times[-1] - times[::-1])[1:]
        exit_t, exit_p = np.concatenate([fwd_t, back_t]), np.vstack([fwd_p, pos[::-1][1:]])
    data = inst.to_dict()
    data["keyframes"] = {"time_s": key_t.tolist(), "pos": key_p.tolist(), "quat": [[1.0, 0.0, 0.0, 0.0]] * key_t.size}
    data["exit_keyframes"] = {"time_s": exit_t.tolist(), "pos": exit_p.tolist(),
                              "quat": [[1.0, 0.0, 0.0, 0.0]] * exit_t.size}
    return HazardInstance.from_dict(data)


def playback(inst):
    """The concatenated playback path (the junction sample once)."""
    frames, exit_ = inst.keyframes, inst.exit_keyframes
    return SampledPath(list(frames.time_s) + list(exit_.time_s[1:]), list(frames.pos) + list(exit_.pos[1:]), None)


# ------------------------------------------------------------------------- 1. a robot on the nominal path

@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", ["slide", "reach"])
def test_robot_on_the_nominal_path_gets_the_old_nominal_release(family, mode, dt):
    outcomes = []
    for with_path in (False, True):
        host, runtime = _world(dt)
        inst = _instance(host, family)
        if with_path:
            inst = with_exit(inst, mode)
        log = W.run_episode(host, runtime, inst, 1.2)
        outcomes.append((log, runtime.episode_info(0), runtime.contact_events(0)))
    (old_log, old, old_events), (new_log, new, new_events) = outcomes
    assert old["release"]["kind"] == new["release"]["kind"] == "nominal"
    assert new["release"]["time_s"] == old["release"]["time_s"]
    assert np.array_equal(new["release"]["lin_vel_mps"], old["release"]["lin_vel_mps"])
    assert np.array_equal(new["release"]["readback_lin_vel_mps"], old["release"]["readback_lin_vel_mps"])
    assert new["phase"] == old["phase"] == RELEASED and new["exit_t_s"] is None and old["exit_t_s"] is None
    assert old_log.first_official_contact_t_s is not None
    assert new_log.first_official_contact_t_s == old_log.first_official_contact_t_s
    assert new_log.released_t_s == old_log.released_t_s
    assert new_log.to_dict() == old_log.to_dict()
    assert [(e.kind, e.time_s, e.partner_name) for e in new_events] == [(e.kind, e.time_s, e.partner_name)
                                                                         for e in old_events]


# ---------------------------------------------------------------------- 2. a robot far from the hit point

def _far_world(dt=0.002):
    far = _task((0.0, 20.0), ((0.0, 0.6, 0.3), (0.0, 0.6, 0.3)))
    return _world(dt, {"far": far})


@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", ["slide", "reach"])
def test_far_robot_never_causes_a_release_and_the_hazard_exits(family, mode, dt):
    host, runtime = _far_world(dt)
    inst = with_exit(_instance(host, family), mode)
    path = playback(inst)
    end = path.t_end_s
    seen = []

    def probe(t):
        info = runtime.episode_info(0)
        seen.append((t, info["phase"], np.array(runtime.hazard_state(0).pos)))
        return FakeAction.follow()

    drained = []   # the phase at every drain of hazard_contacts (one per physics step)
    original = host.hazard_contacts
    host.hazard_contacts = lambda h: (drained.append(runtime.episode_info(0)["phase"]), original(h))[1]
    log = W.run_episode(host, runtime, inst, end + 0.4, task="far", actions=probe)
    info = runtime.episode_info(0)

    assert info["phase"] == EXITED and info["release"] is None and not info["setup_failed"]
    assert abs(info["exit_t_s"] - end) <= 0.5 * dt + 1e-9
    assert log.released_t_s is None and log.first_official_contact_t_s is None
    assert log.kinematic_overlap_count == 0 and runtime.contact_events(0) == []
    phases = [p for _, p, _ in seen]
    assert RELEASED not in phases and phases[0] == PARKED
    assert [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]] == [PARKED, KINEMATIC, EXITED]
    moving = [(t, pos) for t, phase, pos in seen if phase == KINEMATIC]
    assert moving[-1][0] > inst.release_time_s + 0.2          # it kept moving well past the nominal release
    for t, pos in moving:
        np.testing.assert_allclose(pos, path.pos(t), rtol=0, atol=1e-9)
    parked = [pos for t, phase, pos in seen if phase == EXITED]
    assert parked and all(np.array_equal(pos, np.array(W.PARK)) for pos in parked)
    # after the exit the hazard is parked: every post_step drains the contacts, none becomes an event
    substeps = int(round(0.02 / dt))
    assert drained.count(EXITED) >= (len(parked) - 1) * substeps > 0


def test_hazard_in_the_robots_former_way_after_the_exit_produces_no_event():
    host, runtime = _far_world()
    inst = with_exit(W.slide_instance(host), "pass")
    end = playback(inst).t_end_s

    def actions(t):  # after the exit the hand drives through the hazard's whole former path
        return FakeAction.follow() if t < end + 0.1 else W.away((0.0, -1.0, 0.0))

    log = W.run_episode(host, runtime, inst, end + 0.8, task="far", actions=actions)
    assert runtime.episode_info(0)["phase"] == EXITED
    assert runtime.contact_events(0) == [] and log.kinematic_overlap_count == 0
    assert log.first_official_contact_t_s is None and log.first_scene_contact_t_s is None


# ------------------------------------------- 3. away at the hit time, into the hazard during the follow-through

#: hand paths that wait far away at the nominal release and then cross the exit path of the default instances
AMBUSH = {("slide", "pass"): ((0.0, 0.5, 0.8, 20.0), ((0.3, 0.3, 0.3), (0.3, 0.3, 0.3), (0.3, 0.0, 0.3),
                                                      (0.3, 0.0, 0.3))),
          ("reach", "retract"): ((0.0, 0.5, 0.62, 20.0), ((0.08, 0.3, 0.3), (0.08, 0.3, 0.3), (0.08, 0.0, 0.3),
                                                          (0.08, 0.0, 0.3)))}


@pytest.mark.parametrize("family,mode", sorted(AMBUSH))
def test_robot_that_moves_into_the_follow_through_gets_the_rollback_release(family, mode):
    host, runtime = _world(0.002, {"ambush": _task(*AMBUSH[(family, mode)])})
    inst = with_exit(_instance(host, family), mode)
    log = W.run_episode(host, runtime, inst, 1.5, task="ambush")
    info, events = runtime.episode_info(0), runtime.contact_events(0)
    overlaps = [e for e in events if e.kind is EventKind.KINEMATIC_OVERLAP]
    assert overlaps and log.kinematic_overlap_count == len(overlaps)
    assert info["release"]["kind"] == "rollback" and info["phase"] == RELEASED and info["exit_t_s"] is None
    assert info["release"]["rollback_t_s"] > inst.release_time_s          # not the nominal release
    assert log.released_t_s is not None and log.released_t_s > inst.release_time_s
    expected = playback(inst).finite_difference_velocity(info["release"]["rollback_t_s"], W.WINDOW_S)
    np.testing.assert_allclose(info["release"]["lin_vel_mps"], expected, rtol=0, atol=1e-12)
    assert log.first_official_contact_t_s is not None
    assert log.first_official_contact_t_s >= overlaps[-1].time_s - 1e-12


# ------------------------------------------------------------------------------ the engagement threshold

def _static_hand(offset_m):
    """A hand parked ``offset_m`` beyond the nominal hit point along the slide direction."""
    point = (offset_m, 0.0, 0.3)
    return {"parked": _task((0.0, 20.0), (point, point))}


@pytest.mark.parametrize("offset_m,engaged", [(0.005, True), (0.012, False)])
def test_engagement_is_the_latest_separation_against_the_r1_limit(offset_m, engaged):
    cfg = W.runtime_config()
    limit = cfg.max_step_ratio * W.HAZARD_R                      # R1: max_step_ratio x the smallest half extent
    host, runtime = _world(0.002, _static_hand(offset_m))
    inst = with_exit(W.slide_instance(host), "pass")
    W.run_episode(host, runtime, inst, 1.2, task="parked")
    info = runtime.episode_info(0)
    # the last sample before the release step is the hazard 0.002 m before its nominal touching pose
    gap = offset_m + 0.002
    assert (gap <= limit) is engaged
    assert info["release"]["kind"] == ("nominal" if engaged else "rollback")
    assert info["phase"] == RELEASED
    reference_host, reference_runtime = _world(0.002, _static_hand(offset_m))
    W.run_episode(reference_host, reference_runtime, W.slide_instance(reference_host), 1.2, task="parked")
    reference = reference_runtime.episode_info(0)["release"]
    assert reference["kind"] == "nominal"                       # without an exit path the release is unconditional
    if engaged:
        assert info["release"]["time_s"] == reference["time_s"]
        assert np.array_equal(info["release"]["lin_vel_mps"], reference["lin_vel_mps"])
    else:
        # not engaged: the hazard went on along the exit path and met the hand right after the junction, so the
        # rollback window ends 0.01 s after the release time and straddles the junction
        rollback_t = info["release"]["rollback_t_s"]
        assert inst.release_time_s < rollback_t < inst.release_time_s + W.WINDOW_S
        expected = playback(inst).finite_difference_velocity(rollback_t, W.WINDOW_S)
        np.testing.assert_allclose(info["release"]["lin_vel_mps"], expected, rtol=0, atol=1e-12)
        assert any(e.kind is EventKind.KINEMATIC_OVERLAP for e in runtime.contact_events(0))


# ----------------------------------------------------------------------------- capability / old code path

@pytest.mark.parametrize("family", ["slide", "reach"])
def test_follow_through_needs_a_geometric_distance(family):
    host, runtime = _world(host_kwargs={"disable": ("min_distance",)})
    assert host.describe().distance == "surrogate"
    inst = _instance(host, family)
    runtime.arm(0, inst)                                          # without an exit path the instance is fine
    host.reset_episode(0, "static", "empty", 0)
    runtime.end_episode(0)
    with pytest.raises(CapabilityError, match="geometric"):
        runtime.arm(0, with_exit(inst, "pass"))


def test_instance_without_exit_keyframes_keeps_the_old_runtime_path():
    host, runtime = _world()
    inst = W.reach_instance(host)
    assert inst.exit_keyframes is None
    log = W.run_episode(host, runtime, inst, 1.0)
    info = runtime.episode_info(0)
    assert info["release"]["kind"] == "nominal" and info["exit_t_s"] is None
    assert log.released_t_s is not None
    assert dataclasses.replace(inst, exit_keyframes=None) == inst
