"""Follow-through (decision D19) in the pipeline: the builder's exit path, hash stability of instances without it,
G0's R1 check of the exit segment and the G1 Oracle / G2 avoider resuming once the hazard has exited."""
import copy

import numpy as np
import pytest

import pipeline_world as W
from hazardlib.driving import ACTIVE_PHASES
from hazardlib.gates import run_g0, run_g1
from hazardlib.gates.episodes import AvoiderController, OracleController
from hazardlib.instantiate import REJECT_REASONS, InstanceRejected
from hazardlib.kinematics import SampledPath
from hazardlib.runtime import EXITED, KINEMATIC, PARKED, RELEASED
from hazardlib.spec import HazardInstance, HazardSpec

# Hashes of the builder instances below *without* follow_through, ``W.build(spec, task, seed=3).hash()``, computed with
# the unmodified sources of commit a57349c (an exported copy of HEAD, run on both interpreters before any change of
# this task). The builder's float arithmetic differs in the last bits between numpy versions, so the literal depends
# on the numpy of the interpreter (the pure-Python serialisation check is in test_spec_exit_keyframes.py). An
# environment that is not listed skips the check instead of guessing.
HASH_BEFORE_FOLLOW_THROUGH = {
    "2.4.6": {"slide": "a714f8d24325eceb4ae96b73b13ed18d898f1bc2364f5e65862640d56c7716e4",
              "reach": "600f80c9fa52165501260d515219a53a75098b8a890c857ee191ef1c0043b197"},
    "1.24.4": {"slide": "a4f816bad8beb6d636ac71430843fac58e61ba3985debd08cd1a0c2af2a5a8ca",
               "reach": "1e6eba0c5691f5eddb115efc9175c26e8cb0fd1dafc4683ec96f97aba5987b16"},
}
CASES = {"slide": (W.slide_spec, "low"), "reach": (W.reach_spec, "pick")}
DT = W.instantiate_settings().keyframe_dt_s


def with_follow_through(spec, mode):
    data = spec.to_dict()
    data["motion"]["params"]["follow_through"] = mode
    return HazardSpec.from_dict(data)


def build(family, mode=None, seed=3, **spec_kwargs):
    make, task = CASES[family]
    spec = make(**spec_kwargs)
    return W.build(spec if mode is None else with_follow_through(spec, mode), task, seed=seed)


def widened_workspace(inst, task):
    ws = W.task_context(task).workspace_bounds
    h = inst.agent.half_extents_m
    radius = max(h) if inst.agent.shape == "sphere" else float(np.linalg.norm(h))
    return np.array(ws.min_m) - radius, np.array(ws.max_m) + radius, radius


def outside(p, lo, hi):
    return bool(np.any(np.asarray(p) < lo) or np.any(np.asarray(p) > hi))


# --------------------------------------------------------------------------------------- hash stability

@pytest.mark.parametrize("family", ["slide", "reach"])
def test_instances_without_follow_through_hash_as_before(family):
    inst = build(family)
    assert inst.exit_keyframes is None and "exit_keyframes" not in inst.to_dict()
    expected = HASH_BEFORE_FOLLOW_THROUGH.get(np.__version__)
    if expected is None:
        pytest.skip("no recorded pre-change hash for numpy %s" % np.__version__)
    assert inst.hash() == expected[family]


# ------------------------------------------------------------------------------------------- the builder

@pytest.mark.parametrize("family", ["slide", "reach"])
def test_pass_leaves_the_widened_workspace_in_a_straight_line(family):
    inst = build(family, "pass")
    task = CASES[family][1]
    lo, hi, radius = widened_workspace(inst, task)
    exit_, frames = inst.exit_keyframes, inst.keyframes
    assert exit_.time_s[0] == inst.release_time_s == frames.time_s[-1]
    assert exit_.pos[0] == frames.pos[-1] and exit_.quat[0] == frames.quat[-1]
    assert outside(exit_.pos[-1], lo, hi)
    assert all(not outside(p, lo, hi) for p in exit_.pos[:-1])          # the end is the first sample outside
    assert all(q == [1.0, 0.0, 0.0, 0.0] for q in exit_.quat)
    steps = np.diff(exit_.time_s)
    assert np.all(steps > 0) and np.max(np.abs(steps - DT)) < 1e-9 * DT + 1e-12
    velocity = np.array(inst.release_velocity.lin_mps)
    exit_velocity = np.diff(np.array(exit_.pos), axis=0) / steps[:, None]
    np.testing.assert_allclose(exit_velocity, np.tile(velocity, (len(steps), 1)), rtol=0, atol=1e-9)
    record = inst.provenance["follow_through"]
    assert record["mode"] == "pass" and record["workspace_widened_by_m"] == radius
    assert record["exit_duration_s"] == pytest.approx(exit_.time_s[-1] - inst.release_time_s)
    assert record["exit_keyframe_count"] == len(exit_.time_s) and record["keyframe_dt_s"] == DT


def test_pass_end_uses_the_bounding_sphere_of_a_box():
    spec = W.slide_spec()
    data = spec.to_dict()
    data["agent"] = {"kind": "object", "shape": "box", "half_extents_m": [0.03, 0.04, 0.02], "mass_kg": 0.2}
    inst = W.build(with_follow_through(HazardSpec.from_dict(data), "pass"), "low")
    lo, hi, radius = widened_workspace(inst, "low")
    assert radius == pytest.approx(np.linalg.norm([0.03, 0.04, 0.02]))
    assert outside(inst.exit_keyframes.pos[-1], lo, hi) and not outside(inst.exit_keyframes.pos[-2], lo, hi)


@pytest.mark.parametrize("family", ["slide", "reach"])
def test_retract_ends_at_the_start_pose(family):
    inst = build(family, "retract")
    exit_, frames = inst.exit_keyframes, inst.keyframes
    assert exit_.time_s[0] == inst.release_time_s == frames.time_s[-1]
    assert exit_.pos[0] == frames.pos[-1]
    np.testing.assert_allclose(exit_.pos[-1], frames.pos[0], rtol=0, atol=1e-12)
    assert exit_.quat[-1] == frames.quat[0]
    steps = np.diff(exit_.time_s)
    assert np.all(steps > 0) and np.max(steps) <= DT * (1 + 1e-9)
    record = inst.provenance["follow_through"]
    assert record["mode"] == "retract"
    assert record["exit_duration_s"] == pytest.approx(record["continuation_s"] + record["retrace_s"])
    assert record["max_interp_error_m"] <= inst.provenance["keyframes"]["tol_m"]


def test_retract_of_a_slide_retraces_from_the_hit_pose_and_a_reach_continues_first():
    slide, reach = build("slide", "retract"), build("reach", "retract")
    assert slide.provenance["follow_through"]["continuation_s"] == 0.0
    assert slide.provenance["follow_through"]["retrace_s"] == pytest.approx(slide.provenance["tau_s"])
    assert reach.provenance["mapping"]["overshoot_m"] > 0
    assert reach.provenance["follow_through"]["continuation_s"] > 0
    # the reach goes on to the end of its analytic path, hit_point + direction * overshoot, before it turns round
    plan = W.analytic_plan(reach, W.task_context("pick"))
    turn = np.array(plan.path.pos(plan.path.t_end_s))
    distances = np.linalg.norm(np.array(reach.exit_keyframes.pos) - turn, axis=1)
    assert distances.min() < 1e-12
    turn_t = reach.exit_keyframes.time_s[int(np.argmin(distances))]
    assert turn_t == pytest.approx(plan.path.t_end_s)
    # the retrace is the time-reversed primitive path: the pose after the turn at +s is the pose before it at -s
    path = SampledPath(reach.exit_keyframes.time_s, reach.exit_keyframes.pos, None)
    for s in (0.05, 0.1, 0.2, 0.3):
        np.testing.assert_allclose(path.pos(turn_t + s), plan.path.pos(turn_t - s), rtol=0, atol=2e-4)


@pytest.mark.parametrize("mode", ["pass", "retract"])
@pytest.mark.parametrize("family", ["slide", "reach"])
def test_follow_through_changes_nothing_but_the_exit_path(family, mode):
    for kwargs in ({}, {"direction": None}):                  # an open direction is drawn from the random stream
        base = build(family, None, **kwargs)
        inst = build(family, mode, **kwargs)
        a, b = base.to_dict(), inst.to_dict()
        for key in ("keyframes", "exit_keyframes", "provenance"):
            a.pop(key, None), b.pop(key, None)
        assert a == b                                          # onset, hit, release time / velocity, ids, agent
        release = base.release_time_s
        n = len(inst.keyframes.time_s)
        assert inst.keyframes.time_s == [t for t in base.keyframes.time_s if t <= release]
        assert inst.keyframes.pos == base.keyframes.pos[:n]
        pa, pb = dict(base.provenance), dict(inst.provenance)
        assert pb.pop("follow_through")["mode"] == mode
        assert pb.pop("spec_hash") != pa.pop("spec_hash")      # the spec itself differs
        assert pb["keyframes"].pop("count") == n <= pa["keyframes"].pop("count")
        assert pa == pb                                        # standoff, contact point, mapping, direction, ...
        assert pb["standoff_m"] == pa["standoff_m"] and pb["direction_source"] == pa["direction_source"]


@pytest.mark.parametrize("bad", ["sideways", "Pass", "", None, 1, True, ["pass"]])
def test_invalid_follow_through_values_are_rejected(bad):
    with pytest.raises(ValueError, match="follow_through"):
        W.build(with_follow_through(W.slide_spec(), bad), "low")


def test_follow_through_is_rejected_for_a_fall():
    data = W.fall_spec().to_dict()
    data["motion"]["params"]["follow_through"] = "pass"
    with pytest.raises(ValueError, match="not supported for intrusion.fall"):
        W.build(HazardSpec.from_dict(data), "pick")


def test_exit_path_rejection_reason_is_registered():
    assert "exit_path_invalid" in REJECT_REASONS
    assert issubclass(InstanceRejected, Exception)


# ------------------------------------------------------------------------------------------------ G0

def g0(inst, task, ccd):
    host = W.make_host(ccd=ccd)
    return run_g0(inst, W.task_context(task), host.describe(), W.gate_settings().g0)


def faster_exit(inst, factor):
    """The same instance whose exit path is ``factor`` times as long (continuity at the junction kept)."""
    data = inst.to_dict()
    pos = np.array(data["exit_keyframes"]["pos"])
    data["exit_keyframes"]["pos"] = (pos[0] + (pos - pos[0]) * factor).tolist()
    return HazardInstance.from_dict(data)


def test_g0_holds_the_exit_segment_to_the_kinematic_r1_rule():
    inst = build("slide", "pass")
    plain = g0(inst, "low", ccd=False)
    assert plain.passed and plain.checks["r1_check"]["exit_kinematic_exceeds"] is False
    fast = g0(faster_exit(inst, 100.0), "low", ccd=False)
    assert fast.reasons == ["r1_step_ratio"]                  # the approach and the release speed are unchanged
    assert fast.checks["r1_check"]["kinematic_exceeds"] is True
    assert fast.checks["r1_check"]["exit_kinematic_exceeds"] is True
    assert g0(faster_exit(inst, 100.0), "low", ccd=True).checks["r1_check"] == "skipped: BD hazard_ccd is true"


def test_g0_checks_of_an_instance_without_an_exit_path_are_unchanged():
    result = g0(build("slide"), "low", ccd=False)
    assert "exit_kinematic_exceeds" not in result.checks["r1_check"]


@pytest.mark.parametrize("mode", ["pass", "retract"])
def test_g0_ignores_the_exit_path_otherwise(mode):
    base, inst = g0(build("slide"), "low", ccd=True), g0(build("slide", mode), "low", ccd=True)
    assert inst.passed == base.passed and inst.checks == base.checks


# ------------------------------------------------------------------------------- G1 Oracle / G2 avoider

class _Session:
    """Just enough of an EpisodeSession for the controllers: the hazard phase and a hold action."""

    class _Runtime:
        def hazard_state(self, env):
            raise AssertionError("an exited hazard must not be read")

    class _Host:
        def hold_action(self, env):
            return "hold"

        def robot_state(self, env):
            return "robot"

    def __init__(self, phase):
        self._phase, self.runtime, self.host = phase, self._Runtime(), self._Host()

    def phase(self, env):
        return self._phase


class _Policy:
    def act(self, observation):
        return "policy"


def test_exited_is_no_active_hazard_for_the_controllers_and_the_driver():
    assert EXITED not in ACTIVE_PHASES and set(ACTIVE_PHASES) == {KINEMATIC, RELEASED}
    oracle = OracleController(0.05, 0.1, [-1, -1, -1], [1, 1, 1])
    assert oracle.decide(_Session(KINEMATIC), 0, 0.0, {}, _Policy(), 0.02) == "hold"   # still held while it moves
    assert oracle.decide(_Session(EXITED), 0, 0.0, {}, _Policy(), 0.02) == "policy"    # resumed at once, no settle
    assert oracle.decide(_Session(PARKED), 0, 0.0, {}, _Policy(), 0.02) == "policy"

    seen = []

    class Avoider:
        def filter_action(self, action, robot_state, hazard_state):
            seen.append(hazard_state)
            return action

    avoider = AvoiderController(Avoider())
    assert avoider.decide(_Session(EXITED), 0, 0.0, {}, _Policy(), 0.02) == "policy"
    assert seen == [None]                                     # no hazard state is handed over after the exit


@pytest.mark.parametrize("mode", ["pass", "retract"])
def test_oracle_resumes_after_the_exit_and_the_task_completes(mode):
    host = W.make_host()
    inst = build("slide", mode, seed=0)
    exit_end = inst.exit_keyframes.time_s[-1]
    run = run_g1(W.session(host), inst, W.PathFollower(host), W.task_context("low"), W.gate_settings(), 0, 0)
    (episode,) = run.episodes
    assert run.passed and episode["success"] is True and episode["official_contacts"] == 0
    controller = episode["controller"]
    assert controller["pauses"] == 1
    # held while the hazard moved kinematically, i.e. until it exited, and not a settle period longer
    assert abs(controller["paused_s"] - (exit_end - inst.onset_time_s)) <= 3 * W.CONTROL_DT
    # the Oracle did not wait for a RELEASED hazard that never came: the success time is just the paused time later
    nominal_success = W.task_context("low").phases[-1].t_end_s
    assert episode["success_t_s"] is not None
    assert episode["success_t_s"] == pytest.approx(nominal_success + controller["paused_s"], abs=3 * W.CONTROL_DT)
    assert episode["success_t_s"] > exit_end


def test_oracle_pauses_less_for_a_retracting_hazard_than_for_one_that_passes():
    paused = {}
    for mode in ("pass", "retract"):
        host = W.make_host()
        run = run_g1(W.session(host), build("slide", mode, seed=0), W.PathFollower(host), W.task_context("low"),
                     W.gate_settings(), 0, 0)
        paused[mode] = run.episodes[0]["controller"]["paused_s"]
    assert paused["retract"] < paused["pass"]
