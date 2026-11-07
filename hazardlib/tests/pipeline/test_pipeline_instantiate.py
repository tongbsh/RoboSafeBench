"""HazardSpec + TaskContext -> HazardInstance (T06 item 2; acceptance 3, 9, 10, 11, 14)."""
import dataclasses
import random

import numpy as np
import pytest

import pipeline_world as W
from hazardlib.config import ConfigError
from hazardlib.instantiate import (InstanceRejected, InstantiateSettings, fall_release_height_m, instantiate,
                                   reach_geometry, slide_start_distance_m)
from hazardlib.kinematics import SampledPath
from hazardlib.spec import TaskContext

analytic_plan = W.analytic_plan  # re-solves the instance with T02's build_hit_path from what it records

CASES = [(lambda: W.fall_spec(), "pick"), (lambda: W.slide_spec(), "low"), (lambda: W.reach_spec(), "pick")]


@pytest.mark.parametrize("make_spec,task", CASES, ids=["fall", "slide", "reach"])
def test_instance_is_a_pure_deterministic_function(make_spec, task):
    np.random.seed(123)
    random.seed(123)
    np_state, py_state = np.random.get_state(), random.getstate()
    a = W.build(make_spec(), task, seed=7)
    b = W.build(make_spec(), task, seed=7)
    assert a.to_dict() == b.to_dict() and a.hash() == b.hash()
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) and after[2:] == np_state[2:]
    assert random.getstate() == py_state
    assert W.build(make_spec(), task, seed=8).hash() != a.hash()


def test_rng_stream_depends_on_spec_tc_and_seed_only():
    spec = W.fall_spec(tau=[0.1, 0.15, 0.2, 0.25, 0.3])
    taus = {s: W.build(spec, "pick", seed=s).provenance["tau_s"] for s in range(12)}
    assert len(set(taus.values())) > 1
    assert all(W.build(spec, "pick", seed=s).provenance["tau_s"] == taus[s] for s in (0, 5, 11))
    open_direction = W.slide_spec(direction=None)
    d0 = W.build(open_direction, "low", seed=0).provenance["direction_source"]
    assert d0 == "rng"


def test_instance_describes_itself_and_records_inputs():
    spec = W.reach_spec()
    inst = W.build(spec, "pick", seed=3)
    assert (inst.spec_id, inst.family, inst.agent.to_dict()) == (spec.id, spec.family, spec.agent.to_dict())
    prov = inst.provenance
    tc = W.task_context("pick")
    assert prov["spec_hash"] == spec.hash() and prov["task_context_hash"] == tc.hash()
    assert prov["bd_hash"] == W.make_host().describe().hash()
    assert (prov["task_id"], prov["layout_id"], prov["instance_seed"]) == ("pick", "L0", 3)
    assert inst.instance_id == "reach.pick.L0.s3"
    assert prov["hazard_quat_wxyz"] == [1.0, 0.0, 0.0, 0.0] and "identity" in prov["hazard_quat_source"]
    assert all(q == [1.0, 0.0, 0.0, 0.0] for q in inst.keyframes.quat)
    assert prov["release_window_s"] == W.instantiate_settings().release_velocity_window_s


@pytest.mark.parametrize("make_spec,task", CASES, ids=["fall", "slide", "reach"])
def test_standoff_converts_the_contact_point_to_the_centroid(make_spec, task):
    inst = W.build(make_spec(), task)
    prov = inst.provenance
    assert prov["standoff_m"] == pytest.approx(W.STANDOFF)
    offset = np.array(inst.nominal_hit_point) - np.array(prov["anchor_point"])
    assert np.linalg.norm(offset) == pytest.approx(W.STANDOFF, abs=1e-9)
    assert np.allclose(offset / np.linalg.norm(offset), prov["contact_normal"], atol=1e-9)
    nominal = SampledPath.from_pose_sequence(W.task_context(task).nominal_ee_path)
    assert np.allclose(prov["anchor_point"], nominal.pos(inst.nominal_hit_time_s), atol=1e-12)


def test_box_agent_standoff_uses_the_support_radius_along_the_normal():
    spec = W.fall_spec()
    data = spec.to_dict()
    data["agent"] = {"kind": "object", "shape": "box", "half_extents_m": [0.05, 0.04, 0.02], "mass_kg": 0.2}
    inst = W.build(type(spec).from_dict(data), "pick")
    assert inst.provenance["standoff_m"] == pytest.approx(0.02 + W.HAND_R)


@pytest.mark.parametrize("make_spec,task", CASES, ids=["fall", "slide", "reach"])
def test_keyframe_replay_matches_the_analytic_path(make_spec, task):
    settings = W.instantiate_settings()
    inst = W.build(make_spec(), task)
    plan = analytic_plan(inst, W.task_context(task))
    times = np.array(inst.keyframes.time_s)
    assert np.allclose(np.diff(times)[np.diff(times) > settings.keyframe_dt_s * 0.999].max(), settings.keyframe_dt_s)
    replay = SampledPath(times, inst.keyframes.pos, None)
    dense = np.linspace(times[0], times[-1], 4001)
    error = np.max(np.linalg.norm(replay.pos(dense) - plan.path.pos(dense), axis=1))
    assert error <= settings.keyframe_interp_tol_m
    assert inst.provenance["keyframes"]["max_interp_error_m"] <= settings.keyframe_interp_tol_m
    assert inst.nominal_hit_time_s in list(times)


@pytest.mark.parametrize("family", ["fall", "slide", "reach"])
def test_keyframes_have_no_round_off_intervals(family):
    make, task = {"fall": (W.fall_spec, "pick"), "slide": (W.slide_spec, "low"), "reach": (W.reach_spec, "pick")}[family]
    dt = W.instantiate_settings().keyframe_dt_s
    for frac in (0.05, 0.06, 0.07, 0.5, 0.75):      # reach at 0.06: t_hit lands one ulp off a grid point
        for tau in (0.1, 0.15, 0.2, 0.25, 0.3):
            try:
                inst = W.build(make(tau=tau, frac=frac), task)
            except InstanceRejected:
                continue
            times = np.array(inst.keyframes.time_s)
            assert np.diff(times).min() > 1e-6 * dt, (frac, tau)
            for instant in (inst.nominal_hit_time_s, inst.release_time_s):
                assert instant in list(times) or not times[0] < instant < times[-1]


@pytest.mark.parametrize("make_spec,task", CASES[1:], ids=["slide", "reach"])
def test_keyframe_release_velocity_equals_the_fixed_window_difference(make_spec, task):
    settings = W.instantiate_settings()
    inst = W.build(make_spec(), task)
    plan = analytic_plan(inst, W.task_context(task))
    window = settings.release_velocity_window_s
    expected = plan.path.finite_difference_velocity(inst.release_time_s, window)
    assert np.allclose(inst.release_velocity.lin_mps, expected, atol=1e-12)
    replay = SampledPath(inst.keyframes.time_s, inst.keyframes.pos, None)
    from_keyframes = replay.finite_difference_velocity(inst.release_time_s, window)
    assert np.allclose(from_keyframes, expected, atol=2.0 * settings.keyframe_interp_tol_m / window)
    assert np.allclose(inst.provenance["keyframe_release_velocity_mps"], from_keyframes)


def test_too_coarse_keyframes_are_rejected():
    settings = dataclasses.replace(W.instantiate_settings(), keyframe_dt_s=0.05, keyframe_interp_tol_m=1e-4)
    with pytest.raises(InstanceRejected) as info:
        W.build(W.fall_spec(), "pick", settings=settings)
    assert info.value.reason == "keyframe_interp_error"


def test_fall_release_height_uses_bd_gravity():
    for tau, height in ((0.4, 0.78), (0.5, 1.23), (1.0, 4.9)):
        assert fall_release_height_m(tau, 9.81) == pytest.approx(height, abs=0.01)
    inst = W.build(W.fall_spec(tau=0.2), "pick")
    release_z = inst.keyframes.pos[0][2]
    assert release_z - inst.nominal_hit_point[2] == pytest.approx(W.fall_height(0.2), abs=1e-12)
    assert inst.release_time_s == inst.onset_time_s and inst.release_velocity.lin_mps == [0.0, 0.0, 0.0]
    moon = W.make_host(gravity_mps2=1.62)
    inst_moon = W.build(W.fall_spec(tau=0.2), "pick", host=moon)
    assert inst_moon.keyframes.pos[0][2] - inst_moon.nominal_hit_point[2] == pytest.approx(W.fall_height(0.2, 1.62))


def test_reach_with_zero_overshoot_is_rejected():
    zero = dataclasses.replace(W.instantiate_settings(), reach_overshoot_fraction=0.0)
    with pytest.raises(InstanceRejected) as info:
        W.build(W.reach_spec(), "pick", settings=zero)
    assert info.value.reason == "reach_zero_overshoot"
    inst = W.build(W.reach_spec(speed=0.6, tau=0.3), "pick")
    mapping = inst.provenance["mapping"]
    assert mapping["overshoot_m"] > 0
    assert mapping["overshoot_m"] / mapping["reach_distance_m"] == pytest.approx(0.25)
    assert inst.provenance["hit_diagnostics"]["hit_speed_mps"] == pytest.approx(0.6, rel=1e-9)


def test_reach_mapping_linear_profile():
    g = reach_geometry(0.8, 0.25, 0.25, "linear")
    assert g.reach_distance_m == pytest.approx(0.8 * 0.25)
    assert g.overshoot_m == pytest.approx(0.25 * 0.8 * 0.25)
    assert slide_start_distance_m(0.5, 0.3) == pytest.approx(0.15)


def test_slide_runs_on_the_support_surface():
    inst = W.build(W.slide_spec(), "low")
    pos = np.array(inst.keyframes.pos)
    top = inst.provenance["support_surface"]["top_m"]
    assert np.all(np.abs(pos[:, 2] - (top + W.HAZARD_R)) <= 1e-12)
    assert inst.release_velocity.lin_mps[2] == pytest.approx(0.0, abs=1e-12)
    assert inst.provenance["support_surface"]["source"] == "fakehost_config"
    assert inst.provenance["mapping"]["start_distance_m"] == pytest.approx(0.15)


def test_slide_without_a_low_enough_phase_is_rejected():
    with pytest.raises(InstanceRejected) as info:
        W.build(W.slide_spec(), "pick")                 # the EEF stays 0.3 m above the table
    assert info.value.reason == "no_support_surface"
    tc = W.task_context("low")
    bare = TaskContext.from_dict(dict(tc.to_dict(), support_surfaces=[]))
    with pytest.raises(InstanceRejected) as info:
        W.build(W.slide_spec(), "low", tc=bare)
    assert info.value.reason == "no_support_surface"


def test_intensity_outside_the_registered_range_is_a_spec_error():
    with pytest.raises(ValueError, match="registered range"):
        W.build(W.fall_spec(tau=2.0), "pick")


def test_instantiate_settings_refuse_placeholders_of_the_repository_config():
    with pytest.raises(ConfigError, match="overshoot_fraction"):
        InstantiateSettings.load(W.REPO_CONFIG_DIR)


def test_instantiate_rejects_wrong_inputs():
    tc = W.task_context("pick")
    bd = W.make_host().describe()
    with pytest.raises(ValueError, match="seed"):
        instantiate(W.fall_spec(), tc, 1.5, bd, W.instantiate_settings())
    with pytest.raises(ValueError, match="anchor phase"):
        instantiate(W.fall_spec(phase="lift"), tc, 0, bd, W.instantiate_settings())
