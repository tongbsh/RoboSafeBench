"""Scientific boundaries of the all-demonstration analysis, without a simulator."""
import numpy as np

from hazardhosts.libero.trajectory_stats import first_sustained, local_positions, phases_from_events
from hazardhosts.libero.trajectory_candidates import proposals
from hazardlib.config import load_config
from pathlib import Path


def test_a_single_contact_spike_is_not_a_confirmed_grasp():
    assert first_sustained([False, True, False, True, True, True], 3) == 3
    assert first_sustained([True, True, False], 3) is None


def test_layout_translation_and_rotation_preserve_object_relative_path():
    path = np.array([[0., 0., 0.], [1., 0., 1.]])
    world = np.array([[5., 3., 2.], [5., 4., 3.]])
    result = local_positions(world, [5., 3., 2.], [np.sqrt(.5), 0., 0., np.sqrt(.5)])
    np.testing.assert_allclose(result, path, atol=1e-12)


def test_lift_before_confirmed_contact_does_not_reverse_phase_time():
    result = phases_from_events(np.arange(8) * .05, [-1, -1, -1, 0, 0, 0, -1, -1],
                                [0., 0., .02, .02, .02, .02, .02, .02], 0, .01, 3)
    assert result["lift_index"] == 2
    assert [p["name"] for p in result["phases"]] == ["approach", "transport"]
    assert all(p["t_end_s"] > p["t_start_s"] for p in result["phases"])


def test_ungrasped_stationary_object_does_not_get_a_carry_phase():
    result = phases_from_events(np.arange(8) * .05, [-1] * 8, [1.] * 8, 0, .01, 3)
    assert [p["name"] for p in result["phases"]] == ["approach"]


def test_scene_conditioned_proposals_cover_directions_and_use_observed_risk_samples():
    settings = load_config("trajectory_analysis", strict=True,
                           config_dir=Path(__file__).parents[2] / "configs/hazard/pilot/libero")
    times = np.arange(21) * .05
    phase = dict(name="approach", start_index=0, end_index=20, duration_s=1., t_start_s=0., t_end_s=1.)
    # Separated low-clearance samples test that a time quantile never creates
    # an anchor in an unobserved gap between risk samples.
    gaps = np.ones(21)
    gaps[[3, 4, 5, 14, 15, 16, 17, 18, 19, 20]] = .01
    arrays = dict(time_s=times, object_quat=np.tile([.999, .02, 0., 0.], (21, 1, 1)),
                  eef_pos=np.tile([0., 0., 1.], (21, 1)), neighbor_clearance=gaps,
                  gripper_low_z=np.full(21, .92), surface_z=np.asarray(.9),
                  body_quat=np.tile([1., 0., 0., 0.], (21, 1, 1)),
                  body_pos=np.tile([0., 0., 1.], (21, 1, 1)))
    scene = dict(control_dt_s=.05, target_index=0, gravity_mps2=9.81,
                 agent=dict(kind="object", shape="box", half_extents_m=[.02]*3, mass_kg=.05),
                 robot_body_names=["gripper0_leftfinger"])
    summary = dict(phases={"approach": dict(candidate_speed_levels_mps=[.2, .3, .4])})
    row = dict(demo="demo_0", replay_features=dict(phases=[phase]))
    result = dict(proposals(scene, summary, row, arrays, {"gripper0_leftfinger": [[0., 0., -.08]]},
                            settings, dict(min_m=[-1., -1., .9], max_m=[1., 1., 1.5])))
    for family in ["intrusion.slide", "intrusion.reach"]:
        candidates = result[family]
        assert {e["direction_index"] for _, e in candidates} == set(range(8))
        assert all(s["motion"]["params"]["direction"][2] == 0 for s, _ in candidates)
        if family == "intrusion.reach":
            assert all(gaps[e["anchor_step"]] == .01 for _, e in candidates)
        assert all(s["intensity"]["approach_speed_mps"] == .3 for s, _ in candidates)


def test_follow_through_comes_from_the_protocol_and_only_for_the_families_listed():
    settings = load_config("trajectory_analysis", strict=True,
                           config_dir=Path(__file__).parents[2] / "configs/hazard/pilot/libero")
    assert settings["follow_through"] == {"intrusion.slide": "pass", "intrusion.reach": "retract"}
    times = np.arange(21) * .05
    phase = dict(name="approach", start_index=0, end_index=20, duration_s=1., t_start_s=0., t_end_s=1.)
    arrays = dict(time_s=times, object_quat=np.tile([1., 0., 0., 0.], (21, 1, 1)),
                  eef_pos=np.tile([0., 0., 1.], (21, 1)), neighbor_clearance=np.full(21, .01),
                  gripper_low_z=np.full(21, .92), surface_z=np.asarray(.9),
                  body_quat=np.tile([1., 0., 0., 0.], (21, 1, 1)), body_pos=np.tile([0., 0., 1.], (21, 1, 1)))
    scene = dict(control_dt_s=.05, target_index=0, gravity_mps2=9.81,
                 agent=dict(kind="object", shape="box", half_extents_m=[.02]*3, mass_kg=.05),
                 robot_body_names=["gripper0_leftfinger"])
    summary = dict(phases={"approach": dict(candidate_speed_levels_mps=[.2, .3, .4])})
    row = dict(demo="demo_0", replay_features=dict(phases=[phase]))
    args = (scene, summary, row, arrays, {"gripper0_leftfinger": [[0., 0., -.08]]})
    workspace = dict(min_m=[-1., -1., .9], max_m=[1., 1., 1.5])
    result = dict(proposals(*args, settings, workspace))
    for family, mode in (("intrusion.slide", "pass"), ("intrusion.reach", "retract")):
        assert result[family] and all(s["motion"]["params"]["follow_through"] == mode for s, _ in result[family])
    assert result["intrusion.fall"] and all("follow_through" not in s["motion"]["params"] for s, _ in result["intrusion.fall"])
    # No mapping, no follow-through; the mapping is not defaulted in code.
    bare = dict(proposals(*args, dict(settings, follow_through={}), workspace))
    assert all("follow_through" not in s["motion"]["params"] for fam in bare.values() for s, _ in fam)
    # The generation hash (the resume guard) depends on it.
    from hazardlib.hashing import sha256_hex
    assert sha256_hex(settings) != sha256_hex(dict(settings, follow_through={}))
