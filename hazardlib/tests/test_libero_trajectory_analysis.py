"""Scientific boundaries of the all-demonstration analysis, without a simulator."""
import numpy as np
import pytest

from hazardhosts.libero.trajectory_stats import first_sustained, local_positions, phases_from_events
from hazardhosts.libero.trajectory_candidates import proposals
from hazardlib.config import load_config
from pathlib import Path


AGENT = dict(kind="object", shape="box", half_extents_m=[.02]*3, mass_kg=.05)
TOP, REACH = .9, 2 * .02 + .02


def SUPPORT(point, t_s):
    """Stand-in of ``slide_support_rule`` (the builder's rule has its own consistency test): finger point over a
    table of top 0.9 m within ``2 h_z + r_eef`` of it."""
    return 0. <= point[2] - TOP <= REACH


def flat(proposed):
    """{(family, target): (candidates, info)} of a proposals() generator."""
    return {(f, t): (c, i) for f, t, c, i in proposed}


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
    result = flat(proposals(scene, summary, row, arrays, {"gripper0_leftfinger": [[0., 0., -.08]]},
                            settings, dict(min_m=[-1., -1., .9], max_m=[1., 1., 1.5]), .2, SUPPORT))
    for family in ["intrusion.slide", "intrusion.reach"]:
        candidates, _ = result[(family, 0.75)]
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
    result = {k: c for k, (c, _) in flat(proposals(*args, settings, workspace, .2, SUPPORT)).items()}
    for family, mode in (("intrusion.slide", "pass"), ("intrusion.reach", "retract")):
        cell = result[(family, 0.75)]
        assert cell and all(s["motion"]["params"]["follow_through"] == mode for s, _ in cell)
    cell = result[("intrusion.fall", 0.75)]
    assert cell and all("follow_through" not in s["motion"]["params"] for s, _ in cell)
    # No mapping, no follow-through; the mapping is not defaulted in code.
    bare = flat(proposals(*args, dict(settings, follow_through={}), workspace, .2, SUPPORT))
    assert all("follow_through" not in s["motion"]["params"] for c, _ in bare.values() for s, _ in c)
    # The generation hash (the resume guard) depends on it.
    from hazardlib.hashing import sha256_hex
    assert sha256_hex(settings) != sha256_hex(dict(settings, follow_through={}))


def sigma_fixture(gaps=None, n=61, t_phase=(0., 3.)):
    """A 3 s approach phase sampled at 20 Hz; the finger hovers 2 cm above the table unless ``gaps`` says."""
    times = np.arange(n) * .05
    low = np.full(n, .92) if gaps is None else .9 + np.asarray(gaps)
    arrays = dict(time_s=times, object_quat=np.tile([1., 0., 0., 0.], (n, 1, 1)),
                  eef_pos=np.tile([0., 0., 1.], (n, 1)), neighbor_clearance=np.full(n, .01),
                  gripper_low_z=low, surface_z=np.asarray(.9),
                  body_quat=np.tile([1., 0., 0., 0.], (n, 1, 1)),
                  body_pos=np.stack([np.zeros(n), np.zeros(n), low + .08], axis=1)[:, None, :])
    scene = dict(control_dt_s=.05, target_index=0, gravity_mps2=9.81, agent=AGENT,
                 robot_body_names=["gripper0_leftfinger"])
    phase = dict(name="approach", start_index=0, end_index=n - 1, duration_s=float(times[-1]), t_start_s=0.,
                 t_end_s=float(times[-1]))
    summary = dict(phases={"approach": dict(candidate_speed_levels_mps=[.2, .3, .4])})
    return scene, summary, dict(demo="demo_0", replay_features=dict(phases=[phase])), arrays, \
        {"gripper0_leftfinger": [[0., 0., -.08]]}


def test_slide_anchors_are_only_where_the_finger_is_low_enough_for_the_builder():
    settings = load_config("trajectory_analysis", strict=True,
                           config_dir=Path(__file__).parents[2] / "configs/hazard/pilot/libero")
    # reach = 2 * 0.02 + 0.02 = 0.06 m above the table top; the low-gap half of the phase is NOT enough
    gaps = np.where(np.arange(61) % 2 == 0, .5, .03)        # every other sample 0.5 m high
    gaps[:10] = .02                                          # early samples valid
    gaps[10:20] = .5
    scene, summary, row, arrays, geometry = sigma_fixture(gaps)
    workspace = dict(min_m=[-1., -1., .9], max_m=[1., 1., 1.5])
    result = flat(proposals(scene, summary, row, arrays, geometry, dict(settings, anchor_quantiles=[0., .5, 1.]),
                            workspace, .3, SUPPORT))
    candidates, info = result[("intrusion.slide", 1.5)]
    anchors = {e["anchor_step"] for _, e in candidates}
    assert anchors and all(gaps[a] <= REACH for a in anchors)
    # The unsupported samples would have been the "low half" had the filter not applied first.
    high = np.full(61, .5)
    scene, summary, row, arrays, geometry = sigma_fixture(high)
    result = flat(proposals(scene, summary, row, arrays, geometry, settings, workspace, .3, SUPPORT))
    candidates, info = result[("intrusion.slide", 1.5)]
    assert candidates == [] and info["unreachable"] == "no_supported_anchor"
    with pytest.raises(ValueError, match="slide_support"):
        list(proposals(scene, summary, row, arrays, geometry, settings, workspace, .3))


def test_tau_follows_sigma_times_t_stop_rounded_down_and_never_below_the_minimum_lead():
    from hazardhosts.libero.trajectory_candidates import tau_for_sigma
    dt = .05
    assert tau_for_sigma(1.5, .3, dt, 2) == (pytest.approx(.45), None)           # 0.45 / 0.05 = 8.999999...
    assert tau_for_sigma(3.0, .3, dt, 2)[0] == pytest.approx(.9)
    tau, _ = tau_for_sigma(.75, .3, dt, 2)                                        # 0.225 -> 4 steps
    assert tau == pytest.approx(.2) and tau / .3 < .75                            # achieved Sigma <= target
    assert tau_for_sigma(1.5, .33, dt, 2)[0] == pytest.approx(.45)                # 0.495 -> 9 steps, down
    assert tau_for_sigma(.3, .3, dt, 2) == (None, "below_minimum_lead")           # 1 step < 2
    assert tau_for_sigma(.5, .3, dt, 2)[0] == pytest.approx(.15)


def test_sigma_targets_give_one_tau_each_and_unreachable_ones_are_named():
    settings = load_config("trajectory_analysis", strict=True,
                           config_dir=Path(__file__).parents[2] / "configs/hazard/pilot/libero")
    assert settings["sigma_targets"] == [.75, 1.5, 3.0]
    assert settings["max_validated_per_family_layout"] == len(settings["sigma_targets"])
    assert "tau_phase_fractions" not in settings
    scene, summary, row, arrays, geometry = sigma_fixture()
    workspace = dict(min_m=[-1., -1., .9], max_m=[1., 1., 1.5])
    result = flat(proposals(scene, summary, row, arrays, geometry, settings, workspace, .3, SUPPORT))
    for family in settings["families"]:
        assert sorted(t for f, t in result if f == family) == [.75, 1.5, 3.0]
    for family in ("intrusion.reach", "intrusion.slide"):
        for target, tau in ((.75, .2), (1.5, .45), (3.0, .9)):
            candidates, info = result[(family, target)]
            assert info["unreachable"] is None and info["tau_s"] == pytest.approx(tau)
            assert candidates and all(s["intensity"]["tau_s"] == pytest.approx(tau) and
                                      e["sigma_target"] == target and e["tau_s"] == pytest.approx(tau)
                                      for s, e in candidates)
            assert all(e["tau_s"] <= e["available_before_hit_s"] + 1e-9 for _, e in candidates)
    # Fall: free height = 1.5 - 1.0 - 0.02 = 0.48 m -> at most sqrt(2 * 0.48 / 9.81) = 0.31 s of free fall.
    assert result[("intrusion.fall", .75)][1]["unreachable"] is None
    assert result[("intrusion.fall", 1.5)][1]["unreachable"] == "free_fall_height"
    assert result[("intrusion.fall", 3.0)][0] == []
    # Available time: an anchor 0.25 s into the phase cannot carry tau = 0.9 s.
    scene, summary, row, arrays, geometry = sigma_fixture(n=6)
    row["replay_features"]["phases"][0].update(end_index=5, duration_s=.25, t_end_s=.25)
    result = flat(proposals(scene, summary, row, arrays, geometry, settings, workspace, .3, SUPPORT))
    assert result[("intrusion.reach", 3.0)][1]["unreachable"] == "phase_time"
    # A short T_stop makes the smallest target fall below the minimum lead (2 steps = 0.1 s).
    result = flat(proposals(scene, summary, row, arrays, geometry, settings, workspace, .1, SUPPORT))
    assert result[("intrusion.reach", .75)][1]["unreachable"] == "below_minimum_lead"
    assert result[("intrusion.reach", .75)][1]["tau_s"] is None
    # Censored / failed protocol: no Sigma is targeted at all.
    result = flat(proposals(scene, summary, row, arrays, geometry, settings, workspace, None, SUPPORT))
    assert len(result) == 9 and all(c == [] and i["unreachable"] == "t_stop_censored" for c, i in result.values())


def test_candidate_budget_applies_per_target():
    settings = load_config("trajectory_analysis", strict=True,
                           config_dir=Path(__file__).parents[2] / "configs/hazard/pilot/libero")
    scene, summary, row, arrays, geometry = sigma_fixture()
    workspace = dict(min_m=[-1., -1., .9], max_m=[1., 1., 1.5])
    result = flat(proposals(scene, summary, row, arrays, geometry, dict(settings, max_candidates_per_family_layout=5),
                            workspace, .3, SUPPORT))
    for family in ("intrusion.reach", "intrusion.slide"):
        for target in settings["sigma_targets"]:
            candidates, info = result[(family, target)]
            assert len(candidates) == 5 and info["n_proposed"] > 5
            assert len({s["id"] for s, _ in candidates}) == 5
    ids = [s["id"] for c, _ in result.values() for s, _ in c]
    assert len(ids) == len(set(ids))
