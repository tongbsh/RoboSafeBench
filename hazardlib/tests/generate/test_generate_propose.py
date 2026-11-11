"""Core proposals on a synthetic V1 / V2: 3D entry directions, trajectory-intersection coverage, Sigma / tau,
budgets, prefilter and settings (no simulator)."""
import math

import numpy as np
import pytest

import generate_synthetic as S
from hazardlib.config import ConfigError
from hazardlib.generate import (CoverageIndex, GenerateSettings, cell_id, layout_probes, propose, sweep,
                                tau_for_sigma)
from hazardlib.generate.propose import (entry_direction, finger_table_gap, phase_speed_levels,
                                        robot_neighbour_clearance, thin_points)

FAMILIES = ("intrusion.reach", "intrusion.fall", "intrusion.slide")


def run(ts, settings=None, demo="demo_0", t_stop=.3, **kwargs):
    gs = settings or S.settings()
    cells = list(propose(ts.trajectories[int(demo.split("_")[1])], ts, gs, S.WORKSPACE, t_stop, S.AGENT, 9.81,
                         slide_support=S.support, **kwargs))
    return {(c.family, c.elevation_rad, c.sigma): c for c in cells}


def test_pilot_settings_load_and_are_validated():
    gs = S.settings()
    assert gs.entry_directions["intrusion.reach"].elevations_rad == (0.0, 0.5236, 1.0472)
    assert gs.entry_directions["intrusion.slide"].elevations_rad == (0.0,)
    assert gs.elevations("intrusion.fall") == (None,)
    assert gs.coverage_radius_m == .055
    raw = S.raw_settings()
    with pytest.raises(ConfigError, match="coverage_radius_m"):
        GenerateSettings.from_mapping({k: v for k, v in raw.items() if k != "coverage_radius_m"})
    bad = dict(raw, entry_directions={"intrusion.reach": {"azimuth_count": 8, "elevations_rad": [0.0]},
                                      "intrusion.slide": {"azimuth_count": 8, "elevations_rad": [0.0, 0.5]}})
    with pytest.raises(ConfigError, match="slide"):
        GenerateSettings.from_mapping(bad)
    with pytest.raises(ConfigError, match="elevations_rad"):
        GenerateSettings.from_mapping(dict(raw, entry_directions={
            "intrusion.reach": {"azimuth_count": 8, "elevations_rad": [1.6]},
            "intrusion.slide": {"azimuth_count": 8, "elevations_rad": [0.0]}}))


def test_no_numeric_default_in_code_every_threshold_is_in_the_yaml():
    raw = S.raw_settings()
    for key in ("coverage_radius_m", "thinned_point_spacing_m", "step_tolerance_fraction", "risk_search_m",
                "layout_prefilter_slack_m", "entry_directions", "direction_frame", "coverage_ordering"):
        assert key in raw


def test_tau_follows_sigma_times_t_stop_rounded_down():
    dt, tol = .05, 1e-6
    assert tau_for_sigma(1.5, .3, dt, 2, tol) == (pytest.approx(.45), None)
    assert tau_for_sigma(.75, .3, dt, 2, tol)[0] == pytest.approx(.2)
    assert tau_for_sigma(.3, .3, dt, 2, tol) == (None, "below_minimum_lead")


def test_entry_directions_are_3d_unit_and_move_downward_from_above():
    for el in (0.0, .5236, 1.0472):
        for i in range(8):
            d = entry_direction(.4, i, 8, el)
            assert np.isclose(np.linalg.norm(d), 1.0)
            assert np.isclose(d[2], -math.sin(el))
            assert np.isclose(math.atan2(d[1], d[0]), (.4 + 2 * math.pi * i / 8 + math.pi) % (2 * math.pi) - math.pi)
    assert entry_direction(.4, 0, 8, 0.0)[2] == 0.0           # exactly horizontal (the builder checks z == 0)


def test_cells_cover_family_elevation_and_sigma_with_the_entity_frame_azimuths():
    yaw = math.radians(30)
    ts = S.task_set(n_demos=3, yaw=yaw)
    result = run(ts)
    keys = {(f, e) for f, e, _ in result}
    assert keys == {("intrusion.reach", 0.0), ("intrusion.reach", .5236), ("intrusion.reach", 1.0472),
                    ("intrusion.fall", None), ("intrusion.slide", 0.0)}
    assert all(len([k for k in result if k[:2] == key]) == 3 for key in keys)     # three Sigma targets each
    for (family, el, sigma), cell in result.items():
        assert cell.info["cell_id"] == cell_id(family, el)
        if family == "intrusion.fall" and sigma > .75:
            continue                                     # free-fall height budget (see the unreachable test)
        assert cell.candidates and cell.info["unreachable"] is None
        for spec, evidence in cell.candidates:
            params = spec["motion"]["params"]
            if family == "intrusion.fall":
                assert "direction" not in params and "approach_speed_mps" not in spec["intensity"]
                continue
            d = np.array(params["direction"])
            assert np.isclose(d[2], -math.sin(el or 0.0)) and np.isclose(np.linalg.norm(d), 1.0)
            azimuth = (math.atan2(d[1], d[0]) - yaw) % (2 * math.pi)
            assert any(np.isclose(azimuth, 2 * math.pi * i / 8) or np.isclose(azimuth, 2 * math.pi * i / 8 - 2 * math.pi)
                       for i in range(8))
            assert evidence["elevation_rad"] == el and evidence["direction_index"] in range(8)
            if family == "intrusion.slide":
                assert d[2] == 0.0
    ids = [s["id"] for c in result.values() for s, _ in c.candidates]
    assert len(ids) == len(set(ids))
    assert all(".e" not in s["id"] for s, _ in result[("intrusion.reach", 0.0, .75)].candidates)
    assert all(s["id"].endswith(".e1") for s, _ in result[("intrusion.reach", .5236, .75)].candidates)
    assert all(s["id"].endswith(".e2") for s, _ in result[("intrusion.reach", 1.0472, .75)].candidates)


def test_direction_frame_option_uses_the_first_entity_or_the_segment_entity():
    ts = S.task_set(n_demos=2, yaw=math.radians(30))
    t = ts.trajectories[0]
    t.segments[1].entity = t.segments[0].entity = "bowl"       # one entity: both frames agree
    a = run(ts, S.settings(direction_frame="segment_entity"))
    b = run(ts, S.settings(direction_frame="demo_first_entity"))
    assert [s for s, _ in a[("intrusion.reach", 0.0, .75)].candidates] == \
           [s for s, _ in b[("intrusion.reach", 0.0, .75)].candidates]


def test_reach_start_outside_the_workspace_is_dropped_with_a_reason():
    ts = S.task_set(n_demos=2)
    result = run(ts, reach_start_m=lambda speed, tau: 5.0)
    for el in (0.0, .5236, 1.0472):
        cell = result[("intrusion.reach", el, .75)]
        assert cell.candidates == [] and cell.info["unreachable"] == "reach_start_outside_workspace"
    assert result[("intrusion.slide", 0.0, .75)].candidates          # only the reach is checked
    small = run(ts, reach_start_m=lambda speed, tau: .01)
    assert small[("intrusion.reach", 0.0, .75)].candidates


def test_coverage_is_the_fraction_of_other_demos_passing_near_the_hit_point_and_orders_candidates():
    ts = S.task_set(n_demos=5, deviating=0, y_until=20)
    ordered = run(ts, S.settings(coverage_ordering=True))
    plain = run(ts, S.settings(coverage_ordering=False))
    cell = ordered[("intrusion.reach", 0.0, .75)]
    by_anchor = {}
    for _, e in cell.candidates:
        by_anchor[e["anchor_step"]] = e["coverage"]
    # demo_0 runs 0.5 m beside the common path during its first 20 samples: anchors there are not covered by the
    # four other demos; the anchor on the common path is covered by all of them
    assert any(c == 0.0 for c in by_anchor.values()) and any(c == 1.0 for c in by_anchor.values())
    assert all(c == 1.0 for a, c in by_anchor.items() if a >= 20) and all(c == 0.0 for a, c in by_anchor.items() if a < 20)
    assert all(e["coverage_n_demos"] == 4 for _, e in cell.candidates)
    first = [e["coverage"] for _, e in cell.candidates]
    assert first == sorted(first, reverse=True)                              # coverage descending
    off = [e["coverage"] for _, e in plain[("intrusion.reach", 0.0, .75)].candidates]
    assert off != sorted(off, reverse=True)                                  # not ordered when switched off
    assert {s["id"] for s, _ in cell.candidates} == {s["id"] for s, _ in plain[("intrusion.reach", 0.0, .75)].candidates}
    # the deviating demonstration's neighbours: demo_1 is covered by demo_0 only after sample 20
    other = run(ts, demo="demo_1")[("intrusion.reach", 0.0, .75)]
    assert all(e["coverage"] == pytest.approx(3 / 4) for _, e in other.candidates if e["anchor_step"] < 12)
    assert all(e["coverage"] == pytest.approx(1.0) for _, e in other.candidates if e["anchor_step"] >= 24)


def test_coverage_radius_and_entity_frame_mapping():
    ts = S.task_set(n_demos=2, yaw=math.radians(50))
    gs = S.settings()
    index = CoverageIndex(ts, gs)
    t, seg = ts.trajectories[0], ts.trajectories[0].segments[0]
    on = t.eef_pos[10]
    assert index.score(t, seg, on) == (1.0, 1)
    # the finger cloud spans z +-0.02 about the EEF: a point radius - 1 mm above the cloud top is covered, 5 mm further is not
    top = on + [0., 0., .02 + gs.coverage_radius_m - .001]
    assert index.score(t, seg, top)[0] == 1.0
    assert index.score(t, seg, on + [0., 0., .02 + gs.coverage_radius_m + .005])[0] == 0.0
    # a segment that no other demonstration has (or on another entity) gives no coverage value
    ts.trajectories[1].segments = ts.trajectories[1].segments[1:]
    assert CoverageIndex(ts, gs).score(t, seg, on) == (None, 0)


def test_single_demo_has_no_coverage_and_still_proposes():
    ts = S.task_set(n_demos=1)
    cell = run(ts)[("intrusion.reach", 0.0, .75)]
    assert cell.candidates and all(e["coverage"] is None for _, e in cell.candidates)


def test_unreachable_targets_are_named_and_never_clamped():
    ts = S.task_set(n_demos=2)
    censored = run(ts, t_stop=None)
    assert len(censored) == 15 and all(c.candidates == [] and c.info["unreachable"] == "t_stop_censored"
                                       for c in censored.values())
    short = run(ts, t_stop=.1)
    assert short[("intrusion.reach", 0.0, .75)].info["unreachable"] == "below_minimum_lead"
    assert short[("intrusion.reach", 0.0, .75)].info["tau_s"] is None
    normal = run(ts, t_stop=.3)
    for target, tau in ((.75, .2), (1.5, .45), (3.0, .9)):
        cell = normal[("intrusion.slide", 0.0, target)]
        assert cell.info["tau_s"] == pytest.approx(tau)
        assert all(s["intensity"]["tau_s"] == pytest.approx(tau) for s, _ in cell.candidates)
        assert all(e["tau_s"] <= e["available_before_hit_s"] + 1e-9 for _, e in cell.candidates)
    # free height = 1.5 - 0.95 - 0.02 = 0.53 m: at most sqrt(2 * 0.53 / 9.81) = 0.33 s of free fall
    assert normal[("intrusion.fall", None, .75)].info["unreachable"] is None
    assert normal[("intrusion.fall", None, 1.5)].info["unreachable"] == "free_fall_height"


def test_budget_per_cell_and_round_robin_over_directions():
    ts = S.task_set(n_demos=3)
    result = run(ts, S.settings(max_candidates_per_family_layout=5))
    for key, cell in result.items():
        if cell.info["n_proposed"] > 5:
            assert len(cell.candidates) == 5
    full = run(ts)[("intrusion.reach", 0.0, .75)]
    assert len({e["direction_index"] for _, e in full.candidates}) == 8


def test_slide_needs_a_support_rule_and_anchors_only_where_it_holds():
    ts = S.task_set(n_demos=2)
    with pytest.raises(ValueError, match="slide_support"):
        list(propose(ts.trajectories[0], ts, S.settings(), S.WORKSPACE, .3, S.AGENT, 9.81))
    cell = run(ts, finger_gap=np.full(S.N, .02))[("intrusion.slide", 0.0, .75)]
    assert cell.candidates
    # eef z .95, finger sample .02 lower, offsets lift the exposure point to the finger point
    for spec, _ in cell.candidates:
        assert spec["exposure"]["offset_m"][2] == pytest.approx(-.02)
    nothing = list(propose(ts.trajectories[0], ts, S.settings(), S.WORKSPACE, .3, S.AGENT, 9.81,
                           slide_support=lambda p, t: False))
    slide = [c for c in nothing if c.family == "intrusion.slide"][0]
    assert slide.candidates == [] and slide.info["unreachable"] == "no_supported_anchor"


def test_follow_through_comes_from_the_settings():
    ts = S.task_set(n_demos=2)
    result = run(ts)
    assert all(s["motion"]["params"]["follow_through"] == "retract"
               for s, _ in result[("intrusion.reach", 1.0472, .75)].candidates)
    assert all(s["motion"]["params"]["follow_through"] == "pass" for s, _ in result[("intrusion.slide", 0.0, .75)].candidates)
    assert all("follow_through" not in s["motion"]["params"] for s, _ in result[("intrusion.fall", None, .75)].candidates)
    bare = run(ts, S.settings(follow_through={}))
    assert all("follow_through" not in s["motion"]["params"] for c in bare.values() for s, _ in c.candidates)


def test_speed_levels_come_from_all_trajectories_and_empty_levels_drop_the_anchor():
    ts = S.task_set(n_demos=3)
    gs = S.settings()
    levels = phase_speed_levels(ts.trajectories, gs)
    assert set(levels) == {"approach_0", "manipulate_0"}
    assert all(v == pytest.approx([.2, .2, .2]) for v in levels.values())        # 0.01 m per 0.05 s
    demo = S.settings(speed_source="demo")         # the paradigm speed needs no demo speed (see below)
    cell = run(ts, demo, speed_levels={"approach_0": [], "manipulate_0": [.3]})[("intrusion.reach", 0.0, .75)]
    assert cell.candidates and all(e["phase"] == "manipulate_0" for _, e in cell.candidates)
    none = run(ts, demo, speed_levels={"approach_0": [], "manipulate_0": []})[("intrusion.reach", 0.0, .75)]
    assert none.candidates == [] and none.info["unreachable"] == "no_speed"


def test_risk_comes_from_robot_points_and_neighbour_boxes_unless_injected():
    ts = S.task_set(n_demos=2)
    t = ts.trajectories[0]
    gs = S.settings()
    risk = robot_neighbour_clearance(t, ts.layouts[t.demo], ts.robot_points, gs)
    # cup at (1, 1, .95) half .05; finger at (0.01 k, 0, .95)
    k = 40
    expected = math.hypot(1.0 - .05 - .01 * k, 1.0 - .05)
    assert abs(risk[k] - expected) < gs.thinned_point_spacing_m
    assert risk[0] > risk[60]                                                  # the finger approaches the cup
    gap = finger_table_gap(t, ts.layouts[t.demo], ts.robot_points, gs.thinned_point_spacing_m, ("gripper",))
    assert gap == pytest.approx(np.full(S.N, .95 - .02 - S.TOP))
    injected = np.linspace(1, 0, S.N)
    cell = run(ts, risk=injected)[("intrusion.fall", None, .75)]
    assert all(e["risk_clearance_m"] == pytest.approx(injected[e["anchor_step"]]) for _, e in cell.candidates)


def test_thinning_keeps_one_point_per_voxel():
    pts = np.array([[0., 0., 0.], [.001, 0., 0.], [.05, 0., 0.]])
    assert len(thin_points(pts, .02)) == 2


def test_layout_prefilter_probes_replay_frozen_support_and_held():
    ts = S.task_set(n_demos=1)
    layout = ts.layouts["demo_0"]
    half = [.02, .02, .02]
    through_cup = [(t, np.array([1.0 - .03 + .01 * t, 1.0, .95]), np.array([1., 0, 0, 0])) for t in range(5)]
    probes = layout_probes(layout, 0.0, ("support",), half)
    assert not sweep(through_cup, probes, .001)["passed"]
    result = sweep(through_cup, probes, .001)
    assert result["partners"] == ["cup_0"] and result["state"] in ("replay", "frozen")
    # 1 cm into the table is not an overlap when the support is ignored, and is one when it is not
    on_table = [(0.0, np.array([0., 0., S.TOP + .01]), np.array([1., 0, 0, 0]))]
    assert sweep(on_table, layout_probes(layout, 0.0, ("support",), half), .001)["passed"]
    assert sweep(on_table, layout_probes(layout, 0.0, (), half), .001)["partners"] == ["table"]
    # the entity the robot holds is robot-side
    at_bowl = [(0.0, np.array([.3, 0., S.TOP + .03]), np.array([1., 0, 0, 0]))]
    assert sweep(at_bowl, layout_probes(layout, 0.0, ("support",), half), .001)["partners"] == ["bowl"]
    held = ["bowl"] * S.N
    assert sweep(at_bowl, layout_probes(layout, 0.0, ("support",), half, held), .001)["passed"]
    # depth is a real penetration depth
    shallow = [(0.0, np.array([1.0, 1.0, .95 + .05 + .02 - .004]), np.array([1., 0, 0, 0]))]
    depth = sweep(shallow, layout_probes(layout, 0.0, ("support",), half), .001)["by_state"]["replay"]["partners"]["cup_0"]
    assert depth == pytest.approx(.004, abs=1e-6)


# ---- approach speed source (D20 / D21): demo-derived or the fixed paradigm speed ----

LEVELS = {"approach_0": [.1, .3, .5], "manipulate_0": [.2, .4, .6]}
DEMO_SPEED = {"approach_0": .3, "manipulate_0": .4}          # medians of LEVELS


def _specs(result):
    return {k: [s for s, _ in c.candidates] for k, c in result.items()}


def test_speed_source_is_a_required_validated_setting_without_a_code_default():
    raw = S.raw_settings()
    assert raw["speed_source"] in ("demo", "paradigm")
    assert set(raw["paradigm_approach_speed_mps"]) == {"intrusion.reach", "intrusion.slide"}
    for key in ("speed_source", "paradigm_approach_speed_mps"):
        with pytest.raises(ConfigError, match=key):
            GenerateSettings.from_mapping({k: v for k, v in raw.items() if k != key})
    with pytest.raises(ConfigError, match="speed_source"):
        GenerateSettings.from_mapping(dict(raw, speed_source="fastest"))
    with pytest.raises(ConfigError, match="intrusion.slide"):
        GenerateSettings.from_mapping(dict(raw, paradigm_approach_speed_mps={"intrusion.reach": .6}))
    with pytest.raises(ConfigError, match="paradigm_approach_speed_mps"):
        GenerateSettings.from_mapping(dict(raw, paradigm_approach_speed_mps={"intrusion.reach": .6,
                                                                              "intrusion.slide": 0.0}))


def test_paradigm_speeds_are_the_first_declared_level_inside_the_pilot_intensity_range():
    from hazardlib.config import load_config
    intensity = load_config("intensity", strict=True, config_dir=S.CONFIG)["intensity"]
    for family, speed in S.raw_settings()["paradigm_approach_speed_mps"].items():
        axis = intensity[family]["approach_speed_mps"]
        assert axis["lo"] <= speed <= axis["hi"]
        assert speed == axis["levels"][0]


def test_paradigm_speed_replaces_the_demo_speed_of_slide_and_reach_and_nothing_else():
    ts = S.task_set(n_demos=3)
    demo = run(ts, S.settings(speed_source="demo"), speed_levels=LEVELS)
    para = run(ts, S.settings(speed_source="paradigm"), speed_levels=LEVELS)
    assert set(demo) == set(para)
    expected = S.raw_settings()["paradigm_approach_speed_mps"]
    seen = set()
    for key, cell in demo.items():
        family = key[0]
        assert cell.info == para[key].info
        assert len(cell.candidates) == len(para[key].candidates)
        for (a, ea), (b, eb) in zip(cell.candidates, para[key].candidates):
            assert (ea["speed_source"], eb["speed_source"]) == ("demo", "paradigm")
            if family == "intrusion.fall":
                assert "approach_speed_mps" not in a["intensity"] and a == b
                continue
            assert a["intensity"]["approach_speed_mps"] == DEMO_SPEED[ea["phase"]]
            assert b["intensity"]["approach_speed_mps"] == expected[family]
            seen.add(family)
            del a["intensity"]["approach_speed_mps"], b["intensity"]["approach_speed_mps"]
            assert a == b                                   # same id, tau, anchor, direction, offset, follow-through
    assert seen == {"intrusion.reach", "intrusion.slide"}


def test_demo_source_does_not_read_the_paradigm_speeds():
    ts = S.task_set(n_demos=3)
    base = _specs(run(ts, S.settings(speed_source="demo"), speed_levels=LEVELS))
    other = _specs(run(ts, S.settings(speed_source="demo", paradigm_approach_speed_mps={
        "intrusion.reach": 1.9, "intrusion.slide": 1.7}), speed_levels=LEVELS))
    assert base == other
    assert any(s["intensity"].get("approach_speed_mps") == DEMO_SPEED["approach_0"]
               for specs in base.values() for s in specs)


def test_paradigm_speed_needs_no_demo_speed_and_feeds_the_reach_stroke_budget():
    ts = S.task_set(n_demos=3)
    calls = []

    def stroke(speed, tau):
        calls.append(speed)
        return 5.0
    gs = S.settings(speed_source="paradigm")
    result = run(ts, gs, speed_levels={"approach_0": [], "manipulate_0": []}, reach_start_m=stroke)
    assert set(calls) == {gs.paradigm_approach_speed_mps["intrusion.reach"]}
    assert all(c.candidates == [] and c.info["unreachable"] == "reach_start_outside_workspace"
               for k, c in result.items() if k[0] == "intrusion.reach")
    slide = result[("intrusion.slide", 0.0, .75)]
    assert slide.candidates and all(s["intensity"]["approach_speed_mps"] ==
                                    gs.paradigm_approach_speed_mps["intrusion.slide"] for s, _ in slide.candidates)


def test_speed_source_is_part_of_the_generation_identity():
    from hazardlib.hashing import sha256_hex
    raw = S.raw_settings()
    other = "demo" if raw["speed_source"] == "paradigm" else "paradigm"
    assert sha256_hex(raw) != sha256_hex(dict(raw, speed_source=other))
    assert sha256_hex(raw) != sha256_hex(dict(raw, paradigm_approach_speed_mps={
        "intrusion.reach": 1.0, "intrusion.slide": 1.0}))

