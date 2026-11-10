"""Task-agnostic interaction segmentation of demonstrations, on synthetic recordings (no simulator)."""
from pathlib import Path

import numpy as np

from hazardhosts.libero.trajectory_segments import (build_phases, find_interactions, merge_gaps, runs,
                                                    segment_features)
from hazardlib.config import load_config

CONFIG = Path(__file__).parents[2] / "configs/hazard/pilot/libero"
SETTINGS = dict(load_config("trajectory_analysis", strict=True, config_dir=CONFIG),
                segmentation=load_config("segmentation", strict=True, config_dir=CONFIG))
SPACING = 0.005
DT = 0.05


def recording(n, objects=0, joints=0):
    """A still recording of ``n`` samples; the end effector drifts so that path / speed are defined."""
    eef = np.zeros((n, 3))
    eef[:, 0] = np.arange(n) * 0.004
    return {"time_s": np.arange(n) * DT, "eef_pos": eef, "held_index": -np.ones(n, dtype=int),
            "gripper_low_z": np.full(n, 0.95), "surface_z": np.asarray(0.9), "neighbor_clearance": np.full(n, 0.2),
            "object_pos": np.tile([[0.1 * i, 0.0, 0.9] for i in range(max(objects, 1))], (n, 1, 1))[:, :objects],
            "joint_pos": np.zeros((n, joints))}


def carry(a, index, grasp, release, lift=0.15, advance=0.2):
    """Object ``index`` is held from ``grasp`` to ``release - 1``: lifted, moved, put down at the release."""
    a["held_index"][grasp:release] = index
    n = release - grasp
    shape = np.sin(np.linspace(0, np.pi, n))
    a["object_pos"][grasp:release, index, 2] += lift * shape
    a["object_pos"][grasp:release, index, 0] += advance * np.linspace(0, 1, n)
    a["object_pos"][release:, index, 0] += advance


FREE = lambda name, i: {"name": name, "kind": "free", "object_index": i}
ARTICULATED = lambda name, i, kinds: {"name": name, "kind": "articulated", "object_index": i, "joint_types": kinds,
                                      "joint_indices": list(range(len(kinds)))}


def names(result):
    return [p["name"] for p in result["phases"]]


def test_single_pick_and_place_is_approach_manipulate_retreat():
    a = recording(70, objects=2)
    carry(a, 0, 20, 50)
    result = segment_features(a, [FREE("bowl", 0), FREE("plate", 1)], SETTINGS, SPACING)
    assert names(result) == ["approach_0", "manipulate_0", "retreat"]
    assert result["n_interactions"] == 1 and result["status"] == "ok"
    assert {p["entity"] for p in result["phases"]} == {"bowl"} and result["first_entity"] == "bowl"
    approach, manipulate, retreat = result["phases"]
    # The interaction spans the grasp (sample 20) to the sample after the last held one; it starts at the first
    # motion of the held object, which is the grasp sample itself here.
    assert (approach["start_index"], approach["end_index"]) == (0, 20)
    assert manipulate["start_index"] == 20 and manipulate["end_index"] in (50, 51)
    assert retreat["end_index"] == 69 and retreat["start_index"] == manipulate["end_index"]
    assert all(p["duration_s"] > 0 and p["t_end_s"] > p["t_start_s"] for p in result["phases"])
    assert result["interactions"][0]["grasp_index"] == 20 and result["interactions"][0]["lift_index"] is not None
    assert all(set(p["speed_mps"]) >= {"n", "quantiles"} and "neighbor_clearance_m" in p for p in result["phases"])


def test_two_sequential_pick_and_places_give_two_interactions_in_time_order():
    a = recording(120, objects=3)       # soup, sauce, basket (the basket stays where it is)
    carry(a, 1, 15, 45)                 # the sauce first, though listed second: order is by time
    carry(a, 0, 65, 100)
    result = segment_features(a, [FREE("soup", 0), FREE("sauce", 1), FREE("basket", 2)], SETTINGS, SPACING)
    assert names(result) == ["approach_0", "manipulate_0", "approach_1", "manipulate_1", "retreat"]
    assert [i["entity"] for i in result["interactions"]] == ["sauce", "soup"]
    assert result["first_entity"] == "sauce" and result["first_entity_index"] == 1
    assert [p["entity"] for p in result["phases"]] == ["sauce", "sauce", "soup", "soup", "soup"]
    ends = [p["end_index"] for p in result["phases"]]
    starts = [p["start_index"] for p in result["phases"]]
    assert starts[1:] == ends[:-1]      # the segments tile the recording


def test_drawer_opening_is_an_articulated_interaction_with_the_drawer_as_entity():
    a = recording(80, joints=1)
    a["joint_pos"][25:45, 0] = np.linspace(0, 0.18, 20)     # slide joint, 18 cm of travel
    a["joint_pos"][45:, 0] = 0.18
    result = segment_features(a, [ARTICULATED("cabinet_middle_region", 0, ["slide"])], SETTINGS, SPACING)
    assert names(result) == ["approach_0", "manipulate_0", "retreat"]
    interaction = result["interactions"][0]
    assert interaction["entity"] == "cabinet_middle_region" and interaction["entity_kind"] == "articulated"
    assert interaction["start_index"] == 24 or interaction["start_index"] == 25
    assert interaction["end_index"] >= 44 and interaction["change_unit"] == "m"
    assert abs(interaction["change"] - 0.18) < 1e-9


def test_knob_turning_uses_the_hinge_thresholds():
    a = recording(60, joints=1)
    a["joint_pos"][20:30, 0] = np.linspace(0, 1.2, 10)      # radians
    a["joint_pos"][30:, 0] = 1.2
    result = segment_features(a, [ARTICULATED("stove", 0, ["hinge"])], SETTINGS, SPACING)
    assert result["n_interactions"] == 1 and result["interactions"][0]["change_unit"] == "rad"
    assert names(result) == ["approach_0", "manipulate_0", "retreat"]


def test_a_nudge_below_the_articulation_threshold_is_not_an_interaction():
    a = recording(60, joints=2)
    a["joint_pos"][20:30, 0] = np.linspace(0, 0.008, 10)    # 8 mm of a drawer
    a["joint_pos"][20:30, 1] = np.linspace(0, 0.04, 10)     # 2 degrees of a door
    a["joint_pos"][30:] = a["joint_pos"][29]
    result = segment_features(a, [ARTICULATED("cabinet", 0, ["slide", "hinge"])], SETTINGS, SPACING)
    assert result["n_interactions"] == 0


def test_a_demonstration_without_interaction_is_reported_not_an_error():
    a = recording(50, objects=2, joints=1)
    result = segment_features(a, [FREE("bowl", 0), ARTICULATED("drawer", 1, ["slide"])], SETTINGS, SPACING)
    assert result["status"] == "no_interaction" and result["n_interactions"] == 0
    assert result["phases"] == [] and result["interactions"] == [] and result["first_entity"] is None
    assert result["first_entity_index"] is None and result["duration_s"] > 0
    # and a recording too short to contain a segment
    short = segment_features(recording(2, objects=1), [FREE("bowl", 0)], SETTINGS, SPACING)
    assert short["status"] == "no_interaction"


def test_a_single_contact_sample_is_not_an_interaction():
    a = recording(40, objects=1)
    a["held_index"][10] = 0
    a["object_pos"][10, 0, 0] += 0.01
    assert segment_features(a, [FREE("bowl", 0)], SETTINGS, SPACING)["n_interactions"] == 0


def test_pushing_counts_only_beyond_the_displacement_threshold():
    a = recording(60, objects=1)
    a["object_pos"][20:30, 0, 0] += np.linspace(0, 0.12, 10)    # pushed 12 cm, never held
    a["object_pos"][30:, 0, 0] += 0.12
    result = segment_features(a, [FREE("plate", 0)], SETTINGS, SPACING)
    assert names(result) == ["approach_0", "manipulate_0", "retreat"]
    assert not result["interactions"][0]["held"] and result["interactions"][0]["grasp_index"] is None
    bump = recording(60, objects=1)
    bump["object_pos"][20:26, 0, 0] += np.linspace(0, 0.02, 6)  # bumped by 2 cm, as a released bowl does
    bump["object_pos"][26:, 0, 0] += 0.02
    assert segment_features(bump, [FREE("plate", 0)], SETTINGS, SPACING)["n_interactions"] == 0


def test_regrasp_is_one_interaction_a_long_pause_is_two():
    series = lambda: [{"name": "mug", "kind": "free", "held_index": 0, "pos": None}]
    for pause, expected in ((5, 1), (30, 2)):
        a = recording(120, objects=1)
        carry(a, 0, 10, 40)
        carry(a, 0, 40 + pause, 80 + pause)
        entity = series()
        entity[0]["pos"] = a["object_pos"][:, 0]
        found, dropped = find_interactions(a["time_s"], a["held_index"], entity, SETTINGS, SPACING)
        assert len(found) == expected and not dropped


def test_an_interest_object_nudged_while_another_is_carried_is_dropped_not_double_counted():
    a = recording(100, objects=2)
    carry(a, 0, 20, 70)
    a["object_pos"][35:45, 1, 0] += np.linspace(0, 0.08, 10)    # the basket is shoved during the carry
    a["object_pos"][45:, 1, 0] += 0.08
    entities = [{"name": "soup", "kind": "free", "held_index": 0, "pos": a["object_pos"][:, 0]},
                {"name": "basket", "kind": "free", "held_index": 1, "pos": a["object_pos"][:, 1]}]
    found, dropped = find_interactions(a["time_s"], a["held_index"], entities, SETTINGS, SPACING)
    assert [i["entity"] for i in found] == ["soup"]
    assert dropped and dropped[0]["entity"] == "basket" and "inside" in dropped[0]["reason"]


def test_open_the_drawer_then_put_the_bowl_inside_orders_an_articulated_and_a_free_interaction():
    a = recording(140, objects=1, joints=1)
    a["joint_pos"][15:35, 0] = np.linspace(0, 0.2, 20)
    a["joint_pos"][35:, 0] = 0.2
    carry(a, 0, 60, 110)
    result = segment_features(a, [FREE("bowl", 0), ARTICULATED("drawer_region", 1, ["slide"])], SETTINGS, SPACING)
    assert [i["entity"] for i in result["interactions"]] == ["drawer_region", "bowl"]
    assert result["first_entity"] == "drawer_region" and result["first_entity_index"] == 1
    assert names(result) == ["approach_0", "manipulate_0", "approach_1", "manipulate_1", "retreat"]


def test_helpers_runs_and_merge_gaps():
    assert runs([0, 1, 1, 0, 1, 1, 1, 0], 3) == [[4, 6]]
    assert runs([1, 1, 1], 3) == [[0, 2]]
    assert merge_gaps([[0, 4], [8, 12], [30, 35]], 3) == [[0, 12], [30, 35]]
    assert build_phases(np.arange(5) * DT, []) == []
