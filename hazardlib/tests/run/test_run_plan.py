"""Episode plan: paired controls and their de-duplication (R1 item 2), episode length, label filter."""
import pytest

from hazardlib.run.plan import CONTROL, HAZARD, episode_horizon_s, plan_episodes, tau_s
from hazardlib.spec import HazardInstance, TaskContext

EXTRA = 0.5


def _tc(task, layout, success_t):
    return TaskContext.from_dict({
        "task_id": task, "layout_id": layout, "entities": {},
        "phases": [{"name": "approach", "t_start_s": 0.0, "t_end_s": success_t}],
        "nominal_ee_path": {"time_s": [0.0, success_t], "pos": [[0.0, 0.0, 0.3], [0.5, 0.0, 0.3]],
                            "quat": [[1.0, 0.0, 0.0, 0.0]] * 2},
        "occupancy": {"points": [], "time_s": []},
        "workspace_bounds": {"min_m": [-1.0, -1.0, 0.0], "max_m": [1.0, 1.0, 1.0]},
        "support_surfaces": [], "provenance": {}})


def _inst(instance_id, task, layout, onset, hit, label):
    return HazardInstance.from_dict({
        "instance_id": instance_id, "spec_id": instance_id.split(".")[0], "family": "intrusion.fall",
        "agent": {"kind": "object", "shape": "sphere", "half_extents_m": [0.03], "mass_kg": 0.2},
        "keyframes": {"time_s": [onset, hit], "pos": [[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
                      "quat": [[1.0, 0.0, 0.0, 0.0]] * 2},
        "nominal_hit_point": [0.0, 0.0, 0.5], "nominal_hit_time_s": hit, "onset_time_s": onset,
        "release_time_s": onset, "release_velocity": {"lin_mps": [0.0] * 3, "ang_radps": [0.0] * 3},
        "certificate": {"g0": True, "g1": label == "wait_solvable", "g2": None, "label": label, "notes": ""},
        "provenance": {"task_id": task, "layout_id": layout}})


CONTEXTS = {("t", "A"): _tc("t", "A", 1.0), ("t", "B"): _tc("t", "B", 1.0)}
# b.t.A and a.t.A: same layout, onset and length -> one control; c.t.A: other onset; d.t.B: other layout;
# e.t.A: same onset as a/b but t_hit beyond the success time -> other length.
INSTANCES = [_inst("b.t.A", "t", "A", 0.2, 0.6, "wait_solvable"), _inst("c.t.A", "t", "A", 0.3, 0.6, "uncertified"),
             _inst("a.t.A", "t", "A", 0.2, 0.9, "wait_solvable"), _inst("d.t.B", "t", "B", 0.2, 0.6, "wait_solvable"),
             _inst("e.t.A", "t", "A", 0.2, 1.4, "unsolvable_by_probes")]


def _brief(plan):
    return [(p.kind, p.episode_id, p.seed, p.control_id) for p in plan]


def test_controls_are_paired_deduplicated_and_ordered():
    plan = plan_episodes(INSTANCES, CONTEXTS, [7, 3], EXTRA, None)
    one_seed = [(CONTROL, "control:a.t.A", "control:a.t.A"), (HAZARD, "b.t.A", "control:a.t.A"),
                (CONTROL, "control:c.t.A", "control:c.t.A"), (HAZARD, "c.t.A", "control:c.t.A"),
                (HAZARD, "a.t.A", "control:a.t.A"),
                (CONTROL, "control:d.t.B", "control:d.t.B"), (HAZARD, "d.t.B", "control:d.t.B"),
                (CONTROL, "control:e.t.A", "control:e.t.A"), (HAZARD, "e.t.A", "control:e.t.A")]
    expected = [(k, e, seed, c) for seed in (7, 3) for k, e, c in one_seed]
    assert _brief(plan) == expected
    controls = {(p.episode_id, p.seed): p for p in plan if p.kind == CONTROL}
    shared = controls[("control:a.t.A", 7)]
    assert shared.paired_instance_ids == ("a.t.A", "b.t.A") and shared.instance is None
    assert shared.control_onset_s == 0.2 and shared.horizon_s == pytest.approx(1.0 + EXTRA)
    assert controls[("control:e.t.A", 3)].horizon_s == pytest.approx(1.4 + EXTRA)
    assert (shared.task_id, shared.layout_id) == ("t", "A")
    hazards = [p for p in plan if p.kind == HAZARD]
    assert all(p.control_onset_s is None and p.instance.instance_id == p.episode_id for p in hazards)


def test_horizon_tau_and_label_filter():
    assert episode_horizon_s(CONTEXTS[("t", "A")], INSTANCES[0], EXTRA) == pytest.approx(1.5)
    assert episode_horizon_s(CONTEXTS[("t", "A")], INSTANCES[4], EXTRA) == pytest.approx(1.9)
    assert tau_s(INSTANCES[1]) == pytest.approx(0.3)
    plan = plan_episodes(INSTANCES, CONTEXTS, [0], EXTRA, ["wait_solvable"])
    assert [p.episode_id for p in plan] == ["control:a.t.A", "b.t.A", "a.t.A", "control:d.t.B", "d.t.B"]
    only_b = plan_episodes(INSTANCES[:1], CONTEXTS, [0], EXTRA, None)
    assert only_b[0].episode_id == "control:b.t.A" and only_b[0].paired_instance_ids == ("b.t.A",)
    assert plan_episodes(INSTANCES, CONTEXTS, [0], EXTRA, ["evade_solvable"]) == []


@pytest.mark.parametrize("seeds, labels, message", [([], None, "seeds"), ([1, 1], None, "seeds"),
                                                    ([0], ["solvable"], "--labels"), ([0], [], "--labels")])
def test_bad_seeds_or_labels_are_refused(seeds, labels, message):
    with pytest.raises(ValueError, match=message):
        plan_episodes(INSTANCES, CONTEXTS, seeds, EXTRA, labels)
