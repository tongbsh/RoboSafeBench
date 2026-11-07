"""Probe -> TaskContext (T06 item 1; acceptance 7)."""
import numpy as np
import pytest
import yaml

import pipeline_world as W
from hazardlib.config import ConfigError
from hazardlib.probe import (PROBE_KEYS, ProbeSettings, ProbeSourceError, ProbeTrace, build_task_context,
                             read_probe_trace, record_probe_trace, write_probe_trace)

TABLE = {"name": "table", "min_m": [-1.0, -1.0, 0.0], "max_m": [1.0, 1.0, 0.0], "source": "scene_yaml"}


def _state(events, u):
    closed = False
    for _, t in events:
        if u >= t - 1e-9:
            closed = not closed
    return closed


def synth(kind="scripted", name="s", seed=0, scale=1.0, events=(("close", 0.6),), success=True, y_offset=0.0,
          contact_at=None, phase=None, under_test=False, surfaces=(TABLE,), quat=True, lid_moves=True):
    """A probe trace of the 'pick' geometry: +x to (0.3, 0, 0.3) by nominal t = 0.6, then +y to (0.3, 0.3, 0.3)
    by 1.2; run ``scale`` times faster. ``events`` toggle the gripper at nominal times."""
    end = 1.2 / scale
    times = np.round(np.arange(0.0, end + 1e-9, 0.02), 10)
    steps = []
    for t in times:
        u = t * scale
        eef = [0.5 * min(u, 0.6), 0.5 * max(0.0, u - 0.6) + y_offset, 0.3]
        closed = _state(events, u)
        cube = [eef[0], eef[1], 0.05] if closed else [0.3, 0.0, 0.05]
        lid = [0.6, 0.6 + (0.05 if (lid_moves and u > 0.9) else 0.0), 0.0]
        steps.append({
            "t_s": float(t), "eef_pos": eef, "eef_quat": [1.0, 0.0, 0.0, 0.0] if quat else None,
            "eef_lin_vel_mps": None, "gripper": 0.0 if closed else 0.08, "held_object": "cube" if closed else None,
            "robot_bodies": {"robot0_hand": eef}, "in_contact": contact_at is not None and abs(t - contact_at) < 1e-9,
            "object_poses": {"cube": {"pos": cube, "quat": [1.0, 0.0, 0.0, 0.0]},
                             "lid": {"pos": lid, "quat": [1.0, 0.0, 0.0, 0.0]}},
            "phase": phase,
        })
    header = {"schema_version": "1", "task_id": "pick", "layout_id": "L0", "seed": seed,
              "source": {"kind": kind, "name": name, "under_test": under_test}, "success": success,
              "success_t_s": float(times[-1]) if success else None, "control_dt_s": 0.02,
              "support_surfaces": None if surfaces is None else [dict(s) for s in surfaces], "n_steps": len(steps)}
    return ProbeTrace.from_dict({"header": header, "steps": steps})


def three(**kwargs):
    return [synth("scripted", "follower", 0, 1.0, **kwargs), synth("demo_replay", "demo", 1, 1.0, **kwargs),
            synth("scripted", "follower_fast", 2, 1.25, **kwargs)]


def settings():
    return W.probe_settings()


def test_under_test_source_is_rejected():
    with pytest.raises(ProbeSourceError, match="R4"):
        build_task_context(three() + [synth("external_policy", "vla", 3, under_test=True)], settings())
    with pytest.raises(ProbeSourceError, match="R4"):
        build_task_context(three(), settings(), policies_under_test=["demo"])


def test_under_test_policy_recorded_on_fakehost_is_rejected():
    host = W.make_host()
    traces = W.record_traces(host, "pick")
    sess = W.session(host)            # one session (runtime) per host at a time: it owns the host's hooks
    traces.append(record_probe_trace(sess, W.PathFollower(host, name="vla", kind="external_policy", under_test=True),
                                     "pick", "L0", 9, W.PROBE_HORIZON_S, 0))
    assert traces[-1].header.source.under_test is True
    with pytest.raises(ProbeSourceError, match="policy under test"):
        build_task_context(traces, settings(), declared_workspace=W.WORKSPACE)


def test_at_least_two_source_kinds_are_required():
    runs = [synth("scripted", "a", 0), synth("scripted", "b", 1), synth("scripted", "c", 2, 1.25)]
    with pytest.raises(ValueError, match="R4 requires at least 2"):
        build_task_context(runs, settings())


def test_phases_are_cut_at_median_gripper_events_and_success():
    tc = build_task_context(three(), settings())
    assert [(p.name, p.t_start_s, p.t_end_s) for p in tc.phases] == [("approach", 0.0, 0.6), ("carry", 0.6, 1.2)]
    assert tc.provenance["phase_signature"] == ["open", "close"]
    assert tc.provenance["nominal_success_time_s"] == 1.2

    events = (("close", 0.6), ("open", 1.0))
    tc2 = build_task_context(three(events=events), settings())
    assert [p.name for p in tc2.phases] == ["approach", "carry", "approach_2"]
    assert [p.t_end_s for p in tc2.phases] == pytest.approx([0.6, 1.0, 1.2])


def test_nominal_path_is_the_per_phase_median():
    # event times are quantised to the 0.02 s samples, so the scales put them on the grid
    runs = [synth("scripted", "a", 0, 1.0, y_offset=0.01), synth("demo_replay", "b", 1, 0.75, y_offset=0.0),
            synth("scripted", "c", 2, 1.25, y_offset=-0.02)]
    tc = build_task_context(runs, settings())
    # boundaries: close at 0.6 / 0.8 / 0.48 -> median 0.6; success 1.2 / 1.6 / 0.96 -> 1.2
    assert [p.t_end_s for p in tc.phases] == pytest.approx([0.6, 1.2])
    path = tc.nominal_ee_path
    t = np.array(path.time_s)
    pos = np.array(path.pos)
    assert len(t) == 2 * settings().path_samples_per_phase + 1
    expected_x = 0.5 * np.minimum(t, 0.6)
    expected_y = 0.5 * np.maximum(0.0, t - 0.6)          # the median of the offsets +0.01, 0, -0.02 is 0
    assert np.allclose(pos[:, 0], expected_x, atol=1e-9)
    assert np.allclose(pos[:, 1], expected_y, atol=1e-9)
    assert tc.provenance["nominal_path_aggregation"].startswith("per-phase normalised time")
    assert tc.provenance["nominal_quat"] == "median_of_probe_quaternions"


def test_missing_quaternions_fall_back_to_identity_and_say_so():
    tc = build_task_context(three(quat=False), settings())
    assert np.allclose(np.array(tc.nominal_ee_path.quat), [1.0, 0.0, 0.0, 0.0])
    assert tc.provenance["nominal_quat"].startswith("identity")


def test_entities_follow_r3_order_and_keep_their_source():
    tc = build_task_context(three(), settings(), static_entities=W.STATIC_ENTITIES,
                            annotations={"container": "drawer", "goal": "goal_marker"})
    assert tc.entities["goal"].to_dict() == {"name": "goal_marker", "source": "oracle_config"}
    assert tc.entities["grasped_0"].to_dict() == {"name": "cube", "source": "probe_inferred"}
    assert tc.entities["moved_0"].to_dict() == {"name": "lid", "source": "probe_inferred"}
    assert tc.entities["container"].to_dict() == {"name": "drawer", "source": "annotation"}
    stages = [d["stage"] for d in tc.provenance["entity_discovery"]]
    assert stages == ["static", "grasped", "moved", "annotation"]


def test_conflicting_annotation_raises():
    with pytest.raises(ValueError, match="conflicts"):
        build_task_context(three(), settings(), annotations={"grasped_0": "sponge"})
    with pytest.raises(ValueError, match="static declaration"):
        build_task_context(three(), settings(), static_entities={"x": {"name": "a", "source": "probe_inferred"}})


def test_unusable_runs_are_dropped_and_counted():
    runs = three() + [synth("scripted", "failed", 5, success=False),
                      synth("scripted", "touched", 6, contact_at=0.4),
                      synth("scripted", "armed", 7, phase="kinematic"),
                      synth("scripted", "no_grasp", 8, events=())]
    tc = build_task_context(runs, settings())
    probe_set = tc.provenance["probe_set"]
    assert probe_set["excluded"] == {"hazard_present": 2, "inconsistent_gripper_events": 1, "not_successful": 1}
    assert probe_set["n_runs_total"] == 7 and probe_set["n_runs_used"] == 3


def test_probe_set_description_is_in_provenance_and_in_the_hash():
    runs = three() + [synth("external_policy", "octo_ref", 4)]
    tc = build_task_context(runs, settings())
    probe_set = tc.provenance["probe_set"]
    assert probe_set["source_kinds_used"] == ["demo_replay", "external_policy", "scripted"]
    assert probe_set["reference_policies"] == ["octo_ref"]
    assert {(s["kind"], s["name"], s["n_runs"], s["n_used"]) for s in probe_set["sources"]} == {
        ("scripted", "follower", 1, 1), ("demo_replay", "demo", 1, 1), ("scripted", "follower_fast", 1, 1),
        ("external_policy", "octo_ref", 1, 1)}
    assert len(probe_set["trace_hashes"]) == 4

    again = build_task_context(three() + [synth("external_policy", "octo_ref", 4)], settings())
    assert again.hash() == tc.hash()
    fewer = build_task_context(three(), settings())
    assert fewer.provenance["probe_set_hash"] != tc.provenance["probe_set_hash"]
    assert fewer.hash() != tc.hash()


def test_support_surfaces_from_host_or_estimated():
    tc = build_task_context(three(), settings())
    assert tc.provenance["support_surface_source"] == "host"
    assert [s.to_dict() for s in tc.support_surfaces] == [TABLE]

    est = build_task_context(three(surfaces=None), settings())
    assert est.provenance["support_surface_source"] == "probe_estimate"
    (surface,) = est.support_surfaces
    assert surface.source == "probe_estimate"
    assert surface.max_m[2] == pytest.approx(0.0 - settings().surface_below_object_m)   # lowest object centre: lid z = 0

    other = dict(TABLE, max_m=[1.0, 1.0, 0.1], min_m=[-1.0, -1.0, 0.1])
    mixed = three()[:2] + [synth("scripted", "x", 9, 1.25, surfaces=(other,))]
    with pytest.raises(ValueError, match="different support surfaces"):
        build_task_context(mixed, settings())


def test_workspace_is_declared_or_the_probe_hull():
    declared = build_task_context(three(), settings(), declared_workspace=W.WORKSPACE)
    assert declared.workspace_bounds.to_dict() == W.WORKSPACE
    assert declared.provenance["workspace_source"] == "declared"
    hull = build_task_context(three(), settings())
    assert hull.provenance["workspace_source"] == "probe_hull"
    assert hull.workspace_bounds.min_m == pytest.approx([0.0, 0.0, 0.0])
    assert hull.workspace_bounds.max_m == pytest.approx([0.6, 0.65, 0.3])


def test_trace_file_roundtrip(tmp_path):
    host = W.make_host()
    (trace, _, _) = W.record_traces(host, "pick")
    path = tmp_path / "probe.jsonl"
    write_probe_trace(path, trace)
    back = read_probe_trace(path)
    assert back.hash() == trace.hash()
    lines = path.read_text().splitlines()
    assert list(yaml.safe_load(lines[0])) == ["probe_trace"] and all(l.startswith('{"step"') for l in lines[1:])
    path.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(ValueError, match="before the header"):
        read_probe_trace(path)


def test_fakehost_task_context():
    tc = W.task_context("pick")
    assert [(p.name, p.t_start_s, p.t_end_s) for p in tc.phases] == [("approach", 0.0, 0.6), ("carry", 0.6, 1.2)]
    assert tc.entities["grasped_0"].name == W.HELD
    path = tc.nominal_ee_path
    i = list(np.round(path.time_s, 9)).index(0.3)
    assert np.allclose(path.pos[i], [0.15, 0.0, 0.3], atol=1e-9)
    assert tc.provenance["eef_speed_median_mps"] == pytest.approx(0.5)
    assert sorted(tc.provenance["robot_body_paths"]) == ["robot0_hand", "robot0_link"]   # camera is excluded
    assert tc.provenance["occupancy_content"] == "nominal_eef_and_robot_bodies"
    assert tc.provenance["probe_set"]["source_kinds_used"] == ["demo_replay", "scripted"]
    assert W.task_context("low").phases[0].name == "approach" and len(W.task_context("low").phases) == 1


def test_probe_settings_refuse_placeholders_of_the_repository_config():
    with pytest.raises(ConfigError, match="gripper_closed_below"):
        ProbeSettings.load(W.REPO_CONFIG_DIR)
    assert ProbeSettings.load(W.FIXTURE_CONFIG_DIR).min_source_kinds == 2


def test_repository_probe_yaml_structure():
    data = yaml.safe_load((W.REPO_CONFIG_DIR / "probe.yaml").read_text(encoding="utf-8"))
    assert sorted(data) == sorted(PROBE_KEYS)
    assert data["min_source_kinds"] == 2
    assert all(data[k] == "PLACEHOLDER" for k in PROBE_KEYS if k != "min_source_kinds")
    fixture = yaml.safe_load((W.FIXTURE_CONFIG_DIR / "probe.yaml").read_text(encoding="utf-8"))
    assert sorted(fixture) == sorted(PROBE_KEYS)
