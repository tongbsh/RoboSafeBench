"""Gate G0 (T06 item 3; acceptance 11, 13)."""
import numpy as np
import pytest

import pipeline_world as W
from hazardlib.gates import G0_REASONS, G0Settings, gate_instances, run_g0
from hazardlib.kinematics import SampledPath
from hazardlib.spec import TaskContext


def g0(instance, task, host=None, tc=None):
    host = host or W.make_host()
    return run_g0(instance, tc or W.task_context(task), host.describe(), W.gate_settings().g0)


@pytest.mark.parametrize("make_spec,task", [(lambda: W.fall_spec(), "pick"), (lambda: W.slide_spec(), "low"),
                                            (lambda: W.reach_spec(), "pick")], ids=["fall", "slide", "reach"])
def test_nominal_instances_pass(make_spec, task):
    result = g0(W.build(make_spec(), task), task)
    assert result.passed, result.reasons
    assert result.checks["early_contact_check"] == "eef_and_bodies"
    assert result.checks["r1_check"] == "skipped: BD hazard_ccd is true"


def test_negative_onset_margin_is_rejected():
    inst = W.build(W.fall_spec(tau=0.2, frac=0.1), "pick")       # t_hit = 0.06 s, onset = -0.14 s
    assert inst.provenance["hit_diagnostics"]["onset_margin_s"] < 0
    result = g0(inst, "pick")
    assert result.reasons == ["onset_margin_negative"]
    assert result.checks["onset_margin_s"] == pytest.approx(-0.14)


def test_chase_counterexample_is_rejected():
    """EEF at 0.5 m/s, hazard chasing in the same direction at 0.3 m/s (T06 prompt): already inside the
    standoff at the onset and opening at t_hit."""
    inst = W.build(W.slide_spec(tau=0.1, speed=0.3, direction=(1.0, 0.0, 0.0)), "low")
    result = g0(inst, "low")
    assert set(result.reasons) == {"non_closing", "early_contact"}
    assert result.checks["closing_speed_mps"] < 0
    early = result.checks["early_contact"]
    assert early["time_s"] == pytest.approx(inst.onset_time_s)
    assert early["min_distance_m"] < W.STANDOFF - W.gate_settings().g0.early_contact_tol_m
    assert early["min_distance_m"] == pytest.approx(0.0417, abs=5e-4)


def test_hazard_inside_the_standoff_at_onset_is_rejected():
    """Closing at t_hit but overlapping the EEF at the onset. With constant velocities that cannot happen, so
    the EEF turns: a linear reach along +x at 0.2 m/s starts on the EEF (which moves +x at 0.5 m/s), and hits
    it 20 ms after the corner of the pick path, when the EEF moves +y."""
    inst = W.build(W.reach_spec(tau=0.25, speed=0.2, direction=(1.0, 0.0, 0.0), phase="carry", frac=1.0 / 30.0,
                                profile="linear"), "pick")
    result = g0(inst, "pick")
    assert result.reasons == ["early_contact"]
    assert result.checks["closing_speed_mps"] > 0
    nominal = W.task_context("pick")
    eef_at_onset = SampledPath.from_pose_sequence(nominal.nominal_ee_path).pos(inst.onset_time_s)
    hazard_at_onset = np.array(inst.keyframes.pos[0])
    assert np.linalg.norm(hazard_at_onset - eef_at_onset) < W.STANDOFF
    assert result.checks["early_contact"]["min_distance_m"] < result.checks["early_contact"]["threshold_m"]


def test_early_contact_uses_robot_bodies_when_the_probe_has_them():
    """A wrist sphere 0.08 m above the hand is hit first by a fall onto the hand; with an EEF-only trace the
    same instance passes (the limitation the report names)."""
    tc = W.task_context("pick")
    inst = W.build(W.fall_spec(), "pick")
    data = tc.to_dict()
    hand = np.array(data["provenance"]["robot_body_paths"]["robot0_hand"])
    with_wrist = dict(data, provenance=dict(data["provenance"], robot_body_paths={
        "robot0_hand": hand.tolist(), "robot0_wrist": (hand + [0.0, 0.0, 0.08]).tolist()}))
    result = g0(inst, "pick", tc=TaskContext.from_dict(with_wrist))
    assert result.reasons == ["early_contact"] and result.checks["early_contact"]["body"] == "robot0_wrist"
    eef_only = dict(data, provenance=dict(data["provenance"], robot_body_paths=None))
    result = g0(inst, "pick", tc=TaskContext.from_dict(eef_only))
    assert result.passed and result.checks["early_contact_check"] == "eef_only"


def test_fall_release_above_the_workspace_is_rejected():
    inst = W.build(W.fall_spec(tau=0.5, phase="carry", frac=0.5), "pick")   # h = 1.23 m above z = 0.36
    result = g0(inst, "pick")
    assert result.reasons == ["release_height_out_of_workspace"]
    assert result.checks["release_height"]["release_z_m"] == pytest.approx(0.36 + W.fall_height(0.5))


def test_path_outside_the_workspace_is_rejected():
    inst = W.build(W.slide_spec(tau=0.3, speed=3.0), "low")                 # starts 0.9 m away, y < -0.65
    result = g0(inst, "low")
    assert result.reasons == ["out_of_workspace"]
    assert result.checks["out_of_workspace_points"] > 0


def test_penetration_of_a_support_surface_is_rejected():
    inst = W.build(W.reach_spec(direction=(0.0, 0.0, 1.0), frac=0.5, phase="approach"), "low")  # from below
    result = g0(inst, "low")
    assert "scene_penetration" in result.reasons
    assert result.checks["max_penetration_m"] > W.gate_settings().g0.max_penetration_m


def test_slide_leaving_its_support_is_rejected():
    tc = W.task_context("low")
    small = dict(W.TABLE_SURFACE, min_m=[-0.5, -0.3, 0.0], max_m=[0.5, 0.3, 0.0])
    narrow = TaskContext.from_dict(dict(tc.to_dict(), support_surfaces=[small]))
    inst = W.build(W.slide_spec(tau=0.3, speed=1.0), "low", tc=narrow)        # starts at y = -0.36
    result = g0(inst, "low", tc=narrow)
    assert result.reasons == ["slide_off_support"]


def test_r1_is_checked_only_without_ccd():
    small = W.reach_spec(speed=2.0, radius=0.005)                # 2 m/s x 2 ms = 4 mm > 0.5 x 5 mm
    with_ccd = W.make_host(ccd=True)
    without = W.make_host(ccd=False)
    assert with_ccd.describe().hazard_ccd is True and without.describe().hazard_ccd is False
    inst = W.build(small, "pick", host=without)
    rejected = g0(inst, "pick", host=without)
    assert rejected.reasons == ["r1_step_ratio"]
    assert rejected.checks["r1_check"]["dynamic_step_m"] > rejected.checks["r1_check"]["limit_m"]
    accepted = g0(inst, "pick", host=with_ccd)
    assert accepted.passed and accepted.checks["r1_check"] == "skipped: BD hazard_ccd is true"
    slow = W.build(W.reach_spec(speed=0.6, radius=0.005), "pick", host=without)
    assert g0(slow, "pick", host=without).passed


def test_r1_kinematic_phase_uses_needs_substeps():
    inst = W.build(W.reach_spec(speed=2.0, radius=0.005), "pick")
    result = g0(inst, "pick", host=W.make_host(ccd=False))
    assert result.checks["r1_check"]["kinematic_exceeds"] is True


def test_g0_settings_validate():
    with pytest.raises(ValueError):
        G0Settings(max_penetration_m=0.0, workspace_margin_m=0.0, early_contact_tol_m=0.0, max_step_ratio=0.5)


def test_reason_counts_appear_in_the_gate_output():
    host = W.make_host(ccd=False)
    narrow_tc = TaskContext.from_dict(dict(W.task_context("low").to_dict(), support_surfaces=[
        dict(W.TABLE_SURFACE, min_m=[-0.5, -0.3, 0.0], max_m=[0.5, 0.3, 0.0])]))
    instances = [
        W.build(W.fall_spec(tau=0.2, frac=0.1, spec_id="onset"), "pick", host=host),
        W.build(W.slide_spec(tau=0.1, speed=0.3, direction=(1.0, 0.0, 0.0), spec_id="chase"), "low", host=host),
        W.build(W.fall_spec(tau=0.5, phase="carry", spec_id="high"), "pick", host=host),
        W.build(W.slide_spec(tau=0.3, speed=3.0, spec_id="far"), "low", host=host),
        W.build(W.reach_spec(direction=(0.0, 0.0, 1.0), frac=0.5, spec_id="below"), "low", host=host),
        W.build(W.reach_spec(speed=2.0, radius=0.005, spec_id="fast"), "pick", host=host),
    ]
    output = gate_instances(W.session(host), instances, W.tc_map("pick", "low"), W.PathFollower(host), None,
                            W.gate_settings(), 0, 0)
    off = W.build(W.slide_spec(tau=0.3, speed=1.0, spec_id="off"), "low", host=host, tc=narrow_tc)
    output_off = gate_instances(W.session(host), [off], {("low", "L0"): narrow_tc}, W.PathFollower(host), None,
                                W.gate_settings(), 0, 0)
    assert output.instances == [] and output_off.instances == []
    counts = dict(output.summary["rejections_by_reason"])
    for reason, n in output_off.summary["rejections_by_reason"].items():
        counts[reason] = counts.get(reason, 0) + n
    assert set(counts) == set(G0_REASONS), sorted(set(G0_REASONS) - set(counts))
    assert output.summary["n_rejected"] == len(instances)
    assert {r["instance_id"] for r in output.rejections} == {i.instance_id for i in instances}
    assert all(r["stage"] == "g0" and r["reasons"] for r in output.rejections)
