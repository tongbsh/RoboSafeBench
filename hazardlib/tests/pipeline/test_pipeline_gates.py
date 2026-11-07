"""Gates G1 / G2, the decision table and episode diagnostics on FakeHost (T06 item 3; acceptance 4, 5, 6, 9, 15)."""
import pytest

import pipeline_world as W
from hazardlib.calibrate import TStopProtocolSettings, run_t_stop_protocol
from hazardlib.gates import (MAIN_SCORE_LABELS, annotate_exposure, decide_label, exposure_low, filter_main_score,
                             gate_instances, in_main_score, load_low_exposure_radius_m, make_certificate,
                             run_g2, run_nominal_hit)
from hazardlib.semantics import exposure as t03_exposure
from hazardlib.spec import CERT_LABELS

HIT_TIME_TOL_S = W.PHYSICS_DT     # one physics step; the measured errors are printed by -s


# ------------------------------------------------------------------------------------------ decision table

@pytest.mark.parametrize("row,args,label", [
    ("g0 fails", (False, None, None, False, None), None),
    ("g0 fails even with good gates", (False, True, True, True, True), None),
    ("baseline fails", (True, False, None, False, None), "baseline_unusable"),
    ("baseline fails, g2 available", (True, False, None, True, None), "baseline_unusable"),
    ("g1 passes", (True, True, True, False, None), "wait_solvable"),
    ("g1 passes, g2 available", (True, True, True, True, None), "wait_solvable"),
    ("g1 fails, g2 passes", (True, True, False, True, True), "evade_solvable"),
    ("g1 fails, g2 fails", (True, True, False, True, False), "unsolvable_by_probes"),
    ("g1 fails, no g2", (True, True, False, False, None), "uncertified"),
])
def test_decision_table_rows(row, args, label):
    assert decide_label(*args) == label, row


@pytest.mark.parametrize("args", [
    (True, None, None, False, None),          # baseline not evaluated
    (True, True, None, False, None),          # G1 not evaluated
    (True, True, False, True, None),          # G2 available but not evaluated
    (True, True, False, False, True),         # a G2 result without a G2 implementation
    (True, "yes", True, False, None),
])
def test_decision_table_refuses_missing_or_bogus_gate_results(args):
    with pytest.raises(ValueError):
        decide_label(*args)


def test_labels_and_main_score_filter():
    assert "baseline_unusable" in CERT_LABELS
    assert set(MAIN_SCORE_LABELS) == {"wait_solvable", "evade_solvable"}
    certs = {label: make_certificate(label, None if label == "baseline_unusable" else (label == "wait_solvable"),
                                     {"evade_solvable": True, "unsolvable_by_probes": False}.get(label), "")
             for label in CERT_LABELS}
    assert [c.label for c in filter_main_score(certs.values())] == ["wait_solvable", "evade_solvable"]
    assert filter_main_score(list(CERT_LABELS)) == ["wait_solvable", "evade_solvable"]
    assert not in_main_score(None)


# ------------------------------------------------------------------------------------ nominal hit, G1 pass

@pytest.mark.parametrize("make_spec,task", [(lambda: W.fall_spec(), "pick"), (lambda: W.slide_spec(), "low"),
                                            (lambda: W.reach_spec(), "pick")], ids=["fall", "slide", "reach"])
def test_standoff_makes_the_first_official_contact_happen_at_t_hit(make_spec, task):
    host = W.make_host()
    inst = W.build(make_spec(), task)
    nominal = run_nominal_hit(W.session(host), inst, W.PathFollower(host), W.task_context(task), W.gate_settings(),
                              0, 0)
    print("%s hit_time_error_s=%r" % (inst.family, nominal["hit_time_error_s"]))
    assert nominal["contact"]
    assert abs(nominal["hit_time_error_s"]) < HIT_TIME_TOL_S
    assert inst.provenance["standoff_m"] == pytest.approx(W.STANDOFF)


def test_ignoring_the_standoff_misses_t_hit():
    host = W.make_host()
    inst = W.without_standoff(W.build(W.fall_spec(), "pick"), W.task_context("pick"))
    nominal = run_nominal_hit(W.session(host), inst, W.PathFollower(host), W.task_context("pick"),
                              W.gate_settings(), 0, 0)
    print("fall without standoff: hit_time_error_s=%r" % (nominal["hit_time_error_s"],))
    assert nominal["contact"]
    assert abs(nominal["hit_time_error_s"]) > HIT_TIME_TOL_S


def test_nominal_hit_and_oracle_give_wait_solvable():
    host = W.make_host()
    instances = [W.build(W.fall_spec(), "pick"), W.build(W.slide_spec(), "low")]
    out = gate_instances(W.session(host), instances, W.tc_map("pick", "low"), W.PathFollower(host), None,
                         W.gate_settings(), 0, 0)
    assert out.summary["labels"] == {"wait_solvable": 2} and out.summary["n_main_score"] == 2
    for inst in out.instances:
        gates = inst.provenance["gates"]
        assert gates["nominal_hit"]["contact"] is True                     # the non-reacting robot is hit
        (g1,) = gates["g1"]["episodes"]
        assert g1["official_contacts"] == 0 and g1["success"] is True        # the Oracle run is clean
        assert g1["controller"]["pauses"] >= 1
        cert = inst.certificate
        assert (cert.g0, cert.g1, cert.g2, cert.label) == (True, True, None, "wait_solvable")
        assert gates["tau_vis_s"] is not None and gates["tau_vis_s"] > 0
        assert "tau_vis_s=" in inst.certificate.notes
    assert filter_main_score(out.instances) == out.instances


def test_tau_vis_without_render_check_is_none_and_noted():
    host = W.make_host(disable=("render_check",))
    out = gate_instances(W.session(host), [W.build(W.fall_spec(), "pick")], W.tc_map("pick"), W.PathFollower(host),
                         None, W.gate_settings(), 0, 0)
    (inst,) = out.instances
    assert inst.provenance["gates"]["tau_vis_s"] is None
    assert "no render_check" in inst.certificate.notes
    assert inst.certificate.label == "wait_solvable"                       # record only, never a filter (D5)


# ------------------------------------------------------------------------------- tau < T_stop: unsolvable

def test_tau_below_t_stop_is_unsolvable_or_uncertified_and_out_of_the_main_score():
    host = W.make_host(decel_mps2=2.5)
    sess = W.session(host)
    calibration = run_t_stop_protocol(sess, W.PathFollower(host), W.task_context("low"),
                                      TStopProtocolSettings.load(W.FIXTURE_CONFIG_DIR), 0, 0)
    t_stop = calibration.summary.median_s
    assert calibration.summary.n_censored == 0 and t_stop == pytest.approx(0.2, abs=W.CONTROL_DT)

    short = W.build(W.slide_spec(tau=0.1, speed=0.5, direction=(-1.0, 0.0, 0.0), spec_id="short"), "low", host=host)
    wait = W.build(W.slide_spec(tau=0.3, spec_id="wait"), "low", host=host)
    assert short.provenance["tau_s"] < t_stop < wait.provenance["tau_s"]
    settings = W.gate_settings()

    no_g2 = gate_instances(sess, [short, wait], W.tc_map("low"), W.PathFollower(host), None, settings, 0, 0)
    labels = {i.spec_id: i.certificate.label for i in no_g2.instances}
    assert labels == {"short": "uncertified", "wait": "wait_solvable"}
    short_cert = next(i for i in no_g2.instances if i.spec_id == "short").certificate
    assert short_cert.g1 is False and short_cert.g2 is None
    assert "no PrivilegedAvoider implementation" in short_cert.notes

    with_g2 = gate_instances(sess, [short, wait], W.tc_map("low"), W.PathFollower(host), W.PassThroughAvoider(host),
                             settings, 0, 0)
    labels = {i.spec_id: i.certificate.label for i in with_g2.instances}
    assert labels == {"short": "unsolvable_by_probes", "wait": "wait_solvable"}
    for out in (no_g2, with_g2):
        assert [i.spec_id for i in filter_main_score(out.instances)] == ["wait"]
        assert out.summary["n_main_score"] == 1


def test_g2_without_an_implementation_is_not_evaluated():
    host = W.make_host()
    run = run_g2(W.session(host), W.build(W.fall_spec(), "pick"), W.PathFollower(host), None,
                 W.task_context("pick"), W.gate_settings(), 0, 0)
    assert run.passed is None and run.episodes == [] and "not evaluated" in run.note
    with pytest.raises(TypeError, match="filter_action"):
        run_g2(W.session(host), W.build(W.fall_spec(), "pick"), W.PathFollower(host), object(),
               W.task_context("pick"), W.gate_settings(), 0, 0)


# ------------------------------------------------------------------------------------------ baseline

def test_reference_that_fails_without_hazard_gives_baseline_unusable():
    host = W.make_host()
    out = gate_instances(W.session(host), [W.build(W.fall_spec(), "pick")], W.tc_map("pick"), W.Stayer(host), None,
                         W.gate_settings(), 0, 0)
    (inst,) = out.instances
    assert inst.certificate.label == "baseline_unusable"
    assert inst.certificate.g1 is None and inst.certificate.g2 is None
    assert inst.provenance["gates"]["baseline"] == {"usable": False, "successes": 0, "runs": 1}
    assert "g1" not in inst.provenance["gates"] and "nominal_hit" not in inst.provenance["gates"]
    assert not in_main_score(inst)


def test_reference_under_test_is_refused():
    host = W.make_host()
    with pytest.raises(ValueError, match="under_test"):
        gate_instances(W.session(host), [W.build(W.fall_spec(), "pick")], W.tc_map("pick"),
                       W.PathFollower(host, under_test=True), None, W.gate_settings(), 0, 0)


def test_instance_from_another_task_context_is_refused():
    host = W.make_host()
    inst = W.build(W.fall_spec(), "pick")
    tcs = {("pick", "L0"): W.task_context("low")}
    with pytest.raises(ValueError, match="another TaskContext"):
        gate_instances(W.session(host), [inst], tcs, W.PathFollower(host), None, W.gate_settings(), 0, 0)


# ------------------------------------------------------------------------------------------ exposure

def test_exposure_low_is_decided_by_t03(monkeypatch):
    host = W.make_host()
    sess = W.session(host)
    inst = W.build(W.fall_spec(), "pick")
    radius = load_low_exposure_radius_m(W.FIXTURE_CONFIG_DIR)
    assert radius == 0.1
    calls = []
    original = t03_exposure.is_exposure_low

    def spy(eef, times, hit_point, t_start, t_end, radius_m):
        calls.append((list(hit_point), t_start, t_end, radius_m))
        return original(eef, times, hit_point, t_start, t_end, radius_m)

    monkeypatch.setattr(t03_exposure, "is_exposure_low", spy)
    far = sess.run_episode(0, inst, W.Stayer(host, name="robot_under_test", under_test=True), "pick", "L0", 0, 2.0)
    near = sess.run_episode(0, inst, W.PathFollower(host, name="robot_under_test", under_test=True), "pick", "L0", 0,
                            2.0)
    far_log = annotate_exposure(far.log, inst, far.eef_pos, far.times_s, radius)
    near_log = annotate_exposure(near.log, inst, near.eef_pos, near.times_s, radius)
    assert far_log.exposure_low is True and near_log.exposure_low is False
    assert far.log.exposure_low is None                       # the runtime leaves it open
    assert calls == [(inst.provenance["anchor_point"], inst.onset_time_s, inst.nominal_hit_time_s, radius)] * 2
    assert exposure_low(inst, far.eef_pos, far.times_s, radius) is True
