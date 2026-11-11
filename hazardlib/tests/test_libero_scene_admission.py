"""D19 scene admission of the LIBERO analysis, without a simulator: clearance sweep, audit verdict,
selection and the ledger statistics."""
import pytest
import numpy as np

from hazardhosts.libero.scene_audit import audit_verdict, rejection_reason as audit_reason
from hazardhosts.libero.scene_clearance import (control_grid, drop_samples, playback_path, rejection_reason,
                                                subdivide, support_height, sweep, sweep_samples)
from hazardhosts.libero.trajectory_candidates import ledger_statistics, rejection_keys, select_admitted
from hazardlib.spec.instance import HazardInstance

QUAT = [1., 0., 0., 0.]


def instance(exit_=True):
    data = {
        "instance_id": "i", "spec_id": "s", "family": "intrusion.slide",
        "agent": {"kind": "object", "shape": "box", "half_extents_m": [.02, .02, .02], "mass_kg": .05},
        "keyframes": {"time_s": [0., .5, 1.], "pos": [[0., 0., 1.], [.1, 0., 1.], [.2, 0., 1.]], "quat": [QUAT] * 3},
        "nominal_hit_point": [.2, 0., 1.], "nominal_hit_time_s": 1., "onset_time_s": 0., "release_time_s": 1.,
        "release_velocity": {"lin_mps": [.2, 0., 0.], "ang_radps": [0., 0., 0.]},
        "certificate": {"g0": True, "g1": None, "g2": None, "label": "uncertified", "notes": ""}, "provenance": {}}
    if exit_:
        data["exit_keyframes"] = {"time_s": [1., 2., 3.], "pos": [[.2, 0., 1.], [.4, 0., 1.], [.6, 0., 1.]],
                                  "quat": [QUAT] * 3}
    return HazardInstance.from_dict(data)


def wall(x0, x1, name="cookies_1", depth=.05):
    """Overlap callback: a slab occupying x in [x0, x1]; the penetration is a constant ``depth``."""
    return lambda t, pos, quat: {name: depth} if x0 <= pos[0] <= x1 else {}


def test_playback_path_is_keyframes_then_exit_with_the_junction_once():
    path = playback_path(instance())
    assert path.t_start_s == 0. and path.t_end_s == 3.
    np.testing.assert_allclose(path.pos(1.5), [.3, 0., 1.])
    np.testing.assert_allclose(path.pos(.75), [.15, 0., 1.])
    # Without an exit path it is the keyframe path itself.
    assert playback_path(instance(False)).t_end_s == 1.


def test_control_grid_has_onset_steps_and_end_without_duplicates():
    grid = control_grid(.02, .27, .05)
    np.testing.assert_allclose(grid, [.02, .05, .1, .15, .2, .25, .27])
    np.testing.assert_allclose(control_grid(0., .1, .05), [0., .05, .1])


def test_subdivide_bounds_the_spacing():
    pos = lambda t: np.array([t, 0., 0.])
    out = subdivide(np.array([0., 1.]), pos, .3)
    assert len(out) == 5 and np.max(np.diff(out)) <= .3 + 1e-12 and out[0] == 0. and out[-1] == 1.
    np.testing.assert_allclose(subdivide(np.array([0., .1]), pos, .3), [0., .1])


def test_sweep_samples_cover_the_exit_path_and_respect_spacing():
    path = playback_path(instance())
    samples = sweep_samples(path, 0., path.t_end_s, .05, .02)
    times = np.array([s[0] for s in samples])
    xs = np.array([s[1][0] for s in samples])
    assert times[0] == 0. and times[-1] == 3. and xs[-1] == np.float64(.6)
    assert np.max(np.diff(xs)) <= .02 + 1e-12 and np.all(np.diff(times) > 0)
    assert all(np.allclose(s[2], QUAT) for s in samples)


def test_a_path_through_an_object_fails_and_names_the_partner_and_first_time():
    samples = sweep_samples(playback_path(instance()), 0., 3., .05, .02)
    result = sweep(samples, {"replay": wall(.45, .5)}, .001)  # only reached on the exit path
    assert not result["passed"] and result["partners"] == ["cookies_1"] and result["state"] == "replay"
    assert 2.1 < result["first_t_s"] < 2.3          # x = .45 is reached at t = 2.25 - spacing
    assert rejection_reason(result) == "scene_clearance:cookies_1"
    # The same slab behind the end of the path is not met; so is one passed by the path end only.
    assert sweep(samples, {"replay": wall(.7, .8)}, .001)["passed"]
    assert not sweep(samples, {"replay": wall(.59, .8)}, .001)["passed"]


def test_tolerance_and_state_attribution():
    samples = sweep_samples(playback_path(instance()), 0., 3., .05, .02)
    assert sweep(samples, {"replay": wall(0., 1., depth=.0005)}, .001)["passed"]
    result = sweep(samples, {"replay": wall(.5, .6, "plate_1"), "frozen": wall(.1, .15, "ramekin_1")}, .001)
    assert result["partners"] == ["plate_1", "ramekin_1"] and result["state"] == "frozen"
    assert set(result["by_state"]) == {"replay", "frozen"} and result["by_state"]["frozen"]["first_t_s"] < \
        result["by_state"]["replay"]["first_t_s"]
    only_replay = sweep(samples, {"replay": wall(.5, .6, "plate_1"), "frozen": wall(5., 6.)}, .001)
    assert only_replay["state"] == "replay" and list(only_replay["by_state"]) == ["replay"]


def test_drop_samples_fall_to_the_support_with_free_fall_times():
    samples = drop_samples([.1, .2, 1.4], .92, 9.81, 2.45, .02, t_after=9.)
    zs = np.array([s[1][2] for s in samples[:-1]])
    assert zs[0] == 1.4 and abs(zs[-1] - .92) < 1e-12 and np.max(-np.diff(zs)) <= .02 + 1e-12
    assert all(s[1][0] == .1 and s[1][1] == .2 for s in samples)
    assert abs(samples[0][0] - 2.45) < 1e-12
    assert abs(samples[-2][0] - (2.45 + np.sqrt(2 * .48 / 9.81))) < 1e-9
    assert samples[-1][0] == 9. and samples[-1][1][2] == zs[-1]          # landing pose at the replay's end
    assert drop_samples([0, 0, 1.], .92, 9.81, 0., .02)[-1][1][2] == .92
    # Released below the rest height (or at it): one sample, no negative drop.
    assert len(drop_samples([0, 0, .9], .92, 9.81, 0., .02)) == 2


def test_a_landing_column_over_an_object_fails():
    samples = drop_samples([.1, .2, 1.4], .92, 9.81, 2.45, .02)
    column = lambda t, pos, quat: {"plate_1": .01} if pos[2] < .95 else {}
    assert not sweep(samples, {"replay": column, "frozen": lambda *a: {}}, .001)["passed"]
    assert sweep(samples, {"replay": lambda *a: {}}, .001)["passed"]


def test_support_height_uses_the_surface_under_the_point():
    surfaces = [{"min_m": [-.5, -.5, .9], "max_m": [.5, .5, .9]}, {"min_m": [1., 1., 1.2], "max_m": [2., 2., 1.2]}]
    assert support_height(surfaces, [0., 0.]) == .9 and support_height(surfaces, [1.5, 1.5]) == 1.2
    assert support_height(surfaces, [9., 9.]) == 1.2


def audit(**over):
    record = {"hold_complete": True, "kinematic_overlap": {}, "released_contact": {}, "objects_moved_m": {"a": .001}}
    record.update(over)
    return record


def test_audit_verdict():
    assert audit_verdict(audit(), .002) == []
    assert audit_verdict(audit(kinematic_overlap={"cookies_1": [1., 2.]}, released_contact={"plate_1": [3., 4.]},
                               objects_moved_m={"a": .003, "b": .002}), .002) == [
        "kinematic_overlap:cookies_1", "released_contact:plate_1", "moved:a"]
    assert audit_verdict(audit(hold_complete=False), .002) == ["hold_incomplete"]
    assert audit_reason(["moved:a", "released_contact:b"]) == "scene_audit:moved:a;released_contact:b"


def test_selection_takes_the_first_admitted_and_budgets_only_attempted_candidates():
    items = [{} for _ in range(6)]
    shortlist = [(1., "i%d" % k, item) for k, item in enumerate(items)]
    outcomes = {"i0": (False, False), "i1": (True, False), "i2": (True, False), "i3": (True, True), "i4": (True, True)}
    chosen, attempts = select_admitted(shortlist, lambda inst, item: outcomes.get(inst, (True, False)), 3)
    assert chosen[1] == "i3" and attempts == 3          # i0 (e.g. G0 failure) used no attempt
    assert [i.get("not_evaluated") for i in items] == [None] * 4 + ["selected_earlier"] * 2
    # Budget exhausted before any admission.
    items = [{} for _ in range(5)]
    shortlist = [(1., "i%d" % k, item) for k, item in enumerate(items)]
    chosen, attempts = select_admitted(shortlist, lambda inst, item: (True, False), 2)
    assert chosen is None and attempts == 2
    assert [i.get("not_evaluated") for i in items] == [None, None, "attempt_budget", "attempt_budget", "attempt_budget"]
    assert select_admitted([], lambda *a: (True, True), 3) == (None, 0)
    # Every candidate evaluated, none admitted, budget not reached.
    items = [{}, {}]
    assert select_admitted([(1., "a", items[0]), (0., "b", items[1])], lambda i, it: (False, False), 3) == (None, 0)
    assert all("not_evaluated" not in i for i in items)


def test_rejection_keys_split_partners_and_reasons():
    assert rejection_keys("scene_clearance:cookies_1,plate_1") == (
        "scene_clearance", ["scene_clearance:cookies_1", "scene_clearance:plate_1"])
    assert rejection_keys("scene_audit:moved:a;released_contact:b")[0] == "scene_audit"
    assert rejection_keys("scene_audit:released_contact:b;kinematic_overlap:c")[1][0] == "scene_audit:released_contact:b"
    assert rejection_keys("outside_declared_pilot_range: tau_s") == ("outside_declared_pilot_range", [])
    assert rejection_keys("g0:early_contact,r1_step_ratio")[1] == ["g0:early_contact", "g0:r1_step_ratio"]


def test_ledger_statistics():
    sel = {"spec": {}, "selected": True, "nominal_hit": {"release_kind": "nominal"},
           "scene_audit": {"exited": True, "released": False, "release_kind": None, "hold_complete": True,
                           "success": True, "admitted": True}, "g1": {"passed": False}}
    ledger = [sel,
              {"rejection": "scene_clearance:cookies_1"}, {"rejection": "scene_clearance:cookies_1,plate_1"},
              {"rejection": "scene_audit:moved:a", "nominal_hit": {"release_kind": "rollback"},
               "scene_audit": {"exited": False, "released": True, "release_kind": "rollback",
                               "hold_complete": True, "success": False, "admitted": False}},
              {"not_evaluated": "selected_earlier"}]
    stats = ledger_statistics([ledger, []])
    assert stats["rejections_by_category"] == {"scene_clearance": 2, "scene_audit": 1}
    assert stats["rejections_by_detail"]["scene_clearance:cookies_1"] == 2
    assert stats["not_evaluated"] == {"selected_earlier": 1}
    assert stats["nominal_release_kinds_all_attempts"] == {"nominal": 1, "rollback": 1}
    assert stats["nominal_release_kinds_selected"] == {"nominal": 1}
    assert stats["audit_all_attempts"]["n"] == 2 and stats["audit_all_attempts"]["exited"] == 1
    assert stats["audit_selected"]["admitted"] == 1 and stats["g1_diagnostic_selected"] == {"n": 1, "passed": 0}


# ---- L2: per-(family, Sigma target) results, report aggregation, T_stop record ------------------------------

def _cell_result(status, sigma, tau=None, lead=None, g1=None, unreachable=None, n_candidates=3):
    selected = None
    if status in ("wait_solvable", "uncertified"):
        selected = {"g1": {"passed": g1}, "sigma": {"tau_s": tau, "sigma_achieved": tau / .3,
                                                      "visible_lead_s": lead}}
    return {"sigma_target": sigma, "status": status, "unreachable": unreachable, "selected": selected,
            "n_candidates": 0 if unreachable else n_candidates, "n_scene_clear": 1, "n_visible": 1,
            "dynamic_attempts": 1}


def test_cell_statistics_aggregate_layouts_per_family_and_target():
    from hazardhosts.libero.trajectory_candidates import cell_statistics
    q = [.5]
    ledger = [{"rejection": "visibility:insufficient_visible_lead"}, {"rejection": "scene_clearance:cookies_1"},
              {"rejection": "visibility:insufficient_visible_lead"}]
    entries = [(_cell_result("wait_solvable", .75, .2, .1, True), ledger),
               (_cell_result("uncertified", .75, .2, .15, False), []),
               (_cell_result("unreachable", .75, unreachable="phase_time"), []),
               (_cell_result("unreachable", .75, unreachable="phase_time"), []),
               (_cell_result("unreachable", .75, unreachable="free_fall_height"), []),
               (_cell_result("no_admitted_candidate", .75), [])]
    stats = cell_statistics(entries, q)
    assert (stats["n_layouts"], stats["admitted"], stats["wait_solvable"], stats["uncertified"]) == (6, 2, 1, 1)
    assert stats["unreachable"] == 3 and stats["unreachable_by_reason"] == {"free_fall_height": 1, "phase_time": 2}
    assert stats["no_admitted_candidate"] == 1
    assert stats["g1_diagnostic"] == {"n": 2, "passed": 1, "pass_rate": .5}
    assert stats["top_rejection_categories"] == {"visibility": 2, "scene_clearance": 1}
    assert stats["top_rejection_details"]["visibility:insufficient_visible_lead"] == 2
    assert stats["tau_s"]["quantiles"]["0.5"] == pytest.approx(.2)
    assert stats["sigma_achieved"]["quantiles"]["0.5"] == pytest.approx(.2 / .3)
    assert stats["visible_lead_s"]["quantiles"]["0.5"] == pytest.approx(.125)
    empty = cell_statistics([(_cell_result("unreachable", 3., unreachable="t_stop_censored"), [])], q)
    assert empty["admitted"] == 0 and empty["g1_diagnostic"]["pass_rate"] is None and empty["tau_s"]["n"] == 0


def test_t_stop_statistics_exclude_censored_layouts():
    from hazardhosts.libero.trajectory_candidates import t_stop_statistics
    records = [{"status": "ok", "median_s": .3}, {"status": "ok", "median_s": .35}, {"status": "ok", "median_s": .25},
               {"status": "censored", "median_s": .1}, {"status": "protocol_failed", "error": "x"}]
    stats = t_stop_statistics(records)
    assert stats["median_s"] == {"min": .25, "median": .3, "max": .35}
    assert stats["n_censored_layouts"] == 1 and stats["status"] == {"censored": 1, "ok": 3, "protocol_failed": 1}


def test_each_target_has_its_own_selection_and_attempt_budget():
    # select_admitted is called once per (family, layout, target): a target that exhausts its attempts does not
    # take the budget of another, and each target can admit its own instance.
    def cell(admit_at):
        items = [{} for _ in range(5)]
        shortlist = [(1., "i%d" % k, item) for k, item in enumerate(items)]
        return select_admitted(shortlist, lambda inst, item: (True, inst == admit_at), 2)[0]
    chosen = [cell("i1"), cell("i4"), cell("i0")]      # target 2 never reaches i4 (2 attempts)
    assert [c[1] if c else None for c in chosen] == ["i1", None, "i0"]


def test_t_stop_calibration_is_cached_and_a_censored_value_is_never_targeted(tmp_path, monkeypatch):
    import hazardhosts.libero.trajectory_candidates as module
    from hazardlib.calibrate.protocol import CalibrationResult, TStopProtocolSettings
    from hazardlib.calibrate.stop import aggregate_t_stop
    calls = []

    class Context:
        def hash(self):
            return "h"

    def fake(session, policy, tc, settings, seed, env):
        calls.append(seed)
        samples = [.3, None, .3] if len(calls) == 1 else [.3, .3, .3]
        return CalibrationResult("t", "l", "h", "b", "libero", {"kind": "x"}, .2, .16,
                                 TStopProtocolSettings(.0172, 1, 3, .8, 1.5), samples, [.1] * 3,
                                 aggregate_t_stop(samples))
    monkeypatch.setattr(module, "run_t_stop_protocol", fake)
    monkeypatch.setattr(module.TStopProtocolSettings, "load", classmethod(lambda cls, d: None))
    record, summary = module.t_stop_calibration(tmp_path, None, None, Context(), None, 0)
    assert record["status"] == "censored" and summary is None and record["n_censored"] == 1
    # Resumed: the stored protocol output is read back, the protocol does not run again.
    record, summary = module.t_stop_calibration(tmp_path, None, None, Context(), None, 0)
    assert record["status"] == "censored" and summary is None and len(calls) == 1
    (tmp_path / "calibration.yaml").unlink()
    record, summary = module.t_stop_calibration(tmp_path, None, None, Context(), None, 0)
    assert record["status"] == "ok" and summary.median_s == pytest.approx(.3) and len(calls) == 2
    # A protocol that cannot run (the EEF never reaches the trigger speed) is recorded, not retried.
    (tmp_path / "calibration.yaml").unlink()
    monkeypatch.setattr(module, "run_t_stop_protocol", lambda *a: (_ for _ in ()).throw(RuntimeError("never fast")))
    record, summary = module.t_stop_calibration(tmp_path, None, None, Context(), None, 0)
    assert record["status"] == "protocol_failed" and summary is None
    assert module.t_stop_calibration(tmp_path, None, None, Context(), None, 0)[0]["error"] == "never fast"
