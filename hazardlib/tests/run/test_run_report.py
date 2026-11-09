"""``hazardctl report`` on a small hand-made data set: every number checked by hand (R1 items 5, 6)."""
import json

import pytest
import yaml

from hazardlib.run.report import ReportInputError, load_runs, render_markdown, summarize, write_report
from hazardlib.run.runner import EPISODES_FILE, RECORD_VERSION, RUN_META_FILE
from hazardlib.run.settings import ReportSettings
from hazardlib.semantics.event_log import HazardEventLog

SETTINGS = ReportSettings(tau_bin_edges_s=[0.2, 0.4], sigma_bin_edges=[1.0, 2.0], interval_method="wilson",
                          confidence=0.95)
T_STOP = 0.2
META = {"meta_version": "1", "pilot": False, "manifest_hash": "a" * 64, "benchmark": "toy", "bd_hash": "b" * 64,
        "calibration_hash": "c" * 64, "policy_name": "p", "run_config_hash": "d" * 64,
        "episode_settings": {"horizon_extra_s": 1.0, "stop_on_success": False}}


def _log(episode_id, contact=False, j=None, vn=None, sep=None, ttc=None, clamp=False, toppled=None,
         exposure=None, invalid="none", force=None):
    return HazardEventLog.new(
        episode_id, first_official_contact_t_s=0.5 if contact else None, impulse_analytic_ns=j, v_rel_normal_mps=vn,
        min_separation_m=sep, ttc_min_s=ttc, constraint_state=("clamp" if clamp else "unconstrained") if contact
        else None, toppled=toppled, exposure_low=exposure, invalid_class=invalid, attribution_ambiguous=False,
        engine_peak_force_n=force, force_source="toy_engine" if force is not None else None).to_dict()


def _hazard(name, label, tau, control, success, seed=0, policy="p", **log):
    return {"record_version": RECORD_VERSION, "kind": "hazard", "episode_id": name, "instance_id": name,
            "control_id": control, "policy_name": policy, "seed": seed, "label": label, "family": "intrusion.fall",
            "tau_s": tau, "sigma": tau / T_STOP, "sigma_is_upper_bound": False,
            "success_t_s": 1.0 if success else None, "pilot": False, "manifest_hash": "a" * 64,
            "log": _log(name, **log)}


def _control(name, success, toppled, seed=0, policy="p", invalid="none"):
    return {"record_version": RECORD_VERSION, "kind": "control", "episode_id": name, "instance_id": None,
            "control_id": name, "policy_name": policy, "seed": seed, "label": None, "family": None, "tau_s": None,
            "sigma": None, "sigma_is_upper_bound": None, "success_t_s": 1.0 if success else None, "pilot": False,
            "manifest_hash": "a" * 64, "log": _log(name, toppled=toppled, invalid=invalid)}


RECORDS = [
    _control("c1", True, False),
    _hazard("h1", "wait_solvable", 0.30, "c1", True, contact=True, j=1.0, vn=2.0, sep=0.0, ttc=0.0, clamp=True,
            toppled=True, exposure=False, force=10.0),
    _hazard("h2", "wait_solvable", 0.30, "c1", False, contact=True, j=3.0, vn=4.0, sep=0.0, ttc=0.1, toppled=False,
            exposure=False, invalid="hazard_induced_unstable", force=30.0),
    _control("c2", False, True),
    _hazard("h3", "wait_solvable", 0.35, "c2", True, sep=0.2, ttc=0.5, toppled=False, exposure=False),
    _hazard("h4", "wait_solvable", 0.30, "c1", False, sep=0.4, exposure=True),
    _hazard("h5", "evade_solvable", 0.25, "c2", False, exposure=False, invalid="setup_invalid"),
    _control("c3", True, None),
    _hazard("h6", "uncertified", 0.10, "c3", True, contact=True, j=5.0, vn=6.0, sep=0.0, ttc=0.0, exposure=False),
    _hazard("h7", "unsolvable_by_probes", 0.30, "c1", False, contact=True, j=7.0, vn=8.0, exposure=False),
    _hazard("h8", "baseline_unusable", 0.30, "c1", False, exposure=False),
    _control("c4", True, False, invalid="sim_unstable"),
]


def _wilson(k, n):   # z = 1.96 by hand; the stats module is checked against R separately
    z, p = 1.959963984540054, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [c - h, c + h]


def _write_run(path, meta, records):
    path.mkdir()
    (path / RUN_META_FILE).write_text(yaml.safe_dump(meta, sort_keys=False))
    (path / EPISODES_FILE).write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_every_number_of_the_main_row_by_hand(tmp_path):
    metas, records = load_runs([_write_run(tmp_path / "r", META, RECORDS)])
    summary = summarize(metas, records, SETTINGS, "e" * 64)
    assert summary["counts"] == {"hazard_episodes": 8, "control_episodes": 4,
                                 "placement": {"baseline_unusable": 1, "exposure_low": 1, "invalid": 1, "main": 4,
                                               "unsolvable_by_probes": 1}}
    rows = {(r["cert"], r["tau_bin_s"]): r for r in summary["main_tau"]}
    assert list(rows) == [("certified", "[0.2,0.4)"), ("uncertified", "<0.2")]
    row = rows[("certified", "[0.2,0.4)")]
    # eligible: h1-h5; main: h1, h2, h3 (h4 exposure_low, h5 setup_invalid; h2's hazard_induced_unstable stays)
    assert (row["policy"], row["family"], row["n_eligible"], row["n_episodes"]) == ("p", "intrusion.fall", 5, 3)
    assert row["success"] == {"k": 2, "n": 3, "rate": 2 / 3, "ci": pytest.approx(_wilson(2, 3))}
    assert row["success"]["ci"] == pytest.approx([0.2077, 0.9385], abs=1e-4)
    # paired controls: c1 (h1, h2) and c2 (h3); c1 succeeded, c2 did not
    assert (row["n_controls"], row["n_controls_invalid"], row["n_controls_missing"]) == (2, 0, 0)
    assert row["control_success"]["k"] == 1 and row["control_success"]["n"] == 2
    assert row["official_contact"]["k"] == 2 and row["official_contact"]["n"] == 3
    assert row["impulse_analytic_ns"] == {"n": 2, "median": 2.0, "q1": 1.5, "q3": 2.5}
    assert row["v_rel_normal_mps"] == {"n": 2, "median": 3.0, "q1": 2.5, "q3": 3.5}
    assert row["min_separation_m"]["median"] == 0.0 and row["min_separation_m"]["n"] == 3
    assert row["ttc_min_s"]["median"] == 0.1
    assert (row["clamp"]["k"], row["clamp"]["n"]) == (1, 3)
    assert (row["toppled"]["k"], row["toppled"]["n"]) == (1, 3)
    assert (row["control_toppled"]["k"], row["control_toppled"]["n"]) == (1, 2)
    assert row["toppled_net"] == pytest.approx(1 / 3 - 1 / 2)
    assert (row["exposure_low"]["k"], row["exposure_low"]["n"]) == (1, 5)
    assert row["exposure_low"]["ci"] == pytest.approx(_wilson(1, 5))
    assert row["invalid"] == {"n_total": 5, "n_none": 3, "n_setup_invalid": 1, "n_sim_unstable": 0,
                              "n_hazard_induced_unstable": 1, "n_attribution_ambiguous": 0, "n_in_denominator": 4}
    footnoted = rows[("uncertified", "<0.2")]
    assert footnoted["n_episodes"] == 1 and footnoted["success"]["k"] == 1
    assert footnoted["toppled"]["n"] == 0 and footnoted["control_toppled"]["n"] == 0
    assert footnoted["toppled_net"] is None and footnoted["control_success"]["k"] == 1


def test_buckets_controls_sigma_and_appendix(tmp_path):
    metas, records = load_runs([_write_run(tmp_path / "r", META, RECORDS)])
    summary = summarize(metas, records, SETTINGS, "e" * 64)
    assert [(b["bucket"], b["n_episodes"]) for b in summary["buckets"]] == [
        ("unsolvable_by_probes", 1), ("baseline_unusable", 1), ("invalid", 1), ("exposure_low", 1)]
    unavoidable = summary["buckets"][0]
    assert unavoidable["official_contact"]["k"] == 1 and unavoidable["impulse_analytic_ns"]["median"] == 7.0
    assert summary["buckets"][2]["invalid"]["n_setup_invalid"] == 1
    assert summary["controls"] == [{
        "policy": "p", "n_controls": 4, "success": {"k": 2, "n": 3, "rate": 2 / 3, "ci": pytest.approx(_wilson(2, 3))},
        "toppled": {"k": 1, "n": 2, "rate": 0.5, "ci": pytest.approx(_wilson(1, 2))},
        "invalid": {"n_none": 3, "n_setup_invalid": 0, "n_sim_unstable": 1}}]
    sigma = {(r["cert"], r["sigma_bin"]): r for r in summary["main_sigma"]}
    assert list(sigma) == [("certified", "[1,2)"), ("uncertified", "<1")]
    tau_row = summary["main_tau"][0]
    assert {k: v for k, v in sigma[("certified", "[1,2)")].items() if k != "sigma_bin"} == \
        {k: v for k, v in tau_row.items() if k != "tau_bin_s"}
    assert summary["sigma_is_upper_bound"] is False
    appendix = summary["appendix_engine_peak_force"]
    assert appendix[0]["engine_peak_force_n"] == {"n": 2, "median": 20.0, "q1": 15.0, "q3": 25.0}
    assert appendix[0]["force_source"] == ["toy_engine"] and appendix[1]["engine_peak_force_n"]["n"] == 0
    assert all("engine_peak_force_n" not in row for row in summary["main_tau"])
    text = render_markdown(summary)
    assert "| p | certified | intrusion.fall | [0.2,0.4) | 3 | 0.67 [0.21, 0.94] (2/3) |" in text
    assert "uncertified†" in text and "† uncertified: G1 failed" in text and "PILOT" not in text
    paths = write_report(tmp_path / "rep", summary)
    assert yaml.safe_load(open(paths["summary"]).read())["main_tau"][0]["n_episodes"] == 3


def test_tau_bins_are_closed_open_with_open_ends(tmp_path):
    records = [_control("c1", True, False)] + [
        _hazard("x%d" % k, "wait_solvable", tau, "c1", True, exposure=False)
        for k, tau in enumerate([0.1, 0.2, 0.3999, 0.4, 0.9])]
    metas, records = load_runs([_write_run(tmp_path / "r", META, records)])
    rows = summarize(metas, records, SETTINGS, "e" * 64)["main_tau"]
    assert [(r["tau_bin_s"], r["n_episodes"]) for r in rows] == [("<0.2", 1), ("[0.2,0.4)", 2), (">=0.4", 2)]


@pytest.mark.parametrize("field, value, message", [
    ("pilot", True, "pilot and formal results cannot be mixed"),
    ("manifest_hash", "f" * 64, "results of different Manifests cannot be mixed"),
    ("calibration_hash", "0" * 64, "calibration_hash: they differ"),
    ("episode_settings", {"horizon_extra_s": 2.0, "stop_on_success": False}, "episode_settings"),
])
def test_runs_that_differ_are_not_mixed(tmp_path, field, value, message):
    first = _write_run(tmp_path / "a", META, RECORDS[:2])
    other_records = [dict(r, policy_name="q") for r in RECORDS[:2]]
    if field in ("pilot", "manifest_hash"):
        other_records = [dict(r, **{field: value}) for r in other_records]
    second = _write_run(tmp_path / "b", dict(META, policy_name="q", **{field: value}), other_records)
    with pytest.raises(ReportInputError, match=message):
        load_runs([first, second])
    load_runs([first, _write_run(tmp_path / "c", dict(META, policy_name="q"),
                                 [dict(r, policy_name="q") for r in RECORDS[:2]])])


def test_duplicates_and_inconsistent_records_are_refused(tmp_path):
    a = _write_run(tmp_path / "a", META, RECORDS[:2])
    b = _write_run(tmp_path / "b", META, RECORDS[1:3])
    with pytest.raises(ReportInputError, match="episode \\('h1', 'p', 0\\) appears in"):
        load_runs([a, b])
    c = _write_run(tmp_path / "c", META, [dict(RECORDS[1], pilot=True)])
    with pytest.raises(ReportInputError, match="has pilot=True, its run_meta.yaml False"):
        load_runs([c])
    with pytest.raises(ReportInputError, match="not a hazardctl run directory"):
        load_runs([tmp_path])
