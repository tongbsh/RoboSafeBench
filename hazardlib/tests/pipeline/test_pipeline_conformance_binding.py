"""``check_conformance_report`` (T06-F, D13 a): each refusal reason on a real report changed field by field, all
reasons at once, and a report generated on the repository's configuration (``runtime.yaml`` still PLACEHOLDER),
which the suite runs on its ``probe_runtime`` / ``probe_calibration`` fallbacks."""
import pytest

import pipeline_world as W
from pipeline_world import conformance_dir, conformance_reports  # noqa: F401 (fixtures)
from hazardlib.manifest import FALLBACK_SOURCE_PREFIX, REPORT_CONFIGS, check_conformance_report, config_hashes

PROBE_RUNTIME = "conformance.yaml:probe_runtime.restitution_e"
PROBE_CALIBRATION = "conformance.yaml:probe_calibration.t_stop_speed_threshold_mps"


def check(report, bd=None, adapter_version=None, hashes=None, adapter_of=None):
    """The problems of ``report`` for a freeze of the pipeline host on the fixture configs (by default with the
    adapter version of ``adapter_of``, the unmodified report)."""
    adapter = adapter_version if adapter_version is not None else (adapter_of or report)["adapter_version"]
    return check_conformance_report(report, bd or W.make_host().describe(), adapter,
                                    hashes if hashes is not None else config_hashes(W.FIXTURE_CONFIG_DIR))


def only(problems, fragment):
    assert len(problems) == 1 and fragment in problems[0], problems
    return problems[0]


def c10_settings(report):
    (c10,) = [c for c in report["checks"] if c["id"] == "C10"]
    return c10["measured"]["t_stop_settings"]


def test_the_real_report_is_accepted(conformance_reports):
    report = conformance_reports()
    assert report["verdict"] == "pass" and report["summary"]["pass"] == len(report["checks"])
    assert check(report) == []


@pytest.mark.parametrize("verdict", ["fail", "unverified"])
def test_a_verdict_other_than_pass(conformance_reports, verdict):
    report = conformance_reports()
    report["verdict"] = verdict
    only(check(report), "verdict is %r, not 'pass'" % verdict)


def test_the_bd_of_another_host(conformance_reports):
    report = conformance_reports()
    other = W.make_host(ccd=False).describe()
    assert "BD 'fakehost'" in only(check(report, bd=other), "is not the hash %s" % other.hash())
    report["bd_hash"] = "0" * 64
    only(check(report), "bd_hash %s" % ("0" * 64))


def test_another_adapter_version(conformance_reports):
    report = conformance_reports()
    only(check(report, adapter_version="fakehost-test"), "adapter_version %r is not the freeze's 'fakehost-test'"
         % report["adapter_version"])


@pytest.mark.parametrize("name", REPORT_CONFIGS)
def test_a_missing_config_hash(conformance_reports, name):
    report = conformance_reports()
    del report["config_hashes"][name]
    only(check(report), "config_hashes has no %r" % name)


@pytest.mark.parametrize("name", REPORT_CONFIGS)
def test_a_different_config_hash(conformance_reports, name):
    report = conformance_reports()
    report["config_hashes"][name] = "0" * 64
    only(check(report), "config %r hash %s is not the frozen %s.yaml's" % (name, "0" * 64, name))


def test_frozen_config_other_than_the_one_the_suite_ran_on(tmp_path, conformance_reports):
    cfg = W.copy_fixture_configs(tmp_path / "cfg")
    W.edit_yaml(cfg / "semantics.yaml", lambda d: d["contact"].update(gap_tolerance_s=0.05))
    only(check(conformance_reports(), hashes=config_hashes(cfg)), "config 'semantics' hash")


def test_a_runtime_setting_from_the_probe_runtime_fallback(conformance_reports):
    report = conformance_reports()
    report["runtime_config"]["restitution_e"]["source"] = PROBE_RUNTIME
    problem = only(check(report), "runtime_config.restitution_e")
    assert PROBE_RUNTIME in problem and "fallback" in problem


def test_a_runtime_config_without_sources(conformance_reports):
    report = conformance_reports()
    del report["runtime_config"]["t_q_s"]
    only(check(report), "runtime_config does not give the source of ['t_q_s']")
    report = conformance_reports()
    del report["runtime_config"]["t_q_s"]["source"]
    only(check(report), "runtime_config.t_q_s has no source")


def test_a_calibration_setting_from_the_probe_calibration_fallback(conformance_reports):
    report = conformance_reports()
    c10_settings(report)["v_threshold_mps"]["source"] = PROBE_CALIBRATION
    problem = only(check(report), "check C10: measured.t_stop_settings.v_threshold_mps")
    assert PROBE_CALIBRATION in problem and "fallback" in problem


def test_every_problem_is_listed_at_once(conformance_reports):
    clean = conformance_reports()
    report = conformance_reports()
    report["verdict"] = "fail"
    report["bd_hash"] = "0" * 64
    report["adapter_version"] = "other"
    del report["config_hashes"]["intensity"]
    report["config_hashes"]["semantics"] = "1" * 64
    report["runtime_config"]["restitution_e"]["source"] = PROBE_RUNTIME
    c10_settings(report)["v_threshold_mps"]["source"] = PROBE_CALIBRATION
    problems = check(report, adapter_of=clean)
    fragments = ["verdict is 'fail'", "bd_hash %s" % ("0" * 64), "adapter_version 'other'",
                 "config_hashes has no 'intensity'", "config 'semantics' hash", "runtime_config.restitution_e",
                 "check C10: measured.t_stop_settings.v_threshold_mps"]
    assert len(problems) == len(fragments), problems
    for fragment in fragments:
        assert sum(fragment in p for p in problems) == 1, (fragment, problems)


def test_a_report_on_the_repository_configuration_is_refused_for_its_fallbacks():
    """``runtime.yaml`` and the T_stop keys of ``intensity.yaml`` are still PLACEHOLDER in the repository, so the
    suite takes them from ``conformance.yaml``; the report passes, but cannot back a freeze."""
    report = W.conformance_report(W.REPO_CONFIG_DIR)
    assert report["verdict"] == "pass"
    runtime_fallbacks = sorted(name for name, entry in report["runtime_config"].items()
                               if entry["source"].startswith(FALLBACK_SOURCE_PREFIX))
    assert runtime_fallbacks
    own = {name: report["config_hashes"][name] for name in REPORT_CONFIGS}  # the very files the suite read
    problems = check(report, hashes=own)
    assert all("fallback" in p for p in problems), problems
    for name in runtime_fallbacks:
        assert sum(p.startswith("runtime_config.%s " % name) for p in problems) == 1, (name, problems)
    calibration = [p for p in problems if p.startswith("check C10: measured.t_stop_settings.")]
    assert calibration and all("conformance.yaml:probe_calibration." in p for p in calibration)
    assert len(problems) == len(runtime_fallbacks) + len(calibration)
    against_fixtures = check(report)                        # freezing on the fixture configs: hashes differ too
    assert len(against_fixtures) == len(problems) + len(REPORT_CONFIGS)
