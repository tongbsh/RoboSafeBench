"""Freeze / verify (T06 item 5; acceptance 3, 16; T06-F: the report and calibration binding, D13 a, b)."""
import json

import pytest
import yaml

import pipeline_world as W
from pipeline_world import calibration_file, conformance_dir, conformance_reports  # noqa: F401 (fixtures)
from hazardlib.config import ConfigError
from hazardlib.conformance import read_report
from hazardlib.gates import make_certificate
from hazardlib.hashing import sha256_hex
from hazardlib.manifest import (CALIBRATION_FILE, CONFORMANCE_REPORT_FILE, MANIFEST_VERSION, USED_CONFIGS,
                                ManifestError, config_hashes, freeze, verify_manifest, write_frozen)
from hazardlib.probe import build_task_context
from hazardlib.spec import BenchmarkDescriptor, HazardInstance

G1_OF = {"wait_solvable": True, "uncertified": False, "baseline_unusable": None}


@pytest.fixture
def bind(tmp_path, conformance_reports, calibration_file):
    """``bind(**target)``: the freeze inputs of D13 a, b for the BD of ``W.make_host(**target)``: a real
    conformance report (written under ``tmp_path``), its adapter version and the calibration file."""
    def make(**target):
        report = conformance_reports(**target)
        name = "conformance_%s.json" % "_".join("%s-%s" % kv for kv in sorted(target.items()))
        return {"conformance_report": W.write_json(tmp_path / name, report), "calibration": calibration_file,
                "adapter_version": report["adapter_version"]}
    return make


def certified(instance, label):
    cert = make_certificate(label, G1_OF[label], None, "test certificate")
    return HazardInstance.from_dict(dict(instance.to_dict(), certificate=cert.to_dict()))


def specs():
    return [W.fall_spec(spec_id="fall"), W.slide_spec(spec_id="slide")]


def gated(host=None, tcs=None, instance_seed=0, labels=("wait_solvable", "uncertified")):
    host = host or W.make_host()
    tcs = tcs or {"pick": W.task_context("pick"), "low": W.task_context("low")}
    fall, slide = specs()
    return [certified(W.build(fall, "pick", seed=instance_seed, host=host, tc=tcs["pick"]), labels[0]),
            certified(W.build(slide, "low", seed=instance_seed, host=host, tc=tcs["low"]), labels[1])]


def do_freeze(binding, config_dir=W.FIXTURE_CONFIG_DIR, seed=0, host=None, tcs=None, instance_seed=0,
              instances=None):
    host = host or W.make_host()
    tcs = tcs or {"pick": W.task_context("pick"), "low": W.task_context("low")}
    instances = instances or gated(host, tcs, instance_seed)
    return freeze(instances, list(tcs.values()), specs(), host.describe(), config_dir, binding["adapter_version"],
                  seed, binding["conformance_report"], binding["calibration"])


def test_freeze_refuses_placeholder_configs(tmp_path, bind):
    with pytest.raises(ConfigError) as info:
        do_freeze(bind(), config_dir=W.REPO_CONFIG_DIR)
    message = str(info.value)
    assert "cannot freeze" in message and all("%s.yaml" % name in message for name in USED_CONFIGS)
    one = W.copy_fixture_configs(tmp_path / "cfg")
    W.edit_yaml(one / "gates.yaml", lambda d: d["g1"].update(resume_settle_s="PLACEHOLDER"))
    with pytest.raises(ConfigError, match=r"(?s)1 config file\(s\) unresolved.*gates.yaml.*g1.resume_settle_s"):
        do_freeze(bind(), config_dir=one)


def test_freeze_writes_a_verifiable_manifest(tmp_path, bind):
    frozen = do_freeze(bind())
    out = write_frozen(tmp_path / "frozen", frozen)
    assert sorted(p.name for p in out.iterdir()) == [CALIBRATION_FILE, CONFORMANCE_REPORT_FILE, "freeze_report.yaml",
                                                     "instances.jsonl", "manifest.sha256", "manifest.yaml"]
    digest = verify_manifest(out, config_dir=W.FIXTURE_CONFIG_DIR, bd=W.make_host().describe())
    assert digest == frozen.manifest.hash() == (out / "manifest.sha256").read_text().strip()
    manifest = frozen.manifest
    assert sorted(manifest.config_hashes) == sorted(USED_CONFIGS)
    assert manifest.bd_hash == W.make_host().describe().hash()
    assert [e.instance_id for e in manifest.entries] == ["fall.pick.L0.s0", "slide.low.L0.s0"]
    assert frozen.report["labels"] == {"uncertified": 1, "wait_solvable": 1}


def test_manifest_binds_the_conformance_report_and_the_calibration(tmp_path, bind):
    binding = bind()
    frozen = do_freeze(binding)
    manifest = frozen.manifest
    assert manifest.manifest_version == MANIFEST_VERSION == "2"
    report = read_report(binding["conformance_report"])
    calibration = yaml.safe_load(binding["calibration"].read_text())
    assert manifest.conformance_report_hash == sha256_hex(report)        # parsed content, like config_hashes
    assert manifest.calibration_hash == sha256_hex(calibration)
    assert report["bd_hash"] == manifest.bd_hash and report["adapter_version"] == manifest.adapter_version
    assert frozen.report["conformance_report_hash"] == manifest.conformance_report_hash
    assert frozen.report["calibration_hash"] == manifest.calibration_hash
    out = write_frozen(tmp_path / "frozen", frozen)
    assert read_report(out / CONFORMANCE_REPORT_FILE) == report
    assert yaml.safe_load((out / CALIBRATION_FILE).read_text()) == calibration
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(report, sort_keys=False))         # formatting does not count, values do
    assert do_freeze(dict(binding, conformance_report=reformatted)).manifest.hash() == manifest.hash()


def test_verify_detects_a_modified_report_and_calibration(tmp_path, bind):
    out = write_frozen(tmp_path / "frozen", do_freeze(bind()))
    report = json.loads((out / CONFORMANCE_REPORT_FILE).read_text())
    report["checks"][0]["measured"]["tampered"] = 1.0                    # still a well-formed report
    W.write_json(out / CONFORMANCE_REPORT_FILE, report)
    W.edit_yaml(out / CALIBRATION_FILE, lambda d: d.update(samples_s=[0.1 for _ in d["samples_s"]]))
    with pytest.raises(ManifestError) as info:
        verify_manifest(out)
    message = str(info.value)
    assert "2 problem(s)" in message
    assert "%s was modified" % CONFORMANCE_REPORT_FILE in message and "%s was modified" % CALIBRATION_FILE in message


def test_verify_detects_an_unreadable_report(tmp_path, bind):
    out = write_frozen(tmp_path / "frozen", do_freeze(bind()))
    report = json.loads((out / CONFORMANCE_REPORT_FILE).read_text())
    report["verdict"] = "fail"                                           # no longer matches its checks
    W.write_json(out / CONFORMANCE_REPORT_FILE, report)
    with pytest.raises(ManifestError, match="%s cannot be read: verdict 'fail'" % CONFORMANCE_REPORT_FILE):
        verify_manifest(out)


def test_verify_detects_a_missing_report_and_calibration(tmp_path, bind):
    out = write_frozen(tmp_path / "frozen", do_freeze(bind()))
    (out / CONFORMANCE_REPORT_FILE).unlink()
    (out / CALIBRATION_FILE).unlink()
    with pytest.raises(ManifestError) as info:
        verify_manifest(out)
    message = str(info.value)
    assert "2 problem(s)" in message
    assert "%s is missing" % CONFORMANCE_REPORT_FILE in message and "%s is missing" % CALIBRATION_FILE in message


def test_freeze_refuses_a_malformed_calibration(tmp_path, bind):
    bad = tmp_path / "calibration.yaml"
    data = yaml.safe_load(bind()["calibration"].read_text())
    data["t_stop_is_lower_bound"] = not data["t_stop_is_lower_bound"]
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="contradicts n_censored"):
        do_freeze(dict(bind(), calibration=bad))
    bad.write_text(yaml.safe_dump({"t_stop_median_s": 0.2}))
    with pytest.raises(ValueError, match="not a calibration file"):
        do_freeze(dict(bind(), calibration=bad))


def test_verify_lists_every_tampering(tmp_path, bind):
    out = write_frozen(tmp_path / "frozen", do_freeze(bind()))
    lines = (out / "instances.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["onset_time_s"] += 0.01
    extra = dict(first, instance_id="extra.pick.L0.s9")
    (out / "instances.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(extra) + "\n")  # modified, missing, extra
    (out / "manifest.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ManifestError) as info:
        verify_manifest(out)
    message = str(info.value)
    assert "4 problem(s)" in message
    assert "was modified" in message and "listed but missing" in message and "manifest.sha256" in message
    assert "'extra.pick.L0.s9' is not listed in the manifest" in message


def test_verify_detects_config_and_bd_drift(tmp_path, bind):
    out = write_frozen(tmp_path / "frozen", do_freeze(bind()))
    cfg = W.copy_fixture_configs(tmp_path / "cfg")
    W.edit_yaml(cfg / "semantics.yaml", lambda d: d["exposure"].update(low_exposure_radius_m=0.2))
    with pytest.raises(ManifestError, match="config 'semantics' changed"):
        verify_manifest(out, config_dir=cfg)
    with pytest.raises(ManifestError, match="bd_hash"):
        verify_manifest(out, bd=W.make_host(ccd=False).describe())


@pytest.mark.parametrize("name", USED_CONFIGS)
def test_config_hashes_follow_each_file(tmp_path, name):
    cfg = W.copy_fixture_configs(tmp_path / "cfg")
    before = config_hashes(cfg)
    path = cfg / ("%s.yaml" % name)
    path.write_text(path.read_text() + "# a comment does not change the parsed content\n")
    assert config_hashes(cfg) == before

    def bump(data):
        node = data
        while True:
            key = sorted(k for k, v in node.items() if isinstance(v, (dict, int, float)) and not isinstance(v, bool))[0]
            if isinstance(node[key], dict):
                node = node[key]
            else:
                node[key] = node[key] + 1
                return

    W.edit_yaml(path, bump)
    after = config_hashes(cfg)
    assert [n for n in USED_CONFIGS if after[n] != before[n]] == [name]


def test_manifest_hash_is_deterministic_and_follows_every_input(tmp_path, bind):
    binding = bind()
    base = do_freeze(binding).manifest.hash()
    assert do_freeze(binding).manifest.hash() == base

    assert do_freeze(binding, seed=1).manifest.hash() != base
    assert do_freeze(binding, instance_seed=1).manifest.hash() != base

    host = W.make_host()
    fewer = W.record_traces(host, "pick", policies=W.probe_policies(host)[:2])
    tc_pick = build_task_context(fewer, W.probe_settings(), static_entities=W.STATIC_ENTITIES,
                                 declared_workspace=W.WORKSPACE)
    assert tc_pick.provenance["probe_set_hash"] != W.task_context("pick").provenance["probe_set_hash"]
    assert do_freeze(binding, tcs={"pick": tc_pick, "low": W.task_context("low")}).manifest.hash() != base

    assert do_freeze(bind(ccd=False), host=W.make_host(ccd=False)).manifest.hash() != base

    cfg = W.copy_fixture_configs(tmp_path / "cfg")
    W.edit_yaml(cfg / "probe.yaml", lambda d: d.update(moving_speed_min_mps=0.02))
    changed = do_freeze(binding, config_dir=cfg).manifest
    assert changed.hash() != base and \
        changed.config_hashes["probe"] != do_freeze(binding).manifest.config_hashes["probe"]

    calibration = tmp_path / "calibration_low.yaml"
    W.write_calibration_file(calibration, task="low")
    assert do_freeze(dict(binding, calibration=calibration)).manifest.hash() != base
    report = read_report(binding["conformance_report"])
    report["checks"][0]["measured"]["extra"] = 1.0
    other = W.write_json(tmp_path / "other_report.json", report)
    assert do_freeze(dict(binding, conformance_report=other)).manifest.hash() != base


def test_baseline_unusable_is_counted_not_frozen(bind):
    host = W.make_host()
    instances = gated(host, labels=("wait_solvable", "baseline_unusable"))
    frozen = do_freeze(bind(), host=host, instances=instances)
    assert [e.spec_id for e in frozen.manifest.entries] == ["fall"]
    assert frozen.report["excluded"] == [{"instance_id": "slide.low.L0.s0", "label": "baseline_unusable"}]
    assert frozen.report["labels"] == {"baseline_unusable": 1, "wait_solvable": 1}


def test_freeze_refuses_incomplete_inputs(bind):
    host = W.make_host()
    binding = bind()
    bound = (binding["adapter_version"], 0, binding["conformance_report"], binding["calibration"])
    raw = W.build(W.fall_spec(spec_id="fall"), "pick", host=host)
    with pytest.raises(ValueError, match="no certificate"):
        do_freeze(binding, host=host, instances=[raw])
    with pytest.raises(ValueError, match="not among the given ones"):
        freeze(gated(host), [W.task_context("pick")], specs(), host.describe(), W.FIXTURE_CONFIG_DIR, *bound)
    with pytest.raises(ValueError, match="not in the library"):
        freeze(gated(host), [W.task_context("pick"), W.task_context("low")], specs()[:1], host.describe(),
               W.FIXTURE_CONFIG_DIR, *bound)
    data = host.describe().to_dict()
    data["determinism"]["same_seed_repro"] = "unknown"
    with pytest.raises(Exception, match="unknown"):
        freeze(gated(host), [W.task_context("pick"), W.task_context("low")], specs(),
               BenchmarkDescriptor.from_dict(data), W.FIXTURE_CONFIG_DIR, *bound)
