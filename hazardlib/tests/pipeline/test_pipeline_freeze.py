"""Freeze / verify (T06 item 5; acceptance 3, 16)."""
import json

import pytest

import pipeline_world as W
from hazardlib.config import ConfigError
from hazardlib.gates import make_certificate
from hazardlib.manifest import USED_CONFIGS, ManifestError, config_hashes, freeze, verify_manifest, write_frozen
from hazardlib.probe import build_task_context
from hazardlib.spec import BenchmarkDescriptor, HazardInstance

G1_OF = {"wait_solvable": True, "uncertified": False, "baseline_unusable": None}


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


def do_freeze(config_dir=W.FIXTURE_CONFIG_DIR, seed=0, host=None, tcs=None, instance_seed=0, instances=None):
    host = host or W.make_host()
    tcs = tcs or {"pick": W.task_context("pick"), "low": W.task_context("low")}
    instances = instances or gated(host, tcs, instance_seed)
    return freeze(instances, list(tcs.values()), specs(), host.describe(), config_dir, "fakehost-test", seed)


def test_freeze_refuses_placeholder_configs(tmp_path):
    with pytest.raises(ConfigError) as info:
        do_freeze(config_dir=W.REPO_CONFIG_DIR)
    message = str(info.value)
    assert "cannot freeze" in message and all("%s.yaml" % name in message for name in USED_CONFIGS)
    one = W.copy_fixture_configs(tmp_path / "cfg")
    W.edit_yaml(one / "gates.yaml", lambda d: d["g1"].update(resume_settle_s="PLACEHOLDER"))
    with pytest.raises(ConfigError, match=r"(?s)1 config file\(s\) unresolved.*gates.yaml.*g1.resume_settle_s"):
        do_freeze(config_dir=one)


def test_freeze_writes_a_verifiable_manifest(tmp_path):
    frozen = do_freeze()
    out = write_frozen(tmp_path / "frozen", frozen)
    assert sorted(p.name for p in out.iterdir()) == ["freeze_report.yaml", "instances.jsonl", "manifest.sha256",
                                                     "manifest.yaml"]
    digest = verify_manifest(out, config_dir=W.FIXTURE_CONFIG_DIR, bd=W.make_host().describe())
    assert digest == frozen.manifest.hash() == (out / "manifest.sha256").read_text().strip()
    manifest = frozen.manifest
    assert sorted(manifest.config_hashes) == sorted(USED_CONFIGS)
    assert manifest.bd_hash == W.make_host().describe().hash()
    assert [e.instance_id for e in manifest.entries] == ["fall.pick.L0.s0", "slide.low.L0.s0"]
    assert frozen.report["labels"] == {"uncertified": 1, "wait_solvable": 1}


def test_verify_lists_every_tampering(tmp_path):
    out = write_frozen(tmp_path / "frozen", do_freeze())
    lines = (out / "instances.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["onset_time_s"] += 0.01
    (out / "instances.jsonl").write_text(json.dumps(first) + "\n")              # one modified, one missing
    (out / "manifest.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ManifestError) as info:
        verify_manifest(out)
    message = str(info.value)
    assert "3 problem(s)" in message
    assert "was modified" in message and "listed but missing" in message and "manifest.sha256" in message


def test_verify_detects_config_and_bd_drift(tmp_path):
    out = write_frozen(tmp_path / "frozen", do_freeze())
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


def test_manifest_hash_is_deterministic_and_follows_every_input(tmp_path):
    base = do_freeze().manifest.hash()
    assert do_freeze().manifest.hash() == base

    assert do_freeze(seed=1).manifest.hash() != base
    assert do_freeze(instance_seed=1).manifest.hash() != base

    host = W.make_host()
    fewer = W.record_traces(host, "pick", policies=W.probe_policies(host)[:2])
    tc_pick = build_task_context(fewer, W.probe_settings(), static_entities=W.STATIC_ENTITIES,
                                 declared_workspace=W.WORKSPACE)
    assert tc_pick.provenance["probe_set_hash"] != W.task_context("pick").provenance["probe_set_hash"]
    assert do_freeze(tcs={"pick": tc_pick, "low": W.task_context("low")}).manifest.hash() != base

    assert do_freeze(host=W.make_host(ccd=False)).manifest.hash() != base

    cfg = W.copy_fixture_configs(tmp_path / "cfg")
    W.edit_yaml(cfg / "probe.yaml", lambda d: d.update(moving_speed_min_mps=0.02))
    changed = do_freeze(config_dir=cfg).manifest
    assert changed.hash() != base and changed.config_hashes["probe"] != do_freeze().manifest.config_hashes["probe"]


def test_baseline_unusable_is_counted_not_frozen():
    host = W.make_host()
    instances = gated(host, labels=("wait_solvable", "baseline_unusable"))
    frozen = do_freeze(host=host, instances=instances)
    assert [e.spec_id for e in frozen.manifest.entries] == ["fall"]
    assert frozen.report["excluded"] == [{"instance_id": "slide.low.L0.s0", "label": "baseline_unusable"}]
    assert frozen.report["labels"] == {"baseline_unusable": 1, "wait_solvable": 1}


def test_freeze_refuses_incomplete_inputs():
    host = W.make_host()
    raw = W.build(W.fall_spec(spec_id="fall"), "pick", host=host)
    with pytest.raises(ValueError, match="no certificate"):
        do_freeze(host=host, instances=[raw])
    with pytest.raises(ValueError, match="not among the given ones"):
        freeze(gated(host), [W.task_context("pick")], specs(), host.describe(), W.FIXTURE_CONFIG_DIR, "v", 0)
    with pytest.raises(ValueError, match="not in the library"):
        freeze(gated(host), [W.task_context("pick"), W.task_context("low")], specs()[:1], host.describe(),
               W.FIXTURE_CONFIG_DIR, "v", 0)
    data = host.describe().to_dict()
    data["determinism"]["same_seed_repro"] = "unknown"
    with pytest.raises(Exception, match="unknown"):
        freeze(gated(host), [W.task_context("pick"), W.task_context("low")], specs(),
               BenchmarkDescriptor.from_dict(data), W.FIXTURE_CONFIG_DIR, "v", 0)
