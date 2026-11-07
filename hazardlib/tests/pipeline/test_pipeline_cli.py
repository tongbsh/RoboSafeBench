"""End to end on FakeHost through ``hazardctl`` (T06 item 6; acceptance 2, 3).

record x3 -> probe -> instantiate -> gate -> calibrate -> freeze -> verify, each a separate process exchanging
YAML / JSONL files, on the braking FakeHost (T_stop = 0.2 s). Run twice: the Manifest hash must be identical.
The freeze is bound to a real conformance report of the same host class and timebase (``W.conformance_target``,
same BD hash) and to the calibration of the chain (D13 a, b); a report of the same host generated on the
repository configuration is refused with every problem listed, and verify reports a modified calibration and a
missing report together.
"""
import json
import os
import shutil
import subprocess
import sys

import yaml

import pipeline_world as W
from pipeline_world import conformance_dir, conformance_reports  # noqa: F401 (fixtures)

HOST = ["--host", "pipeline_world:make_host", "--host-arg", "decel_mps2=%r" % W.DECEL_MPS2]
SPECS = [W.fall_spec(tau=0.3, frac=0.75, spec_id="fall_wait"),
         W.fall_spec(tau=0.1, frac=0.5, spec_id="fall_short"),
         W.fall_spec(tau=0.5, phase="carry", frac=0.5, spec_id="fall_high"),
         W.slide_spec(spec_id="slide_on_pick"),
         W.reach_spec(tau=0.3, speed=0.6, frac=0.75, spec_id="reach_wait")]


def run_cli(cwd, config_dir, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(W.HERE.parents[1] / "src"), str(W.HERE)])
    return subprocess.run([sys.executable, "-m", "hazardlib.cli", "--config-dir", str(config_dir)]
                          + [str(a) for a in args], cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)


def hazardctl(cwd, *args):
    proc = run_cli(cwd, W.FIXTURE_CONFIG_DIR, *args)
    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip().splitlines()[-1]
    print(line)
    return json.loads(line)


def chain(work, report):
    work.mkdir()
    W.write_json(work / "conformance.json", report)
    (work / "specs.yaml").write_text(yaml.safe_dump({"specs": [s.to_dict() for s in SPECS]}, sort_keys=False))
    (work / "entities.yaml").write_text(yaml.safe_dump(W.STATIC_ENTITIES))
    (work / "workspace.yaml").write_text(yaml.safe_dump(W.WORKSPACE))
    follower = ["--policy", "pipeline_world:make_follower"]
    records = [(follower + ["--policy-arg", "name=follower"], 0),
               (["--policy", "pipeline_world:make_demo"], 1),
               (follower + ["--policy-arg", "name=follower_fast", "--policy-arg", "time_scale=1.25"], 2)]
    for k, (policy, seed) in enumerate(records):
        hazardctl(work, "record", *HOST, *policy, "--task", "pick", "--layout", "L0", "--seed", seed,
                  "--horizon-s", W.PROBE_HORIZON_S, "--env", 0, "--out", "trace_%d.jsonl" % k, "--bd-out", "bd.yaml")
    probe = hazardctl(work, "probe", "--traces", "trace_0.jsonl", "trace_1.jsonl", "trace_2.jsonl",
                      "--out", "tc_pick.yaml", "--entities", "entities.yaml", "--workspace", "workspace.yaml")
    inst = hazardctl(work, "instantiate", "--specs", "specs.yaml", "--task-context", "tc_pick.yaml", "--bd", "bd.yaml",
                     "--seed", 0, "--count", 1, "--out", "instances.jsonl")
    gate = hazardctl(work, "gate", "--instances", "instances.jsonl", "--task-context", "tc_pick.yaml", *HOST,
                     *follower, "--seed", 0, "--env", 0, "--out", "gated.jsonl", "--report", "gate_report.yaml")
    cal = hazardctl(work, "calibrate", "--task-context", "tc_pick.yaml", *HOST, *follower, "--seed", 0, "--env", 0,
                    "--out", "calibration.yaml")
    frz = hazardctl(work, "freeze", "--instances", "gated.jsonl", "--task-context", "tc_pick.yaml", "--specs",
                    "specs.yaml", "--bd", "bd.yaml", "--seed", 0, "--adapter-version", report["adapter_version"],
                    "--conformance", "conformance.json", "--calibration", "calibration.yaml", "--out-dir", "frozen")
    ver = hazardctl(work, "verify", "--manifest-dir", "frozen", "--bd", "bd.yaml")
    return probe, inst, gate, cal, frz, ver


def test_end_to_end_cli_chain_is_deterministic(tmp_path, conformance_reports):
    conformance = conformance_reports(decel_mps2=W.DECEL_MPS2)
    probe, inst, gate, cal, frz, ver = chain(tmp_path / "run1", conformance)
    assert probe["phases"] == ["approach", "carry"] and probe["runs_used"] == 3
    assert inst["instances"] == 4 and inst["rejected_by_reason"] == {"no_support_surface": 1}
    assert gate["rejections_by_reason"] == {"release_height_out_of_workspace": 1}
    assert gate["labels"] == {"uncertified": 1, "wait_solvable": 2} and gate["n_main_score"] == 2
    assert cal["t_stop_is_lower_bound"] is False and abs(cal["t_stop_median_s"] - 0.2) < 1e-9
    assert frz["entries"] == 3 and ver["ok"] is True and ver["manifest_hash"] == frz["manifest_hash"]

    report = yaml.safe_load((tmp_path / "run1" / "gate_report.yaml").read_text())
    assert report["upstream_rejections_by_reason"] == {"no_support_surface": 1}
    labels = {}
    for line in (tmp_path / "run1" / "gated.jsonl").read_text().splitlines():
        record = json.loads(line)
        if "instance" in record:
            labels[record["instance"]["spec_id"]] = record["instance"]["certificate"]["label"]
    assert labels == {"fall_wait": "wait_solvable", "fall_short": "uncertified", "reach_wait": "wait_solvable"}

    again = chain(tmp_path / "run2", conformance)
    assert again[4]["manifest_hash"] == frz["manifest_hash"]
    assert (tmp_path / "run1" / "frozen" / "instances.jsonl").read_text() == \
        (tmp_path / "run2" / "frozen" / "instances.jsonl").read_text()

    work = tmp_path / "run1"
    refused = run_cli(work, W.REPO_CONFIG_DIR, "freeze", "--instances", "gated.jsonl", "--task-context",
                      "tc_pick.yaml", "--specs", "specs.yaml", "--bd", "bd.yaml", "--seed", 0, "--adapter-version",
                      conformance["adapter_version"], "--conformance", "conformance.json", "--calibration",
                      "calibration.yaml", "--out-dir", "frozen_repo")
    assert refused.returncode == 2 and "cannot freeze: 5 config file(s) unresolved" in refused.stderr
    assert not (work / "frozen_repo").exists()

    # same host and BD, but generated on the repository configuration: PLACEHOLDER runtime.yaml -> fallbacks
    W.write_json(work / "repo_conformance.json", W.conformance_report(W.REPO_CONFIG_DIR, decel_mps2=W.DECEL_MPS2))
    bad = run_cli(work, W.FIXTURE_CONFIG_DIR, "freeze", "--instances", "gated.jsonl", "--task-context",
                  "tc_pick.yaml", "--specs", "specs.yaml", "--bd", "bd.yaml", "--seed", 0, "--adapter-version",
                  conformance["adapter_version"], "--conformance", "repo_conformance.json", "--calibration",
                  "calibration.yaml", "--out-dir", "frozen_bad")
    print(bad.stderr.splitlines()[0])
    header, problems = bad.stderr.splitlines()[0], bad.stderr.strip().splitlines()[1:]
    assert bad.returncode == 2 and "cannot freeze with the conformance report repo_conformance.json" in header
    assert header.endswith("%d problem(s):" % len(problems)) and "bd_hash" not in bad.stderr
    for fragment in ("config 'runtime' hash", "runtime_config.restitution_e = ",
                     "v_threshold_mps comes from conformance.yaml:probe_calibration."):
        assert sum(fragment in p for p in problems) == 1, bad.stderr
    assert not (work / "frozen_bad").exists()

    shutil.copytree(str(work / "frozen"), str(work / "frozen_copy"))
    calibration = yaml.safe_load((work / "frozen_copy" / "calibration.yaml").read_text())
    calibration["t_stop"]["median_s"] = 0.1
    (work / "frozen_copy" / "calibration.yaml").write_text(yaml.safe_dump(calibration, sort_keys=False))
    (work / "frozen_copy" / "conformance_report.json").unlink()
    broken = run_cli(work, W.FIXTURE_CONFIG_DIR, "verify", "--manifest-dir", "frozen_copy")
    print(broken.stderr.strip().splitlines()[0])
    assert broken.returncode == 2 and "2 problem(s)" in broken.stderr
    assert "conformance_report.json is missing" in broken.stderr and "calibration.yaml was modified" in broken.stderr

    lines = (work / "frozen" / "instances.jsonl").read_text().splitlines()
    record = json.loads(lines[0])
    record["onset_time_s"] += 0.01
    (work / "frozen" / "instances.jsonl").write_text("\n".join([json.dumps(record)] + lines[1:]) + "\n")
    tampered = run_cli(work, W.FIXTURE_CONFIG_DIR, "verify", "--manifest-dir", "frozen")
    assert tampered.returncode == 2 and "was modified" in tampered.stderr
