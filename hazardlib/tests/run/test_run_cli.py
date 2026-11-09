"""End to end on FakeHost through ``hazardctl`` (R1 item 6; acceptance 2).

Formal: record x3 -> probe -> instantiate -> gate -> calibrate -> freeze -> verify -> run (three policies under test,
two seeds; one with the ``.npy`` frame sink) -> report, each a separate process. Pilot: the gated instances with a
pilot configuration directory and ``--pilot`` -> run -> report. Then the refusals of the command line: mixing pilot
and formal runs in one report, a run without ``--policy-name``, non-frozen inputs without ``--pilot``, a frozen
directory with ``--pilot``, the repository's PLACEHOLDER ``run.yaml``, and a policy that is a probe source.
"""
import json
import os
import subprocess
import sys

import numpy as np
import yaml

import run_world as RW

W = RW.W
HOST = ["--host", "run_world:make_host"]


def run_cli(cwd, config_dir, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(W.HERE.parents[1] / "src"), str(RW.HERE), str(W.HERE)])
    return subprocess.run([sys.executable, "-m", "hazardlib.cli", "--config-dir", str(config_dir)]
                          + [str(a) for a in args], cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)


def hazardctl(cwd, config_dir, *args):
    proc = run_cli(cwd, config_dir, *args)
    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip().splitlines()[-1]
    print(line)
    return json.loads(line)


def prepare(work):
    work.mkdir()
    cfg = RW.config_dir(work / "configs", conformance=True)
    W.write_json(work / "conformance.json", RW.conformance_report(cfg))
    (work / "specs.yaml").write_text(yaml.safe_dump({"specs": [s.to_dict() for s in RW.specs()]}, sort_keys=False))
    (work / "entities.yaml").write_text(yaml.safe_dump(RW.ENTITIES))
    (work / "workspace.yaml").write_text(yaml.safe_dump(RW.WORKSPACE))
    return cfg


def pipeline(work, cfg):
    follower = ["--policy", "pipeline_world:make_follower"]
    records = [(follower + ["--policy-arg", "name=follower"], 0), (["--policy", "pipeline_world:make_demo"], 1),
               (follower + ["--policy-arg", "name=follower_fast", "--policy-arg", "time_scale=1.25"], 2)]
    for k, (policy, seed) in enumerate(records):
        hazardctl(work, cfg, "record", *HOST, *policy, "--task", RW.TASK, "--layout", RW.LAYOUT, "--seed", seed,
                  "--horizon-s", W.PROBE_HORIZON_S, "--env", 0, "--out", "trace_%d.jsonl" % k, "--bd-out", "bd.yaml")
    hazardctl(work, cfg, "probe", "--traces", "trace_0.jsonl", "trace_1.jsonl", "trace_2.jsonl", "--out", "tc.yaml",
              "--entities", "entities.yaml", "--workspace", "workspace.yaml")
    hazardctl(work, cfg, "instantiate", "--specs", "specs.yaml", "--task-context", "tc.yaml", "--bd", "bd.yaml",
              "--seed", 0, "--count", 1, "--out", "instances.jsonl")
    gate = hazardctl(work, cfg, "gate", "--instances", "instances.jsonl", "--task-context", "tc.yaml", *HOST,
                     *follower, "--seed", 0, "--env", 0, "--out", "gated.jsonl", "--report", "gate_report.yaml")
    hazardctl(work, cfg, "calibrate", "--task-context", "tc.yaml", *HOST, *follower, "--seed", 0, "--env", 0,
              "--out", "calibration.yaml")
    adapter = json.loads((work / "conformance.json").read_text())["adapter_version"]
    frz = hazardctl(work, cfg, "freeze", "--instances", "gated.jsonl", "--task-context", "tc.yaml", "--specs",
                    "specs.yaml", "--bd", "bd.yaml", "--seed", 0, "--adapter-version", adapter, "--conformance",
                    "conformance.json", "--calibration", "calibration.yaml", "--out-dir", "frozen")
    hazardctl(work, cfg, "verify", "--manifest-dir", "frozen", "--bd", "bd.yaml")
    return gate, frz


def run_args(policy, out, *extra):
    return ["run", *HOST, "--policy", "run_world:make_%s" % policy, "--policy-name", policy, "--seeds", 0, 1,
            "--env", 0, "--out-dir", out, "--task-context", "tc.yaml", "--bd", "bd.yaml", *extra]


def test_formal_and_pilot_runs_end_to_end(tmp_path):
    work = tmp_path / "chain"
    cfg = prepare(work)
    gate, frz = pipeline(work, cfg)
    assert gate["labels"] == {"uncertified": 1, "wait_solvable": 2}

    formal = ["--manifest-dir", "frozen"]
    runs = {}
    for policy in ("nominal", "statue"):
        runs[policy] = hazardctl(work, cfg, *run_args(policy, "run_" + policy, *formal))
    runs["cautious"] = hazardctl(work, cfg, *run_args("cautious", "run_cautious", *formal, "--frame-keys", "rgb",
                                                      "--frame-sink", "hazardlib.run.frames:make_npy_sink"))
    for result in runs.values():
        assert result["pilot"] is False and result["manifest_hash"] == frz["manifest_hash"]
        assert result["planned"] == result["ran"] == 10 and result["ran_by_kind"] == {"control": 4, "hazard": 6}
    again = hazardctl(work, cfg, *run_args("nominal", "run_nominal", *formal))
    assert again["ran"] == 0 and again["skipped"] == 10

    lines = (work / "run_cautious" / "episodes.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    frames = np.load(str(work / "run_cautious" / first["frames"]["rgb"]))
    assert frames.shape == (first["n_control_steps"], 32, 32, 3)
    assert first["frames"]["rgb"].startswith("frames/") and first["kind"] == "control"

    rep = hazardctl(work, cfg, "report", "--runs", "run_nominal", "run_statue", "run_cautious", "--out-dir", "report")
    assert rep["pilot"] is False and rep["counts"]["hazard_episodes"] == 18 and rep["counts"]["control_episodes"] == 12
    summary = yaml.safe_load((work / "report" / "summary.yaml").read_text())
    rows = {(r["policy"], r["cert"], r["tau_bin_s"]): r for r in summary["main_tau"]}
    assert rows[("nominal", "certified", "[0.2,0.4)")]["official_contact"]["k"] == 2
    assert rows[("cautious", "certified", "[0.2,0.4)")]["official_contact"]["k"] == 0
    assert rows[("cautious", "uncertified", "<0.2")]["official_contact"]["k"] == 2
    assert rows[("cautious", "certified", ">=0.4")]["exposure_low"]["k"] == 2
    assert rows[("statue", "certified", "[0.2,0.4)")]["n_episodes"] == 0
    assert all(r["toppled_net"] == 1.0 for r in summary["main_tau"] if r["n_episodes"])
    assert [r["sigma_bin"] for r in summary["main_sigma"]][:3] == ["[1,2)", ">=2", "<1"]
    report_md = (work / "report" / "report.md").read_text()
    print(report_md)
    assert "formal, Manifest %s" % frz["manifest_hash"] in report_md

    pilot_cfg = RW.copy_tree(cfg, work / "pilot_configs")
    pilot = ["--instances", "gated.jsonl", "--calibration", "calibration.yaml", "--pilot"]
    result = run_cli(work, pilot_cfg, *run_args("nominal", "pilot_nominal", *pilot))
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps(out, sort_keys=True))
    assert out["pilot"] is True and out["manifest_hash"] is None and out["ran"] == 10
    assert all(json.loads(line)["pilot"] is True for line in
               (work / "pilot_nominal" / "episodes.jsonl").read_text().splitlines())
    pilot_rep = hazardctl(work, pilot_cfg, "report", "--runs", "pilot_nominal", "--out-dir", "pilot_report")
    assert pilot_rep["pilot"] is True
    assert "PILOT (not a formal result)" in (work / "pilot_report" / "report.md").read_text()

    mixed = run_cli(work, cfg, "report", "--runs", "run_nominal", "pilot_nominal", "--out-dir", "mixed")
    print(mixed.stderr.strip().splitlines()[0])
    assert mixed.returncode == 2 and "pilot and formal results cannot be mixed" in mixed.stderr
    assert not (work / "mixed").exists()

    no_name = run_args("nominal", "x1", *formal)
    index = no_name.index("--policy-name")
    del no_name[index:index + 2]
    proc = run_cli(work, cfg, *no_name)
    assert proc.returncode == 2 and "--policy-name" in proc.stderr
    proc = run_cli(work, cfg, *run_args("nominal", "x2", "--instances", "gated.jsonl"))
    assert proc.returncode == 2 and "needs --pilot" in proc.stderr
    proc = run_cli(work, cfg, *run_args("nominal", "x3", *formal, "--pilot"))
    assert proc.returncode == 2 and "is a formal run" in proc.stderr
    proc = run_cli(work, W.REPO_CONFIG_DIR, *run_args("nominal", "x4", *formal))
    assert proc.returncode == 2 and "run.yaml: 6 unresolved value(s)" in proc.stderr
    proc = run_cli(work, cfg, "run", *HOST, "--policy", "pipeline_world:make_follower", "--policy-name", "follower",
                   "--seeds", 0, "--env", 0, "--out-dir", "x5", "--task-context", "tc.yaml", "--bd", "bd.yaml",
                   *formal)
    assert proc.returncode == 2 and "is a probe source of TaskContext sweep/L0" in proc.stderr
    assert not any((work / name).exists() for name in ("x1", "x2", "x3", "x4", "x5"))
