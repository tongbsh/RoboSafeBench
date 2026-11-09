"""``run_stage`` on FakeHost, in process (R1 items 1-3): pairing, ``control_onset_s``, refusals, resume, records."""
import json

import pytest
import yaml

import run_world as RW
from run_world import run_world_chain  # noqa: F401 (fixture)
from hazardlib.calibrate import CensoredTStopError
from hazardlib.gates import load_low_exposure_radius_m
from hazardlib.manifest.freeze import ManifestError
from hazardlib.pipeline_io import read_task_context, write_yaml
from hazardlib.run import (EPISODES_FILE, PolicyInProbeSetError, RunDirectoryError, RunInputError, RunSettings,
                           load_formal, load_pilot, read_episode_records, run_stage, summarize)
from hazardlib.run.runner import RUN_META_FILE


def _run(world, out, make=RW.make_nominal, seeds=(0,), inputs=None, labels=None, policy_name=None, host=None,
         allow_censored=False, monitor_topple=True, settings=None, **policy_args):
    inputs = inputs or load_formal(world.frozen_dir, [world.tc_path], world.bd, world.config_dir)
    host = host or RW.make_host()
    policy = make(host, **policy_args)
    session = world.session(host, inputs.provenance)
    settings = settings or RunSettings.load(world.config_dir)
    result = run_stage(session, inputs, policy, policy_name or policy.name, settings, out,
                       list(seeds), 0, load_low_exposure_radius_m(world.config_dir), labels, None, None,
                       monitor_topple, allow_censored)
    return result, read_episode_records(out / EPISODES_FILE, repair=False)


def test_controls_are_paired_and_the_topple_reference_is_the_nominal_onset(run_world_chain, tmp_path):
    world = run_world_chain
    result, records = _run(world, tmp_path / "out")
    assert result.planned == result.ran == 5 and result.ran_by_kind == {"hazard": 3, "control": 2}
    onsets = {i.instance_id: i.onset_time_s for i in world.instances}
    controls = [r for r in records if r["kind"] == "control"]
    assert [(c["episode_id"], c["paired_instance_ids"]) for c in controls] == [
        ("control:fall_long.sweep.L0.s0", ["fall_long.sweep.L0.s0", "fall_wait.sweep.L0.s0"]),
        ("control:fall_short.sweep.L0.s0", ["fall_short.sweep.L0.s0"])]
    for control in controls:
        assert control["control_onset_s"] == onsets[control["paired_instance_ids"][0]]
        assert control["log"]["instance_id"] == control["episode_id"]
        assert control["log"]["toppled"] is False and control["log"]["max_tilt_deg"] == pytest.approx(0.0, abs=1e-9)
        assert control["task_entities"] == [RW.VASE] and control["instance_id"] is None
    for hazard in (r for r in records if r["kind"] == "hazard"):
        assert hazard["control_id"] in {c["episode_id"] for c in controls}
        assert hazard["log"]["toppled"] is True and hazard["log"]["max_tilt_deg"] == pytest.approx(60.0, abs=1e-6)
    settings = RunSettings.load(world.config_dir)
    meta = yaml.safe_load((tmp_path / "out" / RUN_META_FILE).read_text())
    summary = summarize([dict(meta, run_dir="out")], records, settings.report, settings.config_hash)
    assert [row["toppled_net"] for row in summary["main_tau"]] == [1.0, 1.0, 1.0]

    # the same control referenced at the episode start instead of the onset would count the settling as a topple
    session = world.session()
    late = session.run_episode(0, None, RW.Nominal(session.host), RW.TASK, RW.LAYOUT, 0, 1.0, control_id="c",
                               task_entities=[RW.VASE], control_onset_s=0.0)
    assert late.log.toppled is True and late.log.max_tilt_deg == pytest.approx(90.0, abs=1e-6)


def test_run_episode_passes_control_onset_and_entities_to_arm(run_world_chain):
    session = run_world_chain.session()
    calls = []
    arm = session.runtime.arm

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return arm(*args, **kwargs)

    session.runtime.arm = spy
    session.run_episode(0, None, RW.Nominal(session.host), RW.TASK, RW.LAYOUT, 0, 0.2, control_id="c",
                        task_entities=[RW.VASE], control_onset_s=0.15)
    session.run_episode(0, None, RW.Nominal(session.host), RW.TASK, RW.LAYOUT, 0, 0.2, control_id="d")
    assert calls[0] == ((0, None), {"task_entities": [RW.VASE], "control_id": "c", "control_onset_s": 0.15})
    assert calls[1] == ((0, None), {"task_entities": None, "control_id": "d", "control_onset_s": None})


def test_records_carry_the_run_identity_and_the_instance_data(run_world_chain, tmp_path):
    world = run_world_chain
    _, records = _run(world, tmp_path / "out", labels=["wait_solvable"])
    hazards = {r["spec_id"]: r for r in records if r["kind"] == "hazard"}
    assert sorted(hazards) == ["fall_long", "fall_wait"]
    wait = hazards["fall_wait"]
    instance = [i for i in world.instances if i.spec_id == "fall_wait"][0]
    assert wait["tau_s"] == pytest.approx(0.3) and wait["sigma"] == pytest.approx(0.3 / 0.2)
    assert wait["sigma_is_upper_bound"] is False and wait["family"] == "intrusion.fall"
    assert wait["label"] == "wait_solvable" and wait["in_main_score"] is True
    assert wait["pilot"] is False and wait["manifest_hash"] == world.manifest_hash
    assert wait["log"]["manifest_hash"] == world.manifest_hash and wait["log"]["bd_hash"] == world.bd.hash()
    assert wait["log"]["exposure_low"] is False and wait["log"]["first_official_contact_t_s"] is not None
    assert wait["official_contact_count"] >= 1 and wait["success_t_s"] is not None
    assert wait["horizon_s"] == pytest.approx(1.0 + 1.5) and wait["n_control_steps"] == 125
    assert wait["inference_time_s"]["n_steps"] == 125 and wait["inference_time_s"]["total_s"] > 0
    assert wait["policy_name"] == "nominal_bot" and wait["policy"] == {"kind": "external_policy",
                                                                      "name": "nominal_bot", "under_test": True}
    assert wait["first_visible_t_s"] is not None and wait["frames"] is None
    assert wait["instance_id"] == instance.instance_id == wait["episode_id"] and wait["seed"] == 0


def test_a_probe_source_cannot_be_the_policy_under_test(run_world_chain, tmp_path):
    with pytest.raises(PolicyInProbeSetError) as err:
        _run(run_world_chain, tmp_path / "a", make=RW.W.make_follower)
    assert "policy 'follower' is a probe source of TaskContext sweep/L0" in str(err.value)
    with pytest.raises(PolicyInProbeSetError, match="'demo'"):          # the policy's own name counts too
        _run(run_world_chain, tmp_path / "b", make=RW.W.make_demo, policy_name="renamed_demo")
    with pytest.raises(PolicyInProbeSetError, match="'follower_fast'"):
        _run(run_world_chain, tmp_path / "c", policy_name="follower_fast")
    assert not any((tmp_path / name).exists() for name in "abc")


def test_formal_inputs_must_verify(run_world_chain, tmp_path):
    world = run_world_chain
    frozen = RW.copy_tree(world.frozen_dir, tmp_path / "frozen")
    lines = (frozen / "instances.jsonl").read_text().splitlines()
    record = json.loads(lines[0])
    record["onset_time_s"] += 0.01
    (frozen / "instances.jsonl").write_text("\n".join([json.dumps(record)] + lines[1:]) + "\n")
    with pytest.raises(ManifestError, match="was modified"):
        load_formal(frozen, [world.tc_path], world.bd, world.config_dir)
    other_bd = RW.RunHost(RW.W.world_config(tasks=RW.tasks()), RW.DECEL_MPS2, ccd=False).describe()
    with pytest.raises(ManifestError, match="manifest bd_hash"):
        load_formal(world.frozen_dir, [world.tc_path], other_bd, world.config_dir)
    tc = read_task_context(world.tc_path).to_dict()
    tc["provenance"]["eef_speed_median_mps"] = 0.4
    write_yaml(tmp_path / "tc_other.yaml", tc)
    with pytest.raises(RunInputError, match="was built on TaskContext"):
        load_formal(world.frozen_dir, [tmp_path / "tc_other.yaml"], world.bd, world.config_dir)
    configs = RW.copy_tree(world.config_dir, tmp_path / "configs")
    RW.W.edit_yaml(configs / "semantics.yaml", lambda d: d["toppling"].update(tilt_threshold_deg=45.0))
    with pytest.raises(ManifestError, match="config 'semantics' changed"):
        load_formal(world.frozen_dir, [world.tc_path], world.bd, configs)


def test_the_host_must_have_the_inputs_bd(run_world_chain, tmp_path):
    host = RW.RunHost(RW.W.world_config(tasks=RW.tasks()), RW.DECEL_MPS2, ccd=False)
    with pytest.raises(ValueError, match="is not the run's BD"):
        _run(run_world_chain, tmp_path / "out", host=host)


def test_resume_skips_finished_episodes_and_drops_an_interrupted_line(run_world_chain, tmp_path):
    world, out = run_world_chain, tmp_path / "out"
    first, records = _run(world, out, labels=["uncertified"])
    assert first.ran == 2 and len(records) == 2
    text = (out / EPISODES_FILE).read_text()
    again, _ = _run(world, out, labels=["uncertified"])
    assert again.ran == 0 and again.skipped == 2 and (out / EPISODES_FILE).read_text() == text
    lines = text.splitlines(keepends=True)
    (out / EPISODES_FILE).write_text(lines[0] + lines[1][: len(lines[1]) // 2])      # interrupted write
    resumed, records = _run(world, out, labels=["uncertified"])
    assert resumed.ran == 1 and resumed.skipped == 1 and [r["kind"] for r in records] == ["control", "hazard"]
    more, records = _run(world, out, labels=["uncertified"], seeds=(0, 1))
    assert more.ran == 2 and more.skipped == 2 and [r["seed"] for r in records] == [0, 0, 1, 1]
    rebinned = RW.copy_tree(world.config_dir, tmp_path / "rebinned")       # report settings may change ...
    RW.W.edit_yaml(rebinned / "run.yaml", lambda d: d["report"].update(tau_bin_edges_s=[0.25]))
    same, _ = _run(world, out, labels=["uncertified"], seeds=(0, 1), settings=RunSettings.load(rebinned))
    assert same.ran == 0 and same.skipped == 4
    longer = RW.copy_tree(world.config_dir, tmp_path / "longer")           # ... the episode settings may not
    RW.W.edit_yaml(longer / "run.yaml", lambda d: d["episode"].update(horizon_extra_s=2.0))
    with pytest.raises(RunDirectoryError, match="episode_settings"):
        _run(world, out, labels=["uncertified"], settings=RunSettings.load(longer))
    with pytest.raises(RunDirectoryError, match="policy_name: directory has 'nominal_bot', this run 'cautious'"):
        _run(world, out, make=RW.make_cautious, labels=["uncertified"])
    with pytest.raises(RunDirectoryError, match="monitor_topple"):
        _run(world, out, labels=["uncertified"], monitor_topple=False)


def test_pilot_inputs_and_a_censored_calibration(run_world_chain, tmp_path):
    world = run_world_chain
    censored = yaml.safe_load(world.calibration_path.read_text())
    censored["t_stop"].update(n_valid=censored["t_stop"]["n_valid"] - 1, n_censored=1)
    censored["t_stop_is_lower_bound"] = True
    write_yaml(tmp_path / "censored.yaml", censored)
    inputs = load_pilot(world.gated_path, [world.tc_path], world.bd, tmp_path / "censored.yaml")
    assert inputs.pilot is True and inputs.manifest_hash is None
    with pytest.raises(CensoredTStopError):
        _run(world, tmp_path / "refused", inputs=inputs)
    assert not (tmp_path / "refused").exists()
    _, records = _run(world, tmp_path / "out", inputs=inputs, allow_censored=True, labels=["uncertified"],
                      monitor_topple=False)
    hazard = records[1]
    assert hazard["pilot"] is True and hazard["manifest_hash"] is None and hazard["log"]["manifest_hash"] is None
    assert hazard["sigma_is_upper_bound"] is True and hazard["sigma"] == pytest.approx(0.1 / censored["t_stop"]
                                                                                       ["median_s"])
    assert hazard["task_entities"] is None and hazard["log"]["toppled"] is None
    uncalibrated = load_pilot(world.gated_path, [world.tc_path], world.bd, None)
    _, records = _run(world, tmp_path / "nocal", inputs=uncalibrated, labels=["uncertified"])
    assert records[1]["sigma"] is None and records[1]["sigma_is_upper_bound"] is None
    ungated = tmp_path / "ungated.jsonl"
    ungated.write_text("\n".join(json.dumps({"instance": dict(i.to_dict(), certificate=None)})
                                 for i in world.instances) + "\n")
    with pytest.raises(RunInputError, match="has no certificate"):
        load_pilot(ungated, [world.tc_path], world.bd, None)
