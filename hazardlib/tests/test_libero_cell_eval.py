"""Layout selection, the global queue, resume bookkeeping and the per-cell aggregation of the generated-cell
evaluation (no simulator; synthetic generation folders and run records, every number checked by hand)."""
import json
import os

import pytest
import yaml

from hazardhosts.libero import cell_eval
from hazardhosts.libero.cell_report import build_cell_report, render_markdown
from hazardhosts.libero.cell_select import (IncompleteTask, build_queue, expected_counts, layout_cells,
                                            plan_layout_episodes, select_all, select_task, selection_document)
from hazardlib.pipeline_io import read_records
from hazardlib.run.plan import plan_episodes
from hazardlib.run.runner import RECORD_VERSION, append_record, read_episode_records
from hazardlib.run.settings import ReportSettings
from hazardlib.semantics.event_log import HazardEventLog
from hazardlib.spec import HazardInstance, TaskContext

SUITE, TASK = "libero_toy", "toy task"
TASK_ID = "libero_toy/" + TASK
SETTINGS = ReportSettings(tau_bin_edges_s=[0.2], sigma_bin_edges=[1.0], interval_method="clopper_pearson",
                          confidence=0.95)


def _instance(instance_id, demo, onset, hit, label="uncertified", family="intrusion.reach"):
    return {
        "instance_id": instance_id, "spec_id": instance_id.split(".")[0], "family": family,
        "agent": {"kind": "object", "shape": "sphere", "half_extents_m": [0.03], "mass_kg": 0.2},
        "keyframes": {"time_s": [onset, hit], "pos": [[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
                      "quat": [[1.0, 0.0, 0.0, 0.0]] * 2},
        "nominal_hit_point": [0.0, 0.0, 0.5], "nominal_hit_time_s": hit, "onset_time_s": onset,
        "release_time_s": onset, "release_velocity": {"lin_mps": [0.0] * 3, "ang_radps": [0.0] * 3},
        "certificate": {"g0": True, "g1": label == "wait_solvable", "g2": None, "label": label, "notes": ""},
        "provenance": {"task_id": TASK_ID, "layout_id": "%s/%s" % (SUITE, demo)}}


def _cell_result(instance, g1_passed, sigma_target):
    return {"selected": {"instance": instance, "g1": {"passed": g1_passed},
                         "sigma": {"target": sigma_target, "sigma_achieved": sigma_target * 0.95, "tau_s": 0.4,
                                   "t_stop_s": 0.3}}}


def _write_layout(task_dir, demo, instances):
    """``instances``: [(instance dict, cell, sigma key, g1)]; cells without an entry are rejected."""
    folder = task_dir / "trajectories" / demo
    folder.mkdir(parents=True)
    families = {"intrusion.fall@vertical": {"0.75": {"selected": None}, "1.5": {"selected": None}},
                "intrusion.reach@el30": {"0.75": {"selected": None}, "1.5": {"selected": None}}}
    for instance, cell, key, g1 in instances:
        families[cell][key] = _cell_result(instance, g1, float(key))
    (folder / "validation.json").write_text(json.dumps({"demo": demo, "status": "processed", "families": families}))
    (folder / "context_metadata.yaml").write_text(yaml.safe_dump({"phases": [{"name": "p", "t_start_s": 0.0,
                                                                              "t_end_s": 1.0}]}))
    (folder / "instances.jsonl").write_text("".join(json.dumps({"instance": i[0]}) + "\n" for i in instances))
    return folder


def _generation(tmp_path, name="libero_toy_task0"):
    """Coverage order demo_3, demo_1 (replay failed, unusable), demo_0, demo_2, demo_4: demo_3 admits two
    instances (same onset, so one shared control), demo_0 none, demo_2 one, demo_4 two."""
    task_dir = tmp_path / name
    rows = [{"demo": d, "replay_success": d != "demo_1"} for d in ("demo_3", "demo_1", "demo_0", "demo_2", "demo_4")]
    task_dir.mkdir()
    (task_dir / "coverage.json").write_text(json.dumps({"demonstrations": rows}))
    _write_layout(task_dir, "demo_3", [
        (_instance("a.demo_3", "demo_3", 0.2, 0.6), "intrusion.reach@el30", "0.75", False),
        (_instance("b.demo_3", "demo_3", 0.2, 0.6, "wait_solvable", "intrusion.fall"), "intrusion.fall@vertical",
         "1.5", True)])
    _write_layout(task_dir, "demo_0", [])
    _write_layout(task_dir, "demo_2", [(_instance("a.demo_2", "demo_2", 0.3, 0.7), "intrusion.reach@el30", "1.5",
                                        False)])
    _write_layout(task_dir, "demo_4", [
        (_instance("a.demo_4", "demo_4", 0.2, 0.6), "intrusion.reach@el30", "0.75", True),
        (_instance("b.demo_4", "demo_4", 0.25, 0.6), "intrusion.reach@el30", "1.5", True)])
    return tmp_path


# ---------------------------------------------------------------------------------------------- selection

def test_layout_cells_maps_instances_and_admission(tmp_path):
    root = _generation(tmp_path)
    validation = json.loads((root / "libero_toy_task0/trajectories/demo_3/validation.json").read_text())
    cells, coverage = layout_cells(validation)
    assert cells["a.demo_3"] == {"cell": "intrusion.reach@el30", "family": "intrusion.reach", "elevation": "el30",
                                 "target": "0.75", "g1_passed": False, "label": "uncertified",
                                 "sigma_achieved": pytest.approx(0.7125), "tau_s": 0.4, "t_stop_s": 0.3}
    assert cells["b.demo_3"]["g1_passed"] is True and cells["b.demo_3"]["label"] == "wait_solvable"
    assert coverage == {"intrusion.fall@vertical|0.75": False, "intrusion.fall@vertical|1.5": True,
                        "intrusion.reach@el30|0.75": True, "intrusion.reach@el30|1.5": False}


def test_first_n_layouts_with_an_admitted_instance_in_coverage_order(tmp_path):
    root = _generation(tmp_path)
    chosen = select_task(root, "libero_toy", 0, 2, 1.0, 0)
    assert [c.demo for c in chosen] == ["demo_3", "demo_2"]          # demo_1 unusable, demo_0 admits nothing
    assert [c.demo for c in select_task(root, "libero_toy", 0, 9, 1.0, 0)] == ["demo_3", "demo_2", "demo_4"]
    assert [c.demo for c in select_task(root, "libero_toy", 0, 1, 1.0, 0)] == ["demo_3"]
    first = chosen[0]
    assert set(first.cells) == {"a.demo_3", "b.demo_3"} and first.cells["a.demo_3"]["cell"] == "intrusion.reach@el30"


def test_episode_plan_shares_controls_and_matches_the_core_planner(tmp_path):
    root = _generation(tmp_path)
    first = select_task(root, "libero_toy", 0, 1, 1.0, 0)[0]
    # a and b: same layout, onset 0.2 and length -> one control, planned right before the first hazard
    assert first.episodes == [("control:a.demo_3", "control"), ("a.demo_3", "hazard"), ("b.demo_3", "hazard")]
    assert (first.n_hazard, first.n_control) == (2, 1)
    instances, _ = read_records(root / "libero_toy_task0/trajectories/demo_3/instances.jsonl")
    context = TaskContext.from_dict({
        "task_id": TASK_ID, "layout_id": "libero_toy/demo_3", "entities": {},
        "phases": [{"name": "p", "t_start_s": 0.0, "t_end_s": 1.0}],
        "nominal_ee_path": {"time_s": [0.0, 1.0], "pos": [[0.0, 0.0, 0.3], [0.5, 0.0, 0.3]],
                            "quat": [[1.0, 0.0, 0.0, 0.0]] * 2},
        "occupancy": {"points": [], "time_s": []},
        "workspace_bounds": {"min_m": [-1.0, -1.0, 0.0], "max_m": [1.0, 1.0, 1.0]},
        "support_surfaces": [], "provenance": {}})
    real = plan_episodes(instances, {(TASK_ID, "libero_toy/demo_3"): context}, [0], 1.0, None)
    assert [(p.episode_id, p.kind) for p in real] == plan_layout_episodes(instances, 1.0, 1.0, 0)
    other_extra = plan_layout_episodes(instances, 1.0, 2.5, 0)
    assert other_extra == first.episodes                      # the ids do not depend on the horizon here


def test_incomplete_tasks_are_reported_not_half_selected(tmp_path):
    root = _generation(tmp_path)
    (root / "libero_toy_task0/trajectories/demo_2/instances.jsonl").unlink()     # still being validated
    with pytest.raises(IncompleteTask, match="demo_2"):
        select_task(root, "libero_toy", 0, 2, 1.0, 0)
    assert [c.demo for c in select_task(root, "libero_toy", 0, 1, 1.0, 0)] == ["demo_3"]   # before it: complete
    assert [c.demo for c in select_task(root, "libero_toy", 0, 2, 1.0, 0, allow_incomplete=True)] == ["demo_3",
                                                                                                    "demo_4"]
    selected, skipped = select_all(root, [("libero_toy", 0), ("libero_toy", 1)], 2, 1.0, 0)
    assert selected == {} and "demo_2" in skipped["libero_toy_task0"]
    assert "coverage.json" in skipped["libero_toy_task1"]


def test_validation_and_instances_must_agree(tmp_path):
    root = _generation(tmp_path)
    path = root / "libero_toy_task0/trajectories/demo_2/validation.json"
    data = json.loads(path.read_text())
    data["families"]["intrusion.reach@el30"]["1.5"] = {"selected": None}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disagree"):
        select_task(root, "libero_toy", 0, 2, 1.0, 0)


# ------------------------------------------------------------------------------------------ queue / resume

def test_queue_is_layout_major_and_resume_drops_finished_episodes(tmp_path):
    root = _generation(tmp_path)
    selected = {"libero_toy_task0": select_task(root, "libero_toy", 0, 3, 1.0, 0)}
    queue = build_queue(selected, {})
    assert [(w.demo, w.episode_id, w.kind) for w in queue] == [
        ("demo_3", "control:a.demo_3", "control"), ("demo_3", "a.demo_3", "hazard"), ("demo_3", "b.demo_3", "hazard"),
        ("demo_2", "control:a.demo_2", "control"), ("demo_2", "a.demo_2", "hazard"),
        ("demo_4", "control:a.demo_4", "control"), ("demo_4", "a.demo_4", "hazard"),
        ("demo_4", "control:b.demo_4", "control"), ("demo_4", "b.demo_4", "hazard")]
    assert expected_counts(selected) == {"libero_toy_task0": {"layouts": 3, "hazard": 5, "control": 4}}
    rest = build_queue(selected, {"libero_toy_task0": {"control:a.demo_3", "a.demo_3", "a.demo_2"}})
    assert [w.episode_id for w in rest][:3] == ["b.demo_3", "control:a.demo_2", "control:a.demo_4"]
    assert len(rest) == 6 and build_queue(selected, {"libero_toy_task0": {w.episode_id for w in queue}}) == []


def _record(kind, episode_id, instance_id=None, success=True, contact=False, invalid="none", exposure=None,
            control_id=None, paired=(), label=None, seed=0):
    log = HazardEventLog.new(episode_id, first_official_contact_t_s=0.5 if contact else None,
                             exposure_low=exposure, invalid_class=invalid).to_dict()
    return {"record_version": RECORD_VERSION, "kind": kind, "episode_id": episode_id, "instance_id": instance_id,
            "control_id": control_id or episode_id, "paired_instance_ids": list(paired), "task_id": TASK_ID,
            "layout_id": "libero_toy/demo", "policy_name": "toy", "seed": seed, "label": label,
            "family": None if instance_id is None else "intrusion.reach", "tau_s": None, "sigma": None,
            "success_t_s": 1.0 if success else None, "n_control_steps": 20, "pilot": True, "manifest_hash": None,
            "frames": None, "log": log}


def test_done_ids_repairs_a_torn_line_and_ignores_other_seeds_and_policies(tmp_path):
    out = tmp_path / "out"
    directory = cell_eval.run_dir(out, "libero_toy_task0", "toy")
    directory.mkdir(parents=True)
    append_record(directory / "episodes.jsonl", _record("hazard", "a.demo_3", "a.demo_3"))
    append_record(directory / "episodes.jsonl", _record("hazard", "b.demo_3", "b.demo_3", seed=1))
    with open(str(directory / "episodes.jsonl"), "a") as handle:
        handle.write('{"record_version": "1", "kind": "haz')       # a worker killed mid-write
    done = cell_eval.done_ids(out, {"libero_toy_task0": None, "libero_toy_task9": None}, "toy", 0)
    assert done == {"libero_toy_task0": {"a.demo_3"}, "libero_toy_task9": set()}
    assert (directory / "episodes.jsonl").read_text().endswith("}\n")       # repaired on disk


def test_append_locked_never_writes_an_episode_twice(tmp_path):
    directory = tmp_path / "run"
    record = _record("hazard", "a.demo_3", "a.demo_3")
    assert cell_eval.append_locked(directory, record) is True
    assert cell_eval.append_locked(directory, record) is False         # a retry after a lost reply
    assert cell_eval.append_locked(directory, _record("hazard", "b.demo_3", "b.demo_3")) is True
    assert [r["episode_id"] for r in read_episode_records(directory / "episodes.jsonl", repair=False)] == \
        ["a.demo_3", "b.demo_3"]


def test_merge_meta_copies_once_and_refuses_another_run(tmp_path):
    base = {"meta_version": "1", "pilot": True, "manifest_hash": None, "benchmark": "libero", "bd_hash": "b",
            "calibration_hash": None, "policy_name": "toy", "policy": {}, "episode_settings": {"x": 1},
            "monitor_topple": True}
    for name, meta in (("s1", base), ("s2", dict(base, sources={"layout_folder": "elsewhere"})),
                       ("s3", dict(base, monitor_topple=False))):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run_meta.yaml").write_text(yaml.safe_dump(meta))
    directory = tmp_path / "run"
    cell_eval.merge_meta(tmp_path / "s1", directory)
    cell_eval.merge_meta(tmp_path / "s2", directory)                    # same identity, other sources: fine
    with pytest.raises(Exception, match="monitor_topple"):
        cell_eval.merge_meta(tmp_path / "s3", directory)


def test_failed_file_holds_only_the_latest_invocation_and_skipped_entities_are_listed(tmp_path):
    assert cell_eval.archive_failures(tmp_path) is None
    (tmp_path / "failed.jsonl").write_text('{"episode_id": "a"}\n')
    archived = cell_eval.archive_failures(tmp_path)
    assert archived.name.startswith("failed.2") and archived.read_text() == '{"episode_id": "a"}\n'
    assert not (tmp_path / "failed.jsonl").exists()
    cell_eval.note_skipped_entities(tmp_path / "new" / "run", ["wooden_cabinet_1_middle_region"])   # creates the dir
    cell_eval.note_skipped_entities(tmp_path, ["wooden_cabinet_1_middle_region"])
    cell_eval.note_skipped_entities(tmp_path, ["b_region", "wooden_cabinet_1_middle_region"])
    assert (tmp_path / cell_eval.SKIPPED_ENTITIES_FILE).read_text().split() == \
        ["b_region", "wooden_cabinet_1_middle_region"]


def test_worker_command_and_environment_pin_gpu_and_server(tmp_path):
    class Args:
        gen_root, out, model, checkpoint, client_src, seed, video = "g", "o", "pi05", "ck", "cs", 3, "none"
    command = cell_eval.worker_command(Args, 5, 18012)
    assert command[command.index("--worker-index") + 1] == "5" and "--worker" in command
    assert command[command.index("--port") + 1] == "18012" and command[command.index("--seed") + 1] == "3"
    env = cell_eval.worker_env(0)
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["MUJOCO_EGL_DEVICE_ID"] == "0"
    with pytest.raises(SystemExit, match="GPU 1"):
        cell_eval.main(["--gen-root", "g", "--out", str(tmp_path), "--checkpoint", "c", "--client-src", "s",
                        "--gpu", "1"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES") != "7"        # the parent environment is not modified


# ------------------------------------------------------------------------------------------- aggregation

def _selection(tmp_path):
    root = _generation(tmp_path)
    selected = {"libero_toy_task0": select_task(root, "libero_toy", 0, 3, 1.0, 0)}
    return selection_document(selected, {"libero_toy_task7": "not validated"}, {"seed": 0})


def _records():
    """demo_3: a (reach el30 0.75, success, contact), b (fall 1.5, failure, no contact); demo_2: a (reach el30
    1.5, success, contact but low exposure); demo_4: a (reach el30 0.75, success, no contact), b (reach el30
    1.5, sim_unstable); controls: all succeed except control:a.demo_2 and a stray record not in the selection."""
    haz = [("a.demo_3", True, True, "none", None, "uncertified"), ("b.demo_3", False, False, "none", None,
                                                                     "wait_solvable"),
           ("a.demo_2", True, True, "none", True, "uncertified"), ("a.demo_4", True, False, "none", None, "uncertified"),
           ("b.demo_4", True, True, "sim_unstable", None, "uncertified")]
    records = [_record("control", "control:a.demo_3", paired=["a.demo_3", "b.demo_3"]),
               _record("control", "control:a.demo_2", success=False, paired=["a.demo_2"]),
               _record("control", "control:a.demo_4", paired=["a.demo_4"]),
               _record("control", "control:b.demo_4", paired=["b.demo_4"])]
    for episode_id, success, contact, invalid, exposure, label in haz:
        control = {"a.demo_3": "control:a.demo_3", "b.demo_3": "control:a.demo_3", "a.demo_2": "control:a.demo_2",
                   "a.demo_4": "control:a.demo_4", "b.demo_4": "control:b.demo_4"}[episode_id]
        records.append(_record("hazard", episode_id, episode_id, success, contact, invalid, exposure, control,
                               [episode_id], label))
    records.append(_record("hazard", "stray", "stray", True, False, "none", None, "control:a.demo_3", ["stray"],
                           "uncertified"))
    return records


def test_cell_aggregation_counts_and_wilson_intervals(tmp_path):
    report = build_cell_report(_selection(tmp_path), _records(), SETTINGS, "toy")
    assert report["interval"]["method"] == "wilson"                    # whatever the run configuration says
    assert report["n_hazard_records"] == 5 and report["n_stray_hazard_records"] == 1
    assert report["n_control_records"] == 4 and report["controls"]["success"]["k"] == 3
    rows = {(r["cell"], r["target"]): r for r in report["tables"]["overall"]}
    reach = rows[("intrusion.reach@el30", "0.75")]               # a.demo_3 and a.demo_4
    assert (reach["n_episodes"], reach["n_main"]) == (2, 2)
    assert (reach["success"]["k"], reach["contact"]["k"], reach["contact_free_success"]["k"]) == (2, 1, 1)
    assert reach["success"]["ci"][0] == pytest.approx(0.3424, abs=1e-3)       # Wilson 2/2 at 95%
    assert (reach["g1_passed"], reach["g1_failed"], reach["labels"]) == (1, 1, {"uncertified": 2})
    assert reach["sigma_achieved_mean"] == pytest.approx(0.7125) and reach["control_success"]["k"] == 2
    assert (reach["layouts_admitted"], reach["layouts"]) == (2, 3)
    heavy = rows[("intrusion.reach@el30", "1.5")]                 # a.demo_2 (low exposure), b.demo_4 (invalid)
    assert (heavy["n_episodes"], heavy["n_main"], heavy["exposure_low"], heavy["invalid"]) == (2, 0, 1, 1)
    assert heavy["success"]["rate"] is None and heavy["n_controls"] == 0
    fall = rows[("intrusion.fall@vertical", "1.5")]
    assert (fall["success"]["k"], fall["success"]["n"], fall["g1_passed"], fall["labels"]) == \
        (0, 1, 1, {"wait_solvable": 1})
    total = rows[("ALL", "all")]
    assert (total["n_episodes"], total["n_main"], total["exposure_low"], total["invalid"]) == (5, 3, 1, 1)
    assert (total["n_controls"], total["control_success"]["k"], total["control_success"]["n"]) == (2, 2, 2)
    per_cell = rows[("intrusion.reach@el30", "all")]
    assert per_cell["n_episodes"] == 4 and per_cell["labels"] == {"uncertified": 4}
    assert set(report["tables"]) == {"overall", "libero_toy"} and report["skipped_tasks"] == \
        {"libero_toy_task7": "not validated"}
    assert report["progress"]["libero_toy_task0"] == {"layouts": 3, "hazard_expected": 5, "hazard_done": 5,
                                                      "control_expected": 4, "control_done": 4}
    text = render_markdown(report)
    assert "## Overall" in text and "## libero_toy" in text and "intrusion.reach@el30" in text
    assert "libero_toy_task7 (not validated)" in text and "5/5" in text
    assert " slots |" in text and "cell slots" in text


def test_aggregation_of_an_unfinished_run_shows_progress(tmp_path):
    report = build_cell_report(_selection(tmp_path), _records()[:2] + _records()[4:6], SETTINGS, "toy")
    progress = report["progress"]["libero_toy_task0"]
    assert (progress["hazard_done"], progress["hazard_expected"]) == (2, 5)
    assert (progress["control_done"], progress["control_expected"]) == (2, 4)
    json.dumps(report)                                              # the JSON report must be serialisable


def test_generation_level_follows_the_task_configs_and_refuses_a_mix(tmp_path):
    from hazardhosts.libero.pilot import assemble_config
    assert cell_eval.generation_level(tmp_path) is None
    assemble_config(tmp_path / "libero_toy_task0" / "config", "M")
    assemble_config(tmp_path / "libero_toy_task1" / "config", "M")
    assert cell_eval.generation_level(tmp_path) == "M"
    assemble_config(tmp_path / "libero_toy_task2" / "config")
    with pytest.raises(SystemExit, match="mixes hazard scale levels"):
        cell_eval.generation_level(tmp_path)


def test_episodes_left_when_every_worker_gave_up_are_failures(tmp_path):
    import queue
    (tmp_path / "gen").mkdir()
    root = _generation(tmp_path / "gen")
    selected = {"libero_toy_task0": select_task(root, "libero_toy", 0, 3, 1.0, 0)}
    pending = queue.Queue()
    for work in build_queue(selected, {})[:2]:
        pending.put(work)
    progress = cell_eval.Progress(2, tmp_path / "eval.log")
    assert cell_eval.abandon_left(pending, progress, tmp_path) == 2
    assert progress.failed == 2
    lines = [json.loads(x) for x in (tmp_path / "failed.jsonl").read_text().splitlines()]
    assert [x["reason"] for x in lines] == ["no worker left"] * 2
    assert "ABANDONED 2 episodes" in (tmp_path / "eval.log").read_text()
    assert cell_eval.abandon_left(pending, progress, tmp_path) == 0
