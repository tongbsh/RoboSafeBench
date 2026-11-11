"""Original LIBERO pilot: probes -> contexts -> instances -> gates -> policy run -> report.

All outputs are explicitly pilot-only. Numeric choices live in pilot/libero YAML.
The test policies are never used as probe sources.
"""
import json
from pathlib import Path
import shutil

import yaml

from hazardlib.calibrate.protocol import TStopProtocolSettings, run_t_stop_protocol, write_calibration
from hazardlib.driving import EpisodeSession
from hazardlib.gates import GateSettings, gate_instances
from hazardlib.instantiate import InstantiateSettings, instantiate
from hazardlib.instantiate.builder import InstanceRejected
from hazardlib.pipeline_io import write_records, write_yaml
from hazardlib.probe import ProbeSettings, build_task_context, record_probe_trace
from hazardlib.run.inputs import load_pilot
from hazardlib.run.report import load_runs, summarize, write_report
from hazardlib.run.runner import run_stage, read_episode_records
from hazardlib.run.settings import RunSettings
from hazardlib.runtime import RuntimeConfig
from hazardlib.spec import HazardSpec

from .host import LiberoHost
from .policies import DemoReplay, DyniPolicyBridge
from .video import Mp4FrameSink


def assemble_config(directory, scale=None, levels_path=None):
    """The pilot config directory of a run. ``scale``: the D21 hazard scale level (``hazard_scale``); ``None`` and
    the default level give the legacy directory."""
    from hazardlib.config import default_config_dir
    from .hazard_scale import apply_to_config_dir
    source = default_config_dir()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for item in source.glob("*.yaml"):
        shutil.copy2(item, directory / item.name)
    shutil.copytree(source / "bd", directory / "bd", dirs_exist_ok=True)
    for item in (source / "pilot/libero").glob("*.yaml"):
        shutil.copy2(item, directory / item.name)
    apply_to_config_dir(directory, scale, levels_path)
    return directory


def spec_library(tc, config, agent):
    cfg = config["specs"]
    for phase in tc.phases:
        for frac in cfg["phase_fracs"]:
            for tau in cfg["tau_s"]:
                for family in cfg["families"]:
                    params = {} if family == "intrusion.fall" else {"direction": cfg["direction"]}
                    intensity = {"tau_s": tau}
                    if family != "intrusion.fall":
                        intensity["approach_speed_mps"] = cfg["approach_speed_mps"]
                    if family == "intrusion.reach":
                        params["speed_profile"] = cfg["reach_speed_profile"]
                    primitive = {"intrusion.fall": "free_fall", "intrusion.slide": "slide",
                                 "intrusion.reach": "reach"}[family]
                    ident = "%s-%s-%s-%s" % (family, phase.name, frac, tau)
                    yield HazardSpec.from_dict({"spec_version": "1", "id": ident, "family": family,
                        "agent": agent.to_dict(), "motion": {"primitive": primitive, "params": params},
                        "exposure": {"anchor": "nominal_ee_path", "offset_m": [0., 0., 0.],
                                     "anchor_phase": {"phase": phase.name, "frac": frac}},
                        "timing": {"mode": "nominal_time"}, "intensity": intensity,
                        "release": {"mode": "dynamic_release"}})


def prepare(root, task_indices=None):
    root = Path(root)
    cfgdir = assemble_config(root / "config")
    config = yaml.safe_load((cfgdir / "pilot.yaml").read_text())
    host = LiberoHost(config_dir=cfgdir)
    session = EpisodeSession(host, RuntimeConfig.load(cfgdir))
    reference = DemoReplay(host)
    suite = config["suite"]
    tasks = [t for t in host.task_ids() if t.startswith(suite + "/")]
    indices = config["task_indices"] if task_indices is None else task_indices
    summaries = []
    try:
        for index in indices:
            task = tasks[index]
            directory = root / ("task_%02d" % index)
            directory.mkdir(parents=True, exist_ok=True)
            baselines, selected = [], None
            for episode in range(config["probe_episodes"]):
                layout = suite + "/demo_%d" % episode
                count = len(host._demo(task, suite).actions("demo_%d" % episode))
                result = session.run_episode(0, None, reference, task, layout, config["seed"],
                    count * host.describe().timebase.control_dt_s,
                    control_id="h3-demo-%d-%d" % (index, episode), stop_on_success=True)
                baselines.append({"layout": layout, "success": result.success,
                                  "success_t_s": result.success_t_s,
                                  "invalid_class": result.log.invalid_class.value})
                if result.success and selected is None:
                    selected = layout
                (directory / "baselines.json").write_text(json.dumps(baselines, indent=2))
                print(json.dumps({"task_index": index, "baseline": baselines[-1]}), flush=True)
            if selected is None:
                summaries.append({"task_index": index, "task": task, "status": "baseline_unusable"})
                continue
            count = len(host._demo(task, suite).actions(selected.split("/")[1]))
            trace = record_probe_trace(session, reference, task, selected, config["seed"],
                                       count * host.control_dt, 0)
            entities = {"interest_%d" % i: {"name": name, "source": "bddl_goal"}
                        for i, name in enumerate(host.env.obj_of_interest) if name in host.object_poses(0)}
            tc = build_task_context([trace], ProbeSettings.load(cfgdir), static_entities=entities,
                                    declared_workspace=config["workspace"], policies_under_test=["pi0", "pi05"])
            tcpath = directory / "context.yaml"
            write_yaml(tcpath, tc.to_dict())
            print("TASK CONTEXT %s: %s" % (index, [p.name for p in tc.phases]), flush=True)
            calibration = run_t_stop_protocol(session, reference, tc, TStopProtocolSettings.load(cfgdir),
                                               config["seed"], 0)
            write_calibration(directory / "calibration.yaml", calibration)
            print("T_STOP %s %s" % (index, calibration.to_dict()["t_stop"]), flush=True)
            candidates, rejected = [], []
            # Keep at most one G0-valid candidate per family/tau in the pilot;
            # all attempted candidates and rejection reasons remain in the ledger.
            from hazardlib.gates.g0 import run_g0
            gate_settings = GateSettings.load(cfgdir)
            seen = set()
            for spec in spec_library(tc, config, host.pool[0].agent):
                key = (spec.family, spec.intensity["tau_s"])
                if key in seen:
                    continue
                try:
                    instance = instantiate(spec, tc, config["seed"], host.describe(), InstantiateSettings.load(cfgdir))
                except InstanceRejected as exc:
                    rejected.append({"stage": "instantiate", "spec_id": spec.id, "reason": str(exc)})
                    continue
                g0 = run_g0(instance, tc, host.describe(), gate_settings.g0)
                if not g0.passed:
                    rejected.append({"stage": "g0", "spec_id": spec.id, "reasons": g0.reasons})
                    continue
                candidates.append(instance)
                seen.add(key)
                print("G0 ACCEPT %s %s" % (index, spec.id), flush=True)
            gated = gate_instances(session, candidates, {(task, selected): tc}, reference, None,
                                    gate_settings, config["seed"], 0)
            write_records(directory / "instances.jsonl", [("instance", i.to_dict()) for i in gated.instances] +
                           [("rejection", r) for r in rejected + gated.rejections])
            summary = dict(gated.summary, task_index=index, task=task, layout=selected, pilot=True,
                           calibration=calibration.to_dict()["t_stop"], baseline_replays=baselines)
            write_yaml(directory / "preparation.yaml", summary)
            summaries.append(summary)
            print("GATES %s %s" % (index, gated.summary), flush=True)
            write_yaml(root / "preparation.yaml", {"pilot": True, "tasks": summaries})
    finally:
        host.close()


def evaluate(root, model, checkpoint, port, client_src, task_indices=None):
    root = Path(root)
    cfgdir = root / "config"
    config = yaml.safe_load((cfgdir / "pilot.yaml").read_text())
    settings = RunSettings.load(cfgdir)
    host = LiberoHost(config_dir=cfgdir)
    session = EpisodeSession(host, RuntimeConfig.load(cfgdir))
    from hazardlib.config import default_config_dir, load_config
    model_yaml = default_config_dir().parent / "models" / (model + ".yaml")
    policy = DyniPolicyBridge(host, model_yaml, checkpoint, "127.0.0.1", port, client_src)
    directories = []
    try:
        for index in config["task_indices"] if task_indices is None else task_indices:
            directory = root / ("task_%02d" % index)
            inputs = load_pilot(directory / "instances.jsonl", [directory / "context.yaml"], host.describe(), None)
            from .seeded import PROTOCOL
            inputs.sources.update(policy_checkpoint=checkpoint, policy_rng_protocol=PROTOCOL,
                                  adapter_version=host.config["adapter_version"])
            if not inputs.instances:
                print("NO GATED INSTANCES task %s" % index, flush=True)
                continue
            out = directory / model
            video = config["video"]
            sink = Mp4FrameSink(out / "frames", video["fps"], video["crf"], video["preset"])
            result = run_stage(session, inputs, policy, policy.name, settings, out, [config["seed"]], 0,
                exposure_radius_m=load_config("semantics", config_dir=cfgdir)["exposure"]["low_exposure_radius_m"],
                labels=None, frame_sink=sink, frame_keys=["primary", "wrist"], monitor_topple=True,
                allow_censored_t_stop=False)
            directories.append(out)
            print("RUN %s %s" % (index, result), flush=True)
        if directories:
            directories = [root / ("task_%02d" % i) / model for i in config["task_indices"]
                           if (root / ("task_%02d" % i) / model / "episodes.jsonl").is_file()]
            metas, records = load_runs(directories)
            summary = summarize(metas, records, settings.report, settings.config_hash)
            report_dir = root / ("report_" + model)
            report_dir.mkdir(exist_ok=True)
            write_report(report_dir, summary)
            compact_report(root, model, config)
    finally:
        policy.close()
        host.close()


def compact_report(root, model, config):
    """Task-level counts alongside the core's certificate-separated tables.

    Raw hazard counts are explicit, so excluded low-exposure/invalid episodes
    cannot silently inflate the apparent task success or safety rate.
    """
    rows = []
    for index in config["task_indices"]:
        directory = root / ("task_%02d" % index)
        path = directory / model / "episodes.jsonl"
        if not path.is_file():
            continue
        records = read_episode_records(path, repair=False)
        controls = [r for r in records if r["kind"] == "control"]
        hazards = [r for r in records if r["kind"] == "hazard"]
        rows.append({"task_index": index, "task_id": records[0]["task_id"],
            "controls": len(controls), "control_success": sum(r["success_t_s"] is not None for r in controls),
            "hazards": len(hazards), "hazard_success": sum(r["success_t_s"] is not None for r in hazards),
            "hazard_contacts": sum(r["log"]["first_official_contact_t_s"] is not None for r in hazards),
            "certified": sum(r["label"] in ("wait_solvable", "evade_solvable") for r in hazards),
            "uncertified": sum(r["label"] == "uncertified" for r in hazards),
            "exposure_low": sum(r["log"]["exposure_low"] is True for r in hazards),
            "invalid": sum(r["log"]["invalid_class"] not in (None, "none") for r in hazards)})
    complete = len(rows) == len(config["task_indices"])
    output = root / ("report_" + model)
    (output / "tasks.json").write_text(json.dumps({"pilot": True, "complete_task_coverage": complete,
                                                  "policy": model, "tasks": rows}, indent=2))
    lines = ["# Original LIBERO spatial pilot (%s)" % model, "",
             "PILOT ONLY. One demonstration layout and seed per task. Raw episode counts below;",
             "certificate-separated eligible denominators and exact intervals are in report.md.", "",
             "| Task | Control success | Hazard success | Hazard contact | Certified / uncertified | Low exposure | Invalid |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| %d | %d/%d | %d/%d | %d/%d | %d / %d | %d | %d |" % (
            row["task_index"], row["control_success"], row["controls"], row["hazard_success"], row["hazards"],
            row["hazard_contacts"], row["hazards"], row["certified"], row["uncertified"],
            row["exposure_low"], row["invalid"]))
    lines += ["", "Coverage: %s." % ("all four configured tasks" if complete else "partial"),
              "Uncertified instances have no solvability claim. Slide construction rejections are kept in instances.jsonl.",
              "Fall visible lead times can be zero. This small pilot does not support a formal model ranking."]
    (output / "tasks.md").write_text("\n".join(lines) + "\n")
