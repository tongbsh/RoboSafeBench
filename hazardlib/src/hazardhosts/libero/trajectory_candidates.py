"""Scene-conditioned proposals and real-physics checks of the 50-demo analysis.

The development protocol is separate from formal gate configuration. Visibility
and nominal timing are explicit admission criteria here, not retroactive changes
to the shared gate semantics. No model under test participates in selection.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

from hazardlib.driving import EpisodeSession
from hazardlib.gates.episodes import run_g1, run_nominal_hit
from hazardlib.gates.g0 import run_g0
from hazardlib.gates.settings import GateSettings
from hazardlib.instantiate import InstantiateSettings, instantiate
from hazardlib.instantiate.builder import InstanceRejected
from hazardlib.kinematics.path import SampledPath
from hazardlib.pipeline_io import write_yaml
from hazardlib.runtime import RuntimeConfig
from hazardlib.spec import HazardSpec, TaskContext
from hazardlib.spec.certificate import Certificate

from .geometry import quat_to_mat
from .host import LiberoHost
from .pilot import assemble_config
from .policies import DemoReplay
from .scene_audit import rejection_reason as audit_reason, run_audit
from .scene_clearance import (drop_samples, host_probes, playback_path, rejection_reason as clearance_reason,
                              support_height, sweep, sweep_samples)
from .trajectory_analysis import save_json
from .trajectory_stats import distribution


class AnalysisContext(TaskContext):
    """An immutable-by-convention context: avoid re-hashing the shared dense mesh per proposal."""
    def hash(self):
        if not hasattr(self, "_analysis_hash"):
            self._analysis_hash = super().hash()
        return self._analysis_hash


def context(scene, row, arrays, geometry, workspace, identity):
    times = arrays["time_s"].tolist()
    bodies = scene["robot_body_names"]
    phase_list = [{k: p[k] for k in ("name", "t_start_s", "t_end_s")}
                  for p in row["replay_features"]["phases"]]
    return AnalysisContext.from_dict({
        "task_id": scene["task_id"], "layout_id": "libero_spatial/" + row["demo"],
        "entities": {"interest_%d" % i: {"name": name, "source": "bddl_goal"}
                     for i, name in enumerate(scene["task_entities"])},
        "phases": phase_list,
        "nominal_ee_path": {"time_s": times, "pos": arrays["eef_pos"].tolist(),
                            "quat": arrays["eef_quat"].tolist()},
        "occupancy": {"points": arrays["eef_pos"].tolist(), "time_s": times},
        "workspace_bounds": workspace, "support_surfaces": scene["support_surfaces"],
        "provenance": {"builder": "libero.trajectory_analysis; individual physical replay, no path averaging",
            "analysis_identity": identity, "source_demo": row["demo"],
            "robot_geometry": {"spacing_m": scene["geometry_spacing_m"], "points": geometry},
            "robot_body_paths": {n: arrays["body_pos"][:, i].tolist() for i, n in enumerate(bodies)},
            "robot_body_quat_paths": {n: arrays["body_quat"][:, i].tolist() for i, n in enumerate(bodies)},
            "probe_set": {"source_kinds_used": ["demo_replay"], "reference_policies": [],
                          "sources": [{"kind": "demo_replay", "name": "libero_official_demos", "under_test": False}]},
            "eef_speed_median_mps": row["replay_features"]["speed_mps"]["quantiles"]["0.5"]}})


def scene_extent(host, target):
    lo, hi = [], []
    for g in range(host.m.ngeom):
        if host.label_of_body[int(host.m.geom_bodyid[g])] != target:
            continue
        if not (host.m.geom_contype[g] or host.m.geom_conaffinity[g]):
            continue
        bounds = host.m.geom_aabb[g]
        rotation = host.d.geom_xmat[g].reshape(3, 3)
        c = host.d.geom_xpos[g] + rotation.dot(bounds[:3])
        r = np.abs(rotation).dot(bounds[3:])
        lo.append(c - r)
        hi.append(c + r)
    return np.max(hi, axis=0) - np.min(lo, axis=0)


def proposals(scene, summary, row, arrays, geometry, settings, workspace):
    """Quantile anchors of low-clearance observations; speeds from ALL 50 demos.

    Directions rotate with the initial target. Tau is a fraction of that
    phase's observed duration, bounded by available time and geometric space.
    A slide anchors the physical finger surface, not a hand-centre proxy. ``settings["follow_through"]``
    maps a family to its D19 exit mode (``pass`` / ``retract``); a family not listed gets none (a fall).
    """
    dt = scene["control_dt_s"]
    min_tau = settings["minimum_lead_control_steps"] * dt
    target_index = scene["target_index"]
    rotation = quat_to_mat(arrays["object_quat"][0, target_index])
    # Object pose can have a small roll/pitch. Surface motion uses its yaw
    # projected into the horizontal plane, not the tilted full body frame.
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    direction_base = [np.array([math.cos(yaw + 2 * math.pi * i / settings["direction_count"]),
                               math.sin(yaw + 2 * math.pi * i / settings["direction_count"]), 0.])
                      for i in range(settings["direction_count"])]
    for family in settings["families"]:
        offered = []
        for phase in row["replay_features"]["phases"]:
            if phase["name"] not in summary["phases"]:
                continue
            start, end = phase["start_index"], phase["end_index"]
            indices = np.arange(start + settings["minimum_lead_control_steps"], end)
            if not len(indices):
                continue
            if family == "intrusion.slide":
                risk = arrays["gripper_low_z"][indices] - arrays["surface_z"]
            else:
                risk = arrays["neighbor_clearance"][indices]
            cutoff = np.quantile(risk, .5)
            indices = indices[risk <= cutoff]
            anchors = sorted(set(int(indices[int(math.floor(q * (len(indices)-1)))])
                                 for q in settings["anchor_quantiles"]))
            levels = summary["phases"][phase["name"]]["candidate_speed_levels_mps"]
            speed = float(np.median(levels))
            for anchor in anchors:
                t_hit = float(arrays["time_s"][anchor])
                offset = np.zeros(3)
                if family == "intrusion.slide":
                    # Real finger sample at the selected instant; its z and x/y
                    # differ from the EEF frame origin, often substantially.
                    points = []
                    for i, name in enumerate(scene["robot_body_names"]):
                        if name.startswith("gripper"):
                            pts = np.asarray(geometry[name])
                            points.append(pts.dot(quat_to_mat(arrays["body_quat"][anchor, i]).T) +
                                          arrays["body_pos"][anchor, i])
                    cloud = np.concatenate(points)
                    low = cloud[int(np.argmin(cloud[:, 2]))]
                    offset = low - arrays["eef_pos"][anchor]
                directions = [None] if family == "intrusion.fall" else direction_base
                for fraction in settings["tau_phase_fractions"]:
                    available = t_hit - phase["t_start_s"]
                    tau = min(phase["duration_s"] * fraction, available)
                    if family == "intrusion.fall":
                        # Conservative release budget above the entire observed gripper.
                        free_height = max(0., workspace["max_m"][2] - arrays["eef_pos"][anchor, 2] -
                                          np.max(scene["agent"]["half_extents_m"]))
                        tau = min(tau, math.sqrt(2 * free_height / scene["gravity_mps2"]))
                    tau = math.floor((tau + np.finfo(float).eps) / dt) * dt
                    if tau < min_tau or speed <= 0:
                        continue
                    for di, direction in enumerate(directions):
                        ident = "%s.%s.%s.k%d.f%s.d%d" % (row["demo"], family, phase["name"], anchor, fraction, di)
                        params = {} if direction is None else {"direction": direction.tolist()}
                        intensity = {"tau_s": tau}
                        if family != "intrusion.fall":
                            intensity["approach_speed_mps"] = speed
                        if family == "intrusion.reach":
                            params["speed_profile"] = "min_jerk"
                        if family in settings["follow_through"]:
                            params["follow_through"] = settings["follow_through"][family]
                        spec = {"spec_version": "1", "id": ident, "family": family, "agent": scene["agent"],
                            "motion": {"primitive": {"intrusion.fall": "free_fall", "intrusion.slide": "slide",
                                                       "intrusion.reach": "reach"}[family], "params": params},
                            "exposure": {"anchor": "nominal_ee_path", "offset_m": offset.tolist(),
                                "anchor_phase": {"phase": phase["name"], "frac":
                                    (t_hit-phase["t_start_s"]) / phase["duration_s"]}},
                            "timing": {"mode": "nominal_time"}, "intensity": intensity,
                            "release": {"mode": "dynamic_release"}}
                        evidence = {"phase": phase["name"], "anchor_step": anchor,
                            "source": "all-50 phase speed quantiles + individual low-clearance replay samples",
                            "speed_levels_mps": levels, "tau_phase_fraction": fraction,
                            "risk_clearance_m": float(arrays["neighbor_clearance"][anchor]),
                            "finger_table_gap_m": float(arrays["gripper_low_z"][anchor]-arrays["surface_z"]),
                            "direction_frame": "initial target object", "direction_index": di}
                        offered.append((spec, evidence))
        # Round-robin over direction x phase, then expand anchors/tau values.
        # A finite budget must not be exhausted by the first direction alone.
        buckets = {}
        for proposal in offered:
            evidence = proposal[1]
            buckets.setdefault((evidence["direction_index"], evidence["phase"]), []).append(proposal)
        for bucket in buckets.values():
            bucket.sort(key=lambda p: (-p[1]["tau_phase_fraction"], p[1]["risk_clearance_m"], p[1]["anchor_step"]))
        ordered = []
        depth = 0
        while any(depth < len(b) for b in buckets.values()):
            for key in sorted(buckets):
                if depth < len(buckets[key]):
                    ordered.append(buckets[key][depth])
            depth += 1
        yield family, ordered[:settings["max_candidates_per_family_layout"]]


def visibility_probe(host, instance, states, times, settings):
    """Exact geometry/segmentation from saved PHYSICAL replay states; no dynamics. Only ``[onset, t_hit)`` is
    probed, so the path is the instance's ``keyframes`` (an exit path starts at the release, ``t_hit``)."""
    handle = host.alloc_hazard(0, instance.agent)
    mem = handle.native
    path = SampledPath.from_pose_sequence(instance.keyframes)
    geom_ids = set(mem.geoms)
    visible = []
    blocked = []
    indices = np.flatnonzero((times >= instance.onset_time_s) & (times < instance.nominal_hit_time_s))
    try:
        for idx in indices:
            host.sim.set_state_from_flattened(states[idx])
            host.mj.mj_forward(host.m, host.d)
            host._fresh_at = -1
            host.set_hazard_visible(handle, True)
            host.set_hazard_pose(handle, path.pos(float(times[idx])), [1., 0., 0., 0.])
            host._collision(mem, True)
            host.mj.mj_forward(host.m, host.d)
            if idx == indices[0]:
                for c in host.d.contact[:host.d.ncon]:
                    if c.dist < 0 and (c.geom1 in geom_ids or c.geom2 in geom_ids):
                        other = c.geom2 if c.geom1 in geom_ids else c.geom1
                        blocked.append(host.label_of_body[int(host.m.geom_bodyid[other])])
                if blocked:
                    return {"passed": False, "reason": "onset_overlap", "partners": sorted(set(blocked))}
            counts = {}
            for key, camera in host._cameras.items():
                segmentation = np.asarray(host.sim.render(host._hw, host._hw, camera_name=camera, segmentation=True))
                mask = (segmentation[..., 0] == int(host.mj.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[..., 1], list(geom_ids))
                counts[key] = int(mask.sum())
            visible.append({"t_s": float(times[idx]), "pixels": counts})
            if len(visible) >= settings["minimum_visible_observations"]:
                for key in host._cameras:
                    start = next((j for j in range(len(visible)-settings["minimum_visible_observations"]+1)
                        if all(v["pixels"][key] >= settings["minimum_visible_pixels"]
                               for v in visible[j:j+settings["minimum_visible_observations"]])), None)
                    if start is not None:
                        lead = instance.nominal_hit_time_s - visible[start]["t_s"]
                        if lead + np.finfo(float).eps >= settings["minimum_lead_control_steps"] * host.control_dt:
                            return {"passed": True, "camera": key, "lead_s": lead, "samples": visible}
        return {"passed": False, "reason": "insufficient_visible_lead", "samples": visible}
    finally:
        host.free_hazard(handle)


def sweep_candidate(host, instance, states, times, settings, surfaces, supports):
    """Geometric scene-clearance sweep of one instance (module ``scene_clearance``): needs no dynamics."""
    handle = host.alloc_hazard(0, instance.agent)
    try:
        probes = host_probes(host, handle, states, instance.onset_time_s, supports)
        spacing = settings["scene_clearance_spacing_half_extent_fraction"] * min(instance.agent.half_extents_m)
        if instance.family == "intrusion.fall":
            release = instance.keyframes.pos[0]
            rest_z = support_height(surfaces, release[:2]) + instance.agent.half_extents_m[2]
            samples = drop_samples(release, rest_z, abs(float(host.m.opt.gravity[2])),
                                   instance.keyframes.time_s[0], spacing, t_after=float(times[-1]))
        else:
            path = playback_path(instance)
            samples = sweep_samples(path, instance.onset_time_s, path.t_end_s, host.control_dt, spacing)
        result = sweep(samples, probes, settings["scene_clearance_tol_m"])
        result["n_samples"] = len(samples)
        return result
    finally:
        host.free_hazard(handle)


def select_admitted(shortlisted, check, budget):
    """First candidate (in the given order) that ``check(instance, item)`` admits. ``check`` returns
    ``(attempted, admitted)``; only an attempted candidate (a physical episode ran) uses the ``budget``.
    Candidates never reached are marked ``not_evaluated``. Returns ``(chosen entry or None, attempts)``."""
    attempts, chosen = 0, None
    for position, entry in enumerate(shortlisted):
        _, instance, item = entry
        attempted, admitted = check(instance, item)
        attempts += bool(attempted)
        if admitted:
            chosen = entry
            reached = position + 1
            break
        if attempts >= budget:
            reached = position + 1
            break
    else:
        reached = len(shortlisted)
    for _, _, item in shortlisted[reached:]:
        item["not_evaluated"] = "selected_earlier" if chosen is not None else "attempt_budget"
    return chosen, attempts


def certify(instance, g1_passed):
    label = "wait_solvable" if g1_passed else "uncertified"
    instance.certificate = Certificate.from_dict({"g0": True, "g1": g1_passed, "g2": None, "label": label,
        "notes": "all-50 scene-conditioned pilot, D19: admitted by validity (scene clearance, nominal hit, "
                 "robot-absent audit); G1 stop-wait-resume is a diagnostic only; G2 unavailable"})
    return label


CHOSEN_KEYS = ("spec", "evidence", "scene_clearance", "visibility_probe", "nominal_hit", "scene_audit", "g1")


def family_result(ledger, chosen_item, instance):
    chosen = None
    if chosen_item is not None:
        chosen = {"instance": instance.to_dict(), **{k: chosen_item[k] for k in CHOSEN_KEYS}}
    return {"n_candidates": len(ledger),
            "n_scene_clear": sum(p.get("scene_clearance", {}).get("passed", False) for p in ledger),
            "n_visible": sum(p.get("visibility_probe", {}).get("passed", False) for p in ledger),
            "dynamic_attempts": sum("nominal_hit" in p for p in ledger), "selected": chosen,
            "status": "no_admitted_candidate" if chosen is None else chosen["instance"]["certificate"]["label"]}


def validate(root, settings, demos=None):
    root = Path(root)
    scene = json.loads((root / "scene.json").read_text())
    coverage = json.loads((root / "coverage.json").read_text())
    summary = coverage["summary"]
    identity = json.loads((root / "identity.json").read_text())
    config = root / "config"
    if not config.exists():
        assemble_config(config)
    from hazardlib.hashing import sha256_hex
    generation_hash = sha256_hex(settings)
    write_yaml(root / "generation_protocol.yaml", settings)
    host = LiberoHost(config_dir=config)
    session = EpisodeSession(host, RuntimeConfig.load(config))
    reference = DemoReplay(host)
    gate_settings, instance_settings = GateSettings.load(config), InstantiateSettings.load(config)
    geometry = {k: v.tolist() for k, v in np.load(root / "robot_geometry.npz").items()}
    supports = set(host.config["support_bodies"])
    # Timing bound is inherited from the shared C13 tolerance.
    from hazardlib.config import load_config
    tolerance = load_config("conformance", config_dir=config)["c13_nominal_hit"]["hit_time_abs_tol_s"]
    try:
        for row in coverage["demonstrations"]:
            if demos is not None and row["demo"] not in demos:
                continue
            folder = root / "trajectories" / row["demo"]
            if (folder / "validation.json").exists():
                prior = json.loads((folder / "validation.json").read_text())
                if prior.get("generation_hash") != generation_hash:
                    raise ValueError("generation protocol changed: use a separate validation output directory")
                print("VALIDATION CACHED " + row["demo"], flush=True)
                continue
            if not row["replay_success"]:
                save_json(folder / "validation.json", {"demo": row["demo"], "status": "baseline_unusable",
                                                       "generation_hash": generation_hash, "families": {}})
                continue
            # Capture physical states for geometry/visibility probes, checked
            # against the previous independent baseline before using them.
            host.install_hooks(None)
            host.reset_episode(0, scene["task_id"], settings["suite"] + "/" + row["demo"], settings["seed"])
            arrays = dict(np.load(folder / "replay.npz"))
            cache_path = folder / "physical_states.npz"
            if cache_path.exists():
                states = np.load(cache_path)["states"]
                if len(states) != len(arrays["time_s"]):
                    raise ValueError("cached replay length mismatch")
                for index in (0, len(states)//2, len(states)-1):
                    host.sim.set_state_from_flattened(states[index])
                    host.mj.mj_forward(host.m, host.d)
                    host._fresh_at = -1
                    if not np.allclose(host.robot_state(0).eef_pose.pos, arrays["eef_pos"][index], atol=1e-12, rtol=0):
                        raise ValueError("cached replay geometry mismatch")
                # mj_forward rebuilds contacts at the saved end state. The
                # native evaluator read mj_step's pre-integration contacts;
                # predicates on a touching boundary can differ. Preserve the
                # goal event recorded during the real baseline, never certify
                # a goal from state forcing.
                if not bool(arrays["native_success"][-1]):
                    raise ValueError("cached physical replay did not record native success")
            else:
                actions = host._demo(scene["task_id"], settings["suite"]).actions(row["demo"])
                states = [host.sim.get_state().flatten()]
                actual_path = [host.robot_state(0).eef_pose.pos]
                for action in actions:
                    host.step(0, action)
                    states.append(host.sim.get_state().flatten())
                    actual_path.append(host.robot_state(0).eef_pose.pos)
                    if host.task_success(0):
                        break
                if len(states) != len(arrays["time_s"]) or not np.allclose(
                        actual_path, arrays["eef_pos"], atol=1e-12, rtol=0):
                    raise RuntimeError("baseline replay drifted for " + row["demo"])
                if not host.task_success(0):
                    raise RuntimeError("baseline replay no longer successful: " + row["demo"])
                states = np.asarray(states)
                np.savez_compressed(cache_path, states=states)
            host.install_hooks(session.runtime)
            extent = scene_extent(host, scene["target"])
            margin = float(max(extent[:2])) * settings["workspace_object_margin_multiple"]
            workspace = {"min_m": (np.asarray(summary["measured_workspace_m"]["min"]) - margin).tolist(),
                         "max_m": (np.asarray(summary["measured_workspace_m"]["max"]) + margin).tolist()}
            tc = context(scene, row, arrays, geometry, workspace, identity)
            tc.hash()
            results = {}
            records = []
            for family, candidates in proposals(scene, summary, row, arrays, geometry, settings, workspace):
                family_path = folder / (family + "_result.json")
                ledger_path = folder / (family + "_candidates.json")
                if family_path.exists():
                    results[family] = json.loads(family_path.read_text())
                    if results[family]["selected"]:
                        records.append(results[family]["selected"]["instance"])
                    continue
                # Recover an interrupted run's completed family without repeating physical checks. The ledger
                # is written once the family is complete (G1 diagnostic included); the selected candidate is
                # the item flagged ``selected``. The complete per-candidate evidence is retained.
                if ledger_path.exists():
                    prior = json.loads(ledger_path.read_text())
                    picked = next((p for p in prior if p.get("selected")), None)
                    instance = None
                    if picked is not None:
                        instance = instantiate(HazardSpec.from_dict(picked["spec"]), tc, settings["seed"],
                                               host.describe(), instance_settings)
                        certify(instance, picked["g1"]["passed"])
                    results[family] = family_result(prior, picked, instance)
                    if picked is not None:
                        records.append(results[family]["selected"]["instance"])
                    save_json(family_path, results[family])
                    continue
                ledger = []
                shortlisted = []
                for spec, evidence in candidates:
                    print("PROPOSAL " + spec["id"], flush=True)
                    item = {"spec": spec, "evidence": evidence}
                    ledger.append(item)
                    outside = [axis for axis, value in spec["intensity"].items()
                               if not instance_settings.intensity_ranges[family][axis][0] <= value <=
                               instance_settings.intensity_ranges[family][axis][1]]
                    if outside:
                        item["rejection"] = "outside_declared_pilot_range: " + ",".join(outside)
                        continue
                    try:
                        instance = instantiate(HazardSpec.from_dict(spec), tc, settings["seed"], host.describe(), instance_settings)
                    except InstanceRejected as exc:
                        item["rejection"] = str(exc)
                        continue
                    clearance = sweep_candidate(host, instance, states, arrays["time_s"], settings,
                                                scene["support_surfaces"], supports)
                    item["scene_clearance"] = clearance
                    if not clearance["passed"]:
                        item["rejection"] = clearance_reason(clearance)
                        continue
                    probe = visibility_probe(host, instance, states, arrays["time_s"], settings)
                    item["visibility_probe"] = probe
                    if probe["passed"]:
                        shortlisted.append((probe["lead_s"], instance, item))
                    else:
                        item["rejection"] = "visibility:" + probe["reason"]
                shortlisted.sort(key=lambda x: -x[0])

                def physical(instance, item):
                    print("G0 " + instance.spec_id, flush=True)
                    g0 = run_g0(instance, tc, host.describe(), gate_settings.g0)
                    item["g0"] = g0.to_dict()
                    if not g0.passed:
                        item["rejection"] = "g0:" + ",".join(g0.reasons)
                        return False, False
                    nominal = run_nominal_hit(session, instance, reference, tc, gate_settings, settings["seed"], 0)
                    info = session.runtime.episode_info(0)
                    nominal["release_kind"] = None if info["release"] is None else info["release"]["kind"]
                    nominal["exit_t_s"] = info["exit_t_s"]
                    print("NOMINAL " + json.dumps(nominal), flush=True)
                    item["nominal_hit"] = nominal
                    failure = None
                    if not nominal["contact"]:
                        failure = "no_contact"
                    elif abs(nominal["hit_time_error_s"]) > tolerance:
                        failure = "hit_time"
                    elif nominal["tau_vis_s"] + 32*np.finfo(float).eps < \
                            settings["minimum_lead_control_steps"] * scene["control_dt_s"]:
                        failure = "tau_vis"
                    if failure:
                        item["rejection"] = "physical_nominal_hit_or_visibility:" + failure
                        return True, False
                    audit = run_audit(session, instance, reference, tc, gate_settings, settings["seed"], supports,
                                      settings["object_moved_tolerance_m"])
                    print("AUDIT " + json.dumps({k: audit[k] for k in ("reasons", "release_kind", "exited",
                                                                       "hold_complete", "success")}), flush=True)
                    item["scene_audit"] = audit
                    if not audit["admitted"]:
                        item["rejection"] = audit_reason(audit["reasons"])
                        return True, False
                    return True, True

                entry, dynamic_attempts = select_admitted(shortlisted, physical,
                                                          settings["max_dynamic_attempts_per_family_layout"])
                instance = None
                if entry is not None:
                    _, instance, item = entry
                    # Diagnostic only: a failed stop-wait-resume does not reject an admitted candidate.
                    print("G1 " + instance.spec_id, flush=True)
                    g1 = run_g1(session, instance, reference, tc, gate_settings, settings["seed"], 0)
                    item["g1"] = g1.to_dict()
                    certify(instance, g1.passed)
                    item["selected"] = True
                save_json(folder / (family + "_candidates.json"), ledger)
                results[family] = family_result(ledger, None if entry is None else item, instance)
                save_json(family_path, results[family])
                if results[family]["selected"]:
                    records.append(results[family]["selected"]["instance"])
                print("FAMILY " + json.dumps({"demo": row["demo"], "family": family,
                                              **{k: v for k, v in results[family].items() if k != "selected"}}), flush=True)
            # The context references shared geometry in the development bundle;
            # not a formal frozen/standalone TaskContext.
            metadata = tc.to_dict()
            metadata["provenance"]["robot_geometry"] = {"external_npz": "../../robot_geometry.npz",
                                                        "spacing_m": scene["geometry_spacing_m"]}
            write_yaml(folder / "context_metadata.yaml", metadata)
            save_json(folder / "validation.json", {"demo": row["demo"], "status": "processed", "families": results,
                "generation_hash": generation_hash,
                "workspace": workspace, "target_collision_bounds_size_m": extent.tolist(),
                "geometry_source": "shared robot_geometry.npz; context hash includes full geometry"})
            (folder / "instances.jsonl").write_text("".join(json.dumps({"instance": i}, allow_nan=False) + "\n" for i in records))
    finally:
        host.close()


def rejection_keys(reason):
    """``(category, [category:detail, ...])`` of a ledger rejection string; details are the partners or
    reasons after the first colon (separated by commas or semicolons)."""
    category, _, rest = reason.partition(":")
    details = [d.strip() for d in rest.replace(";", ",").split(",") if d.strip()]
    return category, ([category + ":" + d for d in details] if category in DETAILED else [])


DETAILED = ("scene_clearance", "scene_audit", "g0", "physical_nominal_hit_or_visibility", "visibility")


def ledger_statistics(ledgers):
    """Counts over the candidate ledgers of one family (one list of items per layout): rejection reasons by
    category and by detail, candidates not evaluated, the release kind of every physical nominal-hit episode,
    the audit outcomes and the same for the selected candidates (with the G1 diagnostic)."""
    def count(items):
        out = {}
        for value in items:
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], str(kv[0]))))
    category, detail, not_evaluated = [], [], []
    nominal_kinds, audits, selected = [], [], []
    for ledger in ledgers:
        for item in ledger:
            if "rejection" in item:
                c, d = rejection_keys(item["rejection"])
                category.append(c)
                detail.extend(d)
            if "not_evaluated" in item:
                not_evaluated.append(item["not_evaluated"])
            if "nominal_hit" in item:
                nominal_kinds.append(str(item["nominal_hit"].get("release_kind")))
            if "scene_audit" in item:
                audits.append(item["scene_audit"])
            if item.get("selected"):
                selected.append(item)
    def audit_counts(records):
        return {"n": len(records), "exited": sum(a["exited"] for a in records),
                "released": sum(a["released"] for a in records),
                "release_kinds": count([str(a["release_kind"]) for a in records]),
                "hold_incomplete": sum(not a["hold_complete"] for a in records),
                "task_success": sum(bool(a["success"]) for a in records),
                "admitted": sum(a["admitted"] for a in records)}
    return {"rejections_by_category": count(category), "rejections_by_detail": count(detail),
            "not_evaluated": count(not_evaluated),
            "nominal_release_kinds_all_attempts": count(nominal_kinds),
            "nominal_release_kinds_selected": count([str(i["nominal_hit"].get("release_kind")) for i in selected]),
            "audit_all_attempts": audit_counts(audits),
            "audit_selected": audit_counts([i["scene_audit"] for i in selected]),
            "g1_diagnostic_selected": {"n": len(selected),
                                       "passed": sum(bool(i["g1"]["passed"]) for i in selected)}}


def report(root, settings):
    root = Path(root)
    coverage = json.loads((root / "coverage.json").read_text())
    rows = coverage["demonstrations"]
    validations = [json.loads((root / "trajectories" / r["demo"] / "validation.json").read_text())
                   for r in rows if (root / "trajectories" / r["demo"] / "validation.json").exists()]
    results = {f: {"admitted": 0, "wait_solvable": 0, "uncertified": 0, "no_admitted_candidate": 0,
                   "parameters": {"tau_s": [], "speed_mps": [], "hit_time_s": [], "visible_lead_s": []}}
               for f in settings["families"]}
    ledgers = {f: [] for f in settings["families"]}
    for row in rows:
        for family in settings["families"]:
            path = root / "trajectories" / row["demo"] / (family + "_candidates.json")
            if path.exists():
                ledgers[family].append(json.loads(path.read_text()))
    csvrows = []
    for row in rows:
        validation = next((v for v in validations if v["demo"] == row["demo"]), None)
        cr = {"demo": row["demo"], "source_success": row["source_success"], "replay_success": row["replay_success"],
              "replay_success_t_s": row["replay_success_t_s"], "source_duration_s": row["source_features"]["duration_s"]}
        for family in settings["families"]:
            record = validation["families"].get(family) if validation else None
            status = record["status"] if record else ("baseline_unusable" if not row["replay_success"] else "pending")
            cr[family] = status
            if record:
                results[family][status] += 1
                chosen = record["selected"]
                if chosen:
                    results[family]["admitted"] += 1
                    results[family]["parameters"]["tau_s"].append(chosen["spec"]["intensity"]["tau_s"])
                    speed = chosen["spec"]["intensity"].get("approach_speed_mps")
                    if speed is not None:
                        results[family]["parameters"]["speed_mps"].append(speed)
                    results[family]["parameters"]["hit_time_s"].append(chosen["instance"]["nominal_hit_time_s"])
                    results[family]["parameters"]["visible_lead_s"].append(chosen["nominal_hit"]["tau_vis_s"])
        csvrows.append(cr)
    for family in results:
        results[family]["parameters"] = {k: distribution(v, settings["quantiles"])
                                          for k, v in results[family]["parameters"].items()}
        results[family]["layouts_with_ledger"] = len(ledgers[family])
        results[family].update(ledger_statistics(ledgers[family]))
    write_yaml(root / "validated_summary.yaml", {"n_processed": len(validations), "families": results})
    with (root / "coverage.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csvrows[0]))
        writer.writeheader()
        writer.writerows(csvrows)
    print("REPORT " + json.dumps({"processed": len(validations), "families": results}))
    from .trajectory_report import write_analysis_report
    write_analysis_report(root)
