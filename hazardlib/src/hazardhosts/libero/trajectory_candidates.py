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
import yaml

from hazardlib.calibrate.protocol import (CensoredTStopError, TStopProtocolSettings, read_calibration_summary,
                                          run_t_stop_protocol, sigma_from_summary, write_calibration)
from hazardlib.driving import EpisodeSession
from hazardlib.gates.episodes import run_g1, run_nominal_hit
from hazardlib.gates.g0 import run_g0
from hazardlib.gates.settings import GateSettings
from hazardlib.generate import (GenerateSettings, TrajectorySet, cell_id, cell_statistics,
                                generation_table, layout_probes, propose, select_admitted, target_key)
from hazardlib.generate.propose import CoverageIndex
from hazardlib.generate.stats import distribution
from hazardlib.generate.select import ledger_statistics, rejection_keys, t_stop_statistics  # noqa: F401  (re-exported)
from hazardlib.instantiate import InstantiateSettings, instantiate, reach_geometry, slide_support_rule
from hazardlib.instantiate.builder import InstanceRejected
from hazardlib.kinematics.path import SampledPath
from hazardlib.pipeline_io import write_yaml
from hazardlib.runtime import RuntimeConfig
from hazardlib.spec import HazardSpec, TaskContext
from hazardlib.spec.hazard_spec import AgentSpec
from hazardlib.spec.certificate import Certificate

from .geometry import quat_to_mat
from .host import LiberoHost
from .pilot import assemble_config
from .policies import DemoReplay
from .scene_audit import rejection_reason as audit_reason, run_audit
from .scene_clearance import (drop_samples, host_probes, playback_path, rejection_reason as clearance_reason,
                              support_height, sweep, sweep_samples)
from .trajectory_analysis import save_json
from .trajectory_export import rebuild_states


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
        "task_id": scene["task_id"], "layout_id": scene["suite"] + "/" + row["demo"],
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


def scene_extent(host, scene):
    """Collision-bounds size of the scene's target entity: the one collect measured for the entity (a free object,
    or the moving part / site of an articulated fixture); the host's groups only when collect has none."""
    size = scene.get("target_collision_bounds_size_m")
    if size is not None:
        return np.asarray(size, dtype=float)
    lo, hi = [], []
    for g in range(host.m.ngeom):
        if host.label_of_body[int(host.m.geom_bodyid[g])] != scene["target"]:
            continue
        if not (host.m.geom_contype[g] or host.m.geom_conaffinity[g]):
            continue
        bounds = host.m.geom_aabb[g]
        rotation = host.d.geom_xmat[g].reshape(3, 3)
        c = host.d.geom_xpos[g] + rotation.dot(bounds[:3])
        r = np.abs(rotation).dot(bounds[3:])
        lo.append(c - r)
        hi.append(c + r)
    if not lo:
        raise ValueError("no collision bounds for the target %r: collect again" % scene["target"])
    return np.max(hi, axis=0) - np.min(lo, axis=0)


STEP_TOLERANCE = 1e-6   # fraction of a control step: float round-off of Sigma * T_stop / dt (0.45 / 0.05)


def tau_for_sigma(sigma, t_stop_s, dt, minimum_steps):
    """``(tau_s, None)`` for a Sigma target: ``Sigma * T_stop`` rounded DOWN to whole control steps (the
    achieved Sigma is therefore <= the target), or ``(None, "below_minimum_lead")`` when that is fewer than
    ``minimum_steps`` steps (the hazard could not be seen for the required number of control steps)."""
    steps = int(math.floor(sigma * t_stop_s / dt + STEP_TOLERANCE))
    if steps < minimum_steps:
        return None, "below_minimum_lead"
    return steps * dt, None


def finger_low_points(scene, geometry, arrays, indices):
    """The lowest real finger/gripper surface sample at each control index (world frame)."""
    names = [(i, n) for i, n in enumerate(scene["robot_body_names"]) if n.startswith("gripper")]
    out = []
    for index in indices:
        cloud = np.concatenate([np.asarray(geometry[n]).dot(quat_to_mat(arrays["body_quat"][index, i]).T) +
                                arrays["body_pos"][index, i] for i, n in names])
        out.append(cloud[int(np.argmin(cloud[:, 2]))])
    return np.asarray(out).reshape(-1, 3)


def proposals_legacy(scene, summary, row, arrays, geometry, settings, workspace, t_stop_s, slide_support=None):
    """The L2 proposal function, kept only as the reference of the regression against ``hazardlib.generate``
    (``_logs/l4a/regression.py``); validate uses ``hazardlib.generate.propose``. Reads ``direction_count``.

    Per family and Sigma target: ``(family, target, candidates, info)``; one Sigma target = one difficulty bin.

    Anchors are quantiles of low-clearance observations per phase (speeds from ALL 50 demos); directions rotate
    with the initial target. ``tau`` of a target is ``Sigma * T_stop`` rounded down to the control step
    (:func:`tau_for_sigma`); it is NOT clamped: an anchor whose available time before the hit (since the phase
    start) or free-fall height budget is shorter than ``tau`` offers nothing for that target, and a target no
    anchor can serve is ``info["unreachable"]`` (reason = the most frequent drop reason; every count in
    ``info["dropped"]``). ``t_stop_s=None`` (censored / failed protocol, D6/D8) makes every target unreachable.
    A slide anchors the physical finger surface, not a hand-centre proxy, and only at instants where
    ``slide_support`` (:func:`slide_support_rule`, ``(point, t_s) -> bool``) holds, i.e. where the builder can find
    a support surface for the box.
    ``settings["follow_through"]`` maps a family to its D19 exit mode (``pass`` / ``retract``); a family not
    listed gets none (a fall). ``info`` also has ``tau_s`` and ``n_proposed`` (before the candidate budget).
    """
    dt = scene["control_dt_s"]
    min_steps = settings["minimum_lead_control_steps"]
    target_index = scene["target_index"]
    rotation = quat_to_mat(arrays["object_quat"][0, target_index])
    # Object pose can have a small roll/pitch. Surface motion uses its yaw
    # projected into the horizontal plane, not the tilted full body frame.
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    direction_base = [np.array([math.cos(yaw + 2 * math.pi * i / settings["direction_count"]),
                               math.sin(yaw + 2 * math.pi * i / settings["direction_count"]), 0.])
                      for i in range(settings["direction_count"])]
    for family in settings["families"]:
        if family == "intrusion.slide" and slide_support is None:
            raise ValueError("a slide proposal needs slide_support (slide_support_rule of the TaskContext)")
        anchors_found = []      # (phase, anchor index, finger offset, speed levels, speed)
        for phase in row["replay_features"]["phases"]:
            if phase["name"] not in summary["phases"]:
                continue
            start, end = phase["start_index"], phase["end_index"]
            indices = np.arange(start + min_steps, end)
            if not len(indices):
                continue
            lows = None
            if family == "intrusion.slide":
                # Only instants where the builder finds a support surface under the finger (the hit time of a
                # slide is where the finger is low enough above it for a box riding on the table to touch it).
                lows = finger_low_points(scene, geometry, arrays, indices)
                supported = np.array([bool(slide_support(point, arrays["time_s"][i])) for point, i in zip(lows, indices)])
                indices, lows = indices[supported], lows[supported]
                if not len(indices):
                    continue
                risk = arrays["gripper_low_z"][indices] - arrays["surface_z"]
            else:
                risk = arrays["neighbor_clearance"][indices]
            cutoff = np.quantile(risk, .5)
            keep = risk <= cutoff
            indices = indices[keep]
            lows = None if lows is None else lows[keep]
            positions = sorted(set(int(math.floor(q * (len(indices)-1))) for q in settings["anchor_quantiles"]))
            levels = summary["phases"][phase["name"]]["candidate_speed_levels_mps"]
            speed = float(np.median(levels))
            for position in positions:
                anchor = int(indices[position])
                offset = np.zeros(3)
                if lows is not None:
                    # Real finger sample at the selected instant; its z and x/y
                    # differ from the EEF frame origin, often substantially.
                    offset = lows[position] - arrays["eef_pos"][anchor]
                anchors_found.append((phase, anchor, offset, levels, speed))
        directions = [None] if family == "intrusion.fall" else direction_base
        for sigma in settings["sigma_targets"]:
            info = {"sigma_target": sigma, "t_stop_s": t_stop_s, "tau_s": None, "unreachable": None,
                    "dropped": {}, "n_proposed": 0}
            dropped = info["dropped"]

            def drop(why, n=1):
                dropped[why] = dropped.get(why, 0) + n
            if t_stop_s is None:
                info["unreachable"] = "t_stop_censored"
                yield family, sigma, [], info
                continue
            tau, why = tau_for_sigma(sigma, t_stop_s, dt, min_steps)
            info["tau_s"] = tau
            if tau is None:
                drop(why, max(len(anchors_found), 1))
            elif not anchors_found:
                drop("no_supported_anchor" if family == "intrusion.slide" else "no_anchor")
            offered = []
            for phase, anchor, offset, levels, speed in (anchors_found if tau is not None else []):
                t_hit = float(arrays["time_s"][anchor])
                if tau > t_hit - phase["t_start_s"] + STEP_TOLERANCE * dt:
                    drop("phase_time")
                    continue
                if family == "intrusion.fall":
                    # Conservative release budget above the entire observed gripper.
                    free_height = max(0., workspace["max_m"][2] - arrays["eef_pos"][anchor, 2] -
                                      np.max(scene["agent"]["half_extents_m"]))
                    if tau > math.sqrt(2 * free_height / scene["gravity_mps2"]) + STEP_TOLERANCE * dt:
                        drop("free_fall_height")
                        continue
                if speed <= 0:
                    drop("no_speed")
                    continue
                key = target_key(sigma)
                for di, direction in enumerate(directions):
                    ident = "%s.%s.%s.k%d.sig%s.d%d" % (row["demo"], family, phase["name"], anchor, key, di)
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
                        "speed_levels_mps": levels, "sigma_target": sigma, "t_stop_s": t_stop_s,
                        "tau_s": tau, "available_before_hit_s": t_hit - phase["t_start_s"],
                        "risk_clearance_m": float(arrays["neighbor_clearance"][anchor]),
                        "finger_table_gap_m": float(arrays["gripper_low_z"][anchor]-arrays["surface_z"]),
                        "direction_frame": "initial target object", "direction_index": di}
                    offered.append((spec, evidence))
            info["n_proposed"] = len(offered)
            if not offered:
                info["unreachable"] = max(sorted(dropped), key=lambda k: dropped[k]) if dropped else "no_anchor"
            # Round-robin over direction x phase, then expand anchors. A finite budget must not be
            # exhausted by the first direction alone.
            buckets = {}
            for proposal in offered:
                evidence = proposal[1]
                buckets.setdefault((evidence["direction_index"], evidence["phase"]), []).append(proposal)
            for bucket in buckets.values():
                bucket.sort(key=lambda p: (p[1]["risk_clearance_m"], p[1]["anchor_step"]))
            ordered = []
            depth = 0
            while any(depth < len(b) for b in buckets.values()):
                for key in sorted(buckets):
                    if depth < len(buckets[key]):
                        ordered.append(buckets[key][depth])
                depth += 1
            yield family, sigma, ordered[:settings["max_candidates_per_family_layout"]], info


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


def candidate_samples(instance, t_end_s, settings, surfaces, gravity_mps2, control_dt_s):
    """Sample poses ``[(t, pos, quat)]`` of the hazard's own trajectory for the scene-clearance sweep (a kinematic
    path, or a fall's release pose dropped onto the support): pure geometry, no host."""
    spacing = settings["scene_clearance_spacing_half_extent_fraction"] * min(instance.agent.half_extents_m)
    if instance.family == "intrusion.fall":
        release = instance.keyframes.pos[0]
        rest_z = support_height(surfaces, release[:2]) + instance.agent.half_extents_m[2]
        return drop_samples(release, rest_z, gravity_mps2, instance.keyframes.time_s[0], spacing, t_after=t_end_s)
    path = playback_path(instance)
    return sweep_samples(path, instance.onset_time_s, path.t_end_s, control_dt_s, spacing)


def sweep_candidate(host, instance, states, samples, settings, supports):
    """Exact scene-clearance sweep of one instance (module ``scene_clearance``, MuJoCo contacts): the authority."""
    handle = host.alloc_hazard(0, instance.agent)
    try:
        probes = host_probes(host, handle, states, instance.onset_time_s, supports)
        result = sweep(samples, probes, settings["scene_clearance_tol_m"])
        result["n_samples"] = len(samples)
        return result
    finally:
        host.free_hazard(handle)


def layout_prefilter(instance, samples, layout, held, settings):
    """Geometric prefilter of the same sweep from the V2 layout (``hazardlib.generate.layout_probes``): the
    support is ignored, an overlap counts when deeper than ``scene_clearance_tol_m + layout_prefilter_slack_m``
    (the layout boxes are bounding boxes)."""
    probes = layout_probes(layout, instance.onset_time_s, ("support",), instance.agent.half_extents_m, held)
    result = sweep(samples, probes, settings["scene_clearance_tol_m"] + settings["layout_prefilter_slack_m"])
    result["n_samples"] = len(samples)
    return result


def reach_stroke_m(instance_settings):
    """Stroke length before the hit of a reach (speed, tau) -> m, from the builder's own mapping."""
    def length(speed, tau):
        return reach_geometry(speed, tau, instance_settings.reach_overshoot_fraction, "min_jerk").reach_distance_m
    return length


CONSEQUENCE_FIELDS = ("task_success", "first_scene_contact_t_s", "v_rel_normal_mps", "impulse_analytic_ns",
                      "hazard_mass_kg", "contact_duration_s", "constraint_state")


def consequence_record(log):
    """D21 diagnostic (never an admission rule) of the physical nominal-hit episode, whose policy is the
    non-reacting reference replay: whether the task still succeeded after the hit, and what the log knows of the
    hit (analytic impulse ``m (1 + e) v_n`` and normal speed of the first robot-side contact, hazard mass, the
    time the hazard first touched a task / scene body). ``engine_peak_force_n`` is not measured by the runtime."""
    record = {}
    for key in CONSEQUENCE_FIELDS:
        value = getattr(log, key, None)
        record[key] = getattr(value, "value", value)
    return record


def consequence_statistics(entries, quantiles):
    """D21 diagnostic of one cell over its layouts (``entries`` as in ``cell_statistics``): over the admitted
    instances, how often the non-reacting reference replay of the physical nominal-hit episode fails the task
    (``reference_task_failed`` / ``n_admitted``), and the distribution of the analytic impulse of the hit where
    the ledger has it (ledgers of runs before D21 only have ``nominal_hit.success``). No admission rule."""
    rows = []
    for _, ledger in entries:
        item = next((p for p in ledger if p.get("selected")), None)
        if item is None or "nominal_hit" not in item:
            continue
        c = item.get("consequence") or {}
        rows.append({"failed": not bool(item["nominal_hit"]["success"]),
                     "impulse_ns": c.get("impulse_analytic_ns"), "v_rel_normal_mps": c.get("v_rel_normal_mps"),
                     "scene_contact": c.get("first_scene_contact_t_s") is not None})
    failed = sum(r["failed"] for r in rows)
    return {"n_admitted": len(rows), "reference_task_failed": failed,
            "failure_rate": None if not rows else failed / len(rows),
            "n_scene_contact": sum(r["scene_contact"] for r in rows),
            "impulse_ns": distribution([r["impulse_ns"] for r in rows if r["impulse_ns"] is not None], quantiles),
            "v_rel_normal_mps": distribution([r["v_rel_normal_mps"] for r in rows
                                              if r["v_rel_normal_mps"] is not None], quantiles)}


def consequence_by_family(cells):
    """Roll-up of ``consequence_statistics`` over cells: ``{family: {n_admitted, reference_task_failed,
    failure_rate}}`` plus ``all``. ``cells``: ``{cell id: {sigma key: consequence statistics}}``."""
    out = {}
    for cid, targets in cells.items():
        for key in ("all", cid.partition("@")[0]):
            row = out.setdefault(key, {"n_admitted": 0, "reference_task_failed": 0})
            for c in targets.values():
                row["n_admitted"] += c["n_admitted"]
                row["reference_task_failed"] += c["reference_task_failed"]
    for row in out.values():
        row["failure_rate"] = None if not row["n_admitted"] else row["reference_task_failed"] / row["n_admitted"]
    return out


def consequence_table(cells, by_family):
    def rate(c):
        return "-" if not c["n_admitted"] else "%d/%d (%.0f%%)" % (
            c["reference_task_failed"], c["n_admitted"], 100.0 * c["failure_rate"])

    def impulse(c):
        q = c["impulse_ns"]
        return "-" if not q["n"] else "%.3f" % q["quantiles"]["0.5"]
    lines = ["| cell | Sigma | admitted | reference replay fails the task after the hit | median impulse (N s) |",
             "|---|---:|---:|---|---:|"]
    for cid, targets in cells.items():
        for key, c in targets.items():
            lines.append("| %s | %s | %d | %s | %s |" % (cid, key, c["n_admitted"], rate(c), impulse(c)))
    lines += ["", "| family | admitted | reference replay fails the task after the hit |", "|---|---:|---|"]
    lines += ["| %s | %d | %s |" % (name, c["n_admitted"], rate(c)) for name, c in sorted(by_family.items())]
    return "\n".join(lines) + "\n"


def certify(instance, g1_passed):
    label = "wait_solvable" if g1_passed else "uncertified"
    instance.certificate = Certificate.from_dict({"g0": True, "g1": g1_passed, "g2": None, "label": label,
        "notes": "all-50 scene-conditioned pilot, D19: admitted by validity (scene clearance, nominal hit, "
                 "robot-absent audit); G1 stop-wait-resume is a diagnostic only; G2 unavailable"})
    return label


CHOSEN_KEYS = ("spec", "evidence", "scene_clearance", "visibility_probe", "nominal_hit", "scene_audit", "g1")


def sigma_record(item, summary):
    """Per-instance Sigma bookkeeping: T_stop, target, achieved tau and Sigma (through the protected
    ``sigma_from_summary``), and (once the nominal hit ran) the visible lead."""
    evidence, tau = item["evidence"], item["spec"]["intensity"]["tau_s"]
    record = {"target": evidence["sigma_target"], "t_stop_s": evidence["t_stop_s"], "tau_s": tau,
              "sigma_achieved": sigma_from_summary(tau, summary, False)[0]}
    if "nominal_hit" in item:
        record["visible_lead_s"] = item["nominal_hit"]["tau_vis_s"]
    return record


def family_result(ledger, chosen_item, instance, info, summary):
    chosen = None
    if chosen_item is not None:
        chosen = {"instance": instance.to_dict(), **{k: chosen_item[k] for k in CHOSEN_KEYS},
                  "sigma": sigma_record(chosen_item, summary)}
        if "layout_prefilter" in chosen_item:
            chosen["layout_prefilter"] = chosen_item["layout_prefilter"]
    if info["unreachable"] is not None:
        status = "unreachable"
    else:
        status = "no_admitted_candidate" if chosen is None else chosen["instance"]["certificate"]["label"]
    return {"sigma_target": info["sigma_target"], "t_stop_s": info["t_stop_s"], "tau_target_s": info["tau_s"],
            "unreachable": info["unreachable"], "tau_dropped_anchors": info["dropped"],
            "n_proposed": info["n_proposed"], "n_candidates": len(ledger),
            "n_prefilter_clear": sum(p.get("layout_prefilter", {}).get("passed", False) for p in ledger),
            "n_scene_clear": sum(p.get("scene_clearance", {}).get("passed", False) for p in ledger),
            "n_visible": sum(p.get("visibility_probe", {}).get("passed", False) for p in ledger),
            "dynamic_attempts": sum("nominal_hit" in p for p in ledger), "selected": chosen, "status": status}


def t_stop_calibration(folder, session, reference, tc, settings_dir, seed):
    """T_stop of one layout by the core protocol (``run_t_stop_protocol``, settings of the intensity config),
    cached in ``calibration.yaml`` (``calibration_failed.yaml`` if the protocol cannot run). Returns
    ``(record, summary)``; ``summary`` is ``None`` unless every repetition stopped: a censored T_stop is a
    lower bound, so Sigma is never targeted from it (D6/D8, ``sigma_from_summary``)."""
    good, bad = folder / "calibration.yaml", folder / "calibration_failed.yaml"
    if bad.exists():
        return {"status": "protocol_failed", "error": yaml.safe_load(bad.read_text())["error"]}, None
    if good.exists():
        data = yaml.safe_load(good.read_text())
        if data["task_context_hash"] != tc.hash():
            raise ValueError("calibration.yaml of %s belongs to another TaskContext: use a separate output directory"
                             % folder)
        summary = read_calibration_summary(good)[0]
    else:
        try:
            result = run_t_stop_protocol(session, reference, tc, TStopProtocolSettings.load(settings_dir), seed, 0)
        except RuntimeError as exc:   # the EEF never reached the trigger speed: no T_stop exists
            write_yaml(bad, {"status": "protocol_failed", "error": str(exc)})
            return {"status": "protocol_failed", "error": str(exc)}, None
        write_calibration(good, result)
        summary = result.summary
    record = {"status": "ok", "file": "calibration.yaml", "n_total": summary.n_total, "n_valid": summary.n_valid,
              "n_censored": summary.n_censored, "median_s": summary.median_s, "iqr_s": summary.iqr_s}
    try:
        sigma_from_summary(1.0, summary, False)
    except (CensoredTStopError, ValueError):
        record["status"] = "censored" if summary.n_valid else "no_valid_repetition"
        return record, None
    return record, summary


def validate(root, settings, demos=None, max_demos=None):
    """Proposals, checks and selection per usable demonstration and cell (family x entry-direction class x Sigma).

    ``max_demos``: only the first N usable demonstrations in coverage order (the V1 / V2 set of the coverage
    scoring is the same N). Needs ``export`` first (``<root>/generation_inputs``)."""
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
    if settings["max_validated_per_family_layout"] != len(settings["sigma_targets"]):
        raise ValueError("max_validated_per_family_layout (%r) must equal the number of sigma_targets (%d): one "
                         "admitted instance per cell (family, entry-direction class, Sigma target)"
                         % (settings["max_validated_per_family_layout"], len(settings["sigma_targets"])))
    generate = GenerateSettings.from_mapping(settings)
    inputs = root / "generation_inputs"
    if not (inputs / "index.json").exists():
        raise ValueError("%s has no generation_inputs: run the export stage first" % root)
    task_set = TrajectorySet.load(inputs, max_demos)
    trajectories = {t.demo: t for t in task_set.trajectories}
    write_yaml(root / "generation_protocol.yaml", settings)
    host = LiberoHost(config_dir=config)
    session = EpisodeSession(host, RuntimeConfig.load(config))
    reference = DemoReplay(host)
    gate_settings, instance_settings = GateSettings.load(config), InstantiateSettings.load(config)
    reach_start = reach_stroke_m(instance_settings)
    coverage_index = CoverageIndex(task_set, generate)
    # Timing bound is inherited from the shared C13 tolerance.
    from hazardlib.config import load_config
    tolerance = load_config("conformance", config_dir=config)["c13_nominal_hit"]["hit_time_abs_tol_s"]
    try:
        for row in coverage["demonstrations"]:
            if demos is not None and row["demo"] not in demos:
                continue
            if row["replay_success"] and row["demo"] not in trajectories:
                continue        # beyond --max-demos (or not exported)
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
            trajectory, layout = trajectories[row["demo"]], task_set.layouts[row["demo"]]
            # Physical states for geometry/visibility probes, checked against the previous independent baseline
            # before using them.
            arrays = dict(np.load(folder / "replay.npz"))
            states = rebuild_states(host, scene, row, folder, arrays, settings)
            if host.pool[0].agent.to_dict() != scene["agent"]:      # the pool exists once an episode was reset
                raise ValueError("scene.json agent %s is not the host pool agent %s of %s: the root belongs to "
                                 "another hazard scale (hazard_scale.prepare_root)"
                                 % (scene["agent"], host.pool[0].agent.to_dict(), config))
            host.install_hooks(session.runtime)
            supports = host.support_labels()      # primary supports if the scene has one, else the fallback (floor)
            extent = scene_extent(host, scene)
            margin = float(max(extent[:2])) * settings["workspace_object_margin_multiple"]
            workspace = {"min_m": (np.asarray(summary["measured_workspace_m"]["min"]) - margin).tolist(),
                         "max_m": (np.asarray(summary["measured_workspace_m"]["max"]) + margin).tolist()}
            tc = context(scene, row, arrays, geometry_of(root), workspace, identity)
            tc.hash()
            t_stop, summary_t = t_stop_calibration(folder, session, reference, tc, config, settings["seed"])
            print("T_STOP %s %s" % (row["demo"], json.dumps(t_stop)), flush=True)
            support = slide_support_rule(tc, host.describe(), AgentSpec.from_dict(scene["agent"]))
            results = {}
            records = []
            for cell in propose(trajectory, task_set, generate, workspace,
                                None if summary_t is None else summary_t.median_s, scene["agent"],
                                scene["gravity_mps2"], slide_support=support, reach_start_m=reach_start,
                                coverage=coverage_index):
                family, sigma, candidates, info = cell.family, cell.sigma, cell.candidates, cell.info
                cid = info["cell_id"]
                cells = results.setdefault(cid, {})
                stem = "%s.sigma%s" % (cid, target_key(sigma))
                family_path = folder / (stem + "_result.json")
                ledger_path = folder / (stem + "_candidates.json")
                if family_path.exists():
                    cells[target_key(sigma)] = json.loads(family_path.read_text())
                    if cells[target_key(sigma)]["selected"]:
                        records.append(cells[target_key(sigma)]["selected"]["instance"])
                    continue
                # Recover an interrupted run's completed cell without repeating physical checks. The ledger is
                # written once the cell is complete (G1 diagnostic included); the selected candidate is the item
                # flagged ``selected``. The complete per-candidate evidence is retained.
                if ledger_path.exists():
                    prior = json.loads(ledger_path.read_text())
                    picked = next((p for p in prior if p.get("selected")), None)
                    instance = None
                    if picked is not None:
                        instance = instantiate(HazardSpec.from_dict(picked["spec"]), tc, settings["seed"],
                                               host.describe(), instance_settings)
                        certify(instance, picked["g1"]["passed"])
                    result = family_result(prior, picked, instance, info, summary_t)
                    cells[target_key(sigma)] = result
                    if picked is not None:
                        records.append(result["selected"]["instance"])
                    save_json(family_path, result)
                    continue
                ledger = []
                shortlisted = []
                for spec, evidence in candidates:
                    print("PROPOSAL " + spec["id"], flush=True)
                    item = {"spec": spec, "evidence": evidence}
                    item["sigma"] = {k: evidence[k] for k in ("sigma_target", "t_stop_s", "tau_s")}
                    item["sigma"]["sigma_achieved"] = sigma_from_summary(spec["intensity"]["tau_s"], summary_t,
                                                                         False)[0]
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
                    samples = candidate_samples(instance, float(arrays["time_s"][-1]), settings,
                                                scene["support_surfaces"], abs(float(host.m.opt.gravity[2])),
                                                host.control_dt)
                    prefilter = None
                    if settings["layout_prefilter"] != "off":
                        prefilter = layout_prefilter(instance, samples, layout, trajectory.held, settings)
                        item["layout_prefilter"] = prefilter
                        if settings["layout_prefilter"] == "reject" and not prefilter["passed"]:
                            item["rejection"] = "layout_prefilter:" + ",".join(prefilter["partners"])
                            continue
                    clearance = sweep_candidate(host, instance, states, samples, settings, supports)
                    item["scene_clearance"] = clearance
                    if prefilter is not None:
                        prefilter["agrees_with_exact"] = prefilter["passed"] == clearance["passed"]
                    if not clearance["passed"]:
                        item["rejection"] = clearance_reason(clearance)
                        continue
                    probe = visibility_probe(host, instance, states, arrays["time_s"], settings)
                    item["visibility_probe"] = probe
                    if probe["passed"]:
                        shortlisted.append((probe["lead_s"], instance, item))
                    else:
                        item["rejection"] = "visibility:" + probe["reason"]
                # Trajectory-intersection first (D20 b), then the longest visible lead.
                shortlisted.sort(key=lambda x: ((-(x[2]["evidence"]["coverage"] or 0.0)
                                                 if settings["coverage_ordering"] else 0.0), -x[0]))

                def physical(instance, item):
                    print("G0 " + instance.spec_id, flush=True)
                    g0 = run_g0(instance, tc, host.describe(), gate_settings.g0)
                    item["g0"] = g0.to_dict()
                    if not g0.passed:
                        item["rejection"] = "g0:" + ",".join(g0.reasons)
                        return False, False
                    nominal = run_nominal_hit(session, instance, reference, tc, gate_settings, settings["seed"], 0)
                    info_ = session.runtime.episode_info(0)
                    nominal["release_kind"] = None if info_["release"] is None else info_["release"]["kind"]
                    nominal["exit_t_s"] = info_["exit_t_s"]
                    print("NOMINAL " + json.dumps(nominal), flush=True)
                    item["nominal_hit"] = nominal
                    item["consequence"] = consequence_record(info_["log"])
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
                save_json(ledger_path, ledger)
                result = family_result(ledger, None if entry is None else item, instance, info, summary_t)
                cells[target_key(sigma)] = result
                save_json(family_path, result)
                if result["selected"]:
                    records.append(result["selected"]["instance"])
                print("FAMILY " + json.dumps({"demo": row["demo"], "cell": cid, "sigma_target": sigma,
                                              **{k: v for k, v in result.items() if k != "selected"}}), flush=True)
            # The context references shared geometry in the development bundle;
            # not a formal frozen/standalone TaskContext.
            metadata = tc.to_dict()
            metadata["provenance"]["robot_geometry"] = {"external_npz": "../../robot_geometry.npz",
                                                        "spacing_m": scene["geometry_spacing_m"]}
            write_yaml(folder / "context_metadata.yaml", metadata)
            save_json(folder / "validation.json", {"demo": row["demo"], "status": "processed", "t_stop": t_stop,
                "sigma_targets": settings["sigma_targets"], "families": results,
                "generation_hash": generation_hash,
                "workspace": workspace, "target_collision_bounds_size_m": extent.tolist(),
                "geometry_source": "shared robot_geometry.npz; context hash includes full geometry"})
            (folder / "instances.jsonl").write_text("".join(json.dumps({"instance": i}, allow_nan=False) + "\n" for i in records))
    finally:
        host.close()


_GEOMETRY = {}


def geometry_of(root):
    """Shared robot surface samples of the task as lists (the TaskContext's own representation), cached."""
    key = str(root)
    if key not in _GEOMETRY:
        _GEOMETRY[key] = {k: v.tolist() for k, v in np.load(Path(root) / "robot_geometry.npz").items()}
    return _GEOMETRY[key]


def cell_ids(settings):
    """``[(cell id, family)]`` of every (family, entry-direction class) of the settings, in report order."""
    generate = GenerateSettings.from_mapping(settings)
    return [(cell_id(f, e), f) for f in generate.families for e in generate.elevations(f)]


def scene_agent(root):
    path = Path(root) / "scene.json"
    return json.loads(path.read_text())["agent"] if path.exists() else None


def report(root, settings):
    root = Path(root)
    coverage = json.loads((root / "coverage.json").read_text())
    rows = coverage["demonstrations"]
    validations = [json.loads((root / "trajectories" / r["demo"] / "validation.json").read_text())
                   for r in rows if (root / "trajectories" / r["demo"] / "validation.json").exists()]
    keys = [target_key(t) for t in settings["sigma_targets"]]
    cells = cell_ids(settings)
    processed = [v for v in validations if v["status"] == "processed"]
    results = {cid: {k: [] for k in keys} for cid, _ in cells}
    csvrows = []
    for row in rows:
        validation = next((v for v in validations if v["demo"] == row["demo"]), None)
        cr = {"demo": row["demo"], "source_success": row["source_success"], "replay_success": row["replay_success"],
              "replay_success_t_s": row["replay_success_t_s"], "source_duration_s": row["source_features"]["duration_s"],
              "t_stop_s": None if not validation or validation["status"] != "processed" else
              validation["t_stop"].get("median_s"), "t_stop_status": None if not validation or
              validation["status"] != "processed" else validation["t_stop"]["status"]}
        for cid, _ in cells:
            for key in keys:
                record = validation["families"].get(cid, {}).get(key) if validation else None
                cr["%s.sigma%s" % (cid, key)] = record["status"] if record else (
                    "baseline_unusable" if not row["replay_success"] else "pending")
                if record:
                    path = root / "trajectories" / row["demo"] / ("%s.sigma%s_candidates.json" % (cid, key))
                    results[cid][key].append((record, json.loads(path.read_text()) if path.exists() else []))
        csvrows.append(cr)
    quantiles = settings["quantiles"]
    families = {cid: {"targets": {k: cell_statistics(results[cid][k], quantiles) for k in keys}} for cid, _ in cells}
    consequence = {cid: {k: consequence_statistics(results[cid][k], quantiles) for k in keys} for cid, _ in cells}
    by_family = consequence_by_family(consequence)
    from .hazard_scale import config_level
    summary = {"n_processed": len(validations), "n_baseline_unusable": len(validations) - len(processed),
               "hazard_scale": config_level(root / "config") if (root / "config").exists() else None,
               "hazard_agent": scene_agent(root),
               "sigma_targets": settings["sigma_targets"], "consequence": consequence,
               "consequence_by_family": by_family,
               "t_stop": t_stop_statistics([v["t_stop"] for v in processed]), "families": families}
    write_yaml(root / "validated_summary.yaml", summary)
    with (root / "coverage.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csvrows[0]))
        writer.writeheader()
        writer.writerows(csvrows)
    compact = {f: {k: {n: v[n] for n in ("admitted", "unreachable_by_reason", "no_admitted_candidate", "g1_diagnostic")}
                   for k, v in d["targets"].items()} for f, d in families.items()}
    (root / "generation_report.md").write_text(
        "# Generation report\n\nDemonstrations processed: %d (baseline unusable: %d). One row per cell "
        "(family x entry-direction class x Sigma target); `admitted` counts demonstrations (layouts).\n\n%s\n"
        "## Physical consequence of the nominal hit (D21, diagnostic only)\n\nHazard scale %s, agent %s. The "
        "reference replay does not react to the hazard; task failure is a physical outcome, not an admission rule.\n\n%s" % (
            len(processed), summary["n_baseline_unusable"], generation_table(families),
            summary["hazard_scale"], summary["hazard_agent"], consequence_table(consequence, by_family)))
    print("REPORT " + json.dumps({"processed": len(validations), "t_stop": summary["t_stop"], "families": compact}))
    from .trajectory_report import write_analysis_report
    write_analysis_report(root)
