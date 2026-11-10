"""All-demonstration, scene-conditioned analysis (development, not a benchmark release).

collect: forward recorded states for geometry, then independently replay actions.
Saved-state reconstruction is NEVER used as a dynamic hazard validation.
Every source trajectory, including local replay failures, remains in the ledger.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from hazardlib.config import load_config
from hazardlib.hashing import sha256_hex
from hazardlib.pipeline_io import write_yaml

from .demos import map_plain_state
from .geometry import is_collision_geom, mat_to_quat
from .host import LiberoHost
from .trajectory_segments import segment_features
from .trajectory_stats import distribution, local_positions, phases_from_events

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SCHEMA_VERSION = "2"


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def scene_groups(host):
    """Collision geoms by object / fixture name. The moving parts of a fixture (drawers, doors, knobs: rigid
    groups of their own) belong to the fixture they hang from."""
    collision = [g for g in range(host.m.ngeom) if is_collision_geom(host.m, g)]
    groups = {name: [g for g in collision if host.label_of_body[int(host.m.geom_bodyid[g])] == name]
              for name in list(host.objects) + list(host.fixtures) if not name.startswith("hazardlib_")}
    for fixture, root in host.fixtures.items():
        groups[fixture] += [g for g in collision if host.label_of_body[int(host.m.geom_bodyid[g])] not in groups
                            and host._descends(int(host.m.geom_bodyid[g]), root)]
    return groups


def describe_entities(host, names):
    """The task's entities of interest (BDDL ``obj_of_interest``) as analysis entities.

    free object -> ``kind: free`` (pose, grasp); an object / fixture / site with hinge or slide joints (drawer,
    door, stove knob) -> ``kind: articulated`` with its joints, its pose being the site's (a region such as
    ``wooden_cabinet_1_middle_region`` moves with its drawer) or the body's; everything else (a goal region,
    a joint-less fixture such as the caddy) is ``static``: it is a destination, never interacted with.
    Returns ``(entities, static, probes)``; ``entities`` get ``object_index`` into ``object_names + extra`` and
    ``probes`` (not serialised) says how to read each articulated entity's pose and joints in the simulator."""
    inner, m, mj = host.env.env, host.m, host.mj
    entities, static, extra, joint_names, probes = [], [], [], [], []
    for name in host.env.obj_of_interest:
        if name in host.free_objects:
            entities.append({"name": name, "kind": "free", "object_index": names.index(name)})
            continue
        site = inner.object_sites_dict.get(name)
        if site is not None:
            joints, parent, source = list(getattr(site, "joints", None) or []), getattr(site, "parent_name", None), "site"
        else:
            obj = inner.get_object(name)
            joints, parent, source = list(getattr(obj, "joints", None) or []), None, "body"
        kinds = {int(mj.mjtJoint.mjJNT_HINGE): "hinge", int(mj.mjtJoint.mjJNT_SLIDE): "slide"}
        joints = [j for j in joints if j in host.joints and int(m.jnt_type[m.joint(j).id]) in kinds]
        if not joints:
            static.append({"name": name, "parent": parent, "reason": "no hinge/slide joint"})
            continue
        entity = {"name": name, "kind": "articulated", "pose_source": source, "parent": parent,
                  "joint_names": joints, "joint_types": [kinds[int(m.jnt_type[m.joint(j).id])] for j in joints],
                  "joint_indices": list(range(len(joint_names), len(joint_names) + len(joints))),
                  "object_index": len(names) + len(extra)}
        entity["pose_name"] = name
        joint_names += joints
        extra.append(name)
        probes.append(("site", m.site(name).id) if source == "site" else
                      ("body", host.fixtures.get(name, host.objects.get(name, -1))))
        entities.append(entity)
    return entities, static, extra, joint_names, probes


def snapshot(host, names, groups, neighbors, gripper_geoms, finger_points, probes=(), joint_addrs=()):
    r, s = host.robot_state(0), host._fresh()
    own = [i for i, name in enumerate(r.collision_body_names) if name in host.robot_groups]
    poses = host.object_poses(0)
    low = min(float((p.dot(np.asarray(s.xmat[body]).reshape(3, 3).T) + s.xpos[body])[:, 2].min())
              for body, p in finger_points)
    gap = host._search
    fromto = np.zeros(6)
    for a in gripper_geoms:
        for name in neighbors:
            for b in groups[name]:
                gap = min(gap, float(host.mj.mj_geomDistance(host.m, s, a, b, gap, fromto)))
    return {"eef_pos": r.eef_pose.pos, "eef_quat": r.eef_pose.quat,
            "body_pos": np.asarray(r.collision_body_pos)[own],
            "body_quat": np.asarray(r.collision_body_quat)[own],
            # free objects first, then the pose of every articulated entity (site or body of its moving part)
            "object_pos": np.array([poses[n].pos for n in names] + [articulated_pose(s, kind, i)[0]
                                                                    for kind, i in probes]).reshape(-1, 3),
            "object_quat": np.array([poses[n].quat for n in names] + [articulated_pose(s, kind, i)[1]
                                                                      for kind, i in probes]).reshape(-1, 4),
            "joint_pos": np.array([s.qpos[a] for a in joint_addrs], dtype=float),
            "gripper": r.gripper_opening_m, "held_index": names.index(r.held_object) if r.held_object in names else -1,
            "gripper_low_z": low, "neighbor_clearance": gap, "native_success": host.task_success(0)}


def articulated_pose(s, kind, index):
    if kind == "site":
        return np.array(s.site_xpos[index]), mat_to_quat(np.asarray(s.site_xmat[index]).reshape(3, 3))
    return np.array(s.xpos[index]), np.array(s.xquat[index])


def packed(samples, dt, surface_z):
    result = {key: np.asarray([s[key] for s in samples]) for key in samples[0]}
    result["time_s"] = np.arange(len(samples)) * dt
    result["surface_z"] = np.asarray(surface_z)
    return result


def features(arrays, scene, settings):
    """Interaction segmentation of one recording; for a first entity that is a free object the earlier
    grasp / lift / transport / placement phases are kept as ``legacy_phases`` for comparison."""
    result = segment_features(arrays, scene["entities"], settings, scene["geometry_spacing_m"])
    first = scene["entities"][0]
    if first["kind"] == "free":
        target = np.asarray(arrays["object_pos"])[:, first["object_index"], 2]
        result["legacy_phases"] = phases_from_events(
            arrays["time_s"], arrays["held_index"], target, first["object_index"],
            scene["geometry_spacing_m"] * settings["lift_spacing_multiple"], settings["event_consecutive_samples"])
    return result


def target_frame(arrays, feats, scene):
    """The end effector in the initial frame of this demonstration's first-interaction entity (the scene's
    first entity when nothing was interacted with), and the index of that entity in ``object_pos``."""
    index = scene["entities"][0 if feats["first_entity_index"] is None else feats["first_entity_index"]]["object_index"]
    return (local_positions(arrays["eef_pos"], arrays["object_pos"][0, index], arrays["object_quat"][0, index]),
            np.asarray(index))


def collect(root, settings):
    root = Path(root)
    (root / "trajectories").mkdir(parents=True, exist_ok=True)
    host = LiberoHost()
    try:
        suite = settings["suite"]
        task = [t for t in host.task_ids() if t.startswith(suite + "/")][settings["task_index"]]
        host.reset_episode(0, task, suite + "/demo_0", settings["seed"])
        file = host._demo(task, suite)
        keys = file.episodes()
        if len(keys) != settings["expected_demonstrations"]:
            raise ValueError("expected the entire set of %d demonstrations, found %d" %
                             (settings["expected_demonstrations"], len(keys)))
        source_path = file.path
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        identity = {"task_id": task, "source_sha256": source_hash, "bd_hash": host.describe().hash(),
                    "settings_hash": sha256_hex(settings), "schema_version": SCHEMA_VERSION,
                    "suite": suite, "task_index": settings["task_index"]}
        identity_path = root / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("analysis directory identity mismatch: use another output directory")
        save_json(identity_path, identity)
        write_yaml(root / "analysis_protocol.yaml", settings)
        geometry = host.robot_collision_points(0)
        np.savez_compressed(root / "robot_geometry.npz", **geometry)
        # Free objects are recorded by name; the pose of each articulated entity of interest follows them in
        # object_pos / object_quat (see describe_entities), so ``object_*[:, target_index]`` is the pose of
        # whichever entity is the target, free object or fixture part.
        names = sorted(n for n in host.objects if not n.startswith("hazardlib_"))
        interests = list(host.env.obj_of_interest)
        entities, static, extra, joint_names, probes = describe_entities(host, names)
        if not entities:
            raise ValueError("%s: no entity of interest can be interacted with (static: %s)" % (task, static))
        target = entities[0]["name"]
        target_index = entities[0]["object_index"]
        joint_addrs = [host.joints[j][0] for j in joint_names]
        groups = scene_groups(host)
        # Neighbours: every other free object and fixture, except the ones the task itself uses (the fixture a
        # task region / joint belongs to is touched on purpose, like the target).
        used = set(interests).union(e["parent"] for e in entities + static if e.get("parent"))
        neighbors = sorted(set(names).union(host.fixtures) - used)
        gripper_names = [n for n in host.robot_groups if n.startswith("gripper")]
        finger_points = [(host.groups.frame_body[n], geometry[n]) for n in gripper_names]
        gripper_geoms = [g for g in range(host.m.ngeom) if is_collision_geom(host.m, g) and
                         host.groups.group_of[int(host.m.geom_bodyid[g])] in gripper_names]
        surfaces = host.support_surfaces(0)
        surface_z = max(s.min_m[2] for s in surfaces)
        scene = {"task_id": task, "suite": suite, "task_index": settings["task_index"],
                 "language_instruction": host.language_instruction(0),
                 "source_path": str(source_path), "object_names": names + extra, "free_object_names": names,
                 "target": target, "target_index": target_index, "target_selection": "provisional: first entity",
                 "task_entities": interests, "entities": entities, "static_task_entities": static,
                 "joint_names": joint_names, "neighbor_objects": neighbors,
                 "robot_body_names": list(host.robot_groups), "geometry_spacing_m": host.describe().robot_point_spacing_m,
                 "control_dt_s": host.control_dt, "support_surfaces": [s.to_dict() for s in surfaces],
                 "gravity_mps2": abs(float(host.m.opt.gravity[2])), "agent": host.pool[0].agent.to_dict(),
                 "camera_hw": host._hw, "cameras": host._cameras,
                 "state_observation_alignment": "states[k] is before action k; obs[k] is after action k",
                 "source_success_definition": "final stored reward=1 and done=1; reconstructed success is separate",
                 "replay_definition": "native actions from each demo initial state; no state forcing during replay"}
        for entity in entities:
            if entity["kind"] == "free":
                geoms = groups[entity["name"]]
            else:   # the moving part: collision geoms below the body of each of its joints
                bodies = [int(host.m.jnt_bodyid[host.m.joint(j).id]) for j in entity["joint_names"]]
                geoms = [g for g in range(host.m.ngeom) if is_collision_geom(host.m, g) and
                         any(host._descends(int(host.m.geom_bodyid[g]), b) for b in bodies)]
            bound_lo, bound_hi = [], []
            for g in geoms:
                aabb = host.m.geom_aabb[g]
                rotation = host.d.geom_xmat[g].reshape(3, 3)
                centre = host.d.geom_xpos[g] + rotation.dot(aabb[:3])
                extent = np.abs(rotation).dot(aabb[3:])
                bound_lo.append(centre - extent)
                bound_hi.append(centre + extent)
            entity["collision_bounds_size_m"] = ((np.max(bound_hi, axis=0) - np.min(bound_lo, axis=0)).tolist()
                                                 if bound_lo else None)
        scene["target_collision_bounds_size_m"] = entities[0]["collision_bounds_size_m"]
        save_json(root / "scene.json", scene)
        for key in keys:
            out = root / "trajectories" / key
            out.mkdir(exist_ok=True)
            if (out / "complete.json").is_file():
                print("CACHED " + key, flush=True)
                continue
            host.reset_episode(0, task, suite + "/" + key, settings["seed"])
            demo = host._demo(task, suite)
            episode = demo.episode(key)
            states, actions = demo.states(key), demo.actions(key)
            layout = demo.joint_layout(key)
            stored_success = bool(episode["rewards"][-1] == 1 and episode["dones"][-1] == 1)
            original = []
            for state in states:
                mapped, _ = map_plain_state(state, layout, host.joints, 1 + host.m.nq + host.m.nv)
                host.sim.set_state_from_flattened(mapped)
                for member in host.members:
                    host._park(member)
                host.mj.mj_forward(host.m, host.d)
                host._fresh_at = -1
                original.append(snapshot(host, names, groups, neighbors, gripper_geoms, finger_points, probes, joint_addrs))
            source = packed(original, host.control_dt, surface_z)
            source_features = features(source, scene, settings)
            source["eef_target_frame"], source["target_index"] = target_frame(source, source_features, scene)
            source["source_clock_s"] = states[:, 0]
            np.savez_compressed(out / "source.npz", **source)
            # A fresh reset after state reconstruction restores controller and rendering history.
            host.reset_episode(0, task, suite + "/" + key, settings["seed"])
            replay = [snapshot(host, names, groups, neighbors, gripper_geoms, finger_points, probes, joint_addrs)]
            success_t = None
            for index, action in enumerate(actions):
                host.step(0, action)
                replay.append(snapshot(host, names, groups, neighbors, gripper_geoms, finger_points, probes, joint_addrs))
                if replay[-1]["native_success"]:
                    success_t = (index + 1) * host.control_dt
                    break
            actual = packed(replay, host.control_dt, surface_z)
            actual_features = features(actual, scene, settings)
            actual["eef_target_frame"], actual["target_index"] = target_frame(actual, actual_features, scene)
            np.savez_compressed(out / "replay.npz", **actual)
            n = min(len(source["eef_pos"]), len(actual["eef_pos"]))
            report = {"demo": key, "source_success": stored_success, "source_samples": len(states),
                      "source_native_success": bool(source["native_success"].any()),
                      "replay_success": success_t is not None, "replay_success_t_s": success_t,
                      "replay_samples": len(actual["time_s"]),
                      "replay_source_eef_rmse_m": float(np.sqrt(np.mean(np.sum(
                          (source["eef_pos"][:n] - actual["eef_pos"][:n]) ** 2, axis=1)))),
                      "source_features": source_features, "replay_features": actual_features}
            save_json(out / "complete.json", report)
            print("COLLECT " + json.dumps({k: report[k] for k in ("demo", "source_success", "replay_success",
                                                                  "replay_success_t_s")}), flush=True)
    finally:
        host.close()


def _counted(items):
    out = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def derive(root, settings):
    root = Path(root)
    scene = json.loads((root / "scene.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((root / "trajectories").glob("demo_*/complete.json"),
                                                     key=lambda p: int(p.parent.name.split("_")[1]))]
    if len(rows) != settings["expected_demonstrations"]:
        raise ValueError("derive requires all %d demonstrations, got %d" % (settings["expected_demonstrations"], len(rows)))
    successes = [r for r in rows if r["replay_success"]]
    # Recompute features from the saved arrays so phase fixes never require
    # repeating collection or physics. The source/replay arrays remain untouched.
    if scene.get("entities") is None:
        raise ValueError("scene.json predates the interaction segmentation (schema 1): collect again")
    for row in rows:
        for source in ("source", "replay"):
            with np.load(root / "trajectories" / row["demo"] / (source + ".npz")) as arrays:
                row[source + "_features"] = features(arrays, scene, settings)
        save_json(root / "trajectories" / row["demo"] / "complete.json", row)
    q = settings["quantiles"]
    # The scene's target: the entity most demonstrations interact with first (replayed ones; ties go to the
    # first-listed entity of interest). Demonstrations that start with another entity keep their own entity as
    # the frame of ``eef_target_frame`` (``replay_features.first_entity_index``).
    first_counts = _counted(r["replay_features"]["first_entity"] for r in (successes or rows)
                            if r["replay_features"]["first_entity"] is not None)
    if first_counts:
        order = [e["name"] for e in scene["entities"]]
        target = max(first_counts, key=lambda n: (first_counts[n], -order.index(n)))
        chosen = scene["entities"][order.index(target)]
        scene.update(target=target, target_index=chosen["object_index"],
                     target_collision_bounds_size_m=chosen["collision_bounds_size_m"],
                     target_selection="entity of the first interaction, most frequent over replayed demonstrations",
                     first_entity_counts=first_counts)
        save_json(root / "scene.json", scene)
    phases = sorted({p["name"] for r in rows for p in r["source_features"]["phases"]})
    interaction_counts = {}
    for source in ("source", "replay"):
        interaction_counts[source] = dict(sorted(_counted(str(r[source + "_features"]["n_interactions"])
                                                          for r in rows).items()))
    summary = {"task_id": scene["task_id"], "suite": scene["suite"],
               "language_instruction": scene["language_instruction"],
               "target": scene["target"], "entities": [e["name"] for e in scene["entities"]],
               "static_task_entities": [e["name"] for e in scene["static_task_entities"]],
               "n_interactions_distribution": interaction_counts,
               "no_interaction_demos": {s: [r["demo"] for r in rows if r[s + "_features"]["n_interactions"] == 0]
                                        for s in ("source", "replay")},
               "n_demonstrations": len(rows),
               "n_source_success": sum(r["source_success"] for r in rows),
               "n_replay_success": len(successes), "failed_replays": [r["demo"] for r in rows if not r["replay_success"]],
               "source_replay_rmse_m": distribution([r["replay_source_eef_rmse_m"] for r in rows], q), "phases": {}}
    for phase in phases:
        observed = [p for r in rows for p in r["source_features"]["phases"] if p["name"] == phase]
        summary["phases"][phase] = {"n_demonstrations": len(observed),
                                   "entities": dict(sorted(_counted(p["entity"] for p in observed).items())),
                                   "duration_s": distribution([p["duration_s"] for p in observed], q),
                                   "path_length_m": distribution([p["path_length_m"] for p in observed], q)}
    # Pool per-frame speeds as a declared time-weighted distribution; duration uses one value per episode.
    moving_speeds, gaps, local_paths, phase_arrays = [], [], [], {}
    workspace = []
    for row in rows:
        with np.load(root / "trajectories" / row["demo"] / "source.npz") as a:
            speeds = np.linalg.norm(np.diff(a["eef_pos"], axis=0) / np.diff(a["time_s"])[:, None], axis=1)
            moving_speeds.extend(speeds[speeds > scene["geometry_spacing_m"] / scene["control_dt_s"]])
            gaps.extend(a["neighbor_clearance"])
            local_paths.append(a["eef_target_frame"])   # in the frame of the demonstration's first entity
            workspace.extend(a["eef_pos"])
            workspace.extend(a["object_pos"].reshape(-1, 3))
            for p in row["source_features"]["phases"]:
                lo, hi = p["start_index"], p["end_index"]
                phase_arrays.setdefault(p["name"], []).append({
                    "speeds": speeds[lo:hi], "gaps": a["neighbor_clearance"][lo:hi+1],
                    "surface_gaps": a["gripper_low_z"][lo:hi+1] - a["surface_z"]})
    summary["moving_speed_mps"] = distribution(moving_speeds, q)
    summary["gripper_neighbor_clearance_m"] = distribution(gaps, q)
    for name, arrays in phase_arrays.items():
        summary["phases"][name]["gripper_neighbor_clearance_m"] = distribution(
            np.concatenate([a["gaps"] for a in arrays]), q)
        summary["phases"][name]["gripper_table_gap_m"] = distribution(
            np.concatenate([a["surface_gaps"] for a in arrays]), q)
        medians = [float(np.median(a["speeds"])) for a in arrays if len(a["speeds"])]
        summary["phases"][name]["per_trajectory_median_speed_mps"] = distribution(medians, q)
        summary["phases"][name]["candidate_speed_levels_mps"] = [
            float(np.quantile(medians, quantile)) for quantile in settings["speed_quantiles"]] if medians else []
    summary["target_frame_path_bounds_m"] = {"min": np.concatenate(local_paths).min(axis=0).tolist(),
                                             "max": np.concatenate(local_paths).max(axis=0).tolist()}
    summary["measured_workspace_m"] = {"min": np.min(workspace, axis=0).tolist(),
                                        "max": np.max(workspace, axis=0).tolist()}
    summary["provenance"] = {"analysis_identity": json.loads((root / "identity.json").read_text()),
        "speed_weighting": "moving frames, all 50 source trajectories; threshold=geometry_spacing/control_dt",
        "trajectory_aggregation": "scalar distributions and target-frame envelopes; no averaged path",
        "mass_material_size": "existing declared 4 cm / 50 g proxy, not inferred physical human properties",
        "phase_threshold": "interaction segmentation (segmentation.yaml): activity sustained for "
                           "event_consecutive_samples; lift = lift_spacing_multiple geometry sampling intervals",
        "phase_speed_weighting": "quantiles of per-trajectory phase median speeds; one vote per demonstration"}
    write_yaml(root / "derived.yaml", summary)
    save_json(root / "coverage.json", {"demonstrations": rows, "summary": summary})
    print("DERIVED " + json.dumps({k: summary[k] for k in ("n_demonstrations", "n_source_success", "n_replay_success")}))


def validate_in_workers(out, task_index, workers, suite=None):
    """Split the demonstrations round-robin over ``workers`` processes of the ``validate`` stage (each
    has its own simulator); logs go to ``<out>/worker_<k>.log``. Returns the worst return code."""
    import subprocess
    import sys
    root = Path(out)
    rows = json.loads((root / "coverage.json").read_text())["demonstrations"]
    processes = []
    for worker in range(workers):
        demos = [row["demo"] for i, row in enumerate(rows) if i % workers == worker]
        if not demos:
            continue
        command = [sys.executable, "-u", "-m", "hazardhosts.libero.trajectory_analysis", "validate", "--out",
                   str(root), "--demos", ",".join(demos)]
        if suite is not None:
            command += ["--suite", suite]
        if task_index is not None:
            command += ["--task-index", str(task_index)]
        log = (root / ("worker_%d.log" % worker)).open("w")
        processes.append((worker, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    codes = {}
    for worker, process, log in processes:
        codes[worker] = process.wait()
        log.close()
        print("WORKER " + json.dumps({"worker": worker, "returncode": codes[worker]}), flush=True)
    save_json(root / "workers_result.json", codes)
    return max(abs(code) for code in codes.values()) if codes else 0


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["collect", "derive", "validate", "report"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--demos", help="validation subset for debugging; omitted processes every demonstration")
    parser.add_argument("--suite", choices=SUITES, help="LIBERO suite instead of the protocol file's; with "
                        "--task-index it selects the task (one output directory per suite and task)")
    parser.add_argument("--task-index", type=int, help="task of the suite instead of the protocol file's; "
                        "one output directory per task (the directory identity includes it)")
    parser.add_argument("--workers", type=int, help="validate only: number of parallel simulator processes")
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[4] / "configs/hazard/pilot/libero"
    settings = load_config("trajectory_analysis", strict=True, config_dir=config_dir)
    settings = dict(settings, segmentation=load_config("segmentation", strict=True, config_dir=config_dir))
    if args.suite is not None:
        settings = dict(settings, suite=args.suite)
    if args.task_index is not None:
        settings = dict(settings, task_index=args.task_index)
    if args.stage == "collect":
        collect(args.out, settings)
    elif args.stage == "derive":
        derive(args.out, settings)
    elif args.stage == "validate" and args.workers:
        raise SystemExit(validate_in_workers(args.out, args.task_index, args.workers, args.suite))
    else:
        from .trajectory_candidates import validate, report
        if args.stage == "validate":
            validate(args.out, settings, None if args.demos is None else args.demos.split(","))
        else:
            report(args.out, settings)


if __name__ == "__main__":
    main()
