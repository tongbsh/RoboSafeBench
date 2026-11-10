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
from .geometry import is_collision_geom
from .host import LiberoHost
from .trajectory_stats import distribution, local_positions, phase_features


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def scene_groups(host):
    collision = [g for g in range(host.m.ngeom) if is_collision_geom(host.m, g)]
    return {name: [g for g in collision if host.label_of_body[int(host.m.geom_bodyid[g])] == name]
            for name in list(host.objects) + list(host.fixtures) if not name.startswith("hazardlib_")}


def snapshot(host, names, groups, neighbors, gripper_geoms, finger_points):
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
            "object_pos": np.array([poses[n].pos for n in names]),
            "object_quat": np.array([poses[n].quat for n in names]),
            "gripper": r.gripper_opening_m, "held_index": names.index(r.held_object) if r.held_object in names else -1,
            "gripper_low_z": low, "neighbor_clearance": gap, "native_success": host.task_success(0)}


def packed(samples, dt, surface_z):
    result = {key: np.asarray([s[key] for s in samples]) for key in samples[0]}
    result["time_s"] = np.arange(len(samples)) * dt
    result["surface_z"] = np.asarray(surface_z)
    return result


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
                    "settings_hash": sha256_hex(settings), "schema_version": "1"}
        identity_path = root / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("analysis directory identity mismatch: use another output directory")
        save_json(identity_path, identity)
        write_yaml(root / "analysis_protocol.yaml", settings)
        geometry = host.robot_collision_points(0)
        np.savez_compressed(root / "robot_geometry.npz", **geometry)
        # Only physical, task-relevant free objects are compared; target contacts are intentional.
        names = sorted(n for n in host.objects if not n.startswith("hazardlib_"))
        interests = list(host.env.obj_of_interest)
        target = next(n for n in interests if n in host.free_objects)
        target_index = names.index(target)
        groups = scene_groups(host)
        neighbors = sorted(set(names) - set(interests))
        gripper_names = [n for n in host.robot_groups if n.startswith("gripper")]
        finger_points = [(host.groups.frame_body[n], geometry[n]) for n in gripper_names]
        gripper_geoms = [g for g in range(host.m.ngeom) if is_collision_geom(host.m, g) and
                         host.groups.group_of[int(host.m.geom_bodyid[g])] in gripper_names]
        surfaces = host.support_surfaces(0)
        surface_z = max(s.min_m[2] for s in surfaces)
        scene = {"task_id": task, "source_path": str(source_path), "object_names": names, "target": target,
                 "target_index": target_index, "task_entities": interests, "neighbor_objects": neighbors,
                 "robot_body_names": list(host.robot_groups), "geometry_spacing_m": host.describe().robot_point_spacing_m,
                 "control_dt_s": host.control_dt, "support_surfaces": [s.to_dict() for s in surfaces],
                 "gravity_mps2": abs(float(host.m.opt.gravity[2])), "agent": host.pool[0].agent.to_dict(),
                 "camera_hw": host._hw, "cameras": host._cameras,
                 "state_observation_alignment": "states[k] is before action k; obs[k] is after action k",
                 "source_success_definition": "final stored reward=1 and done=1; reconstructed success is separate",
                 "replay_definition": "native actions from each demo initial state; no state forcing during replay"}
        bound_lo, bound_hi = [], []
        for g in groups[target]:
            aabb = host.m.geom_aabb[g]
            rotation = host.d.geom_xmat[g].reshape(3, 3)
            centre = host.d.geom_xpos[g] + rotation.dot(aabb[:3])
            extent = np.abs(rotation).dot(aabb[3:])
            bound_lo.append(centre - extent)
            bound_hi.append(centre + extent)
        scene["target_collision_bounds_size_m"] = (np.max(bound_hi, axis=0) - np.min(bound_lo, axis=0)).tolist()
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
                original.append(snapshot(host, names, groups, neighbors, gripper_geoms, finger_points))
            source = packed(original, host.control_dt, surface_z)
            source["eef_target_frame"] = local_positions(source["eef_pos"], source["object_pos"][0, target_index],
                                                         source["object_quat"][0, target_index])
            source["source_clock_s"] = states[:, 0]
            np.savez_compressed(out / "source.npz", **source)
            # A fresh reset after state reconstruction restores controller and rendering history.
            host.reset_episode(0, task, suite + "/" + key, settings["seed"])
            replay = [snapshot(host, names, groups, neighbors, gripper_geoms, finger_points)]
            success_t = None
            for index, action in enumerate(actions):
                host.step(0, action)
                replay.append(snapshot(host, names, groups, neighbors, gripper_geoms, finger_points))
                if replay[-1]["native_success"]:
                    success_t = (index + 1) * host.control_dt
                    break
            actual = packed(replay, host.control_dt, surface_z)
            actual["eef_target_frame"] = local_positions(actual["eef_pos"], actual["object_pos"][0, target_index],
                                                         actual["object_quat"][0, target_index])
            np.savez_compressed(out / "replay.npz", **actual)
            n = min(len(source["eef_pos"]), len(actual["eef_pos"]))
            report = {"demo": key, "source_success": stored_success, "source_samples": len(states),
                      "source_native_success": bool(source["native_success"].any()),
                      "replay_success": success_t is not None, "replay_success_t_s": success_t,
                      "replay_samples": len(actual["time_s"]),
                      "replay_source_eef_rmse_m": float(np.sqrt(np.mean(np.sum(
                          (source["eef_pos"][:n] - actual["eef_pos"][:n]) ** 2, axis=1)))),
                      "source_features": phase_features(source, target_index, settings, scene["geometry_spacing_m"]),
                      "replay_features": phase_features(actual, target_index, settings, scene["geometry_spacing_m"])}
            save_json(out / "complete.json", report)
            print("COLLECT " + json.dumps({k: report[k] for k in ("demo", "source_success", "replay_success",
                                                                  "replay_success_t_s")}), flush=True)
    finally:
        host.close()


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
    for row in rows:
        for source in ("source", "replay"):
            with np.load(root / "trajectories" / row["demo"] / (source + ".npz")) as arrays:
                row[source + "_features"] = phase_features(arrays, scene["target_index"], settings,
                                                           scene["geometry_spacing_m"])
        save_json(root / "trajectories" / row["demo"] / "complete.json", row)
    q = settings["quantiles"]
    phases = sorted({p["name"] for r in rows for p in r["source_features"]["phases"]})
    summary = {"task_id": scene["task_id"], "n_demonstrations": len(rows),
               "n_source_success": sum(r["source_success"] for r in rows),
               "n_replay_success": len(successes), "failed_replays": [r["demo"] for r in rows if not r["replay_success"]],
               "source_replay_rmse_m": distribution([r["replay_source_eef_rmse_m"] for r in rows], q), "phases": {}}
    for phase in phases:
        observed = [p for r in rows for p in r["source_features"]["phases"] if p["name"] == phase]
        summary["phases"][phase] = {"n_demonstrations": len(observed),
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
            local_paths.append(a["eef_target_frame"])
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
            float(np.quantile(medians, quantile)) for quantile in settings["speed_quantiles"]]
    summary["target_frame_path_bounds_m"] = {"min": np.concatenate(local_paths).min(axis=0).tolist(),
                                             "max": np.concatenate(local_paths).max(axis=0).tolist()}
    summary["measured_workspace_m"] = {"min": np.min(workspace, axis=0).tolist(),
                                        "max": np.max(workspace, axis=0).tolist()}
    summary["provenance"] = {"analysis_identity": json.loads((root / "identity.json").read_text()),
        "speed_weighting": "moving frames, all 50 source trajectories; threshold=geometry_spacing/control_dt",
        "trajectory_aggregation": "scalar distributions and target-frame envelopes; no averaged path",
        "mass_material_size": "existing declared 4 cm / 50 g proxy, not inferred physical human properties",
        "phase_threshold": "lift=2 geometry sampling intervals; contact events sustained for 3 control samples",
        "phase_speed_weighting": "quantiles of per-trajectory phase median speeds; one vote per demonstration"}
    write_yaml(root / "derived.yaml", summary)
    save_json(root / "coverage.json", {"demonstrations": rows, "summary": summary})
    print("DERIVED " + json.dumps({k: summary[k] for k in ("n_demonstrations", "n_source_success", "n_replay_success")}))


def validate_in_workers(out, task_index, workers):
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
    parser.add_argument("--task-index", type=int, help="task of the suite instead of the protocol file's; "
                        "one output directory per task (the directory identity includes it)")
    parser.add_argument("--workers", type=int, help="validate only: number of parallel simulator processes")
    args = parser.parse_args()
    settings = load_config("trajectory_analysis", strict=True, config_dir=Path(__file__).resolve().parents[4] /
                           "configs/hazard/pilot/libero")
    if args.task_index is not None:
        settings = dict(settings, task_index=args.task_index)
    if args.stage == "collect":
        collect(args.out, settings)
    elif args.stage == "derive":
        derive(args.out, settings)
    elif args.stage == "validate" and args.workers:
        raise SystemExit(validate_in_workers(args.out, args.task_index, args.workers))
    else:
        from .trajectory_candidates import validate, report
        if args.stage == "validate":
            validate(args.out, settings, None if args.demos is None else args.demos.split(","))
        else:
            report(args.out, settings)


if __name__ == "__main__":
    main()
