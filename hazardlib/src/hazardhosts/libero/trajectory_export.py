"""LIBERO exporter of the generator's two input variables (D20): ``export`` writes, for each usable
demonstration of a collected task, V1 (``hazardlib.generate.Trajectory``) and V2 (``SceneLayout``) under
``<task dir>/generation_inputs``.

The physical replay states are rebuilt exactly as ``validate`` always did (:func:`rebuild_states`: the cached
``physical_states.npz`` when present, else a replay of the recorded actions whose EEF path must equal the
collected one to 1e-12); the world pose of every non-robot body that carries collision geoms is then read at
every control step. A body is a rigid group of the host (``label_of_body`` groups geoms; a fixture's moving
parts - drawers, doors, knobs - are groups of their own, so they get per-step poses; a group that never moves
gets one pose). Its collision boxes are the axis-aligned bounds of each collision geom in the geom frame
(``geom_aabb``) expressed in the group's frame: bounding boxes, exact for box geoms and conservative for
meshes and curved geoms (MuJoCo collides a mesh as its convex hull). Infinite planes (the floor) are not boxes
and are skipped; the support surfaces of V2 are the host's.
"""
import json
from pathlib import Path

import numpy as np

from hazardlib.generate import LayoutBody, SceneLayout, Segment, Trajectory, TrajectorySet

from .geometry import _body_to_frame, is_collision_geom, quat_to_mat as geom_quat_to_mat
from .pilot import assemble_config


def rebuild_states(host, scene, row, folder, arrays, settings):
    """Flattened physical replay states ``(T, n)`` of one usable demonstration, one per control sample.

    Reuses ``<folder>/physical_states.npz`` (checked against the recorded EEF path at three samples and for
    native success), else replays the recorded actions from the demonstration's initial state, requires the EEF
    path to equal the collected replay to 1e-12 and native success at the end, and saves the file. The
    caller installs the runtime hooks afterwards."""
    host.install_hooks(None)
    host.reset_episode(0, scene["task_id"], settings["suite"] + "/" + row["demo"], settings["seed"])
    cache_path = Path(folder) / "physical_states.npz"
    if cache_path.exists():
        states = np.load(str(cache_path))["states"]
        if len(states) != len(arrays["time_s"]):
            raise ValueError("cached replay length mismatch")
        for index in (0, len(states) // 2, len(states) - 1):
            host.sim.set_state_from_flattened(states[index])
            host.mj.mj_forward(host.m, host.d)
            host._fresh_at = -1
            if not np.allclose(host.robot_state(0).eef_pose.pos, arrays["eef_pos"][index], atol=1e-12, rtol=0):
                raise ValueError("cached replay geometry mismatch")
        # mj_forward rebuilds contacts at the saved end state. The native evaluator read mj_step's
        # pre-integration contacts; predicates on a touching boundary can differ. Preserve the goal event
        # recorded during the real baseline, never certify a goal from state forcing.
        if not bool(arrays["native_success"][-1]):
            raise ValueError("cached physical replay did not record native success")
        return states
    actions = host._demo(scene["task_id"], settings["suite"]).actions(row["demo"])
    states = [host.sim.get_state().flatten()]
    actual_path = [host.robot_state(0).eef_pose.pos]
    for action in actions:
        host.step(0, action)
        states.append(host.sim.get_state().flatten())
        actual_path.append(host.robot_state(0).eef_pose.pos)
        if host.task_success(0):
            break
    if len(states) != len(arrays["time_s"]) or not np.allclose(actual_path, arrays["eef_pos"], atol=1e-12, rtol=0):
        raise RuntimeError("baseline replay drifted for " + row["demo"])
    if not host.task_success(0):
        raise RuntimeError("baseline replay no longer successful: " + row["demo"])
    states = np.asarray(states)
    np.savez_compressed(str(cache_path), states=states)
    return states


def _task_names(scene):
    """Labels of the task's own entities and the fixtures they hang from."""
    names = set(scene["task_entities"])
    for e in list(scene["entities"]) + list(scene["static_task_entities"]):
        names.add(e["name"])
        if e.get("parent"):
            names.add(e["parent"])
    return names


def layout_geometry(host, scene):
    """Static part of V2: a list of ``{label, role, frame, centre, rot, half}`` (frame = body id whose pose
    track gives the group's world pose), one per (host label, rigid group, role)."""
    m = host.m
    supports = host.support_labels()
    task = _task_names(scene)
    neighbours = set(scene["neighbor_objects"])
    members = set(mem.body for mem in host.members)
    plane = int(host.mj.mjtGeom.mjGEOM_PLANE)
    entries = {}
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        label = host.label_of_body[b]
        if not is_collision_geom(m, g) or b in host.robot_body_set or b in members or label.startswith("hazardlib_"):
            continue
        if int(m.geom_type[g]) == plane:
            if not (m.geom_size[g][0] > 0 and m.geom_size[g][1] > 0):
                continue
            centre, half = np.zeros(3), np.array([m.geom_size[g][0], m.geom_size[g][1], 0.0])
        else:
            centre, half = np.asarray(m.geom_aabb[g][:3], dtype=float), np.asarray(m.geom_aabb[g][3:], dtype=float)
        group = host.groups.group_of[b]
        frame = host.groups.frame_body.get(group, b)
        owner = label
        if owner not in host.objects and owner not in host.fixtures:
            owner = next((n for n, root in host.fixtures.items() if host._descends(frame, root)), label)
        if label in supports or host.body_name[b] in supports or owner in supports:
            role = "support"
        elif owner in task:
            role = "task_entity"
        elif owner in neighbours:
            role = "neighbour"
        else:
            role = "fixture"
        rot_bf, pos_bf = _body_to_frame(m, b, frame)
        rot_gf = rot_bf.dot(geom_quat_to_mat(m.geom_quat[g]))
        centre_f = rot_bf.dot(np.asarray(m.geom_pos[g], dtype=float)) + pos_bf + rot_gf.dot(centre)
        entry = entries.setdefault((label, frame, role), {"label": label, "role": role, "frame": frame,
                                                          "centre": [], "rot": [], "half": []})
        entry["centre"].append(centre_f)
        entry["rot"].append(rot_gf)
        entry["half"].append(half)
    out = []
    taken = set()
    for key in sorted(entries, key=lambda k: (k[0], k[1], k[2])):
        e = entries[key]
        name = e["label"] if e["label"] not in taken else "%s#%d.%s" % (e["label"], e["frame"], e["role"])
        taken.add(name)
        out.append(dict(e, label=name))
    return out


def body_tracks(host, states, geometry):
    """World pose of every group frame at every state: ``{frame: (pos (T, 3), quat (T, 4))}``."""
    frames = sorted(set(e["frame"] for e in geometry))
    pos = {f: [] for f in frames}
    quat = {f: [] for f in frames}
    for state in states:
        host.sim.set_state_from_flattened(state)
        host.mj.mj_forward(host.m, host.d)
        for f in frames:
            pos[f].append(np.array(host.d.xpos[f]))
            quat[f].append(np.array(host.d.xquat[f]))
    return {f: (np.asarray(pos[f]), np.asarray(quat[f])) for f in frames}


def scene_layout(host, scene, row, states, geometry, measured, layout_id):
    tracks = body_tracks(host, states, geometry)
    bodies = []
    for e in geometry:
        pos, quat = tracks[e["frame"]]
        if np.ptp(pos, axis=0).max() == 0.0 and np.ptp(quat, axis=0).max() == 0.0:
            pos, quat = pos[0], quat[0]          # never moves: one pose
        bodies.append(LayoutBody(label=e["label"], role=e["role"], box_centre=np.array(e["centre"]),
                                 box_rot=np.array(e["rot"]), box_half=np.array(e["half"]), pos=pos, quat=quat))
    return SceneLayout(layout_id=layout_id, control_dt_s=scene["control_dt_s"], n_steps=len(states), bodies=bodies,
                       support_surfaces=scene["support_surfaces"], workspace_measured_m=measured)


def trajectory_of(scene, row, arrays):
    """V1 of a collected replay (``replay.npz`` arrays + the segmentation of ``complete.json``)."""
    entities = scene["entities"]
    index = [e["object_index"] for e in entities]
    object_names = scene["object_names"]
    held = [object_names[i] if i >= 0 else None for i in arrays["held_index"]]
    segments = [Segment(name=p["name"], kind=p["kind"], entity=p["entity"], start_index=p["start_index"],
                        end_index=p["end_index"], t_start_s=p["t_start_s"], t_end_s=p["t_end_s"])
                for p in row["replay_features"]["phases"]]
    return Trajectory(demo=row["demo"], control_dt_s=scene["control_dt_s"], time_s=arrays["time_s"],
                      eef_pos=arrays["eef_pos"], eef_quat=arrays["eef_quat"], body_names=list(scene["robot_body_names"]),
                      body_pos=arrays["body_pos"], body_quat=arrays["body_quat"],
                      entity_names=[e["name"] for e in entities], entity_pos=arrays["object_pos"][:, index],
                      entity_quat=arrays["object_quat"][:, index], held=held, segments=segments)


def export(root, settings, max_demos=None):
    """Write ``<root>/generation_inputs`` (module docstring) for the first ``max_demos`` usable demonstrations
    in coverage order (all if ``None``). Existing demonstrations whose files are present are kept."""
    from .host import LiberoHost
    root = Path(root)
    scene = json.loads((root / "scene.json").read_text())
    coverage = json.loads((root / "coverage.json").read_text())
    measured = {"min_m": coverage["summary"]["measured_workspace_m"]["min"],
                "max_m": coverage["summary"]["measured_workspace_m"]["max"]}
    config = root / "config"
    if not config.exists():
        assemble_config(config)
    out = root / "generation_inputs"
    usable = [r for r in coverage["demonstrations"] if r["replay_success"]]
    if max_demos is not None:
        usable = usable[:max_demos]
    points = {k: v for k, v in np.load(str(root / "robot_geometry.npz")).items()}
    host = LiberoHost(config_dir=config)
    trajectories, layouts, hashes = [], {}, {}
    try:
        geometry = None
        for row in usable:
            folder = root / "trajectories" / row["demo"]
            arrays = dict(np.load(str(folder / "replay.npz")))
            states = rebuild_states(host, scene, row, folder, arrays, settings)
            geometry = layout_geometry(host, scene)     # the model is the same for every demonstration
            trajectory = trajectory_of(scene, row, arrays)
            layout = scene_layout(host, scene, row, states, geometry, measured, scene["suite"] + "/" + row["demo"])
            trajectories.append(trajectory)
            layouts[row["demo"]] = layout
            print("EXPORT %s steps=%d bodies=%d boxes=%d roles=%s" % (
                row["demo"], len(states), len(layout.bodies), sum(len(b.box_half) for b in layout.bodies),
                json.dumps({r: len(layout.by_role(r)) for r in ("task_entity", "neighbour", "fixture", "support")})),
                flush=True)
    finally:
        host.close()
    task_set = TrajectorySet(scene["task_id"], points, float(scene["geometry_spacing_m"]), trajectories, layouts)
    task_set.save_shared(out)
    for t in trajectories:
        hashes[t.demo] = task_set.save_demo(out, t)
    task_set.save_index(out, hashes)
    print("EXPORTED %d demonstrations -> %s" % (len(trajectories), out), flush=True)
    return task_set
