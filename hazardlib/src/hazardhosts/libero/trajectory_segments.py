"""Task-agnostic interaction segmentation of one demonstration (pure numpy, no simulator).

Every LIBERO task names its *entities of interest* (BDDL ``:obj_of_interest``). An entity is either a free
object (pose only; the gripper can hold it) or an articulated one (a drawer / door / knob: the position of its
moving joints). One *interaction* is a contiguous stretch of time in which the robot acts on one entity:

* free object: the gripper holds it (``held_index``) or it moves by more than ``entity_step_motion_m`` per
  control step (a push) and it is displaced by at least ``interaction_min_displacement_m`` (or held);
* articulated entity: a joint moves by more than ``articulation_step_motion`` per step and the joint changes by
  at least ``articulation_min_change`` (a drawer is not "opened" by a nudge).

An activity run must last ``event_consecutive_samples`` control samples (a single contact sample is no event);
runs of one entity separated by at most ``interaction_gap_fill_samples`` are one interaction (regrasp, a drawer
pulled in two strokes). Interactions of all entities are put in time order; an interaction that lies inside an
earlier one (a neighbouring interest object nudged while another one is carried) is dropped and listed, one that
overlaps its predecessor starts where that ends. The demonstration is then cut into

* ``approach_<k>``   before interaction k (from the end of interaction k-1, or sample 0),
* ``manipulate_<k>`` interaction k (grasp ... release, or first motion ... articulation stops),
* ``retreat``        after the last interaction until the end of the recording;

a segment of zero length is omitted. A demonstration without any interaction yields no segment and
``status == "no_interaction"``; it is reported, never an error.
"""
import numpy as np

from .trajectory_stats import distribution, first_sustained


def runs(mask, consecutive):
    """Maximal ``[start, end]`` (inclusive) true runs of ``mask`` that are at least ``consecutive`` long."""
    mask = np.asarray(mask, dtype=bool)
    out, start = [], None
    for index, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= consecutive:
                out.append([start, index - 1])
            start = None
    return out


def merge_gaps(spans, gap):
    """Join consecutive spans separated by at most ``gap`` samples (the samples between them)."""
    out = []
    for span in spans:
        if out and span[0] - out[-1][1] - 1 <= gap:
            out[-1][1] = span[1]
        else:
            out.append(list(span))
    return out


def step_moved(series, threshold):
    """``moved[t]``: the (T,) scalar ``series`` changes by more than ``threshold`` between samples t and t+1
    (the last sample is False)."""
    series = np.asarray(series, dtype=float)
    moved = np.zeros(len(series), dtype=bool)
    if len(series) > 1:
        moved[:-1] = np.abs(np.diff(series)) > threshold
    return moved


def _free_runs(entity, held, seg, consecutive, lift_threshold):
    pos = np.asarray(entity["pos"], dtype=float)
    step = np.zeros(len(pos))
    if len(pos) > 1:
        step[:-1] = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    held_mask = np.asarray(held) == entity["held_index"]
    active = np.logical_or(step > seg["entity_step_motion_m"], held_mask)
    out = []
    for start, end in merge_gaps(runs(active, consecutive), seg["interaction_gap_fill_samples"]):
        stop = min(end + 1, len(pos) - 1)
        displacement = float(np.linalg.norm(pos[start:stop + 1] - pos[start], axis=1).max())
        was_held = bool(held_mask[start:end + 1].any())
        if not was_held and displacement < seg["interaction_min_displacement_m"]:
            continue
        grasp = first_sustained(held_mask[start:stop + 1], consecutive) if was_held else None
        lifted = first_sustained(pos[start:stop + 1, 2] - pos[start, 2] > lift_threshold, consecutive)
        out.append({"start_index": int(start), "end_index": int(stop), "held": was_held,
                    "grasp_index": None if grasp is None else int(start + grasp),
                    "lift_index": None if lifted is None else int(start + lifted),
                    "change": displacement, "change_unit": "m"})
    return out


def _articulated_runs(entity, seg, consecutive):
    q = np.asarray(entity["joints"], dtype=float).reshape(len(entity["joints"]), -1)
    types = list(entity["joint_types"])
    step_limit, change_limit = seg["articulation_step_motion"], seg["articulation_min_change"]
    active = np.zeros(len(q), dtype=bool)
    for j, kind in enumerate(types):
        active |= step_moved(q[:, j], step_limit[kind])
    out = []
    for start, end in merge_gaps(runs(active, consecutive), seg["interaction_gap_fill_samples"]):
        stop = min(end + 1, len(q) - 1)
        change = [float(np.abs(q[start:stop + 1, j] - q[start, j]).max()) for j in range(len(types))]
        if not any(c >= change_limit[kind] for c, kind in zip(change, types)):
            continue
        out.append({"start_index": int(start), "end_index": int(stop), "held": False, "grasp_index": None,
                    "lift_index": None, "change": float(max(change)),
                    "change_unit": "m" if types[int(np.argmax(change))] == "slide" else "rad"})
    return out


def find_interactions(time, held_index, entities, settings, geometry_spacing):
    """Time-ordered, non-overlapping interactions over all ``entities`` and the ones dropped for overlap.

    ``entities`` items: ``name``, ``kind`` (``free`` | ``articulated``) and either ``pos`` (T, 3) and
    ``held_index`` (the value of ``held_index`` while the gripper holds it) or ``joints`` (T, J) and
    ``joint_types`` (``slide`` | ``hinge`` per column)."""
    seg = settings["segmentation"]
    consecutive = settings["event_consecutive_samples"]
    lift = geometry_spacing * settings["lift_spacing_multiple"]
    found = []
    for position, entity in enumerate(entities):
        if entity["kind"] == "free":
            spans = _free_runs(entity, held_index, seg, consecutive, lift)
        elif entity["kind"] == "articulated":
            spans = _articulated_runs(entity, seg, consecutive)
        else:
            raise ValueError("entity %r: unknown kind %r" % (entity["name"], entity["kind"]))
        for span in spans:
            found.append(dict(span, entity=entity["name"], entity_index=position, entity_kind=entity["kind"]))
    found.sort(key=lambda s: (s["start_index"], s["end_index"], s["entity_index"]))
    kept, dropped = [], []
    for item in found:
        if kept and item["end_index"] <= kept[-1]["end_index"]:
            dropped.append({"entity": item["entity"], "start_index": item["start_index"],
                            "end_index": item["end_index"], "reason": "inside interaction of " + kept[-1]["entity"]})
            continue
        if kept and item["start_index"] < kept[-1]["end_index"]:
            if item["end_index"] - kept[-1]["end_index"] < consecutive:
                dropped.append({"entity": item["entity"], "start_index": item["start_index"],
                                "end_index": item["end_index"], "reason": "overlap leaves < consecutive samples"})
                continue
            item = dict(item, start_index=kept[-1]["end_index"], truncated_overlap=True)
        kept.append(item)
    for k, item in enumerate(kept):
        item["k"] = k
        item["t_start_s"] = float(time[item["start_index"]])
        item["t_end_s"] = float(time[item["end_index"]])
    return kept, dropped


def build_phases(time, interactions):
    """``approach_k`` / ``manipulate_k`` / ``retreat`` from ordered interactions; zero-length segments omitted."""
    last = len(time) - 1
    phases, cursor = [], 0
    for item in interactions:
        if item["start_index"] > cursor:
            phases.append(_phase("approach_%d" % item["k"], "approach", cursor, item["start_index"], time, item))
        phases.append(_phase("manipulate_%d" % item["k"], "manipulate", item["start_index"], item["end_index"], time,
                             item))
        cursor = item["end_index"]
    if interactions and last > cursor:
        phases.append(_phase("retreat", "retreat", cursor, last, time, interactions[-1]))
    return phases


def _phase(name, kind, start, end, time, item):
    return {"name": name, "kind": kind, "interaction": int(item["k"]), "entity": item["entity"],
            "entity_kind": item["entity_kind"], "start_index": int(start), "end_index": int(end),
            "t_start_s": float(time[start]), "t_end_s": float(time[end])}


def entity_series(arrays, entities):
    """The per-entity series of one saved recording (``object_pos`` over ``object_names`` and ``joint_pos``);
    ``entities`` are the scene's descriptors (``scene["entities"]``)."""
    out = []
    for e in entities:
        if e["kind"] == "free":
            out.append({"name": e["name"], "kind": "free", "held_index": e["object_index"],
                        "pos": np.asarray(arrays["object_pos"])[:, e["object_index"]]})
        else:
            out.append({"name": e["name"], "kind": "articulated", "joint_types": e["joint_types"],
                        "joints": np.asarray(arrays["joint_pos"])[:, e["joint_indices"]]})
    return out


def segment_features(arrays, entities, settings, geometry_spacing):
    """Segment one recording and attach per-segment scalar features (the layout of ``phase_features``).

    Keys: ``phases`` (consumed by the candidate generator: name, start_index, end_index, t_start_s, t_end_s,
    duration_s, ...), ``interactions``, ``n_interactions``, ``status``, ``dropped_overlaps``,
    ``first_entity`` / ``first_entity_index`` (entity of the first interaction), plus whole-recording speed."""
    time, eef = np.asarray(arrays["time_s"]), np.asarray(arrays["eef_pos"])
    series = entity_series(arrays, entities)
    interactions, dropped = find_interactions(time, np.asarray(arrays["held_index"]), series, settings,
                                              geometry_spacing)
    phases = build_phases(time, interactions)
    q = settings["quantiles"]
    speed = np.linalg.norm(np.diff(eef, axis=0) / np.diff(time)[:, None], axis=1)
    for phase in phases:
        a, b = phase["start_index"], phase["end_index"]
        phase["duration_s"] = float(time[b] - time[a])
        phase["path_length_m"] = float(np.linalg.norm(np.diff(eef[a:b + 1], axis=0), axis=1).sum())
        phase["speed_mps"] = distribution(speed[a:b], q)
        phase["gripper_table_gap_m"] = distribution(np.asarray(arrays["gripper_low_z"])[a:b + 1] -
                                                     np.asarray(arrays["surface_z"]), q)
        phase["neighbor_clearance_m"] = distribution(np.asarray(arrays["neighbor_clearance"])[a:b + 1], q)
    first = interactions[0] if interactions else None
    return {"phases": phases, "interactions": interactions, "n_interactions": len(interactions),
            "status": "ok" if interactions else "no_interaction", "dropped_overlaps": dropped,
            "first_entity": None if first is None else first["entity"],
            "first_entity_index": None if first is None else first["entity_index"],
            "speed_mps": distribution(speed, q), "duration_s": float(time[-1] - time[0]),
            "path_length_m": float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum())}
