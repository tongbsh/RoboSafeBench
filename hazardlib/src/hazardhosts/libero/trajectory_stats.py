"""Statistics for complete demonstration sets; no simulator or learned policy.

Each trajectory keeps its own layout. Only distributions of scalar features and
object-relative paths are pooled; no averaged robot trajectory is instantiated.
"""
from typing import Any, Dict, List

import numpy as np

from hazardlib.generate.stats import distribution  # noqa: F401  (kept importable from here)


def first_sustained(mask, consecutive):
    """First index of a complete true run (not the index confirming the run)."""
    length = 0
    for index, value in enumerate(mask):
        length = length + 1 if value else 0
        if length >= consecutive:
            return index - consecutive + 1
    return None


def phases_from_events(time, held, target_z, target_index, lift_threshold, consecutive):
    time = np.asarray(time)
    held = np.asarray(held)
    grasp = first_sustained(held == target_index, consecutive)
    lifted = first_sustained(np.asarray(target_z) - target_z[0] > lift_threshold, consecutive)
    # Bilateral contacts can disappear briefly while a securely lifted object is
    # moving. A later confirmed contact must not place "grasp" after "lift".
    if lifted is not None:
        grasp = lifted if grasp is None else min(grasp, lifted)
    boundaries = [("approach", 0)]
    if grasp is not None and grasp > 0:
        boundaries.append(("grasp", grasp))
    if lifted is not None:
        boundaries.append(("transport", lifted))
    if lifted is not None:
        release = first_sustained(held[lifted:] != target_index, consecutive)
        if release is not None and lifted + release > lifted:
            boundaries.append(("placement", lifted + release))
    out = []
    for index, (name, start) in enumerate(boundaries):
        end = boundaries[index + 1][1] if index + 1 < len(boundaries) else len(time) - 1
        if end > start:
            out.append({"name": name, "start_index": int(start), "end_index": int(end),
                        "t_start_s": float(time[start]), "t_end_s": float(time[end])})
    return {"grasp_index": grasp, "lift_index": lifted, "phases": out}


def local_positions(positions, origin, quat_wxyz):
    """Align points to ONE layout's initial target frame, retaining all paths."""
    q = np.asarray(quat_wxyz, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.array([[1 - 2 * (y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return (np.asarray(positions) - np.asarray(origin)).dot(rotation)


def phase_features(arrays, target_index, settings, geometry_spacing):
    time, eef = arrays["time_s"], arrays["eef_pos"]
    target = arrays["object_pos"][:, target_index]
    events = phases_from_events(time, arrays["held_index"], target[:, 2], target_index,
                               geometry_spacing * settings["lift_spacing_multiple"],
                               settings["event_consecutive_samples"])
    speed = np.linalg.norm(np.diff(eef, axis=0) / np.diff(time)[:, None], axis=1)
    for phase in events["phases"]:
        a, b = phase["start_index"], phase["end_index"]
        phase["duration_s"] = float(time[b] - time[a])
        phase["path_length_m"] = float(np.linalg.norm(np.diff(eef[a:b+1], axis=0), axis=1).sum())
        phase["speed_mps"] = distribution(speed[a:b], settings["quantiles"])
        phase["gripper_table_gap_m"] = distribution(arrays["gripper_low_z"][a:b+1] - arrays["surface_z"],
                                                     settings["quantiles"])
        phase["neighbor_clearance_m"] = distribution(arrays["neighbor_clearance"][a:b+1], settings["quantiles"])
    events["speed_mps"] = distribution(speed, settings["quantiles"])
    events["duration_s"] = float(time[-1] - time[0])
    events["path_length_m"] = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum())
    return events
