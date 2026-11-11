"""Scene clearance of a hazard's own trajectory (D19 c) on the LIBERO host.

The simulator-free core (playback path, sweep samples, the verdict, the V2 layout prefilter) lives in
:mod:`hazardlib.generate.clearance` and is re-exported here. This module holds the exact check, the authority:
:func:`host_probes` places a pool hazard with its collision on (as ``visibility_probe`` does) in saved physical
replay states and reads every MuJoCo contact with ``dist < 0``; robot links, the object the robot holds in that
state, the support bodies (the table the hazard rides on or lands on) and pool members are ignored. The two
scene states (``replay`` / ``frozen``) are those of the core's module docstring.
"""
from hazardlib.generate.clearance import (control_grid, drop_samples, layout_probes, playback_path,  # noqa: F401
                                          rejection_reason, subdivide, support_height, sweep, sweep_samples)


def host_probes(host, handle, states, onset_s, supports):
    """``{"replay": fn, "frozen": fn}`` for :func:`sweep` on a LIBERO host. ``handle`` is the allocated pool
    hazard, ``states`` the flattened physical replay states (one per control step), ``supports`` the labels
    of the support bodies."""
    dt = host.control_dt
    mem = handle.native
    geoms = set(mem.geoms)
    members = set(m.body for m in host.members)
    loaded = {"index": None}

    def index_of(t):
        return int(min(max(round(t / dt), 0), len(states) - 1))

    def overlaps(index, pos, quat):
        if loaded["index"] != index:
            host.sim.set_state_from_flattened(states[index])
            host._fresh_at = -1
            loaded["index"] = index
        host.set_hazard_pose(handle, pos, quat)
        host._collision(mem, True)
        host.mj.mj_forward(host.m, host.d)
        held = host._held_object()
        found = {}
        d, m = host.d, host.m
        for c in d.contact[:d.ncon]:
            if c.dist >= 0 or not (c.geom1 in geoms or c.geom2 in geoms):
                continue
            other = c.geom2 if c.geom1 in geoms else c.geom1
            if other in geoms:
                continue
            body = int(m.geom_bodyid[other])
            label = host.label_of_body[body]
            if (body in host.robot_body_set or body in members or label == held
                    or label in supports or host.body_name[body] in supports):
                continue
            found[label] = max(found.get(label, 0.0), -float(c.dist))
        return found

    return {"replay": lambda t, pos, quat: overlaps(index_of(t), pos, quat),
            "frozen": lambda t, pos, quat: overlaps(index_of(onset_s), pos, quat)}
