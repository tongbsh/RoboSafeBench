"""Scene clearance of a hazard's own trajectory (D19 c): does it pass through or land on task objects?

Two layers.

* A **pure core** (numpy only, no simulator): the kinematic playback path of an instance
  (:func:`playback_path`: ``keyframes`` followed by ``exit_keyframes``, the junction sample counted once, as
  the runtime plays it), the sample times of a sweep (:func:`sweep_samples`, :func:`drop_samples`) and the
  verdict over an overlap callback (:func:`sweep`). The callback maps ``(t_s, pos, quat)`` to
  ``{partner: penetration depth in m}`` for one scene state; the core only applies the tolerance.
* A **host layer** (:func:`host_probes`): the callbacks for a LIBERO host. A pool hazard is placed with its
  collision on (as ``visibility_probe`` does) in saved physical replay states and every MuJoCo contact with
  ``dist < 0`` is read; robot links, the object the robot holds in that state, the support bodies (the
  table the hazard rides on or lands on) and pool members are ignored.

Two scene states are checked, because the benchmark cannot know what the model under test does:

* ``replay`` - the scene as the successful robot leaves it: the saved replay state of time ``t``, clamped to
  the last state after the replay ends;
* ``frozen`` - the replay state at the hazard's onset, held: the scene of a robot that stopped when the
  hazard appeared.

A fall has no kinematic path; its release pose is dropped straight down onto the support surface
(:func:`drop_samples`) and the landing pose is also checked against the last replay state (the scene the
successful robot leaves behind, which the debris then occupies).
"""
import math

import numpy as np

from hazardlib.kinematics.path import SampledPath


def playback_path(instance):
    """The path the runtime plays: ``keyframes`` + ``exit_keyframes[1:]`` (the junction pose is shared)."""
    frames = instance.keyframes
    times, pos, quat = list(frames.time_s), list(frames.pos), list(frames.quat)
    exit_ = instance.exit_keyframes
    if exit_ is not None:
        times += list(exit_.time_s[1:])
        pos += list(exit_.pos[1:])
        quat += list(exit_.quat[1:])
    return SampledPath(times, pos, quat)


def control_grid(t_start, t_end, dt, eps=1e-9):
    """``t_start``, every multiple of ``dt`` inside ``[t_start, t_end]``, and ``t_end`` (no duplicates)."""
    first = int(math.ceil(t_start / dt - eps))
    last = int(math.floor(t_end / dt + eps))
    times = [t_start] + [k * dt for k in range(first, last + 1)] + [t_end]
    out = []
    for t in sorted(times):
        if not out or t - out[-1] > eps:
            out.append(float(t))
    return np.asarray(out)


def subdivide(times, pos_fn, spacing):
    """Insert samples (uniform in time) so that consecutive samples are at most ``spacing`` apart."""
    out = [float(times[0])]
    for a, b in zip(times[:-1], times[1:]):
        gap = float(np.linalg.norm(np.asarray(pos_fn(b), dtype=float) - np.asarray(pos_fn(a), dtype=float)))
        pieces = max(1, int(math.ceil(gap / spacing - 1e-12)))
        out.extend(float(a + (b - a) * k / pieces) for k in range(1, pieces))
        out.append(float(b))
    return np.asarray(out)


def sweep_samples(path, t_start, t_end, dt, spacing):
    """``[(t, pos, quat)]`` along ``path`` from ``t_start`` to ``t_end``: every control step, the end, and
    enough extra samples that consecutive positions are at most ``spacing`` apart."""
    times = subdivide(control_grid(t_start, t_end, dt), path.pos, spacing)
    quats = path.quat(times) if path.has_orientation else None
    out = []
    for i, t in enumerate(times):
        quat = np.array([1., 0., 0., 0.]) if quats is None else np.asarray(quats[i], dtype=float)
        out.append((float(t), np.asarray(path.pos(float(t)), dtype=float), quat))
    return out


def drop_samples(release_pos, rest_z, gravity, t_release, spacing, t_after=None):
    """Vertical drop of a box centre from ``release_pos`` to ``rest_z`` (the centre height at rest on the
    support), at most ``spacing`` apart, each with its free-fall time ``t_release + sqrt(2 dz / g)``. With
    ``t_after`` the landing pose is repeated at that time (the scene state at the end of the replay)."""
    release_pos = np.asarray(release_pos, dtype=float)
    z0 = float(release_pos[2])
    z_rest = min(float(rest_z), z0)
    pieces = max(1, int(math.ceil((z0 - z_rest) / spacing - 1e-12)))
    quat = np.array([1., 0., 0., 0.])
    out = []
    for k in range(pieces + 1):
        z = z0 - (z0 - z_rest) * k / pieces
        pos = np.array([release_pos[0], release_pos[1], z])
        out.append((float(t_release + math.sqrt(2.0 * (z0 - z) / gravity)), pos, quat))
    if t_after is not None and t_after > out[-1][0]:
        out.append((float(t_after), out[-1][1].copy(), quat))
    return out


def sweep(samples, probes, tol_m):
    """Verdict of ``samples`` against every named probe.

    ``probes`` maps a scene-state name to ``fn(t, pos, quat) -> {partner: depth_m}``; a partner counts when
    its depth exceeds ``tol_m``. Returns ``passed``, the sorted union of ``partners``, the earliest offending
    time ``first_t_s`` with its scene ``state`` (ties: probe order) and ``by_state`` (partners with their
    largest depth, and the first offending time, per state that offended)."""
    by_state = {}
    for name, probe in probes.items():
        partners, first = {}, None
        for t, pos, quat in samples:
            for partner, depth in probe(t, pos, quat).items():
                if depth > tol_m:
                    partners[partner] = max(partners.get(partner, 0.0), float(depth))
                    first = t if first is None else first
        if partners:
            by_state[name] = {"partners": {p: round(d, 6) for p, d in sorted(partners.items())},
                              "first_t_s": float(first)}
    if not by_state:
        return {"passed": True, "partners": [], "first_t_s": None, "state": None, "by_state": {}}
    state = min(by_state, key=lambda s: by_state[s]["first_t_s"])
    return {"passed": False, "partners": sorted({p for v in by_state.values() for p in v["partners"]}),
            "first_t_s": by_state[state]["first_t_s"], "state": state, "by_state": by_state}


def rejection_reason(result):
    return "scene_clearance:" + ",".join(result["partners"])


# ------------------------------------------------------------------------------------------- host layer

def support_height(surfaces, xy):
    """Top of the highest support surface whose horizontal footprint contains ``xy``; else the highest top."""
    tops = [s["max_m"][2] for s in surfaces
            if s["min_m"][0] <= xy[0] <= s["max_m"][0] and s["min_m"][1] <= xy[1] <= s["max_m"][1]]
    return max(tops) if tops else max(s["max_m"][2] for s in surfaces)


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
