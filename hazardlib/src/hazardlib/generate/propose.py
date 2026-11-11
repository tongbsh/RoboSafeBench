"""Hazard proposals of the perturbation paradigm (D20): one function of the two input variables.

``propose`` reads only V1 (the successful trajectories: robot geometry over time, interaction segments, timing),
V2 (the 3D scene layout: every non-robot body with its collision boxes and pose over time, supports), the
settings (YAML, no defaults in code), ``T_stop`` and what the builder needs (``agent``, gravity, and the slide
support rule of :mod:`hazardlib.instantiate`). It knows no benchmark and no simulator.

One **cell** is (family, entry-direction class, Sigma target); a family's classes are its elevations
(:class:`~hazardlib.generate.settings.EntrySettings`): the hazard moves down along
``(cos el cos az, cos el sin az, -sin el)`` with ``az`` taken in the initial frame of the segment's entity
(``direction_frame: segment_entity``; ``demo_first_entity`` is the L2 frame) and ``el = 0`` from the side; a
fall is vertical and has no direction. Per cell:

* **Anchors** are quantiles of the low-clearance samples of every interaction segment (the lower half by robot
  to neighbour clearance, computed from V1 points x V2 boxes unless ``risk`` is injected); a slide anchors the
  real finger surface point and only where ``slide_support`` holds.
* **tau** of a Sigma target is ``Sigma * T_stop`` rounded down to the control step (:func:`tau_for_sigma`),
  never clamped: an anchor whose available time before the hit (since the segment start) or whose free-fall
  height budget is shorter than tau offers nothing for that target, and a target no anchor can serve is
  ``info["unreachable"]`` (the most frequent drop reason; every count in ``info["dropped"]``). A reach whose
  start point would lie outside the workspace is dropped (``reach_start_outside_workspace``) when
  ``reach_start_m`` (stroke length before the hit for a speed and tau) is given.
* **Coverage (D20 b)** of a hit point ``p`` (segment ``s``, entity ``e``): the fraction of the OTHER
  demonstrations that have segment ``s`` on the same entity and whose robot surface points (thinned to
  ``thinned_point_spacing_m``; the EEF is not used) come within ``coverage_radius_m`` of ``p`` mapped into their
  own initial ``e`` frame during that segment. It is recorded in the evidence; with ``coverage_ordering`` the
  candidates of a bucket (direction x segment) are ordered by coverage first, and the budgeted list is stably
  re-sorted by coverage (the shortlist of the host sorts by it too).
* **Approach speed** of a slide / reach (``speed_source``): ``demo`` is the median of the per-segment demo speed
  levels (a benchmark-derived intensity); ``paradigm`` is ``paradigm_approach_speed_mps[family]``, the same for
  every benchmark (D20: the hazard's physical intensity is not read from the demonstrations). Tau stays
  ``Sigma * T_stop``, so the stroke ``speed * tau`` grows with the speed; a reach whose start leaves the
  workspace is dropped, a slide's start is judged by G0 (``out_of_workspace``) and the scene sweep.
* **Budget**: round-robin over (azimuth, segment) buckets, then ``max_candidates_per_family_layout``.
"""
import collections
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..kinematics.hit import FAMILY_FALL, FAMILY_REACH, FAMILY_SLIDE
from .geometry import points_boxes_distance, quat_to_mat
from .inputs import Segment, Trajectory, TrajectorySet
from .settings import GenerateSettings, cell_id, elevation_tag

Cell = collections.namedtuple("Cell", "family elevation_rad sigma candidates info")

PRIMITIVE = {FAMILY_FALL: "free_fall", FAMILY_SLIDE: "slide", FAMILY_REACH: "reach"}


def target_key(sigma: float) -> str:
    """Text key of a Sigma target (``0.75``, ``1.5``, ``3``): JSON key, file-name tag, spec-id part."""
    return "%g" % sigma


def tau_for_sigma(sigma: float, t_stop_s: float, dt: float, minimum_steps: int,
                  step_tolerance: float) -> Tuple[Optional[float], Optional[str]]:
    """``(tau_s, None)`` for a Sigma target: ``Sigma * T_stop`` rounded DOWN to whole control steps (the achieved
    Sigma is therefore <= the target; ``step_tolerance`` absorbs the round-off of ``Sigma * T_stop / dt``), or
    ``(None, "below_minimum_lead")`` when that is fewer than ``minimum_steps`` steps."""
    steps = int(math.floor(sigma * t_stop_s / dt + step_tolerance))
    if steps < minimum_steps:
        return None, "below_minimum_lead"
    return steps * dt, None


def entry_direction(yaw: float, azimuth_index: int, azimuth_count: int, elevation_rad: float) -> np.ndarray:
    """Unit direction of motion: azimuth ``yaw + 2 pi i / n`` about +z, ``elevation_rad`` below the horizontal."""
    az = yaw + 2 * math.pi * azimuth_index / azimuth_count
    return np.array([math.cos(elevation_rad) * math.cos(az), math.cos(elevation_rad) * math.sin(az),
                     0.0 - math.sin(elevation_rad)])


def frame_yaw(trajectory: Trajectory, entity: str) -> float:
    """Yaw of an entity's initial frame, projected into the horizontal plane (a small roll / pitch of the body
    must not tilt the surface motion)."""
    _, rotation = trajectory.entity_frame(entity, 0)
    return math.atan2(rotation[1, 0], rotation[0, 0])


def thin_points(points, spacing: float) -> np.ndarray:
    """One surface sample per ``spacing`` voxel (the first in order)."""
    points = np.asarray(points, dtype=np.float64)
    keys = np.floor(points / spacing).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def robot_points_world(trajectory: Trajectory, local_points: Mapping[str, np.ndarray], steps: Sequence[int],
                       bodies: Optional[Sequence[str]] = None) -> np.ndarray:
    """World-frame robot surface points ``(len(steps), N, 3)`` of the chosen bodies at the given samples."""
    out = []
    for i, name in enumerate(trajectory.body_names):
        if bodies is not None and name not in bodies:
            continue
        rot = quat_to_mat(trajectory.body_quat[list(steps), i])                       # (S, 3, 3)
        out.append(np.einsum("np,sqp->snq", local_points[name], rot) + trajectory.body_pos[list(steps), i][:, None, :])
    return np.concatenate(out, axis=1)


def finger_low_points(trajectory: Trajectory, robot_points: Mapping[str, np.ndarray], indices,
                      prefixes: Sequence[str]) -> np.ndarray:
    """The lowest real finger / gripper surface sample at each control index (world frame)."""
    names = [(i, n) for i, n in enumerate(trajectory.body_names) if n.startswith(tuple(prefixes))]
    out = []
    for index in indices:
        cloud = np.concatenate([np.asarray(robot_points[n]).dot(quat_to_mat(trajectory.body_quat[index, i]).T) +
                                trajectory.body_pos[index, i] for i, n in names])
        out.append(cloud[int(np.argmin(cloud[:, 2]))])
    return np.asarray(out).reshape(-1, 3)


def surface_top_m(layout) -> float:
    return float(max(s["max_m"][2] for s in layout.support_surfaces))


def finger_table_gap(trajectory: Trajectory, layout, robot_points: Mapping[str, np.ndarray], spacing: float,
                     prefixes: Sequence[str]) -> np.ndarray:
    """Height of the lowest gripper surface sample above the highest support top, per control sample."""
    local = {n: thin_points(robot_points[n], spacing) for n in trajectory.body_names if n.startswith(tuple(prefixes))}
    world = robot_points_world(trajectory, local, range(len(trajectory.time_s)), list(local))
    return world[:, :, 2].min(axis=1) - surface_top_m(layout)


def robot_neighbour_clearance(trajectory: Trajectory, layout, robot_points: Mapping[str, np.ndarray],
                              settings: GenerateSettings) -> np.ndarray:
    """Per control sample: the smallest distance from the gripper's surface samples (thinned) to the collision
    boxes of the ``neighbour`` bodies of V2, capped at ``risk_search_m``."""
    prefixes = tuple(settings.gripper_body_prefixes)
    local = {n: thin_points(robot_points[n], settings.thinned_point_spacing_m)
             for n in trajectory.body_names if n.startswith(prefixes)}
    steps = list(range(len(trajectory.time_s)))
    world = robot_points_world(trajectory, local, steps, list(local))
    out = np.full(len(steps), float(settings.risk_search_m))
    for k in steps:
        centres, rots, halves, _ = layout.boxes_at(k, ("neighbour",))
        if len(halves):
            out[k] = min(out[k], float(points_boxes_distance(world[k], centres, rots, halves).min()))
    return out


def phase_speed_levels(trajectories: Sequence[Trajectory], settings: GenerateSettings) -> Dict[str, List[float]]:
    """Per segment name: quantiles (``speed_quantiles``) of the per-trajectory median EEF speed in that segment
    (one vote per trajectory). Names no trajectory has a moving sample in get ``[]``."""
    medians = {}  # type: Dict[str, List[float]]
    for t in trajectories:
        speeds = np.linalg.norm(np.diff(t.eef_pos, axis=0) / np.diff(t.time_s)[:, None], axis=1)
        for s in t.segments:
            part = speeds[s.start_index:s.end_index]
            medians.setdefault(s.name, [])
            if len(part):
                medians[s.name].append(float(np.median(part)))
    return {name: ([float(np.quantile(m, q)) for q in settings.speed_quantiles] if m else [])
            for name, m in medians.items()}


class CoverageIndex:
    """Coverage of a hit point by the other demonstrations (module docstring); clouds are cached per
    (demonstration, segment)."""

    def __init__(self, trajectory_set: TrajectorySet, settings: GenerateSettings) -> None:
        self.set = trajectory_set
        self.radius = float(settings.coverage_radius_m)
        self.local = {n: thin_points(p, settings.thinned_point_spacing_m)
                      for n, p in trajectory_set.robot_points.items()}
        self._clouds = {}  # type: Dict[Tuple[str, str], np.ndarray]

    def _cloud(self, trajectory: Trajectory, segment: Segment) -> np.ndarray:
        key = (trajectory.demo, segment.name)
        if key not in self._clouds:
            steps = list(range(segment.start_index, segment.end_index + 1))
            world = robot_points_world(trajectory, self.local, steps).reshape(-1, 3)
            origin, rotation = trajectory.entity_frame(segment.entity, 0)
            self._clouds[key] = (world - origin).dot(rotation)
        return self._clouds[key]

    def score(self, trajectory: Trajectory, segment: Segment, point_world) -> Tuple[Optional[float], int]:
        """``(coverage, n_other)``: ``coverage`` is ``None`` when no other demonstration has the segment."""
        origin, rotation = trajectory.entity_frame(segment.entity, 0)
        p = (np.asarray(point_world, dtype=np.float64) - origin).dot(rotation)
        hits = total = 0
        for other in self.set.trajectories:
            if other.demo == trajectory.demo:
                continue
            match = next((s for s in other.segments if s.name == segment.name and s.entity == segment.entity), None)
            if match is None:
                continue
            total += 1
            cloud = self._cloud(other, match)
            hits += bool(float(np.min(np.sum((cloud - p) ** 2, axis=1))) <= self.radius ** 2)
        return (None if total == 0 else hits / total), total


def propose(trajectory: Trajectory, trajectory_set: TrajectorySet, settings: GenerateSettings,
            workspace: Mapping[str, Any], t_stop_s: Optional[float], agent: Mapping[str, Any],
            gravity_mps2: float, slide_support: Optional[Callable[[Any, float], bool]] = None,
            reach_start_m: Optional[Callable[[float, float], float]] = None, risk: Optional[Any] = None,
            finger_gap: Optional[Any] = None, speed_levels: Optional[Mapping[str, Sequence[float]]] = None,
            coverage: Optional[CoverageIndex] = None):
    """Per family, entry-direction class and Sigma target: a :data:`Cell` ``(family, elevation_rad, sigma,
    candidates, info)``; ``candidates`` are ``(spec dict, evidence dict)``, ``info`` has ``sigma_target``,
    ``t_stop_s``, ``tau_s``, ``unreachable``, ``dropped``, ``n_proposed``, ``elevation_rad``, ``cell_id``.

    ``trajectory`` is one demonstration of ``trajectory_set`` (its layout is ``trajectory_set.layouts``);
    ``workspace`` is ``{"min_m", "max_m"}``. ``t_stop_s=None`` (censored / failed protocol, D6 / D8) makes every
    target unreachable. Injectable overrides (regression against recorded host data): ``risk`` (per sample robot
    to neighbour clearance), ``finger_gap`` (per sample finger-to-table gap), ``speed_levels`` (per segment
    name)."""
    layout = trajectory_set.layouts[trajectory.demo]
    robot_points = trajectory_set.robot_points
    dt = trajectory.control_dt_s
    min_steps = settings.minimum_lead_control_steps
    tol = settings.step_tolerance_fraction
    times = trajectory.time_s
    levels_by_name = (phase_speed_levels(trajectory_set.trajectories, settings) if speed_levels is None
                      else {k: list(v) for k, v in speed_levels.items()})
    if coverage is None and settings.coverage_radius_m:
        coverage = CoverageIndex(trajectory_set, settings)
    state = {}  # type: Dict[str, Any]

    def risk_array():
        if "risk" not in state:
            state["risk"] = (np.asarray(risk, dtype=float) if risk is not None else
                             robot_neighbour_clearance(trajectory, layout, robot_points, settings))
        return state["risk"]

    def gap_array():
        if "gap" not in state:
            state["gap"] = (np.asarray(finger_gap, dtype=float) if finger_gap is not None else
                            finger_table_gap(trajectory, layout, robot_points, settings.thinned_point_spacing_m,
                                             settings.gripper_body_prefixes))
        return state["gap"]

    def yaw_of(segment):
        entity = segment.entity if settings.direction_frame == "segment_entity" else (
            trajectory.first_entity or trajectory.entity_names[0])
        return frame_yaw(trajectory, entity)

    for family in settings.families:
        if family == FAMILY_SLIDE and slide_support is None:
            raise ValueError("a slide proposal needs slide_support (hazardlib.instantiate.slide_support_rule "
                             "of the TaskContext)")
        anchors_found = []      # (segment, anchor index, finger offset, speed levels, speed, coverage record)
        for segment in trajectory.segments:
            if segment.name not in levels_by_name:
                continue
            indices = np.arange(segment.start_index + min_steps, segment.end_index)
            if not len(indices):
                continue
            lows = None
            if family == FAMILY_SLIDE:
                # Only instants where the builder finds a support surface under the finger (the hit time of a
                # slide is where the finger is low enough above it for a box riding on the table to touch it).
                lows = finger_low_points(trajectory, robot_points, indices, settings.gripper_body_prefixes)
                supported = np.array([bool(slide_support(point, times[i])) for point, i in zip(lows, indices)])
                indices, lows = indices[supported], lows[supported]
                if not len(indices):
                    continue
                clearance = gap_array()[indices]
            else:
                clearance = risk_array()[indices]
            cutoff = np.quantile(clearance, .5)
            keep = clearance <= cutoff
            indices = indices[keep]
            lows = None if lows is None else lows[keep]
            positions = sorted(set(int(math.floor(q * (len(indices) - 1))) for q in settings.anchor_quantiles))
            levels = levels_by_name[segment.name]
            if settings.speed_source == "paradigm" and family != FAMILY_FALL:
                speed = settings.paradigm_approach_speed_mps[family]
            else:
                speed = float(np.median(levels)) if len(levels) else 0.0
            for position in positions:
                anchor = int(indices[position])
                offset = np.zeros(3)
                if lows is not None:
                    # Real finger sample at the selected instant; its z and x/y differ from the EEF origin.
                    offset = lows[position] - trajectory.eef_pos[anchor]
                point = trajectory.eef_pos[anchor] + offset
                cover = None if coverage is None else coverage.score(trajectory, segment, point)
                anchors_found.append((segment, anchor, offset, levels, speed, cover))
        for elevation in settings.elevations(family):
            azimuths = 1 if family == FAMILY_FALL else settings.entry_directions[family].azimuth_count
            for sigma in settings.sigma_targets:
                info = {"sigma_target": sigma, "t_stop_s": t_stop_s, "tau_s": None, "unreachable": None,
                        "dropped": {}, "n_proposed": 0, "elevation_rad": elevation,
                        "cell_id": cell_id(family, elevation)}
                dropped = info["dropped"]

                def drop(why, n):
                    dropped[why] = dropped.get(why, 0) + n
                if t_stop_s is None:
                    info["unreachable"] = "t_stop_censored"
                    yield Cell(family, elevation, sigma, [], info)
                    continue
                tau, why = tau_for_sigma(sigma, t_stop_s, dt, min_steps, tol)
                info["tau_s"] = tau
                if tau is None:
                    drop(why, max(len(anchors_found), 1))
                elif not anchors_found:
                    drop("no_supported_anchor" if family == FAMILY_SLIDE else "no_anchor", 1)
                offered = []
                for segment, anchor, offset, levels, speed, cover in (anchors_found if tau is not None else []):
                    t_hit = float(times[anchor])
                    if tau > t_hit - segment.t_start_s + tol * dt:
                        drop("phase_time", 1)
                        continue
                    if family == FAMILY_FALL:
                        # Conservative release budget above the entire observed gripper.
                        free_height = max(0., workspace["max_m"][2] - trajectory.eef_pos[anchor, 2] -
                                          np.max(agent["half_extents_m"]))
                        if tau > math.sqrt(2 * free_height / gravity_mps2) + tol * dt:
                            drop("free_fall_height", 1)
                            continue
                    if speed <= 0:
                        drop("no_speed", 1)
                        continue
                    key = target_key(sigma)
                    yaw = None if family == FAMILY_FALL else yaw_of(segment)
                    for di in range(azimuths):
                        direction = None
                        if family != FAMILY_FALL:
                            direction = entry_direction(yaw, di, azimuths, elevation)
                            if family == FAMILY_REACH and reach_start_m is not None:
                                start = trajectory.eef_pos[anchor] - direction * reach_start_m(speed, tau)
                                if np.any(start < np.asarray(workspace["min_m"])) or \
                                        np.any(start > np.asarray(workspace["max_m"])):
                                    drop("reach_start_outside_workspace", 1)
                                    continue
                        ei = 0 if elevation is None else settings.elevations(family).index(elevation)
                        ident = "%s.%s.%s.k%d.sig%s.d%d%s" % (trajectory.demo, family, segment.name, anchor, key, di,
                                                              "" if ei == 0 else ".e%d" % ei)
                        params = {} if direction is None else {"direction": direction.tolist()}
                        intensity = {"tau_s": tau}
                        if family != FAMILY_FALL:
                            intensity["approach_speed_mps"] = speed
                        if family == FAMILY_REACH:
                            params["speed_profile"] = "min_jerk"
                        if family in settings.follow_through:
                            params["follow_through"] = settings.follow_through[family]
                        spec = {"spec_version": "1", "id": ident, "family": family, "agent": dict(agent),
                                "motion": {"primitive": PRIMITIVE[family], "params": params},
                                "exposure": {"anchor": "nominal_ee_path", "offset_m": offset.tolist(),
                                             "anchor_phase": {"phase": segment.name,
                                                              "frac": (t_hit - segment.t_start_s) / segment.duration_s}},
                                "timing": {"mode": "nominal_time"}, "intensity": intensity,
                                "release": {"mode": "dynamic_release"}}
                        evidence = {"phase": segment.name, "segment_entity": segment.entity, "anchor_step": anchor,
                                    "source": "all-demo segment speed quantiles + individual low-clearance "
                                              "replay samples",
                                    "speed_levels_mps": list(levels), "speed_source": settings.speed_source,
                                    "sigma_target": sigma, "t_stop_s": t_stop_s,
                                    "tau_s": tau, "available_before_hit_s": t_hit - segment.t_start_s,
                                    "risk_clearance_m": float(risk_array()[anchor]),
                                    "finger_table_gap_m": float(gap_array()[anchor]),
                                    "direction_frame": "initial frame of the %s" % (
                                        "segment entity" if settings.direction_frame == "segment_entity"
                                        else "demonstration's first entity"),
                                    "direction_index": di, "elevation_rad": elevation,
                                    "elevation_class": elevation_tag(elevation),
                                    "coverage": None if cover is None else cover[0],
                                    "coverage_n_demos": None if cover is None else cover[1],
                                    "coverage_radius_m": settings.coverage_radius_m,
                                    "coverage_source": "robot surface points of the other demonstrations, "
                                                       "same segment and entity"}
                        offered.append((spec, evidence))
                info["n_proposed"] = len(offered)
                if not offered:
                    info["unreachable"] = max(sorted(dropped), key=lambda k: dropped[k]) if dropped else "no_anchor"
                # Round-robin over direction x segment, then expand anchors. A finite budget must not be
                # exhausted by the first direction alone.
                buckets = {}
                for proposal in offered:
                    buckets.setdefault((proposal[1]["direction_index"], proposal[1]["phase"]), []).append(proposal)

                def covered(proposal):
                    c = proposal[1]["coverage"]
                    return -(c if c is not None else 0.0) if settings.coverage_ordering else 0.0
                for bucket in buckets.values():
                    bucket.sort(key=lambda p: (covered(p), p[1]["risk_clearance_m"], p[1]["anchor_step"]))
                ordered = []
                depth = 0
                while any(depth < len(b) for b in buckets.values()):
                    for key in sorted(buckets):
                        if depth < len(buckets[key]):
                            ordered.append(buckets[key][depth])
                    depth += 1
                ordered = ordered[:settings.max_candidates_per_family_layout]
                if settings.coverage_ordering:
                    ordered.sort(key=covered)
                yield Cell(family, elevation, sigma, ordered, info)
