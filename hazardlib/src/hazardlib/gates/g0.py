"""Gate G0: geometric validity of an instance (paradigm §8; decisions D1, D5, D11 d). Pure function.

``run_g0(instance, task_context, bd, settings)`` evaluates every check below and rejects the instance with
every reason that applies (``G0Result.reasons``). Positions are hazard centroids from the instance keyframes
(what the runtime plays back) up to the nominal hit; the workspace is the TaskContext's box widened by
``g0.workspace_margin_m``.

``onset_margin_negative``            the onset precedes the start of the nominal EEF path.
``release_height_out_of_workspace``  fall: the release point is above the workspace top.
``out_of_workspace``                 any other centroid outside the widened workspace.
``scene_penetration``                the hazard crosses a support surface by more than ``g0.max_penetration_m``.
``slide_off_support``                slide: the centroid leaves the xy extent of its support surface.
``non_closing``                      at ``t_hit`` the hazard does not approach the nominal EEF along the contact
                                     normal (closing speed ``-(v_hazard - v_eef) . n <= 0``).
``early_contact``                    some time in ``[onset, t_hit)`` the centroid is closer than
                                     ``standoff - g0.early_contact_tol_m`` to the nominal robot: the EEF, plus
                                     the robot-side collision bodies when the probe recorded them
                                     (``checks.early_contact_check`` = ``eef_only`` / ``eef_and_bodies``). The
                                     standoff along the hit normal is used for every point and direction.
``r1_step_ratio``                    (only when the BD says ``hazard_ccd: false``) a hook interval of the
                                     kinematic phase moves the hazard more than ``max_step_ratio`` x its
                                     smallest half extent (``needs_substeps`` at the BD physics step), or the
                                     predicted dynamic speed (fall ``g tau``, slide / reach the release speed)
                                     times the physics step does.

The parking pose is the host's (checked by conformance C2), not by G0. ``tau_vis`` needs the host and is
measured by the gate stage (record only, D5).
"""
import dataclasses
from typing import Any, Dict, List

import numpy as np

from .._records import is_unknown
from ..instantiate.builder import support_radius_m
from ..kinematics.hit import FAMILY_FALL, FAMILY_SLIDE
from ..kinematics.path import SampledPath
from ..kinematics.subdivide import hook_times, needs_substeps
from ..spec.benchmark_descriptor import BenchmarkDescriptor
from ..spec.instance import HazardInstance
from ..spec.task_context import TaskContext
from .settings import G0Settings

G0_REASONS = ("onset_margin_negative", "release_height_out_of_workspace", "out_of_workspace", "scene_penetration",
              "slide_off_support", "non_closing", "early_contact", "r1_step_ratio")
_UP = np.array([0.0, 0.0, 1.0])


@dataclasses.dataclass
class G0Result:
    passed: bool
    reasons: List[str]
    checks: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons), "checks": dict(self.checks)}


def _provenance(instance: HazardInstance, key: str) -> Any:
    if key not in instance.provenance:
        raise ValueError("instance %r has no provenance.%s; G0 needs instances from hazardlib.instantiate"
                         % (instance.instance_id, key))
    return instance.provenance[key]


def run_g0(instance: HazardInstance, task_context: TaskContext, bd: BenchmarkDescriptor,
           settings: G0Settings) -> G0Result:
    reasons = []  # type: List[str]
    checks = {}  # type: Dict[str, Any]
    nominal = SampledPath.from_pose_sequence(task_context.nominal_ee_path)
    frames = SampledPath(instance.keyframes.time_s, instance.keyframes.pos, None)
    onset, t_hit = instance.onset_time_s, instance.nominal_hit_time_s
    standoff = float(_provenance(instance, "standoff_m"))
    normal = np.array(_provenance(instance, "contact_normal"), dtype=np.float64)

    checks["onset_margin_s"] = onset - nominal.t_start_s
    if checks["onset_margin_s"] < 0:
        reasons.append("onset_margin_negative")

    kept = frames.times_s[frames.times_s <= t_hit]
    times = np.unique(np.concatenate([kept, [t_hit]]))
    points = frames.pos(times)
    ws = task_context.workspace_bounds
    margin = settings.workspace_margin_m
    lo, hi = np.array(ws.min_m) - margin, np.array(ws.max_m) + margin
    below, above = points < lo, points > hi
    if instance.family == FAMILY_FALL:
        release_z = float(frames.pos(onset)[2])
        checks["release_height"] = {"release_z_m": release_z, "workspace_top_m": float(hi[2])}
        if release_z > hi[2]:
            reasons.append("release_height_out_of_workspace")
        above = above.copy()
        above[:, 2] = False
    outside = int(np.sum(np.any(np.logical_or(below, above), axis=1)))
    checks["out_of_workspace_points"] = outside
    if outside:
        reasons.append("out_of_workspace")

    h_down = support_radius_m(instance.agent, -_UP)
    h_up = support_radius_m(instance.agent, _UP)
    worst = 0.0
    for surface in task_context.support_surfaces:
        top, bottom = surface.max_m[2], surface.min_m[2]
        inside = np.ones(len(points), dtype=bool)
        for axis in range(2):
            inside = np.logical_and(inside, points[:, axis] >= surface.min_m[axis])
            inside = np.logical_and(inside, points[:, axis] <= surface.max_m[axis])
        crossing = np.logical_and(inside, points[:, 2] + h_up > bottom)
        if np.any(crossing):
            worst = max(worst, float(np.max(top - (points[crossing, 2] - h_down))))
    checks["max_penetration_m"] = worst
    if worst > settings.max_penetration_m:
        reasons.append("scene_penetration")

    if instance.family == FAMILY_SLIDE:
        surface = _provenance(instance, "support_surface")
        on = all(np.all(np.logical_and(points[:, a] >= surface["min_m"][a], points[:, a] <= surface["max_m"][a]))
                 for a in range(2))
        checks["slide_on_support"] = bool(on)
        if not on:
            reasons.append("slide_off_support")

    v_hazard = np.array(instance.provenance.get("hit_velocity_mps", frames.vel(t_hit)), dtype=np.float64)
    v_eef = nominal.vel(t_hit)
    closing = float(-np.dot(v_hazard - v_eef, normal))
    checks["closing_speed_mps"] = closing
    if closing <= 0:
        reasons.append("non_closing")

    grid = [frames.times_s, nominal.times_s]
    dt = bd.timebase.physics_dt_s
    if not is_unknown(dt):
        grid.append(hook_times(onset, t_hit, dt))
    probe_t = np.unique(np.concatenate(grid))
    probe_t = probe_t[np.logical_and(probe_t >= onset, probe_t < t_hit)]
    if probe_t.size == 0:
        probe_t = np.array([onset])
    hazard = frames.pos(probe_t)
    robot = {"eef": nominal.pos(probe_t)}
    bodies = task_context.provenance.get("robot_body_paths")
    if bodies:
        for name, positions in bodies.items():
            robot[name] = SampledPath(nominal.times_s, positions, None).pos(probe_t)
    nearest = None
    for name, positions in robot.items():
        distances = np.linalg.norm(hazard - positions, axis=1)
        index = int(np.argmin(distances))
        if nearest is None or distances[index] < nearest[0]:
            nearest = (float(distances[index]), name, float(probe_t[index]))
    threshold = standoff - settings.early_contact_tol_m
    checks["early_contact_check"] = "eef_and_bodies" if bodies else "eef_only"
    checks["early_contact"] = {"min_distance_m": nearest[0], "body": nearest[1], "time_s": nearest[2],
                               "threshold_m": threshold, "samples": int(probe_t.size)}
    if nearest[0] < threshold:
        reasons.append("early_contact")

    if bd.hazard_ccd is True:
        checks["r1_check"] = "skipped: BD hazard_ccd is true"
    else:
        if is_unknown(dt):
            raise ValueError("BD %r: the R1 check needs timebase.physics_dt_s" % (bd.name,))
        limit = settings.max_step_ratio * min(instance.agent.half_extents_m)
        kinematic = False
        if instance.family != FAMILY_FALL and instance.release_time_s > onset:
            release = instance.release_time_s
            sub_t = np.unique(np.concatenate([frames.times_s[np.logical_and(frames.times_s >= onset,
                                                                            frames.times_s <= release)],
                                              [onset, release]]))
            kinematic = needs_substeps(SampledPath(sub_t, frames.pos(sub_t), None), limit, dt)
        if instance.family == FAMILY_FALL:
            dynamic_speed = float(bd.gravity_mps2) * (t_hit - onset)
        else:
            dynamic_speed = float(np.linalg.norm(instance.release_velocity.lin_mps))
        dynamic = dynamic_speed * dt > limit
        checks["r1_check"] = {"limit_m": limit, "hook_dt_s": dt, "kinematic_exceeds": bool(kinematic),
                              "dynamic_speed_mps": dynamic_speed, "dynamic_step_m": dynamic_speed * dt}
        if kinematic or dynamic:
            reasons.append("r1_step_ratio")
    return G0Result(passed=not reasons, reasons=reasons, checks=checks)
