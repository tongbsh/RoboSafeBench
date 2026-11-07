"""C3: pose fidelity and no tunnelling.

* A kinematic path (translation and rotation) reads back through ``hazard_state`` as commanded, with zero velocity
  (01 §3.1: a teleport leaves no residual velocity; §3.3 item 15: wxyz quaternions).
* A kinematic intrusion into a robot body is reported in the step of the first overlapping pose (§3.3 item 7);
  the runtime's rollback pose reads back as commanded and is free of overlap (R8).
* A hazard crossing a robot body at the fastest speed the BD admits produces a contact event: with
  ``hazard_ccd: true`` (G0 then admits any speed) it crosses the whole overlap window within a fraction of a
  physics step, centred between two step ends, so only continuous detection can see it; with ``hazard_ccd:
  false`` it moves at a fraction of the R1 limit (what G0 admits). The runtime's ``r1_violation_count`` must
  match the per-step displacement (R1, D11 d).
"""
import math

import numpy as np

from hazardlib.conformance.common import (contact_clock_violations, expected_partner, still_snapshot,
                                          strike_probe)
from hazardlib.conformance.harness import body_position, path_instance, quat_angle_rad, support_radius
from hazardlib.conformance.suite import Check
from hazardlib.semantics import EventKind

SECTION = "c3_pose_fidelity"


def _pose_errors(records):
    pos = [float(np.linalg.norm(r.readback.pos - r.pos)) for r in records]
    rot = [quat_angle_rad(r.readback.quat, r.quat) for r in records]
    vel = [float(np.linalg.norm(r.readback.lin_vel_mps)) for r in records]
    return max(pos, default=0.0), max(rot, default=0.0), max(vel, default=0.0)


def _free_path(ctx, probe, env, tol):
    scenario = ctx.scenario
    speed = ctx.number("probe", "speed_mps")
    travel = ctx.number("probe", "free_travel_m")
    angle = ctx.number("probe", "rotation_rad")
    keyframe_dt = ctx.number("probe", "keyframe_dt_s")
    tail = ctx.number("probe", "tail_s")
    onset = ctx.on_grid(ctx.number("probe", "onset_s"))
    release = ctx.on_grid(onset + travel / speed)
    direction = np.array(scenario.free_direction)
    instance = path_instance("c3-free-path", scenario.probe_agent, scenario.free_point_m, direction * speed, onset,
                             release, np.zeros(3), keyframe_dt,
                             quat_at=lambda u: (math.cos(0.5 * u * angle), 0.0, 0.0, math.sin(0.5 * u * angle)))
    run = probe.run(env, instance, scenario.still, release + tail, probe.hold(env))
    records = probe.records(probe.rec.poses, run)
    ctx.require(records, "the runtime posed the hazard but no set_hazard_pose was recorded")
    pos, rot, vel = _pose_errors(records)
    ctx.record("free_path", {"poses": len(records), "max_pos_error_m": pos, "max_rot_error_rad": rot,
                             "max_readback_speed_mps": vel})
    ctx.expect(pos <= tol["pos"], "free path: hazard_state reads back %.3g m from the commanded pose (> %r)"
               % (pos, tol["pos"]))
    ctx.expect(rot <= tol["rot"], "free path: hazard_state orientation is %.3g rad off the commanded wxyz "
                                  "quaternion (> %r)" % (rot, tol["rot"]))
    ctx.expect(vel <= tol["vel"], "free path: a kinematic pose reads back with velocity %.3g m/s (> %r): "
                                  "set_hazard_pose leaves residual velocity" % (vel, tol["vel"]))


def _intrusion(ctx, probe, env, tol):
    scenario, strike, agent = ctx.scenario, ctx.scenario.strike, ctx.scenario.probe_agent
    speed = ctx.number("probe", "speed_mps")
    keyframe_dt = ctx.number("probe", "keyframe_dt_s")
    tail = ctx.number("probe", "tail_s")
    dt = ctx.physics_dt
    robot, _ = still_snapshot(ctx, probe, env)
    body = body_position(robot, strike.body)
    direction = np.array(strike.direction)
    reach = strike.radius_m + support_radius(agent, direction)
    # n free poses (room for the release window before the rollback time), then the first overlapping pose,
    # each half a pose step away from first touch
    n = int(math.ceil(ctx.runtime_cfg.release_velocity_window_s / dt)) + 1
    onset = ctx.on_grid(ctx.number("probe", "onset_s"))
    release = onset + 2 * (n + 1) * dt
    start = body - direction * (reach + speed * dt * (n + 1.5))
    instance = path_instance("c3-intrusion", agent, start, direction * speed, onset, release, direction * speed,
                             keyframe_dt)
    run = probe.run(env, instance, scenario.still, release + tail, probe.hold(env))
    expected = expected_partner(ctx, robot, strike.body)
    records = probe.records(probe.rec.poses, run)
    pos, rot, vel = _pose_errors(records)
    distances = [float(np.linalg.norm(r.pos - body)) for r in records]
    first_overlap = next((r.t_s - run.t_start + dt for r, d in zip(records, distances) if d < reach), None)
    release_info = run.info["release"] or {}
    t_rb = release_info.get("rollback_t_s")
    overlaps = [e for e in run.events if e.kind is EventKind.KINEMATIC_OVERLAP]
    clearance = None
    if t_rb is not None:
        clearance = float(np.linalg.norm(start + direction * speed * (t_rb - onset) - body))
    ctx.record("intrusion", {"poses": len(records), "max_pos_error_m": pos, "max_rot_error_rad": rot,
                             "first_overlapping_pose_hook_end_s": first_overlap,
                             "kinematic_overlap_events": len(overlaps),
                             "first_overlap_event_t_s": overlaps[0].time_s if overlaps else None,
                             "release_kind": release_info.get("kind"), "rollback_t_s": t_rb,
                             "rollback_centre_distance_m": clearance, "touch_distance_m": reach,
                             "official_partner": run.log.partner_class})
    ctx.expect(pos <= tol["pos"], "intrusion: hazard_state reads back %.3g m from a commanded pose (> %r), "
                                  "e.g. the rollback pose was not applied" % (pos, tol["pos"]))
    ctx.require(first_overlap is not None, "intrusion probe never reached the body (scenario geometry)")
    ctx.expect(overlaps, "a kinematic hazard overlapping %r was never reported as an overlap (01 §3.3 item 7)"
               % strike.body)
    if overlaps:
        ctx.expect(overlaps[0].time_s <= first_overlap + tol["clock"],
                   "the first overlap is reported at %.6f s, the first overlapping pose was posed in the step ending "
                   "at %.6f s" % (overlaps[0].time_s, first_overlap))
        ctx.expect(overlaps[0].partner_class is expected, "the overlap partner is %s, expected %s"
                   % (overlaps[0].partner_class.value, expected.value))
    ctx.expect(release_info.get("kind") == "rollback", "release kind %r after a kinematic overlap, expected "
                                                       "'rollback'" % (release_info.get("kind"),))
    if clearance is not None:
        ctx.expect(clearance >= reach - tol["clearance"],
                   "the rollback pose (t=%.6f s) is %.3g m from the body centre, inside the touch distance %.3g m: "
                   "the overlap was reported after the step that caused it" % (t_rb, clearance, reach))
    ctx.expect(run.log.partner_class is expected, "after the rollback release the official partner is %r, expected "
                                                  "%s" % (run.log.partner_class, expected.value))


def _crossing(ctx, probe, env, tol):
    scenario, strike, agent = ctx.scenario, ctx.scenario.strike, ctx.scenario.probe_agent
    dt = ctx.physics_dt
    tail = ctx.number("probe", "tail_s")
    direction = np.array(strike.direction)
    reach = strike.radius_m + support_radius(agent, direction)
    chord = 2.0 * reach
    r1_limit = ctx.runtime_cfg.max_step_ratio * min(agent.half_extents_m)
    if ctx.bd.hazard_ccd:
        speed = chord / (ctx.number(SECTION, "ccd_crossing_physics_steps") * dt)
    else:
        speed = ctx.number(SECTION, "r1_speed_fraction") * r1_limit / dt
    # start a whole number of half-and-more steps before the body centre: the centre is crossed mid-step
    n = max(0, int(math.ceil(chord / (speed * dt) - 0.5)))
    gap = speed * dt * (n + 0.5) - reach
    run, geometry = strike_probe(ctx, probe, env, strike, speed, "c3-crossing", tail, gap_m=gap)
    expected = geometry["expected_partner"]
    log = run.log
    step_displacement = speed * dt
    ctx.record("crossing", {"bd_hazard_ccd": ctx.bd.hazard_ccd, "speed_mps": speed,
                            "displacement_per_step_m": step_displacement, "overlap_window_m": chord,
                            "r1_limit_m": r1_limit, "first_official_contact_t_s": log.first_official_contact_t_s,
                            "partner_class": log.partner_class, "r1_violation_count": log.r1_violation_count})
    ctx.expect(log.first_official_contact_t_s is not None,
               "a hazard crossing %r at %.3g m/s (%.3g m per physics step, overlap window %.3g m) produced no contact"
               "%s" % (strike.body, speed, step_displacement, chord,
                       " although the BD declares hazard_ccd: true" if ctx.bd.hazard_ccd else ""))
    if log.first_official_contact_t_s is not None:
        ctx.expect(log.partner_class is expected, "crossing contact partner %r, expected %s"
                   % (log.partner_class, expected.value))
    if step_displacement > r1_limit:
        ctx.expect(log.r1_violation_count >= 1, "the hazard moved %.3g m per step (R1 limit %.3g m) but the runtime "
                                                "counted no R1 violation: hazard_state does not follow the body"
                   % (step_displacement, r1_limit))
    else:
        ctx.expect(log.r1_violation_count == 0, "the hazard moved %.3g m per step (below the R1 limit %.3g m) but the "
                                                "runtime counted %d R1 violation(s)"
                   % (step_displacement, r1_limit, log.r1_violation_count))
    violations = contact_clock_violations(probe, run, dt, tol["clock"])
    ctx.expect(not violations, "crossing: %s" % "; ".join(violations[:3]))


def run(ctx):
    tol = {"pos": ctx.number(SECTION, "pose_error_m_tol"), "rot": ctx.number(SECTION, "orientation_error_rad_tol"),
           "vel": ctx.number(SECTION, "kinematic_velocity_abs_tol_mps"),
           "clearance": ctx.number(SECTION, "rollback_clearance_abs_tol_m"),
           "clock": ctx.number("c4_partner_class", "contact_time_slack_physics_steps") * ctx.physics_dt}
    probe = ctx.probe()
    _free_path(ctx, probe, 0, tol)
    _intrusion(ctx, probe, 0, tol)
    _crossing(ctx, probe, 0, tol)


CHECK = Check("C3", "pose fidelity, rollback and no tunnelling",
              "kinematic poses read back as commanded with zero velocity; a kinematic overlap is reported in its own "
              "step and rolled back to a free pose; crossing a robot body at the fastest admissible speed yields a "
              "contact event and a consistent r1_violation_count", run)


def test_c03_pose_fidelity(conformance):
    conformance.execute(CHECK)
