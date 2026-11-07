"""Probe constructions shared by several checks (T05). Generic: geometry comes from the target's ``Scenario``,
magnitudes from ``conformance.yaml: probe``."""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..host import PartnerClass, RobotState
from ..semantics import classify_partner
from .harness import EpisodeRun, Probe, body_position, strike_instance, support_radius
from .target import StrikeTarget

PROBE = "probe"


def robot_side_names(robot: RobotState) -> List[str]:
    if robot.held_object_bodies is not None:
        return list(robot.held_object_bodies)
    return [] if robot.held_object is None else [robot.held_object]


def expected_partner(ctx: Any, robot: RobotState, body: str) -> PartnerClass:
    bodies = ctx.bd.robot_contact_bodies
    return classify_partner(body, list(bodies.include), list(bodies.exclude), robot_side_names(robot), [])


def still_snapshot(ctx: Any, probe: Probe, env: int) -> Tuple[RobotState, Dict[str, Any]]:
    """Reset to ``scenario.still`` in a one-control-step control episode and return the robot state and object
    poses right after the reset. It also moves the host clock past zero (contact-clock checks need it)."""
    ref = ctx.scenario.still
    probe.start(env, None, ref, control_id="conformance-still-snapshot")
    try:
        robot = ctx.host.robot_state(env)
        poses = ctx.host.object_poses(env)
        probe.step(env, ctx.host.hold_action(env))
    finally:
        probe.finish(env)
    return robot, poses


def flight_start(body_pos: np.ndarray, strike: StrikeTarget, agent: Any, gap_m: float) -> np.ndarray:
    """Centroid ``gap_m`` before first touch on the line through the body along ``strike.direction``."""
    direction = np.array(strike.direction)
    reach = strike.radius_m + support_radius(agent, direction)
    return body_pos - direction * (reach + gap_m)


def strike_probe(ctx: Any, probe: Probe, env: int, strike: StrikeTarget, speed_mps: float, instance_id: str,
                 tail_s: float, gap_m: Optional[float] = None) -> Tuple[EpisodeRun, Dict[str, Any]]:
    """A hazard parked until ``probe.onset_s``, still at a gap before ``strike.body`` until
    ``onset + probe.exposure_s``, then released at ``speed_mps`` along ``strike.direction`` while the robot holds
    still. The default gap is ``probe.strike_gap_physics_steps`` of flight, so the first touch falls mid-step."""
    robot, _ = still_snapshot(ctx, probe, env)
    agent = ctx.scenario.probe_agent
    dt = ctx.physics_dt
    if gap_m is None:
        gap_m = speed_mps * dt * ctx.number(PROBE, "strike_gap_physics_steps")
    onset = ctx.on_grid(ctx.number(PROBE, "onset_s"))
    release = ctx.on_grid(onset + ctx.number(PROBE, "exposure_s"))
    body = body_position(robot, strike.body)
    start = flight_start(body, strike, agent, gap_m)
    velocity = np.array(strike.direction) * speed_mps
    # the hazard leaves its pose one physics step before release_time (D12 b)
    hit_time = release - dt + gap_m / speed_mps
    instance = strike_instance(instance_id, agent, start, velocity, onset, release, hit_time,
                               body - np.array(strike.direction) * strike.radius_m)
    run = probe.run(env, instance, ctx.scenario.still, hit_time + tail_s, probe.hold(env))
    geometry = {"body_pos": body, "start": start, "gap_m": gap_m, "speed_mps": speed_mps,
                "release_time_s": release, "expected_hit_time_s": hit_time,
                "expected_partner": expected_partner(ctx, robot, strike.body)}
    return run, geometry


def contact_clock_violations(probe: Probe, run: EpisodeRun, dt: float, slack_s: float) -> List[str]:
    """01 §3.3 items 4 and 6: every contact returned by ``hazard_contacts`` carries an absolute time inside the
    physics step that just ended, ``[t1 - dt, t1]`` with ``t1`` the clock at the call (to within ``slack_s``)."""
    out = []
    for record in probe.records(probe.rec.contacts, run):
        for obs in record.contacts:
            if not (record.t_s - dt - slack_s <= obs.time_s <= record.t_s + slack_s):
                out.append("contact with %r stamped %.6f s returned by the call at %.6f s (step [%.6f, %.6f])"
                           % (obs.partner_name, obs.time_s, record.t_s, record.t_s - dt, record.t_s))
    return out


def first_index(values: List[bool]) -> Optional[int]:
    for i, value in enumerate(values):
        if value:
            return i
    return None


def angle_between(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.acos(max(-1.0, min(1.0, cos)))


def oracle_note(times_s: List[float]) -> str:
    """Message suffix naming the sample times at which two runs' task_success differ (empty if none)."""
    return "; the task oracle differs at t = %s s" % times_s[:5] if times_s else ""
