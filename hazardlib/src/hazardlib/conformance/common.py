"""Probe constructions shared by several checks (T05). Generic: geometry comes from the target's ``Scenario``,
magnitudes from ``conformance.yaml: probe``.

Statistical hosts (BD ``same_seed_repro: statistical``, 01 §5, D13 h): C2 / C7 / C12 do not require identity with the
hazard-free run there. Each observation kind is compared with C8's bound of that kind (``c8_determinism``: state
vectors; images, every camera on its own, in grey levels); a bound that is still PLACEHOLDER leaves the kind measured
but not judged (``unverified``). ``task_success`` must still agree with the hazard-free run when it agreed between
C8's same-seed repeats; when those repeats disagree on it, it is not judged. Hosts that declare ``exact`` keep the
identity checks of each check.
"""
import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..host import PartnerClass, RobotState
from ..semantics import classify_partner
from .harness import (EpisodeRun, Probe, body_position, merge_max, observation_diff, strike_instance,
                      support_radius)
from .target import StrikeTarget

PROBE = "probe"
C8_SECTION = "c8_determinism"
STATISTICAL_VECTOR_BOUND = "statistical_state_vector_max_abs_diff"
STATISTICAL_IMAGE_BOUND = "statistical_image_max_abs_diff"
_REPEATS = "c8_same_seed_repeats"


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


# ------------------------------------------------------------------------------------- statistical hosts

def statistical_repro(ctx: Any) -> bool:
    return ctx.bd.determinism.same_seed_repro == "statistical"


@dataclasses.dataclass
class SameSeedRepeats:
    """C8's two same-seed resets of ``scenario.motion`` replayed with ``motion_action``: the largest difference per
    state-vector key and per camera, and the sample times at which their ``task_success`` differs."""
    vectors: Dict[str, float]
    images: Dict[str, float]
    oracle_differs_at_s: List[float]


def same_seed_repeats(ctx: Any, probe: Optional[Probe] = None) -> SameSeedRepeats:
    """Measured once per suite run (``SuiteSession.shared``) on ``probe`` or a new one. A new probe takes over the
    host's hooks, so call this before building the check's own probe, or pass that probe."""
    cached = ctx.session.shared.get(_REPEATS)
    if cached is not None:
        return cached
    scenario = ctx.scenario
    probe = probe if probe is not None else ctx.probe()
    runs = [probe.run(0, None, scenario.motion, scenario.motion_duration_s, ctx.target.motion_action,
                      control_id="c8-repeat-%d" % i) for i in range(2)]
    ctx.require(len(runs[0].samples) == len(runs[1].samples), "the same-seed repeats have different lengths")
    vectors, images, oracle = {}, {}, []  # type: Dict[str, float], Dict[str, float], List[float]
    for a, b in zip(runs[0].samples, runs[1].samples):
        diff = observation_diff(a.obs, b.obs)
        ctx.require(not diff.mismatch, "observation keys or shapes differ between same-seed repeats: %s"
                    % diff.mismatch)
        merge_max(vectors, diff.vectors)
        merge_max(images, diff.images)
        if a.success != b.success:
            oracle.append(round(b.t_s, 6))
    repeats = SameSeedRepeats(vectors, images, oracle)
    ctx.session.shared[_REPEATS] = repeats
    return repeats


def judge_statistical_kinds(ctx: Any, what: str, vectors: Dict[str, float],
                            images: Optional[Dict[str, float]]) -> None:
    """Compare ``vectors`` (largest difference per state-vector key) and ``images`` (per camera, grey levels; None
    when the check does not compare images) with C8's per-kind bounds; ``what`` names the compared runs."""
    vector_max = max(vectors.values(), default=0.0)
    ctx.record("statistical", {"state_vectors": vectors, "state_vector_max_abs_diff": vector_max,
                               "images": images})
    if vectors:
        bound = ctx.number_if_set(C8_SECTION, STATISTICAL_VECTOR_BOUND)
        if bound is None:
            ctx.unjudged("measured, not judged: %s: state vectors differ by %.3g, but %s.%s is not set (statistical "
                         "host)" % (what, vector_max, C8_SECTION, STATISTICAL_VECTOR_BOUND))
        else:
            ctx.expect(vector_max <= bound, "%s: state vectors differ by %.3g, above the statistical bound %r (%s.%s)"
                       % (what, vector_max, bound, C8_SECTION, STATISTICAL_VECTOR_BOUND))
    if images:
        bound = ctx.number_if_set(C8_SECTION, STATISTICAL_IMAGE_BOUND)
        shown = ", ".join("%s %.3g" % (camera, d) for camera, d in sorted(images.items()))
        if bound is None:
            ctx.unjudged("measured, not judged: %s: images differ by up to %s grey levels per camera, but %s.%s is "
                         "not set (statistical host)" % (what, shown, C8_SECTION, STATISTICAL_IMAGE_BOUND))
        else:
            over = sorted(camera for camera, d in images.items() if d > bound)
            ctx.expect(not over, "%s: camera(s) %s differ by more than the statistical bound %r grey levels (%s.%s; "
                                 "per camera: %s)" % (what, over, bound, C8_SECTION, STATISTICAL_IMAGE_BOUND, shown))


def judge_statistical_oracle(ctx: Any, repeats: SameSeedRepeats, what: str, oracle_differs_at_s: List[float],
                             end_of_episode: Tuple[Any, Any]) -> None:
    """``task_success`` of a hazard run against its hazard-free control (sample times at which it differs; the two
    end-of-episode values): required equal when C8's same-seed repeats agree on it, otherwise not judged."""
    ctx.record("statistical_oracle", {"repeats_oracle_differs_at_s": repeats.oracle_differs_at_s[:10],
                                      "oracle_differs_at_s": oracle_differs_at_s[:10],
                                      "end_of_episode": list(end_of_episode)})
    if repeats.oracle_differs_at_s:
        ctx.unjudged("task_success not judged: same-seed repeats (C8) disagree on it at t = %s s (%s: %s)"
                     % (repeats.oracle_differs_at_s[:5], what,
                        "differs at t = %s s" % oracle_differs_at_s[:5] if oracle_differs_at_s else "agrees"))
        return
    ctx.expect(not oracle_differs_at_s, "%s: task_success differs at t = %s s, although same-seed repeats (C8) agree "
                                        "on it" % (what, oracle_differs_at_s[:5]))
    ctx.expect(end_of_episode[0] == end_of_episode[1],
               "%s: end-of-episode task_success %r vs %r, although same-seed repeats (C8) agree on it"
               % (what, end_of_episode[0], end_of_episode[1]))
