"""C6: relative normal speed and analytic impulse.

A hazard moved kinematically toward a still robot body at the probe speed and then released at the (different)
strike speed hits it head-on. The runtime's ``v_rel_normal_mps`` (centre-of-mass velocity of the last hook before
the first contact, D10 c) must equal the strike speed and ``impulse_analytic_ns`` must equal ``m (1 + e) v`` (R9).
It fails when the contact is seen one hook late (post-impact velocity), when the release ignores the commanded
velocity, when a teleport leaves residual velocity, or when the collision bodies carry no velocities / names
(01 §3.3 item 9: then ``v_n`` is ``None``). Skipped on Tier B hosts (01 §5).
"""
import numpy as np

from hazardlib.conformance.common import expected_partner, flight_start, still_snapshot
from hazardlib.conformance.harness import body_position, path_instance
from hazardlib.conformance.suite import Check
from hazardlib.host import Tier

SECTION = "c6_relative_velocity"


def _skip(session):
    # an undetermined tier (unknown BD fields) is not Tier B: C6 runs
    if session.tier is Tier.B:
        return "Tier B host: no reliable contact information, severity is not reported (01 §5 allows skipping C6)"
    return None


def run(ctx):
    scenario, strike, agent = ctx.scenario, ctx.scenario.strike, ctx.scenario.probe_agent
    vn_tol = ctx.number(SECTION, "normal_speed_rel_tol")
    j_tol = ctx.number(SECTION, "impulse_rel_tol")
    u = ctx.number("probe", "speed_mps")
    v = ctx.number("probe", "strike_speed_mps")
    gap_steps = ctx.number("probe", "strike_gap_physics_steps")
    exposure = ctx.number("probe", "exposure_s")
    dt = ctx.physics_dt
    probe = ctx.probe()
    robot, _ = still_snapshot(ctx, probe, 0)
    body = body_position(robot, strike.body)
    direction = np.array(strike.direction)
    gap = v * dt * gap_steps
    onset = ctx.on_grid(ctx.number("probe", "onset_s"))
    release = ctx.on_grid(onset + exposure)
    # the hazard leaves p(release - dt) (D12 b): put that pose `gap` before first touch
    start = flight_start(body, strike, agent, gap) - direction * u * (release - dt - onset)
    instance = path_instance("c6-strike", agent, start, direction * u, onset, release, direction * v,
                             ctx.number("probe", "keyframe_dt_s"))
    hit_time = release - dt + gap / v
    run_ = probe.run(0, instance, scenario.still, hit_time + ctx.number("probe", "tail_s"), probe.hold(0))
    log = run_.log
    wanted = expected_partner(ctx, robot, strike.body)
    restitution = ctx.runtime_cfg.restitution_e
    j_expected = agent.mass_kg * (1.0 + restitution) * v
    ctx.record("strike", {"approach_speed_mps": u, "strike_speed_mps": v, "expected_hit_time_s": hit_time,
                          "first_official_contact_t_s": log.first_official_contact_t_s,
                          "partner_class": log.partner_class, "v_rel_normal_mps": log.v_rel_normal_mps,
                          "impulse_analytic_ns": log.impulse_analytic_ns, "impulse_expected_ns": j_expected,
                          "restitution_e": restitution})
    ctx.require(log.first_official_contact_t_s is not None, "the strike on %r produced no official contact"
                % strike.body)
    ctx.expect(log.partner_class is wanted, "official partner %r, expected %s" % (log.partner_class, wanted.value))
    ctx.require(log.v_rel_normal_mps is not None,
                "v_rel_normal_mps is None: the struck body %r has no velocity in RobotState (names / "
                "collision_body_lin_vel_mps, 01 §3.3 item 9) or no hook preceded the contact" % strike.body)
    vn_error = abs(log.v_rel_normal_mps - v) / v
    ctx.record("v_rel_normal_rel_error", vn_error)
    ctx.expect(vn_error <= vn_tol, "v_rel_normal_mps = %.6g, the strike speed is %.6g (rel. error %.3g > %r)"
               % (log.v_rel_normal_mps, v, vn_error, vn_tol))
    ctx.require(log.impulse_analytic_ns is not None, "impulse_analytic_ns is None although v_rel_normal_mps is set")
    j_error = abs(log.impulse_analytic_ns - j_expected) / j_expected
    ctx.record("impulse_rel_error", j_error)
    ctx.expect(j_error <= j_tol, "impulse_analytic_ns = %.6g, m (1 + e) v = %.6g (rel. error %.3g > %r)"
               % (log.impulse_analytic_ns, j_expected, j_error, j_tol))


CHECK = Check("C6", "relative normal speed and analytic impulse",
              "a head-on strike at a known speed on a still robot body gives v_rel_normal_mps = that speed (previous "
              "hook's velocity, D10 c) and impulse_analytic_ns = m (1 + e) v", run, skip_when=_skip)


def test_c06_relative_velocity(conformance):
    conformance.execute(CHECK)
