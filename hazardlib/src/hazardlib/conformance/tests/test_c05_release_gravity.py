"""C5: release and gravity.

* A hazard moved kinematically along the free direction and released *backwards* reads back, right after
  ``release_to_physics``, exactly the commanded velocity: not one derived from the last kinematic poses, and
  without residual teleport velocity added (01 §3.1, §3.3 item 8, R7).
* A hazard released at rest falls freely: the gravity fitted to its ``hazard_state`` heights matches the BD's
  ``gravity_mps2``, and the landing time on the support surface matches ``sqrt(2 h / g_measured)`` (zero damping,
  R7; the BD has no damping field, so zero is what is checked).
"""
import math

import numpy as np

from hazardlib.conformance.harness import drop_instance, path_instance, support_radius
from hazardlib.conformance.suite import Check

SECTION = "c5_release_gravity"


def _release(ctx, probe, tol):
    scenario = ctx.scenario
    u = ctx.number("probe", "speed_mps")
    v = ctx.number("probe", "release_speed_mps")
    travel = ctx.number("probe", "free_travel_m")
    onset = ctx.on_grid(ctx.number("probe", "onset_s"))
    release = ctx.on_grid(onset + travel / u)
    direction = np.array(scenario.free_direction)
    commanded = -direction * v
    instance = path_instance("c5-release", scenario.probe_agent, scenario.free_point_m, direction * u, onset,
                             release, commanded, ctx.number("probe", "keyframe_dt_s"))
    run = probe.run(0, instance, scenario.still, release + ctx.number("probe", "tail_s"), probe.hold(0))
    releases = probe.records(probe.rec.releases, run)
    ctx.require(len(releases) == 1, "expected one release_to_physics call, got %d" % len(releases))
    rel = releases[0]
    sent = float(np.max(np.abs(rel.lin_vel - commanded)))
    error = float(np.linalg.norm(rel.readback.lin_vel_mps - commanded))
    info = run.info["release"] or {}
    ctx.record("release", {"kind": info.get("kind"), "commanded_mps": commanded,
                           "readback_mps": rel.readback.lin_vel_mps, "readback_error_mps": error,
                           "kinematic_speed_before_mps": u})
    ctx.expect(info.get("kind") == "nominal", "release kind %r, expected 'nominal'" % (info.get("kind"),))
    ctx.expect(sent == 0.0, "the runtime sent %r instead of the instance velocity %r" % (rel.lin_vel, commanded))
    ctx.expect(error <= tol["velocity"], "hazard_state right after release_to_physics reads %s m/s, commanded %s "
                                         "(error %.3g > %r)" % (np.round(rel.readback.lin_vel_mps, 6),
                                                                np.round(commanded, 6), error, tol["velocity"]))


def _free_fall(ctx, probe, tol):
    scenario, agent = ctx.scenario, ctx.scenario.probe_agent
    up = np.array([0.0, 0.0, 1.0])
    height = scenario.drop_point_m[2] - scenario.drop_surface_height_m - support_radius(agent, up)
    g_declared = ctx.bd.gravity_mps2
    nominal_fall = math.sqrt(2.0 * height / g_declared)
    onset = ctx.on_grid(ctx.number("probe", "onset_s"))
    instance = drop_instance("c5-drop", agent, scenario.drop_point_m, onset, nominal_fall)
    run = probe.run(0, instance, scenario.still, onset + 2.0 * nominal_fall, probe.hold(0))
    releases = probe.records(probe.rec.releases, run)
    ctx.require(len(releases) == 1, "expected one release_to_physics call, got %d" % len(releases))
    t_release = releases[0].t_s
    landing = run.log.first_scene_contact_t_s
    ctx.require(landing is not None, "the dropped hazard never touched the support surface")
    t_land_abs = run.t_start + landing
    samples = [(s.t_s - t_release, float(s.state.pos[2])) for s in probe.records(probe.rec.states, run)
               if t_release < s.t_s < t_land_abs]
    ctx.require(len(samples) >= 3, "only %d hazard_state samples during the fall" % len(samples))
    t = np.array([s[0] for s in samples])
    z = np.array([s[1] for s in samples])
    coeffs = np.polyfit(t, z, 2)
    g_measured = -2.0 * float(coeffs[0])
    ctx.require(g_measured > 0, "the fitted gravity is %r: the released hazard does not fall" % g_measured)
    fall_measured = t_land_abs - t_release
    fall_expected = math.sqrt(2.0 * height / g_measured)
    g_error = abs(g_declared - g_measured) / g_measured
    t_error = abs(fall_measured - fall_expected) / fall_expected
    ctx.record("free_fall", {"height_m": height, "samples": len(samples), "g_measured_mps2": g_measured,
                             "g_declared_mps2": g_declared, "g_rel_error": g_error,
                             "fall_time_measured_s": fall_measured, "fall_time_expected_s": fall_expected,
                             "fall_time_rel_error": t_error,
                             "fit_residual_max_m": float(np.max(np.abs(np.polyval(coeffs, t) - z)))})
    ctx.expect(g_error <= tol["gravity"], "free fall measures g = %.6g m/s^2, the BD declares %.6g (rel. error %.3g > "
                                          "%r)" % (g_measured, g_declared, g_error, tol["gravity"]))
    ctx.expect(t_error <= tol["fall_time"], "landing after %.6g s, sqrt(2h/g_measured) = %.6g s (rel. error %.3g > "
                                            "%r): damping or a contact offset" % (fall_measured, fall_expected, t_error,
                                                                                  tol["fall_time"]))


def run(ctx):
    tol = {"velocity": ctx.number(SECTION, "release_velocity_abs_tol_mps"),
           "gravity": ctx.number(SECTION, "gravity_rel_tol"),
           "fall_time": ctx.number(SECTION, "free_fall_time_rel_tol")}
    probe = ctx.probe()
    _release(ctx, probe, tol)
    _free_fall(ctx, probe, tol)


CHECK = Check("C5", "release velocity and gravity",
              "hazard_state right after release reads the commanded velocity; free-fall gravity matches the BD; "
              "landing time matches sqrt(2h/g_measured)", run)


def test_c05_release_gravity(conformance):
    conformance.execute(CHECK)
