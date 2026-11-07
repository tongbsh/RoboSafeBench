"""C13: nominal hit timing (D1). The robot replays the scripted motion; hazards built with ``build_hit_path`` on the
known nominal EEF path, with ``standoff = r_hazard + eef_contact_radius_m``, must make their first official contact
within ``hit_time_abs_tol_s`` of the nominal ``t_hit`` (residuals reported). Instances come from the suite's own D1
builder and, when the target provides it, from the instance builder used on this host
(``ConformanceTarget.build_hit_instance``).

The motion must be repeatable: two replays are compared first; if they differ by more than
``replay_repeat_abs_tol_m`` the timing cannot be verified (status ``unverified``, not ``pass``). Without a declared
nominal path the recorded EEF trace of the first replay is the nominal path. The tolerance must cover at least one
physics step of the host (01 §5); a smaller configured value fails the check.
"""
import numpy as np

from hazardlib.conformance.harness import build_nominal_hit_instance
from hazardlib.conformance.suite import Check
from hazardlib.host import PartnerClass
from hazardlib.kinematics.hit import FAMILY_FALL, FAMILY_REACH, FAMILY_SLIDE
from hazardlib.kinematics.path import SampledPath

SECTION = "c13_nominal_hit"
ROBOT_SIDE = (PartnerClass.ROBOT_LINK, PartnerClass.HELD_OBJECT)


def _params(ctx, family):
    # the release window of slide / reach is the runtime's R7 window (kinematics.hit), not a C13 setting
    direction = list(ctx.scenario.motion_hit_direction)
    window = ctx.runtime_cfg.release_velocity_window_s
    if family == FAMILY_FALL:
        return {"g": ctx.bd.gravity_mps2}
    if family == FAMILY_SLIDE:
        return {"direction": direction, "speed_mps": ctx.number(SECTION, "slide_speed_mps"),
                "release_window_s": window}
    if family == FAMILY_REACH:
        return {"direction": direction, "reach_distance_m": ctx.number(SECTION, "reach_distance_m"),
                "overshoot_m": ctx.number(SECTION, "reach_overshoot_m"),
                "speed_profile": ctx.value(SECTION, "reach_speed_profile"),
                "release_window_s": window}
    raise ValueError("c13_nominal_hit.families: unsupported family %r" % (family,))


def _trace(probe, run):
    times, pos, _ = probe.robot_trace(run)
    begin = run.begin_scene["eef"]
    return np.concatenate([[0.0], times]), np.vstack([begin[None, :], pos])


def run(ctx):
    scenario, env, target = ctx.scenario, 0, ctx.target
    tol = ctx.number(SECTION, "hit_time_abs_tol_s")
    repeat_tol = ctx.number(SECTION, "replay_repeat_abs_tol_m")
    tau = ctx.number(SECTION, "tau_s")
    tail = ctx.number(SECTION, "tail_s")
    families = ctx.value(SECTION, "families")
    keyframe_dt = ctx.number("probe", "keyframe_dt_s")
    dt = ctx.physics_dt
    ctx.record("physics_dt_s", dt)
    ctx.record("release_velocity_window_s", ctx.runtime_cfg.release_velocity_window_s)
    ctx.require(tol >= dt, "c13_nominal_hit.hit_time_abs_tol_s = %r is below one physics step (%r) of this host; "
                           "01 §5 requires it to cover at least one" % (tol, dt))
    probe = ctx.probe()
    replays = [probe.run(env, None, scenario.motion, scenario.motion_duration_s, target.motion_action,
                         control_id="c13-replay-%d" % i) for i in range(2)]
    traces = [_trace(probe, r) for r in replays]
    same_shape = traces[0][1].shape == traces[1][1].shape
    repeat_error = float(np.max(np.abs(traces[0][1] - traces[1][1]))) if same_shape else float("inf")
    ctx.record("replay_repeat_max_abs_diff_m", repeat_error)
    if not repeat_error <= repeat_tol:
        ctx.unverified("the scripted motion is not repeatable (two replays differ by %.3g m > %r), so the nominal "
                       "hit time cannot be verified" % (repeat_error, repeat_tol))
    if target.nominal_eef_path is not None:
        nominal = target.nominal_eef_path
        times, pos = traces[0]
        ctx.record("declared_path_vs_replay_max_m", float(np.max(np.linalg.norm(nominal.pos(times) - pos, axis=1))))
        ctx.record("nominal_path_source", "declared by the target")
    else:
        nominal = SampledPath(traces[0][0], traces[0][1], None)
        ctx.record("nominal_path_source", "recorded EEF trace of the first replay")

    builders = [("suite", lambda **kw: build_nominal_hit_instance(
        nominal_eef_path=nominal, eef_contact_radius_m=ctx.bd.eef_contact_radius_m, **kw))]
    if target.build_hit_instance is not None:
        builders.append(("target", target.build_hit_instance))
    else:
        ctx.note("the target provides no build_hit_instance: only the suite's D1 builder was checked")
    residuals = {}
    for family in families:
        params = _params(ctx, family)
        for name, builder in builders:
            label = "%s/%s" % (name, family)
            instance = builder(instance_id="c13-%s-%s" % (name, family.split(".")[-1]), family=family,
                               agent=scenario.probe_agent, t_hit_s=scenario.motion_hit_time_s, tau_s=tau,
                               params=params, keyframe_dt_s=keyframe_dt)
            run_ = probe.run(env, instance, scenario.motion, instance.nominal_hit_time_s + tail, target.motion_action)
            log = run_.log
            first = log.first_official_contact_t_s
            residual = None if first is None else first - instance.nominal_hit_time_s
            residuals[label] = {"residual_s": residual, "nominal_hit_time_s": instance.nominal_hit_time_s,
                                "first_official_contact_t_s": first, "partner_class": log.partner_class,
                                "release_kind": (run_.info["release"] or {}).get("kind"),
                                "kinematic_overlaps": log.kinematic_overlap_count,
                                "standoff_m": instance.provenance.get("standoff_m")}
            if not ctx.expect(first is not None, "%s: no official contact (nominal hit at %.4f s)"
                              % (label, instance.nominal_hit_time_s)):
                continue
            ctx.expect(log.partner_class in ROBOT_SIDE, "%s: first official partner %r" % (label, log.partner_class))
            ctx.expect(abs(residual) < tol, "%s: first official contact %.5f s, nominal t_hit %.5f s, residual %+.5f s "
                                            "(|.| >= %r): re-measure eef_contact_radius_m or fix the builder's "
                                            "standoff, do not widen the tolerance"
                       % (label, first, instance.nominal_hit_time_s, residual, tol))
    ctx.record("hits", residuals)


CHECK = Check("C13", "nominal hit timing",
              "on a repeatable scripted motion, hazards built on the nominal EEF path with the D1 standoff first "
              "touch the robot within hit_time_abs_tol_s of t_hit (suite builder and the host's builder)", run)


def test_c13_nominal_hit(conformance):
    conformance.execute(CHECK)
