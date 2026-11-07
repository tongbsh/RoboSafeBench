"""C10: hold action and T_stop. Under ``hold_action`` the robot does not drift; issued during the scripted motion,
the hold stops the EEF reproducibly: over ``repeats`` repetitions no T_stop is censored and the inter-quartile
range stays within the tolerance (paradigm 00 §7; T_stop math from ``calibrate.stop``)."""
import numpy as np

from hazardlib.calibrate.stop import aggregate_t_stop, estimate_t_stop
from hazardlib.conformance.suite import Check

SECTION = "c10_hold_t_stop"


def run(ctx):
    host, scenario, env = ctx.host, ctx.scenario, 0
    drift_tol = ctx.number(SECTION, "drift_m_tol")
    hold_duration = ctx.number(SECTION, "hold_duration_s")
    repeats = ctx.integer(SECTION, "repeats")
    iqr_tol = ctx.number(SECTION, "t_stop_iqr_s_tol")
    record_s = ctx.number(SECTION, "t_stop_record_s")
    probe = ctx.probe()

    still = probe.run(env, None, scenario.still, hold_duration, probe.hold(env), control_id="c10-hold")
    _, eef, _ = probe.robot_trace(still)
    start = still.begin_scene["eef"]
    drift = float(np.max(np.linalg.norm(eef - start, axis=1))) if eef.size else 0.0
    ctx.record("hold_drift_m", drift)
    ctx.expect(drift <= drift_tol, "under hold_action the EEF drifts %.3g m in %.3g s (> %r)"
               % (drift, hold_duration, drift_tol))

    threshold, k, sources = ctx.config.t_stop_settings(ctx.bd.timebase.substeps_per_action)
    ctx.record("t_stop_settings", sources)
    control_dt = host.time_state(env).control_dt_s
    t_hold = round(scenario.motion_hold_time_s / control_dt) * control_dt
    actions = lambda t: ctx.target.motion_action(t) if t < t_hold - 0.5 * control_dt else host.hold_action(env)  # noqa
    samples, speeds_at_hold = [], []
    for i in range(repeats):
        run_ = probe.run(env, None, scenario.motion, t_hold + record_s, actions, control_id="c10-t-stop-%d" % i)
        times, _, vel = probe.robot_trace(run_)
        speeds = np.linalg.norm(vel, axis=1)
        before = speeds[times <= t_hold]
        speeds_at_hold.append(float(before[-1]) if before.size else 0.0)
        samples.append(estimate_t_stop(times, speeds, threshold, k, t_hold))
    summary = aggregate_t_stop(samples)
    ctx.record("t_hold_s", t_hold)
    ctx.record("speed_at_hold_mps", speeds_at_hold)
    ctx.record("t_stop_samples_s", samples)
    ctx.record("t_stop_summary", {"n_total": summary.n_total, "n_valid": summary.n_valid,
                                  "n_censored": summary.n_censored, "median_s": summary.median_s,
                                  "q1_s": summary.q1_s, "q3_s": summary.q3_s, "iqr_s": summary.iqr_s})
    ctx.require(min(speeds_at_hold) > threshold,
                "scenario: the EEF moves at %.3g m/s when the hold is issued (threshold %r), so T_stop is not "
                "measured on a moving robot" % (min(speeds_at_hold), threshold))
    ctx.expect(summary.n_censored == 0, "%d of %d repetitions never stopped within %.3g s of the hold"
               % (summary.n_censored, summary.n_total, record_s))
    if summary.iqr_s is not None:
        ctx.expect(summary.iqr_s <= iqr_tol, "T_stop IQR %.3g s over %d repetitions (> %r)"
                   % (summary.iqr_s, summary.n_valid, iqr_tol))


CHECK = Check("C10", "hold action and T_stop",
              "hold_action keeps the EEF within drift_m_tol; T_stop after a hold during the scripted motion is never "
              "censored and its IQR over the repeats is within t_stop_iqr_s_tol", run)


def test_c10_hold_t_stop(conformance):
    conformance.execute(CHECK)
