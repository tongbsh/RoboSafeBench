"""C1: the BD timebase matches the host, and the hooks keep the clock semantics of 01 §3.3 items 1-3."""
import numpy as np

from hazardlib._records import is_unknown
from hazardlib.conformance.suite import Check

SECTION = "c1_timebase"


def _rel(measured: float, declared: float) -> float:
    """Error of a declared value relative to the measured truth."""
    return abs(declared - measured) / abs(measured)


def run(ctx):
    host, bd, env = ctx.host, ctx.bd, 0
    rel_phys = ctx.number(SECTION, "physics_dt_rel_tol")
    rel_ctrl = ctx.number(SECTION, "control_dt_rel_tol")
    sub_tol = ctx.integer(SECTION, "substeps_abs_tol")
    hook_tol_steps = ctx.number(SECTION, "hook_time_tol_physics_steps")
    scene_tol = ctx.number(SECTION, "begin_scene_abs_tol_m")
    n_steps = ctx.integer(SECTION, "control_steps")
    timebase = bd.timebase
    unknown = [k for k in ("physics_dt_s", "control_dt_s", "substeps_per_action") if is_unknown(getattr(timebase, k))]
    ctx.require(not unknown, "BD leaves timebase.%s unknown" % ", timebase.".join(unknown))

    probe = ctx.probe()
    # A moving episode first: a reset hook fired before the scene reset would then see the end of this one.
    probe.run(env, None, ctx.scenario.motion, ctx.scenario.motion_duration_s, ctx.target.motion_action,
              control_id="c1-motion")
    first_mark = len(probe.tap.marks)
    run_ = probe.start(env, None, ctx.scenario.still, control_id="c1-still")
    robot_after_reset = host.robot_state(env)
    objects_after_reset = host.object_poses(env)
    control_dts, substeps = [], []
    for _ in range(n_steps):
        t_before = host.time_state(env).sim_time_s
        n_before = len(probe.tap.marks)
        probe.step(env, host.hold_action(env))
        control_dts.append(host.time_state(env).sim_time_s - t_before)
        substeps.append(sum(1 for m in probe.tap.marks[n_before:] if m.env == env and m.kind == "pre"))
    probe.finish(env)

    marks = [m for m in probe.tap.marks[first_mark:] if m.env == env]
    ctx.require(marks and marks[0].kind == "begin", "reset_episode did not fire the reset hook first")
    hooks = marks[1:]
    kinds = [m.kind for m in hooks]
    ctx.require(kinds == ["pre", "post"] * (len(kinds) // 2) and kinds,
                "the step hooks do not alternate pre_step / post_step (got %s ...)" % kinds[:6])
    pre = np.array([m.sim_time_s for m in hooks[0::2]])
    post = np.array([m.sim_time_s for m in hooks[1::2]])
    measured_dt = float(np.median(post - pre))
    ctx.require(measured_dt > 0, "the clock does not advance inside the step hooks (post - pre = %r)" % measured_dt)
    hook_tol = hook_tol_steps * measured_dt
    reported_dt = float(np.median([m.physics_dt_s for m in hooks]))
    reported_ctrl = float(np.median([m.control_dt_s for m in hooks]))
    measured_ctrl = float(np.median(control_dts))
    ctx.record("measured_physics_dt_s", measured_dt)
    ctx.record("measured_control_dt_s", measured_ctrl)
    ctx.record("measured_substeps", sorted(set(substeps)))
    ctx.record("declared", {"physics_dt_s": timebase.physics_dt_s, "control_dt_s": timebase.control_dt_s,
                            "substeps_per_action": timebase.substeps_per_action})
    ctx.record("time_state", {"physics_dt_s": reported_dt, "control_dt_s": reported_ctrl})

    # declared vs measured (C1 proper)
    ctx.expect(_rel(measured_dt, timebase.physics_dt_s) <= rel_phys,
               "BD physics_dt_s=%.6g but the hooks measure %.6g (rel. error %.3g > %r)"
               % (timebase.physics_dt_s, measured_dt, _rel(measured_dt, timebase.physics_dt_s), rel_phys))
    ctx.expect(_rel(measured_ctrl, timebase.control_dt_s) <= rel_ctrl,
               "BD control_dt_s=%.6g but one step() advances the clock by %.6g (rel. error %.3g > %r)"
               % (timebase.control_dt_s, measured_ctrl, _rel(measured_ctrl, timebase.control_dt_s), rel_ctrl))
    ctx.expect(all(abs(s - timebase.substeps_per_action) <= sub_tol for s in substeps),
               "BD substeps_per_action=%r but step() wraps %s physics steps in hooks"
               % (timebase.substeps_per_action, sorted(set(substeps))))
    # time_state() is what the runtime uses for half-step rounding and R1 (item 1)
    ctx.expect(_rel(measured_dt, reported_dt) <= rel_phys,
               "time_state().physics_dt_s=%.6g but the hooks measure %.6g" % (reported_dt, measured_dt))
    ctx.expect(_rel(measured_ctrl, reported_ctrl) <= rel_ctrl,
               "time_state().control_dt_s=%.6g but step() advances the clock by %.6g" % (reported_ctrl, measured_ctrl))
    # item 1: pre_step at t0, post_step at t0 + dt, one hook pair per physics step, back to back
    spread = float(np.max(np.abs((post - pre) - measured_dt)))
    gaps = pre[1:] - post[:-1]
    ctx.record("hook_dt_spread_s", spread)
    ctx.expect(spread <= hook_tol, "post_step - pre_step varies by %.3g s (> %.3g s)" % (spread, hook_tol))
    ctx.expect(gaps.size == 0 or float(np.max(np.abs(gaps))) <= hook_tol,
               "consecutive physics steps are not contiguous on the hook clock (max gap %.3g s)"
               % (float(np.max(np.abs(gaps))) if gaps.size else 0.0))
    # item 3: the episode zero is the clock at begin_episode, and the first step starts there
    offset = pre[0] - marks[0].sim_time_s
    ctx.record("first_pre_step_minus_begin_s", float(offset))
    ctx.expect(abs(offset) <= hook_tol, "the first pre_step starts %.3g s after begin_episode (> %.3g s): the reset "
                                        "advanced the clock after the reset hook" % (offset, hook_tol))
    ctx.expect(abs(run_.samples[-1].t_s - n_steps * measured_ctrl) <= hook_tol * n_steps,
               "episode time after %d steps is %.6g s, expected %.6g s" % (n_steps, run_.samples[-1].t_s,
                                                                           n_steps * measured_ctrl))
    # item 2: begin_episode sees the scene after the reset
    begin = run_.begin_scene
    eef_gap = float(np.max(np.abs(begin["eef"] - robot_after_reset.eef_pose.pos)))
    ctx.record("begin_scene_eef_diff_m", eef_gap)
    ctx.expect(eef_gap <= scene_tol, "begin_episode saw the EEF %.3g m away from its pose after the reset: the reset "
                                     "hook fires before the scene reset is complete" % eef_gap)
    ctx.expect(set(begin["objects"]) == set(objects_after_reset),
               "begin_episode saw objects %s, after the reset there are %s"
               % (sorted(begin["objects"]), sorted(objects_after_reset)))
    for name in sorted(set(begin["objects"]) & set(objects_after_reset)):
        gap = float(np.max(np.abs(begin["objects"][name][0] - objects_after_reset[name].pos)))
        ctx.expect(gap <= scene_tol, "begin_episode saw object %r %.3g m away from its pose after the reset"
                   % (name, gap))


CHECK = Check("C1", "BD timebase and hook clock",
              "measured physics_dt / control_dt / substeps match the BD and time_state(); hooks wrap every physics "
              "step back to back with pre at t0 and post at t0 + dt; begin_episode runs after the scene reset at "
              "the episode zero (01 §3.3 items 1-3)", run)


def test_c01_timebase(conformance):
    conformance.execute(CHECK)
