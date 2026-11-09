"""Probe: what ``hold_action`` does on CALVIN, next to the T_stop protocol (H1 item 5: target-actual lag).

CALVIN integrates relative actions into a target pose (``use_target_pose``, ``robot.py:228-237``) that the IK
follows, so a zero relative action holds the *target*, not the robot: at the hold the robot still has to cover
the distance between its TCP and the target. For every TaskContext given, the demo replay runs like the T_stop
protocol (hold from the control step after the EEF speed first reaches ``trigger_fraction`` x the context's
nominal speed, for ``record_s``) and the probe records, per physics step, the TCP speed and position and the
integrated target ``Robot.target_pos``:

* ``speed_hold``: EEF speed at the last control step before the hold (what the protocol triggered on);
* ``lag``: ``|target_pos - tcp|`` when the hold starts (the distance the robot still travels);
* ``drift``: how far the TCP moves after the hold starts (final minus start);
* ``tstop_ctrl``: first control step from which ``threshold`` holds for ``k`` samples (the protocol's
  estimate, resolution 1/30 s); ``tstop_phys``: the same on the physics-step speeds (1/240 s, k x 8 samples);
* ``lag/v``: ``lag / speed_hold``, the time a robot moving on at that speed would need to cover the lag.

    python -m hazardhosts.calvin.probes.hold_lag --egl-device 2 --threshold 0.0172 --k 1 \
        --trigger-fraction 0.8 --record-s 1.0 CONTEXT.yaml [...]
"""
import sys

import numpy as np

from hazardhosts.calvin.probes.common import probe_args


def _t_stop(speeds, dt, threshold, k):
    for i in range(len(speeds) - k + 1):
        if all(s < threshold for s in speeds[i:i + k]):
            return (i + 1) * dt
    return None


def main() -> int:
    parser = probe_args(__doc__)
    parser.add_argument("contexts", nargs="+")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--trigger-fraction", type=float, required=True)
    parser.add_argument("--record-s", type=float, required=True)
    args = parser.parse_args()
    from hazardlib.pipeline_io import read_task_context
    from hazardhosts.calvin.host import CalvinHost
    from hazardhosts.calvin.policies import DemoReplay
    host = CalvinHost(dataset_dir=args.dataset_dir, egl_device=args.egl_device)
    robot = host.env.robot
    phys = []

    def tcp():
        state = host.p.getLinkState(robot.robot_uid, robot.tcp_link_id, computeLinkVelocity=1,
                                    computeForwardKinematics=1, physicsClientId=host.cid)
        return np.array(state[4]), float(np.linalg.norm(state[6]))

    original = host.after_physics_step

    def after(env):
        original(env)
        pos, speed = tcp()
        phys.append((pos, speed))

    host.after_physics_step = after
    rows = []
    print("%-42s %7s %7s %7s %7s %7s %8s %8s %7s" % ("context", "v_nom", "v_hold", "lag_mm", "drift_mm", "lag/v",
                                                    "tstop_c", "tstop_p", "lag_end"))
    try:
        for path in args.contexts:
            tc = read_task_context(path)
            trigger = args.trigger_fraction * float(tc.provenance["eef_speed_median_mps"])
            host.reset_episode(0, tc.task_id, tc.layout_id, 0)
            policy = DemoReplay(host)
            policy.reset(tc.task_id, tc.layout_id, 0)
            horizon = int(round((float(tc.phases[-1].t_end_s) + args.record_s) / host.control_dt))
            hold_from, hold_steps, record = None, int(round(args.record_s / host.control_dt)), None
            speed_ctrl = []
            for step in range(horizon):
                if hold_from is None:
                    action = policy.act(host.policy_observation(0))
                else:
                    action = host.hold_action(0)
                if hold_from is not None and record is None:
                    pos0 = tcp()[0]
                    record = dict(lag=float(np.linalg.norm(np.array(robot.target_pos) - pos0)), pos0=pos0,
                                  phys0=len(phys))
                host.step(0, action)
                speed = float(np.linalg.norm(host.robot_state(0).eef_lin_vel_mps))
                if hold_from is not None:
                    speed_ctrl.append(speed)
                    if len(speed_ctrl) >= hold_steps:
                        break
                elif speed >= trigger:
                    hold_from, v_hold = step + 1, speed
            if record is None:
                print("%-42s never reached the trigger %.4f m/s" % (tc.task_id, trigger))
                continue
            phys_hold = phys[record["phys0"]:]
            drift = float(np.linalg.norm(phys_hold[-1][0] - record["pos0"]))
            lag_end = float(np.linalg.norm(np.array(robot.target_pos) - phys_hold[-1][0]))
            t_c = _t_stop(speed_ctrl, host.control_dt, args.threshold, args.k)
            t_p = _t_stop([s for _, s in phys_hold], host.physics_dt, args.threshold, args.k * 8)
            name = path.rsplit("/", 1)[-1][:-5]
            row = dict(name=name, v_nom=float(tc.provenance["eef_speed_median_mps"]), v_hold=v_hold,
                       lag=record["lag"], drift=drift, t_c=t_c, t_p=t_p, lag_end=lag_end)
            rows.append(row)
            print("%-42s %7.4f %7.4f %7.2f %7.2f %7.3f %8s %8s %7.2f" % (
                name, row["v_nom"], v_hold, 1e3 * row["lag"], 1e3 * drift, row["lag"] / v_hold,
                "%.4f" % t_c if t_c is not None else "-", "%.4f" % t_p if t_p is not None else "-",
                1e3 * lag_end))
    finally:
        host.close()
    if rows:
        def stats(key, scale=1.0):
            vals = np.array([r[key] for r in rows if r[key] is not None]) * scale
            return "median %.4f IQR [%.4f, %.4f] min %.4f max %.4f (n %d)" % (
                np.median(vals), np.percentile(vals, 25), np.percentile(vals, 75), vals.min(), vals.max(), len(vals))
        print("lag_mm    ", stats("lag", 1e3))
        print("drift_mm  ", stats("drift", 1e3))
        print("tstop_ctrl", stats("t_c"))
        print("tstop_phys", stats("t_p"))
        both = [(r["lag"] / r["v_hold"], r["t_p"]) for r in rows if r["t_p"] is not None]
        if len(both) > 2:
            a, b = np.array(both).T
            print("corr(lag/v_hold, tstop_phys) %.3f; corr(v_hold, tstop_phys) %.3f" % (
                np.corrcoef(a, b)[0, 1],
                np.corrcoef([r["v_hold"] for r in rows if r["t_p"] is not None], b)[0, 1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
