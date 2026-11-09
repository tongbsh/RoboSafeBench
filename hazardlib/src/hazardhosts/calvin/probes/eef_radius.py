"""Probe: effective EEF contact radius of the CALVIN gripper (D1, BD ``eef_contact_radius_m``).

For a probe sphere of radius r placed on the ray ``TCP + s * n``, the first-contact standoff ``s*(n)`` is the
largest ``s`` at which the sphere touches the robot (signed getClosestPoints distance 0, found by bisection after
a scan inward in 1 mm steps from 0.3 m horizontally and 1.5 m along +z, where the arm stands above the downward
gripper). The effective radius along ``n`` is ``r_eff(n) = s*(n) - r``: the D1
builder puts the hazard centroid at ``p_ref + (r + eef_contact_radius_m) n``, so a single radius is exact only
where ``r_eff(n)`` equals it. Measured at dataset frames with the gripper open and closed, for the vertical
(fall: hazard above the TCP), the four horizontal axes and the four horizontal diagonals, and the first robot body
touched.

    python -m hazardhosts.calvin.probes.eef_radius --egl-device 2 --frames val/553636 val/553700
"""
import sys

import numpy as np

from hazardhosts.calvin.dataset import CalvinDataset
from hazardhosts.calvin.probes.common import build_env, probe_args

DIRECTIONS = {"+z": (0, 0, 1), "+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
              "+x+y": (1, 1, 0), "+x-y": (1, -1, 0), "-x+y": (-1, 1, 0), "-x-y": (-1, -1, 0)}


def signed_distance(env, sphere, robot, at):
    p, cid = env.p, env.cid
    p.resetBasePositionAndOrientation(sphere, at.tolist(), [0, 0, 0, 1], physicsClientId=cid)
    pts = p.getClosestPoints(sphere, robot, 5.0, physicsClientId=cid)
    best = min(pts, key=lambda c: c[8])
    return best[8], best[4]


def first_contact(env, sphere, robot, tcp, n, start):
    prev = start
    d_prev, _ = signed_distance(env, sphere, robot, tcp + prev * n)
    if d_prev <= 0:
        return None, None
    s = prev
    while s > 0:
        s = round(s - 0.001, 6)
        d, _ = signed_distance(env, sphere, robot, tcp + s * n)
        if d <= 0:
            lo, hi = s, prev
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if signed_distance(env, sphere, robot, tcp + mid * n)[0] <= 0:
                    lo = mid
                else:
                    hi = mid
            return hi, signed_distance(env, sphere, robot, tcp + lo * n)[1]
        prev = s
    return 0.0, None


def main() -> int:
    parser = probe_args(__doc__)
    parser.add_argument("--frames", nargs="+", required=True, help="layout ids <split>/<frame>")
    parser.add_argument("--radii", nargs="+", type=float, default=[0.02, 0.03])
    args = parser.parse_args()
    env = build_env(args)
    p, cid = env.p, env.cid
    robot = env.robot.robot_uid
    links = {j: p.getJointInfo(robot, j, physicsClientId=cid)[12].decode() for j in range(p.getNumJoints(robot))}
    links[-1] = "panda_link0"
    data = CalvinDataset(args.dataset_dir, {"val": "validation", "train": "training"})
    closed = {}  # radius -> r_eff of the horizontal directions at frames with the gripper closed (opening < 5 mm)
    try:
        for layout_id in args.frames:
            layout = data.layout(layout_id)
            robot_obs, scene_obs = data.state(layout.split, layout.frame)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            tcp_state = p.getLinkState(robot, env.robot.tcp_link_id, computeForwardKinematics=1, physicsClientId=cid)
            tcp = np.array(tcp_state[0])
            rot = np.array(p.getMatrixFromQuaternion(tcp_state[1])).reshape(3, 3)
            print("== %s: TCP %s, gripper opening %.4f m, TCP z axis (world) %s, finger axis (TCP y) %s"
                  % (layout_id, np.round(tcp, 4).tolist(), robot_obs[6], np.round(rot[:, 2], 3).tolist(),
                     np.round(rot[:, 1], 3).tolist()))
            for r in args.radii:
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=cid)
                sphere = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=[5, 5, 5],
                                           physicsClientId=cid)
                p.setCollisionFilterGroupMask(sphere, -1, 0, 0, physicsClientId=cid)
                row = []
                for label, d in DIRECTIONS.items():
                    n = np.array(d, dtype=float) / np.linalg.norm(d)
                    s, link = first_contact(env, sphere, robot, tcp, n, 1.5 if d[2] else 0.3)
                    if robot_obs[6] < 0.005 and not d[2] and s is not None:
                        closed.setdefault(r, []).append(s - r)
                    row.append("%s %s(%s)" % (label, "n/a" if s is None else "%.4f" % (s - r),
                                              "-" if link is None else links[link]))
                print("  r = %.3f: r_eff  %s" % (r, "  ".join(row)))
                p.removeBody(sphere, physicsClientId=cid)
    finally:
        env.close()
        env.cid = -1
    for r, values in sorted(closed.items()):
        print("closed gripper, horizontal, r = %.3f: %d values, median %.4f, min %.4f, max %.4f"
              % (r, len(values), float(np.median(values)), min(values), max(values)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
