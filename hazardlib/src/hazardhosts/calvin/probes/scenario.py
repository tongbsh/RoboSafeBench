"""Probe: the scene facts of the CALVIN conformance scenario (``configs/hazard/calvin.yaml: scenario``), measured
on ``CalvinHost`` itself (no runtime installed, so the hazard is placed directly).

``still``  reset to a layout and send ``hold_action`` for ``--steps`` control steps: success oracle, held object,
           TCP displacement and gripper opening over time; the robot collision bodies at the reset.
``motion`` reset to a layout and replay the recorded play data from its frame (``policies.relative_actions``):
           TCP position and velocity, gripper opening, held object and the scene bodies the robot touches.
``free``   candidate free points on a grid: clearance of a probe sphere to every body, its pixels in each policy
           camera (segmentation mask), and the free travel along the six axis directions.
``line``   a strike line through a collision body's reported position along a direction: every body the probe
           sphere touches along the line (only the target may appear), the target's extent along the line from its
           reported position, and the contact normal there versus ``-direction`` (head-on).
``drop``   the vertical column below a drop point: what a sphere released there lands on first, and at what
           height.

    python -m hazardhosts.calvin.probes.scenario --egl-device 2 still --task lift_red_block_table \
        --layout val/554085..554110 --steps 60
"""
import argparse
import sys

import numpy as np

from hazardlib.spec.hazard_spec import AgentSpec

from hazardhosts.calvin.host import CalvinHost
from hazardhosts.calvin.policies import _scales, relative_actions


def _agent(radius):
    return AgentSpec.from_dict({"kind": "probe", "shape": "sphere", "half_extents_m": [radius], "mass_kg": 0.05})


def _clearance(host, uid, skip=()):
    p, cid = host.p, host.cid
    best, name = np.inf, None
    for other in host.bodies.collision_uids:
        if other in skip:
            continue
        for c in p.getClosestPoints(uid, other, 1.0, physicsClientId=cid):
            if c[8] < best:
                best, name = c[8], host.bodies.partner[(c[2], c[4])]
    return best, name


def _touching(host, uid):
    p, cid = host.p, host.cid
    names = {}
    for other in host.bodies.collision_uids:
        for c in p.getClosestPoints(uid, other, 0.0, physicsClientId=cid):
            label = host.bodies.partner[(c[2], c[4])]
            names[label] = min(names.get(label, 0.0), c[8])
    return names


def _pixels(host, uid):
    out = {}
    for camera in host.env.cameras:
        view, proj = host._camera_matrices(camera)
        image = host.p.getCameraImage(width=camera.width, height=camera.height, viewMatrix=view,
                                      projectionMatrix=proj, physicsClientId=host.cid)
        mask = np.reshape(np.asarray(image[4]), (camera.height, camera.width))
        out[camera.name] = int(np.count_nonzero(mask == uid))
    return out


def _place(host, handle, pos):
    host.set_hazard_pose(handle, list(pos), [1.0, 0.0, 0.0, 0.0])
    host.set_hazard_visible(handle, True)


def cmd_still(host, args):
    host.reset_episode(0, args.task, args.layout, 0)
    robot = host.robot_state(0)
    print("collision bodies at the reset (name: reported position):")
    for name, pos in zip(robot.collision_body_names, robot.collision_body_pos):
        print("  %-26s %s" % (name, np.round(pos, 4).tolist()))
    tcp0 = np.array(robot.eef_pose.pos)
    print("TCP %s  opening %.4f  held %r  base %s" % (np.round(tcp0, 4).tolist(), robot.gripper_opening_m,
                                                       robot.held_object, np.round(robot.base_pose.pos, 3).tolist()))
    print("step  t_s     success  held          |dTCP| m   opening")
    for k in range(args.steps + 1):
        if k:
            host.step(0, host.hold_action(0))
        if k % 5 == 0:
            robot = host.robot_state(0)
            print("%4d  %.4f  %-7s  %-12s  %.6f  %.4f" % (k, k * host.control_dt, host.task_success(0),
                                                       robot.held_object,
                                                       np.linalg.norm(np.array(robot.eef_pose.pos) - tcp0),
                                                       robot.gripper_opening_m))


def cmd_motion(host, args):
    host.reset_episode(0, args.task, args.layout, 0)
    layout = host.dataset.layout(args.layout)
    actions = relative_actions(host.dataset, layout.split, layout.frame, args.steps, _scales(host.env))
    robot_uid = host.bodies.robot_uid
    print("step  t_s     frame   TCP                        TCP vel                    |v|    opening held      "
          "robot touches")
    for k in range(args.steps + 1):
        if k:
            host.step(0, actions[k - 1])
        robot = host.robot_state(0)
        touching = sorted({host.bodies.partner[(c[2], c[4])]
                           for c in host.p.getContactPoints(bodyA=robot_uid, physicsClientId=host.cid)
                           if c[2] != robot_uid})
        print("%4d  %.4f  %d  %-26s %-26s %.3f  %.4f  %-8s  %s"
              % (k, k * host.control_dt, layout.frame + k, np.round(robot.eef_pose.pos, 3).tolist(),
                 np.round(robot.eef_lin_vel_mps, 3).tolist(), np.linalg.norm(robot.eef_lin_vel_mps),
                 robot.gripper_opening_m, robot.held_object, touching))


def cmd_free(host, args):
    host.reset_episode(0, args.task, args.layout, 0)
    handle = host.alloc_hazard(0, _agent(args.radius))
    uid = host.hazard_uid(handle)
    center = np.array(args.center)
    axes = [np.arange(-h, h + 1e-9, args.step) for h in args.half]
    rows = []
    for dx in axes[0]:
        for dy in axes[1]:
            for dz in axes[2]:
                point = center + [dx, dy, dz]
                _place(host, handle, point)
                gap, near = _clearance(host, uid)
                if gap < args.min_clearance:
                    continue
                px = _pixels(host, uid)
                travel = {}
                for label, d in (("+x", (1, 0, 0)), ("-x", (-1, 0, 0)), ("+y", (0, 1, 0)), ("-y", (0, -1, 0)),
                                 ("+z", (0, 0, 1)), ("-z", (0, 0, -1))):
                    s = 0.0
                    while s < args.travel:
                        _place(host, handle, point + (s + 0.005) * np.array(d))
                        if _clearance(host, uid)[0] < 0.0:
                            break
                        s += 0.005
                    travel[label] = s
                rows.append((min(px.values()), point, gap, near, px, travel))
    rows.sort(key=lambda r: -r[0])
    print("%d points with clearance >= %.3f m (radius %.3f); best by the smaller camera pixel count:"
          % (len(rows), args.min_clearance, args.radius))
    for _, point, gap, near, px, travel in rows[:args.top]:
        print("  %s clearance %.4f (%s) pixels %s travel %s" % (np.round(point, 3).tolist(), gap, near, px,
                                                                 {k: round(v, 3) for k, v in travel.items()}))
    host.free_hazard(handle)


def cmd_line(host, args):
    host.reset_episode(0, args.task, args.layout, 0)
    robot = host.robot_state(0)
    names = robot.collision_body_names
    if args.body not in names:
        raise SystemExit("%r is not a collision body (%s)" % (args.body, names))
    origin = np.array(robot.collision_body_pos[names.index(args.body)])
    d = np.array(args.direction, dtype=float)
    d /= np.linalg.norm(d)
    handle = host.alloc_hazard(0, _agent(args.radius))
    uid = host.hazard_uid(handle)
    seen = {}
    first = None
    for s in np.arange(-args.reach, args.reach + 1e-9, 0.001):
        _place(host, handle, origin + s * d)
        for name, depth in _touching(host, uid).items():
            seen.setdefault(name, [s, s, depth])
            seen[name][1] = s
            seen[name][2] = min(seen[name][2], depth)
            if first is None and name == args.body:
                first = s
    print("line through %s %s along %s, sphere r %.3f:" % (args.body, np.round(origin, 4).tolist(),
                                                           np.round(d, 3).tolist(), args.radius))
    for name, (a, b, depth) in sorted(seen.items(), key=lambda kv: kv[1][0]):
        print("  touches %-26s for s in [%.3f, %.3f] (deepest %.4f)" % (name, a, b, depth))
    if first is not None:
        lo, hi = first - 0.001, first
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            _place(host, handle, origin + mid * d)
            if args.body in _touching(host, uid):
                hi = mid
            else:
                lo = mid
        _place(host, handle, origin + hi * d)
        pts = [c for other in host.bodies.collision_uids
               for c in host.p.getClosestPoints(uid, other, 0.0, physicsClientId=host.cid)
               if host.bodies.partner[(c[2], c[4])] == args.body]
        c = min(pts, key=lambda c: c[8])
        normal = np.array(c[7])
        angle = np.degrees(np.arccos(np.clip(np.dot(normal, -d), -1, 1)))
        print("  first touch of %s at s = %.4f: extent along the line %.4f m (radius_m), normal %s, %.1f deg from "
              "-direction" % (args.body, hi, -hi - args.radius, np.round(normal, 3).tolist(), angle))
    host.free_hazard(handle)


def cmd_drop(host, args):
    host.reset_episode(0, args.task, args.layout, 0)
    handle = host.alloc_hazard(0, _agent(args.radius))
    uid = host.hazard_uid(handle)
    point = np.array(args.point)
    z = point[2]
    while z > 0.0:
        _place(host, handle, [point[0], point[1], z])
        hit = _touching(host, uid)
        if hit:
            print("drop column at %s (r %.3f): first touch at centre z %.4f with %s; fall height %.4f m"
                  % (np.round(point[:2], 3).tolist(), args.radius, z, sorted(hit), point[2] - z))
            break
        z -= 0.0005
    _place(host, handle, point)
    print("pixels at the drop point: %s" % _pixels(host, uid))
    host.free_hazard(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--egl-device", type=int, required=True)
    parser.add_argument("--dataset-dir", default=None)
    sub = parser.add_subparsers(dest="cmd")
    sub.required = True
    for name in ("still", "motion", "free", "line", "drop"):
        p = sub.add_parser(name)
        p.add_argument("--task", required=True)
        p.add_argument("--layout", required=True)
        p.add_argument("--radius", type=float, default=0.02)
    sub.choices["still"].add_argument("--steps", type=int, default=60)
    sub.choices["motion"].add_argument("--steps", type=int, default=60)
    p = sub.choices["free"]
    p.add_argument("--center", type=float, nargs=3, required=True)
    p.add_argument("--half", type=float, nargs=3, default=[0.1, 0.1, 0.05])
    p.add_argument("--step", type=float, default=0.025)
    p.add_argument("--min-clearance", type=float, default=0.03)
    p.add_argument("--travel", type=float, default=0.15)
    p.add_argument("--top", type=int, default=12)
    p = sub.choices["line"]
    p.add_argument("--body", required=True)
    p.add_argument("--direction", type=float, nargs=3, required=True)
    p.add_argument("--reach", type=float, default=0.4)
    sub.choices["drop"].add_argument("--point", type=float, nargs=3, required=True)
    args = parser.parse_args()
    host = CalvinHost(dataset_dir=args.dataset_dir, egl_device=args.egl_device)
    try:
        {"still": cmd_still, "motion": cmd_motion, "free": cmd_free, "line": cmd_line,
         "drop": cmd_drop}[args.cmd](host, args)
    finally:
        host.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
