"""Probe: PyBullet primitives in the CALVIN world that the BD entries depend on.

1. default linear / angular damping of a free body (``createMultiBody``, maximal and reduced coordinates) and
   whether ``changeDynamics`` sets it to zero (R7, D13 h);
2. gravity from a free-fall fit (D2) and the drop time against ``sqrt(2 h / g)``;
3. kinematic pose set: a body with mass 0 stays where it is put, reads back the commanded pose with zero velocity;
   mass m again makes it fall (``kinematic_pose_set``);
4. release with velocity: ``resetBaseVelocity`` reads back immediately (``release_with_velocity``);
5. contacts of a dropped sphere on the table: when they appear, their distances, the normal direction, and whether
   the reported points belong to the pose before or after the step (``contact_tier``);
6. collision filter mask 0: no contacts, and whether ``getClosestPoints`` still measures (``distance``);
7. continuous collision detection: what a fast sphere crossing a thin robot link reports with the default
   settings (``hazard_ccd``).

    python -m hazardhosts.calvin.probes.physics --egl-device 2
"""
import math
import sys

import numpy as np

from hazardhosts.calvin.probes.common import build_env, probe_args

FREE_POINT = (0.8, 0.6, 2.0)  # far from the table and the robot, above the floor plane


def sphere(env, radius, mass, pos, maximal):
    p, cid = env.p, env.cid
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0.5, 0, 1], physicsClientId=cid)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                             basePosition=pos, useMaximalCoordinates=maximal, physicsClientId=cid)


def state(env, uid):
    p, cid = env.p, env.cid
    pos, quat = p.getBasePositionAndOrientation(uid, physicsClientId=cid)
    lin, ang = p.getBaseVelocity(uid, physicsClientId=cid)
    return np.array(pos), np.array(quat), np.array(lin), np.array(ang)


def step(env, n=1):
    for _ in range(n):
        env.p.stepSimulation(physicsClientId=env.cid)


def damping(env, dt):
    print("== 1. damping (horizontal velocity after 60 steps of free flight from v0 = (1, 0, 0), w0 = (0, 0, 5))")
    p, cid = env.p, env.cid
    for maximal in (True, False):
        for zero in (False, True):
            uid = sphere(env, 0.03, 0.1, FREE_POINT, maximal)
            if zero:
                p.changeDynamics(uid, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=cid)
            p.resetBaseVelocity(uid, [1.0, 0.0, 0.0], [0.0, 0.0, 5.0], physicsClientId=cid)
            step(env, 60)
            _, _, lin, ang = state(env, uid)
            n = 60
            lin_rate = (1.0 - lin[0] ** (1.0 / n)) / dt if lin[0] > 0 else float("nan")
            ang_rate = (1.0 - (ang[2] / 5.0) ** (1.0 / n)) / dt if ang[2] > 0 else float("nan")
            print("  maximal=%-5s set_zero=%-5s vx=%.9f wz/5=%.9f -> per-step decay rate linear %.4g /s angular "
                  "%.4g /s" % (maximal, zero, lin[0], ang[2] / 5.0, lin_rate, ang_rate))
            p.removeBody(uid, physicsClientId=cid)


def gravity(env, dt):
    print("== 2. gravity: free fall of a zero-damping sphere (maximal coordinates), quadratic fit of z(t)")
    p, cid = env.p, env.cid
    uid = sphere(env, 0.03, 0.1, FREE_POINT, True)
    p.changeDynamics(uid, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=cid)
    z0 = state(env, uid)[0][2]
    ts, zs = [], []
    for k in range(1, 121):
        step(env)
        ts.append(k * dt)
        zs.append(state(env, uid)[0][2])
    coeffs = np.polyfit(ts, zs, 2)
    g = -2.0 * coeffs[0]
    exact = z0 - 9.8 * dt * dt * np.arange(1, 121) * np.arange(2, 122) / 2.0
    print("  g_fit = %.9f m/s^2 (setGravity -9.8); fit residual max %.3g m; semi-implicit Euler "
          "z_n = z0 - g dt^2 n(n+1)/2 max deviation %.3g m" % (g, float(np.max(np.abs(np.polyval(coeffs, ts) - zs))),
                                                               float(np.max(np.abs(exact - np.array(zs))))))
    p.removeBody(uid, physicsClientId=cid)
    return g


def kinematic(env, dt):
    print("== 3. kinematic pose set (mass toggled 0 <-> m, maximal coordinates)")
    p, cid = env.p, env.cid
    uid = sphere(env, 0.03, 0.1, FREE_POINT, True)
    p.changeDynamics(uid, -1, mass=0.0, linearDamping=0.0, angularDamping=0.0, physicsClientId=cid)
    target = np.array([0.7, 0.5, 1.5])
    quat = p.getQuaternionFromEuler([0.3, -0.2, 1.0])
    p.resetBasePositionAndOrientation(uid, target.tolist(), quat, physicsClientId=cid)
    p.resetBaseVelocity(uid, [0, 0, 0], [0, 0, 0], physicsClientId=cid)
    pos, q, lin, ang = state(env, uid)
    print("  readback: pos error %.3g m, quat error %.3g, |v| %.3g, |w| %.3g"
          % (np.linalg.norm(pos - target), np.max(np.abs(q - np.array(quat))), np.linalg.norm(lin),
             np.linalg.norm(ang)))
    step(env, 50)
    pos, _, lin, _ = state(env, uid)
    print("  after 50 steps with mass 0: displacement %.3g m, |v| %.3g m/s" % (np.linalg.norm(pos - target),
                                                                               np.linalg.norm(lin)))
    p.changeDynamics(uid, -1, mass=0.1, physicsClientId=cid)
    step(env, 50)
    pos, _, lin, _ = state(env, uid)
    print("  mass 0.1 again, after 50 steps: dz %.6f m (free fall predicts %.6f), vz %.6f m/s (predicts %.6f)"
          % (pos[2] - target[2], -9.8 * dt * dt * 50 * 51 / 2.0, lin[2], -9.8 * dt * 50))
    p.removeBody(uid, physicsClientId=cid)


def release(env):
    print("== 4. release with velocity: readback right after resetBaseVelocity")
    p, cid = env.p, env.cid
    uid = sphere(env, 0.03, 0.1, FREE_POINT, True)
    p.changeDynamics(uid, -1, mass=0.0, physicsClientId=cid)
    p.changeDynamics(uid, -1, mass=0.1, linearDamping=0.0, angularDamping=0.0, physicsClientId=cid)
    lin_cmd, ang_cmd = [-0.7, 0.3, 0.123456789], [0.5, -2.0, 1.0]
    p.resetBaseVelocity(uid, lin_cmd, ang_cmd, physicsClientId=cid)
    _, _, lin, ang = state(env, uid)
    print("  commanded %s / %s, read %s / %s, max error %.3g" % (lin_cmd, ang_cmd, lin.tolist(), ang.tolist(),
                                                                 max(np.max(np.abs(lin - lin_cmd)),
                                                                     np.max(np.abs(ang - ang_cmd)))))
    p.removeBody(uid, physicsClientId=cid)


def contacts(env, dt, table_uid):
    print("== 5. contacts of a sphere (r 0.03 m) dropped from 0.3 m onto the table top")
    p, cid = env.p, env.cid
    uid = sphere(env, 0.03, 0.1, (0.25, -0.1, 0.9), True)
    p.changeDynamics(uid, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=cid)
    first = None
    for k in range(1, 1200):
        before = state(env, uid)
        step(env)
        after = state(env, uid)
        pts = [c for c in p.getContactPoints(bodyA=uid, physicsClientId=cid)]
        closest = p.getClosestPoints(uid, table_uid, 0.02, physicsClientId=cid)
        if pts and first is None:
            first = k
        if first is None and closest and k >= 60:
            print("  step %3d (no engine contact yet) getClosestPoints at t1: dist %+.5f; z(sphere) before %.5f "
                  "after %.5f" % (k, min(c[8] for c in closest), before[0][2], after[0][2]))
        if first is not None and (k <= first + 3 or k % 200 == 0):
            for c in pts:
                on_a = np.array(c[5])
                print("  step %3d body %d link %d dist %+.5f normalOnB %s | z(sphere) before %.5f after %.5f | "
                      "posOnA.z %.5f (= before - r: %+.2e, = after - r: %+.2e) vz before %.4f after %.4f force %.4g"
                      " | getClosestPoints at t1 %+.5f"
                      % (k, c[2], c[4], c[8], np.round(c[7], 4).tolist(), before[0][2], after[0][2], on_a[2],
                         on_a[2] - (before[0][2] - 0.03), on_a[2] - (after[0][2] - 0.03), before[2][2],
                         after[2][2], c[9], min(c[8] for c in closest) if closest else float("nan")))
    p.removeBody(uid, physicsClientId=cid)
    return first


def masks(env, table_uid):
    print("== 6. collision filter mask 0: contacts and getClosestPoints")
    p, cid = env.p, env.cid
    uid = sphere(env, 0.03, 0.1, (0.25, -0.1, 0.45), True)
    p.changeDynamics(uid, -1, mass=0.0, physicsClientId=cid)
    p.setCollisionFilterGroupMask(uid, -1, 0, 0, physicsClientId=cid)
    p.resetBasePositionAndOrientation(uid, [0.25, -0.1, 0.40], [0, 0, 0, 1], physicsClientId=cid)  # inside the top
    step(env)
    print("  mask 0, sphere 2 cm inside the table top: getContactPoints %d, getClosestPoints(table, 0) %s"
          % (len(p.getContactPoints(bodyA=uid, physicsClientId=cid)),
             [round(c[8], 5) for c in p.getClosestPoints(uid, table_uid, 0.0, physicsClientId=cid)]))
    p.setCollisionFilterGroupMask(uid, -1, 1, -1, physicsClientId=cid)
    p.changeDynamics(uid, -1, mass=0.1, physicsClientId=cid)
    p.resetBasePositionAndOrientation(uid, [0.25, -0.1, 0.40], [0, 0, 0, 1], physicsClientId=cid)
    step(env)
    print("  group 1 / mask -1 again: getContactPoints %d" % len(p.getContactPoints(bodyA=uid, physicsClientId=cid)))
    p.removeBody(uid, physicsClientId=cid)


def ccd(env, dt):
    print("== 7. tunnelling: sphere (r 0.01 m) shot at a static 0.01 m thick wall in free space (default CCD "
          "settings, ccdSweptSphereRadius 0)")
    p, cid = env.p, env.cid
    wall_pos = np.array(FREE_POINT)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.005, 0.2, 0.2], physicsClientId=cid)
    wall = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=wall_pos.tolist(),
                             physicsClientId=cid)
    for speed in (1.0, 3.0, 6.0, 12.0, 30.0):
        start = wall_pos - [0.03 + 0.01 + 0.005 + 0.5 * speed * dt, 0, 0]
        uid = sphere(env, 0.01, 0.01, start.tolist(), True)
        p.changeDynamics(uid, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=cid)
        p.resetBaseVelocity(uid, [speed, 0, 0], [0, 0, 0], physicsClientId=cid)
        hits = 0
        for _ in range(int(math.ceil(0.1 / (speed * dt))) + 2):
            step(env)
            hits += len(p.getContactPoints(bodyA=uid, bodyB=wall, physicsClientId=cid))
        pos, _, lin, _ = state(env, uid)
        print("  speed %5.1f m/s (%.4f m per step, wall + sphere span 0.03 m): contacts %d, final x - wall x %+.3f m, "
              "vx %+.3f" % (speed, speed * dt, hits, pos[0] - wall_pos[0], lin[0]))
        p.removeBody(uid, physicsClientId=cid)
    p.removeBody(wall, physicsClientId=cid)


def main() -> int:
    args = probe_args(__doc__).parse_args()
    env = build_env(args)
    try:
        env.reset()
        dt = env.p.getPhysicsEngineParameters(physicsClientId=env.cid)["fixedTimeStep"]
        table = [o.uid for o in env.scene.fixed_objects if o.name == "table"][0]
        print("physics dt %.12f" % dt)
        damping(env, dt)
        gravity(env, dt)
        kinematic(env, dt)
        release(env)
        contacts(env, dt, table)
        masks(env, table)
        ccd(env, dt)
    finally:
        env.close()
        env.cid = -1
    return 0


if __name__ == "__main__":
    sys.exit(main())
