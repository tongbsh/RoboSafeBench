"""Probe: engine version and parameters, every collision body (name, link, shape, dynamics), AABBs of the table
and blocks, the robot's link names, and on which GPU the EGL renderer runs (nvidia-smi).

    python -m hazardhosts.calvin.probes.scene_survey --egl-device 2 [--cameras static,gripper,tactile]
"""
import os
import sys

import numpy as np

from hazardhosts.calvin.probes.common import body_names, build_env, collision_links, nvidia_smi_processes, probe_args

GEOM = {2: "sphere", 3: "box", 4: "cylinder", 5: "mesh", 6: "plane", 7: "capsule"}


def main() -> int:
    args = probe_args(__doc__).parse_args()
    env = build_env(args)
    try:
        p, cid = env.p, env.cid
        import pkg_resources
        print("pybullet", pkg_resources.get_distribution("pybullet").version, "API", p.getAPIVersion())
        params = p.getPhysicsEngineParameters(physicsClientId=cid)
        print("engine parameters:", {k: params[k] for k in sorted(params)})
        print("EGL env: EGL_VISIBLE_DEVICES=%s EGL_DEVICE_ID=%s PYOPENGL_PLATFORM=%s"
              % (os.environ.get("EGL_VISIBLE_DEVICES"), os.environ.get("EGL_DEVICE_ID"),
                 os.environ.get("PYOPENGL_PLATFORM")))
        print("cameras:", [type(c).__name__ + ":" + c.name for c in env.cameras])
        print("action_repeat", env.action_repeat, "control_freq", env.control_freq)
        names = body_names(env)
        print("bodies:", names)
        print("collision links (uid, link, body/link, shapes, mass, damping-free dynamics info):")
        for uid, link, body, link_name, shapes in collision_links(env):
            info = p.getDynamicsInfo(uid, link, physicsClientId=cid)
            kinds = sorted({GEOM.get(s[2], str(s[2])) for s in shapes})
            aabb = p.getAABB(uid, link, physicsClientId=cid)
            print("  %3d %3d %-36s n=%d %-14s mass=%.4g friction=%.3g restitution=%.3g margin=%.4g "
                  "aabb=%s..%s" % (uid, link, "%s/%s" % (body, link_name), len(shapes), ",".join(kinds), info[0],
                                   info[1], info[5], info[11], np.round(aabb[0], 4).tolist(),
                                   np.round(aabb[1], 4).tolist()))
        robot = env.robot
        print("robot tcp_link_id", robot.tcp_link_id, "end_effector_link_id", robot.end_effector_link_id,
              "gripper joints", robot.gripper_joint_ids, "arm joints", robot.arm_joint_ids)
        obs = env.reset()
        for _ in range(3):
            obs, _, _, info = env.step(np.array([0.0, 0, 0, 0, 0, 0, 1]))
        print("obs keys:", {k: (sorted(v) if isinstance(v, dict) else np.asarray(v).shape) for k, v in obs.items()})
        for group in ("rgb_obs", "depth_obs"):
            for key, value in obs[group].items():
                print("  %s %s %s" % (key, np.asarray(value).shape, np.asarray(value).dtype))
        print("robot_obs", np.round(obs["robot_obs"], 5).tolist())
        print("scene_obs", np.round(obs["scene_obs"], 5).tolist())
        print("info keys", sorted(info), "scene_info keys", sorted(info["scene_info"]))
        print(nvidia_smi_processes(os.getpid()))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
