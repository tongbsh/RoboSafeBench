"""Compare unmodified original LIBERO to the host with its hazard parked."""
import json
import os
from pathlib import Path

import numpy as np

from .host import LiberoHost
from .paths import import_libero


def compare_original(out, suite_name, task_index, episode):
    import_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    host = LiberoHost()
    suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    task = suite.get_task(task_index)
    task_id = suite_name + "/" + task.name
    demo = host._demo(task_id, suite_name)
    key = "demo_%d" % episode
    actions, initial = demo.actions(key), demo.states(key)[0]
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=host._hw, camera_widths=host._hw,
                             camera_names=["agentview", "robot0_eye_in_hand"], hard_reset=False)
    reference, plain_success = [], []
    try:
        env.seed(0)
        env.reset()
        env.robots[0].gripper.current_action = np.zeros_like(env.robots[0].gripper.current_action)
        env.set_init_state(initial)
        controller = env.robots[0].controller
        controller.update(force=True)
        controller.reset_goal()
        for action in actions:
            obs, _, _, _ = env.step(action)
            reference.append({"eef": obs["robot0_eef_pos"].copy(),
                              "joints": env.sim.data.qpos.copy(),
                              "image": obs["agentview_image"].copy()})
            plain_success.append(bool(env.check_success()))
    finally:
        env.close()
    try:
        host.reset_episode(0, task_id, suite_name + "/" + key, 0)
        max_eef, max_joints, max_pixels, changed_pixels = 0., 0., 0, 0
        augmented_success = []
        for action, original in zip(actions, reference):
            host.step(0, action)
            max_eef = max(max_eef, float(np.max(np.abs(host._obs["robot0_eef_pos"] - original["eef"]))))
            max_joints = max(max_joints, float(np.max(np.abs(host.d.qpos[:len(original["joints"])] - original["joints"]))))
            diff = np.abs(host._obs["agentview_image"].astype(int) - original["image"].astype(int))
            max_pixels = max(max_pixels, int(diff.max()))
            changed_pixels += int(np.count_nonzero(np.any(diff, axis=2)))
            augmented_success.append(host.task_success(0))
        report = {"suite": suite_name, "task_index": task_index, "demo": episode, "steps": len(actions),
                  "max_eef_difference_m": max_eef, "max_joint_difference": max_joints,
                  "max_image_difference": max_pixels, "changed_pixels": changed_pixels,
                  "plain_success": any(plain_success), "parked_success": any(augmented_success),
                  "success_timeline_identical": plain_success == augmented_success}
        path = Path(out)
        path.mkdir(parents=True, exist_ok=True)
        (path / ("%s_%d_%d.json" % (suite_name, task_index, episode))).write_text(json.dumps(report, indent=2))
        print("ORIGINAL PARITY " + json.dumps(report), flush=True)
    finally:
        host.close()


def compare_initial_layouts(out, suite_name, task_indices, count):
    """Feed the same official action sequence to demo and official init layouts.

    This is a protocol diagnostic, not an evaluation of a learned policy.
    First held-object and success events describe phase-timing differences.
    """
    host = LiberoHost()
    tasks = [t for t in host.task_ids() if t.startswith(suite_name + "/")]
    rows = []
    try:
        for index in task_indices:
            task = tasks[index]
            for k in range(count):
                actions = host._demo(task, suite_name).actions("demo_%d" % k)
                runs = []
                for layout in (suite_name + "/demo_%d" % k, "init/%d" % k):
                    host.reset_episode(0, task, layout, 0)
                    objects = {n: p.pos.copy() for n, p in host.object_poses(0).items()
                               if n in host.free_objects}
                    path, first_held, success_t = [], None, None
                    for j, action in enumerate(actions):
                        host.step(0, action)
                        robot = host.robot_state(0)
                        path.append(robot.eef_pose.pos.copy())
                        if robot.held_object is not None and first_held is None:
                            first_held = (j + 1) * host.control_dt
                        if host.task_success(0) and success_t is None:
                            success_t = (j + 1) * host.control_dt
                    runs.append({"layout": layout, "objects": objects, "eef": np.asarray(path),
                                 "first_held_t_s": first_held, "success_t_s": success_t})
                a, b = runs
                object_gaps = {n: float(np.linalg.norm(a["objects"][n] - b["objects"][n]))
                               for n in sorted(set(a["objects"]) & set(b["objects"]))}
                gap = np.linalg.norm(a["eef"] - b["eef"], axis=1)
                rows.append({"task_index": index, "demo_index": k, "object_initial_differences_m": object_gaps,
                             "eef_path_rmse_m": float(np.sqrt(np.mean(gap ** 2))),
                             "eef_path_max_difference_m": float(gap.max()),
                             "events": [{key: run[key] for key in ("layout", "first_held_t_s", "success_t_s")}
                                        for run in runs]})
                target = Path(out)
                target.mkdir(parents=True, exist_ok=True)
                (target / "initial_layout_comparison.json").write_text(json.dumps(rows, indent=2))
                print("INITIAL LAYOUT " + json.dumps(rows[-1]), flush=True)
    finally:
        host.close()
