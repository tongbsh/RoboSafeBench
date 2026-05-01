"""Test both dynamic levels: Level 1 (obstacle) and Level 2 (arm)."""
import sys
sys.path.insert(0, "simulator_env/dyniSafety_libero")
sys.path.insert(0, "openvla-oft")

import os, numpy as np, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.dynisafety_libero.dynisafety_utils import (
    DynamicObstacleController, HumanArmController,
    get_robot_geom_ids, read_obstacle_table_pose,
    pad_init_state_for_hand, hide_all_obstacles,
)
import mujoco

bm = benchmark.get_benchmark_dict()
ts = bm["safelibero_spatial"](safety_level="I")
task = ts.get_task(0)

bf = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
os.makedirs("./test_output", exist_ok=True)

for level in [1, 2]:
    print(f"\n{'='*40}\nTesting Level {level}\n{'='*40}")
    env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
    env.seed(0)
    init_states = ts.get_task_init_states(0)
    env.reset()

    padded = pad_init_state_for_hand(init_states[0], env)
    obs = env.set_init_state(padded)

    obs_pos, obs_quat = read_obstacle_table_pose(env, "moka_pot_obstacle_1")
    robot_geom_ids = get_robot_geom_ids(env)

    if level == 1:
        # Hide hand
        hand_obj = env.env.objects_dict.get("hand_pusher_1")
        if hand_obj:
            sim = env.env.sim
            sim.data.set_joint_qpos(hand_obj.joints[-1],
                np.array([10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
            mujoco.mj_forward(sim.model._model, sim.data._data)
        ctrl = DynamicObstacleController(
            env=env, obstacle_name="moka_pot_obstacle_1",
            final_pos=obs_pos, final_quat=obs_quat,
            start_offset=(0.25, 0.0, 0.0), move_start_step=30, move_end_step=50,
        )
    else:
        hide_all_obstacles(env)
        ctrl = HumanArmController(
            env=env, target_pos=obs_pos, target_quat=obs_quat,
            start_offset=(0.25, 0.0, 0.0), move_start_step=30, move_end_step=50,
        )

    frames = []
    for step in range(80):
        ctrl.update_before_step(step)
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        ctrl.check_robot_collision(robot_geom_ids)
        if step % 20 == 0 or step == 30 or step == 50:
            print(f"  Step {step:3d}: blocked={ctrl.blocked}")
        frames.append(obs["agentview_image"][::-1, ::-1])

    path = f"./test_output/level{level}_test.mp4"
    imageio.mimsave(path, frames, fps=20)
    print(f"Saved: {path}")
    env.close()
