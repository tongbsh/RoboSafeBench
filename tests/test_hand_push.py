"""Test: obstacle-only dynamic movement (no hand), using dyniSafety_libero."""
import sys
sys.path.insert(0, "simulator_env/dyniSafety_libero")
sys.path.insert(0, "openvla-oft")

import os, numpy as np, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.dynisafety_libero.dynisafety_utils import (
    DynamicObstacleController, get_robot_geom_ids, read_obstacle_table_pose,
)

bm = benchmark.get_benchmark_dict()
ts = bm["safelibero_spatial"](safety_level="I")
task = ts.get_task(0)

bf = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
env.seed(0)

init_states = ts.get_task_init_states(0)
env.reset()
obs = env.set_init_state(init_states[0])

obs_pos, obs_quat = read_obstacle_table_pose(env, "moka_pot_obstacle_1")
print(f"Obstacle on-table: {obs_pos}")

ctrl = DynamicObstacleController(
    env=env,
    obstacle_name="moka_pot_obstacle_1",
    final_pos=obs_pos,
    final_quat=obs_quat,
    start_offset=(0.25, 0.0, 0.0),
    move_start_step=30,
    move_end_step=50,
)

robot_geom_ids = get_robot_geom_ids(env)
os.makedirs("./test_output", exist_ok=True)
frames = []

for step in range(80):
    ctrl.update_before_step(step)
    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    ctrl.check_robot_collision(robot_geom_ids)

    if step % 10 == 0 or step == 30 or step == 50:
        op = ctrl._get_pos()
        st = "BLOCKED" if ctrl.blocked else ("MOVE" if 30 <= step < 50 else "STATIC")
        print(f"  Step {step:3d}: obs=({op[0]:.3f},{op[1]:.3f}) [{st}]")

    frames.append(obs["agentview_image"][::-1, ::-1])

imageio.mimsave("./test_output/obstacle_only_test.mp4", frames, fps=20)
print(f"\nSaved: ./test_output/obstacle_only_test.mp4")
env.close()
