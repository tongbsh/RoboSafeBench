"""Verify physics-based obstacle movement: pushes objects, stops at robot arm."""
import sys
sys.path.insert(0, "simulator_env/dyniSafety_libero")

import os, numpy as np, mujoco, torch, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["safelibero_spatial"](safety_level="I")
task = task_suite.get_task(0)

task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
env.seed(0)
init_states = task_suite.get_task_init_states(0)
env.reset()
obs = env.set_init_state(init_states[0])

# --- Import our controller ---
sys.path.insert(0, "openvla-oft")
from experiments.robot.dynisafety_libero.dynisafety_utils import (
    DynamicObstacleController, get_robot_geom_ids,
)

obstacle_name = "moka_pot_obstacle_1"
inner = env.env
sim = inner.sim

final_pos = sim.data.body_xpos[sim.model.body_name2id(
    inner.objects_dict[obstacle_name].root_body)].copy()
final_quat = sim.data.body_xquat[sim.model.body_name2id(
    inner.objects_dict[obstacle_name].root_body)].copy()

print(f"Final pos (on table): {final_pos}")
start_offset = (0.25, 0.0, 0.0)
print(f"Start pos: {final_pos + np.array(start_offset)}")

ctrl = DynamicObstacleController(
    env=env,
    obstacle_name=obstacle_name,
    final_pos=final_pos,
    final_quat=final_quat,
    start_offset=start_offset,
    move_start_step=30,
    move_end_step=50,
)

robot_geom_ids = get_robot_geom_ids(env)

os.makedirs("./test_output", exist_ok=True)
frames = []
total_steps = 80

for step in range(total_steps):
    task_step = step  # no wait phase in this test

    ctrl.update_before_step(task_step)
    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    blocked = ctrl.check_robot_collision(robot_geom_ids)

    pos = ctrl._get_pos()
    if step % 10 == 0 or step == 30 or step == 50 or blocked:
        status = "BLOCKED" if ctrl.blocked else ("MOVING" if 30 <= step < 50 else "STATIC")
        print(f"  Step {step:3d}: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}  [{status}]")

    img = obs["agentview_image"][::-1, ::-1]
    frames.append(img)

imageio.mimsave("./test_output/dynamic_obstacle_physics_test.mp4", frames, fps=20)
print(f"\nSaved video: ./test_output/dynamic_obstacle_physics_test.mp4")
print(f"Blocked: {ctrl.blocked}")
env.close()
