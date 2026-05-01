"""Debug: check arm vs robot collision detection."""
import sys
sys.path.insert(0, "simulator_env/dyniSafety_libero")
sys.path.insert(0, "openvla-oft")

import os, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.dynisafety_libero.dynisafety_utils import (
    HumanArmController, get_robot_geom_ids, read_obstacle_table_pose,
    pad_init_state_for_hand, hide_all_obstacles, detect_collisions,
)
import mujoco

bm = benchmark.get_benchmark_dict()
ts = bm["safelibero_spatial"](safety_level="I")
task = ts.get_task(0)
bf = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
env.seed(0)
init_states = ts.get_task_init_states(0)
env.reset()
padded = pad_init_state_for_hand(init_states[0], env)
obs = env.set_init_state(padded)

# Read obstacle position BEFORE hiding
obs_pos, obs_quat = read_obstacle_table_pose(env, "moka_pot_obstacle_1")
print(f"Obstacle pos (before hide): {obs_pos}")

# Now hide obstacles
hide_all_obstacles(env)

# Create arm controller
ctrl = HumanArmController(env=env, target_pos=obs_pos, target_quat=obs_quat,
    start_offset=(0.25, 0.0, 0.0), move_start_step=10, move_end_step=30)

robot_geom_ids = get_robot_geom_ids(env)
arm_geom_ids = ctrl.obstacle_geom_ids
sim = env.env.sim

print(f"Robot collision geom IDs: {len(robot_geom_ids)}")
print(f"Arm geom IDs: {sorted(arm_geom_ids)}")
print(f"Arm start: {ctrl.start_pos}, final: {ctrl.final_pos}")
print(f"Robot EEF: {obs['robot0_eef_pos']}")

for step in range(60):
    ctrl.update_before_step(step)
    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    # Check contacts
    dyn_colliding = detect_collisions(env, robot_geom_ids, arm_geom_ids)

    ctrl.check_robot_collision(robot_geom_ids)

    if step % 5 == 0 or dyn_colliding:
        arm_pos = sim.data.body_xpos[ctrl._body_id]
        print(f"Step {step:3d}: arm=({arm_pos[0]:.3f},{arm_pos[1]:.3f},{arm_pos[2]:.3f}) "
              f"blocked={ctrl.blocked} dyn_collisions={len(dyn_colliding)}")

env.close()
