"""Test: can we load the scene with the hand_pusher object?"""
import sys
sys.path.insert(0, "simulator_env/dyniSafety_libero")

import os, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["safelibero_spatial"](safety_level="I")
task = task_suite.get_task(0)

task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"BDDL: {task_bddl_file}")

env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()

inner = env.env
sim = inner.sim
print(f"Model nq={sim.model.nq}, nv={sim.model.nv}")

# Check hand exists
if "hand_pusher_1" in inner.objects_dict:
    obj = inner.objects_dict["hand_pusher_1"]
    bid = sim.model.body_name2id(obj.root_body)
    pos = sim.data.body_xpos[bid].copy()
    print(f"hand_pusher_1 body_xpos: {pos}")
    print(f"hand_pusher_1 joint: {obj.joints}")
else:
    print("hand_pusher_1 NOT in objects_dict!")
    print("objects:", list(inner.objects_dict.keys()))

# Save test image
import imageio
obs = env.env._get_observations()
img = obs["agentview_image"][::-1, ::-1]
os.makedirs("./test_output", exist_ok=True)
imageio.imwrite("./test_output/hand_scene_test.png", img)
print("Saved: ./test_output/hand_scene_test.png")

env.close()
