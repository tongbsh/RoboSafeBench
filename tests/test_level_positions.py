"""Check obstacle positions in Level I vs Level II init states."""
import sys
sys.path.insert(0, "simulator_env/dyniSafety_libero")

import os, numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()

for level in ["I", "II"]:
    task_suite = benchmark_dict["safelibero_spatial"](safety_level=level)
    task = task_suite.get_task(0)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
    env.seed(0)
    init_states = task_suite.get_task_init_states(0)
    env.reset()
    obs = env.set_init_state(init_states[0])
    inner = env.env
    sim = inner.sim
    print(f"\n=== Level {level} ===")
    for name in sorted(inner.objects_dict.keys()):
        obj = inner.objects_dict[name]
        bid = sim.model.body_name2id(obj.root_body)
        pos = sim.data.body_xpos[bid].copy()
        print(f"  {name:40s}  xpos={pos}")
    env.close()
