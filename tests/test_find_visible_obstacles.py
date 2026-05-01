"""Find which obstacle is actually on the table (not hidden) for each task."""
import sys
sys.path.insert(0, "simulator_env/dyniSafety_libero")
import os, numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()

for suite_name in ["safelibero_spatial", "safelibero_object", "safelibero_goal", "safelibero_long"]:
    task_suite = benchmark_dict[suite_name](safety_level="I")
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=128, camera_widths=128)
        env.seed(0)
        init_states = task_suite.get_task_init_states(task_id)
        env.reset()
        obs = env.set_init_state(init_states[0])
        inner = env.env
        sim = inner.sim
        visible = []
        for name in sorted(inner.objects_dict.keys()):
            if "obstacle" not in name and "box_base" not in name:
                continue
            obj = inner.objects_dict[name]
            bid = sim.model.body_name2id(obj.root_body)
            pos = sim.data.body_xpos[bid].copy()
            if abs(pos[0]) < 2 and abs(pos[1]) < 2:
                visible.append((name, pos))
        env.close()
        vis_str = ", ".join([f"{n}: ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})" for n, p in visible])
        print(f"[{suite_name}][task{task_id}] {task.language[:60]:60s} visible: {vis_str}")
