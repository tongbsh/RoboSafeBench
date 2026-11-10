"""Exercise the pinned, unmodified LIBERO evaluator with host lifecycle hooks.

Only evaluate_one_task_success is loaded with AST, avoiding unrelated training
dependencies. Its loop body is unchanged. A single-environment vector proxy
routes its reset/set_init_state/step/close to the HazardHost and ends each
episode once, including the last one when the upstream loop closes the env.
These hold-policy episodes test instrumentation, not policy performance.
"""
import ast
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from hazardlib.driving import EpisodeSession
from hazardlib.runtime import RuntimeConfig

from .host import LiberoHost
from .paths import tree_root

LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
EVALUATOR = "libero/lifelong/metric.py"


class Timer:
    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        self.elapsed = time.monotonic() - self.start

    def get_elapsed_time(self):
        return self.elapsed


def validate_native_loop(config_dir, out, episodes, max_steps):
    import torch
    root = tree_root()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != LIBERO_COMMIT:
        raise RuntimeError("native evaluator is pinned to %s, found %s" % (LIBERO_COMMIT, commit))
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", EVALUATOR], check=True)
    path = root / EVALUATOR
    tree = ast.parse(path.read_text(), filename=str(path))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_one_task_success"]
    if len(body) != 1:
        raise RuntimeError("upstream evaluator not found")
    host = LiberoHost(config_dir=config_dir)
    session = EpisodeSession(host, RuntimeConfig.load(config_dir))
    task_id = host.task_ids()[0]
    logs, begun, ended = [], [], []

    class Vec:
        def __init__(self, factories):
            self.active = False
            self.index = 0

        def finish(self):
            if self.active:
                log = session.runtime.end_episode(0)
                logs.append(log.to_dict())
                ended.append(self.index - 1)
                self.active = False

        def reset(self):
            self.finish()

        def set_init_state(self, states):
            session.runtime.arm(0, None, control_id="native-%d" % self.index)
            host.reset_episode(0, task_id, "init/%d" % self.index, 0)
            begun.append(self.index)
            self.index += 1
            self.active = True
            return [host.policy_observation(0)]

        def step(self, actions):
            host.step(0, actions[0])
            return [host.policy_observation(0)], [0.], [host.task_success(0)], [{}]

        def close(self):
            self.finish()

    class Hold:
        def get_action(self, data):
            return np.asarray([host.hold_action(0)])

    algo = SimpleNamespace(eval=lambda: None, reset=lambda: None, policy=Hold())
    from libero.libero import get_libero_path
    suite, index = host._suite_task(task_id)
    task = suite.get_task(index)
    cfg = SimpleNamespace(lifelong=SimpleNamespace(algo="HoldProbe"),
        eval=SimpleNamespace(num_procs=1, n_eval=episodes, use_mp=False, max_steps=max_steps),
        data=SimpleNamespace(img_h=host._hw, img_w=host._hw),
        bddl_folder=get_libero_path("bddl_files"), init_states_folder=get_libero_path("init_states"))
    namespace = {"np": np, "gc": gc, "os": os, "torch": torch, "Timer": Timer,
                 "DummyVectorEnv": Vec, "SubprocVectorEnv": Vec,
                 "OffScreenRenderEnv": lambda **kwargs: None,
                 "raw_obs_to_tensor_obs": lambda obs, emb, config: obs}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    try:
        rate = namespace["evaluate_one_task_success"](cfg, algo, task, None, index)
        assert len(begun) == len(ended) == len(logs) == episodes
        target = Path(out)
        target.mkdir(parents=True, exist_ok=True)
        (target / "hazard_events.jsonl").write_text("".join(json.dumps(log) + "\n" for log in logs))
        report = {"benchmark_commit": commit, "evaluator": EVALUATOR,
                  "begun": len(begun), "ended": len(ended), "logs": len(logs),
                  "success_rate": rate, "purpose": "native loop lifecycle probe, hold policy"}
        (target / "summary.json").write_text(json.dumps(report, indent=2))
        print("ITEM 14 PASS " + json.dumps(report), flush=True)
    finally:
        host.close()
