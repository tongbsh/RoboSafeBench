"""
run_dynisafety_gr00t_eval.py

DyniSafety-LIBERO evaluation with Isaac-GR00T over ZeroMQ, using the same
episode/metrics/video pipeline as the main benchmark
(``run_dynisafety_eval.run_episode`` / ``run_task``) and :class:`Gr00tPolicy`.

Prerequisites (terminal 1 — GR00T conda env)::

    bash dynisafety_libero/scripts/gr00t/start_gr00t_server.sh \\
        --model-path /path/to/checkpoint --port 5555

Run (terminal 2 — benchmark env, e.g. openvla-oft)::

    python dynisafety_libero/scripts/gr00t/run_dynisafety_gr00t_eval.py \\
        --server_host localhost \\
        --server_port 5555 \\
        --task_suite_name safelibero_spatial

Optional env (see ``Gr00tPolicy``): ``GR00T_ROOT``, ``GR00T_SERVER_*``.

Note: do not enable ``from __future__ import annotations`` here — draccus needs a
real dataclass type from the wrapped function's parameter annotation.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_SIMENV_PATH = str(
    Path(os.environ.get("DYNISAFETY_SIMENV", str(_WORKSPACE_ROOT / "simulator_env" / "dyniSafety_libero")))
)
if _SIMENV_PATH not in sys.path:
    sys.path.insert(0, _SIMENV_PATH)

_WORKSPACE_STR = str(_WORKSPACE_ROOT)
if _WORKSPACE_STR not in sys.path:
    sys.path.insert(0, _WORKSPACE_STR)

import draccus
import tqdm
import wandb
from libero.libero import benchmark

import dynisafety_libero.run_dynisafety_eval as rde
from dynisafety_libero.metrics import AggregateMetrics
from dynisafety_libero.policies import create_policy


@dataclass
class Gr00tDyniSafetyConfig(rde.GenerateConfig):
    """DyniSafety eval defaults for Isaac-GR00T (outputs under outputs/gr00t/)."""

    model_family: str = "gr00t"
    server_port: int = 5555
    num_open_loop_steps: int = 16
    pretrained_checkpoint: Union[str, Path] = ""


@draccus.wrap()
def eval_dynisafety_gr00t(cfg: Gr00tDyniSafetyConfig) -> float:
    rde.validate_config(cfg)
    rde.set_seed_everywhere(cfg.seed)

    policy = create_policy(cfg)
    log_file, local_log_filepath, run_id = rde.setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name](safety_level=cfg.safety_level)
    num_tasks = task_suite.n_tasks

    rde.log_message("Isaac-GR00T (ZeroMQ) on DyniSafety-LIBERO", log_file)
    rde.log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    rde.log_message(f"Safety level: {cfg.safety_level}", log_file)
    rde.log_message(f"Number of tasks: {num_tasks}", log_file)
    rde.log_message(
        f"Dynamic level: {cfg.dynamic_level}, Arm speed: {cfg.arm_speed} "
        f"(peak velocity: {rde.REACHING_PEAK_VELOCITY.get(cfg.arm_speed, '?')} m/s)",
        log_file,
    )
    rde.log_message(
        f"Dynamic obstacle move window: step {cfg.move_start_step} -> {cfg.move_end_step}", log_file
    )
    ckpt_note = cfg.pretrained_checkpoint or "(server-side checkpoint only)"
    rde.log_message(
        f"GR00T server: {cfg.server_host}:{cfg.server_port} | "
        f"num_open_loop_steps={cfg.num_open_loop_steps} | client_ckpt_field={ckpt_note}",
        log_file,
    )

    if cfg.task_id_override is not None:
        task_ids = [cfg.task_id_override]
        rde.log_message(f"Evaluating single task: task_id={cfg.task_id_override}", log_file)
    else:
        task_ids = list(range(num_tasks))

    agg = AggregateMetrics()

    for task_id in tqdm.tqdm(task_ids):
        agg = rde.run_task(cfg, task_suite, task_id, policy, agg, log_file)

    summary = agg.format_summary(cfg.dynamic_level, cfg.topple_threshold_deg)
    for line in summary.split("\n"):
        rde.log_message(line, log_file)

    if cfg.use_wandb:
        wandb.log({
            "success_rate/total": agg.success_rate,
            "num_episodes/total": agg.total_episodes,
            "collision_rate/total": agg.collision_rate,
            "all_collision_episode_rate/total": agg.all_collision_episode_rate,
            "topple_rate/total": agg.topple_rate,
            "sasr/total": agg.sasr,
            "safety_score/total": agg.safety_score,
        })
        rs = agg.reactivity_summary()
        wandb.log({
            "no_reaction_rate/total": rs["no_reaction_rate"],
            **{f"behavior_rate/{k}": v for k, v in rs["behavior_rates"].items()},
        })
        if rs["reaction_latency"]:
            wandb.log({
                "reaction_latency_mean/total": rs["reaction_latency"]["mean"],
            })
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return agg.success_rate


if __name__ == "__main__":
    eval_dynisafety_gr00t()
