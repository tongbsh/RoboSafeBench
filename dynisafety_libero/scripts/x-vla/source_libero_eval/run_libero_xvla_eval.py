"""
run_libero_xvla_eval.py

Evaluates the X-VLA policy on the original (unmodified) LIBERO benchmark.

Connects to a running X-VLA FastAPI inference server and rolls out each task
suite.  Videos and logs are saved under:
    dynisafety_libero/outputs/xvla/source_libero/

Prerequisites
-------------
1. Start the X-VLA server (in 'XVLA' conda env):
       bash dynisafety_libero/scripts/x-vla/start_xvla_server.sh

2. Run this script (in 'openvla' conda env, where LIBERO is installed):
       bash dynisafety_libero/scripts/x-vla/source_libero_eval/run_libero_eval.sh \\
           --task-suite libero_spatial

API reference
-------------
X-VLA server POST /act payload:
    {
        "image0":               json_numpy-encoded np.ndarray (H,W,3)  [agentview, flipped]
        "image1":               json_numpy-encoded np.ndarray (H,W,3)  [wrist, optional]
        "language_instruction": str
        "proprio":              json_numpy-encoded np.ndarray (20,)
                                = [pos(3), rot6d(6), grip(1), zeros(10)]
        "domain_id":            int  (3 for LIBERO)
        "steps":                int  diffusion denoising steps (default 10)
    }
Response:
    {"action": [[pos3, rot6d6, grip1], ...]}  shape (T, 10)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List

import imageio
import json_numpy
import numpy as np

json_numpy.patch()

# ---------------------------------------------------------------------------
# Path bootstrap: add simulator_env/LIBERO so `libero` is importable.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = (_SCRIPT_DIR / "../../../..").resolve()   # dynamic_safty_bench/

_LIBERO_PATH = str(_PROJ_ROOT / "simulator_env" / "LIBERO")
if _LIBERO_PATH not in sys.path:
    sys.path.insert(0, _LIBERO_PATH)

_PROJ_ROOT_STR = str(_PROJ_ROOT)
if _PROJ_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJ_ROOT_STR)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from dynisafety_libero.xvla_source_client import (
    XVLAClient,
    dummy_action,
    get_flipped_agentview,
    get_wrist_image,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATE      = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# Maximum env steps per episode (excluding stabilisation warm-up).
# X-VLA uses absolute actions so more steps are needed than for relative-action
# models to allow the controller to reach each target pose.
TASK_MAX_STEPS = {
    "libero_spatial": 600,
    "libero_object":  600,
    "libero_goal":    600,
    "libero_10":      700,
    "libero_90":      600,
}

VALID_SUITES = list(TASK_MAX_STEPS.keys())

# ---------------------------------------------------------------------------
# LIBERO env helpers
# ---------------------------------------------------------------------------

def get_libero_env(task, resolution: int = 256):
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": resolution,
        "camera_widths":  resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task.language

# ---------------------------------------------------------------------------
# Logging / video helpers
# ---------------------------------------------------------------------------

def log_message(msg: str, log_file=None) -> None:
    logger.info(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def save_video(
    images: List[np.ndarray],
    episode_idx: int,
    success: bool,
    task_description: str,
    video_dir: str,
    log_file=None,
) -> str:
    os.makedirs(video_dir, exist_ok=True)
    safe_task = (
        task_description.lower()
        .replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    mp4_path = os.path.join(
        video_dir,
        f"{DATE_TIME}--xvla--episode={episode_idx}"
        f"--success={success}--task={safe_task}.mp4",
    )
    writer = imageio.get_writer(mp4_path, fps=30)
    for img in images:
        writer.append_data(img)
    writer.close()
    log_message(f"Saved video: {mp4_path}", log_file)
    return mp4_path

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env,
    task_description: str,
    client:           XVLAClient,
    initial_state,
    max_steps:        int,
    num_steps_wait:   int,
    save_video_flag:  bool,
    log_file=None,
):
    env.reset()
    obs = env.set_init_state(initial_state)

    # ---- stabilisation warm-up (relative mode, zero displacement = stay put) ----
    # Must run BEFORE setting use_delta=False: dummy_action=[0,0,0,0,0,0,-1] means
    # "no movement" only in relative mode; in abs mode it would target position (0,0,0).
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(dummy_action())

    # Switch to absolute EEF control AFTER warm-up, matching the original libero_client.py
    for robot in env.env.robots:
        robot.controller.use_delta = False

    t              = 0
    replay_images: List[np.ndarray] = []
    success        = False

    try:
        while t < max_steps:
            main_img  = get_flipped_agentview(obs)
            wrist_img = get_wrist_image(obs)

            if save_video_flag:
                replay_images.append(main_img.copy())

            # Query server only when the action queue is exhausted, matching the
            # original libero_client.py behavior (execute ALL returned actions
            # before re-querying rather than truncating at num_open_loop_steps).
            if not client.has_actions():
                client.query(main_img, wrist_img, task_description, env)

            action = client.pop_action()
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception:
        log_message(f"Episode error:\n{traceback.format_exc()}", log_file)

    return success, replay_images

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> float:
    out_base   = _PROJ_ROOT / "dynisafety_libero" / "outputs" / "xvla" / "source_libero"
    log_dir    = out_base / "logs"
    video_base = out_base / "videos" / args.task_suite / DATE

    log_dir.mkdir(parents=True, exist_ok=True)
    video_base.mkdir(parents=True, exist_ok=True)

    run_id = f"EVAL-xvla-{args.task_suite}-{DATE_TIME}"
    if args.run_note:
        run_id += f"--{args.run_note}"

    log_path = log_dir / f"{run_id}.txt"
    log_file = open(log_path, "w")

    log_message(f"Run ID         : {run_id}", log_file)
    log_message(f"Task suite     : {args.task_suite}", log_file)
    log_message(f"Server         : {args.server_host}:{args.server_port}", log_file)
    log_message(f"Domain ID      : {args.domain_id}", log_file)
    log_message(f"Diff. steps    : {args.steps}", log_file)
    log_message(f"Trials / task  : {args.num_trials}", log_file)
    log_message(f"Warm-up steps  : {args.num_steps_wait}", log_file)
    log_message(f"Log file       : {log_path}", log_file)

    # ------------------------------------------------------------------
    # Build client (waits for server readiness internally)
    # ------------------------------------------------------------------
    client = XVLAClient(
        host=args.server_host,
        port=args.server_port,
        domain_id=args.domain_id,
        steps=args.steps,
    )

    # ------------------------------------------------------------------
    # LIBERO benchmark
    # ------------------------------------------------------------------
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite not in benchmark_dict:
        raise ValueError(
            f"Unknown task suite '{args.task_suite}'. "
            f"Available: {list(benchmark_dict.keys())}"
        )
    task_suite_obj = benchmark_dict[args.task_suite]()
    num_tasks  = task_suite_obj.n_tasks
    max_steps  = TASK_MAX_STEPS.get(args.task_suite, 600)

    log_message(f"Number of tasks: {num_tasks}", log_file)
    log_message("=" * 60, log_file)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    total_episodes  = 0
    total_successes = 0

    for task_id in range(num_tasks):
        task           = task_suite_obj.get_task(task_id)
        initial_states = task_suite_obj.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=args.env_res)

        task_dir = video_base / task_description.lower().replace(" ", "_")[:60]
        task_dir.mkdir(parents=True, exist_ok=True)

        task_episodes  = 0
        task_successes = 0

        log_message(f"\n[Task {task_id + 1}/{num_tasks}] {task_description}", log_file)

        for ep_idx in range(args.num_trials):
            initial_state = initial_states[ep_idx % len(initial_states)]
            log_message(f"  Episode {ep_idx + 1}/{args.num_trials} …", log_file)

            client.reset()
            success, replay_images = run_episode(
                env=env,
                task_description=task_description,
                client=client,
                initial_state=initial_state,
                max_steps=max_steps,
                num_steps_wait=args.num_steps_wait,
                save_video_flag=args.save_video,
                log_file=log_file,
            )

            task_episodes  += 1
            total_episodes += 1
            if success:
                task_successes  += 1
                total_successes += 1

            log_message(
                f"  → success={success} | "
                f"task_sr={task_successes}/{task_episodes} | "
                f"total_sr={total_successes}/{total_episodes} "
                f"({total_successes / total_episodes * 100:.1f}%)",
                log_file,
            )

            if args.save_video and replay_images:
                save_video(
                    images=replay_images,
                    episode_idx=total_episodes,
                    success=success,
                    task_description=task_description,
                    video_dir=str(task_dir),
                    log_file=log_file,
                )

        task_sr = task_successes / task_episodes if task_episodes > 0 else 0.0
        log_message(
            f"  Task SR: {task_successes}/{task_episodes} = "
            f"{task_sr:.4f} ({task_sr * 100:.1f}%)",
            log_file,
        )
        log_message("=" * 60, log_file)
        env.close()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    final_sr = total_successes / total_episodes if total_episodes > 0 else 0.0

    log_message("\n" + "=" * 60, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message(f"  Task suite     : {args.task_suite}", log_file)
    log_message(f"  Total episodes : {total_episodes}", log_file)
    log_message(f"  Total successes: {total_successes}", log_file)
    log_message(f"  Overall SR     : {final_sr:.4f} ({final_sr * 100:.1f}%)", log_file)
    log_message("=" * 60, log_file)

    log_file.close()
    return final_sr

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate X-VLA on original LIBERO benchmark"
    )

    # Server
    p.add_argument("--server-host", default="localhost",
                   help="X-VLA server host (default: localhost)")
    p.add_argument("--server-port", type=int, default=8010,
                   help="X-VLA server port (default: 8010)")
    p.add_argument("--domain-id", type=int, default=3,
                   help="X-VLA domain ID for LIBERO (default: 3)")
    p.add_argument("--steps", type=int, default=10,
                   help="Diffusion denoising steps per inference call (default: 10)")

    # Task
    p.add_argument("--task-suite", default="libero_spatial",
                   choices=VALID_SUITES,
                   help=f"LIBERO task suite. Choices: {VALID_SUITES}")
    p.add_argument("--num-trials", type=int, default=10,
                   help="Rollouts per task (default: 10)")
    p.add_argument("--num-steps-wait", type=int, default=10,
                   help="Stabilisation steps (relative mode) before switching to "
                        "absolute control and querying the policy (default: 10)")
    p.add_argument("--env-res", type=int, default=256,
                   help="Camera resolution in pixels (default: 256)")

    # Output
    p.add_argument("--save-video", action="store_true", default=True,
                   help="Save MP4 rollout videos (default: True)")
    p.add_argument("--no-save-video", dest="save_video", action="store_false",
                   help="Disable video saving")
    p.add_argument("--run-note", default="",
                   help="Optional note appended to the run ID")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    final_sr = run_eval(args)
    print(f"\nFinal success rate: {final_sr:.4f} ({final_sr * 100:.1f}%)")
