"""
run_libero_spatialvla_eval.py

Evaluates the SpatialVLA policy on the original (unmodified) LIBERO benchmark.

The script connects to a running SpatialVLA FastAPI inference server and rolls
out each task suite.  Videos and logs are saved under:
    dynisafety_libero/outputs/spatialvla/source_libero/

Prerequisites
-------------
1. Start the SpatialVLA server (in 'spatialvla' conda env):
       bash dynisafety_libero/scripts/spatialVLA/start_spatialvla_server.sh \\
           --lora-path download_checkpoints/spatialVLA/spatialvla-lora-libero-spatial/checkpoint-16550 \\
           --unnorm-key libero_spatial

2. Run this script (in 'openvla' conda env, where LIBERO is installed):
       bash dynisafety_libero/scripts/spatialVLA/source_libero_eval/run_libero_eval.sh \\
           --task-suite libero_spatial \\
           --unnorm-key libero_spatial

Supported task suites
---------------------
  libero_spatial | libero_object | libero_goal | libero_10 | libero_90

Recommended unnorm-key mapping
-------------------------------
  The processor statistics use "_no_noops/1.0.0" suffix — NOT the bare dataset name.
  Passing the wrong key causes the server to fall back to an unrelated dataset's stats
  (e.g. Bridge) which breaks action un-normalisation and the robot fails every episode.

  libero_spatial  →  libero_spatial_no_noops/1.0.0
  libero_object   →  libero_object_no_noops/1.0.0
  libero_goal     →  libero_goal_no_noops/1.0.0
  libero_10       →  libero_10_no_noops/1.0.0   (used by the libero-long LoRA)
  libero_90       →  bridge_orig/1.0.0           (no fine-tuned LoRA available)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap: add simulator_env/LIBERO so `libero` is importable.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJ_ROOT   = (_SCRIPT_DIR / "../../../..").resolve()   # dynamic_safty_bench/

_LIBERO_PATH = str(_PROJ_ROOT / "simulator_env" / "LIBERO")
if _LIBERO_PATH not in sys.path:
    sys.path.insert(0, _LIBERO_PATH)

# ---------------------------------------------------------------------------
# LIBERO imports (requires the 'openvla' conda env where LIBERO is installed)
# ---------------------------------------------------------------------------
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import imageio
import json_numpy
import numpy as np
import requests

json_numpy.patch()

# ---------------------------------------------------------------------------
# Logging setup
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

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
    "libero_90":      400,
}

VALID_SUITES = list(TASK_MAX_STEPS.keys())

# ---------------------------------------------------------------------------
# LIBERO env helpers
# ---------------------------------------------------------------------------

def get_libero_env(task, resolution: int = 256):
    task_description = task.language
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
    return env, task_description


def get_agentview_image(obs: dict) -> np.ndarray:
    """Third-person camera image, rotated 180° to match training preprocessing."""
    img = obs["agentview_image"]
    return img[::-1, ::-1]


def dummy_action() -> list:
    """No-op action (closed gripper) used during stabilisation."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


# ---------------------------------------------------------------------------
# SpatialVLA HTTP client
# ---------------------------------------------------------------------------

class SpatialVLAClient:
    """Thin HTTP client for the SpatialVLA FastAPI server."""

    def __init__(self, host: str, port: int, unnorm_key: Optional[str],
                 timeout: int = 300):
        self._act_url    = f"http://{host}:{port}/act"
        self._health_url = f"http://{host}:{port}/healthz"
        self._unnorm_key = unnorm_key
        self._wait_for_server(timeout)

    def _wait_for_server(self, timeout: int) -> None:
        deadline = time.monotonic() + timeout
        logger.info("Waiting for SpatialVLA server at %s …", self._health_url)
        while time.monotonic() < deadline:
            try:
                r = requests.get(self._health_url, timeout=5)
                if r.status_code == 200:
                    info = r.json()
                    logger.info(
                        "Server ready — unnorm_key=%s, lora_loaded=%s",
                        info.get("default_unnorm_key"), info.get("lora_loaded"),
                    )
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(
            f"SpatialVLA server did not become ready within {timeout}s."
        )

    def predict(self, image: np.ndarray, task: str) -> List[np.ndarray]:
        """Send one observation and return a list of 7-dim action arrays."""
        payload: dict = {
            "image": json_numpy.dumps(image),
            "task":  task,
        }
        if self._unnorm_key:
            payload["unnorm_key"] = self._unnorm_key

        try:
            resp = requests.post(self._act_url, json=payload, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"SpatialVLA server request failed: {e}") from e

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"SpatialVLA server error:\n{data['error']}")

        actions = np.array(data["actions"], dtype=np.float32)   # (chunk, 7)

        # Gripper convention fix:
        # SpatialVLA decode_actions passes gripper through as {0.0=close, 1.0=open}
        # (the gripper dimension has mask=False in processor statistics so it is
        # NOT unnormalized).
        # Verified: mask=[T,T,T,T,T,T,False], q01[6]=0.0, q99[6]=1.0
        #
        # LIBERO env convention (from robot_utils.invert_gripper_action docstring):
        #   -1 = open gripper,  +1 = close gripper
        # (dummy_action = [0,0,0,0,0,0,-1] confirms -1 = open during stabilisation)
        #
        # model 1.0(open) → env -1.0(open),  model 0.0(close) → env +1.0(close)
        actions[:, -1] = np.where(actions[:, -1] > 0.5, -1.0, 1.0)

        return [actions[i] for i in range(len(actions))]


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_message(msg: str, log_file=None) -> None:
    logger.info(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

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
        f"{DATE_TIME}--spatialvla--episode={episode_idx}"
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
    client: SpatialVLAClient,
    initial_state,
    max_steps: int,
    num_steps_wait: int,
    num_open_loop_steps: int,
    save_video_flag: bool,
    log_file=None,
):
    env.reset()
    obs = env.set_init_state(initial_state)

    from collections import deque
    action_queue: deque = deque()

    t = 0
    replay_images: List[np.ndarray] = []
    success = False

    try:
        while t < max_steps + num_steps_wait:
            # Stabilisation phase
            if t < num_steps_wait:
                obs, _, _, _ = env.step(dummy_action())
                t += 1
                continue

            img = get_agentview_image(obs)
            if save_video_flag:
                replay_images.append(img)

            # Query policy when action queue is empty
            if len(action_queue) == 0:
                actions = client.predict(img, task_description)
                action_queue.extend(actions[:num_open_loop_steps])

            action = action_queue.popleft()
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
    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    out_base   = _PROJ_ROOT / "dynisafety_libero" / "outputs" / "spatialvla" / "source_libero"
    log_dir    = out_base / "logs"
    video_base = out_base / "videos" / args.task_suite / DATE

    log_dir.mkdir(parents=True, exist_ok=True)
    video_base.mkdir(parents=True, exist_ok=True)

    run_id = f"EVAL-spatialvla-{args.task_suite}-{DATE_TIME}"
    if args.run_note:
        run_id += f"--{args.run_note}"

    log_path = log_dir / f"{run_id}.txt"
    log_file = open(log_path, "w")
    log_message(f"Run ID       : {run_id}", log_file)
    log_message(f"Task suite   : {args.task_suite}", log_file)
    log_message(f"Server       : {args.server_host}:{args.server_port}", log_file)
    log_message(f"Unnorm key   : {args.unnorm_key or 'server-default'}", log_file)
    log_message(f"Trials/task  : {args.num_trials}", log_file)
    log_message(f"Open-loop N  : {args.num_open_loop_steps}", log_file)
    log_message(f"Log file     : {log_path}", log_file)

    # ------------------------------------------------------------------
    # SpatialVLA client
    # ------------------------------------------------------------------
    client = SpatialVLAClient(
        host=args.server_host,
        port=args.server_port,
        unnorm_key=args.unnorm_key or None,
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
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks  = task_suite.n_tasks
    max_steps  = TASK_MAX_STEPS.get(args.task_suite, 300)

    log_message(f"Number of tasks: {num_tasks}", log_file)
    log_message("=" * 60, log_file)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    total_episodes  = 0
    total_successes = 0

    for task_id in range(num_tasks):
        task             = task_suite.get_task(task_id)
        initial_states   = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=args.env_res)

        task_dir = video_base / task_description.lower().replace(" ", "_")[:60]
        task_dir.mkdir(parents=True, exist_ok=True)

        task_episodes  = 0
        task_successes = 0

        log_message(f"\n[Task {task_id + 1}/{num_tasks}] {task_description}", log_file)

        for ep_idx in range(args.num_trials):
            initial_state = initial_states[ep_idx % len(initial_states)]
            log_message(f"  Episode {ep_idx + 1}/{args.num_trials} …", log_file)

            success, replay_images = run_episode(
                env=env,
                task_description=task_description,
                client=client,
                initial_state=initial_state,
                max_steps=max_steps,
                num_steps_wait=args.num_steps_wait,
                num_open_loop_steps=args.num_open_loop_steps,
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
            f"  Task SR: {task_successes}/{task_episodes} = {task_sr:.4f} ({task_sr * 100:.1f}%)",
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
        description="Evaluate SpatialVLA on original LIBERO benchmark"
    )

    # Server
    p.add_argument("--server-host", default="localhost",
                   help="SpatialVLA server host (default: localhost)")
    p.add_argument("--server-port", type=int, default=8112,
                   help="SpatialVLA server port (default: 8112)")
    p.add_argument("--unnorm-key", default="",
                   help="Action un-normalisation key (e.g. libero_spatial). "
                        "Empty = use server default.")

    # Task
    p.add_argument("--task-suite", default="libero_spatial",
                   choices=VALID_SUITES,
                   help=f"LIBERO task suite. Choices: {VALID_SUITES}")
    p.add_argument("--num-trials", type=int, default=10,
                   help="Number of rollouts per task (default: 10)")
    p.add_argument("--num-steps-wait", type=int, default=10,
                   help="Stabilisation steps before querying policy (default: 10)")
    p.add_argument("--num-open-loop-steps", type=int, default=1,
                   help="Open-loop steps before re-querying policy (default: 1)")
    p.add_argument("--env-res", type=int, default=256,
                   help="Environment camera resolution in pixels (default: 256)")

    # Output
    p.add_argument("--save-video", action="store_true", default=True,
                   help="Save MP4 rollout videos (default: True)")
    p.add_argument("--no-save-video", dest="save_video", action="store_false",
                   help="Disable video saving")
    p.add_argument("--run-note", default="",
                   help="Optional note appended to run ID")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    final_sr = run_eval(args)
    print(f"\nFinal success rate: {final_sr:.4f} ({final_sr * 100:.1f}%)")
