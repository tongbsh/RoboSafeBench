"""Output path layout for DyniSafety-SimplerEnv (logs + videos)."""

from __future__ import annotations

import os
import time

# Timestamp helpers
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def task_video_subdir(task_id: int, task_description: str) -> str:
    """Filesystem-safe per-task folder under the rollout base path (LIBERO-style)."""
    slug = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    max_slug = 120
    if len(slug) > max_slug:
        slug = slug[:max_slug].rstrip("_")
    return f"task{task_id}_{slug}"


def bench_log_filepath(
    output_root: str,
    model_family: str,
    bench_task: str,
    level: int,
    run_id_note: str | None = None,
    date_time: str | None = None,
) -> str:
    """Full path to the EVAL-dynisafety-*.txt log file."""
    dt = date_time or DATE_TIME
    log_dir = os.path.join(output_root, model_family, "logs")
    os.makedirs(log_dir, exist_ok=True)
    if run_id_note:
        name = f"EVAL-dynisafety-{dt}--{run_id_note}.txt"
    else:
        name = f"EVAL-dynisafety-{model_family}-{bench_task}_level{level}-{dt}.txt"
    return os.path.join(log_dir, name)


def bench_episode_video_dir(
    output_root: str,
    model_family: str,
    bench_task: str,
    level: int,
    task_description: str,
    task_id: int = 0,
    date_time: str | None = None,
) -> str:
    """Directory for one episode's MP4 (mirrors LIBERO: suite_level/date/task_slug/)."""
    dt = date_time or DATE_TIME
    date_part = dt.split("-")[0]
    suite_folder = f"{bench_task}_level{level}"
    sub = task_video_subdir(task_id, task_description)
    d = os.path.join(output_root, model_family, "videos", suite_folder, date_part, sub)
    os.makedirs(d, exist_ok=True)
    return d


def libero_style_mp4_basename(
    episode_idx: int,
    success: bool,
    task_description: str,
    date_time: str | None = None,
) -> str:
    """Return a filesystem-safe video filename for the given task description and timestamp."""
    dt = date_time or DATE_TIME
    processed = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )
    return (
        f"{dt}--dynisafety--episode={episode_idx}--success={success}--task={processed}.mp4"
    )


def infer_bench_task_from_env_name(env_name: str) -> str:
    """Fallback bench_task when --bench-task is omitted."""
    if env_name.startswith("GraspSingle") or "CokeCan" in env_name:
        return "pick_coke_can"
    if env_name.startswith("MoveNear"):
        return "move_near"
    if "Drawer" in env_name:
        return "drawer"
    if "PutCarrotOnPlate" in env_name:
        return "bridge_carrot"
    if "StackGreenCubeOnYellowCube" in env_name:
        return "bridge_stack"
    if "PutSpoonOnTableCloth" in env_name:
        return "bridge_spoon"
    if "PutEggplantInBasket" in env_name:
        return "bridge_eggplant"
    return "simplerenv"
