"""
DyniSafety evaluator for SimplerEnv / ManiSkill2.

Drop-in replacement for ``maniskill2_evaluator`` that wraps each episode
environment with ``DyniSafetyWrapper`` when ``dynamic_level`` is 1–3, or uses
the raw ManiSkill2 env for ``dynamic_level==0`` (baseline). Collects collision
metrics when the wrapper is active.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, TextIO

import numpy as np
from transforms3d.euler import quat2euler

from dynisafety import output_layout as bench_paths
from dynisafety.wrapper import DyniSafetyConfig, DyniSafetyWrapper
from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

logger = logging.getLogger(__name__)


@dataclass
class BenchOutputContext:
    """LIBERO-aligned run: one log file + monotonic episode index for video names."""

    output_root: str
    model_family: str
    bench_task: str
    level: int
    run_id_note: Optional[str]
    fixed_date_time: str
    log_file: TextIO
    log_path: str
    episode_seq: int = 0

    def next_episode_index(self) -> int:
        self.episode_seq += 1
        return self.episode_seq


def run_dynisafety_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    dynisafety_config: DyniSafetyConfig,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
    bench_output: Optional[BenchOutputContext] = None,
    stop_on_success: bool = False,
):
    """Run a single episode with dynamic safety disturbance.

    Returns (success: bool, collision_metrics: dict).
    """
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        additional_env_build_kwargs = ray_tracing_dict

    # Build base env; level 0 = baseline SimplerEnv (no DyniSafety wrapper)
    base_env = build_maniskill2_env(env_name, **additional_env_build_kwargs, **kwargs)
    if dynisafety_config.dynamic_level == 0:
        env = base_env
    else:
        env = DyniSafetyWrapper(base_env, config=dynisafety_config)

    # Reset
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }

    obs, _ = env.reset(options=env_reset_options)
    is_final_subtask = env.is_final_subtask()

    if instruction is not None:
        task_description = instruction
    else:
        task_description = env.get_language_instruction()
    print(task_description)

    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    model.reset(task_description)

    timestep = 0
    success = "failure"
    task_descriptions = []

    while not (predicted_terminated or truncated):
        raw_action, action = model.step(
            image, task_description, eef_pos=obs["agent"]["eef_pos"]
        )
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                predicted_terminated = False
                env.advance_to_next_subtask()

        obs, reward, done, truncated, info = env.step(
            np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]]
            ),
        )

        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.is_final_subtask()

        dynisafety_info = info.get("dynisafety", {})
        step_col = dynisafety_info.get("step_collided", False)
        print(
            f"  step {timestep}  "
            f"collision={'YES' if step_col else 'no'}  "
            f"total_col_steps={dynisafety_info.get('collision_steps', 0)}  "
            f"{info}"
        )

        image = get_image_from_maniskill2_obs_dict(
            env, obs, camera_name=obs_camera_name
        )
        images.append(image)
        task_descriptions.append(task_description)
        timestep += 1

        if stop_on_success and done and is_final_subtask:
            break

    episode_stats = info.get("episode_stats", {})
    collision_metrics = getattr(env, "collision_metrics", None)
    if collision_metrics is None:
        collision_metrics = {"collision_steps": 0, "collision_events": 0}
    ok = success == "success"
    col_steps = collision_metrics["collision_steps"]
    obstacle_toppled = bool(collision_metrics.get("obstacle_toppled", False))
    max_tilt_deg = float(collision_metrics.get("max_tilt_deg", 0.0) or 0.0)
    topple_threshold_deg = float(
        collision_metrics.get("topple_threshold_deg", dynisafety_config.topple_threshold_deg)
        or dynisafety_config.topple_threshold_deg
    )
    max_collision_force = float(collision_metrics.get("max_collision_force", 0.0) or 0.0)
    max_collision_impulse = float(collision_metrics.get("max_collision_impulse", 0.0) or 0.0)
    force_anomaly = bool(collision_metrics.get("force_anomaly", False))
    force_anomaly_steps = int(collision_metrics.get("force_anomaly_steps", 0) or 0)
    pre_physics_collision_prevents = int(
        collision_metrics.get("pre_physics_collision_prevents", 0) or 0
    )
    max_collision_pair = collision_metrics.get("max_collision_pair")
    max_collision_step = collision_metrics.get("max_collision_step")
    force_anomaly_pairs = collision_metrics.get("force_anomaly_pairs", {}) or {}
    if isinstance(max_collision_pair, (list, tuple)) and len(max_collision_pair) == 2:
        max_collision_pair_str = f"{max_collision_pair[0]}->{max_collision_pair[1]}"
    else:
        max_collision_pair_str = ""
    anomaly_pair_str = ";".join(
        f"{key}:{value}" for key, value in sorted(force_anomaly_pairs.items())
    )

    # -- Save video --
    if bench_output is not None:
        ep_idx = bench_output.next_episode_index()
        vid_dir = bench_paths.bench_episode_video_dir(
            bench_output.output_root,
            bench_output.model_family,
            bench_output.bench_task,
            bench_output.level,
            task_description,
            task_id=0,
            date_time=bench_output.fixed_date_time,
        )
        video_name = bench_paths.libero_style_mp4_basename(
            ep_idx, ok, task_description, date_time=bench_output.fixed_date_time
        )
        video_path = os.path.join(vid_dir, video_name)
        write_video(video_path, images, fps=5)
        lf = bench_output.log_file
        lf.write(
            f"episode={ep_idx} success={ok} collision_steps={col_steps} "
            f"obstacle_toppled={obstacle_toppled} "
            f"max_tilt_deg={max_tilt_deg:.1f} "
            f"topple_threshold_deg={topple_threshold_deg:.1f} "
            f"max_collision_force_N={max_collision_force:.2f} "
            f"max_collision_impulse_Ns={max_collision_impulse:.4f} "
            f"force_anomaly={force_anomaly} "
            f"force_anomaly_steps={force_anomaly_steps} "
            f"pre_physics_collision_prevents={pre_physics_collision_prevents} "
            f"max_collision_pair={max_collision_pair_str} "
            f"max_collision_step={'' if max_collision_step is None else max_collision_step} "
            f"force_anomaly_pairs={anomaly_pair_str} "
            f"video={video_path}\n"
        )
        lf.flush()
        logger.info("Saved bench-layout MP4: %s", video_path)
    else:
        level_tag = f"dynlevel{dynisafety_config.dynamic_level}"
        env_save_name = env_name
        for k, v in additional_env_build_kwargs.items():
            env_save_name = env_save_name + f"_{k}_{v}"
        if additional_env_save_tags is not None:
            env_save_name = env_save_name + f"_{additional_env_save_tags}"
        env_save_name = env_save_name + f"_{level_tag}"

        if obj_variation_mode == "xy":
            video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
        elif obj_variation_mode == "episode":
            video_name = f"{success}_obj_episode_{obj_episode_id}"
        for k, v in episode_stats.items():
            video_name = video_name + f"_{k}_{v}"
        video_name += f"_col_steps_{col_steps}"
        video_name += ".mp4"

        if rgb_overlay_path is not None:
            rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
        else:
            rgb_overlay_path_str = "None"
        r, p, y = quat2euler(robot_init_quat)
        video_path = (
            f"{scene_name}/{control_mode}/{env_save_name}/"
            f"rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}"
            f"_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
        )
        video_path = os.path.join(logging_dir, video_path)
        write_video(video_path, images, fps=5)

    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    if hasattr(model, "visualize_epoch") and callable(model.visualize_epoch):
        model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return ok, collision_metrics


def dynisafety_evaluator(model, args, dynisafety_config: DyniSafetyConfig, dyn_args=None):
    """Evaluate model with dynamic safety disturbance across all configurations."""
    layout = "legacy"
    bench_ctx: Optional[BenchOutputContext] = None
    if dyn_args is not None:
        layout = getattr(dyn_args, "output_layout", "bench")

    if layout == "bench":
        bench_task = (getattr(dyn_args, "bench_task", "") or "").strip()
        if not bench_task:
            bench_task = bench_paths.infer_bench_task_from_env_name(args.env_name)
        out_root = os.path.abspath(getattr(dyn_args, "output_root", "./outputs"))
        note = getattr(dyn_args, "run_id_note", "") or ""
        fixed_dt = bench_paths.DATE_TIME
        log_path = bench_paths.bench_log_filepath(
            out_root,
            args.policy_model,
            bench_task,
            dynisafety_config.dynamic_level,
            run_id_note=note if note else None,
            date_time=fixed_dt,
        )
        log_f = open(log_path, "w", encoding="utf-8")
        log_f.write(
            "DyniSafety-SimplerEnv (bench output layout)\n"
            f"  policy_model={args.policy_model}\n"
            f"  bench_task={bench_task}\n"
            f"  dynamic_level={dynisafety_config.dynamic_level}\n"
            f"  collision_force_equiv_dt={dynisafety_config.collision_force_equiv_dt}\n"
            f"  collision_impulse_point_reduce={dynisafety_config.collision_impulse_point_reduce}\n"
            f"  collision_force_report_window_s={dynisafety_config.collision_force_report_window_s}\n"
            f"  env_name={args.env_name}\n"
            f"  scene_name={args.scene_name}\n"
            f"  output_root={out_root}\n"
            f"  log_file={log_path}\n"
            "---\n"
        )
        log_f.flush()
        bench_ctx = BenchOutputContext(
            output_root=out_root,
            model_family=args.policy_model,
            bench_task=bench_task,
            level=dynisafety_config.dynamic_level,
            run_id_note=note if note else None,
            fixed_date_time=fixed_dt,
            log_file=log_f,
            log_path=log_path,
        )
        logger.info("Bench layout: logging to %s", log_path)

    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []
    collision_arr = []

    try:
        for robot_init_x in args.robot_init_xs:
            for robot_init_y in args.robot_init_ys:
                for robot_init_quat in args.robot_init_quats:
                    kwargs = dict(
                        model=model,
                        ckpt_path=args.ckpt_path,
                        robot_name=args.robot,
                        env_name=args.env_name,
                        scene_name=args.scene_name,
                        robot_init_x=robot_init_x,
                        robot_init_y=robot_init_y,
                        robot_init_quat=robot_init_quat,
                        control_mode=control_mode,
                        dynisafety_config=dynisafety_config,
                        additional_env_build_kwargs=args.additional_env_build_kwargs,
                        rgb_overlay_path=args.rgb_overlay_path,
                        control_freq=args.control_freq,
                        sim_freq=args.sim_freq,
                        max_episode_steps=args.max_episode_steps,
                        enable_raytracing=args.enable_raytracing,
                        additional_env_save_tags=args.additional_env_save_tags,
                        obs_camera_name=args.obs_camera_name,
                        logging_dir=args.logging_dir,
                        bench_output=bench_ctx,
                        stop_on_success=bool(getattr(dyn_args, "stop_on_success", False)),
                    )
                    if args.obj_variation_mode == "xy":
                        for obj_init_x in args.obj_init_xs:
                            for obj_init_y in args.obj_init_ys:
                                ok, col = run_dynisafety_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                                success_arr.append(ok)
                                collision_arr.append(col)
                    elif args.obj_variation_mode == "episode":
                        for obj_episode_id in range(
                            args.obj_episode_range[0], args.obj_episode_range[1]
                        ):
                            ok, col = run_dynisafety_eval_single_episode(
                                obj_episode_id=obj_episode_id, **kwargs
                            )
                            success_arr.append(ok)
                            collision_arr.append(col)
                    else:
                        raise NotImplementedError()
    finally:
        if bench_ctx is not None and bench_ctx.log_file is not None:
            bench_ctx.log_file.close()
            bench_ctx.log_file = None  # type: ignore[assignment]

    # Summary
    n = len(success_arr)
    n_success = sum(success_arr)
    n_collided = sum(1 for c in collision_arr if c["collision_steps"] > 0)
    n_toppled = sum(1 for c in collision_arr if c.get("obstacle_toppled", False))
    total_col_steps = sum(c["collision_steps"] for c in collision_arr)
    max_tilt = max((float(c.get("max_tilt_deg", 0.0) or 0.0) for c in collision_arr), default=0.0)
    forces = [float(c.get("max_collision_force", 0.0) or 0.0) for c in collision_arr]
    impulses = [float(c.get("max_collision_impulse", 0.0) or 0.0) for c in collision_arr]
    force_anomaly_episodes = sum(1 for c in collision_arr if c.get("force_anomaly", False))
    total_force_anomaly_steps = sum(
        int(c.get("force_anomaly_steps", 0) or 0) for c in collision_arr
    )
    total_pre_physics_prevents = sum(
        int(c.get("pre_physics_collision_prevents", 0) or 0) for c in collision_arr
    )
    avg_force = sum(forces) / len(forces) if forces else 0.0
    peak_force = max(forces) if forces else 0.0
    avg_impulse = sum(impulses) / len(impulses) if impulses else 0.0
    peak_impulse = max(impulses) if impulses else 0.0

    forces_no_anomaly = [
        float(c.get("max_collision_force", 0.0) or 0.0)
        for c in collision_arr
        if not c.get("force_anomaly", False)
    ]
    impulses_no_anomaly = [
        float(c.get("max_collision_impulse", 0.0) or 0.0)
        for c in collision_arr
        if not c.get("force_anomaly", False)
    ]
    n_no_anomaly = len(forces_no_anomaly)
    avg_force_no_anomaly = (
        sum(forces_no_anomaly) / n_no_anomaly if n_no_anomaly else 0.0
    )
    avg_impulse_no_anomaly = (
        sum(impulses_no_anomaly) / n_no_anomaly if n_no_anomaly else 0.0
    )
    summary_lines = [
        "=" * 60,
        "DyniSafety Evaluation Summary",
        f"  Dynamic level : {dynisafety_config.dynamic_level}",
        (
            f"  Collision force equiv dt: {dynisafety_config.collision_force_equiv_dt} "
            f"(impulse/dt; impact_window => dt=collision_force_report_window_s)"
        ),
        f"  Contact impulse point reduce: {dynisafety_config.collision_impulse_point_reduce}",
        f"  Force report window (s): {dynisafety_config.collision_force_report_window_s}",
        f"  Total episodes: {n}",
        (
            f"  Success rate  : {n_success}/{n} ({n_success / n * 100:.1f}%)"
            if n
            else "  No episodes"
        ),
        (
            f"  Collision episodes: {n_collided}/{n} ({n_collided / n * 100:.1f}%)"
            if n
            else ""
        ),
        (
            f"  Topple episodes: {n_toppled}/{n} ({n_toppled / n * 100:.1f}%)"
            if n
            else ""
        ),
        f"  Max tilt deg: {max_tilt:.1f}",
        f"  Topple threshold deg: {dynisafety_config.topple_threshold_deg:.1f}",
        f"  Avg max collision force (N): {avg_force:.2f}",
        f"  Peak collision force (N): {peak_force:.2f}",
        (
            f"  Avg max collision force (N), excl. force-anomaly episodes: "
            f"{avg_force_no_anomaly:.2f} ({n_no_anomaly}/{n})"
        ),
        f"  Avg max collision impulse (Ns): {avg_impulse:.4f}",
        f"  Peak collision impulse (Ns): {peak_impulse:.4f}",
        (
            f"  Avg max collision impulse (Ns), excl. force-anomaly episodes: "
            f"{avg_impulse_no_anomaly:.4f} ({n_no_anomaly}/{n})"
        ),
        f"  Total collision steps: {total_col_steps}",
        f"  Force anomaly episodes: {force_anomaly_episodes}/{n}",
        f"  Force anomaly steps: {total_force_anomaly_steps}",
        f"  Pre-physics collision prevents: {total_pre_physics_prevents}",
        "=" * 60,
    ]
    for line in summary_lines:
        print(line)
    if bench_ctx is not None:
        with open(bench_ctx.log_path, "a", encoding="utf-8") as lf:
            lf.write("\n")
            for line in summary_lines:
                lf.write(line + "\n")

    return success_arr, collision_arr
