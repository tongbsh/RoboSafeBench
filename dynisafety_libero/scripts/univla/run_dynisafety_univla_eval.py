"""
run_dynisafety_univla_eval.py

DyniSafety-LIBERO evaluation with UniVLA over HTTP, mirroring the episode / metrics /
video pipeline of ``scripts/x-vla/run_dynisafety_xvla_eval.py`` and the inference path of
``vlaModelHome/UniVLA/experiments/robot/libero/run_libero_eval.py`` (flipped agentview,
proprio state, latent action + temporal ActionDecoder on server).

Prerequisites (terminal 1 — ``conda activate univla``)::

    bash dynisafety_libero/scripts/univla/start_univla_server.sh \\
        --task-suite safelibero_object --port 8020

Run (terminal 2 — benchmark env, e.g. openvla-oft)::

    python dynisafety_libero/scripts/univla/run_dynisafety_univla_eval.py \\
        --task_suite_name safelibero_object \\
        --server_host localhost \\
        --server_port 8020

The server must use an ``unnorm_key`` consistent with the checkpoint (e.g.
``libero_object_no_noops`` for object suite). Do **not** set ``use_delta=False``;
UniVLA follows standard LIBERO delta control like OpenVLA.

Note: do not enable ``from __future__ import annotations`` here — draccus needs a
real dataclass type from the wrapped function's parameter annotation.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

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
import numpy as np
import tqdm
import wandb
from libero.libero import benchmark

import dynisafety_libero.run_dynisafety_eval as rde
from dynisafety_libero.collision import (
    compute_body_tilt_deg,
    compute_contact_force,
    detect_collisions,
    get_obstacle_geom_ids,
    get_robot_geom_ids,
    get_task_object_geom_ids,
)
from dynisafety_libero.env_utils import (
    get_dynisafety_dummy_action,
    get_dynisafety_env,
    pad_init_state_for_hand,
    save_rollout_video,
)
from dynisafety_libero.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    ReactivityAnalyzer,
    compute_safety_penalty,
)
from dynisafety_libero.univla_http_client import UniVLAHTTPClient
from dynisafety_libero.xvla_source_client import get_flipped_agentview


# Expected dataset_statistics.json keys for UniVLA LIBERO checkpoints (must match server --unnorm_key).
SAFELIBERO_TO_UNNORM_KEY = {
    "safelibero_spatial": "libero_spatial_no_noops",
    "safelibero_object": "libero_object_no_noops",
    "safelibero_goal": "libero_goal_no_noops",
    "safelibero_long": "libero_10_no_noops",
}


def _task_max_steps(cfg: rde.GenerateConfig) -> int:
    key = cfg.task_suite_name
    if isinstance(key, str):
        key = rde.TaskSuite(key)
    return rde.TASK_MAX_STEPS[key]


def _expected_univla_unnorm_key(task_suite_name: str) -> str:
    return SAFELIBERO_TO_UNNORM_KEY.get(task_suite_name, "(no default mapping; set server unnorm_key manually)")


@dataclass
class UnivlaDyniSafetyConfig(rde.GenerateConfig):
    """Defaults for UniVLA DyniSafety eval; logs and outputs use model_family ``univla``."""

    model_family: str = "univla"
    pretrained_checkpoint: str = ""
    action_decoder_path: str = ""


def _make_univla_client(cfg: rde.GenerateConfig) -> UniVLAHTTPClient:
    return UniVLAHTTPClient(host=cfg.server_host, port=cfg.server_port)


def run_episode_univla(
    cfg: rde.GenerateConfig,
    env,
    task_description: str,
    task_id: int,
    client: UniVLAHTTPClient,
    initial_state=None,
    log_file=None,
    episode_idx: int = 0,
) -> tuple:
    env.reset()
    client.reset()

    if initial_state is not None:
        padded_state = pad_init_state_for_hand(initial_state, env)
        obs = env.set_init_state(padded_state)
    else:
        obs = env.get_observation()

    obstacle_ctrl, arm_ctrl, drop_ctrl = rde._setup_controller(
        cfg,
        env,
        task_id,
        episode_idx,
        log_file,
        task_description=task_description,
    )

    robot_geom_ids = get_robot_geom_ids(env)
    task_obj_geom_ids = get_task_object_geom_ids(env)
    robot_geom_ids = robot_geom_ids | task_obj_geom_ids
    obstacle_geom_ids = get_obstacle_geom_ids(env)
    if obstacle_ctrl is not None:
        obstacle_geom_ids = obstacle_ctrl.obstacle_geom_ids
        obstacle_ctrl.set_robot_geom_ids(robot_geom_ids)
    elif arm_ctrl is not None:
        obstacle_geom_ids = arm_ctrl.obstacle_geom_ids
        arm_ctrl.set_robot_geom_ids(robot_geom_ids)
    elif drop_ctrl is not None:
        obstacle_geom_ids = drop_ctrl.obstacle_geom_ids
    has_obstacles = len(obstacle_geom_ids) > 0

    dyn_obstacle_geom_ids = set()
    if obstacle_ctrl is not None:
        dyn_obstacle_geom_ids = obstacle_ctrl.obstacle_geom_ids
    elif arm_ctrl is not None:
        dyn_obstacle_geom_ids = arm_ctrl.obstacle_geom_ids
    elif drop_ctrl is not None:
        dyn_obstacle_geom_ids = drop_ctrl.obstacle_geom_ids

    active_ctrl = obstacle_ctrl or arm_ctrl or drop_ctrl

    topple_body_id = None
    topple_ref_quat = None
    if obstacle_ctrl is not None:
        topple_body_id = obstacle_ctrl._body_id
        topple_ref_quat = obstacle_ctrl.final_quat
    elif drop_ctrl is not None:
        topple_body_id = drop_ctrl._body_id
        topple_ref_quat = drop_ctrl.target_quat

    t = 0
    replay_images = []
    max_steps = _task_max_steps(cfg)

    metrics = EpisodeMetrics()
    eef_positions = []

    success = False
    sim_unstable = False

    try:
        while t < max_steps + cfg.num_steps_wait:
            task_step = t - cfg.num_steps_wait

            if active_ctrl is not None:
                active_ctrl.update_before_step(task_step)

            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(
                    get_dynisafety_dummy_action(cfg.model_family)
                )
                if np.any(np.isnan(env.env.sim.data.qpos)):
                    rde.log_message(
                        "Simulation unstable (NaN in qpos during wait phase), aborting episode.",
                        log_file,
                    )
                    sim_unstable = True
                    break
                t += 1
                continue

            eef_pos = obs["robot0_eef_pos"].copy()
            eef_positions.append(eef_pos)

            main_img = get_flipped_agentview(obs)
            obs_dict, _ = rde.prepare_observation(obs)
            proprio = obs_dict["state"]

            if cfg.save_video:
                replay_images.append(main_img.copy())

            action = client.act(main_img, task_description, proprio)
            obs, reward, done, info = env.step(action.tolist())

            if np.any(np.isnan(env.env.sim.data.qpos)):
                rde.log_message("Simulation unstable (NaN in qpos), aborting episode.", log_file)
                sim_unstable = True
                break

            step_has_collision = False
            step_collision_events = 0
            step_has_dyn_collision = False
            step_dyn_collision_events = 0

            if has_obstacles:
                colliding = detect_collisions(env, robot_geom_ids, obstacle_geom_ids)
                if colliding:
                    step_has_collision = True
                    step_collision_events += len(colliding)
                    metrics.collision_details.append((t, list(set(colliding))))

            if dyn_obstacle_geom_ids:
                dyn_colliding = detect_collisions(env, robot_geom_ids, dyn_obstacle_geom_ids)
                if dyn_colliding:
                    step_has_dyn_collision = True
                    step_dyn_collision_events += len(dyn_colliding)

            if active_ctrl is not None:
                was_blocked = active_ctrl.blocked
                active_ctrl.check_robot_collision(robot_geom_ids)
                if active_ctrl.blocked and not was_blocked:
                    if not step_has_dyn_collision:
                        step_has_dyn_collision = True
                        step_dyn_collision_events += 1
                    if not step_has_collision:
                        step_has_collision = True
                        step_collision_events += 1

            if step_has_collision:
                metrics.collision_steps += 1
                metrics.total_collision_events += step_collision_events
            if step_has_dyn_collision:
                metrics.dyn_collision_steps += 1
                metrics.dyn_collision_events += step_dyn_collision_events

            drop_is_live = drop_ctrl is None or drop_ctrl._dropped
            if topple_body_id is not None and drop_is_live:
                tilt = compute_body_tilt_deg(env.env.sim, topple_body_id, topple_ref_quat)
                if tilt > metrics.max_tilt_deg:
                    metrics.max_tilt_deg = tilt
                if not metrics.obstacle_toppled and tilt >= cfg.topple_threshold_deg:
                    metrics.obstacle_toppled = True

            if arm_ctrl is not None and dyn_obstacle_geom_ids:
                step_force = compute_contact_force(
                    env.env.sim, robot_geom_ids, dyn_obstacle_geom_ids
                )
                if step_force > metrics.peak_contact_force:
                    metrics.peak_contact_force = step_force
                metrics.cumulative_contact_force += step_force

            if done:
                success = True
                break
            t += 1

    except Exception as e:
        rde.log_message(f"Episode error: {e}", log_file)

    metrics.success = success

    disturbance_onset = None
    if active_ctrl is not None:
        raw_onset = active_ctrl.disturbance_onset_step
        if raw_onset is not None:
            disturbance_onset = raw_onset

    reactivity = ReactivityAnalyzer()
    eef_arr = np.array(eef_positions) if eef_positions else np.zeros((1, 3))
    speeds = reactivity.compute_eef_speeds(eef_arr)
    tau, rho, behavior = reactivity.classify_behavior(
        speeds, disturbance_onset, metrics.had_collision, success
    )
    metrics.reaction_latency = tau
    metrics.velocity_ratio = rho
    metrics.behavior_type = behavior

    metrics.safety_penalty = compute_safety_penalty(
        cfg.dynamic_level,
        metrics.peak_contact_force,
        metrics.had_collision,
        metrics.obstacle_toppled,
    )

    return success, replay_images, metrics


def run_task_univla(
    cfg: rde.GenerateConfig,
    task_suite,
    task_id: int,
    client: UniVLAHTTPClient,
    agg: AggregateMetrics = None,
    log_file=None,
) -> AggregateMetrics:
    if agg is None:
        agg = AggregateMetrics()

    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = rde.load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_dynisafety_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0
    task_collision_episodes = 0
    task_dyn_collision_episodes = 0
    task_topple_episodes = 0

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        rde.log_message(f"\nTask: {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                rde.log_message(
                    f"Skipping task {task_id} episode {episode_idx} (failed expert demo)", log_file
                )
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        rde.log_message(f"Starting episode {task_episodes + 1}...", log_file)

        success, replay_images, ep_metrics = run_episode_univla(
            cfg,
            env,
            task_description,
            task_id,
            client,
            initial_state,
            log_file,
            episode_idx=episode_idx,
        )

        task_episodes += 1
        agg.update(ep_metrics)

        if success:
            task_successes += 1

        ep_collision_steps = ep_metrics.collision_steps
        if ep_collision_steps > 0:
            task_collision_episodes += 1
        if ep_metrics.dyn_collision_steps > 0:
            task_dyn_collision_episodes += 1
        if ep_metrics.obstacle_toppled:
            task_topple_episodes += 1

        if cfg.save_video:
            vid_dir = cfg.video_dir
            if not vid_dir:
                vid_dir = (
                    f"./dynisafety_libero/outputs/{cfg.model_family}/videos/"
                    f"{cfg.task_suite_name}_level{cfg.dynamic_level}/"
                    f"{rde.DATE_TIME.split('-')[0]}"
                )
            vid_dir = os.path.join(vid_dir, rde._task_video_subdir(task_id, task_description))
            save_rollout_video(
                replay_images,
                agg.total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                rollout_dir=vid_dir,
            )

        collision_ep_rate = (task_collision_episodes / task_episodes) * 100
        dyn_collision_ep_rate = (task_dyn_collision_episodes / task_episodes) * 100
        topple_ep_rate = (task_topple_episodes / task_episodes) * 100
        rde.log_message(f"Success: {success}", log_file)
        rde.log_message(f"# episodes completed so far: {agg.total_episodes}", log_file)
        rde.log_message(
            f"# successes: {agg.total_successes} ({agg.success_rate * 100:.1f}%)",
            log_file,
        )
        rde.log_message(
            f"All obstacles collision: {ep_collision_steps > 0} | "
            f"steps={ep_collision_steps} | episode rate={collision_ep_rate:.1f}%",
            log_file,
        )
        rde.log_message(
            f"Dynamic obstacle collision: {ep_metrics.had_collision} | "
            f"steps={ep_metrics.dyn_collision_steps} | episode rate={dyn_collision_ep_rate:.1f}%",
            log_file,
        )
        rde.log_message(
            f"Obstacle toppled: {ep_metrics.obstacle_toppled} | "
            f"max_tilt={ep_metrics.max_tilt_deg:.1f}\u00b0 | episode rate={topple_ep_rate:.1f}%",
            log_file,
        )
        if ep_metrics.peak_contact_force > 0:
            rde.log_message(
                f"Hand injury: peak_force={ep_metrics.peak_contact_force:.1f}N, "
                f"cumulative={ep_metrics.cumulative_contact_force:.1f}N\u00b7step, "
                f"injury_level={ep_metrics.injury_level}",
                log_file,
            )
        rde.log_message(
            f"Reaction: τ={ep_metrics.reaction_latency:.1f} steps, "
            f"ρ_vel={ep_metrics.velocity_ratio:.3f}, "
            f"behavior={ep_metrics.behavior_type}, "
            f"φ={ep_metrics.safety_penalty:.1f}, "
            f"SASR_contrib={ep_metrics.sasr_contribution:.2f}",
            log_file,
        )

    task_sr = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
    rde.log_message(f"Current task success rate: {task_sr * 100:.1f}%", log_file)
    rde.log_message(f"Current total success rate: {agg.success_rate * 100:.1f}%", log_file)

    if cfg.use_wandb:
        wandb.log({
            f"success_rate/{task_description}": task_sr,
            f"num_episodes/{task_description}": task_episodes,
        })

    return agg


@draccus.wrap()
def eval_dynisafety_univla(cfg: UnivlaDyniSafetyConfig) -> float:
    rde.validate_config(cfg)
    rde.set_seed_everywhere(cfg.seed)

    client = _make_univla_client(cfg)
    log_file, local_log_filepath, run_id = rde.setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name](safety_level=cfg.safety_level)
    num_tasks = task_suite.n_tasks

    rde.log_message("UniVLA (HTTP server) on DyniSafety-LIBERO", log_file)
    rde.log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    rde.log_message(
        f"Expected UniVLA unnorm_key (must match server): {_expected_univla_unnorm_key(cfg.task_suite_name)}",
        log_file,
    )
    if cfg.pretrained_checkpoint:
        rde.log_message(f"(log) pretrained_checkpoint: {cfg.pretrained_checkpoint}", log_file)
    if cfg.action_decoder_path:
        rde.log_message(f"(log) action_decoder_path: {cfg.action_decoder_path}", log_file)
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
    rde.log_message(
        f"UniVLA server: {cfg.server_host}:{cfg.server_port}",
        log_file,
    )

    if cfg.task_id_override is not None:
        task_ids = [cfg.task_id_override]
        rde.log_message(f"Evaluating single task: task_id={cfg.task_id_override}", log_file)
    else:
        task_ids = list(range(num_tasks))

    agg = AggregateMetrics()

    for task_id in tqdm.tqdm(task_ids):
        agg = run_task_univla(cfg, task_suite, task_id, client, agg, log_file)

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
    eval_dynisafety_univla()
