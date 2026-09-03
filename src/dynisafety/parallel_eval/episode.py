"""Segmented RoboTwin rollout: begin / collect_obs / apply_chunk / step_once / finish.

The HTTP ``RoboTwinBackend.rollout`` body is the P0 correctness baseline and
must stay untouched. This module splits that same body at the points Infer
has to yield (empty ``ActionChunkQueue``), and carries the in-between state
on ``EpisodeSlot`` so a later Env worker can drive it over a Channel.

Seed search is intentionally absent: ``_seed_cursor`` is sequential (episode
k starts after episode k-1's accepted seed). ``begin`` takes the
**search-start** seed: that is the expert-cache key. ``expert_check_setup``
then returns the warmed accepted seed on a cache hit (plan 0.7).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.backends.robotwin.backend import (
    _apply_action_chunk,
    _capture_fps,
    _copy_camera_rgb,
    _copy_endpose_7d,
    _copy_endpose_xyz,
    _copy_joint_vector,
    _force_equiv_dt_s,
    _require_level_supported,
    _require_robotwin_key,
    _robotwin_extra,
)
from dynisafety.backends.robotwin.config_bridge import build_disturbance_config
from dynisafety.backends.robotwin.contact import l3_episode_extra_from_info
from dynisafety.backends.robotwin.env import (
    RoboTwinEnvHandle,
    apply_ee_action,
    apply_qpos_action,
    build_observation,
    expert_check_setup,
    extract_capture_cameras,
    extract_head_rgb,
    save_episode_video,
    select_instruction,
)
from dynisafety.backends.robotwin.expert_cache import resolve_expert_check_cache_dir
from dynisafety.backends.robotwin.primary_arm import ensure_open_laptop_arm_tag
from dynisafety.backends.robotwin.hooks import (
    disable_dynamic_disturbance,
    enable_dynamic_disturbance,
    snapshot_info,
)
from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.episode import EpisodeRecord
from dynisafety.core.metrics import EpisodeMetricsBuilder
from dynisafety.core.rollout import (
    ActionChunkQueue,
    BoundRun,
    build_episode_record,
    episode_video_dir,
    resolve_save_video,
)
from dynisafety.capture import CaptureSession
from dynisafety.diagnostics import DiagnosticsSession
from dynisafety.parallel_eval.slot_state import SlotState
from dynisafety.policies.action_space import effective_action_space
from dynisafety.policies.base import ActionChunk, Observation


class EpisodeSlot:
    """Mutable bag between rollout segments.

    The old ``rollout`` is one stack frame. Splitting it without this object
    would hide the two contact clocks and the close-once flag in closures,
    which is how those fields drift.
    """

    def __init__(
        self,
        *,
        handle: RoboTwinEnvHandle,
        cfg: RunConfig,
        bound: BoundRun,
        slot_state: SlotState,
        accepted_seed: int,
        candidate_seed_start: int,
        expert_cache_hit: bool,
        episode_index: int,
        instruction: str,
        level: int,
        max_steps: int,
        observation_needs_state: bool,
        warmup_steps: int | None,
        post_warmup_space: str | None,
        extra_robotwin: Mapping[str, Any],
        builder: EpisodeMetricsBuilder,
        diag_session: DiagnosticsSession,
        capture_session: CaptureSession,
        action_queue: ActionChunkQueue,
        force_dt: float,
        t0: float,
        save_video: bool,
    ) -> None:
        self.handle = handle
        self.cfg = cfg
        self.bound = bound
        self.slot_state = slot_state
        self.accepted_seed = int(accepted_seed)
        self.candidate_seed_start = int(candidate_seed_start)
        self.expert_cache_hit = bool(expert_cache_hit)
        self.episode_index = int(episode_index)
        self.instruction = instruction
        self.level = int(level)
        self.max_steps = int(max_steps)
        self.observation_needs_state = bool(observation_needs_state)
        self.warmup_steps = warmup_steps
        self.post_warmup_space = post_warmup_space
        self.extra_robotwin = extra_robotwin
        self.builder = builder
        self.diag_session = diag_session
        self.capture_session = capture_session
        self.action_queue = action_queue
        self.force_dt = float(force_dt)
        self.t0 = float(t0)
        self.save_video = bool(save_video)
        self.frames: list[np.ndarray] = []
        self.raw_obs: Mapping[str, Any] | None = None
        self.step_eef_pos: tuple[float, ...] | None = None
        self.step_eef_pos_secondary: tuple[float, ...] | None = None
        self.step_tcp: tuple[float, ...] | None = None
        self.step_tcp_right: tuple[float, ...] | None = None
        self.step_joint: tuple[float, ...] | None = None
        self.step_primary: np.ndarray | None = None
        self.step_wrist: np.ndarray | None = None
        self.step_wrist_right: np.ndarray | None = None
        self.step_cameras: dict[str, dict[str, Any]] | None = None
        self.success = False
        self.success_step: int | None = None
        self.termination = "max_steps"
        self.task_steps_done = 0
        self.contact_without_force_any = False
        self.builder_extra_err: str | None = None
        self.env_closed = False


def begin(
    *,
    handle: RoboTwinEnvHandle,
    cfg: RunConfig,
    bound: BoundRun,
    slot_state: SlotState,
    episode_index: int,
    start_seed: int,
    resolved_data: Mapping[str, Any],
    workspace: Path,
    observation_needs_state: bool,
    warmup_steps: int | None,
    post_warmup_space: str | None,
) -> EpisodeSlot:
    """Bind one episode. ``start_seed`` is the expert-check search start.

    That value is the cache key (``expert_cache.cache_key(..., start_seed=)``).
    ``begin`` passes it to ``expert_check_setup``; a warm-cache hit returns
    the accepted seed used for ``setup_demo``. This does not walk
    ``_seed_cursor``.
    """
    if not isinstance(handle, RoboTwinEnvHandle):
        raise TypeError(f"expected RoboTwinEnvHandle, got {type(handle)!r}")
    if not isinstance(bound, BoundRun):
        raise TypeError(f"expected BoundRun, got {type(bound)!r}")
    if cfg.metrics is MISSING:
        raise ValueError("cfg.metrics is MISSING")
    if cfg.seed is MISSING:
        raise ValueError("cfg.seed is MISSING")
    if not isinstance(workspace, Path):
        raise TypeError(f"workspace must be a Path, got {type(workspace)!r}")

    level = int(cfg.level)
    _require_level_supported(level)
    extra_robotwin = _robotwin_extra(resolved_data)
    candidate_seed_start = int(start_seed)

    expert_enabled = bool(_require_robotwin_key(extra_robotwin, "expert_check"))
    expert_max_attempts = int(_require_robotwin_key(extra_robotwin, "expert_check_max_attempts"))
    expert_timeout_s = float(_require_robotwin_key(extra_robotwin, "expert_check_timeout_s"))
    expert_log_every = int(_require_robotwin_key(extra_robotwin, "expert_check_log_every"))
    expert_cache_enabled = bool(_require_robotwin_key(extra_robotwin, "expert_check_cache"))
    cache_dir = None
    if expert_enabled and expert_cache_enabled:
        cache_dir = resolve_expert_check_cache_dir(workspace)

    t0 = time.perf_counter()
    expert_outcome = expert_check_setup(
        handle,
        seed=candidate_seed_start,
        episode_index=episode_index,
        enabled=expert_enabled,
        max_attempts=expert_max_attempts,
        timeout_s=expert_timeout_s,
        log_every=expert_log_every,
        cache_dir=cache_dir,
    )
    used_seed = int(expert_outcome.accepted_seed)
    episode_info = expert_outcome.episode_info

    task_env = handle.task_env
    task_env.setup_demo(
        now_ep_num=episode_index,
        seed=used_seed,
        is_test=True,
        **handle.task_args,
    )
    ensure_open_laptop_arm_tag(task_env)

    dist_cfg = build_disturbance_config(cfg, extra_robotwin=extra_robotwin)
    enable_dynamic_disturbance(task_env, dist_cfg)

    episode_info_list = [episode_info] if episode_info else []
    if handle.episode_info and not episode_info_list:
        episode_info_list = [handle.episode_info]
    test_num = int(cfg.runtime.episodes) if cfg.runtime is not MISSING else 1
    instruction = select_instruction(
        task_name=handle.task.task_name,
        episode_info_list=episode_info_list,
        test_num=test_num,
        instruction_type=str(_require_robotwin_key(extra_robotwin, "instruction_type")),
        prompt_mode=str(_require_robotwin_key(extra_robotwin, "prompt_mode")),
        fallback=handle.task.instruction,
    )
    task_env.set_instruction(instruction=instruction)
    handle.instruction = instruction

    slot_state.reset()

    builder = EpisodeMetricsBuilder(cfg.metrics)
    diag_session = DiagnosticsSession.create(
        cfg,
        episode_index=episode_index,
        bound=bound,
        controller=getattr(task_env, "dynisafety_manager", None),
    )
    capture_session = CaptureSession.create(
        cfg,
        episode_index=episode_index,
        bound=bound,
        fps=_capture_fps(extra_robotwin),
    )
    save_video = resolve_save_video(cfg)
    force_dt = _force_equiv_dt_s(cfg)

    step_lim = int(getattr(task_env, "step_lim", handle.task.max_steps))
    cap = int(cfg.max_steps) if cfg.max_steps is not MISSING else step_lim
    max_steps = min(step_lim, cap)

    slot = EpisodeSlot(
        handle=handle,
        cfg=cfg,
        bound=bound,
        slot_state=slot_state,
        accepted_seed=used_seed,
        candidate_seed_start=candidate_seed_start,
        expert_cache_hit=bool(expert_outcome.cache_hit),
        episode_index=episode_index,
        instruction=instruction,
        level=level,
        max_steps=max_steps,
        observation_needs_state=observation_needs_state,
        warmup_steps=warmup_steps,
        post_warmup_space=post_warmup_space,
        extra_robotwin=extra_robotwin,
        builder=builder,
        diag_session=diag_session,
        capture_session=capture_session,
        action_queue=ActionChunkQueue(bound.chunk_size),
        force_dt=force_dt,
        t0=t0,
        save_video=save_video,
    )
    slot.raw_obs = task_env.get_obs()
    if save_video:
        slot.frames.append(extract_head_rgb(slot.raw_obs))
    return slot


def control_steps_remaining(slot: EpisodeSlot) -> bool:
    """Main-loop predicate: control steps (``take_action_cnt``), not sim steps."""
    task_env = slot.handle.task_env
    return int(getattr(task_env, "take_action_cnt", 0)) < slot.max_steps


def chunk_queue_empty(slot: EpisodeSlot) -> bool:
    """``ActionChunkQueue`` only refills when empty; extra predicted rows are dropped."""
    return len(slot.action_queue) == 0


def collect_obs(slot: EpisodeSlot) -> Observation:
    """Copy EEF *before* ``take_action``. ``get_obs`` here would re-render cameras."""
    if slot.raw_obs is None:
        raise RuntimeError("collect_obs requires begin() to have primed raw_obs")
    task_env = slot.handle.task_env
    task_step = int(getattr(task_env, "take_action_cnt", 0))
    slot.step_eef_pos = _copy_endpose_xyz(slot.raw_obs, "left_endpose")
    slot.step_eef_pos_secondary = _copy_endpose_xyz(slot.raw_obs, "right_endpose")
    if slot.capture_session.enabled:
        slot.step_tcp = _copy_endpose_7d(slot.raw_obs, "left_endpose")
        slot.step_tcp_right = _copy_endpose_7d(slot.raw_obs, "right_endpose")
        slot.step_joint = _copy_joint_vector(slot.raw_obs)
        try:
            slot.step_cameras = extract_capture_cameras(slot.raw_obs)
            if not slot.step_cameras:
                slot.step_cameras = None
        except Exception:
            slot.step_cameras = None
        if slot.capture_session.wants_images:
            slot.step_primary = _copy_camera_rgb(slot.raw_obs, "head_camera")
            slot.step_wrist = _copy_camera_rgb(slot.raw_obs, "left_camera")
            slot.step_wrist_right = _copy_camera_rgb(slot.raw_obs, "right_camera")
        else:
            slot.step_primary = None
            slot.step_wrist = None
            slot.step_wrist_right = None
    else:
        slot.step_tcp = None
        slot.step_tcp_right = None
        slot.step_joint = None
        slot.step_primary = None
        slot.step_wrist = None
        slot.step_wrist_right = None
        slot.step_cameras = None
    observation = build_observation(
        slot.handle,
        instruction=slot.instruction,
        step=task_step,
        needs_state=slot.observation_needs_state,
        raw_obs=slot.raw_obs,
    )
    if slot.save_video:
        slot.frames.append(observation.images["primary"].copy())
    return observation


def apply_chunk(slot: EpisodeSlot, actions_20: np.ndarray) -> int:
    """Decode official 20-D → EE 16-D, convert space, refill the shared queue."""
    if slot.raw_obs is None:
        raise RuntimeError("apply_chunk requires a primed raw_obs (call collect_obs first)")
    task_env = slot.handle.task_env
    task_step = int(getattr(task_env, "take_action_cnt", 0))
    ee16 = slot.slot_state.actions_20_to_ee16(actions_20)
    chunk = ActionChunk(actions=ee16, space="abs_ee_16d")
    converted = _apply_action_chunk(
        slot.handle,
        chunk,
        raw_obs=slot.raw_obs,
        configured_space=slot.bound.action_space,
        task_step=task_step,
        warmup_steps=slot.warmup_steps,
        post_warmup_space=slot.post_warmup_space,
    )
    return slot.action_queue.refill(converted)


def step_once(slot: EpisodeSlot) -> bool:
    """One control step = one full ``take_action`` (variable-length TOPP inside).

    Do not split this into ``scene.step``. Returns False when the episode
    should stop (success).
    """
    task_env = slot.handle.task_env
    task_step = int(getattr(task_env, "take_action_cnt", 0))
    action = slot.action_queue.popleft()
    target_space = effective_action_space(
        configured_space=slot.bound.action_space,
        step=task_step,
        warmup_steps=slot.warmup_steps,
        post_warmup_space=slot.post_warmup_space,
    )
    if target_space == "qpos_14d":
        apply_qpos_action(slot.handle, action)
    elif target_space in ("delta_ee_7d", "abs_ee_7d", "abs_ee_16d"):
        apply_ee_action(slot.handle, action)
    else:
        raise ValueError(f"unsupported target action space {target_space!r}")

    slot.raw_obs = task_env.get_obs()
    dyn_info = _dynisafety_info(task_env)

    step_collided = bool(dyn_info.get("step_collided", False))
    step_force = float(dyn_info.get("step_force_n", 0.0) or 0.0)
    step_impulse = float(dyn_info.get("step_impulse_ns", 0.0) or 0.0)
    slot.builder.update_contact(
        in_contact=step_collided,
        force_n=step_force,
        impulse_ns=step_impulse,
    )
    if slot.step_eef_pos is not None:
        slot.diag_session.record_step(
            t=task_step,
            eef_pos=slot.step_eef_pos,
            eef_quat=None,
            eef_pos_secondary=slot.step_eef_pos_secondary,
            action=action,
            in_contact=step_collided,
            force_n=step_force,
            impulse_ns=step_impulse,
        )
    cap_eef = (
        slot.step_tcp[:3]
        if slot.step_tcp is not None
        else slot.step_eef_pos
    )
    slot.capture_session.record_step(
        t=task_step,
        eef_pos=cap_eef or (0.0, 0.0, 0.0),
        eef_quat=(
            slot.step_tcp[3:7]
            if slot.step_tcp is not None and len(slot.step_tcp) >= 7
            else None
        ),
        action=action,
        in_contact=step_collided,
        force_n=step_force,
        reward=0.0,
        joint_position=slot.step_joint,
        primary=slot.step_primary,
        wrist=slot.step_wrist,
        wrist_right=slot.step_wrist_right,
        tcp_pose_right=slot.step_tcp_right,
        cameras=slot.step_cameras,
    )
    if bool(dyn_info.get("contact_without_force", False)):
        slot.contact_without_force_any = True

    if slot.level in (1, 2) and dyn_info.get("max_tilt_deg") is not None:
        slot.builder.update_topple(
            tilt_deg=float(dyn_info.get("max_tilt_deg", 0.0) or 0.0),
            toppled=bool(dyn_info.get("obstacle_toppled", False)),
        )

    slot.task_steps_done = task_step + 1
    if bool(getattr(task_env, "eval_success", False)):
        slot.success = True
        slot.success_step = task_step
        slot.termination = "success"
        return False
    return True


def mark_error(slot: EpisodeSlot, exc: BaseException) -> None:
    """Same as the old rollout ``except``: keep going to finish, do not drop the close."""
    slot.termination = "error"
    slot.builder_extra_err = str(exc)


def close_env_once(slot: EpisodeSlot) -> None:
    """Exactly one ``close_env``. The old HTTP path also closes again in ``loop.py``."""
    if slot.env_closed:
        return
    slot.env_closed = True
    task_env = slot.handle.task_env
    try:
        disable_dynamic_disturbance(task_env)
        clear_cache = (slot.episode_index + 1) % int(slot.handle.clear_cache_freq) == 0
        closer = getattr(task_env, "close_env", None)
        if callable(closer):
            closer(clear_cache=clear_cache)
    except Exception:
        pass


def finish(slot: EpisodeSlot) -> EpisodeRecord:
    """Overlay the two contact clocks, close once, build schema 2.0 record."""
    task_env = slot.handle.task_env
    dyn_final = _dynisafety_info(task_env)
    close_env_once(slot)

    metrics = slot.builder.finalize(
        success=slot.success,
        success_step=slot.success_step,
        steps=slot.task_steps_done,
        force_equiv_dt_s=slot.force_dt,
    )
    contact_control_steps = int(metrics["contact_steps"])
    contact_control_events = int(metrics["contact_events"])
    first_contact_control_step = metrics.get("first_contact_step")
    # first_contact_control_step / contact_control_steps are control-step
    # (take_action_cnt). After the dyn_final overlay below, metrics
    # ["first_contact_step"] / ["contact_steps"] become physical sim-step
    # values (dt=1/250). Passing those overlaid metrics into analyze_trace
    # makes t_react < first_contact_step almost always true and
    # systematically over-counts preemptive reactions. Do not switch these
    # arguments to metrics["first_contact_step"] / metrics["contact_steps"].
    slot.diag_session.finalize(
        success=slot.success,
        success_step=slot.success_step,
        first_contact_step=first_contact_control_step,
        contact_steps=contact_control_steps,
    )
    slot.capture_session.finalize(
        success=slot.success,
        success_step=slot.success_step,
        termination=slot.termination,
        first_contact_step=first_contact_control_step,
        contact_steps=contact_control_steps,
        toppled=metrics.get("toppled") if slot.level in (1, 2) else None,
        task_id=slot.handle.task.task_id,
        task_name=slot.handle.task.task_name,
        instruction=slot.instruction,
        checkpoint=str(slot.cfg.checkpoint),
        config_hash=slot.bound.config_hash,
    )
    metrics["peak_force_n"] = float(dyn_final.get("max_collision_force", 0.0) or 0.0)
    metrics["peak_impulse_ns"] = float(dyn_final.get("max_collision_impulse", 0.0) or 0.0)
    if "cumulative_impulse_ns" in dyn_final:
        metrics["cumulative_impulse_ns"] = float(
            dyn_final.get("cumulative_impulse_ns", 0.0) or 0.0
        )
    metrics["contact_steps"] = int(dyn_final.get("collision_steps", metrics["contact_steps"]))
    metrics["contact_events"] = int(
        dyn_final.get("collision_events", metrics["contact_events"])
    )
    metrics["first_contact_step"] = dyn_final.get("first_collision_step")
    if dyn_final.get("collision_partner_steps"):
        metrics["contact_partners"] = {
            str(k): int(v) for k, v in dict(dyn_final["collision_partner_steps"]).items()
        }
    if slot.level in (1, 2):
        metrics["max_tilt_deg"] = float(dyn_final.get("max_tilt_deg", 0.0) or 0.0)
        metrics["toppled"] = bool(dyn_final.get("obstacle_toppled", False))
    if int(dyn_final.get("contact_without_force_steps", 0) or 0) > 0:
        slot.contact_without_force_any = True

    video_path = _maybe_write_video(slot)
    extra = _episode_extra(
        slot,
        dyn_final=dyn_final,
        contact_control_steps=contact_control_steps,
        contact_control_events=contact_control_events,
        first_contact_control_step=first_contact_control_step,
    )
    wall = time.perf_counter() - slot.t0
    return build_episode_record(
        metrics=metrics,
        run_id=str(slot.cfg.run_id),
        config_hash=slot.bound.config_hash,
        backend=str(slot.cfg.backend),
        suite=str(slot.cfg.suite),
        model=str(slot.cfg.model),
        checkpoint=str(slot.cfg.checkpoint),
        level=slot.level,
        episode_index=slot.episode_index,
        task_id=slot.handle.task.task_id,
        task_name=slot.handle.task.task_name,
        instruction=slot.instruction,
        seed=int(slot.cfg.seed),
        max_steps=slot.max_steps,
        steps=int(metrics["steps"]),
        termination=slot.termination,
        wall_time_s=float(wall),
        video_path=video_path,
        extra=extra,
        diagnostics=slot.diag_session.episode_diagnostics(),
    )


def _dynisafety_info(task_env: Any) -> dict[str, Any]:
    dyn_info = snapshot_info(task_env, finalize=True)
    if hasattr(task_env, "info") and isinstance(task_env.info, dict):
        task_env.info["dynisafety"] = dyn_info
    return dyn_info


def _maybe_write_video(slot: EpisodeSlot) -> str | None:
    if not (slot.save_video and slot.frames):
        return None
    extra_robotwin = slot.extra_robotwin
    if "video_fps" not in extra_robotwin:
        raise ValueError("extra.robotwin.video_fps required when save_video=true")
    fps = int(extra_robotwin["video_fps"])
    results_dir = Path(str(slot.cfg.output.results_dir))
    vid_dir = episode_video_dir(
        results_dir=results_dir,
        backend=str(slot.cfg.backend),
        suite=str(slot.cfg.suite),
        level=slot.level,
        run_id=str(slot.cfg.run_id),
        task_id=slot.handle.task.task_id,
        task_name=slot.handle.task.task_name,
    )
    suffix = "success" if slot.success else "failure"
    path = vid_dir / f"{suffix}_{slot.episode_index}.mp4"
    save_episode_video(slot.frames, path=path, fps=fps)
    return str(path)


def _episode_extra(
    slot: EpisodeSlot,
    *,
    dyn_final: Mapping[str, Any],
    contact_control_steps: int,
    contact_control_events: int,
    first_contact_control_step: Any,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "accepted_seed": slot.accepted_seed,
        "candidate_seed_start": slot.candidate_seed_start,
        "expert_check_cache_hit": bool(slot.expert_cache_hit),
        "contact_without_force": bool(slot.contact_without_force_any),
        "contact_without_force_steps": int(
            dyn_final.get("contact_without_force_steps", 0) or 0
        ),
        "contact_control_steps": contact_control_steps,
        "contact_control_events": contact_control_events,
        "first_contact_control_step": (
            None
            if first_contact_control_step is None
            else int(first_contact_control_step)
        ),
    }
    disturbance_record = dyn_final.get("robotwin_disturbance")
    if isinstance(disturbance_record, dict):
        extra["robotwin_disturbance"] = dict(disturbance_record)
    if slot.level in (1, 2):
        extra["landing_toppled"] = bool(dyn_final.get("landing_toppled", False))
    if slot.level == 3:
        extra.update(l3_episode_extra_from_info(dyn_final))
    spawn_plan = dyn_final.get("spawn_plan")
    if isinstance(spawn_plan, dict) and spawn_plan:
        extra["spawn_plan"] = dict(spawn_plan)
    l3_plan = dyn_final.get("l3_plan")
    if isinstance(l3_plan, dict) and l3_plan:
        extra["l3_plan"] = dict(l3_plan)
    if slot.builder_extra_err is not None:
        extra["error"] = slot.builder_extra_err
    return extra


__all__ = [
    "EpisodeSlot",
    "apply_chunk",
    "begin",
    "chunk_queue_empty",
    "close_env_once",
    "collect_obs",
    "control_steps_remaining",
    "finish",
    "mark_error",
    "step_once",
]
