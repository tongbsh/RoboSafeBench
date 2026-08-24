"""RoboTwin ``Backend`` — single episode path for all models.

Replaces the three legacy loops (``main_inference.infer_once``,
``inference.infer_once``, ``xvla_official_inference.infer_once``) and removes
the ``internvla_a1`` special-case route that forwarded into tyro/`inference.py``.

``make_env`` / ``rollout`` talk to SAPIEN via ``env.py`` + ``hooks.py`` +
``disturbance/`` manager. One ``rollout`` serves every policy.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dynisafety.backends.robotwin.capabilities import ROBOTWIN_CAPABILITIES
from dynisafety.backends.robotwin.config_bridge import build_disturbance_config
from dynisafety.backends.robotwin.env import (
    RoboTwinEnvHandle,
    apply_ee_action,
    apply_qpos_action,
    build_observation,
    expert_check_setup,
    extract_head_rgb,
    make_robotwin_env,
    save_episode_video,
    select_instruction,
)
from dynisafety.backends.robotwin.hooks import (
    disable_dynamic_disturbance,
    enable_dynamic_disturbance,
    snapshot_info,
)
from dynisafety.backends.robotwin.seed import episode_search_start, next_candidate_seed
from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.contact import resolve_force_equiv_dt
from dynisafety.core.episode import EpisodeRecord
from dynisafety.core.metrics import EpisodeMetricsBuilder
from dynisafety.core.protocols import BackendCapabilities, EnvHandle, TaskSpec
from dynisafety.core.rollout import (
    ActionChunkQueue,
    build_episode_record,
    episode_video_dir,
    require_bound,
    resolve_save_video,
)
from dynisafety.diagnostics import DiagnosticsSession, copy_vec
from dynisafety.policies.action_space import convert_action, effective_action_space
from dynisafety.policies.base import ActionChunk, EpisodeContext, Policy

_VALID_SUITES = frozenset({"main20"})


def _find_workspace_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for parent in (here, *here.parents):
        if (parent / "configs" / "backends" / "robotwin").is_dir() and (
            parent / "pyproject.toml"
        ).is_file():
            return parent
    raise FileNotFoundError(
        "cannot locate workspace root (configs/backends/robotwin + pyproject.toml)"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def _require_level_supported(level: int) -> None:
    if level not in ROBOTWIN_CAPABILITIES.levels:
        raise ValueError(
            f"RoboTwin backend does not support level={level}; "
            f"supported levels={ROBOTWIN_CAPABILITIES.levels}"
        )


def _suite_yaml(suite: str, *, workspace: Path) -> dict[str, Any]:
    path = workspace / "configs" / "backends" / "robotwin" / "suites" / f"{suite}.yaml"
    return _load_yaml(path)


def _suite_task_rows(suite: str, *, workspace: Path) -> list[dict[str, Any]]:
    data = _suite_yaml(suite, workspace=workspace)
    rows = data.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"suite {suite!r} YAML must define a non-empty tasks: list "
            f"(configs/backends/robotwin/suites/{suite}.yaml)"
        )
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"suite {suite!r} tasks[{i}] must be a mapping")
        name = row.get("name")
        task_id = row.get("task_id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"suite {suite!r} tasks[{i}] missing name")
        if task_id is None:
            raise ValueError(f"suite {suite!r} task {name!r} missing task_id")
        out.append(dict(row))
    return out


def _suite_max_steps(suite: str, *, workspace: Path) -> int | None:
    data = _suite_yaml(suite, workspace=workspace)
    runtime = data.get("runtime")
    if isinstance(runtime, Mapping) and "max_steps" in runtime:
        return int(runtime["max_steps"])
    return None


def _task_max_steps(
    task_name: str,
    *,
    suite: str,
    workspace: Path,
) -> int:
    task_yaml = workspace / "configs" / "backends" / "robotwin" / "tasks" / f"{task_name}.yaml"
    if task_yaml.is_file():
        data = _load_yaml(task_yaml)
        if "max_steps" in data:
            return int(data["max_steps"])
        runtime = data.get("runtime")
        if isinstance(runtime, Mapping) and "max_steps" in runtime:
            return int(runtime["max_steps"])

    suite_steps = _suite_max_steps(suite, workspace=workspace)
    if suite_steps is not None:
        return suite_steps

    raise ValueError(
        f"max_steps missing for task {task_name!r}: set it in "
        f"configs/backends/robotwin/tasks/{task_name}.yaml or "
        f"suites/{suite}.yaml runtime.max_steps "
        "(vendored _eval_step_limit.yml fallback removed in S3g)"
    )


def _task_instruction(task_name: str, row: Mapping[str, Any]) -> str:
    for key in ("instruction", "prompt"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    extra = row.get("extra")
    if isinstance(extra, Mapping):
        rt = extra.get("robotwin")
        if isinstance(rt, Mapping):
            for key in ("instruction", "prompt"):
                val = rt.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return task_name.replace("_", " ")


def _force_equiv_dt_s(cfg: RunConfig) -> float:
    contact = cfg.contact
    if contact is MISSING:
        raise ValueError("cfg.contact is MISSING")
    mode = contact.force_equiv_dt
    window = contact.force_report_window_s
    if mode is MISSING or window is MISSING:
        raise ValueError("contact.force_equiv_dt and force_report_window_s required")
    return resolve_force_equiv_dt(
        mode,
        sim_timestep_s=None,
        control_freq_hz=None,
        force_report_window_s=float(window),
    )


def _robotwin_extra(resolved_data: Mapping[str, Any]) -> Mapping[str, Any]:
    extra = resolved_data.get("extra") or {}
    if not isinstance(extra, Mapping):
        raise TypeError("resolved_data['extra'] must be a mapping")
    robotwin = extra.get("robotwin")
    if not isinstance(robotwin, Mapping):
        raise ValueError("resolved_data extra.robotwin required")
    return robotwin


def _require_robotwin_key(extra: Mapping[str, Any], key: str) -> Any:
    if key not in extra:
        raise ValueError(f"extra.robotwin.{key}")
    val = extra[key]
    if val is MISSING or val is None:
        raise ValueError(f"extra.robotwin.{key}")
    return val


def _seed_stride(cfg: RunConfig, resolved_data: Mapping[str, Any]) -> int:
    runtime = cfg.runtime
    if runtime is not MISSING and runtime.seed_stride is not MISSING:
        return int(runtime.seed_stride)
    resolved_runtime = resolved_data.get("runtime")
    if isinstance(resolved_runtime, Mapping) and "seed_stride" in resolved_runtime:
        return int(resolved_runtime["seed_stride"])
    raise ValueError("seed_stride missing from cfg.runtime or resolved_data.runtime")


def _copy_endpose_xyz(raw_obs: Mapping[str, Any], key: str) -> tuple[float, ...] | None:
    """Copy xyz from an already-fetched RoboTwin obs. Never calls ``get_obs()``.

    A missing / short pose returns ``None`` instead of raising — diagnostics
    must not turn a rollout into ``termination="error"``.
    """
    try:
        endpose = raw_obs["endpose"]
        return copy_vec(endpose[key][:3])
    except Exception:
        return None


def _eef_context(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    endpose = raw_obs.get("endpose")
    if not isinstance(endpose, Mapping):
        raise ValueError("endpose missing from RoboTwin observation for ee conversion")
    left = np.asarray(endpose["left_endpose"], dtype=np.float64).reshape(-1)
    grip = float(endpose.get("left_gripper", 0.0))
    if left.size >= 6:
        eef = np.concatenate([left[:6], [grip]])
    else:
        raise ValueError("endpose.left_endpose must provide at least 6 values")
    return {"eef_state": eef}


def _apply_action_chunk(
    handle: RoboTwinEnvHandle,
    chunk: ActionChunk,
    *,
    raw_obs: Mapping[str, Any],
    configured_space: str,
    task_step: int,
    warmup_steps: int | None,
    post_warmup_space: str | None,
) -> np.ndarray:
    target = effective_action_space(
        configured_space=configured_space,
        step=task_step,
        warmup_steps=warmup_steps,
        post_warmup_space=post_warmup_space,
    )
    context: Mapping[str, Any] = {}
    if target in ("delta_ee_7d", "abs_ee_7d") or chunk.space in ("delta_ee_7d", "abs_ee_7d"):
        context = _eef_context(raw_obs)
    return convert_action(chunk, target_space=target, context=context)


class RoboTwinBackend:
    """SAPIEN / RoboTwin backend adapting impulse contact to ``EpisodeRecord``.

    All models share this class — there is **no** ``if model == "internvla_a1"``
    branch and no tyro / ``inference.py`` detour.
    """

    name: str = "robotwin"
    capabilities: BackendCapabilities = ROBOTWIN_CAPABILITIES

    def __init__(self) -> None:
        self._config_hash: str | None = None
        self._configured_action_space: str | None = None
        self._chunk_size: int | None = None
        self._warmup_steps: int | None = None
        self._post_warmup_space: str | None = None
        self._needs_state: bool = True
        self._observation_images: tuple[str, ...] = ("primary", "left", "right")
        self._workspace: Path | None = None
        self._resolved_data: dict[str, Any] = {}
        self._seed_cursor: int | None = None

    def bind_run(
        self,
        *,
        config_hash: str,
        resolved_data: Mapping[str, Any] | None = None,
        action_space: str,
        chunk_size: int,
        observation_images: tuple[str, ...] = ("primary", "left", "right"),
        observation_needs_state: bool = True,
        model_extra: Mapping[str, Any] | None = None,
        workspace: Path | None = None,
    ) -> None:
        """Stash ResolvedConfig fields that are not on ``RunConfig``."""
        if not config_hash:
            raise ValueError("config_hash must be non-empty")
        if not action_space:
            raise ValueError("action_space must be non-empty")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self._config_hash = str(config_hash)
        self._configured_action_space = str(action_space)
        self._chunk_size = int(chunk_size)
        self._needs_state = bool(observation_needs_state)
        self._observation_images = tuple(observation_images)
        extra = dict(model_extra or {})
        warm = extra.get("warmup_steps")
        self._warmup_steps = int(warm) if warm is not None else None
        post = extra.get("post_warmup_action_space")
        self._post_warmup_space = str(post) if post is not None else None
        self._workspace = Path(workspace) if workspace is not None else None
        self._resolved_data = dict(resolved_data or {})
        self._seed_cursor = None

    def list_tasks(self, suite: str) -> list[TaskSpec]:
        if suite not in _VALID_SUITES:
            raise ValueError(
                f"unknown RoboTwin suite {suite!r}; expected one of {sorted(_VALID_SUITES)}"
            )
        workspace = self._workspace or _find_workspace_root()
        rows = _suite_task_rows(suite, workspace=workspace)
        specs: list[TaskSpec] = []
        for row in rows:
            task_name = str(row["name"])
            task_id = int(row["task_id"])
            instruction = _task_instruction(task_name, row)
            task_yaml = (
                workspace / "configs" / "backends" / "robotwin" / "tasks" / f"{task_name}.yaml"
            )
            if task_yaml.is_file():
                tdata = _load_yaml(task_yaml)
                instruction = _task_instruction(task_name, {**row, **tdata})
            specs.append(
                TaskSpec(
                    task_id=task_id,
                    task_name=task_name,
                    suite=suite,
                    instruction=instruction,
                    max_steps=_task_max_steps(task_name, suite=suite, workspace=workspace),
                )
            )
        return specs

    def make_env(self, cfg: RunConfig, task: TaskSpec) -> EnvHandle:
        _require_level_supported(int(cfg.level) if cfg.level is not MISSING else -1)
        workspace = self._workspace or _find_workspace_root()
        if self._workspace is None:
            self._workspace = workspace
        extra_robotwin = _robotwin_extra(self._resolved_data)
        return make_robotwin_env(
            cfg,
            task,
            workspace=workspace,
            extra_robotwin=extra_robotwin,
        )

    def rollout(
        self,
        env: EnvHandle,
        policy: Policy,
        cfg: RunConfig,
        episode_index: int,
    ) -> EpisodeRecord:
        """Unified episode loop for all RoboTwin policies."""
        bound = require_bound(
            backend="robotwin",
            config_hash=self._config_hash,
            action_space=self._configured_action_space,
            chunk_size=self._chunk_size,
        )

        if not isinstance(env, RoboTwinEnvHandle):
            raise TypeError(f"expected RoboTwinEnvHandle, got {type(env)!r}")

        handle = env
        level = int(cfg.level)
        _require_level_supported(level)

        extra_robotwin = _robotwin_extra(self._resolved_data)
        metrics_cfg = cfg.metrics
        if metrics_cfg is MISSING:
            raise ValueError("cfg.metrics is MISSING")
        if cfg.seed is MISSING:
            raise ValueError("cfg.seed is MISSING")

        seed_stride = _seed_stride(cfg, self._resolved_data)
        if self._seed_cursor is None:
            self._seed_cursor = episode_search_start(
                run_seed=int(cfg.seed),
                seed_stride=seed_stride,
            )
        candidate_seed = int(self._seed_cursor)

        expert_enabled = bool(_require_robotwin_key(extra_robotwin, "expert_check"))
        expert_max_attempts = int(
            _require_robotwin_key(extra_robotwin, "expert_check_max_attempts")
        )
        expert_timeout_s = float(_require_robotwin_key(extra_robotwin, "expert_check_timeout_s"))
        expert_log_every = int(_require_robotwin_key(extra_robotwin, "expert_check_log_every"))

        t0 = time.perf_counter()
        accepted_seed, episode_info = expert_check_setup(
            handle,
            seed=candidate_seed,
            episode_index=episode_index,
            enabled=expert_enabled,
            max_attempts=expert_max_attempts,
            timeout_s=expert_timeout_s,
            log_every=expert_log_every,
        )

        task_env = handle.task_env
        task_env.setup_demo(
            now_ep_num=episode_index,
            seed=accepted_seed,
            is_test=True,
            **handle.task_args,
        )

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

        ctx = EpisodeContext(
            instruction=instruction,
            task_id=handle.task.task_id,
            task_name=handle.task.task_name,
            episode_index=episode_index,
            seed=accepted_seed,
            level=level,
            backend=str(cfg.backend),
            suite=str(cfg.suite),
            extra={},
        )
        policy.reset(ctx)

        builder = EpisodeMetricsBuilder(metrics_cfg)
        # Pass the DisturbanceManager (controller + dynamic_level). make_probe
        # unwraps it via unwrap_manager; there is no controller.step here.
        diag_session = DiagnosticsSession.create(
            cfg,
            episode_index=episode_index,
            bound=bound,
            controller=getattr(task_env, "dynisafety_manager", None),
        )
        save_video = resolve_save_video(cfg)
        frames: list[np.ndarray] = []
        force_dt = _force_equiv_dt_s(cfg)

        step_lim = int(getattr(task_env, "step_lim", handle.task.max_steps))
        cap = int(cfg.max_steps) if cfg.max_steps is not MISSING else step_lim
        max_steps = min(step_lim, cap)

        action_queue = ActionChunkQueue(bound.chunk_size)
        success = False
        success_step: int | None = None
        termination = "max_steps"
        task_steps_done = 0
        contact_without_force_any = False

        try:
            raw_obs = task_env.get_obs()
            if save_video:
                frames.append(extract_head_rgb(raw_obs))

            while int(getattr(task_env, "take_action_cnt", 0)) < max_steps:
                task_step = int(getattr(task_env, "take_action_cnt", 0))
                # Copy before take_action: next get_obs is t+1, and SAPIEN may
                # reuse the same ndarray buffers in place. Reuse raw_obs already
                # in scope — do not call get_obs() just for EEF (renders cameras).
                # Left arm is the analyzed subject; the right arm is stored so a
                # dual-arm reaction is not lost from the trace.
                step_eef_pos = _copy_endpose_xyz(raw_obs, "left_endpose")
                step_eef_pos_secondary = _copy_endpose_xyz(raw_obs, "right_endpose")
                observation = build_observation(
                    handle,
                    instruction=instruction,
                    step=task_step,
                    needs_state=self._needs_state,
                )
                if save_video:
                    frames.append(observation.images["primary"].copy())

                if len(action_queue) == 0:
                    chunk = policy.act(observation)
                    converted = _apply_action_chunk(
                        handle,
                        chunk,
                        raw_obs=raw_obs,
                        configured_space=self._configured_action_space,
                        task_step=task_step,
                        warmup_steps=self._warmup_steps,
                        post_warmup_space=self._post_warmup_space,
                    )
                    action_queue.refill(converted)

                action = action_queue.popleft()
                target_space = effective_action_space(
                    configured_space=self._configured_action_space,
                    step=task_step,
                    warmup_steps=self._warmup_steps,
                    post_warmup_space=self._post_warmup_space,
                )
                if target_space == "qpos_14d":
                    apply_qpos_action(handle, action)
                elif target_space in ("delta_ee_7d", "abs_ee_7d", "abs_ee_16d"):
                    apply_ee_action(handle, action)
                else:
                    raise ValueError(f"unsupported target action space {target_space!r}")

                raw_obs = task_env.get_obs()
                dyn_info = snapshot_info(task_env)
                if hasattr(task_env, "info") and isinstance(task_env.info, dict):
                    dyn_info = dict(task_env.info.get("dynisafety") or dyn_info)

                step_collided = bool(dyn_info.get("step_collided", False))
                # Per-take_action peaks from update_info (manager action accumulators).
                step_force = float(dyn_info.get("step_force_n", 0.0) or 0.0)
                step_impulse = float(dyn_info.get("step_impulse_ns", 0.0) or 0.0)
                builder.update_contact(
                    in_contact=step_collided,
                    force_n=step_force,
                    impulse_ns=step_impulse,
                )
                if step_eef_pos is not None:
                    diag_session.record_step(
                        t=task_step,
                        eef_pos=step_eef_pos,
                        eef_quat=None,
                        eef_pos_secondary=step_eef_pos_secondary,
                        action=action,
                        in_contact=step_collided,
                        force_n=step_force,
                        impulse_ns=step_impulse,
                    )
                if bool(dyn_info.get("contact_without_force", False)):
                    contact_without_force_any = True

                if level in (1, 2) and dyn_info.get("max_tilt_deg") is not None:
                    builder.update_topple(
                        tilt_deg=float(dyn_info.get("max_tilt_deg", 0.0) or 0.0),
                        toppled=bool(dyn_info.get("obstacle_toppled", False)),
                    )

                task_steps_done = task_step + 1
                if bool(getattr(task_env, "eval_success", False)):
                    success = True
                    success_step = task_step
                    termination = "success"
                    break
        except Exception as exc:  # noqa: BLE001
            termination = "error"
            builder_extra_err = str(exc)
        else:
            builder_extra_err = None

        dyn_final = snapshot_info(task_env)
        if hasattr(task_env, "info") and isinstance(task_env.info, dict):
            dyn_final = dict(task_env.info.get("dynisafety") or dyn_final)

        try:
            disable_dynamic_disturbance(task_env)
            clear_cache = (episode_index + 1) % int(handle.clear_cache_freq) == 0
            close_env = getattr(task_env, "close_env", None)
            if callable(close_env):
                close_env(clear_cache=clear_cache)
        except Exception:
            pass

        self._seed_cursor = next_candidate_seed(accepted_seed)

        metrics = builder.finalize(
            success=success,
            success_step=success_step,
            steps=task_steps_done,
            force_equiv_dt_s=force_dt,
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
        diag_session.finalize(
            success=success,
            success_step=success_step,
            first_contact_step=first_contact_control_step,
            contact_steps=contact_control_steps,
        )
        metrics["peak_force_n"] = float(dyn_final.get("max_collision_force", 0.0) or 0.0)
        metrics["peak_impulse_ns"] = float(dyn_final.get("max_collision_impulse", 0.0) or 0.0)
        # Prefer manager cumulative (sums every sim substep); builder only sees
        # per-take_action peaks and would under-count.
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
        if level in (1, 2):
            metrics["max_tilt_deg"] = float(dyn_final.get("max_tilt_deg", 0.0) or 0.0)
            metrics["toppled"] = bool(dyn_final.get("obstacle_toppled", False))
        if int(dyn_final.get("contact_without_force_steps", 0) or 0) > 0:
            contact_without_force_any = True

        video_path: str | None = None
        if save_video and frames:
            if "video_fps" not in extra_robotwin:
                raise ValueError("extra.robotwin.video_fps required when save_video=true")
            fps = int(extra_robotwin["video_fps"])
            results_dir = Path(str(cfg.output.results_dir))
            vid_dir = episode_video_dir(
                results_dir=results_dir,
                suite=str(cfg.suite),
                level=level,
                task_id=handle.task.task_id,
                task_name=handle.task.task_name,
            )
            suffix = "success" if success else "failure"
            path = vid_dir / f"{suffix}_{episode_index}.mp4"
            save_episode_video(frames, path=path, fps=fps)
            video_path = str(path)

        wall = time.perf_counter() - t0
        extra: dict[str, Any] = {
            "accepted_seed": accepted_seed,
            "candidate_seed_start": candidate_seed,
            "contact_without_force": bool(contact_without_force_any),
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
        if level in (1, 2):
            extra["landing_toppled"] = bool(dyn_final.get("landing_toppled", False))
        if level == 3:
            extra["l3_post_first_overlap_sim_steps"] = int(
                dyn_final.get("l3_post_first_overlap_sim_steps", 0) or 0
            )
            raw_ignore = dyn_final.get("l3_ignore_collision_link_substrings")
            extra["l3_ignore_collision_link_substrings"] = (
                [str(item) for item in raw_ignore] if raw_ignore else []
            )
        spawn_plan = dyn_final.get("spawn_plan")
        if isinstance(spawn_plan, dict) and spawn_plan:
            extra["spawn_plan"] = dict(spawn_plan)
        l3_plan = dyn_final.get("l3_plan")
        if isinstance(l3_plan, dict) and l3_plan:
            extra["l3_plan"] = dict(l3_plan)
        if builder_extra_err is not None:
            extra["error"] = builder_extra_err

        return build_episode_record(
            metrics=metrics,
            run_id=str(cfg.run_id),
            config_hash=bound.config_hash,
            backend=str(cfg.backend),
            suite=str(cfg.suite),
            model=str(cfg.model),
            checkpoint=str(cfg.checkpoint),
            level=level,
            episode_index=episode_index,
            task_id=handle.task.task_id,
            task_name=handle.task.task_name,
            instruction=instruction,
            seed=accepted_seed,
            max_steps=max_steps,
            steps=int(metrics["steps"]),
            termination=termination,
            wall_time_s=float(wall),
            video_path=video_path,
            extra=extra,
            diagnostics=diag_session.episode_diagnostics(),
        )
