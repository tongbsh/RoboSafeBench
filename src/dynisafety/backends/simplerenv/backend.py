"""SimplerEnv ``Backend`` — list_tasks / make_env / rollout."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dynisafety.backends.simplerenv.capabilities import SIMPLERENV_CAPABILITIES
from dynisafety.backends.simplerenv.contact import force_equiv_dt_s, scene_timestep
from dynisafety.backends.simplerenv.env import (
    SimplerEnvHandle,
    build_reset_options,
    get_capture_cameras,
    get_eef_pos,
    get_eef_pos_8d,
    get_eef_state_7d,
    get_primary_image,
    make_simplerenv_env,
    save_episode_video,
)
from dynisafety.config.schema import MISSING, RunConfig
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
from dynisafety.capture import CaptureSession
from dynisafety.diagnostics import DiagnosticsSession, copy_vec
from dynisafety.policies.action_space import convert_action, effective_action_space
from dynisafety.policies.base import ActionChunk, EpisodeContext, Observation, Policy

_VALID_SUITES = frozenset({"bridge"})

_BRIDGE_TASK_ORDER = (
    "bridge_carrot",
    "bridge_stack",
    "bridge_spoon",
    "bridge_eggplant",
)


def _find_workspace_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for parent in (here, *here.parents):
        if (parent / "configs" / "backends" / "simplerenv").is_dir() and (
            parent / "pyproject.toml"
        ).is_file():
            return parent
    raise FileNotFoundError(
        "cannot locate workspace root (configs/backends/simplerenv + pyproject.toml)"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def _suite_max_steps(suite: str, *, workspace: Path | None = None) -> int:
    root = workspace or _find_workspace_root()
    path = root / "configs" / "backends" / "simplerenv" / "suites" / f"{suite}.yaml"
    data = _load_yaml(path)
    runtime = data.get("runtime")
    if not isinstance(runtime, dict) or "max_steps" not in runtime:
        raise ValueError(
            f"suite {suite!r} YAML must define runtime.max_steps "
            f"(configs/backends/simplerenv/suites/{suite}.yaml)"
        )
    return int(runtime["max_steps"])


def _require_level_supported(level: int) -> None:
    if level not in SIMPLERENV_CAPABILITIES.levels:
        raise ValueError(
            f"SimplerEnv backend does not support level={level}; "
            f"supported levels={SIMPLERENV_CAPABILITIES.levels}"
        )


def _task_instruction(task_name: str, extra: dict[str, Any]) -> str:
    for key in ("instruction", "env_name"):
        val = extra.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return task_name


def _resolved_extra(resolved_data: Mapping[str, Any] | None) -> dict[str, Any]:
    if not resolved_data:
        return {}
    extra = resolved_data.get("extra")
    if extra is None:
        return {}
    if not isinstance(extra, Mapping):
        raise TypeError("resolved_data['extra'] must be a mapping")
    return dict(extra)


def _disturbance_controller(env: Any) -> Any | None:
    """Return ``DyniSafetyWrapper._controller`` without touching the sim.

    L0 never constructs the wrapper (``handle.env`` is the bare ManiSkill env).
    L1–L3 set ``handle.env`` to the wrapper after reset. A later gym wrapper
    may sit outside; walk ``.env`` then ``.unwrapped``. Missing → ``None`` so
    ``make_probe`` stays off and the episode still records EEF / action / contact.
    Never raises: a diagnostics lookup must not become ``termination="error"``.
    """
    try:
        current = env
        for _ in range(3):
            if current is None:
                return None
            ctrl = getattr(current, "_controller", None)
            if ctrl is not None:
                return ctrl
            inner = getattr(current, "env", None)
            if inner is None or inner is current:
                inner = getattr(current, "unwrapped", None)
            if inner is None or inner is current:
                return None
            current = inner
    except Exception:
        return None
    return None


def build_observation(
    handle: SimplerEnvHandle,
    obs: Mapping[str, Any],
    *,
    instruction: str,
    step: int,
    needs_state: bool,
    include_images: bool = True,
) -> Observation:
    images: dict[str, np.ndarray] = {}
    if include_images:
        images["primary"] = get_primary_image(
            handle.env, obs, camera_name=handle.camera_name
        )
    state = None
    extra: dict[str, Any] = {}
    # Full WidowX eef_pos (8D) for policies that abs→delta like archive X-VLA;
    # other adapters ignore Observation.extra.
    eef8 = get_eef_pos_8d(obs if isinstance(obs, Mapping) else {})
    if eef8 is not None:
        extra["eef_pos"] = eef8
    # Task id/name for adapters with per-task knobs (e.g. X-VLA gripper thresh).
    extra["task_id"] = int(handle.task.task_id)
    extra["task_name"] = str(handle.task.task_name)
    if needs_state:
        state = get_eef_state_7d(obs)
        if state is None:
            eef = get_eef_pos(handle.env)
            state = np.concatenate([eef, np.zeros(4, dtype=np.float64)])
    return Observation(
        images=images,
        state=state,
        instruction=instruction,
        step=int(step),
        extra=extra,
    )


def apply_action_chunk(
    chunk: ActionChunk,
    *,
    obs: Mapping[str, Any],
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
    eef = get_eef_state_7d(obs)
    context: dict[str, Any] = {}
    if eef is not None:
        context["eef_state"] = eef
    converted = convert_action(chunk, target_space=target, context=context)
    return np.asarray(converted, dtype=np.float64)


class SimplerEnvBackend:
    """ManiSkill2 / SimplerEnv backend adapting SAPIEN contact to ``EpisodeRecord``."""

    name: str = "simplerenv"
    capabilities: BackendCapabilities = SIMPLERENV_CAPABILITIES

    def __init__(self) -> None:
        self._config_hash: str | None = None
        self._configured_action_space: str | None = None
        self._chunk_size: int | None = None
        self._warmup_steps: int | None = None
        self._post_warmup_space: str | None = None
        self._needs_state: bool = True
        self._workspace: Path | None = None
        self._resolved_data: dict[str, Any] = {}

    def bind_run(
        self,
        *,
        config_hash: str,
        resolved_data: dict[str, Any] | Mapping[str, Any] | None = None,
        action_space: str,
        chunk_size: int,
        observation_images: tuple[str, ...] = ("primary",),
        observation_needs_state: bool = True,
        model_extra: dict[str, Any] | Mapping[str, Any] | None = None,
        workspace: Path | None = None,
    ) -> None:
        """Stash ResolvedConfig fields that are not on ``RunConfig``."""
        del observation_images
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
        extra = dict(model_extra or {})
        warm = extra.get("warmup_steps")
        self._warmup_steps = int(warm) if warm is not None else None
        post = extra.get("post_warmup_action_space")
        self._post_warmup_space = str(post) if post is not None else None
        self._workspace = Path(workspace) if workspace is not None else None
        self._resolved_data = dict(resolved_data or {})

    def list_tasks(self, suite: str) -> list[TaskSpec]:
        if suite not in _VALID_SUITES:
            raise ValueError(
                f"unknown SimplerEnv suite {suite!r}; expected one of "
                f"{sorted(_VALID_SUITES)} (D1=A: bridge only)"
            )
        workspace = self._workspace or _find_workspace_root()
        max_steps = _suite_max_steps(suite, workspace=workspace)
        tasks_dir = workspace / "configs" / "backends" / "simplerenv" / "tasks"
        if not tasks_dir.is_dir():
            raise FileNotFoundError(f"tasks directory missing: {tasks_dir}")

        task_names = list(_BRIDGE_TASK_ORDER)
        specs: list[TaskSpec] = []
        for task_id, task_name in enumerate(task_names):
            path = tasks_dir / f"{task_name}.yaml"
            data = _load_yaml(path)
            extra = data.get("extra")
            if not isinstance(extra, dict):
                raise ValueError(f"task {task_name!r} YAML must define mapping extra:")
            if "env_name" not in extra:
                raise ValueError(
                    f"task {task_name!r} YAML must set extra.env_name (ManiSkill2 env id)"
                )
            specs.append(
                TaskSpec(
                    task_id=task_id,
                    task_name=task_name,
                    suite=suite,
                    instruction=_task_instruction(task_name, extra),
                    max_steps=max_steps,
                )
            )
        if not specs:
            raise ValueError(f"no tasks found for suite {suite!r} under {tasks_dir}")
        return specs

    def make_env(self, cfg: RunConfig, task: TaskSpec) -> EnvHandle:
        if cfg.backend is not MISSING and cfg.backend != self.name:
            raise ValueError(
                f"cfg.backend={cfg.backend!r} does not match SimplerEnvBackend name={self.name!r}"
            )
        _require_level_supported(int(cfg.level) if cfg.level is not MISSING else -1)
        if task.suite not in _VALID_SUITES:
            raise ValueError(f"unknown SimplerEnv suite on TaskSpec: {task.suite!r}")
        workspace = self._workspace or _find_workspace_root()
        if self._workspace is None:
            self._workspace = workspace
        return make_simplerenv_env(
            cfg,
            task,
            workspace=workspace,
            resolved_extra=_resolved_extra(self._resolved_data),
        )

    def rollout(
        self,
        env: EnvHandle,
        policy: Policy,
        cfg: RunConfig,
        episode_index: int,
    ) -> EpisodeRecord:
        if not isinstance(env, SimplerEnvHandle):
            raise TypeError(f"expected SimplerEnvHandle, got {type(env)!r}")
        bound = require_bound(
            backend="simplerenv",
            config_hash=self._config_hash,
            action_space=self._configured_action_space,
            chunk_size=self._chunk_size,
        )

        handle = env
        level = int(cfg.level)
        _require_level_supported(level)
        max_steps = int(cfg.max_steps)
        metrics_cfg = cfg.metrics
        if metrics_cfg is MISSING:
            raise ValueError("cfg.metrics is MISSING")
        if cfg.seed is MISSING:
            raise ValueError("cfg.seed is MISSING")

        runtime = cfg.runtime
        if runtime is MISSING:
            raise ValueError("cfg.runtime is MISSING")
        control_freq = runtime.control_freq
        if control_freq is MISSING:
            raise ValueError("runtime.control_freq required (YAML)")

        t0 = time.perf_counter()
        reset_opts = build_reset_options(cfg, handle.task_extra, episode_index=episode_index)

        # Rebind wrapper episode index before reset (make_env may be reused).
        wrapped = handle.env
        if hasattr(wrapped, "_episode_index"):
            wrapped._episode_index = int(episode_index)  # noqa: SLF001

        obs, _info = handle.env.reset(options=reset_opts)
        try:
            instruction = handle.env.get_language_instruction()
        except Exception:
            instruction = handle.instruction
        if not instruction:
            instruction = handle.instruction

        ctx = EpisodeContext(
            instruction=str(instruction),
            task_id=handle.task.task_id,
            task_name=handle.task.task_name,
            episode_index=episode_index,
            seed=int(cfg.seed),
            level=level,
            backend=str(cfg.backend),
            suite=str(cfg.suite),
            extra={},
        )
        policy.reset(ctx)

        # Force dt: L0 uses scene from base env; L1–3 from wrapper after reset.
        scene = getattr(getattr(handle.env, "unwrapped", handle.env), "_scene", None)
        if scene is None and hasattr(handle.env, "scene"):
            scene = handle.env.scene
        if scene is None:
            raise RuntimeError("cannot access SAPIEN scene for force_equiv_dt")
        force_dt = force_equiv_dt_s(
            cfg,
            scene_timestep_s=scene_timestep(scene, control_freq=int(control_freq)),
            control_freq=int(control_freq),
        )

        builder = EpisodeMetricsBuilder(metrics_cfg)
        diag_session = DiagnosticsSession.create(
            cfg,
            episode_index=episode_index,
            bound=bound,
            controller=_disturbance_controller(handle.env),
        )
        capture_session = CaptureSession.create(
            cfg,
            episode_index=episode_index,
            bound=bound,
            fps=int(control_freq),
        )
        action_queue = ActionChunkQueue(bound.chunk_size)
        frames: list[np.ndarray] = []
        save_video = resolve_save_video(cfg)

        success = False
        success_step: int | None = None
        termination = "max_steps"
        task_steps_done = 0
        force_anomaly_any = False
        contact_without_force_steps = 0
        pre_physics_prevents = 0
        builder_extra_err: str | None = None
        is_final_subtask = True
        try:
            is_final_subtask = bool(handle.env.is_final_subtask())
        except Exception:
            is_final_subtask = True

        try:
            for task_step in range(max_steps):
                # Copy before env.step: next obs is t+1, and SAPIEN may
                # reuse the same ndarray buffers in place. This is the
                # pre-action EEF; do not call get_eef_pos again after step.
                step_eef_pos = copy_vec(get_eef_pos(handle.env))
                eef_state = get_eef_state_7d(obs if isinstance(obs, Mapping) else {})
                step_eef_quat = copy_vec(eef_state[3:7]) if eef_state is not None else None
                mapping = obs if isinstance(obs, Mapping) else {}
                if capture_session.enabled:
                    agent = mapping.get("agent")
                    qpos = agent.get("qpos") if isinstance(agent, Mapping) else None
                    step_joint = copy_vec(qpos)
                    # ManiSkill eef_pos quat is wxyz; WRM tcp_pose wants xyzw.
                    if step_eef_quat is not None and len(step_eef_quat) == 4:
                        w, x, y, z = step_eef_quat
                        cap_quat = (x, y, z, w)
                    else:
                        cap_quat = step_eef_quat
                    try:
                        step_cameras = get_capture_cameras(
                            handle.env,
                            mapping,
                            camera_name=handle.camera_name,
                        )
                        if not step_cameras:
                            step_cameras = None
                    except Exception:
                        step_cameras = None
                    if capture_session.wants_images:
                        try:
                            step_primary = np.asarray(
                                get_primary_image(
                                    handle.env,
                                    mapping,
                                    camera_name=handle.camera_name,
                                )
                            ).copy()
                        except Exception:
                            step_primary = None
                    else:
                        step_primary = None
                else:
                    step_joint = None
                    cap_quat = None
                    step_primary = None
                    step_cameras = None
                include_images = True
                needs_visual = getattr(policy, "needs_visual_observation", None)
                if callable(needs_visual):
                    include_images = bool(needs_visual())
                observation = build_observation(
                    handle,
                    obs if isinstance(obs, Mapping) else {},
                    instruction=str(instruction),
                    step=task_step,
                    needs_state=self._needs_state,
                    include_images=include_images,
                )
                if save_video:
                    frames.append(
                        get_primary_image(
                            handle.env,
                            obs if isinstance(obs, Mapping) else {},
                            camera_name=handle.camera_name,
                        )
                    )

                if len(action_queue) == 0:
                    chunk = policy.act(observation)
                    converted = apply_action_chunk(
                        chunk,
                        obs=obs if isinstance(obs, Mapping) else {},
                        configured_space=self._configured_action_space,
                        task_step=task_step,
                        warmup_steps=self._warmup_steps,
                        post_warmup_space=self._post_warmup_space,
                    )
                    action_queue.refill(converted)

                action = action_queue.popleft()
                step_out = handle.env.step(np.asarray(action, dtype=np.float64))
                if len(step_out) == 5:
                    obs, step_reward, done, truncated, info = step_out
                else:  # pragma: no cover — gym classic
                    obs, step_reward, done, info = step_out
                    truncated = False

                dyn = {}
                if isinstance(info, Mapping):
                    raw = info.get("dynisafety")
                    if isinstance(raw, Mapping):
                        dyn = dict(raw)
                elif hasattr(handle.env, "dynisafety_step_info"):
                    dyn = dict(handle.env.dynisafety_step_info)

                if level == 0:
                    in_contact = False
                    force_n = 0.0
                    impulse_ns = 0.0
                    builder.update_contact(
                        in_contact=in_contact,
                        force_n=force_n,
                        impulse_ns=impulse_ns,
                        partners={},
                    )
                else:
                    in_contact = bool(dyn.get("in_contact", False))
                    force_n = float(dyn.get("force_n", 0.0) or 0.0)
                    impulse_ns = float(dyn.get("impulse_ns", 0.0) or 0.0)
                    builder.update_contact(
                        in_contact=in_contact,
                        force_n=force_n,
                        impulse_ns=impulse_ns,
                        partners=dict(dyn.get("partners") or {}),
                    )
                    if bool(dyn.get("contact_without_force", False)):
                        contact_without_force_steps += 1
                    if level in (1, 2):
                        builder.update_topple(
                            tilt_deg=float(dyn.get("tilt_deg", 0.0) or 0.0),
                            toppled=bool(dyn.get("toppled", False)),
                        )
                    if dyn.get("force_anomaly") or dyn.get("force_anomaly_any"):
                        force_anomaly_any = True
                    pre_physics_prevents = int(
                        dyn.get("pre_physics_prevents", pre_physics_prevents) or 0
                    )

                diag_session.record_step(
                    t=task_step,
                    eef_pos=step_eef_pos,
                    eef_quat=step_eef_quat,
                    action=action,
                    in_contact=in_contact,
                    force_n=force_n,
                    impulse_ns=impulse_ns,
                    base_xy_yaw=None,
                )
                capture_session.record_step(
                    t=task_step,
                    eef_pos=step_eef_pos or (0.0, 0.0, 0.0),
                    eef_quat=cap_quat,
                    action=action,
                    in_contact=in_contact,
                    force_n=force_n,
                    reward=step_reward,
                    joint_position=step_joint,
                    primary=step_primary,
                    cameras=step_cameras,
                )

                try:
                    new_instr = handle.env.get_language_instruction()
                    if new_instr and new_instr != instruction:
                        instruction = new_instr
                    is_final_subtask = bool(handle.env.is_final_subtask())
                except Exception:
                    pass

                task_steps_done = task_step + 1
                if done and is_final_subtask:
                    success = True
                    success_step = task_step
                    termination = "success"
                    break
                if done and not is_final_subtask:
                    try:
                        handle.env.advance_to_next_subtask()
                        is_final_subtask = bool(handle.env.is_final_subtask())
                    except Exception:
                        pass
                if truncated:
                    termination = "max_steps" if task_steps_done >= max_steps else "truncated"
                    break
            else:
                termination = "max_steps"
        except Exception as exc:  # noqa: BLE001 — record failure path
            termination = "error"
            builder_extra_err = str(exc)

        metrics = builder.finalize(
            success=success,
            success_step=success_step,
            steps=task_steps_done,
            force_equiv_dt_s=force_dt,
        )
        diag_session.finalize(
            success=success,
            success_step=success_step,
            metrics=metrics,
        )
        capture_session.finalize(
            success=success,
            success_step=success_step,
            termination=termination,
            metrics=metrics,
            task_id=handle.task.task_id,
            task_name=handle.task.task_name,
            instruction=str(instruction),
            checkpoint=str(cfg.checkpoint),
            config_hash=bound.config_hash,
        )

        video_path: str | None = None
        if save_video and frames:
            results_dir = Path(str(cfg.output.results_dir))
            vid_dir = episode_video_dir(
                results_dir=results_dir,
                backend=str(cfg.backend),
                suite=str(cfg.suite),
                level=level,
                run_id=str(cfg.run_id),
                task_id=handle.task.task_id,
                task_name=handle.task.task_name,
            )
            path = vid_dir / (f"ep{episode_index:04d}_{'success' if success else 'fail'}.mp4")
            fps = int(control_freq)
            save_episode_video(frames, path=path, fps=fps)
            video_path = str(path)

        wall = time.perf_counter() - t0
        extra: dict[str, Any] = {
            "force_anomaly": force_anomaly_any,
            "pre_physics_prevents": pre_physics_prevents,
            "control_mode": handle.control_mode,
            "contact_without_force": contact_without_force_steps > 0,
            "contact_without_force_steps": int(contact_without_force_steps),
        }
        ctrl = getattr(handle.env, "_controller", None)
        spawn_plan = getattr(ctrl, "spawn_plan", None) if ctrl is not None else None
        if isinstance(spawn_plan, dict) and spawn_plan:
            extra["spawn_plan"] = dict(spawn_plan)
        l3_plan = getattr(ctrl, "l3_plan", None) if ctrl is not None else None
        if l3_plan is None and ctrl is not None:
            l3_plan = getattr(ctrl, "reach_plan", None)
        if isinstance(l3_plan, dict) and l3_plan:
            extra["l3_plan"] = dict(l3_plan)
        if ctrl is not None and hasattr(ctrl, "home_inside"):
            extra["l1_home_inside"] = bool(ctrl.home_inside)
            extra["l1_home_onset"] = bool(getattr(ctrl, "home_onset", False))
        if builder_extra_err is not None:
            extra["error"] = builder_extra_err
        if level == 3:
            wrap_extra = getattr(handle.env, "_l3_extra", None)
            if isinstance(wrap_extra, dict):
                extra.update(dict(wrap_extra))
            last = getattr(handle.env, "_last_step_info", None)
            if isinstance(last, dict):
                for key in (
                    "l3_force_scheme",
                    "l3_kinematic_overlap_sim_steps",
                    "l3_kinematic_peak_impulse_ns",
                    "l3_kinematic_peak_force_n",
                    "l3_release_speed_mps",
                    "l3_force_inbound_speed_mps",
                    "l3_partner_speed_mps",
                    "l3_closing_speed_mps",
                    "l3_force_from_release_momentum",
                    "released_to_physics",
                    "ghost_contact_sim_steps",
                ):
                    if key in last and key not in extra:
                        extra[key] = last[key]

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
            instruction=str(instruction),
            seed=int(cfg.seed),
            max_steps=max_steps,
            steps=int(metrics["steps"]),
            termination=termination,
            wall_time_s=float(wall),
            video_path=video_path,
            extra=extra,
            diagnostics=diag_session.episode_diagnostics(),
        )
