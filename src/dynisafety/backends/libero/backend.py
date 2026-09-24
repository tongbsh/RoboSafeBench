"""LIBERO ``Backend`` — list_tasks / make_env / rollout."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dynisafety.backends.libero.capabilities import LIBERO_CAPABILITIES
from dynisafety.backends.libero.contact import (
    collect_obstacle_geom_ids,
    collect_robot_geom_ids,
    collect_task_object_geom_ids,
    libero_records_topple,
    measure_libero_episode_contact,
    measure_tilt_deg,
    merge_pre_rollback_contact,
)
from dynisafety.backends.libero.disturbance.assemble import build_controller
from dynisafety.backends.libero.env import (
    LiberoEnvHandle,
    build_video_path,
    dummy_action,
    ensure_libero_sys_path,
    get_capture_cameras,
    get_primary_image,
    get_wrist_image,
    make_libero_env,
    pad_init_state_for_hand,
    quat2axisangle,
    rescale_obstacles,
    save_episode_video,
)
from dynisafety.backends.libero.obstacle_pack import (
    attach_obstacle_pack_reset,
    overlay_obstacle_pack,
    parse_obstacle_pack,
    plan_for_task,
    with_remesh,
)
from dynisafety.backends.libero.safety import (
    LiberoAegisSession,
    LiberoOracleSession,
    LiberoYieldHomeSession,
    aegis_enabled_from_extra,
)
from dynisafety.backends.libero.safety.phases import probe_controller_phase
from dynisafety.backends.libero.safety.yield_home import (
    apply_arm_qpos,
    extend_sim_step_cap,
    refresh_libero_obs,
    reset_task_horizon_clock,
)
from dynisafety.capture import CaptureSession
from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.contact import is_force_anomaly, is_toppled, resolve_force_equiv_dt
from dynisafety.core.episode import EpisodeRecord
from dynisafety.core.metrics import EpisodeMetricsBuilder
from dynisafety.core.protocols import BackendCapabilities, EnvHandle, TaskSpec
from dynisafety.core.rollout import (
    ActionChunkQueue,
    build_episode_record,
    episode_video_dir,
    require_bound,
    resolve_save_video,
    setup_invalid_extra,
)
from dynisafety.diagnostics import DiagnosticsSession, copy_vec
from dynisafety.policies.action_space import convert_action, effective_action_space
from dynisafety.policies.base import ActionChunk, EpisodeContext, Observation, Policy

_VALID_SUITES = frozenset(
    {
        "safelibero_spatial",
        "safelibero_object",
        "safelibero_goal",
        "safelibero_long",
    }
)


def _find_workspace_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for parent in (here, *here.parents):
        if (parent / "configs" / "backends" / "libero").is_dir() and (
            parent / "pyproject.toml"
        ).is_file():
            return parent
    raise FileNotFoundError(
        "cannot locate workspace root (configs/backends/libero + pyproject.toml)"
    )


def _load_suite_yaml(suite: str, *, workspace: Path | None = None) -> dict[str, Any]:
    root = workspace or _find_workspace_root()
    path = root / "configs" / "backends" / "libero" / "suites" / f"{suite}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO suite config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"suite YAML root must be a mapping: {path}")
    return data


def _sim_step_cap(handle: LiberoEnvHandle, *, max_steps: int, num_steps_wait: int) -> int:
    """Upper bound on env steps so L2/L3 pause can extend the episode.

    VLA budget is still ``max_steps``. This only leaves room for god-view
    hold. Clamped below the robosuite/LIBERO horizon so a timeout is not
    reported as success.
    """
    cap = int(num_steps_wait) + 2 * int(max_steps)
    env = handle.env
    horizon = getattr(env, "horizon", None)
    if horizon is None:
        horizon = getattr(getattr(env, "env", None), "horizon", None)
    if horizon is not None:
        floor = int(num_steps_wait) + int(max_steps)
        cap = min(cap, max(int(horizon) - 1, floor))
    return cap


def _env_horizon(handle: LiberoEnvHandle) -> int | None:
    env = handle.env
    horizon = getattr(env, "horizon", None)
    if horizon is None:
        horizon = getattr(getattr(env, "env", None), "horizon", None)
    return None if horizon is None else int(horizon)


def _suite_max_steps(suite: str, *, workspace: Path | None = None) -> int:
    data = _load_suite_yaml(suite, workspace=workspace)
    runtime = data.get("runtime")
    if not isinstance(runtime, dict) or "max_steps" not in runtime:
        raise ValueError(
            f"suite {suite!r} YAML must define runtime.max_steps "
            f"(configs/backends/libero/suites/{suite}.yaml)"
        )
    return int(runtime["max_steps"])


def _suite_safety_level(suite: str, *, workspace: Path | None = None) -> str:
    data = _load_suite_yaml(suite, workspace=workspace)
    extra = data.get("extra")
    if not isinstance(extra, dict) or "libero_safety_level" not in extra:
        raise ValueError(f"suite {suite!r} YAML must set extra.libero_safety_level (e.g. 'I')")
    level = extra["libero_safety_level"]
    if not isinstance(level, str) or not level:
        raise ValueError(f"extra.libero_safety_level must be a non-empty str, got {level!r}")
    return level


def _require_level_supported(level: int) -> None:
    if level not in LIBERO_CAPABILITIES.levels:
        raise ValueError(
            f"LIBERO backend does not support level={level}; "
            f"supported levels={LIBERO_CAPABILITIES.levels}"
        )


def _apply_obstacle_rescale(handle: LiberoEnvHandle, cfg: RunConfig) -> None:
    """Scale named obstacle assets. Must run after ``env.reset()`` when
    robosuite ``hard_reset=True`` (SafeLIBERO default), which rebuilds MjSim
    from XML and drops any earlier in-memory scale."""
    if cfg.disturbance is MISSING or cfg.disturbance.obstacle is MISSING:
        return
    rescale = cfg.disturbance.obstacle.rescale
    if rescale is MISSING or not rescale:
        return
    rescale_obstacles(handle, dict(rescale))


def _force_equiv_dt_s(cfg: RunConfig) -> float:
    contact = cfg.contact
    if contact is MISSING:
        raise ValueError("cfg.contact is MISSING")
    mode = contact.force_equiv_dt
    if mode is MISSING:
        raise ValueError("contact.force_equiv_dt is MISSING")
    window = contact.force_report_window_s
    window_s = None if window is MISSING else float(window)
    control_freq_hz = None
    runtime = cfg.runtime
    if runtime is not MISSING:
        freq = runtime.control_freq
        if freq is not MISSING:
            control_freq_hz = float(freq)
    return resolve_force_equiv_dt(
        mode,
        sim_timestep_s=None,
        control_freq_hz=control_freq_hz,
        force_report_window_s=window_s,
    )


def _eef_state_7d(obs: Mapping[str, Any]) -> np.ndarray:
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    aa = quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    g = float(grip[0]) if grip.size else 0.0
    return np.concatenate([pos, aa, [g]])


def _proprio_state(obs: Mapping[str, Any]) -> np.ndarray:
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    aa = quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    return np.concatenate([pos, aa, grip]).astype(np.float32)


def _controller_ee_proprio_20d(handle: LiberoEnvHandle) -> np.ndarray:
    """Absolute EEF proprio for X-VLA — matches ``old_cache/.../xvla_source_client.get_ee_proprio``."""
    robot = handle.env.env.robots[0]
    ee_pos = np.asarray(robot.controller.ee_pos, dtype=np.float32).reshape(3)
    rot = np.asarray(robot.controller.ee_ori_mat, dtype=np.float64)
    rot6d = np.concatenate([rot[:3, 0], rot[:3, 1]]).astype(np.float32)
    half = np.concatenate([ee_pos, rot6d, np.asarray([0.0], dtype=np.float32)])
    return np.concatenate([half, np.zeros(10, dtype=np.float32)])


def _set_use_delta(handle: LiberoEnvHandle, *, use_delta: bool) -> None:
    for robot in handle.env.env.robots:
        robot.controller.use_delta = bool(use_delta)


def _build_observation(
    obs: Mapping[str, Any],
    *,
    instruction: str,
    step: int,
    needs_wrist: bool,
    needs_state: bool,
    handle: LiberoEnvHandle | None = None,
) -> Observation:
    images: dict[str, np.ndarray] = {"primary": get_primary_image(obs)}
    if needs_wrist:
        images["wrist"] = get_wrist_image(obs)
    state = _proprio_state(obs) if needs_state else None
    extra: dict[str, Any] = {}
    if handle is not None:
        # X-VLA paper client reads controller ee_pos/ee_ori_mat, not obs quat.
        extra["ee_proprio_20d"] = _controller_ee_proprio_20d(handle)
    return Observation(
        images=images,
        state=state,
        instruction=instruction,
        step=step,
        extra=extra,
    )


def _apply_action_chunk(
    handle: LiberoEnvHandle,
    chunk: ActionChunk,
    *,
    obs: Mapping[str, Any],
    configured_space: str,
    task_step: int,
    warmup_steps: int | None,
    post_warmup_space: str | None,
) -> np.ndarray:
    """Convert a full chunk; returns shape ``(C, D)``."""
    target = effective_action_space(
        configured_space=configured_space,
        step=task_step,
        warmup_steps=warmup_steps,
        post_warmup_space=post_warmup_space,
    )
    _set_use_delta(handle, use_delta=(target == "delta_ee_7d"))
    return convert_action(
        chunk,
        target_space=target,
        context={"eef_state": _eef_state_7d(obs)},
    )


class LiberoBackend:
    """SafeLIBERO backend adapting MuJoCo env + disturbance to ``EpisodeRecord``."""

    name: str = "libero"
    capabilities: BackendCapabilities = LIBERO_CAPABILITIES

    def __init__(self) -> None:
        self._config_hash: str | None = None
        self._extra_libero: dict[str, Any] = {}
        self._resolved_extra: dict[str, Any] = {}
        self._configured_action_space: str | None = None
        self._chunk_size: int | None = None
        self._warmup_steps: int | None = None
        self._post_warmup_space: str | None = None
        self._needs_wrist: bool = True
        self._needs_state: bool = True

    def bind_run(
        self,
        *,
        config_hash: str,
        resolved_data: Mapping[str, Any],
        action_space: str,
        chunk_size: int,
        observation_images: tuple[str, ...],
        observation_needs_state: bool,
        model_extra: Mapping[str, Any],
    ) -> None:
        """Stash ResolvedConfig fields that are not on ``RunConfig``."""
        if not config_hash:
            raise ValueError("config_hash must be non-empty")
        if not action_space:
            raise ValueError("action_space must be non-empty")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        extra = resolved_data.get("extra") or {}
        if not isinstance(extra, Mapping):
            raise TypeError("resolved_data['extra'] must be a mapping")
        libero = extra.get("libero")
        if not isinstance(libero, Mapping):
            raise ValueError(
                "resolved_data.extra.libero missing; fill configs/backends/libero.yaml"
            )
        self._config_hash = str(config_hash)
        self._resolved_extra = dict(extra)
        self._extra_libero = dict(libero)
        self._configured_action_space = str(action_space)
        self._chunk_size = int(chunk_size)
        self._needs_wrist = "wrist" in observation_images
        self._needs_state = bool(observation_needs_state)
        warm = model_extra.get("warmup_steps")
        self._warmup_steps = int(warm) if warm is not None else None
        post = model_extra.get("post_warmup_action_space")
        self._post_warmup_space = str(post) if post is not None else None

    def _env_resolution(self, cfg: RunConfig) -> int:
        runtime = cfg.runtime
        if runtime is not MISSING:
            resolution = getattr(runtime, "env_img_res", MISSING)
            if resolution is not MISSING and resolution is not None:
                value = int(resolution)
                if value <= 0:
                    raise ValueError(f"runtime.env_img_res must be positive, got {value}")
                return value
        if "env_img_res" not in self._extra_libero:
            raise ValueError(
                "env_img_res missing: set runtime.env_img_res (M1) or "
                "extra.libero.env_img_res and call bind_run() before make_env"
            )
        value = int(self._extra_libero["env_img_res"])
        if value <= 0:
            raise ValueError(f"extra.libero.env_img_res must be positive, got {value}")
        return value

    def _num_steps_wait(self) -> int:
        if "num_steps_wait" not in self._extra_libero:
            raise ValueError(
                "extra.libero.num_steps_wait missing; set in configs/backends/libero.yaml "
                "and call bind_run()"
            )
        value = int(self._extra_libero["num_steps_wait"])
        if value < 0:
            raise ValueError(f"num_steps_wait must be >= 0, got {value}")
        return value

    def list_tasks(self, suite: str) -> list[TaskSpec]:
        if suite not in _VALID_SUITES:
            raise ValueError(
                f"unknown LIBERO suite {suite!r}; expected one of "
                f"{sorted(_VALID_SUITES)} (full safelibero_* names, D2=A)"
            )
        max_steps = _suite_max_steps(suite)
        safety_level = _suite_safety_level(suite)

        try:
            ensure_libero_sys_path()
            from libero.libero import benchmark, get_libero_path
        except ImportError as exc:
            raise ImportError(
                "LIBERO/SafeLIBERO is required for LiberoBackend.list_tasks "
                f"({exc}). Install the SafeLIBERO env (M8) before listing tasks."
            ) from exc

        benchmark_dict = benchmark.get_benchmark_dict()
        if suite not in benchmark_dict:
            raise KeyError(
                f"suite {suite!r} not in libero benchmark_dict; available={sorted(benchmark_dict)}"
            )
        task_suite = benchmark_dict[suite](safety_level=safety_level)
        n_tasks = int(task_suite.n_tasks)
        bddl_root = Path(get_libero_path("bddl_files"))

        tasks: list[TaskSpec] = []
        for task_id in range(n_tasks):
            raw = task_suite.get_task(task_id)
            instruction = str(raw.language)
            task_name = str(getattr(raw, "name", f"task_{task_id}"))
            bddl_path = bddl_root / raw.problem_folder / raw.bddl_file
            if not bddl_path.is_file():
                raise FileNotFoundError(f"BDDL not found for task {task_id}: {bddl_path}")
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    task_name=task_name,
                    suite=suite,
                    instruction=instruction,
                    max_steps=max_steps,
                )
            )
        return tasks

    def make_env(self, cfg: RunConfig, task: TaskSpec) -> LiberoEnvHandle:
        if cfg.backend is not MISSING and cfg.backend != self.name:
            raise ValueError(
                f"cfg.backend={cfg.backend!r} does not match LiberoBackend name={self.name!r}"
            )
        _require_level_supported(int(cfg.level) if cfg.level is not MISSING else -1)
        if task.suite not in _VALID_SUITES:
            raise ValueError(f"unknown LIBERO suite on TaskSpec: {task.suite!r}")

        resolution = self._env_resolution(cfg)
        safety_level = _suite_safety_level(task.suite)

        try:
            ensure_libero_sys_path()
            from libero.libero import benchmark, get_libero_path
        except ImportError as exc:
            raise ImportError(
                "LIBERO/SafeLIBERO is required for LiberoBackend.make_env "
                f"({exc}). Install the SafeLIBERO env (M8) before make_env."
            ) from exc

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task.suite](safety_level=safety_level)
        raw = task_suite.get_task(task.task_id)
        bddl_file_name = str(
            Path(get_libero_path("bddl_files")) / raw.problem_folder / raw.bddl_file
        )

        pack = parse_obstacle_pack(self._extra_libero)
        plan = plan_for_task(pack, task.suite, task.task_name)
        if pack is not None and plan is not None:
            pack = with_remesh(pack, plan.remesh)
            if plan.extras:
                print(
                    f"[obstacle_pack] add extras {list(plan.extras)} "
                    f"remesh={dict(plan.remesh)}",
                    flush=True,
                )
        with overlay_obstacle_pack(pack):
            handle = make_libero_env(
                task=task,
                resolution=resolution,
                bddl_file_name=bddl_file_name,
            )
        attach_obstacle_pack_reset(handle.env, pack)

        _apply_obstacle_rescale(handle, cfg)
        return handle

    def _load_init_state(self, handle: LiberoEnvHandle, episode_index: int) -> np.ndarray:
        safety_level = _suite_safety_level(handle.suite)
        try:
            ensure_libero_sys_path()
            from libero.libero import benchmark
        except ImportError as exc:
            raise ImportError(f"LIBERO required to load init states ({exc})") from exc
        task_suite = benchmark.get_benchmark_dict()[handle.suite](safety_level=safety_level)
        init_states = task_suite.get_task_init_states(handle.task_id)
        if init_states is None or len(init_states) == 0:
            raise RuntimeError(f"no init states for suite={handle.suite} task_id={handle.task_id}")
        return np.asarray(init_states[int(episode_index) % len(init_states)])

    def rollout(
        self,
        env: EnvHandle,
        policy: Policy,
        cfg: RunConfig,
        episode_index: int,
    ) -> EpisodeRecord:
        if not isinstance(env, LiberoEnvHandle):
            raise TypeError(f"expected LiberoEnvHandle, got {type(env)!r}")
        bound = require_bound(
            backend="libero",
            config_hash=self._config_hash,
            action_space=self._configured_action_space,
            chunk_size=self._chunk_size,
        )

        handle = env
        level = int(cfg.level)
        _require_level_supported(level)
        max_steps = int(cfg.max_steps)
        num_steps_wait = self._num_steps_wait()
        force_dt = _force_equiv_dt_s(cfg)
        contact_cfg = cfg.contact
        if contact_cfg is MISSING:
            raise ValueError("cfg.contact is MISSING")
        impulse_reduce = contact_cfg.impulse_reduce
        if impulse_reduce is MISSING:
            raise ValueError("contact.impulse_reduce is MISSING")
        topple_thr = contact_cfg.topple_threshold_deg
        if topple_thr is MISSING:
            raise ValueError("contact.topple_threshold_deg is MISSING")
        metrics_cfg = cfg.metrics
        if metrics_cfg is MISSING:
            raise ValueError("cfg.metrics is MISSING")
        force_anomaly_n = metrics_cfg.force_anomaly_threshold_n
        impulse_anomaly_ns = metrics_cfg.impulse_anomaly_threshold_ns
        if force_anomaly_n is MISSING or impulse_anomaly_ns is MISSING:
            raise ValueError(
                "metrics.force_anomaly_threshold_n and "
                "metrics.impulse_anomaly_threshold_ns required (YAML)"
            )
        force_anomaly_n = float(force_anomaly_n)
        impulse_anomaly_ns = float(impulse_anomaly_ns)
        force_anomaly_any = False

        t0 = time.perf_counter()
        handle.env.reset()
        # SafeLIBERO/robosuite default hard_reset=True rebuilds the MjSim from
        # XML, so any in-memory geom/mesh scale from make_env is discarded.
        # Re-apply after reset, before the controller reads geom sizes.
        _apply_obstacle_rescale(handle, cfg)
        init_state = self._load_init_state(handle, episode_index)
        padded = pad_init_state_for_hand(init_state, handle)
        obs = handle.env.set_init_state(padded)

        ctx = EpisodeContext(
            instruction=handle.task_description,
            task_id=handle.task_id,
            task_name=handle.task_name,
            episode_index=episode_index,
            seed=int(cfg.seed),
            level=level,
            backend=str(cfg.backend),
            suite=str(cfg.suite),
            extra={},
        )
        policy.reset(ctx)

        controller = build_controller(
            level,
            handle,
            cfg,
            episode_index=episode_index,
            extra_libero=self._extra_libero,
        )
        if level > 0 and controller is None:
            # Assembly failed: do not run a silent L0 trial under the requested level.
            wall = time.perf_counter() - t0
            metrics = EpisodeMetricsBuilder(metrics_cfg).finalize(
                success=False,
                success_step=None,
                steps=0,
                force_equiv_dt_s=force_dt,
            )
            extra = setup_invalid_extra(
                "disturbance_unassembled",
                extra={"active_obstacle": None},
            )
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
                task_id=handle.task_id,
                task_name=handle.task_name,
                instruction=handle.task_description,
                seed=int(cfg.seed),
                max_steps=max_steps,
                steps=0,
                termination="error",
                wall_time_s=float(wall),
                video_path=None,
                extra=extra,
                diagnostics=None,
            )

        # Prompt 11: pause-VLA oracle (explicit extra.libero.oracle.enabled; L0 bypass).
        oracle_session = LiberoOracleSession.create(
            policy=policy,
            cfg=cfg,
            extra_libero=self._extra_libero,
            action_space=str(self._configured_action_space),
        )
        act_policy = oracle_session.active_policy
        if act_policy is not policy:
            act_policy.reset(ctx)

        # Yield-home before AEGIS: CBF is post-resume only when yield applies
        # (parked L2 residual). Oracle SWR stays mutually exclusive with yield.
        yield_session = LiberoYieldHomeSession.create(
            cfg=cfg,
            extra_libero=self._extra_libero,
            oracle_enabled=oracle_session.enabled,
            aegis_enabled=aegis_enabled_from_extra(self._extra_libero),
        )
        aegis_session = LiberoAegisSession.create(
            cfg=cfg,
            extra_libero=self._extra_libero,
            oracle_enabled=oracle_session.enabled,
            instruction=handle.task_description,
            yield_home_enabled=yield_session.applies_to_level(),
        )
        if isinstance(obs, Mapping):
            yield_session.capture_home_if_needed(obs)

        action_queue = ActionChunkQueue(bound.chunk_size)
        last_gripper: float | None = None
        arm_geom_ids = collect_robot_geom_ids(handle.env)
        # L1/L2 metrics: obstacle vs arm ∪ task objects (bowl contact is logged).
        # L3 person-safety metrics use arm/gripper only (see measure_l3_person_contact_step).
        robot_geom_ids = set(arm_geom_ids)
        if level != 3:
            robot_geom_ids = arm_geom_ids | collect_task_object_geom_ids(handle.env)
        if controller is not None and hasattr(controller, "obstacle_geom_ids"):
            obstacle_geom_ids = set(controller.obstacle_geom_ids)  # type: ignore[attr-defined]
            if hasattr(controller, "set_robot_geom_ids"):
                # Flinch / near-EEF relax: arm+gripper only for all levels.
                controller.set_robot_geom_ids(arm_geom_ids)  # type: ignore[attr-defined]
            set_measure = getattr(controller, "set_contact_measure_params", None)
            if callable(set_measure):
                set_measure(
                    force_equiv_dt_s=force_dt,
                    impulse_reduce=str(impulse_reduce),
                    measure_robot_geom_ids=set(robot_geom_ids),
                )
        else:
            obstacle_geom_ids = collect_obstacle_geom_ids(handle.env)

        topple_body_name: str | None = None
        topple_ref_quat: np.ndarray | None = None
        if (
            libero_records_topple(level)
            and controller is not None
            and hasattr(controller, "_body_id")
        ):
            body_id = int(controller._body_id)  # noqa: SLF001
            topple_body_name = handle.env.env.sim.model.body_id2name(body_id)
            if hasattr(controller, "final_quat"):
                topple_ref_quat = np.asarray(controller.final_quat, dtype=np.float64)
            elif hasattr(controller, "target_quat"):
                topple_ref_quat = np.asarray(controller.target_quat, dtype=np.float64)

        builder = EpisodeMetricsBuilder(metrics_cfg)
        diag_session = DiagnosticsSession.create(
            cfg,
            episode_index=episode_index,
            bound=bound,
            controller=controller,
        )
        capture_session = CaptureSession.create(
            cfg,
            episode_index=episode_index,
            bound=bound,
        )
        frames: list[np.ndarray] = []
        save_video = resolve_save_video(cfg)

        success = False
        success_step: int | None = None
        termination = "max_steps"
        t = 0
        task_steps_done = 0
        vla_budget_used = 0
        vla_budget_total = 0
        vla_budget_exempt_steps = 0
        sim_step_cap = _sim_step_cap(handle, max_steps=max_steps, num_steps_wait=num_steps_wait)

        try:
            while t < sim_step_cap:
                task_step = t - num_steps_wait
                if controller is not None:
                    controller.step(task_step, obs if isinstance(obs, Mapping) else {})

                if t < num_steps_wait:
                    obs, _reward, done, _info = handle.env.step(dummy_action())
                    if np.any(np.isnan(handle.env.env.sim.data.qpos)):
                        termination = "sim_unstable"
                        break
                    t += 1
                    continue

                # use_delta is set in _apply_action_chunk from effective_action_space
                # (abs_ee_7d → False, delta_ee_7d → True). Do not hard-force abs here:
                # that was an X-VLA-only legacy path and breaks delta models if act()
                # is skipped on a step.

                # Copy before env.step: next obs is t+1, and robosuite may
                # reuse the same ndarray buffers in place.
                step_eef_pos = copy_vec(obs["robot0_eef_pos"])
                step_eef_quat = copy_vec(obs["robot0_eef_quat"])
                # WRM extras: only copy images / joints when capture is on.
                # get_*_image reads obs already in hand — no extra render.
                if capture_session.enabled:
                    step_joint = copy_vec(
                        obs["robot0_joint_pos"]
                        if "robot0_joint_pos" in obs
                        else obs.get("robot0_joint_positions")
                    )
                    try:
                        step_cameras = get_capture_cameras(handle.env, obs)
                        if not step_cameras:
                            step_cameras = None
                    except Exception:
                        step_cameras = None
                    if capture_session.wants_images:
                        try:
                            step_primary = np.asarray(get_primary_image(obs)).copy()
                        except Exception:
                            step_primary = None
                        try:
                            step_wrist = np.asarray(get_wrist_image(obs)).copy()
                        except Exception:
                            step_wrist = None
                    else:
                        step_primary = None
                        step_wrist = None
                else:
                    step_joint = None
                    step_primary = None
                    step_wrist = None
                    step_cameras = None

                observation: Observation | None = None
                phase_probe = None
                if yield_session.enabled or aegis_session.enabled:
                    phase_probe = probe_controller_phase(
                        controller,
                        level=level,
                        task_step=task_step,
                        settle_steps=aegis_session.settle_steps,
                    )
                yield_taking = yield_session.should_takeover(
                    None if phase_probe is None else phase_probe.phase,
                    controller=controller,
                    task_step=task_step,
                )
                if yield_session.consume_resume_clear(taking=yield_taking):
                    action_queue.clear()
                    if aegis_session.filter is not None:
                        # Re-init separating axis from the home pose, like a
                        # fresh SafeLIBERO episode (upstream main_aegis.py).
                        aegis_session.filter.reset()
                    if yield_session.should_reset_task_clock():
                        reset_task_horizon_clock(handle.env)
                        vla_budget_used = 0
                        sim_step_cap = extend_sim_step_cap(
                            current_t=t,
                            max_steps=max_steps,
                            current_cap=sim_step_cap,
                            horizon=_env_horizon(handle),
                        )
                        yield_session.note_clock_reset(wall_task_step=task_step)
                        act_policy.reset(ctx)
                # Disturbance / diagnostics keep the wall ``task_step``. After a
                # yield resume the VLA and robosuite horizon counters restart so
                # the policy still has a full ``max_steps`` from home.
                policy_step = (
                    vla_budget_used
                    if yield_session.should_reset_task_clock()
                    else task_step
                )
                if yield_taking:
                    action_queue.clear()
                    budget_exempt = True
                elif oracle_session.enabled or len(action_queue) == 0:
                    observation = _build_observation(
                        obs,
                        instruction=handle.task_description,
                        step=policy_step,
                        needs_wrist=self._needs_wrist,
                        needs_state=self._needs_state,
                        handle=handle,
                    )
                    observation = oracle_session.inject_observation(
                        observation,
                        controller=controller,
                        task_step=task_step,
                    )
                    budget_exempt = oracle_session.exempts_vla_budget(observation)
                else:
                    budget_exempt = False
                if not budget_exempt and vla_budget_used >= max_steps:
                    termination = "max_steps"
                    break
                if save_video:
                    frames.append(get_primary_image(obs))

                if observation is not None and oracle_session.should_clear_action_queue(
                    observation
                ):
                    action_queue.clear()

                action_space_now = effective_action_space(
                    configured_space=str(self._configured_action_space),
                    step=policy_step,
                    warmup_steps=self._warmup_steps,
                    post_warmup_space=self._post_warmup_space,
                )
                if yield_taking:
                    action = yield_session.command(
                        obs if isinstance(obs, Mapping) else {},
                        controller=controller,
                        action_space=action_space_now,
                        last_gripper=last_gripper,
                    )
                    _set_use_delta(handle, use_delta=(action_space_now == "delta_ee_7d"))
                    if yield_session.pending_qpos is not None:
                        apply_arm_qpos(handle.env, yield_session.pending_qpos)
                else:
                    if len(action_queue) == 0:
                        if observation is None:
                            raise RuntimeError("LIBERO act requested without Observation")
                        chunk = act_policy.act(observation)
                        oracle_session.note_after_act()
                        converted = _apply_action_chunk(
                            handle,
                            chunk,
                            obs=obs,
                            configured_space=self._configured_action_space,
                            task_step=policy_step,
                            warmup_steps=self._warmup_steps,
                            post_warmup_space=self._post_warmup_space,
                        )
                        action_queue.refill(converted)

                    action = action_queue.popleft()
                    if aegis_session.enabled and yield_session.allows_aegis_filter():
                        # Phase from the same god-view probe Oracle uses; filter only
                        # when activate_after allows. Yield-home defers this until
                        # resume so CBF never fights the joint snap.
                        action = aegis_session.maybe_filter_action(
                            action,
                            obs=obs if isinstance(obs, Mapping) else {},
                            env=handle.env,
                            controller=controller,
                            task_step=task_step,
                            probe_phase=None if phase_probe is None else phase_probe.phase,
                            action_space=action_space_now,
                        )
                if np.asarray(action).shape[-1] >= 7:
                    last_gripper = float(np.asarray(action, dtype=np.float64).reshape(-1)[6])
                obs, step_reward, done, _info = handle.env.step(action.tolist())
                if yield_taking and yield_session.pending_qpos is not None:
                    apply_arm_qpos(handle.env, yield_session.pending_qpos)
                    obs = refresh_libero_obs(handle.env)
                if np.any(np.isnan(handle.env.env.sim.data.qpos)):
                    termination = "sim_unstable"
                    break

                contact = measure_libero_episode_contact(
                    handle.env,
                    level=level,
                    arm_geom_ids=arm_geom_ids,
                    robot_geom_ids=robot_geom_ids,
                    obstacle_geom_ids=obstacle_geom_ids,
                    force_equiv_dt_s=force_dt,
                    impulse_reduce=str(impulse_reduce),
                )
                pre_rollback = None
                consume = getattr(controller, "consume_pre_rollback_contact", None)
                if controller is not None and callable(consume):
                    pre_rollback = consume()
                if controller is not None and hasattr(controller, "check_robot_collision"):
                    was_blocked = bool(getattr(controller, "blocked", False))
                    controller.check_robot_collision(arm_geom_ids)  # type: ignore[attr-defined]
                    now_blocked = bool(getattr(controller, "blocked", False))
                    if now_blocked and not was_blocked and not contact.in_contact:
                        contact = measure_libero_episode_contact(
                            handle.env,
                            level=level,
                            arm_geom_ids=arm_geom_ids,
                            robot_geom_ids=robot_geom_ids,
                            obstacle_geom_ids=obstacle_geom_ids,
                            force_equiv_dt_s=force_dt,
                            impulse_reduce=str(impulse_reduce),
                        )
                contact = merge_pre_rollback_contact(contact, pre_rollback)

                # MuJoCo penetration recovery can turn one step's impulse into
                # a kilonewton "force". Latch it so the episode is reported as
                # a solver artefact instead of contributing silently to the
                # force tail. The force fields themselves are left untouched.
                if is_force_anomaly(
                    force_n=float(contact.force_n),
                    impulse_ns=float(contact.impulse_ns),
                    force_threshold_n=force_anomaly_n,
                    impulse_threshold_ns=impulse_anomaly_ns,
                ):
                    force_anomaly_any = True

                builder.update_contact(
                    in_contact=contact.in_contact,
                    force_n=contact.force_n,
                    impulse_ns=contact.impulse_ns,
                    partners=contact.partners,
                )
                diag_session.record_step(
                    t=task_step,
                    eef_pos=step_eef_pos,
                    eef_quat=step_eef_quat,
                    action=action,
                    in_contact=contact.in_contact,
                    force_n=contact.force_n,
                    impulse_ns=contact.impulse_ns,
                )
                capture_session.record_step(
                    t=task_step,
                    eef_pos=step_eef_pos,
                    eef_quat=step_eef_quat,
                    action=action,
                    in_contact=contact.in_contact,
                    force_n=contact.force_n,
                    reward=step_reward,
                    joint_position=step_joint,
                    primary=step_primary,
                    wrist=step_wrist,
                    cameras=step_cameras,
                )

                drop_live = True
                if controller is not None and hasattr(controller, "_dropped"):
                    drop_live = bool(controller._dropped)  # noqa: SLF001
                if (
                    libero_records_topple(level)
                    and topple_body_name is not None
                    and drop_live
                ):
                    tilt = measure_tilt_deg(
                        handle.env,
                        topple_body_name,
                        ref_quat=topple_ref_quat,
                    )
                    builder.update_topple(
                        tilt_deg=tilt,
                        toppled=is_toppled(tilt, threshold_deg=float(topple_thr)),
                    )

                task_steps_done = task_step + 1
                if budget_exempt:
                    vla_budget_exempt_steps += 1
                else:
                    vla_budget_used += 1
                    vla_budget_total += 1
                if done:
                    success = True
                    success_step = task_step
                    termination = "success"
                    break
                t += 1
        except Exception as exc:  # noqa: BLE001 — record failure path
            termination = "error"
            builder_extra_err = str(exc)
        else:
            builder_extra_err = None

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
            task_id=handle.task_id,
            task_name=handle.task_name,
            instruction=handle.task_description,
            checkpoint=str(cfg.checkpoint),
            config_hash=bound.config_hash,
        )

        video_path: str | None = None
        if save_video and frames:
            task_slug = handle.task_description.lower().replace(" ", "_").replace("\n", "_")[:40]
            vid_dir = episode_video_dir(
                results_dir=str(cfg.output.results_dir),
                backend=str(cfg.backend),
                suite=str(cfg.suite),
                level=level,
                run_id=str(cfg.run_id),
                task_id=handle.task_id,
                task_name=task_slug,
            )
            path = build_video_path(
                vid_dir,
                episode_idx=episode_index,
                success=success,
                task_description=handle.task_description,
            )
            save_episode_video(frames, path=path)
            video_path = str(path)

        wall = time.perf_counter() - t0
        extra: dict[str, Any] = {
            "active_obstacle": (
                controller.active_obstacle_name if controller is not None else None
            ),
            "num_steps_wait": num_steps_wait,
            "vla_budget_steps": int(vla_budget_total),
            "vla_budget_exempt_steps": int(vla_budget_exempt_steps),
            "force_anomaly": bool(force_anomaly_any),
        }
        if controller is not None:
            plan = getattr(controller, "l3_plan", None)
            if isinstance(plan, dict) and plan:
                extra["l3_plan"] = dict(plan)
            onset_plan = getattr(controller, "onset_plan", None)
            if isinstance(onset_plan, dict) and onset_plan:
                extra["onset_plan"] = dict(onset_plan)
            spawn_plan = getattr(controller, "spawn_plan", None)
            if isinstance(spawn_plan, dict) and spawn_plan:
                extra["spawn_plan"] = dict(spawn_plan)
                extra["spawn_plan"]["rollback_steps"] = int(
                    getattr(controller, "rollback_steps", 0)
                )
        if controller is not None and hasattr(controller, "blocked"):
            extra["disturbance_blocked"] = bool(controller.blocked)
        if controller is not None and hasattr(controller, "released"):
            extra["disturbance_released"] = bool(controller.released)
        block_step = getattr(controller, "block_task_step", None) if controller is not None else None
        if block_step is not None:
            extra["disturbance_block_task_step"] = int(block_step)
        meta = getattr(controller, "pre_rollback_meta", None) if controller is not None else None
        if isinstance(meta, dict) and meta:
            extra["pre_rollback_contact"] = dict(meta)
        if controller is not None and hasattr(controller, "workspace_onset"):
            extra["l1_workspace_onset"] = bool(controller.workspace_onset)
        if (
            cfg.disturbance is not MISSING
            and cfg.disturbance.obstacle is not MISSING
            and cfg.disturbance.obstacle.rescale is not MISSING
            and cfg.disturbance.obstacle.rescale
        ):
            extra["obstacle_rescale"] = {
                str(name): float(scale)
                for name, scale in dict(cfg.disturbance.obstacle.rescale).items()
            }
        extra.update(oracle_session.episode_extra())
        extra.update(aegis_session.episode_extra())
        extra.update(yield_session.episode_extra())
        # Stamp matrix condition from resolved --set extra.feasibility_condition=...
        # (Prompt 14/A2; previously missing from EpisodeRecord.extra).
        fc = self._resolved_extra.get("feasibility_condition")
        if fc is not None and str(fc).strip():
            extra["feasibility_condition"] = str(fc).strip()
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
            task_id=handle.task_id,
            task_name=handle.task_name,
            instruction=handle.task_description,
            seed=int(cfg.seed),
            max_steps=max_steps,
            steps=int(metrics["steps"]),
            termination=termination,
            wall_time_s=float(wall),
            video_path=video_path,
            extra=extra,
            diagnostics=diag_session.episode_diagnostics(),
        )
