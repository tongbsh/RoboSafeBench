"""RoboCasa ``Backend`` — L0–L3 NavigateKitchen plus optional PickPlace A9 suite.

``robocasa_nav``: L1 Frontal Sudden Appear, L2 Base-Directed Approach, L3
Approaching Human. ``robocasa_pnp`` is a separate level-0 PickPlace suite
and does not change nav L0–L3.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dynisafety.backends.robocasa.capabilities import ROBOCASA_CAPABILITIES
from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.contact import is_toppled
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

_VALID_SUITES = frozenset({"robocasa_nav", "robocasa_pnp"})
_PNP_SUITE = "robocasa_pnp"
_NAV_ENV_CLASS = "NavigateKitchen"
_PNP_ENV_CLASSES = frozenset(
    {
        "PickPlaceCounterToSink",
        "PickPlaceCounterToCabinet",
        "PickPlaceStoveToCounter",
    }
)
_PNP_TRACKS = frozenset({"sink_beside", "cab_beside", "stove_beside"})
_PNP_TRACK_OBJECT = {
    "sink_beside": "hazard",
    "cab_beside": "hazard",
    "stove_beside": "hazard",
}
_WINDOW_C_FIX = "FIX：完成窗口 C"
_WINDOW_D_FIX = "FIX：窗口 D"
_WINDOW_E_FIX = "FIX：窗口 E"


def _find_workspace_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for parent in (here, *here.parents):
        if (parent / "configs" / "backends" / "robocasa").is_dir() and (
            parent / "pyproject.toml"
        ).is_file():
            return parent
    raise FileNotFoundError(
        "cannot locate workspace root (configs/backends/robocasa + pyproject.toml)"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def _suite_yaml(suite: str, *, workspace: Path) -> dict[str, Any]:
    path = workspace / "configs" / "backends" / "robocasa" / "suites" / f"{suite}.yaml"
    return _load_yaml(path)


def _suite_max_steps(suite: str, *, workspace: Path) -> int:
    data = _suite_yaml(suite, workspace=workspace)
    runtime = data.get("runtime")
    if not isinstance(runtime, Mapping) or "max_steps" not in runtime:
        raise ValueError(
            f"suite {suite!r} YAML must define runtime.max_steps "
            f"(configs/backends/robocasa/suites/{suite}.yaml)"
        )
    return int(runtime["max_steps"])


def _suite_env_class(data: Mapping[str, Any], *, suite: str) -> str | None:
    extra = data.get("extra")
    robocasa = extra.get("robocasa") if isinstance(extra, Mapping) else None
    if suite == _PNP_SUITE:
        family = robocasa.get("env_family") if isinstance(robocasa, Mapping) else None
        if family != "pickplace":
            raise ValueError(
                f"suite {suite!r} YAML must set extra.robocasa.env_family='pickplace'"
            )
        env_class = robocasa.get("env_class") if isinstance(robocasa, Mapping) else None
        if env_class is not None:
            raise ValueError(
                f"suite {suite!r} must not set extra.robocasa.env_class (got "
                f"{env_class!r}); env_class lives on each task row"
            )
        return None
    env_class = robocasa.get("env_class") if isinstance(robocasa, Mapping) else None
    if not isinstance(env_class, str) or not env_class.strip():
        raise ValueError(
            f"suite {suite!r} YAML must set extra.robocasa.env_class="
            f"{_NAV_ENV_CLASS!r} (PickPlace* is not a primary task)"
        )
    if env_class != _NAV_ENV_CLASS:
        raise ValueError(
            f"suite {suite!r} extra.robocasa.env_class={env_class!r}; "
            f"only {_NAV_ENV_CLASS} is allowed (no PickPlace* primary tasks)"
        )
    return env_class


def _task_instruction(task_name: str, row: Mapping[str, Any]) -> str:
    for key in ("instruction", "prompt"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    extra = row.get("extra")
    if isinstance(extra, Mapping):
        rc = extra.get("robocasa")
        if isinstance(rc, Mapping):
            for key in ("instruction", "prompt"):
                val = rc.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return task_name.replace("_", " ")


def _suite_task_rows(suite: str, *, workspace: Path) -> list[dict[str, Any]]:
    data = _suite_yaml(suite, workspace=workspace)
    _suite_env_class(data, suite=suite)
    rows = data.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"suite {suite!r} YAML must define a non-empty tasks: list "
            f"(configs/backends/robocasa/suites/{suite}.yaml)"
        )
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"suite {suite!r} tasks[{i}] must be a mapping")
        name = row.get("name")
        task_id = row.get("task_id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"suite {suite!r} tasks[{i}] missing name")
        if suite == _PNP_SUITE:
            env_class = row.get("env_class")
            track = row.get("pnp_track")
            row_steps = row.get("max_steps")
            if env_class not in _PNP_ENV_CLASSES:
                raise ValueError(
                    f"suite {suite!r} task {name!r} env_class={env_class!r}; "
                    f"whitelist {sorted(_PNP_ENV_CLASSES)}"
                )
            if track not in _PNP_TRACKS:
                raise ValueError(
                    f"suite {suite!r} task {name!r} pnp_track={track!r}; "
                    f"whitelist {sorted(_PNP_TRACKS)}"
                )
            if isinstance(row_steps, bool) or not isinstance(row_steps, int) or row_steps < 1:
                raise ValueError(
                    f"suite {suite!r} task {name!r} must set max_steps as a positive int"
                )
        elif isinstance(name, str) and name.lower().startswith("pickplace"):
            raise ValueError(
                f"suite {suite!r} task {name!r} is PickPlace*; "
                f"only {_NAV_ENV_CLASS} navigation tasks are allowed"
            )
        if task_id is None:
            raise ValueError(f"suite {suite!r} task {name!r} missing task_id")
        out.append(dict(row))
    return out


def _parse_policy_cameras(model_extra: Mapping[str, Any]) -> dict[str, str]:
    raw = model_extra.get("policy_cameras")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(
            "model extra.policy_cameras must be a non-empty mapping of "
            f"Observation.images key → robosuite camera name ({_WINDOW_C_FIX})"
        )
    out: dict[str, str] = {}
    for key, cam in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"policy_cameras keys must be non-empty str, got {key!r}")
        if not isinstance(cam, str) or not cam.strip():
            raise ValueError(f"policy_cameras[{key!r}] must be a non-empty str, got {cam!r}")
        out[key.strip()] = cam.strip()
    return out


def _pnp_task_row(task: TaskSpec, *, workspace: Path) -> dict[str, Any]:
    rows = _suite_task_rows(task.suite, workspace=workspace)
    for row in rows:
        if int(row["task_id"]) == int(task.task_id) or str(row["name"]) == task.task_name:
            return dict(row)
    raise ValueError(
        f"task_id={task.task_id} name={task.task_name!r} not in suite "
        f"{task.suite!r} YAML ({_WINDOW_C_FIX})"
    )


def _target_fixture_for_task(task: TaskSpec, *, workspace: Path) -> str:
    rows = _suite_task_rows(task.suite, workspace=workspace)
    for row in rows:
        if int(row["task_id"]) == int(task.task_id) or str(row["name"]) == task.task_name:
            fx = row.get("target_fixture")
            if not isinstance(fx, str) or not fx.strip():
                raise ValueError(
                    f"suite {task.suite!r} task {task.task_name!r} missing "
                    f"target_fixture ({_WINDOW_C_FIX})"
                )
            if fx == "Fridge":
                raise ValueError(
                    "target_fixture 'Fridge' is not a class; pin "
                    "FridgeFrenchDoor / FridgeSideBySide / FridgeBottomFreezer. "
                    f"{_WINDOW_C_FIX}"
                )
            return fx.strip()
    raise ValueError(
        f"task_id={task.task_id} name={task.task_name!r} not in suite "
        f"{task.suite!r} YAML ({_WINDOW_C_FIX})"
    )


def _obstacle_half_size_mass_rgba(
    cfg: RunConfig,
) -> tuple[list[float], float, list[float]]:
    from dynisafety.backends.robocasa.env import parse_obstacle_rgba

    dist = cfg.disturbance
    if dist is MISSING:
        raise ValueError("cfg.disturbance is MISSING")
    obstacle = dist.obstacle
    if obstacle is MISSING:
        raise ValueError("disturbance.obstacle is MISSING")
    half = obstacle.half_size
    mass = obstacle.mass
    color = obstacle.color
    if half is MISSING or mass is MISSING:
        raise ValueError(
            f"disturbance.obstacle.half_size and mass required in YAML ({_WINDOW_C_FIX})"
        )
    half_list = [float(x) for x in half]
    if len(half_list) != 3:
        raise ValueError(f"disturbance.obstacle.half_size must be length 3, got {half_list!r}")
    if color is MISSING:
        raise ValueError(f"disturbance.obstacle.color required in YAML ({_WINDOW_C_FIX})")
    return half_list, float(mass), parse_obstacle_rgba(color)


def _seed_stride(cfg: RunConfig, resolved_data: Mapping[str, Any]) -> int:
    runtime = cfg.runtime
    if runtime is not MISSING and runtime.seed_stride is not MISSING:
        return int(runtime.seed_stride)
    resolved_runtime = resolved_data.get("runtime")
    if isinstance(resolved_runtime, Mapping) and "seed_stride" in resolved_runtime:
        return int(resolved_runtime["seed_stride"])
    raise ValueError("seed_stride missing from cfg.runtime or resolved_data.runtime")


def _latched_base_xy_yaw(inner: Any) -> tuple[float, float, float] | None:
    """Read-only chassis pose ``(x, y, yaw)`` at the current sim state.

    ``read_base_xy_yaw`` looks up ``mobilebase0_base`` xpos / xmat. It does
    not step the simulator or draw RNG. A failed read degrades to ``None``
    so diagnostics cannot abort the episode with ``termination="error"``.
    """
    try:
        from dynisafety.backends.robocasa.env import read_base_xy_yaw

        xy, yaw = read_base_xy_yaw(inner)
        return (float(xy[0]), float(xy[1]), float(yaw))
    except Exception:
        return None


def _apply_action_chunk(
    chunk: ActionChunk,
    *,
    configured_space: str,
    task_step: int,
    warmup_steps: int | None,
    post_warmup_space: str | None,
) -> np.ndarray:
    from dynisafety.backends.robocasa.env import to_robosuite_action

    target = effective_action_space(
        configured_space=configured_space,
        step=task_step,
        warmup_steps=warmup_steps,
        post_warmup_space=post_warmup_space,
    )
    if target != "hybrid_mobile_12d":
        raise ValueError(
            f"RoboCasa requires target_space hybrid_mobile_12d, got {target!r}. "
            f"7-D pad-with-zeros is forbidden. {_WINDOW_C_FIX}"
        )
    converted = convert_action(chunk, target_space=target, context={})
    rows = [to_robosuite_action(row) for row in converted]
    return np.stack(rows, axis=0)


class RoboCasaBackend:
    """RoboCasa kitchen navigation backend (L0 rollout in Window C)."""

    name: str = "robocasa"
    capabilities: BackendCapabilities = ROBOCASA_CAPABILITIES

    def __init__(self) -> None:
        self._config_hash: str | None = None
        self._extra_robocasa: dict[str, Any] = {}
        self._resolved_extra: dict[str, Any] = {}
        self._resolved_data: dict[str, Any] = {}
        self._configured_action_space: str | None = None
        self._chunk_size: int | None = None
        self._warmup_steps: int | None = None
        self._post_warmup_space: str | None = None
        self._needs_wrist: bool = False
        self._needs_state: bool = False
        self._observation_images: tuple[str, ...] = ()
        self._policy_cameras: dict[str, str] = {}
        self._workspace: Path | None = None

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
        workspace: Path | None = None,
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
        robocasa = extra.get("robocasa")
        if not isinstance(robocasa, Mapping):
            raise ValueError(
                "resolved_data.extra.robocasa missing; fill "
                "configs/backends/robocasa.yaml and suite YAML"
            )
        if "camera" not in robocasa:
            raise ValueError(
                "extra.robocasa.camera missing; set it in suite YAML "
                f"(no Python default). {_WINDOW_C_FIX}"
            )
        self._config_hash = str(config_hash)
        self._resolved_extra = dict(extra)
        self._extra_robocasa = dict(robocasa)
        self._resolved_data = dict(resolved_data)
        self._configured_action_space = str(action_space)
        self._chunk_size = int(chunk_size)
        self._needs_wrist = "wrist" in observation_images
        self._needs_state = bool(observation_needs_state)
        self._observation_images = tuple(observation_images)
        self._policy_cameras = _parse_policy_cameras(model_extra)
        if self._policy_cameras:
            extra_cams = list(dict.fromkeys(self._policy_cameras.values()))
            self._extra_robocasa["extra_cameras"] = extra_cams
            if "policy_camera_width" in model_extra:
                self._extra_robocasa["policy_camera_width"] = int(
                    model_extra["policy_camera_width"]
                )
            if "policy_camera_height" in model_extra:
                self._extra_robocasa["policy_camera_height"] = int(
                    model_extra["policy_camera_height"]
                )
        warm = model_extra.get("warmup_steps")
        self._warmup_steps = int(warm) if warm is not None else None
        post = model_extra.get("post_warmup_action_space")
        self._post_warmup_space = str(post) if post is not None else None
        self._workspace = workspace

    def list_tasks(self, suite: str) -> list[TaskSpec]:
        if suite not in _VALID_SUITES:
            raise ValueError(
                f"unknown RoboCasa suite {suite!r}; expected one of {sorted(_VALID_SUITES)}"
            )
        workspace = _find_workspace_root()
        rows = _suite_task_rows(suite, workspace=workspace)
        max_steps = _suite_max_steps(suite, workspace=workspace)
        specs: list[TaskSpec] = []
        for row in rows:
            task_name = str(row["name"])
            row_steps = row.get("max_steps")
            task_max = int(row_steps) if row_steps is not None else max_steps
            specs.append(
                TaskSpec(
                    task_id=int(row["task_id"]),
                    task_name=task_name,
                    suite=suite,
                    instruction=_task_instruction(task_name, row),
                    max_steps=task_max,
                )
            )
        return specs

    def make_env(self, cfg: RunConfig, task: TaskSpec) -> EnvHandle:
        level = int(cfg.level)
        if level not in ROBOCASA_CAPABILITIES.levels:
            raise ValueError(
                f"RoboCasa backend does not support level={level}; "
                f"supported levels={ROBOCASA_CAPABILITIES.levels}"
            )
        if self._config_hash is None or not self._extra_robocasa:
            raise RuntimeError(
                f"RoboCasaBackend.bind_run(...) must be called before make_env ({_WINDOW_C_FIX})"
            )
        if self._configured_action_space != "hybrid_mobile_12d":
            raise ValueError(
                "RoboCasa action_space must be hybrid_mobile_12d, got "
                f"{self._configured_action_space!r}. {_WINDOW_C_FIX}"
            )
        if self._needs_wrist and "wrist" not in self._policy_cameras:
            raise ValueError(
                "wrist images requested but model extra.policy_cameras has no "
                f"'wrist' camera key. {_WINDOW_C_FIX}"
            )

        workspace = self._workspace or _find_workspace_root()
        runtime = cfg.runtime
        if runtime is MISSING or runtime.control_freq is MISSING:
            raise ValueError("runtime.control_freq missing from YAML")
        if str(task.suite) == _PNP_SUITE:
            if int(cfg.level) != 0:
                raise ValueError(f"robocasa_pnp only allows level 0, got {cfg.level}")
            row = _pnp_task_row(task, workspace=workspace)
            from dynisafety.backends.robocasa.env import make_robocasa_pnp_env

            return make_robocasa_pnp_env(
                task=task,
                extra_robocasa=self._extra_robocasa,
                control_freq=int(runtime.control_freq),
                max_steps=int(row["max_steps"]),
                seed=int(cfg.seed),
                env_class=str(row["env_class"]),
                pnp_track=str(row["pnp_track"]),
                workspace=workspace,
            )
        target_fixture = _target_fixture_for_task(task, workspace=workspace)
        half_size, mass, rgba = _obstacle_half_size_mass_rgba(cfg)
        if level == 1:
            l1_block = self._extra_robocasa.get("l1")
            if not isinstance(l1_block, Mapping) or "half_size" not in l1_block:
                raise ValueError(f"extra.robocasa.l1.half_size required in YAML ({_WINDOW_C_FIX})")
            half_size = [float(x) for x in l1_block["half_size"]]
            if len(half_size) != 3:
                raise ValueError(f"extra.robocasa.l1.half_size must be length 3, got {half_size!r}")
        if level == 2:
            l2_block = self._extra_robocasa.get("l2")
            if not isinstance(l2_block, Mapping) or "half_size" not in l2_block:
                raise ValueError(f"extra.robocasa.l2.half_size required in YAML ({_WINDOW_E_FIX})")
            half_size = [float(x) for x in l2_block["half_size"]]
            if len(half_size) != 3:
                raise ValueError(f"extra.robocasa.l2.half_size must be length 3, got {half_size!r}")
        runtime = cfg.runtime
        if runtime is MISSING or runtime.control_freq is MISSING:
            raise ValueError("runtime.control_freq missing from YAML")
        max_steps = int(cfg.max_steps)
        seed = int(cfg.seed)

        l3_half: list[float] | None = None
        l3_mass: float | None = None
        l3_mesh_path: str | None = None
        l3_mesh_rgba: list[float] | None = None
        l12_visuals = None
        l3_visuals = None
        if level in (1, 2):
            from dynisafety.backends.robocasa.visuals import parse_l12_visual_catalog

            disturbance = cfg.disturbance
            if disturbance is MISSING or disturbance.obstacle is MISSING:
                raise ValueError("disturbance.obstacle required for RoboCasa L1/L2 visuals")
            pool = disturbance.obstacle.pool
            if pool is MISSING or not pool:
                raise ValueError("disturbance.obstacle.pool required for RoboCasa L1/L2")
            l12_visuals = parse_l12_visual_catalog(self._extra_robocasa, [str(x) for x in pool])
        if level == 3:
            from dynisafety.backends.robocasa.env import parse_obstacle_rgba
            from dynisafety.backends.robocasa.visuals import parse_l3_visual_parts

            l3_block = self._extra_robocasa.get("l3")
            if not isinstance(l3_block, Mapping):
                raise ValueError(f"extra.robocasa.l3 missing from resolved YAML ({_WINDOW_E_FIX})")
            required = ("half_size", "mass", "mesh_path", "mesh_rgba")
            missing = [key for key in required if key not in l3_block]
            if missing:
                raise ValueError(
                    "extra.robocasa.l3 missing "
                    + ", ".join(missing)
                    + f" ({_WINDOW_E_FIX})"
                )
            l3_half = [float(x) for x in l3_block["half_size"]]
            if len(l3_half) != 3:
                raise ValueError(f"extra.robocasa.l3.half_size must be length 3, got {l3_half!r}")
            l3_mass = float(l3_block["mass"])
            l3_mesh_path = str(l3_block["mesh_path"]).strip()
            if not l3_mesh_path:
                raise ValueError(f"extra.robocasa.l3.mesh_path must be non-empty ({_WINDOW_E_FIX})")
            l3_mesh_rgba = parse_obstacle_rgba(
                l3_block["mesh_rgba"], where="extra.robocasa.l3.mesh_rgba"
            )
            l3_visuals = parse_l3_visual_parts(l3_block)

        from dynisafety.backends.robocasa.env import make_robocasa_env

        return make_robocasa_env(
            task=task,
            extra_robocasa=self._extra_robocasa,
            control_freq=int(runtime.control_freq),
            max_steps=max_steps,
            seed=seed,
            obstacle_half_size=half_size,
            obstacle_mass=mass,
            obstacle_rgba=rgba,
            target_fixture=target_fixture,
            workspace=workspace,
            level=level,
            l3_half_size=l3_half,
            l3_mass=l3_mass,
            l3_mesh_path=l3_mesh_path,
            l3_mesh_rgba=l3_mesh_rgba,
            l12_visuals=l12_visuals,
            l3_visuals=l3_visuals,
        )

    def rollout(
        self,
        env: EnvHandle,
        policy: Policy,
        cfg: RunConfig,
        episode_index: int,
    ) -> EpisodeRecord:
        from dynisafety.backends.robocasa.contact import (
            collect_obstacle_geom_ids,
            collect_robot_geom_ids,
            force_equiv_dt_s,
            measure_contact_step,
            measure_tilt_deg,
        )
        from dynisafety.backends.robocasa.disturbance._support import (
            layout_has_island,
            snapshot_geom_contact,
        )
        from dynisafety.backends.robocasa.disturbance.assemble import build_controller
        from dynisafety.backends.robocasa.env import (
            RoboCasaEnvHandle,
            _public_l12_meta,
            apply_pinned_target_fixture,
            build_video_path,
            check_navigate_success,
            eef_pos_from_obs,
            get_primary_image,
            hide_l0_obstacle_actor,
            hide_l3_pedestrian_actor,
            read_base_joint_qpos,
            read_base_xy_yaw,
            robocasa365_state_14d,
            save_episode_video,
            to_robosuite_action,
        )

        if not isinstance(env, RoboCasaEnvHandle):
            raise TypeError(f"expected RoboCasaEnvHandle, got {type(env)!r}")
        if str(env.suite) == _PNP_SUITE:
            return self._rollout_pnp(env, policy, cfg, episode_index)
        bound = require_bound(
            backend="robocasa",
            config_hash=self._config_hash,
            action_space=self._configured_action_space,
            chunk_size=self._chunk_size,
        )

        handle = env
        level = int(cfg.level)
        max_steps = int(cfg.max_steps)
        force_dt = force_equiv_dt_s(cfg, missing=MISSING)
        contact_cfg = cfg.contact
        if contact_cfg is MISSING:
            raise ValueError("cfg.contact is MISSING")
        impulse_reduce = contact_cfg.impulse_reduce
        if impulse_reduce is MISSING:
            raise ValueError("contact.impulse_reduce is MISSING")
        topple_thr = contact_cfg.topple_threshold_deg
        if level in (1, 2) and topple_thr is MISSING:
            raise ValueError("contact.topple_threshold_deg is MISSING")
        metrics_cfg = cfg.metrics
        if metrics_cfg is MISSING:
            raise ValueError("cfg.metrics is MISSING")
        runtime = cfg.runtime
        if runtime is MISSING or runtime.control_freq is MISSING:
            raise ValueError("runtime.control_freq missing from YAML")
        control_freq = int(runtime.control_freq)
        seed_stride = _seed_stride(cfg, self._resolved_data)
        ep_seed = int(cfg.seed) + int(episode_index) * seed_stride

        t0 = time.perf_counter()
        inner = handle.env
        inner.rng = np.random.default_rng(ep_seed)
        obs = inner.reset()
        meta_l12 = getattr(inner, "_dynisafety_l12", None)
        if isinstance(meta_l12, dict):
            meta_l12["geom_contact"] = snapshot_geom_contact(inner)
        hide_info = hide_l0_obstacle_actor(inner)
        meta_l3 = getattr(inner, "_dynisafety_l3", None)
        if isinstance(meta_l3, dict):
            from dynisafety.backends.robocasa.disturbance._support import (
                snapshot_actor_geom_contact,
            )

            if not meta_l3.get("geom_contact"):
                meta_l3["geom_contact"] = snapshot_actor_geom_contact(inner, meta_l3)
            if level != 3:
                hide_l3_pedestrian_actor(inner)
        pin_info = apply_pinned_target_fixture(inner, handle.target_fixture_class)
        obs = inner._get_observations() if hasattr(inner, "_get_observations") else obs

        controller = build_controller(
            level,
            handle,
            cfg,
            extra_robocasa=self._extra_robocasa,
            episode_index=episode_index,
        )

        base_xy0, base_yaw0 = read_base_xy_yaw(inner)
        base_qpos0 = read_base_joint_qpos(inner)
        splits = dict(handle.extra.get("action_split_indexes") or {})

        ctx = EpisodeContext(
            instruction=handle.task_description,
            task_id=handle.task_id,
            task_name=handle.task_name,
            episode_index=episode_index,
            seed=int(cfg.seed),
            level=level,
            backend=str(cfg.backend),
            suite=str(cfg.suite),
            extra={"target_fixture": handle.target_fixture_class, **pin_info},
        )
        policy.reset(ctx)

        action_queue = ActionChunkQueue(bound.chunk_size)
        robot_geom_ids = collect_robot_geom_ids(inner)
        if controller is not None:
            obstacle_geom_ids = set(controller.obstacle_geom_ids)
            setter = getattr(controller, "set_robot_geom_ids", None)
            if callable(setter):
                setter(robot_geom_ids)
        else:
            obstacle_geom_ids = collect_obstacle_geom_ids(inner)
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
            fps=control_freq,
        )
        frames: list[np.ndarray] = []
        save_video = resolve_save_video(cfg)
        spawn_snapshot_path: str | None = None
        onset_marked = False

        success = False
        success_step: int | None = None
        termination = "max_steps"
        task_steps_done = 0
        builder_extra_err: str | None = None
        camera = handle.camera
        err_fix = _WINDOW_E_FIX if level >= 2 else (_WINDOW_D_FIX if level == 1 else _WINDOW_C_FIX)

        try:
            for task_step in range(max_steps):
                if controller is not None:
                    controller.step(task_step, obs if isinstance(obs, Mapping) else {})
                    if not onset_marked and controller.disturbance_onset_step is not None:
                        onset_marked = True
                        robot_geom_ids = collect_robot_geom_ids(inner)
                        setter = getattr(controller, "set_robot_geom_ids", None)
                        if callable(setter):
                            setter(robot_geom_ids)
                        if spawn_snapshot_path is None:
                            spawn_snapshot_path = _save_spawn_snapshot(
                                inner,
                                camera=camera,
                                cfg=cfg,
                                handle=handle,
                                level=level,
                                episode_index=episode_index,
                                body_name=getattr(controller, "body_name", None),
                            )
                    elif spawn_snapshot_path is None and bool(
                        getattr(controller, "spawn_failed", False)
                    ):
                        spawn_snapshot_path = _save_spawn_snapshot(
                            inner,
                            camera=camera,
                            cfg=cfg,
                            handle=handle,
                            level=level,
                            episode_index=episode_index,
                            body_name=getattr(controller, "body_name", None),
                        )

                if not isinstance(obs, Mapping):
                    raise TypeError(f"expected mapping obs, got {type(obs)!r}")
                # Copy before inner.step: next obs is t+1, and robosuite may
                # reuse the same ndarray buffers in place. Chassis pose is the
                # same instant as EEF (read-only; no extra sim step).
                step_eef_pos = copy_vec(eef_pos_from_obs(obs, inner))
                step_eef_quat = (
                    copy_vec(obs["robot0_base_to_eef_quat"])
                    if "robot0_base_to_eef_quat" in obs
                    else None
                )
                step_base_xy_yaw = copy_vec(_latched_base_xy_yaw(inner))
                images = {"primary": get_primary_image(obs, camera=camera)}
                for img_key, cam_name in self._policy_cameras.items():
                    images[img_key] = get_primary_image(obs, camera=cam_name)
                state = None
                if self._needs_state:
                    if all(
                        key in obs
                        for key in (
                            "robot0_base_to_eef_pos",
                            "robot0_base_to_eef_quat",
                            "robot0_gripper_qpos",
                            "robot0_base_pos",
                            "robot0_base_quat",
                        )
                    ):
                        state = robocasa365_state_14d(obs)
                    elif "robot0_eef_pos" in obs:
                        state = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                observation = Observation(
                    images=images,
                    state=state,
                    instruction=handle.task_description,
                    step=task_step,
                    extra={
                        "target_fixture": handle.target_fixture_class,
                        **pin_info,
                    },
                )
                if capture_session.enabled:
                    step_joint = copy_vec(
                        obs["robot0_joint_pos"]
                        if "robot0_joint_pos" in obs
                        else obs.get("robot0_joint_positions")
                    )
                    if "robot0_eef_pos" in obs:
                        cap_eef_pos = copy_vec(obs["robot0_eef_pos"])
                        cap_eef_quat = (
                            copy_vec(obs["robot0_eef_quat"])
                            if "robot0_eef_quat" in obs
                            else step_eef_quat
                        )
                    else:
                        cap_eef_pos = step_eef_pos
                        cap_eef_quat = step_eef_quat
                    cap_base = None
                    if "robot0_base_pos" in obs:
                        base_pos = copy_vec(obs["robot0_base_pos"])
                        base_quat = (
                            copy_vec(obs["robot0_base_quat"])
                            if "robot0_base_quat" in obs
                            else None
                        )
                        if base_pos is not None:
                            cap_base = tuple(base_pos) + (
                                tuple(base_quat) if base_quat is not None else (0.0, 0.0, 0.0, 1.0)
                            )
                    if capture_session.wants_images:
                        step_primary = np.asarray(images["primary"]).copy()
                        step_wrist = (
                            np.asarray(images["wrist"]).copy()
                            if "wrist" in images
                            else None
                        )
                    else:
                        step_primary = None
                        step_wrist = None
                else:
                    step_joint = None
                    cap_eef_pos = None
                    cap_eef_quat = None
                    cap_base = None
                    step_primary = None
                    step_wrist = None
                if save_video:
                    frames.append(images["primary"])

                if len(action_queue) == 0:
                    chunk = policy.act(observation)
                    converted = _apply_action_chunk(
                        chunk,
                        configured_space=self._configured_action_space,
                        task_step=task_step,
                        warmup_steps=self._warmup_steps,
                        post_warmup_space=self._post_warmup_space,
                    )
                    action_queue.refill(converted)

                action = to_robosuite_action(action_queue.popleft())
                step_out = inner.step(action)
                if len(step_out) == 5:
                    obs, step_reward, done, _truncated, _info = step_out
                else:
                    obs, step_reward, done, _info = step_out
                del done  # ignore_done=True; success is _check_success()
                if np.any(np.isnan(inner.sim.data.qpos)):
                    termination = "sim_unstable"
                    break

                if controller is not None:
                    checker = getattr(controller, "check_robot_collision", None)
                    if callable(checker):
                        checker(robot_geom_ids)

                contact = measure_contact_step(
                    inner,
                    robot_geom_ids=robot_geom_ids,
                    obstacle_geom_ids=obstacle_geom_ids,
                    force_equiv_dt_s=force_dt,
                    impulse_reduce=str(impulse_reduce),
                )
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
                    base_xy_yaw=step_base_xy_yaw,
                    action=action,
                    in_contact=contact.in_contact,
                    force_n=contact.force_n,
                    impulse_ns=contact.impulse_ns,
                )
                capture_session.record_step(
                    t=task_step,
                    eef_pos=cap_eef_pos or step_eef_pos or (0.0, 0.0, 0.0),
                    eef_quat=cap_eef_quat or step_eef_quat,
                    action=action,
                    in_contact=contact.in_contact,
                    force_n=contact.force_n,
                    reward=step_reward,
                    joint_position=step_joint,
                    primary=step_primary,
                    wrist=step_wrist,
                    base_pose=cap_base,
                    base_xy_yaw=step_base_xy_yaw,
                )
                if (
                    controller is not None
                    and level in (1, 2)
                    and (
                        bool(getattr(controller, "_dropped", False))
                        or bool(getattr(controller, "_spawned", False))
                    )
                    and topple_thr is not MISSING
                ):
                    tilt = measure_tilt_deg(
                        inner,
                        controller.body_name,
                        ref_quat=controller.target_quat,
                    )
                    builder.update_topple(
                        tilt_deg=tilt,
                        toppled=is_toppled(tilt, threshold_deg=float(topple_thr)),
                    )
                task_steps_done = task_step + 1
                if check_navigate_success(inner):
                    success = True
                    success_step = task_step
                    termination = "success"
                    break
        except Exception as exc:  # noqa: BLE001 — record failure path
            termination = "error"
            builder_extra_err = f"{exc} ({err_fix})"

        if (
            controller is not None
            and spawn_snapshot_path is None
            and (
                bool(getattr(controller, "_dropped", False))
                or bool(getattr(controller, "_spawned", False))
                or bool(getattr(controller, "_motion_armed", False))
                or bool(getattr(controller, "blocked", False))
            )
        ):
            spawn_snapshot_path = _save_spawn_snapshot(
                inner,
                camera=camera,
                cfg=cfg,
                handle=handle,
                level=level,
                episode_index=episode_index,
                body_name=getattr(controller, "body_name", None),
            )

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

        base_xy1, base_yaw1 = read_base_xy_yaw(inner)
        base_qpos1 = read_base_joint_qpos(inner)
        xy_delta = float(np.linalg.norm(base_xy1 - base_xy0))
        qpos_delta = 0.0
        for key, v0 in base_qpos0.items():
            v1 = float(base_qpos1.get(key, v0))
            qpos_delta = max(qpos_delta, abs(v1 - float(v0)))
        base_moved = bool(xy_delta > 1e-4 or qpos_delta > 1e-4)

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
            save_episode_video(frames, path=path, fps=control_freq)
            video_path = str(path)

        wall = time.perf_counter() - t0
        extra: dict[str, Any] = {
            "target_fixture_class": handle.target_fixture_class,
            "camera": camera,
            "action_space": "hybrid_mobile_12d",
            "action_split_indexes": splits,
            "freeze_base": False,
            "success_source": "NavigateKitchen._check_success",
            "base_xy_start": [float(base_xy0[0]), float(base_xy0[1])],
            "base_xy_end": [float(base_xy1[0]), float(base_xy1[1])],
            "base_xy_delta_m": xy_delta,
            "base_yaw_start": float(base_yaw0),
            "base_yaw_end": float(base_yaw1),
            "base_joint_qpos_start": {k: float(v) for k, v in base_qpos0.items()},
            "base_joint_qpos_end": {k: float(v) for k, v in base_qpos1.items()},
            "base_joint_qpos_peak_abs_delta": float(qpos_delta),
            "base_moved": base_moved,
            "ep_seed": int(ep_seed),
            "hidden_obstacle": hide_info,
            "l12_obstacle": _public_l12_meta(meta_l12),
            "l3_human": _public_l12_meta(meta_l3) if isinstance(meta_l3, dict) else None,
            "layout_id": (
                None if getattr(inner, "layout_id", None) is None else int(inner.layout_id)
            ),
            "layout_has_island": bool(layout_has_island(inner)),
            "spawn_snapshot": spawn_snapshot_path,
            "skipped_lightwheel_props_missing_objs_lw": handle.extra.get(
                "skipped_lightwheel_props_missing_objs_lw"
            ),
            "gentex_tex_aliases": handle.extra.get("gentex_tex_aliases"),
            **pin_info,
        }
        if builder_extra_err is not None:
            extra["error"] = builder_extra_err
        if termination == "error" and builder_extra_err is None:
            extra["error"] = f"env_error ({err_fix})"
        extra["spawn_failed"] = False
        if controller is not None:
            fin = getattr(controller, "finalize", None)
            if callable(fin):
                fin()
            extra.update(controller.extra_fields())

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

    def _rollout_pnp(
        self,
        env: EnvHandle,
        policy: Policy,
        cfg: RunConfig,
        episode_index: int,
    ) -> EpisodeRecord:
        from dynisafety.backends.robocasa.contact import (
            collect_pnp_hazard_geom_ids,
            collect_robot_geom_ids,
            force_equiv_dt_s,
            measure_contact_step,
        )
        from dynisafety.backends.robocasa.disturbance.assemble import build_controller
        from dynisafety.backends.robocasa.disturbance.pnp import (
            pnp_should_stop_on_official_success,
        )
        from dynisafety.backends.robocasa.env import (
            RoboCasaEnvHandle,
            build_video_path,
            compose_pnp_review_frame,
            eef_pos_from_obs,
            get_primary_image,
            read_base_joint_qpos,
            read_base_xy_yaw,
            robocasa365_state_14d,
            save_episode_video,
            to_robosuite_action,
        )

        if not isinstance(env, RoboCasaEnvHandle):
            raise TypeError(f"expected RoboCasaEnvHandle, got {type(env)!r}")
        bound = require_bound(
            backend="robocasa",
            config_hash=self._config_hash,
            action_space=self._configured_action_space,
            chunk_size=self._chunk_size,
        )
        handle = env
        level = int(cfg.level)
        if level != 0:
            raise ValueError(f"robocasa_pnp only allows level 0, got {level}")
        task_horizon = handle.extra.get("max_steps")
        if task_horizon is None:
            raise ValueError("robocasa_pnp handle.extra['max_steps'] missing")
        max_steps = min(int(cfg.max_steps), int(task_horizon))
        force_dt = force_equiv_dt_s(cfg, missing=MISSING)
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
        runtime = cfg.runtime
        if runtime is MISSING or runtime.control_freq is MISSING:
            raise ValueError("runtime.control_freq missing from YAML")
        control_freq = int(runtime.control_freq)
        seed_stride = _seed_stride(cfg, self._resolved_data)
        ep_seed = int(cfg.seed) + int(episode_index) * seed_stride

        t0 = time.perf_counter()
        inner = handle.env
        inner.rng = np.random.default_rng(ep_seed)
        obs = inner.reset()
        controller = build_controller(
            level,
            handle,
            cfg,
            extra_robocasa=self._extra_robocasa,
            episode_index=episode_index,
        )
        if controller is None:
            raise RuntimeError("robocasa_pnp build_controller returned None")

        base_xy0, base_yaw0 = read_base_xy_yaw(inner)
        base_qpos0 = read_base_joint_qpos(inner)
        splits = dict(handle.extra.get("action_split_indexes") or {})
        track = str(handle.extra.get("pnp_track") or "")
        object_name = _PNP_TRACK_OBJECT.get(track)
        if object_name is None:
            raise ValueError(f"unknown pnp_track {track!r}")

        ctx = EpisodeContext(
            instruction=handle.task_description,
            task_id=handle.task_id,
            task_name=handle.task_name,
            episode_index=episode_index,
            seed=int(cfg.seed),
            level=level,
            backend=str(cfg.backend),
            suite=str(cfg.suite),
            extra={"pnp_track": track, "env_class": handle.extra.get("env_class")},
        )
        policy.reset(ctx)

        action_queue = ActionChunkQueue(bound.chunk_size)
        robot_geom_ids = collect_robot_geom_ids(inner)
        obstacle_geom_ids = collect_pnp_hazard_geom_ids(inner, object_name=object_name)
        setter = getattr(controller, "set_robot_geom_ids", None)
        if callable(setter):
            setter(robot_geom_ids)
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
            fps=control_freq,
        )
        frames: list[np.ndarray] = []
        save_video = resolve_save_video(cfg)
        success = False
        success_step: int | None = None
        termination = "max_steps"
        task_steps_done = 0
        builder_extra_err: str | None = None
        camera = handle.camera
        success_at_reset = bool(inner._check_success())

        try:
            for task_step in range(max_steps):
                controller.step(task_step, obs if isinstance(obs, Mapping) else {})
                if not isinstance(obs, Mapping):
                    raise TypeError(f"expected mapping obs, got {type(obs)!r}")
                step_eef_pos = copy_vec(eef_pos_from_obs(obs, inner))
                step_eef_quat = (
                    copy_vec(obs["robot0_base_to_eef_quat"])
                    if "robot0_base_to_eef_quat" in obs
                    else None
                )
                step_base_xy_yaw = copy_vec(_latched_base_xy_yaw(inner))
                images = {"primary": get_primary_image(obs, camera=camera)}
                for img_key, cam_name in self._policy_cameras.items():
                    images[img_key] = get_primary_image(obs, camera=cam_name)
                state = None
                if self._needs_state:
                    if all(
                        key in obs
                        for key in (
                            "robot0_base_to_eef_pos",
                            "robot0_base_to_eef_quat",
                            "robot0_gripper_qpos",
                            "robot0_base_pos",
                            "robot0_base_quat",
                        )
                    ):
                        state = robocasa365_state_14d(obs)
                    elif "robot0_eef_pos" in obs:
                        state = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                observation = Observation(
                    images=images,
                    state=state,
                    instruction=handle.task_description,
                    step=task_step,
                    extra={"pnp_track": track},
                )
                if capture_session.enabled:
                    step_joint = copy_vec(
                        obs["robot0_joint_pos"]
                        if "robot0_joint_pos" in obs
                        else obs.get("robot0_joint_positions")
                    )
                    if "robot0_eef_pos" in obs:
                        cap_eef_pos = copy_vec(obs["robot0_eef_pos"])
                        cap_eef_quat = (
                            copy_vec(obs["robot0_eef_quat"])
                            if "robot0_eef_quat" in obs
                            else step_eef_quat
                        )
                    else:
                        cap_eef_pos = step_eef_pos
                        cap_eef_quat = step_eef_quat
                    cap_base = None
                    if "robot0_base_pos" in obs:
                        base_pos = copy_vec(obs["robot0_base_pos"])
                        base_quat = (
                            copy_vec(obs["robot0_base_quat"])
                            if "robot0_base_quat" in obs
                            else None
                        )
                        if base_pos is not None:
                            cap_base = tuple(base_pos) + (
                                tuple(base_quat) if base_quat is not None else (0.0, 0.0, 0.0, 1.0)
                            )
                    if capture_session.wants_images:
                        step_primary = np.asarray(images["primary"]).copy()
                        step_wrist = (
                            np.asarray(images["wrist"]).copy()
                            if "wrist" in images
                            else None
                        )
                    else:
                        step_primary = None
                        step_wrist = None
                else:
                    step_joint = None
                    cap_eef_pos = None
                    cap_eef_quat = None
                    cap_base = None
                    step_primary = None
                    step_wrist = None
                if save_video:
                    frames.append(compose_pnp_review_frame(images))

                if len(action_queue) == 0:
                    chunk = policy.act(observation)
                    converted = _apply_action_chunk(
                        chunk,
                        configured_space=self._configured_action_space,
                        task_step=task_step,
                        warmup_steps=self._warmup_steps,
                        post_warmup_space=self._post_warmup_space,
                    )
                    action_queue.refill(converted)

                action = to_robosuite_action(action_queue.popleft())
                step_out = inner.step(action)
                if len(step_out) == 5:
                    obs, step_reward, done, _truncated, _info = step_out
                else:
                    obs, step_reward, done, _info = step_out
                del done
                if np.any(np.isnan(inner.sim.data.qpos)):
                    termination = "sim_unstable"
                    break

                contact = measure_contact_step(
                    inner,
                    robot_geom_ids=robot_geom_ids,
                    obstacle_geom_ids=obstacle_geom_ids,
                    force_equiv_dt_s=force_dt,
                    impulse_reduce=str(impulse_reduce),
                )
                note = getattr(controller, "note_robot_contact", None)
                if callable(note):
                    note(contact.in_contact)
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
                    base_xy_yaw=step_base_xy_yaw,
                    action=action,
                    in_contact=contact.in_contact,
                    force_n=contact.force_n,
                    impulse_ns=contact.impulse_ns,
                )
                capture_session.record_step(
                    t=task_step,
                    eef_pos=cap_eef_pos or step_eef_pos or (0.0, 0.0, 0.0),
                    eef_quat=cap_eef_quat or step_eef_quat,
                    action=action,
                    in_contact=contact.in_contact,
                    force_n=contact.force_n,
                    reward=step_reward,
                    joint_position=step_joint,
                    primary=step_primary,
                    wrist=step_wrist,
                    base_pose=cap_base,
                    base_xy_yaw=step_base_xy_yaw,
                )
                task_steps_done = task_step + 1
                official_now = bool(inner._check_success())
                if pnp_should_stop_on_official_success(
                    track=track,
                    success_at_reset=success_at_reset,
                    official_now=official_now,
                ):
                    success = True
                    success_step = task_step
                    termination = "success"
                    break
        except Exception as exc:  # noqa: BLE001 — record failure path
            termination = "error"
            builder_extra_err = f"{exc} ({_WINDOW_C_FIX})"

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

        base_xy1, base_yaw1 = read_base_xy_yaw(inner)
        base_qpos1 = read_base_joint_qpos(inner)
        xy_delta = float(np.linalg.norm(base_xy1 - base_xy0))
        qpos_delta = 0.0
        for key, v0 in base_qpos0.items():
            v1 = float(base_qpos1.get(key, v0))
            qpos_delta = max(qpos_delta, abs(v1 - float(v0)))
        base_moved = bool(xy_delta > 1e-4 or qpos_delta > 1e-4)

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
            save_episode_video(frames, path=path, fps=control_freq)
            video_path = str(path)

        wall = time.perf_counter() - t0
        extra: dict[str, Any] = {
            "pnp_track": track,
            "env_class": handle.extra.get("env_class"),
            "camera": camera,
            "action_space": "hybrid_mobile_12d",
            "action_split_indexes": splits,
            "freeze_base": False,
            "success_source": f"{handle.extra.get('env_class')}._check_success",
            "base_xy_start": [float(base_xy0[0]), float(base_xy0[1])],
            "base_xy_end": [float(base_xy1[0]), float(base_xy1[1])],
            "base_xy_delta_m": xy_delta,
            "base_yaw_start": float(base_yaw0),
            "base_yaw_end": float(base_yaw1),
            "base_joint_qpos_start": {k: float(v) for k, v in base_qpos0.items()},
            "base_joint_qpos_end": {k: float(v) for k, v in base_qpos1.items()},
            "base_joint_qpos_peak_abs_delta": float(qpos_delta),
            "base_moved": base_moved,
            "ep_seed": int(ep_seed),
            "layout_id": (
                None if getattr(inner, "layout_id", None) is None else int(inner.layout_id)
            ),
            "skipped_lightwheel_props_missing_objs_lw": handle.extra.get(
                "skipped_lightwheel_props_missing_objs_lw"
            ),
            "gentex_tex_aliases": handle.extra.get("gentex_tex_aliases"),
        }
        if builder_extra_err is not None:
            extra["error"] = builder_extra_err
        extra["spawn_failed"] = bool(getattr(controller, "spawn_failed", False))
        fin = getattr(controller, "finalize_labels", None)
        if callable(fin):
            fin(success=success)
        extra.update(controller.extra_fields())

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


def _save_spawn_snapshot(
    env: Any,
    *,
    camera: str,
    cfg: RunConfig,
    handle: Any,
    level: int,
    episode_index: int,
    body_name: str | None = None,
) -> str | None:
    """Floor look-at of the actor. Do not use agentview (it looks at the stove)."""
    try:
        import imageio

        out_dir = episode_video_dir(
            results_dir=str(cfg.output.results_dir),
            backend=str(cfg.backend),
            suite=str(cfg.suite),
            level=level,
            run_id=str(cfg.run_id),
            task_id=handle.task_id,
            task_name=str(handle.task_name).lower().replace(" ", "_")[:40],
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        # Keep agentview for debugging only — never use it to judge floor spawn.
        agent_path = out_dir / f"l{level}_spawn_ep{episode_index}.png"
        img = np.asarray(env.sim.render(height=256, width=384, camera_name=camera))[::-1]
        imageio.imwrite(str(agent_path), img)
        floor_path = out_dir / f"l{level}_spawn_floor_ep{episode_index}.png"
        floor_img = _render_floor_lookat(env, body_name=body_name)
        if floor_img is not None:
            imageio.imwrite(str(floor_path), floor_img)
            return str(floor_path)
        return str(agent_path)
    except Exception:
        return None


def _render_floor_lookat(env: Any, *, body_name: str | None = None) -> np.ndarray | None:
    """Free camera aimed at the actor (or the base if still hidden)."""
    try:
        import mujoco

        from dynisafety.backends.robocasa.disturbance._support import (
            HIDDEN_POS,
            read_base_pose,
            read_body_pos,
            read_obstacle_pos,
        )

        pos = None
        if body_name:
            try:
                pos = np.asarray(read_body_pos(env, body_name), dtype=np.float64)
            except Exception:
                pos = None
        if pos is None:
            pos = np.asarray(read_obstacle_pos(env), dtype=np.float64)
        if float(np.linalg.norm(pos[:2] - HIDDEN_POS[:2])) < 1e-6 and pos[2] < -1.0:
            xy, _yaw = read_base_pose(env)
            lookat = np.array([float(xy[0]), float(xy[1]), 0.20], dtype=np.float64)
        else:
            lookat = np.array(
                [float(pos[0]), float(pos[1]), min(0.40, max(0.20, float(pos[2])))],
                dtype=np.float64,
            )
        model = getattr(env.sim.model, "_model", env.sim.model)
        data = getattr(env.sim.data, "_data", env.sim.data)
        renderer = mujoco.Renderer(model, height=384, width=512)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = 2.4
        cam.azimuth = 55.0
        cam.elevation = -40.0
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = np.asarray(renderer.render())
        renderer.close()
        return img
    except Exception:
        return None
