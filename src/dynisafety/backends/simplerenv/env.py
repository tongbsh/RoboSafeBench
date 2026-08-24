"""SimplerEnv / ManiSkill2 environment helpers (no disturbance / metrics)."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.protocols import TaskSpec

_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_ASSET_DIR = _PACKAGE_DIR / "assets"


@dataclass
class SimplerEnvHandle:
    """Concrete ``EnvHandle`` for ManiSkill2 / SimplerEnv episodes."""

    env: Any
    base_env: Any
    task: TaskSpec
    instruction: str
    control_mode: str
    task_extra: dict[str, Any] = field(default_factory=dict)
    simplerenv_root: Path | None = None
    camera_name: str | None = None

    def close(self) -> None:
        env = self.env
        close = getattr(env, "close", None)
        if callable(close):
            close()


def resolve_simplerenv_root(
    *,
    workspace: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve SimplerEnv-OpenVLA checkout.

    Priority: ``SIMPLERENV_ROOT`` → ``{workspace}/third_party/SimplerEnv-OpenVLA``
    → legacy ``{workspace}/dynamic_SimplerEnv/SimplerEnv-OpenVLA``.
    """
    env_map = env if env is not None else os.environ
    if "SIMPLERENV_ROOT" in env_map:
        root = Path(env_map["SIMPLERENV_ROOT"])
        if root.is_dir():
            return root.resolve()
        raise FileNotFoundError(f"SIMPLERENV_ROOT is not a directory: {root}")

    if workspace is None:
        raise FileNotFoundError(
            "SimplerEnv root not found: set SIMPLERENV_ROOT or provide workspace"
        )
    ws = workspace.resolve()
    candidates = (
        ws / "third_party" / "SimplerEnv-OpenVLA",
        ws / "dynamic_SimplerEnv" / "SimplerEnv-OpenVLA",
    )
    for cand in candidates:
        if cand.is_dir() and (cand / "simpler_env").is_dir():
            return cand.resolve()
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        f"SimplerEnv root not found under workspace {ws}: tried {candidates[0]} and {candidates[1]}"
    )


def ensure_simplerenv_sys_path(simplerenv_root: Path) -> None:
    """Insert SimplerEnv + ManiSkill2 import paths (idempotent)."""
    root = simplerenv_root.resolve()
    for sub in (root, root / "ManiSkill2_real2sim"):
        sub_str = str(sub)
        if sub_str not in sys.path:
            sys.path.insert(0, sub_str)


def default_asset_dir() -> Path:
    if not _DEFAULT_ASSET_DIR.is_dir():
        raise FileNotFoundError(f"SimplerEnv backend assets missing: {_DEFAULT_ASSET_DIR}")
    return _DEFAULT_ASSET_DIR


def models_dir_for_root(simplerenv_root: Path) -> Path:
    path = simplerenv_root.resolve() / "ManiSkill2_real2sim" / "data" / "custom" / "models"
    if not path.is_dir():
        raise FileNotFoundError(f"obstacle models directory missing: {path}")
    return path


def load_task_extra(task: TaskSpec, *, workspace: Path) -> dict[str, Any]:
    path = workspace / "configs" / "backends" / "simplerenv" / "tasks" / f"{task.task_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"task YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    extra = data.get("extra")
    if not isinstance(extra, dict):
        raise ValueError(f"task {task.task_name!r} YAML must define mapping extra:")
    return dict(extra)


def _require_task_key(extra: Mapping[str, Any], key: str) -> Any:
    if key not in extra:
        raise ValueError(
            f"task extra.{key} missing; set it in configs/backends/simplerenv/tasks/*.yaml"
        )
    return extra[key]


def make_simplerenv_env(
    cfg: RunConfig,
    task: TaskSpec,
    *,
    workspace: Path,
    episode_index: int = 0,
    resolved_extra: Mapping[str, Any] | None = None,
) -> SimplerEnvHandle:
    """Build ManiSkill2 env; level∈{1,2,3} wraps with ``DyniSafetyWrapper``.

    Level 0 returns the bare env (no wrapper; iron law #16).
    """
    del episode_index  # reserved for callers that rebuild per episode
    level = int(cfg.level) if cfg.level is not MISSING else -1
    if level not in (0, 1, 2, 3):
        raise ValueError(f"unsupported level={level}")

    runtime = cfg.runtime
    if runtime is MISSING:
        raise ValueError("cfg.runtime is MISSING")
    control_freq = runtime.control_freq
    sim_freq = runtime.sim_freq
    max_steps = cfg.max_steps
    if control_freq is MISSING or sim_freq is MISSING or max_steps is MISSING:
        raise ValueError("runtime.control_freq, runtime.sim_freq, and max_steps required (YAML)")

    simplerenv_root = resolve_simplerenv_root(workspace=workspace)
    ensure_simplerenv_sys_path(simplerenv_root)

    from simpler_env.utils.env.env_builder import (
        build_maniskill2_env,
        get_robot_control_mode,
    )

    task_extra = load_task_extra(task, workspace=workspace)
    env_name = str(_require_task_key(task_extra, "env_name"))
    scene_name = str(_require_task_key(task_extra, "scene_name"))
    robot = str(_require_task_key(task_extra, "robot"))
    overlay_rel = str(_require_task_key(task_extra, "rgb_overlay"))
    rgb_overlay_path = str((simplerenv_root / overlay_rel).resolve())
    if not Path(rgb_overlay_path).is_file():
        raise FileNotFoundError(f"rgb_overlay not found: {rgb_overlay_path}")

    control_mode = get_robot_control_mode(robot, None)
    kwargs: dict[str, Any] = dict(
        obs_mode="rgbd",
        robot=robot,
        sim_freq=int(sim_freq),
        control_mode=control_mode,
        control_freq=int(control_freq),
        max_episode_steps=int(max_steps),
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    base_env = build_maniskill2_env(env_name, **kwargs)

    instruction = task.instruction
    camera_name = task_extra.get("obs_camera_name")
    if camera_name is not None:
        camera_name = str(camera_name)

    if level == 0:
        env: Any = base_env
    else:
        from dynisafety.backends.simplerenv.wrapper import DyniSafetyWrapper

        env = DyniSafetyWrapper(
            base_env,
            level=level,
            run_cfg=cfg,
            episode_index=0,
            seed=int(cfg.seed) if cfg.seed is not MISSING else 0,
            task_id=int(task.task_id),
            simplerenv_root=simplerenv_root,
            workspace=workspace,
            resolved_extra=dict(resolved_extra or {}),
            task_extra=task_extra,
        )

    return SimplerEnvHandle(
        env=env,
        base_env=base_env,
        task=task,
        instruction=instruction,
        control_mode=control_mode,
        task_extra=task_extra,
        simplerenv_root=simplerenv_root,
        camera_name=camera_name,
    )


def build_reset_options(
    cfg: RunConfig,
    task_extra: Mapping[str, Any],
    *,
    episode_index: int,
) -> dict[str, Any]:
    """robot_init_* + obj_variation_mode=episode obj_episode_id."""
    rx = _require_task_key(task_extra, "robot_init_x")
    ry = _require_task_key(task_extra, "robot_init_y")
    quat = _require_task_key(task_extra, "robot_init_quat")
    quat_arr = np.asarray(quat, dtype=np.float64).reshape(4)

    runtime = cfg.runtime
    if runtime is MISSING:
        raise ValueError("cfg.runtime is MISSING")
    ep_range = runtime.obj_episode_range
    if ep_range is MISSING or ep_range is None:
        raise ValueError("runtime.obj_episode_range required (YAML)")
    if len(ep_range) != 2:
        raise ValueError(f"runtime.obj_episode_range must be [lo, hi), got {ep_range!r}")
    lo, hi = int(ep_range[0]), int(ep_range[1])
    if hi <= lo:
        raise ValueError(f"obj_episode_range hi must be > lo, got [{lo}, {hi})")
    span = hi - lo
    obj_episode_id = lo + (int(episode_index) % span)

    return {
        "robot_init_options": {
            "init_xy": np.array([float(rx), float(ry)], dtype=np.float64),
            "init_rot_quat": quat_arr,
        },
        "obj_init_options": {
            "episode_id": int(obj_episode_id),
        },
    }


def get_primary_image(
    env: Any,
    obs: Mapping[str, Any],
    *,
    camera_name: str | None,
) -> np.ndarray:
    """Wrap ``get_image_from_maniskill2_obs_dict`` for Observation.images['primary']."""
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    return np.asarray(get_image_from_maniskill2_obs_dict(env, obs, camera_name=camera_name))


def get_eef_pos(env: Any) -> np.ndarray:
    """EEF world position from ManiSkill2 agent (or obs fallback)."""
    inner = getattr(env, "unwrapped", env)
    try:
        return np.asarray(inner.agent.ee_link.get_pose().p, dtype=np.float64)
    except Exception:
        pass
    return np.zeros(3, dtype=np.float64)


def get_eef_state_7d(obs: Mapping[str, Any]) -> np.ndarray | None:
    """Prefer ``obs['agent']['eef_pos']`` (xyz + wxyz) when present.

    ManiSkill WidowX ``eef_pos`` is typically 8D (xyz + wxyz + gripper); this
    returns the leading 7D pose used by generic EE conversion. Full 8D is
    available via :func:`get_eef_pos_8d`.
    """
    agent = obs.get("agent")
    if not isinstance(agent, Mapping) or "eef_pos" not in agent:
        return None
    arr = np.asarray(agent["eef_pos"], dtype=np.float64).ravel()
    if arr.size >= 7:
        return arr[:7].copy()
    if arr.size == 3:
        return np.concatenate([arr, np.zeros(4, dtype=np.float64)])
    return None


def get_eef_pos_8d(obs: Mapping[str, Any]) -> np.ndarray | None:
    """Return ManiSkill ``agent.eef_pos`` when it has the full 8D WidowX layout.

    Layout: ``[xyz(3), quat_wxyz(4), gripper_open(1)]`` (see ManiSkill2
    ``base_agent.py``). Returns ``None`` if missing or too short.
    """
    agent = obs.get("agent")
    if not isinstance(agent, Mapping) or "eef_pos" not in agent:
        return None
    arr = np.asarray(agent["eef_pos"], dtype=np.float64).ravel()
    if arr.size < 8:
        return None
    return arr[:8].copy()


def save_episode_video(
    frames: Sequence[np.ndarray],
    *,
    path: Path,
    fps: int,
) -> None:
    """Write mp4; fps from cfg/YAML (no module-level DATE_TIME)."""
    if not frames:
        return
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:  # pragma: no cover
        import imageio  # type: ignore

    imageio.mimsave(str(path), list(frames), fps=int(fps))
