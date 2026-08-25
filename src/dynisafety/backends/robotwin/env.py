"""RoboTwin task environment bootstrap (no disturbance / metrics here)."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import yaml

from dynisafety.backends.robotwin.expert_cache import (
    cache_key,
    exclusive_key_lock,
    load_entry,
    store_entry,
)
from dynisafety.config.schema import RunConfig
from dynisafety.core.protocols import TaskSpec
from dynisafety.policies.base import Observation

logger = logging.getLogger(__name__)


@dataclass
class RoboTwinEnvHandle:
    task_env: Any
    task: TaskSpec
    instruction: str
    task_args: dict[str, Any]
    clear_cache_freq: int
    robotwin_root: Path
    episode_info: dict[str, Any] | None = None

    def close(self) -> None:
        close_env = getattr(self.task_env, "close_env", None)
        if callable(close_env):
            close_env()


def resolve_robotwin_root(
    *,
    workspace: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve RoboTwin checkout root.

    Priority: ``ROBOTWIN_ROOT`` env → first tree under workspace that contains
    ``task_config/`` (``third_party/RoboTwin`` then
    ``external/simulators/RoboTwin``).

    The submodule pin currently omits ``task_config``/``assets``, so in practice
    resolution falls through to the local simulator tree under ``external/``.
    """
    env_map = env if env is not None else os.environ
    if "ROBOTWIN_ROOT" in env_map:
        root = Path(env_map["ROBOTWIN_ROOT"])
        if root.is_dir():
            return root.resolve()
        raise FileNotFoundError(f"ROBOTWIN_ROOT is not a directory: {root}")

    if workspace is None:
        raise FileNotFoundError("RoboTwin root not found: set ROBOTWIN_ROOT or provide workspace")
    ws = workspace.resolve()
    candidates = (
        ws / "third_party" / "RoboTwin",
        ws / "external" / "simulators" / "RoboTwin",
    )
    for cand in candidates:
        if cand.is_dir() and (cand / "task_config").is_dir():
            return cand
    # Last resort: any existing dir (caller will fail later on missing files).
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"RoboTwin root not found under workspace {ws}: tried {candidates[0]} and {candidates[1]}"
    )


def ensure_robotwin_sys_path(robotwin_root: Path) -> None:
    """Insert RoboTwin import paths and ``chdir`` into ``robotwin_root``.

    Side effect: mutates ``sys.path`` and the process working directory (matches
    legacy ``main_inference.py`` bootstrap so cwd-relative asset paths resolve).
    """
    root = robotwin_root.resolve()
    for sub in (
        root,
        root / "policy",
        root / "description" / "utils",
    ):
        sub_str = str(sub)
        if sub_str not in sys.path:
            sys.path.insert(0, sub_str)
    os.chdir(str(root))


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        from omegaconf import OmegaConf

        container = OmegaConf.to_container(OmegaConf.load(text), resolve=True)
    except Exception:
        container = yaml.safe_load(text)
    if not isinstance(container, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return container


def _get_embodiment_config(robot_file: str) -> Any:
    robot_config_file = Path(robot_file) / "config.yml"
    try:
        from omegaconf import OmegaConf

        return OmegaConf.load(robot_config_file)
    except Exception:
        with open(robot_config_file, encoding="utf-8") as fh:
            return yaml.safe_load(fh)


def build_task_args(
    *,
    robotwin_root: Path,
    task_config: str,
    task_name: str,
) -> dict[str, Any]:
    """Build RoboTwin ``setup_demo`` kwargs from task + embodiment YAML."""
    task_cfg_file = robotwin_root / "task_config" / f"{task_config}.yml"
    if not task_cfg_file.is_file():
        raise FileNotFoundError(f"task config not found: {task_cfg_file}")
    task_args = _load_yaml_mapping(task_cfg_file)

    from envs import CONFIGS_PATH  # type: ignore  # noqa: WPS433 — RoboTwin layout

    embodiment_types = _load_yaml_mapping(Path(CONFIGS_PATH + "_embodiment_config.yml"))
    camera_cfg = _load_yaml_mapping(Path(CONFIGS_PATH + "_camera_config.yml"))

    def _emb_file(emb: str) -> str:
        rf = embodiment_types[emb]["file_path"]
        if rf is None:
            raise RuntimeError("No embodiment files found")
        return str(rf)

    embodiment_type = task_args["embodiment"]
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        robot_file = str(robotwin_root / _emb_file(embodiment_type[0]))
        task_args["left_robot_file"] = robot_file
        task_args["right_robot_file"] = robot_file
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = str(robotwin_root / _emb_file(embodiment_type[0]))
        task_args["right_robot_file"] = str(robotwin_root / _emb_file(embodiment_type[1]))
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError(
            f"Invalid embodiment type length: {len(embodiment_type)}, expected 1 or 3"
        )

    task_args["left_embodiment_config"] = _get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = _get_embodiment_config(task_args["right_robot_file"])
    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["eval_mode"] = True
    return task_args


def _class_decorator(task_name: str) -> Any:
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def make_robotwin_env(
    cfg: RunConfig,
    task: TaskSpec,
    *,
    workspace: Path,
    extra_robotwin: Mapping[str, Any],
) -> RoboTwinEnvHandle:
    """Instantiate a RoboTwin task env (no ``setup_demo`` / disturbance)."""
    del cfg
    if "task_config" not in extra_robotwin:
        raise ValueError("extra.robotwin.task_config")
    task_config = str(extra_robotwin["task_config"])
    robotwin_root = resolve_robotwin_root(workspace=workspace)
    ensure_robotwin_sys_path(robotwin_root)
    task_args = build_task_args(
        robotwin_root=robotwin_root,
        task_config=task_config,
        task_name=task.task_name,
    )
    task_env = _class_decorator(task.task_name)
    clear_cache_freq = int(task_args["clear_cache_freq"])
    instruction = task.instruction
    return RoboTwinEnvHandle(
        task_env=task_env,
        task=task,
        instruction=instruction,
        task_args=task_args,
        clear_cache_freq=clear_cache_freq,
        robotwin_root=robotwin_root,
    )


def _import_unstable_error() -> type[BaseException]:
    try:
        from envs.utils.create_actor import UnStableError  # type: ignore

        return UnStableError
    except Exception:
        return Exception


@dataclass(frozen=True)
class ExpertCheckOutcome:
    accepted_seed: int
    episode_info: dict[str, Any]
    cache_hit: bool


def expert_check_setup(
    handle: RoboTwinEnvHandle,
    *,
    seed: int,
    episode_index: int,
    enabled: bool,
    max_attempts: int,
    timeout_s: float,
    log_every: int,
    cache_dir: Path | None = None,
) -> ExpertCheckOutcome:
    """Run expert-check for one accepted episode seed.

    When ``cache_dir`` is set, a hit skips ``setup_demo`` / ``play_once`` /
    ``close_env``. The caller still ``setup_demo``s the accepted seed for the
    real policy episode.
    """
    if not enabled:
        return ExpertCheckOutcome(int(seed), {}, False)

    task_name = handle.task.task_name
    start_seed = int(seed)

    if cache_dir is None:
        return _expert_check_search(
            handle,
            start_seed=start_seed,
            episode_index=episode_index,
            max_attempts=max_attempts,
            timeout_s=timeout_s,
            log_every=log_every,
        )

    task_config = handle.task_args.get("task_config")
    if not isinstance(task_config, str) or not task_config.strip():
        raise ValueError("handle.task_args.task_config required for expert-check cache")

    key = cache_key(
        task_name=task_name,
        task_config=task_config,
        start_seed=start_seed,
        episode_index=int(episode_index),
    )
    with exclusive_key_lock(cache_dir, key):
        cached = load_entry(cache_dir, key)
        if cached is not None:
            if cached.accepted_seed < start_seed:
                logger.warning(
                    "Expert-check cache ignored | task=%s start=%s accepted=%s (accepted < start)",
                    task_name,
                    start_seed,
                    cached.accepted_seed,
                )
            else:
                logger.info(
                    "Expert-check cache hit | task=%s start=%s accepted=%s episode=%s",
                    task_name,
                    start_seed,
                    cached.accepted_seed,
                    episode_index,
                )
                handle.episode_info = dict(cached.episode_info)
                return ExpertCheckOutcome(
                    cached.accepted_seed, dict(cached.episode_info), True
                )
        outcome = _expert_check_search(
            handle,
            start_seed=start_seed,
            episode_index=episode_index,
            max_attempts=max_attempts,
            timeout_s=timeout_s,
            log_every=log_every,
        )
        try:
            store_entry(
                cache_dir,
                key,
                accepted_seed=outcome.accepted_seed,
                episode_info=outcome.episode_info,
                task_name=task_name,
                task_config=task_config,
                start_seed=start_seed,
                episode_index=int(episode_index),
            )
        except OSError as exc:
            logger.warning(
                "Expert-check cache store failed | task=%s key=%s err=%s",
                task_name,
                key,
                exc,
            )
        return outcome


def _expert_check_search(
    handle: RoboTwinEnvHandle,
    *,
    start_seed: int,
    episode_index: int,
    max_attempts: int,
    timeout_s: float,
    log_every: int,
) -> ExpertCheckOutcome:
    unstable_err = _import_unstable_error()
    task_env = handle.task_env
    task_name = handle.task.task_name
    now_seed = int(start_seed)
    attempts = 0
    start_ts = time.time()

    while True:
        attempts += 1
        elapsed_s = time.time() - start_ts
        if attempts > max_attempts:
            raise RuntimeError(
                f"Expert-check exceeded max attempts for {task_name}: "
                f"attempts={attempts}, current_seed={now_seed}"
            )
        if elapsed_s > timeout_s:
            raise TimeoutError(
                f"Expert-check exceeded timeout for {task_name}: "
                f"elapsed={elapsed_s:.1f}s, attempts={attempts}, current_seed={now_seed}"
            )
        if attempts == 1 or attempts % log_every == 0:
            logger.info(
                "Expert-check progress | task=%s seed=%s attempts=%s elapsed=%.1fs",
                task_name,
                now_seed,
                attempts,
                elapsed_s,
            )
        try:
            task_env.setup_demo(
                now_ep_num=episode_index,
                seed=now_seed,
                is_test=True,
                **handle.task_args,
            )
            episode_info = task_env.play_once()
            task_env.close_env()
        except (unstable_err, Exception) as exc:
            logger.warning(
                "Expert-check rejected seed=%s for task=%s due to %s: %s",
                now_seed,
                task_name,
                type(exc).__name__,
                exc,
            )
            close_env = getattr(task_env, "close_env", None)
            if callable(close_env):
                close_env()
            now_seed += 1
            continue

        plan_success = bool(getattr(task_env, "plan_success", False))
        expert_success = bool(task_env.check_success())
        logger.info(
            "Expert-check result | task=%s seed=%s plan_success=%s check_success=%s",
            task_name,
            now_seed,
            plan_success,
            expert_success,
        )
        if plan_success and expert_success:
            info_payload: dict[str, Any] = {}
            if isinstance(episode_info, Mapping):
                inner = episode_info.get("info")
                if isinstance(inner, Mapping):
                    info_payload = dict(inner)
                else:
                    info_payload = dict(episode_info)
            handle.episode_info = info_payload
            return ExpertCheckOutcome(now_seed, info_payload, False)

        logger.warning(
            "Expert-check rejected seed=%s for task=%s because plan_success=%s check_success=%s",
            now_seed,
            task_name,
            plan_success,
            expert_success,
        )
        now_seed += 1


def select_instruction(
    *,
    task_name: str,
    episode_info_list: list,
    test_num: int,
    instruction_type: str,
    prompt_mode: str,
    fallback: str,
) -> str:
    """Select natural-language instruction for the policy episode."""
    mode = str(os.environ.get("PROMPT_MODE", prompt_mode)).strip().lower()
    if mode == "default":
        mode = "task_name"
    if mode in {"official", "random"}:
        from generate_episode_descriptions import generate_episode_descriptions  # type: ignore

        results = generate_episode_descriptions(task_name, episode_info_list, test_num)
        return str(np.random.choice(results[0][instruction_type]))
    if mode in {"task_name", "short_name"}:
        return fallback if fallback.strip() else task_name.replace("_", " ")
    raise ValueError(
        f"Unsupported prompt_mode={mode!r}. Expected official, random, task_name, or default."
    )


def extract_head_rgb(raw_obs: Mapping[str, Any]) -> np.ndarray:
    observation = raw_obs.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("observation missing from RoboTwin get_obs() payload")
    head = observation.get("head_camera")
    if not isinstance(head, Mapping) or "rgb" not in head:
        raise ValueError("observation.head_camera.rgb missing")
    return np.asarray(head["rgb"])


def build_observation(
    handle: RoboTwinEnvHandle,
    *,
    instruction: str,
    step: int,
    needs_state: bool,
    raw_obs: Mapping[str, Any] | None = None,
) -> Observation:
    """Pack a policy ``Observation``. Reuse ``raw_obs`` when the caller already
    fetched it — ``get_obs()`` re-renders the three cameras.
    """
    payload = handle.task_env.get_obs() if raw_obs is None else raw_obs
    observation = payload.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("observation missing from RoboTwin get_obs() payload")

    images: dict[str, np.ndarray] = {
        "primary": np.asarray(observation["head_camera"]["rgb"]),
    }
    if "left_camera" in observation and "rgb" in observation["left_camera"]:
        images["left"] = np.asarray(observation["left_camera"]["rgb"])
    if "right_camera" in observation and "rgb" in observation["right_camera"]:
        images["right"] = np.asarray(observation["right_camera"]["rgb"])

    state: np.ndarray | None = None
    extra: dict[str, Any] = {}
    # Dual-arm EE proprio for official X-VLA (and similar); other adapters ignore.
    endpose = payload.get("endpose")
    if isinstance(endpose, Mapping):
        extra["endpose"] = {
            "left_endpose": np.asarray(endpose["left_endpose"], dtype=np.float64),
            "right_endpose": np.asarray(endpose["right_endpose"], dtype=np.float64),
            "left_gripper": float(endpose.get("left_gripper", 0.0)),
            "right_gripper": float(endpose.get("right_gripper", 0.0)),
        }
    if needs_state:
        # joint_action is a top-level key on get_obs(), not under observation{}
        joint_action = payload.get("joint_action")
        if isinstance(joint_action, Mapping) and "vector" in joint_action:
            state = np.asarray(joint_action["vector"], dtype=np.float64)
        else:
            raise ValueError(
                "get_obs() joint_action.vector missing for needs_state=True "
                "(ensure task_args data_type.qpos=true)"
            )

    return Observation(
        images=images,
        state=state,
        instruction=instruction,
        step=step,
        extra=extra,
    )


def save_episode_video(frames: list[np.ndarray], *, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=int(fps))


def apply_qpos_action(handle: RoboTwinEnvHandle, action: np.ndarray) -> None:
    handle.task_env.take_action(action, action_type="qpos")


def apply_ee_action(handle: RoboTwinEnvHandle, action: np.ndarray) -> None:
    handle.task_env.take_action(action, action_type="ee")


__all__ = [
    "ExpertCheckOutcome",
    "RoboTwinEnvHandle",
    "apply_ee_action",
    "apply_qpos_action",
    "build_observation",
    "build_task_args",
    "ensure_robotwin_sys_path",
    "expert_check_setup",
    "extract_head_rgb",
    "make_robotwin_env",
    "resolve_robotwin_root",
    "save_episode_video",
    "select_instruction",
]
