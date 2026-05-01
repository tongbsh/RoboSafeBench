"""FastWAM local Torch policy adapter for DyniSafety-RoboTwin.

The official FastWAM RobotWin/RoboTwin evaluation code exposes a RoboTwin
policy module with ``get_model`` / ``eval`` / ``reset_model`` functions.  This
adapter wraps that implementation in the ``RoboTwinPolicy`` interface used by
``dynamic_RoboTwin/main_inference.py`` so the same DyniSafety level0-3 loop can
drive FastWAM without copying the model-specific preprocessing code.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from policies import register_policy
from policies.base_policy import RoboTwinPolicy

logger = logging.getLogger(__name__)


def _default_workspace_root(this_file: Path) -> Path:
    p = this_file.resolve()
    for ancestor in p.parents:
        if (ancestor / "vlaModelHome").is_dir():
            return ancestor
    return p.parents[2]


_THIS_FILE = Path(__file__).resolve()
_WORKSPACE_ROOT = _default_workspace_root(_THIS_FILE)
_DEFAULT_FASTWAM_ROOT = _WORKSPACE_ROOT / "vlaModelHome" / "FastWAM"
_DEFAULT_FASTWAM_CKPT = (
    _WORKSPACE_ROOT
    / "download_checkpoints"
    / "fastwam"
    / "robotwin"
    / "robotwin_uncond_3cam_384.pt"
)
_DEFAULT_FASTWAM_DATASET_STATS = (
    _WORKSPACE_ROOT
    / "download_checkpoints"
    / "fastwam"
    / "robotwin"
    / "robotwin_uncond_3cam_384_dataset_stats.json"
)


def _resolve_path(value: Any, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default.expanduser().resolve()
    return Path(str(value)).expanduser().resolve()


def _require_file(path: Path, label: str, hint: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}. {hint}")
    return path


def _require_dir(path: Path, label: str, hint: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}. {hint}")
    return path


def _ensure_fastwam_on_path(fastwam_root: Path) -> None:
    for p in (fastwam_root, fastwam_root / "src"):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def _load_deploy_policy_module(fastwam_root: Path):
    """Load FastWAM's official RobotWin deploy_policy.py under a private name."""
    _ensure_fastwam_on_path(fastwam_root)
    deploy_path = fastwam_root / "experiments" / "robotwin" / "fastwam_policy" / "deploy_policy.py"
    if not deploy_path.is_file():
        raise FileNotFoundError(f"FastWAM deploy_policy.py not found: {deploy_path}")

    module_name = "_dynamic_robotwin_fastwam_deploy_policy"
    cached = sys.modules.get(module_name)
    if cached is not None and Path(getattr(cached, "__file__", "")).resolve() == deploy_path:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, deploy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FastWAM deploy policy from {deploy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _pop_first(kwargs: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs.pop(name)
    return default


@register_policy("fastwam")
class FastWAMPolicy(RoboTwinPolicy):
    """Adapter around FastWAM's official local RobotWin policy."""

    name = "fastwam"
    action_type = "qpos"

    def __init__(self, model: Any) -> None:
        self._model = model
        self._instruction = ""

    @classmethod
    def load(cls, **kwargs: Any) -> "FastWAMPolicy":
        kwargs = dict(kwargs)

        fastwam_root = _require_dir(
            _resolve_path(
                _pop_first(kwargs, "fastwam_root", default=os.environ.get("FASTWAM_ROOT")),
                _DEFAULT_FASTWAM_ROOT,
            ),
            "FastWAM root",
            "Set FASTWAM_ROOT or pass fastwam_root in --policy_kwargs.",
        )
        ckpt_path = _require_file(
            _resolve_path(
                _pop_first(
                    kwargs,
                    "ckpt_setting",
                    "ckpt_path",
                    "checkpoint_path",
                    default=os.environ.get("FASTWAM_CKPT"),
                ),
                _DEFAULT_FASTWAM_CKPT,
            ),
            "FastWAM checkpoint",
            "Set FASTWAM_CKPT or pass ckpt_setting in --policy_kwargs.",
        )
        dataset_stats_path = _require_file(
            _resolve_path(
                _pop_first(
                    kwargs,
                    "dataset_stats_path",
                    default=os.environ.get("FASTWAM_DATASET_STATS"),
                ),
                _DEFAULT_FASTWAM_DATASET_STATS,
            ),
            "FastWAM dataset stats",
            "Set FASTWAM_DATASET_STATS or pass dataset_stats_path in --policy_kwargs.",
        )

        usr_args: dict[str, Any] = dict(kwargs)
        usr_args["ckpt_setting"] = str(ckpt_path)
        usr_args["dataset_stats_path"] = str(dataset_stats_path)
        usr_args.setdefault("sim_cfg_path", str(fastwam_root / "configs" / "sim_robotwin.yaml"))
        usr_args.setdefault("sim_cfg_name", "sim_robotwin.yaml")
        usr_args.setdefault("mixed_precision", "bf16")
        usr_args.setdefault("device", "cuda")
        usr_args.setdefault("replan_steps", 24)
        usr_args.setdefault("num_inference_steps", 10)
        usr_args.setdefault("text_cfg_scale", 1.0)
        usr_args.setdefault("negative_prompt", "")
        usr_args.setdefault("rand_device", "cpu")
        usr_args.setdefault("tiled", False)
        usr_args.setdefault("timing_enabled", False)

        deploy_policy = _load_deploy_policy_module(fastwam_root)
        logger.info("Loading FastWAM policy from %s", fastwam_root)
        model = deploy_policy.get_model(usr_args)
        return cls(model=model)

    def reset(
        self,
        instruction: str,
        task_env: Any,
        initial_obs: Mapping[str, Any],
    ) -> None:
        del task_env, initial_obs
        self._instruction = instruction
        self._model.reset()

    def step(self, task_env: Any, observation: Mapping[str, Any]) -> None:
        # Reuse the official FastWAM step implementation; it handles action
        # chunking and dispatches one qpos action to the RoboTwin env.
        obs = observation if self._model.should_request_observation() else None
        self._model.step(task_env, obs)

    def close(self) -> None:
        del self._model


__all__ = ["FastWAMPolicy"]
