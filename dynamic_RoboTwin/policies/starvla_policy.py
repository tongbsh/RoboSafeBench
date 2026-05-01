"""StarVLA websocket-client policy adapter for DyniSafety-RoboTwin.

This adapter mirrors the inference contract used by
``vlaModelHome/starVLA/examples/Robotwin/eval_files/model2robotwin_interface.py``
while fitting the :class:`~policies.base_policy.RoboTwinPolicy` interface used
by ``dynamic_RoboTwin/main_inference.py``.

The StarVLA model itself runs in a separate ``starVLA`` conda env via
``deployment/model_server/server_policy.py``. The RoboTwin side stays in the
existing evaluation env and only talks to the remote server over websocket.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping

import cv2 as cv
import numpy as np
import yaml

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
_STARVLA_ROOT = Path(
    os.environ.get("STARVLA_ROOT", str(_WORKSPACE_ROOT / "vlaModelHome" / "starVLA"))
).resolve()


def _ensure_starvla_on_path() -> Path:
    root_str = str(_STARVLA_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return _STARVLA_ROOT


def _make_websocket_client(host: str, port: int):
    _ensure_starvla_on_path()
    from deployment.model_server.tools.websocket_policy_client import (  # noqa: PLC0415
        WebsocketClientPolicy,
    )

    return WebsocketClientPolicy(host=host, port=port)


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_image_size(value: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return fallback
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, str):
        pieces = [part.strip() for part in value.split(",") if part.strip()]
        if len(pieces) == 1:
            size = int(pieces[0])
            return (size, size)
        if len(pieces) >= 2:
            return (int(pieces[0]), int(pieces[1]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    return fallback


def _load_run_artifacts(ckpt_file: Path) -> tuple[dict, dict, Path]:
    ckpt_file = ckpt_file.expanduser().resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"StarVLA checkpoint not found: {ckpt_file}")
    if ckpt_file.parent.name != "checkpoints":
        raise ValueError(
            "StarVLA ckpt_path must point to a file under `checkpoints/`, "
            f"got parent directory {ckpt_file.parent.name!r}"
        )

    run_dir = ckpt_file.parents[1]
    config_path = run_dir / "config.yaml"
    stats_path = run_dir / "dataset_statistics.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.yaml: {config_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing dataset_statistics.json: {stats_path}")

    with config_path.open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with stats_path.open("r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    return model_cfg, norm_stats, run_dir


def _check_unnorm_key(norm_stats: dict, unnorm_key: str | None) -> str:
    available_keys = sorted(norm_stats.keys())
    if unnorm_key is None:
        if len(available_keys) == 1:
            return available_keys[0]
        raise ValueError(
            "`unnorm_key` must be provided when multiple normalization statistics are available. "
            f"Available keys: {available_keys}"
        )
    if unnorm_key not in norm_stats:
        raise KeyError(
            f"Unknown `unnorm_key`: {unnorm_key!r}. Available keys: {available_keys}"
        )
    return unnorm_key


def _resolve_action_stats(
    norm_stats: dict,
    unnorm_key: str,
    action_mode: str,
) -> dict:
    stats = norm_stats[unnorm_key]
    if action_mode in stats:
        mode_stats = stats[action_mode]
        return mode_stats.get("action", mode_stats)
    if "action" in stats:
        if action_mode != "abs":
            raise ValueError(
                f"Statistics key {unnorm_key!r} only provides abs action stats, "
                f"but action_mode={action_mode!r} was requested."
            )
        return stats["action"]
    raise ValueError(
        f"Invalid statistics format for key {unnorm_key!r}. "
        f"Available keys: {sorted(stats.keys())}"
    )


def _get_normalization_bounds(
    norm_stats: Dict[str, Any],
    normalization_mode: str = "min_max",
) -> tuple[np.ndarray, np.ndarray]:
    if normalization_mode == "q99":
        if "q01" not in norm_stats or "q99" not in norm_stats:
            raise KeyError("Normalization mode `q99` requires `q01` and `q99`.")
        return np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    if normalization_mode == "min_max":
        if "min" not in norm_stats or "max" not in norm_stats:
            raise KeyError("Normalization mode `min_max` requires `min` and `max`.")
        return np.array(norm_stats["max"]), np.array(norm_stats["min"])
    raise ValueError(
        f"Unsupported normalization_mode: {normalization_mode!r}. "
        "Expected one of ['min_max', 'q99']."
    )


def _unnormalize_actions(
    normalized_actions: np.ndarray,
    action_norm_stats: Dict[str, Any],
    normalization_mode: str,
) -> np.ndarray:
    action_high, action_low = _get_normalization_bounds(
        action_norm_stats,
        normalization_mode=normalization_mode,
    )
    mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
    normalized_actions = np.clip(normalized_actions, -1, 1)
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )


class _AdaptiveEnsembler:
    def __init__(self, pred_action_horizon: int, adaptive_ensemble_alpha: float = 0.0) -> None:
        self.pred_action_horizon = pred_action_horizon
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_history: deque[np.ndarray] = deque(maxlen=self.pred_action_horizon)

    def reset(self) -> None:
        self.action_history.clear()

    def ensemble_action(self, cur_action: np.ndarray) -> np.ndarray:
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1),
                        self.action_history,
                    )
                ]
            )

        ref = curr_act_preds[num_actions - 1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * curr_act_preds, axis=0)


@register_policy("starvla")
class StarVLAPolicy(RoboTwinPolicy):
    """StarVLA policy adapter using the official websocket inference server."""

    name = "starvla"
    action_type = "qpos"

    def __init__(
        self,
        client: Any,
        *,
        ckpt_path: str,
        policy_setup: str,
        action_mode: str,
        normalization_mode: str,
        action_chunk_size: int,
        image_size: tuple[int, int],
        use_ddim: bool,
        num_ddim_steps: int,
        action_norm_stats: Dict[str, Any],
        action_ensemble: bool,
        action_ensemble_horizon: int,
        adaptive_ensemble_alpha: float,
    ) -> None:
        self._client = client
        self._ckpt_path = ckpt_path
        self._policy_setup = policy_setup
        self._action_mode = action_mode
        self._normalization_mode = normalization_mode
        self._action_chunk_size = int(action_chunk_size)
        self._image_size = tuple(int(v) for v in image_size)
        self._use_ddim = bool(use_ddim)
        self._num_ddim_steps = int(num_ddim_steps)
        self._action_norm_stats = action_norm_stats

        self._action_ensemble_enabled = bool(action_ensemble)
        self._action_ensembler = (
            _AdaptiveEnsembler(action_ensemble_horizon, adaptive_ensemble_alpha)
            if action_ensemble
            else None
        )

        self._instruction: str = ""
        self._initial_state: np.ndarray | None = None
        self._prev_action: np.ndarray | None = None
        self._raw_actions: np.ndarray | None = None

    @classmethod
    def load(cls, **kwargs: Any) -> "StarVLAPolicy":
        host = str(kwargs.pop("host", os.environ.get("STARVLA_SERVER_HOST", "127.0.0.1")))
        port = int(kwargs.pop("port", os.environ.get("STARVLA_SERVER_PORT", 10093)))
        ckpt_path = str(
            kwargs.pop("ckpt_path", os.environ.get("STARVLA_CHECKPOINT", "")).strip()
        )
        if not ckpt_path:
            raise ValueError(
                "StarVLA policy requires `ckpt_path` (a checkpoint file under run_dir/checkpoints/)."
            )

        use_ddim = _parse_bool(
            kwargs.pop("use_ddim", os.environ.get("STARVLA_USE_DDIM")),
            True,
        )
        num_ddim_steps = int(
            kwargs.pop("num_ddim_steps", os.environ.get("STARVLA_NUM_DDIM_STEPS", 10))
        )
        policy_setup = str(
            kwargs.pop("policy_setup", os.environ.get("STARVLA_POLICY_SETUP", "robotwin"))
        )

        model_cfg, norm_stats, _ = _load_run_artifacts(Path(ckpt_path))
        env_unnorm = _optional_str(os.environ.get("STARVLA_UNNORM_KEY"))
        unnorm_key = _optional_str(kwargs.pop("unnorm_key", None)) or env_unnorm
        resolved_unnorm_key = _check_unnorm_key(norm_stats, unnorm_key)

        cfg_action_mode = (
            model_cfg.get("datasets", {}).get("vla_data", {}).get("action_mode", "abs")
        )
        action_mode = str(
            kwargs.pop("action_mode", os.environ.get("STARVLA_ACTION_MODE", cfg_action_mode))
        )
        normalization_mode = str(
            kwargs.pop(
                "normalization_mode",
                kwargs.pop(
                    "action_normalization_mode",
                    os.environ.get("STARVLA_NORMALIZATION_MODE", "min_max"),
                ),
            )
        )

        cfg_image_size = (
            model_cfg.get("datasets", {}).get("vla_data", {}).get("image_size")
            or model_cfg.get("datasets", {}).get("vla_data", {}).get("default_image_resolution")
            or [224, 224]
        )
        image_size = _parse_image_size(
            kwargs.pop("image_size", os.environ.get("STARVLA_IMAGE_SIZE")),
            fallback=_parse_image_size(cfg_image_size, (224, 224)),
        )

        action_ensemble = _parse_bool(
            kwargs.pop("action_ensemble", os.environ.get("STARVLA_ACTION_ENSEMBLE")),
            False,
        )
        action_ensemble_horizon = int(
            kwargs.pop(
                "action_ensemble_horizon",
                os.environ.get("STARVLA_ACTION_ENSEMBLE_HORIZON", 3),
            )
        )
        adaptive_ensemble_alpha = float(
            kwargs.pop(
                "adaptive_ensemble_alpha",
                os.environ.get("STARVLA_ADAPTIVE_ENSEMBLE_ALPHA", 0.1),
            )
        )

        action_chunk_size = (
            int(model_cfg["framework"]["action_model"]["future_action_window_size"]) + 1
        )
        action_norm_stats = _resolve_action_stats(
            norm_stats,
            resolved_unnorm_key,
            action_mode,
        )
        client = _make_websocket_client(host=host, port=port)

        logger.info(
            "Loaded StarVLA artifacts from %s (unnorm_key=%s, action_mode=%s, chunk=%s)",
            ckpt_path,
            resolved_unnorm_key,
            action_mode,
            action_chunk_size,
        )

        return cls(
            client=client,
            ckpt_path=ckpt_path,
            policy_setup=policy_setup,
            action_mode=action_mode,
            normalization_mode=normalization_mode,
            action_chunk_size=action_chunk_size,
            image_size=image_size,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            action_norm_stats=action_norm_stats,
            action_ensemble=action_ensemble,
            action_ensemble_horizon=action_ensemble_horizon,
            adaptive_ensemble_alpha=adaptive_ensemble_alpha,
        )

    def reset(
        self,
        instruction: str,
        task_env: Any,
        initial_obs: Mapping[str, Any],
    ) -> None:
        del task_env, initial_obs
        self._instruction = str(instruction)
        self._initial_state = None
        self._prev_action = None
        self._raw_actions = None
        if self._action_ensembler is not None:
            self._action_ensembler.reset()

    def step(self, task_env: Any, observation: Mapping[str, Any]) -> None:
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        if self._action_mode in {"delta", "rel"} and self._initial_state is None:
            self._initial_state = state.copy()

        images = [
            self._resize_image(observation["observation"]["head_camera"]["rgb"]),
            self._resize_image(observation["observation"]["left_camera"]["rgb"]),
            self._resize_image(observation["observation"]["right_camera"]["rgb"]),
        ]

        example = {
            "lang": self._instruction,
            "image": images,
            "state": state,
        }
        vla_input = {
            "examples": [{
                "lang": example["lang"],
                "image": example["image"],
            }],
            "do_sample": False,
            "use_ddim": self._use_ddim,
            "num_ddim_steps": self._num_ddim_steps,
        }

        step_idx = int(task_env.take_action_cnt)
        if step_idx % self._action_chunk_size == 0 or self._raw_actions is None:
            response = self._client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"][0]
            except KeyError as exc:
                raise KeyError(
                    "StarVLA server response is missing `data.normalized_actions`."
                ) from exc

            raw_actions = _unnormalize_actions(
                normalized_actions=np.asarray(normalized_actions),
                action_norm_stats=self._action_norm_stats,
                normalization_mode=self._normalization_mode,
            )
            if self._action_mode == "delta":
                self._raw_actions = self._delta_to_absolute(raw_actions)
            elif self._action_mode == "rel":
                self._raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self._raw_actions = raw_actions

        action_idx = step_idx % self._action_chunk_size
        if self._raw_actions is None or action_idx >= len(self._raw_actions):
            raise RuntimeError(
                f"StarVLA action buffer underflow: step={step_idx}, "
                f"chunk={self._action_chunk_size}, raw_actions={None if self._raw_actions is None else len(self._raw_actions)}"
            )

        current_action = np.asarray(self._raw_actions[action_idx]).copy()
        if self._action_ensembler is not None:
            current_action = self._action_ensembler.ensemble_action(current_action)
        if self._action_mode == "delta":
            self._prev_action = current_action.copy()

        # Match StarVLA Robotwin client ordering:
        # [left_arm(6), left_gripper, right_arm(6), right_gripper]
        current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        task_env.take_action(current_action, action_type="qpos")

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()

    def _delta_to_absolute(self, delta_actions: np.ndarray) -> np.ndarray:
        if self._initial_state is None:
            raise RuntimeError("StarVLA delta mode requires an initial joint state.")
        abs_actions = np.zeros_like(delta_actions)
        mask = self._action_norm_stats.get(
            "mask",
            np.ones(delta_actions.shape[-1], dtype=bool),
        )
        base = self._prev_action if self._prev_action is not None else self._initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = np.where(mask, delta_actions[i] + base, delta_actions[i])
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        if self._initial_state is None:
            raise RuntimeError("StarVLA rel mode requires an initial joint state.")
        abs_actions = np.zeros_like(rel_actions)
        mask = self._action_norm_stats.get(
            "mask",
            np.ones(rel_actions.shape[-1], dtype=bool),
        )
        for i in range(len(rel_actions)):
            abs_actions[i] = np.where(mask, rel_actions[i] + self._initial_state, rel_actions[i])
        return abs_actions

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, self._image_size, interpolation=cv.INTER_AREA)


__all__ = ["StarVLAPolicy"]
