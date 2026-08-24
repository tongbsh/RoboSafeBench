"""InternVLA-M1 WebSocket adapter (SimplerEnv).

Wire contract mirrors
``third_party/SimplerEnv-OpenVLA/simpler_env/policies/internvla_m1/internvla_m1_model.py``:
``WebsocketClientPolicy.infer`` with ``batch_images`` / ``instructions`` payload,
client-side ``q01``/``q99`` de-normalisation, AdaptiveEnsembler, euler→axangle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import yaml

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    connect_transport,
    require_image,
)
from dynisafety.policies.adapters.starvla import (
    _AdaptiveEnsembler,
    _euler_rpy_to_axangle,
)
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)
from dynisafety.policies.transports import Transport

logger = logging.getLogger(__name__)


class InternVLAM1Adapter:
    """InternVLA-M1 ``server_policy_M1`` client (StarVLA-family codec)."""

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport

        model_cfg, norm_stats, run_dir = _load_run_artifacts(Path(cfg.checkpoint))
        raw_un = cfg.extra.get("unnorm_key")
        unnorm_key = None if raw_un is None or str(raw_un).strip() == "" else str(raw_un)
        self._unnorm_key = _check_unnorm_key(norm_stats, unnorm_key)
        self._action_norm_stats = norm_stats[self._unnorm_key]["action"]
        if "q01" not in self._action_norm_stats or "q99" not in self._action_norm_stats:
            raise KeyError(
                "InternVLA-M1 action stats must contain 'q01'/'q99'; got: "
                f"{list(self._action_norm_stats.keys())}"
            )

        fam = model_cfg["framework"]["action_model"]
        self._server_chunk = int(fam["future_action_window_size"]) + 1
        self._image_size = _image_size_from_cfg(model_cfg)

        ens_flag = cfg.extra.get("action_ensemble", True)
        if isinstance(ens_flag, str):
            ens_on = ens_flag.strip().lower() in ("1", "true", "yes", "on")
        else:
            ens_on = bool(ens_flag) if ens_flag is not None else True
        self._ensembler: _AdaptiveEnsembler | None = None
        if ens_on:
            horizon = int(cfg.extra.get("action_ensemble_horizon") or 7)
            alpha = float(cfg.extra.get("adaptive_ensemble_alpha") or 0.1)
            self._ensembler = _AdaptiveEnsembler(horizon, alpha)

        logger.info(
            "InternVLAM1Adapter: run_dir=%s unnorm_key=%s chunk=%d image=%s ensemble=%s",
            run_dir,
            self._unnorm_key,
            self._server_chunk,
            self._image_size,
            "off"
            if self._ensembler is None
            else f"on(h={self._ensembler.pred_action_horizon})",
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> InternVLAM1Adapter:
        if cfg.transport != "websocket":
            raise ValueError("InternVLAM1Adapter requires transport='websocket'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        if self._ensembler is not None:
            self._ensembler.reset()
        # InternVLA-M1 client.reset(instruction) clears server-side episode state.
        self._transport.reset({"instruction": ctx.instruction})

    def act(self, obs: Observation) -> ActionChunk:
        primary = _resize_uint8_hwc(require_image(obs, "primary"), self._image_size)
        payload: dict[str, Any] = {
            "batch_images": [[primary]],
            "instructions": [obs.instruction],
            "unnorm_key": self._unnorm_key,
            "do_sample": False,
        }
        for key in ("use_ddim", "num_ddim_steps", "cfg_scale"):
            if key in self._cfg.extra:
                payload[key] = self._cfg.extra[key]

        result = self._transport.infer(payload)
        # Archive client: missing/false ``ok`` is a hard failure.
        if not isinstance(result, dict) or not result.get("ok", False):
            raise RuntimeError(
                f"InternVLA-M1 server inference failed: "
                f"{result.get('error', result) if isinstance(result, dict) else result}"
            )
        try:
            normalized = result["data"]["normalized_actions"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"InternVLA-M1 response missing data.normalized_actions; got {result!r}"
            ) from exc

        normalized = np.asarray(normalized[0], dtype=np.float32)
        if normalized.ndim == 1:
            normalized = normalized[None, :]
        raw = _unnormalize_actions(normalized, self._action_norm_stats)

        if self._ensembler is not None:
            step7 = self._ensembler.ensemble_action(raw)
        else:
            step7 = np.asarray(raw[0], dtype=np.float32).reshape(-1)
        step7 = np.asarray(step7, dtype=np.float32).reshape(-1)
        if step7.shape[0] < 7:
            raise ValueError(f"InternVLA-M1 action too short: {step7.shape}")

        pos = step7[:3]
        rot_axangle = _euler_rpy_to_axangle(step7[3:6].astype(np.float64))
        grip_pm1 = 1.0 if float(step7[6]) > 0.5 else -1.0
        out = np.concatenate(
            [pos.astype(np.float32), rot_axangle.astype(np.float32), [grip_pm1]]
        ).astype(np.float32)
        return actions_to_chunk(out.reshape(1, -1), space=self._cfg.action_space)

    def close(self) -> None:
        self._transport.close()


def _load_run_artifacts(ckpt_file: Path) -> tuple[dict, dict, Path]:
    ckpt_file = ckpt_file.expanduser().resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"InternVLA-M1 checkpoint not found: {ckpt_file}")
    if ckpt_file.parent.name != "checkpoints":
        raise ValueError(
            "InternVLA-M1 checkpoint must live under ``checkpoints/``; "
            f"got parent {ckpt_file.parent.name!r}"
        )
    run_dir = ckpt_file.parents[1]
    config_path = run_dir / "config.yaml"
    stats_path = run_dir / "dataset_statistics.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.yaml: {config_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing dataset_statistics.json: {stats_path}")
    with open(config_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(stats_path, encoding="utf-8") as f:
        norm_stats = json.load(f)
    if not isinstance(model_cfg, dict) or not isinstance(norm_stats, dict):
        raise ValueError("InternVLA-M1 run artifacts must be mappings")
    return model_cfg, norm_stats, run_dir


def _check_unnorm_key(norm_stats: dict, unnorm_key: str | None) -> str:
    if not unnorm_key:
        if len(norm_stats) != 1:
            raise ValueError(
                "Model has multiple dataset stats; set extra.unnorm_key to one of: "
                f"{list(norm_stats.keys())}"
            )
        return next(iter(norm_stats.keys()))
    if unnorm_key not in norm_stats:
        raise KeyError(
            f"unnorm_key {unnorm_key!r} not in stats; choose from: {list(norm_stats.keys())}"
        )
    return unnorm_key


def _image_size_from_cfg(model_cfg: dict) -> tuple[int, int]:
    """Return ``(width, height)`` like archived ``InternVLAM1Inference``."""
    vla_data = model_cfg.get("datasets", {}).get("vla_data", {})
    # Archive prefers default_image_resolution over image_size.
    res = vla_data.get("default_image_resolution") or vla_data.get("image_size")
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        return (int(res[-1]), int(res[-2]))
    return (224, 224)


def _resize_uint8_hwc(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Match archive / official M1Inference: OpenCV ``INTER_AREA``."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    w, h = int(size_wh[0]), int(size_wh[1])
    return cv.resize(arr, (w, h), interpolation=cv.INTER_AREA)


def _unnormalize_actions(
    normalized_actions: np.ndarray, action_norm_stats: dict[str, Any]
) -> np.ndarray:
    action_high = np.asarray(action_norm_stats["q99"], dtype=np.float32)
    action_low = np.asarray(action_norm_stats["q01"], dtype=np.float32)
    mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
    x = np.clip(normalized_actions.astype(np.float32), -1.0, 1.0)
    x = x.copy()
    x[:, 6] = np.where(x[:, 6] < 0.5, 0.0, 1.0)
    return np.where(
        mask,
        0.5 * (x + 1.0) * (action_high - action_low) + action_low,
        x,
    ).astype(np.float32)
