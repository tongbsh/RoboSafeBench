"""StarVLA WebSocket adapter (LIBERO/Simpler EE or RoboTwin qpos via YAML space).

Wire contract mirrors the archived LIBERO client
(``old_cache/dynisafety_libero/policies/starvla_policy.py``) and the RoboTwin /
SimplerEnv counterparts: ``predict_action`` + ``examples`` payload, then client-side
de-normalisation from ``RUN/dataset_statistics.json``.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from dynisafety.policies.adapters._common import (
    actions_to_chunk,
    connect_transport,
    require_image,
)
from dynisafety.policies.base import (
    ActionChunk,
    EpisodeContext,
    ModelConfig,
    Observation,
)
from dynisafety.policies.transports import Transport

logger = logging.getLogger(__name__)

# RoboTwin StarVLA server emits
# [left_arm(6), right_arm(6), left_gripper, right_gripper]; env expects
# [left_arm(6), left_gripper, right_arm(6), right_gripper].
_ROBOTWIN_QPOS_PERM = np.asarray(
    [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13], dtype=np.int64
)


class StarVLAAdapter:
    """StarVLA ``server_policy`` client.

    Requires ``extra.ws_codec: starvla`` and ``server.root``.
    Observation packing follows ``extra.obs_layout``:
    ``libero`` (primary+wrist) | ``simpler`` (primary) | ``robotwin`` (primary/left/right).
    """

    def __init__(self, cfg: ModelConfig, transport: Transport) -> None:
        self._cfg = cfg
        self._transport = transport
        layout = cfg.extra.get("obs_layout")
        if layout is None:
            raise ValueError("StarVLAAdapter requires extra.obs_layout")
        if layout not in ("libero", "simpler", "robotwin"):
            raise ValueError(
                f"extra.obs_layout must be libero|simpler|robotwin, got {layout!r}"
            )
        self._layout = str(layout)

        model_cfg, norm_stats, run_dir = _load_run_artifacts(Path(cfg.checkpoint))
        raw_un = cfg.extra.get("unnorm_key")
        unnorm_key = None if raw_un is None or str(raw_un).strip() == "" else str(raw_un)
        resolved_key = _check_unnorm_key(norm_stats, unnorm_key)
        self._action_norm_stats = _resolve_action_stats(
            norm_stats[resolved_key],
            action_mode=str(cfg.extra.get("action_mode") or "abs"),
        )
        self._norm_bounds = _pick_norm_bounds(
            self._action_norm_stats,
            layout=self._layout,
            normalization_mode=cfg.extra.get("normalization_mode"),
        )

        fam = model_cfg["framework"]["action_model"]
        self._server_chunk = int(fam["future_action_window_size"]) + 1
        self._image_size = _image_size_from_cfg(model_cfg)

        # Bridge (simpler): archive StarVLAInference uses AdaptiveEnsembler +
        # euler→axangle and returns one env step per query.
        self._ensembler: _AdaptiveEnsembler | None = None
        if self._layout == "simpler":
            ens_flag = cfg.extra.get("action_ensemble", True)
            if isinstance(ens_flag, str):
                ens_on = ens_flag.strip().lower() in ("1", "true", "yes", "on")
            else:
                ens_on = bool(ens_flag) if ens_flag is not None else True
            if ens_on:
                horizon = int(cfg.extra.get("action_ensemble_horizon") or 7)
                alpha = float(cfg.extra.get("adaptive_ensemble_alpha") or 0.1)
                self._ensembler = _AdaptiveEnsembler(horizon, alpha)

        logger.info(
            "StarVLAAdapter: layout=%s run_dir=%s unnorm_key=%s chunk=%d image=%s ensemble=%s",
            self._layout,
            run_dir,
            resolved_key,
            self._server_chunk,
            self._image_size,
            "off"
            if self._ensembler is None
            else f"on(h={self._ensembler.pred_action_horizon})",
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> StarVLAAdapter:
        if cfg.transport != "websocket":
            raise ValueError("StarVLAAdapter requires transport='websocket'")
        return cls(cfg, connect_transport(cfg))

    def reset(self, ctx: EpisodeContext) -> None:
        del ctx
        if self._ensembler is not None:
            self._ensembler.reset()
        self._transport.reset(None)

    def act(self, obs: Observation) -> ActionChunk:
        example = self._build_example(obs)
        payload: dict[str, Any] = {
            "examples": [example],
            "do_sample": False,
        }
        for key in ("use_ddim", "num_ddim_steps"):
            if key in self._cfg.extra:
                payload[key] = self._cfg.extra[key]

        result = self._transport.infer(payload)
        try:
            normalized = result["data"]["normalized_actions"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"StarVLA response missing data.normalized_actions; "
                f"keys={sorted(result) if isinstance(result, dict) else type(result)}"
            ) from exc

        normalized = np.asarray(normalized[0], dtype=np.float32)
        if normalized.ndim == 1:
            normalized = normalized[None, :]

        raw = _unnormalize_actions(
            normalized,
            self._action_norm_stats,
            bounds=self._norm_bounds,
            binarize_gripper7=(self._layout in ("libero", "simpler")),
        )
        if self._layout == "libero":
            raw = _libero_binarize_gripper_pm1(raw)
            return actions_to_chunk(raw, space=self._cfg.action_space)
        if self._layout == "simpler":
            return self._act_simpler_bridge(raw)
        # robotwin
        raw = raw[:, _ROBOTWIN_QPOS_PERM]
        return actions_to_chunk(raw, space=self._cfg.action_space)

    def _act_simpler_bridge(self, raw: np.ndarray) -> ActionChunk:
        """Match archived ``StarVLAInference`` widowx_bridge step packing."""
        if self._ensembler is not None:
            step7 = self._ensembler.ensemble_action(raw)
        else:
            step7 = np.asarray(raw[0], dtype=np.float32).reshape(-1)
        step7 = np.asarray(step7, dtype=np.float32).reshape(-1)
        if step7.shape[0] < 7:
            raise ValueError(f"StarVLA Bridge action too short: {step7.shape}")

        pos = step7[:3]
        euler_rpy = step7[3:6].astype(np.float64)
        grip = float(step7[6])
        rot_axangle = _euler_rpy_to_axangle(euler_rpy)
        grip_pm1 = 1.0 if grip > 0.5 else -1.0
        out = np.concatenate(
            [pos.astype(np.float32), rot_axangle.astype(np.float32), [grip_pm1]]
        ).astype(np.float32)
        return actions_to_chunk(out.reshape(1, -1), space=self._cfg.action_space)

    def close(self) -> None:
        self._transport.close()

    def _build_example(self, obs: Observation) -> dict[str, Any]:
        if self._layout == "libero":
            images = [
                _resize_uint8_hwc(require_image(obs, "primary"), self._image_size),
                _resize_uint8_hwc(require_image(obs, "wrist"), self._image_size),
            ]
            return {"image": images, "lang": obs.instruction}
        if self._layout == "simpler":
            # Archive StarVLAInference (Bridge) uses cv2 INTER_AREA, not LANCZOS.
            images = [
                _resize_uint8_hwc_area(
                    require_image(obs, "primary"), self._image_size
                ),
            ]
            return {"image": images, "lang": obs.instruction}
        # robotwin — archived client sends images only inside examples (no state).
        images = [
            _resize_uint8_hwc(require_image(obs, "primary"), self._image_size),
            _resize_uint8_hwc(require_image(obs, "left"), self._image_size),
            _resize_uint8_hwc(require_image(obs, "right"), self._image_size),
        ]
        return {"image": images, "lang": obs.instruction}


def _load_run_artifacts(ckpt_file: Path) -> tuple[dict, dict, Path]:
    """Load ``config.yaml`` + ``dataset_statistics.json`` from StarVLA run dir."""
    ckpt_file = ckpt_file.expanduser().resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"StarVLA checkpoint not found: {ckpt_file}")
    if ckpt_file.parent.name != "checkpoints":
        raise ValueError(
            "StarVLA checkpoint must live under a directory named ``checkpoints`` "
            f"(run layout); got parent {ckpt_file.parent.name!r}"
        )
    run_dir = ckpt_file.parents[1]
    config_path = run_dir / "config.yaml"
    stats_path = run_dir / "dataset_statistics.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.yaml next to checkpoint run: {config_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing dataset_statistics.json: {stats_path}")
    with open(config_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(stats_path, encoding="utf-8") as f:
        norm_stats = json.load(f)
    if not isinstance(model_cfg, dict) or not isinstance(norm_stats, dict):
        raise ValueError("StarVLA run artifacts must be mappings")
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


def _resolve_action_stats(stats_entry: dict, *, action_mode: str) -> dict[str, Any]:
    """Pick action stats; support both flat ``action`` and mode-nested layouts."""
    if action_mode in stats_entry and isinstance(stats_entry[action_mode], dict):
        mode_stats = stats_entry[action_mode]
        return mode_stats.get("action", mode_stats)
    if "action" in stats_entry:
        return stats_entry["action"]
    raise ValueError(
        f"Invalid statistics format; keys={sorted(stats_entry)}; action_mode={action_mode!r}"
    )


def _pick_norm_bounds(
    action_norm_stats: dict[str, Any],
    *,
    layout: str,
    normalization_mode: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (high, low) arrays for unnormalisation.

    * libero: archived client always used ``min``/``max``.
    * simpler: prefer ``q01``/``q99`` when present (OXE), else min/max.
    * robotwin: ``extra.normalization_mode`` ``min_max`` (default) or ``q99``.
    """
    mode = None if normalization_mode is None else str(normalization_mode)
    if layout == "libero" or mode == "min_max":
        return (
            np.asarray(action_norm_stats["max"], dtype=np.float32),
            np.asarray(action_norm_stats["min"], dtype=np.float32),
        )
    if layout == "simpler" or mode == "q99":
        if "q01" in action_norm_stats and "q99" in action_norm_stats:
            return (
                np.asarray(action_norm_stats["q99"], dtype=np.float32),
                np.asarray(action_norm_stats["q01"], dtype=np.float32),
            )
        return (
            np.asarray(action_norm_stats["max"], dtype=np.float32),
            np.asarray(action_norm_stats["min"], dtype=np.float32),
        )
    # robotwin default
    return (
        np.asarray(action_norm_stats["max"], dtype=np.float32),
        np.asarray(action_norm_stats["min"], dtype=np.float32),
    )


def _image_size_from_cfg(model_cfg: dict) -> tuple[int, int]:
    """Return ``(width, height)`` for PIL resize."""
    vla_data = model_cfg.get("datasets", {}).get("vla_data", {})
    res = vla_data.get("image_size") or vla_data.get("default_image_resolution")
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        # configs alternate between [H,W] and [W,H]; StarVLA clients treat last two
        # as H,W when length>=2 for default_image_resolution, and [224,224] as square.
        a, b = int(res[-2]), int(res[-1])
        if a == b:
            return (a, b)
        # Prefer (W, H) like SimplerEnv StarVLAInference when given HxW-ish lists.
        return (b, a)
    return (224, 224)


def _resize_uint8_hwc(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """LIBERO archive client uses PIL LANCZOS."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    pil = Image.fromarray(arr)
    pil = pil.resize(size_wh, Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def _resize_uint8_hwc_area(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Bridge / SimplerEnv archive StarVLAInference uses OpenCV INTER_AREA."""
    import cv2 as cv  # noqa: PLC0415

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    w, h = int(size_wh[0]), int(size_wh[1])
    return cv.resize(arr, (w, h), interpolation=cv.INTER_AREA)


def _unnormalize_actions(
    normalized_actions: np.ndarray,
    action_norm_stats: dict[str, Any],
    *,
    bounds: tuple[np.ndarray, np.ndarray],
    binarize_gripper7: bool,
) -> np.ndarray:
    action_high, action_low = bounds
    mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
    x = np.clip(normalized_actions.astype(np.float32), -1.0, 1.0)
    if binarize_gripper7 and x.shape[-1] >= 7:
        x = x.copy()
        x[:, 6] = np.where(x[:, 6] < 0.5, 0.0, 1.0)
    return np.where(
        mask,
        0.5 * (x + 1.0) * (action_high - action_low) + action_low,
        x,
    ).astype(np.float32)


def _libero_binarize_gripper_pm1(actions: np.ndarray) -> np.ndarray:
    """Match archived LIBERO client: gripper → {-1, +1} after unnorm."""
    out = np.asarray(actions, dtype=np.float32).copy()
    open_vals = out[:, 6]
    out[:, 6] = np.where(open_vals > 0.5, -1.0, 1.0)
    return out


def _euler_rpy_to_axangle(euler_rpy: np.ndarray) -> np.ndarray:
    """Match archived Bridge ``euler2axangle`` → ``ax * angle`` packing."""
    try:
        from transforms3d.euler import euler2axangle  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        rotvec = Rotation.from_euler("xyz", np.asarray(euler_rpy, dtype=np.float64)).as_rotvec()
        return np.asarray(rotvec, dtype=np.float64)

    roll, pitch, yaw = np.asarray(euler_rpy, dtype=np.float64).reshape(3)
    ax, angle = euler2axangle(roll, pitch, yaw)
    return (np.asarray(ax, dtype=np.float64) * float(angle)).astype(np.float64)


class _AdaptiveEnsembler:
    """Same logic as archived SimplerEnv ``starvla_model._AdaptiveEnsembler``."""

    def __init__(self, pred_action_horizon: int, adaptive_ensemble_alpha: float = 0.0) -> None:
        self.pred_action_horizon = int(pred_action_horizon)
        self.adaptive_ensemble_alpha = float(adaptive_ensemble_alpha)
        self.action_history: deque[np.ndarray] = deque(maxlen=self.pred_action_horizon)

    def reset(self) -> None:
        self.action_history.clear()

    def ensemble_action(self, cur_action: np.ndarray) -> np.ndarray:
        self.action_history.append(np.asarray(cur_action))
        num_actions = len(self.action_history)
        cur = self.action_history[-1]
        if cur.ndim == 1:
            curr_act_preds = np.stack(list(self.action_history), axis=0)
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
        return np.sum(weights[:, None] * curr_act_preds, axis=0).astype(np.float32)
