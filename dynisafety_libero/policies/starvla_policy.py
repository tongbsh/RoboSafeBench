"""StarVLA policy adapter for DyniSafety-LIBERO evaluation.

Connects to a running StarVLA WebSocket inference server
(``vlaModelHome/starVLA/deployment/model_server/server_policy.py``) and wraps
it as a :class:`VLAPolicy` aligned with upstream
``examples/LIBERO/eval_files/model2libero_interface.py`` (payload shape,
de-normalisation, gripper binarisation).

Server (StarVLA conda env)::

    bash dynisafety_libero/scripts/starvla/start_starvla_server.sh \\
        --ckpt-path /path/to/RUN/checkpoints/step.pt --port 10093

Client (benchmark env)::

    python dynisafety_libero/run_dynisafety_eval.py \\
        --model_family starvla \\
        --pretrained_checkpoint /path/to/RUN/checkpoints/step.pt \\
        --server_host localhost \\
        --server_port 10093 \\
        --task_suite_name safelibero_spatial

``pretrained_checkpoint`` must be a weight file under ``RUN/checkpoints/``
(``*.pt``, ``*.safetensors``, or other filenames used by your run) so the
client can load ``RUN/config.yaml`` and ``RUN/dataset_statistics.json`` for
chunk size and action stats (no StarVLA torch imports on the client).

Environment variables (optional):

* ``STARVLA_ROOT`` — repo root containing ``deployment/`` (default:
  ``<workspace>/vlaModelHome/starVLA``).
* ``STARVLA_SERVER_HOST`` / ``STARVLA_SERVER_PORT`` — override ``cfg`` server
  fields.
* ``STARVLA_USE_DDIM`` — ``true``/``false`` (default ``true``).
* ``STARVLA_NUM_DDIM_STEPS`` — int (default ``10``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from PIL import Image

from .base_policy import VLAPolicy

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
_STARVLA_ROOT = Path(
    os.environ.get("STARVLA_ROOT", str(_WORKSPACE_ROOT / "vlaModelHome" / "starVLA"))
)
if not _STARVLA_ROOT.is_dir():
    raise RuntimeError(
        f"STARVLA_ROOT not found at '{_STARVLA_ROOT}'. "
        "Set STARVLA_ROOT or clone starVLA under vlaModelHome/starVLA."
    )
_STARVLA_ROOT_STR = str(_STARVLA_ROOT)


def _make_websocket_client(host: str, port: int):
    """Lazy import so other policies work without ``websockets`` installed."""
    if _STARVLA_ROOT_STR not in sys.path:
        sys.path.insert(0, _STARVLA_ROOT_STR)
    from deployment.model_server.tools.websocket_policy_client import (  # noqa: PLC0415
        WebsocketClientPolicy,
    )

    return WebsocketClientPolicy(host, port)


def _resize_uint8_hwc(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = pil.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def _check_unnorm_key(norm_stats: dict, unnorm_key: Optional[str]) -> str:
    if not unnorm_key:
        assert len(norm_stats) == 1, (
            "Model has multiple dataset stats; pass --unnorm_key with one of: "
            f"{list(norm_stats.keys())}"
        )
        return next(iter(norm_stats.keys()))
    assert unnorm_key in norm_stats, (
        f"unnorm_key '{unnorm_key}' not in stats; choose from: {list(norm_stats.keys())}"
    )
    return unnorm_key


def _load_run_artifacts(ckpt_file: Path) -> tuple[dict, dict, Path]:
    """Load config.yaml + dataset_statistics.json from run dir (parent of checkpoints/)."""
    ckpt_file = ckpt_file.resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"StarVLA checkpoint not found: {ckpt_file}")
    if ckpt_file.parent.name != "checkpoints":
        raise ValueError(
            "pretrained_checkpoint must live under a directory named ``checkpoints`` "
            f"(StarVLA run layout); got parent {ckpt_file.parent.name!r}"
        )
    run_dir = ckpt_file.parents[1]
    config_path = run_dir / "config.yaml"
    stats_path = run_dir / "dataset_statistics.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.yaml next to checkpoint run: {config_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing dataset_statistics.json: {stats_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(stats_path, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    return model_cfg, norm_stats, run_dir


def _unnormalize_actions(
    normalized_actions: np.ndarray, action_norm_stats: Dict[str, Any]
) -> np.ndarray:
    mask = action_norm_stats.get(
        "mask", np.ones_like(np.array(action_norm_stats["min"]), dtype=bool)
    )
    action_high = np.array(action_norm_stats["max"], dtype=np.float32)
    action_low = np.array(action_norm_stats["min"], dtype=np.float32)
    x = np.clip(normalized_actions.astype(np.float32), -1.0, 1.0)
    x[:, 6] = np.where(x[:, 6] < 0.5, 0.0, 1.0)
    return np.where(
        mask,
        0.5 * (x + 1.0) * (action_high - action_low) + action_low,
        x,
    )


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


class StarVLAPolicy(VLAPolicy):
    """WebSocket client for the StarVLA ``server_policy`` (LIBERO delta actions)."""

    def __init__(self, cfg) -> None:
        host = os.environ.get(
            "STARVLA_SERVER_HOST", getattr(cfg, "server_host", "localhost")
        )
        port = os.environ.get("STARVLA_SERVER_PORT")
        if port is not None:
            port = int(port)
        else:
            port = int(getattr(cfg, "server_port", 10093))

        ckpt_arg = getattr(cfg, "pretrained_checkpoint", "")
        if ckpt_arg is None or str(ckpt_arg).strip() == "":
            raise ValueError(
                "StarVLAPolicy requires cfg.pretrained_checkpoint (path to .pt under RUN/checkpoints/)."
            )
        ckpt = Path(str(ckpt_arg)).expanduser()

        model_cfg, norm_stats, run_dir = _load_run_artifacts(ckpt)
        raw_un = getattr(cfg, "unnorm_key", "")
        if raw_un is None or str(raw_un).strip() == "":
            unnorm_key = None
        else:
            unnorm_key = str(raw_un)
        resolved_key = _check_unnorm_key(norm_stats, unnorm_key)
        self._action_norm_stats: Dict[str, Any] = norm_stats[resolved_key]["action"]

        fam = model_cfg["framework"]["action_model"]
        self._chunk_size: int = int(fam["future_action_window_size"]) + 1
        self._image_size: tuple[int, int] = (224, 224)
        vla_data = model_cfg.get("datasets", {}).get("vla_data", {})
        res = vla_data.get("default_image_resolution")
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            h, w = int(res[-2]), int(res[-1])
            self._image_size = (w, h)

        self._use_ddim = os.environ.get("STARVLA_USE_DDIM", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        self._num_ddim_steps = int(os.environ.get("STARVLA_NUM_DDIM_STEPS", "10"))

        logger.info(
            "StarVLAPolicy: ws://%s:%s  run_dir=%s  unnorm_key=%s  chunk=%d  image=%s",
            host,
            port,
            run_dir,
            resolved_key,
            self._chunk_size,
            self._image_size,
        )

        self._client = _make_websocket_client(host, port)
        self._last_task_description: Optional[str] = None

    @property
    def action_chunk_size(self) -> int:
        return self._chunk_size

    def reset(self) -> None:
        self._last_task_description = None

    def predict_actions(self, obs: Dict[str, Any], task_description: str) -> List[np.ndarray]:
        if task_description != self._last_task_description:
            self._last_task_description = task_description

        primary = _resize_uint8_hwc(obs["full_image"], self._image_size)
        wrist = _resize_uint8_hwc(obs["wrist_image"], self._image_size)
        example = {
            "image": [primary, wrist],
            "lang": task_description,
        }
        vla_input: Dict[str, Any] = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self._use_ddim,
            "num_ddim_steps": self._num_ddim_steps,
        }

        response = self._client.predict_action(vla_input)
        try:
            normalized = response["data"]["normalized_actions"]
        except KeyError as exc:
            raise KeyError(
                f"StarVLA server response missing normalized_actions: {response.get('data', response)}"
            ) from exc
        normalized = np.asarray(normalized[0], dtype=np.float32)
        if normalized.ndim == 1:
            normalized = normalized[None, :]

        raw = _unnormalize_actions(normalized, self._action_norm_stats)
        out: List[np.ndarray] = []
        for i in range(raw.shape[0]):
            row = raw[i]
            wv = np.asarray(row[:3], dtype=np.float32).reshape(-1)
            rot = np.asarray(row[3:6], dtype=np.float32).reshape(-1)
            og = np.asarray(row[6:7], dtype=np.float32).reshape(-1)
            grip = _binarize_gripper_open(og)
            if wv.size != 3 or rot.size != 3 or grip.size != 1:
                raise ValueError(
                    f"Bad action shape from server: row={row.shape} wv={wv.shape} rot={rot.shape}"
                )
            out.append(np.concatenate([wv, rot, grip], axis=0))
        return out
