"""Isaac-GR00T policy adapter for DyniSafety-LIBERO evaluation.

Connects to a running GR00T ZeroMQ inference server
(``vlaModelHome/Isaac-GR00T/gr00t/eval/run_gr00t_server.py``) with
``--embodiment-tag LIBERO_PANDA`` and ``--use-sim-policy-wrapper``, and wraps
it as a :class:`VLAPolicy`. Observation layout matches
``gr00t/eval/sim/LIBERO/libero_env.py``; gripper post-processing matches the
same file (normalize to [-1,1] + binarize + sign invert).

Server (GR00T conda env)::

    bash dynisafety_libero/scripts/gr00t/start_gr00t_server.sh \\
        --model-path /path/to/checkpoint --port 5555

Client (benchmark env)::

    python dynisafety_libero/scripts/gr00t/run_dynisafety_gr00t_eval.py \\
        --server_host localhost --server_port 5555 \\
        --task_suite_name safelibero_spatial

Environment variables (optional):

* ``GR00T_ROOT`` — Isaac-GR00T repo root (default: ``<workspace>/vlaModelHome/Isaac-GR00T``).
* ``GR00T_SERVER_HOST`` / ``GR00T_SERVER_PORT`` — override ``cfg`` server fields.

The client needs ``gr00t`` importable (add ``GR00T_ROOT`` to ``PYTHONPATH`` or
``pip install -e`` the repo) plus ``pyzmq`` and ``msgpack``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base_policy import VLAPolicy

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
_GR00T_ROOT = Path(os.environ.get("GR00T_ROOT", str(_WORKSPACE_ROOT / "vlaModelHome" / "Isaac-GR00T")))
_GR00T_ROOT_STR = str(_GR00T_ROOT)

# LIBERO_PANDA action horizon from gr00t/configs/data/embodiment_configs.py
_LIBERO_PANDA_ACTION_STEPS = 16

_LANG_KEY = "annotation.human.action.task_description"


def _normalize_gripper_action(action: np.ndarray, *, binarize: bool = True) -> np.ndarray:
    """Match gr00t/eval/sim/LIBERO/libero_env.normalize_gripper_action for a 1D action."""
    a = np.asarray(action, dtype=np.float32).copy()
    orig_low, orig_high = 0.0, 1.0
    a[..., -1] = 2 * (a[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        a[..., -1] = np.sign(a[..., -1])
    return a


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Match gr00t/eval/sim/LIBERO/libero_env.invert_gripper_action for a 1D action."""
    a = np.asarray(action, dtype=np.float32).copy()
    a[..., -1] = a[..., -1] * -1.0
    return a


def _make_policy_client(host: str, port: int):
    if _GR00T_ROOT_STR not in sys.path:
        sys.path.insert(0, _GR00T_ROOT_STR)
    try:
        from gr00t.policy.server_client import PolicyClient  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            f"Failed to import gr00t.policy.server_client (GR00T_ROOT={_GR00T_ROOT_STR}). "
            "Install pyzmq and msgpack, and ensure Isaac-GR00T is on PYTHONPATH or pip install -e."
        ) from e
    return PolicyClient(host=host, port=port, strict=False)


def _dyni_obs_to_gr00t_flat(obs: Dict[str, Any], task_description: str) -> Dict[str, Any]:
    """Build Gr00tSimPolicyWrapper flat observation (batch=1, time=1)."""
    img = np.asarray(obs["full_image"], dtype=np.uint8)
    wrist = np.asarray(obs["wrist_image"], dtype=np.uint8)
    if img.ndim != 3 or wrist.ndim != 3:
        raise ValueError(f"Expected HWC images; got img={img.shape} wrist={wrist.shape}")
    vid = img[None, None, ...]
    wvid = wrist[None, None, ...]

    state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    if state.shape[0] != 8:
        raise ValueError(f"Expected 8D state (eef3+axis3+gripper2); got shape {state.shape}")

    x, y, z = float(state[0]), float(state[1]), float(state[2])
    roll, pitch, yaw = float(state[3]), float(state[4]), float(state[5])
    gripper = state[6:8].astype(np.float32)

    return {
        "video.image": vid,
        "video.wrist_image": wvid,
        "state.x": np.array([[[x]]], dtype=np.float32),
        "state.y": np.array([[[y]]], dtype=np.float32),
        "state.z": np.array([[[z]]], dtype=np.float32),
        "state.roll": np.array([[[roll]]], dtype=np.float32),
        "state.pitch": np.array([[[pitch]]], dtype=np.float32),
        "state.yaw": np.array([[[yaw]]], dtype=np.float32),
        "state.gripper": gripper.reshape(1, 1, -1).astype(np.float32),
        _LANG_KEY: (task_description,),
    }


def _flat_action_to_libero_steps(flat_action: Dict[str, Any]) -> List[np.ndarray]:
    """Convert server flat action dict (B, T, *) to list of 7D LIBERO vectors."""
    keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    parts = []
    for k in keys:
        ak = flat_action[f"action.{k}"]
        ak = np.asarray(ak, dtype=np.float32)
        parts.append(ak)
    b0, t0 = parts[0].shape[0], parts[0].shape[1]
    for p in parts[1:]:
        if p.shape[0] != b0 or p.shape[1] != t0:
            raise ValueError(f"Action horizon mismatch: {[(k, flat_action['action.'+k].shape) for k in keys]}")

    out: List[np.ndarray] = []
    for t in range(t0):
        vec_parts = []
        for k in keys:
            sl = np.asarray(flat_action[f"action.{k}"][0, t], dtype=np.float32).reshape(-1)
            vec_parts.append(sl)
        action_vector = np.concatenate(vec_parts, axis=0)
        if action_vector.shape[0] != 7:
            raise ValueError(f"Expected 7D action at t={t}; got shape {action_vector.shape}")
        action_vector = _normalize_gripper_action(action_vector, binarize=True)
        action_vector = _invert_gripper_action(action_vector)
        out.append(action_vector.astype(np.float32, copy=False))
    return out


class Gr00tPolicy(VLAPolicy):
    """ZeroMQ client for the GR00T ``PolicyServer`` (LIBERO_PANDA + sim wrapper)."""

    def __init__(self, cfg) -> None:
        if not _GR00T_ROOT.is_dir():
            raise RuntimeError(
                f"GR00T_ROOT not found at '{_GR00T_ROOT}'. "
                "Set GR00T_ROOT or clone Isaac-GR00T under vlaModelHome/Isaac-GR00T."
            )

        host = os.environ.get("GR00T_SERVER_HOST", getattr(cfg, "server_host", "localhost"))
        port = os.environ.get("GR00T_SERVER_PORT")
        if port is not None:
            port = int(port)
        else:
            port = int(getattr(cfg, "server_port", 5555))

        self._client = _make_policy_client(host, port)
        if not self._client.ping():
            raise RuntimeError(
                f"GR00T policy server not reachable at {host}:{port}. "
                "Start it with scripts/gr00t/start_gr00t_server.sh (LIBERO_PANDA + sim wrapper)."
            )

        logger.info("Gr00tPolicy: tcp://%s:%s  action_horizon=%d", host, port, _LIBERO_PANDA_ACTION_STEPS)

    @property
    def action_chunk_size(self) -> int:
        return _LIBERO_PANDA_ACTION_STEPS

    def reset(self) -> None:
        self._client.reset()

    def predict_actions(self, obs: Dict[str, Any], task_description: str) -> List[np.ndarray]:
        flat_obs = _dyni_obs_to_gr00t_flat(obs, task_description)
        flat_action, _info = self._client.get_action(flat_obs)
        return _flat_action_to_libero_steps(flat_action)
