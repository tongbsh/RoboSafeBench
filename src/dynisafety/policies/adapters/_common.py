"""Shared helpers for policy adapters (not a registered model)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynisafety.policies.base import ActionChunk, ModelConfig, Observation
from dynisafety.policies.transports import Transport, build_transport


def require_extra(cfg: ModelConfig, key: str) -> Any:
    if key not in cfg.extra:
        raise ValueError(
            f"model {cfg.name!r}: ModelConfig.extra missing required key {key!r}"
        )
    return cfg.extra[key]


def optional_extra(cfg: ModelConfig, key: str) -> Any | None:
    return cfg.extra[key] if key in cfg.extra else None


def require_image(obs: Observation, key: str) -> np.ndarray:
    if key not in obs.images:
        raise KeyError(
            f"Observation.images missing {key!r}; have {sorted(obs.images)}"
        )
    return np.asarray(obs.images[key])


def require_state(obs: Observation) -> np.ndarray:
    if obs.state is None:
        raise ValueError("Observation.state is required but missing")
    return np.asarray(obs.state)


def remote_host_port(cfg: ModelConfig) -> tuple[str, int]:
    host = require_extra(cfg, "server_host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError(f"extra.server_host must be non-empty str, got {host!r}")
    if "server_port" in cfg.extra:
        port = cfg.extra["server_port"]
    elif cfg.server_default_port is not None:
        port = cfg.server_default_port
    else:
        raise ValueError(
            f"model {cfg.name!r}: need extra.server_port or server.default_port"
        )
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError(f"server port must be int, got {port!r}")
    return host, port


def connect_transport(cfg: ModelConfig) -> Transport:
    if cfg.transport == "local":
        raise ValueError(
            f"model {cfg.name!r}: connect_transport called with transport='local'"
        )
    host, port = remote_host_port(cfg)
    return build_transport(cfg, host=host, port=port)


def actions_to_chunk(
    actions: np.ndarray | Sequence[np.ndarray],
    *,
    space: str,
) -> ActionChunk:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"actions must be (chunk, dim), got shape {arr.shape}")
    return ActionChunk(actions=arr, space=space)


def as_jsonable(obj: Any) -> Any:
    """Convert numpy containers to JSON-serializable Python objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    return obj


def ensure_repo_on_path(repo_root: str) -> str:
    """Prepend *repo_root* (and ``repo_root/src`` if present) to ``sys.path``.

    Many VLA checkouts use a src-layout package (e.g. InternVLA-A1 → ``src/lerobot``).
    Editable installs that still point at a moved tree leave ``import lerobot`` broken;
    injecting ``…/src`` restores the historical import path without reinstall.
    """
    import sys
    from pathlib import Path

    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repo_root not a directory: {root}")
    for candidate in (root / "src", root):
        if not candidate.is_dir():
            continue
        s = str(candidate)
        if s not in sys.path:
            sys.path.insert(0, s)
    return str(root)
