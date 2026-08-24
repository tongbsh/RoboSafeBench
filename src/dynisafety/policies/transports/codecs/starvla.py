"""StarVLA / InternVLA-M1 websocket codec — wraps upstream WebsocketClientPolicy."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StarVLACodec:
    """Frame codec for StarVLA-style ``server_policy`` WebSocket servers.

    Also used by InternVLA-M1 (same client module layout under its repo root).

    Upstream APIs differ by repo:
    * StarVLA exposes ``predict_action(query_info)``
    * InternVLA-M1 exposes ``infer(obs)`` (and wraps the payload server-side)

    Parameters
    ----------
    repo_root:
        Absolute/relative path to the model repo that contains
        ``deployment.model_server.tools.websocket_policy_client``.
    """

    def __init__(self, *, repo_root: str) -> None:
        if not isinstance(repo_root, str) or not repo_root.strip():
            raise ValueError(f"repo_root must be a non-empty str, got {repo_root!r}")
        root = Path(repo_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"StarVLA/InternVLA repo_root not a directory: {root}")
        self._repo_root = str(root)
        self._client: Any = None

    def connect(self, host: str, port: int) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError(f"host must be a non-empty str, got {host!r}")
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError(f"port must be int, got {port!r}")
        if self._repo_root not in sys.path:
            sys.path.insert(0, self._repo_root)
        try:
            from deployment.model_server.tools.websocket_policy_client import (  # noqa: PLC0415
                WebsocketClientPolicy,
            )
        except ImportError as exc:
            raise ImportError(
                f"Failed to import StarVLA websocket client from {self._repo_root!r}. "
                "Set model YAML server.root to the starVLA / InternVLA-M1 checkout."
            ) from exc
        logger.info("StarVLACodec: connecting ws://%s:%s (root=%s)", host, port, self._repo_root)
        self._client = WebsocketClientPolicy(host, port)

    def reset(self, payload: Mapping[str, Any] | None) -> None:
        self._ensure()
        reset = getattr(self._client, "reset", None)
        if not callable(reset):
            return
        if payload is None:
            return
        # InternVLA-M1 ``reset(instruction)``; StarVLA has no reset.
        instruction = payload.get("instruction")
        if instruction is not None:
            reset(instruction)
        else:
            reset()

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._ensure()
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload)!r}")
        body = dict(payload)
        # Prefer StarVLA's real method; fall back to InternVLA-M1 / generic callables.
        predict = getattr(self._client, "predict_action", None)
        if callable(predict):
            result = predict(body)
        elif hasattr(self._client, "infer") and callable(self._client.infer):
            result = self._client.infer(body)
        elif callable(self._client):
            result = self._client(body)
        else:
            raise TypeError(
                f"StarVLA client {type(self._client)!r} has neither "
                "predict_action, infer, nor __call__"
            )
        if not isinstance(result, Mapping):
            raise TypeError(f"starvla infer returned {type(result)!r}, expected mapping")
        return dict(result)

    def close(self) -> None:
        if self._client is None:
            return
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()
        self._client = None

    def _ensure(self) -> None:
        if self._client is None:
            raise RuntimeError("StarVLACodec is not connected; call connect() first")
