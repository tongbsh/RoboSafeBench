"""Shared ZMQ transport — thin wrap of GR00T ``PolicyClient`` (M4, S2 D4=A)."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 1.0


class ZmqTransport:
    """Session over ``gr00t.policy.server_client.PolicyClient``.

    Adapters must not import ``zmq`` / ``PolicyClient`` directly; they go through
    this transport. ``host`` / ``port`` / timeouts / ``gr00t_root`` are all
    caller-supplied (no code defaults).
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        startup_timeout_s: int,
        gr00t_root: str,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError(f"host must be a non-empty str, got {host!r}")
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError(f"port must be int, got {port!r}")
        if isinstance(startup_timeout_s, bool) or not isinstance(startup_timeout_s, int):
            raise ValueError(f"startup_timeout_s must be int, got {startup_timeout_s!r}")
        if startup_timeout_s < 1:
            raise ValueError(f"startup_timeout_s must be >= 1, got {startup_timeout_s}")
        if not isinstance(gr00t_root, str) or not gr00t_root.strip():
            raise ValueError(f"gr00t_root must be a non-empty str, got {gr00t_root!r}")

        root = Path(gr00t_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"gr00t_root not a directory: {root}")

        self._host = host
        self._port = port
        self._startup_timeout_s = startup_timeout_s
        self._gr00t_root = str(root)
        self._client: Any = None
        self._closed = False
        self._client = self._make_client()
        self._wait_until_ready()

    def _make_client(self) -> Any:
        if self._gr00t_root not in sys.path:
            sys.path.insert(0, self._gr00t_root)
        try:
            from gr00t.policy.server_client import PolicyClient  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"Failed to import gr00t.policy.server_client "
                f"(gr00t_root={self._gr00t_root!r}). "
                "Install pyzmq/msgpack and ensure Isaac-GR00T is importable."
            ) from exc
        # strict=False matches existing LIBERO Gr00tPolicy client construction.
        return PolicyClient(host=self._host, port=self._port, strict=False)

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout_s
        logger.info(
            "ZmqTransport: waiting for PolicyClient at %s:%s …",
            self._host,
            self._port,
        )
        while time.monotonic() < deadline:
            if self.health():
                logger.info("ZmqTransport: ready at %s:%s", self._host, self._port)
                return
            time.sleep(_POLL_INTERVAL_S)
        raise RuntimeError(
            f"GR00T PolicyClient at {self._host}:{self._port} not reachable within "
            f"{self._startup_timeout_s}s"
        )

    def health(self) -> bool:
        self._ensure_open()
        ping = getattr(self._client, "ping", None)
        if not callable(ping):
            return self._client is not None
        try:
            return bool(ping())
        except Exception:  # noqa: BLE001 — health probes must not raise
            return False

    def reset(self, payload: Mapping[str, Any] | None = None) -> None:
        del payload
        self._ensure_open()
        reset = getattr(self._client, "reset", None)
        if not callable(reset):
            raise TypeError(
                f"PolicyClient {type(self._client)!r} has no callable reset()"
            )
        reset()

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run ``get_action``; return ``{"action": flat_action, "info": info}``."""
        self._ensure_open()
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload)!r}")
        get_action = getattr(self._client, "get_action", None)
        if not callable(get_action):
            raise TypeError(
                f"PolicyClient {type(self._client)!r} has no callable get_action()"
            )
        result = get_action(dict(payload))
        if isinstance(result, tuple) and len(result) == 2:
            action, info = result
            out: dict[str, Any] = {"action": action}
            if info is not None:
                out["info"] = info
            return out
        if isinstance(result, Mapping):
            return dict(result)
        return {"action": result}

    def close(self) -> None:
        if self._closed:
            return
        closer = getattr(self._client, "close", None) if self._client is not None else None
        try:
            if callable(closer):
                closer()
        finally:
            self._client = None
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed or self._client is None:
            raise RuntimeError("ZmqTransport is closed")
