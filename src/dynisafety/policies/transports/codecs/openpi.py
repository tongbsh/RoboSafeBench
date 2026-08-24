"""OpenPI websocket codec — wraps ``openpi_client.WebsocketClientPolicy``."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OpenPiCodec:
    """Frame codec for openpi ``serve_policy.py`` WebSocket servers.

    Parameters
    ----------
    client_src:
        Filesystem path to openpi ``packages/openpi-client/src`` (or any directory
        that makes ``openpi_client`` importable). Required — no default search.
    connect_timeout_s:
        Wall-clock budget for the initial TCP/WebSocket handshake. Upstream
        ``WebsocketClientPolicy`` retries ``ConnectionRefusedError`` forever;
        without a deadline a wrong port (e.g. YAML default_port vs sweep PORT)
        hangs the whole sweep. Pass ``server.startup_timeout_s`` from YAML.
    """

    def __init__(self, *, client_src: str, connect_timeout_s: int) -> None:
        if not isinstance(client_src, str) or not client_src.strip():
            raise ValueError(f"client_src must be a non-empty str, got {client_src!r}")
        if isinstance(connect_timeout_s, bool) or not isinstance(connect_timeout_s, int):
            raise ValueError(
                f"connect_timeout_s must be int, got {connect_timeout_s!r}"
            )
        if connect_timeout_s < 1:
            raise ValueError(f"connect_timeout_s must be >= 1, got {connect_timeout_s}")
        root = Path(client_src).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"openpi client_src not a directory: {root}")
        self._client_src = str(root)
        self._connect_timeout_s = connect_timeout_s
        self._client: Any = None
        self._metadata: dict[str, Any] = {}

    def connect(self, host: str, port: int) -> None:
        import socket
        import time

        if not isinstance(host, str) or not host.strip():
            raise ValueError(f"host must be a non-empty str, got {host!r}")
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError(f"port must be int, got {port!r}")
        if self._client_src not in sys.path:
            sys.path.insert(0, self._client_src)
        try:
            from openpi_client.websocket_client_policy import (  # noqa: PLC0415
                WebsocketClientPolicy,
            )
        except ImportError as exc:
            raise ImportError(
                f"Failed to import openpi_client from {self._client_src!r}. "
                "Install/open the openpi client package or fix extra.openpi_client_src."
            ) from exc

        # Preflight TCP: upstream client has an infinite ConnectionRefused retry.
        deadline = time.monotonic() + self._connect_timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=2.0):
                    break
            except OSError as exc:
                last_err = exc
                time.sleep(0.5)
        else:
            raise TimeoutError(
                f"OpenPiCodec: TCP {host}:{port} not reachable within "
                f"{self._connect_timeout_s}s (last error: {last_err}). "
                "Check that sweep/serve injected extra.server_port (or env PORT) "
                "and that it matches the live server."
            )

        logger.info("OpenPiCodec: connecting ws://%s:%s", host, port)
        self._client = WebsocketClientPolicy(host=host, port=port)
        meta = self._client.get_server_metadata()
        self._metadata = dict(meta) if isinstance(meta, Mapping) else {}

    @property
    def server_metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def reset(self, payload: Mapping[str, Any] | None) -> None:
        del payload
        self._ensure()
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            reset()

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._ensure()
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload)!r}")
        result = self._client.infer(dict(payload))
        if not isinstance(result, Mapping):
            raise TypeError(f"openpi infer returned {type(result)!r}, expected mapping")
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
            raise RuntimeError("OpenPiCodec is not connected; call connect() first")
