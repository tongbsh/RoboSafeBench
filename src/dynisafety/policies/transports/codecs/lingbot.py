"""LingBot-VA websocket codec (msgpack-numpy framing)."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from dynisafety.policies.transports.codecs._msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)


class LingBotCodec:
    """msgpack WebSocket codec matching LingBot-VA ``wan_va_server``.

    Parameters
    ----------
    connect_timeout_s:
        Max seconds to wait for the initial connection (required).
    api_key:
        Optional API key; pass ``None`` explicitly when unused.
    """

    def __init__(self, *, connect_timeout_s: int, api_key: str | None) -> None:
        if isinstance(connect_timeout_s, bool) or not isinstance(connect_timeout_s, int):
            raise ValueError(f"connect_timeout_s must be int, got {connect_timeout_s!r}")
        if connect_timeout_s < 1:
            raise ValueError(f"connect_timeout_s must be >= 1, got {connect_timeout_s}")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError(f"api_key must be None or a non-empty str, got {api_key!r}")
        self._connect_timeout_s = connect_timeout_s
        self._api_key = api_key
        self._packer: Any = None
        self._ws: Any = None
        self._server_metadata: dict[str, Any] = {}

    def connect(self, host: str, port: int) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError(f"host must be a non-empty str, got {host!r}")
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError(f"port must be int, got {port!r}")
        try:
            import websockets.sync.client  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "LingBotCodec requires the 'websockets' package in the eval env."
            ) from exc
        uri = f"ws://{host}:{port}"
        deadline = time.monotonic() + self._connect_timeout_s
        logger.info("LingBotCodec: connecting %s …", uri)
        last_err: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key is not None
                    else None
                )
                conn = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    close_timeout=10,
                )
                raw_meta = conn.recv()
                meta = unpackb(raw_meta)
                self._server_metadata = dict(meta) if isinstance(meta, Mapping) else {}
                self._ws = conn
                return
            except Exception as exc:  # noqa: BLE001 — retry until timeout
                last_err = exc
                time.sleep(1.0)
        raise RuntimeError(
            f"LingBotCodec: failed to connect to {uri} within "
            f"{self._connect_timeout_s}s: {last_err}"
        )

    @property
    def server_metadata(self) -> Mapping[str, Any]:
        return self._server_metadata

    def reset(self, payload: Mapping[str, Any] | None) -> None:
        del payload
        self._ensure()
        # Upstream LingBot client reset is a no-op; keep parity.

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._ensure()
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload)!r}")
        if self._packer is None:
            self._packer = Packer()
        self._ws.send(self._packer.pack(dict(payload)))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in LingBot inference server:\n{response}")
        result = unpackb(response)
        if not isinstance(result, Mapping):
            raise TypeError(f"lingbot infer returned {type(result)!r}, expected mapping")
        return dict(result)

    def close(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.close()
        finally:
            self._ws = None

    def _ensure(self) -> None:
        if self._ws is None:
            raise RuntimeError("LingBotCodec is not connected; call connect() first")
