"""Shared WebSocket transport with pluggable codecs (M4, S2 D3=A)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class WebSocketCodec(Protocol):
    """Per-family framing (openpi / starvla / lingbot)."""

    def connect(self, host: str, port: int) -> None: ...

    def reset(self, payload: Mapping[str, Any] | None) -> None: ...

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class WebSocketTransport:
    """Thin session wrapper around a :class:`WebSocketCodec`.

    ``host`` / ``port`` are injected by serve/sweep (never read from code defaults).
    """

    def __init__(self, *, host: str, port: int, codec: WebSocketCodec) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError(f"host must be a non-empty str, got {host!r}")
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError(f"port must be int, got {port!r}")
        self._host = host
        self._port = port
        self._codec = codec
        self._closed = False
        self._codec.connect(host, port)

    def reset(self, payload: Mapping[str, Any] | None = None) -> None:
        self._ensure_open()
        self._codec.reset(payload)

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._ensure_open()
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload)!r}")
        return self._codec.infer(payload)

    def health(self) -> bool:
        """True while the session is open (codecs lack a uniform ping API)."""
        return not self._closed

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._codec.close()
        finally:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("WebSocketTransport is closed")
