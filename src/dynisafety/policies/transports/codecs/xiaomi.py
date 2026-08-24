"""Xiaomi-Robotics-1 length-prefixed pickle socket (not a WebSocket).

Official ``deploy/client.py`` frames each request as ``>I`` length + pickle.
The session wrapper is still ``WebSocketTransport``; this codec only replaces
the wire format so adapters can use ``connect_transport``.
"""

from __future__ import annotations

import pickle
import socket
import struct
import time
from collections.abc import Mapping
from typing import Any


class XiaomiCodec:
    """TCP pickle codec matching ``Xiaomi-Robotics-1/deploy/client.py``."""

    def __init__(self, *, connect_timeout_s: int) -> None:
        if isinstance(connect_timeout_s, bool) or not isinstance(connect_timeout_s, int):
            raise ValueError(f"connect_timeout_s must be int, got {connect_timeout_s!r}")
        if connect_timeout_s < 1:
            raise ValueError(f"connect_timeout_s must be >= 1, got {connect_timeout_s}")
        self._connect_timeout_s = connect_timeout_s
        self._sock: socket.socket | None = None

    def connect(self, host: str, port: int) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError(f"host must be a non-empty str, got {host!r}")
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError(f"port must be int, got {port!r}")
        deadline = time.time() + self._connect_timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(min(2.0, max(0.1, deadline - time.time())))
                sock.connect((host, port))
                sock.settimeout(None)
                self._sock = sock
                return
            except OSError as exc:
                last_err = exc
                time.sleep(0.5)
        raise ConnectionError(
            f"XiaomiCodec failed to connect to {host}:{port} within "
            f"{self._connect_timeout_s}s: {last_err}"
        )

    def reset(self, payload: Mapping[str, Any] | None) -> None:
        del payload

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._sock is None:
            raise RuntimeError("XiaomiCodec is not connected")
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload)!r}")
        serialized = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
        self._sock.sendall(struct.pack(">I", len(serialized)) + serialized)
        actions = self._recv_pickle()
        return {"actions": actions}

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            sock.close()

    def _recv_all(self, length: int) -> bytes:
        if self._sock is None:
            raise RuntimeError("XiaomiCodec is not connected")
        data = b""
        while len(data) < length:
            packet = self._sock.recv(length - len(data))
            if not packet:
                raise ConnectionError("Xiaomi server closed the connection")
            data += packet
        return data

    def _recv_pickle(self) -> Any:
        len_data = self._recv_all(4)
        data_len = struct.unpack(">I", len_data)[0]
        return pickle.loads(self._recv_all(data_len))
