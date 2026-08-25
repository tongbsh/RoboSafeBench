"""Localhost TCP channel with length-prefixed pickle frames.

Socket recv boundaries are not message boundaries, so every payload is framed
as ``struct.Struct('!I')`` length + pickle bytes. Timeouts and disconnects
raise; this module never retries or returns ``None``. ``SO_REUSEADDR`` is set
on the listen socket so a previous run's TIME_WAIT does not steal the next
eval's port.

Host / port / timeout are constructor-required (iron law 1). Callers read them
from ``extra.parallel_eval``.
"""

from __future__ import annotations

import pickle
import select
import socket
import struct
from collections.abc import Sequence
from typing import Any

from dynisafety.parallel_eval.protocol import validate_payload

_FRAME_HEADER = struct.Struct("!I")


class ChannelError(RuntimeError):
    """IPC failed. Callers must surface this; do not swallow or retry here."""


class ChannelTimeout(ChannelError):
    """connect / accept / put / get exceeded the configured timeout."""


class ChannelClosed(ChannelError):
    """Peer closed this socket. Infer drops the slot; other slots keep running."""


class Channel:
    """Bidirectional pipe after the TCP handshake (Infer or Env side)."""

    def __init__(
        self,
        sock: socket.socket,
        *,
        host: str,
        port: int,
        timeout_s: float,
        role: str,
    ) -> None:
        _check_host(host)
        _check_port(port)
        timeout = _check_timeout_s(timeout_s)
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"role must be a non-empty str, got {role!r}")
        self._sock: socket.socket | None = sock
        self._host = host
        self._port = port
        self._timeout_s = timeout
        self._role = role
        self._apply_connected_options()

    @classmethod
    def bind(cls, *, host: str, port: int, timeout_s: float) -> ChannelListener:
        """Infer-side listen. ``port=0`` lets the OS pick an ephemeral port."""
        return ChannelListener.bind(host=host, port=port, timeout_s=timeout_s)

    @classmethod
    def connect(cls, *, host: str, port: int, timeout_s: float) -> Channel:
        """Env-side connect. Fail-loud on refuse / timeout (no silent retry)."""
        _check_host(host)
        _check_port(port)
        timeout = _check_timeout_s(timeout_s)
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
        except TimeoutError as exc:
            raise ChannelTimeout(
                f"channel client connect timed out after {timeout}s at {host}:{port}"
            ) from exc
        except OSError as exc:
            raise ChannelError(
                f"channel client connect failed at {host}:{port}: {exc}"
            ) from exc
        return cls(sock, host=host, port=port, timeout_s=timeout, role="client")

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def put(self, item: Any) -> None:
        """Send one already-validated wire payload (typically ``to_wire(msg)``)."""
        sock = self._require_open()
        validate_payload(item)
        try:
            payload = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
            sock.sendall(_FRAME_HEADER.pack(len(payload)) + payload)
        except TimeoutError as exc:
            raise ChannelTimeout(self._timeout_message("put")) from exc
        except OSError as exc:
            raise ChannelError(self._io_message("put", exc)) from exc

    def get(self) -> Any:
        """Receive one frame. Raises on timeout, peer close, or unsafe payload."""
        sock = self._require_open()
        try:
            header = _recv_exact(sock, _FRAME_HEADER.size)
            (length,) = _FRAME_HEADER.unpack(header)
            if length < 1:
                raise ChannelError(self._io_message("get", "empty frame"))
            payload = _recv_exact(sock, length)
            item = pickle.loads(payload)
        except ChannelError:
            raise
        except TimeoutError as exc:
            raise ChannelTimeout(self._timeout_message("get")) from exc
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
            raise ChannelClosed(self._io_message("get", exc)) from exc
        except OSError as exc:
            raise ChannelError(self._io_message("get", exc)) from exc
        except pickle.UnpicklingError as exc:
            raise ChannelError(self._io_message("get", exc)) from exc
        validate_payload(item)
        return item

    def fileno(self) -> int:
        """Socket fd for ``select``. Closed channels fail-loud, same as I/O."""
        return self._require_open().fileno()

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is None:
            return
        _close_socket(sock)

    def __enter__(self) -> Channel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _apply_connected_options(self) -> None:
        sock = self._require_open()
        sock.settimeout(self._timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _require_open(self) -> socket.socket:
        if self._sock is None:
            raise ChannelError(
                f"channel {self._role} is closed ({self._host}:{self._port})"
            )
        return self._sock

    def _timeout_message(self, op: str) -> str:
        return (
            f"channel {self._role} {op} timed out after {self._timeout_s}s "
            f"at {self._host}:{self._port}"
        )

    def _io_message(self, op: str, exc: object) -> str:
        return f"channel {self._role} {op} failed at {self._host}:{self._port}: {exc}"


class ChannelListener:
    """Infer-side acceptor. One ``accept()`` per Env worker process."""

    def __init__(
        self,
        sock: socket.socket,
        *,
        host: str,
        port: int,
        timeout_s: float,
    ) -> None:
        _check_host(host)
        _check_port(port)
        self._sock: socket.socket | None = sock
        self._host = host
        self._port = port
        self._timeout_s = _check_timeout_s(timeout_s)

    @classmethod
    def bind(cls, *, host: str, port: int, timeout_s: float) -> ChannelListener:
        _check_host(host)
        _check_port(port)
        timeout = _check_timeout_s(timeout_s)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(timeout)
            sock.bind((host, port))
            sock.listen(socket.SOMAXCONN)
        except OSError as exc:
            _close_socket(sock)
            raise ChannelError(f"channel listener bind failed at {host}:{port}: {exc}") from exc
        bound_host, bound_port = sock.getsockname()
        return cls(sock, host=bound_host, port=int(bound_port), timeout_s=timeout)

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def accept(self) -> Channel:
        sock = self._require_open()
        try:
            conn, addr = sock.accept()
        except TimeoutError as exc:
            raise ChannelTimeout(
                f"channel listener accept timed out after {self._timeout_s}s "
                f"at {self._host}:{self._port}"
            ) from exc
        except OSError as exc:
            raise ChannelError(
                f"channel listener accept failed at {self._host}:{self._port}: {exc}"
            ) from exc
        peer_host = addr[0] if addr else self._host
        peer_port = int(addr[1]) if addr else self._port
        return Channel(
            conn,
            host=peer_host,
            port=peer_port,
            timeout_s=self._timeout_s,
            role="server",
        )

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is None:
            return
        _close_socket(sock)

    def __enter__(self) -> ChannelListener:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require_open(self) -> socket.socket:
        if self._sock is None:
            raise ChannelError(
                f"channel listener is closed ({self._host}:{self._port})"
            )
        return self._sock


def poll_ready(channels: Sequence[Channel], *, timeout_s: float) -> list[Channel]:
    """Return channels that are readable now, or ``[]`` if the wait expires.

    This is the one timeout in the module that is **not** an error. Infer's
    batch window (plan 0.5) must fire with whatever ``ObsMsg`` arrived —
    waiting for a full ``num_envs`` is a straggler barrier because each slot's
    ``take_action`` costs 0.25–0.87 s. An empty list means "window closed, go
    with the current batch" (possibly empty), not "IPC failed". ``get`` /
    ``connect`` / ``accept`` still raise ``ChannelTimeout``.

    ``timeout_s`` is required (iron law 1). ``0`` is a non-blocking poll.
    """
    timeout = _check_poll_timeout_s(timeout_s)
    unique: list[tuple[int, Channel]] = []
    seen: set[int] = set()
    for channel in channels:
        fd = channel.fileno()
        if fd in seen:
            continue
        seen.add(fd)
        unique.append((fd, channel))
    if not unique:
        return []
    try:
        ready_fds, _, _ = select.select([fd for fd, _ in unique], [], [], timeout)
    except OSError as exc:
        raise ChannelError(f"poll_ready failed: {exc}") from exc
    ready = set(ready_fds)
    return [channel for fd, channel in unique if fd in ready]


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ChannelClosed(
                f"channel peer closed during recv ({len(buf)}/{nbytes} bytes)"
            )
        buf.extend(chunk)
    return bytes(buf)


def _close_socket(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


def _check_host(host: str) -> None:
    if not isinstance(host, str) or not host.strip():
        raise ValueError(f"host must be a non-empty str, got {host!r}")


def _check_port(port: int) -> None:
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError(f"port must be int, got {port!r}")
    if port < 0 or port > 65535:
        raise ValueError(f"port must be in [0, 65535], got {port}")


def _check_timeout_s(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError(f"timeout_s must be int or float, got {timeout_s!r}")
    value = float(timeout_s)
    if value <= 0.0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")
    return value


def _check_poll_timeout_s(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError(f"timeout_s must be int or float, got {timeout_s!r}")
    value = float(timeout_s)
    if value < 0.0:
        raise ValueError(f"poll_ready timeout_s must be >= 0, got {timeout_s!r}")
    return value
