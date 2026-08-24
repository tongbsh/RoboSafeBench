"""Shared remote transports for VLA policies (HTTP / WebSocket / ZMQ)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from dynisafety.policies.base import ModelConfig
from dynisafety.policies.transports.http import HttpTransport
from dynisafety.policies.transports.websocket import WebSocketCodec, WebSocketTransport
from dynisafety.policies.transports.zmq import ZmqTransport


class Transport(Protocol):
    def reset(self, payload: Mapping[str, Any] | None = None) -> None: ...

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...

    def health(self) -> bool: ...


def _require_extra(cfg: ModelConfig, key: str) -> Any:
    if key not in cfg.extra:
        raise ValueError(
            f"ModelConfig.extra missing required key {key!r} for "
            f"transport={cfg.transport!r} (model={cfg.name!r})"
        )
    return cfg.extra[key]


def _build_http(cfg: ModelConfig, *, host: str, port: int) -> HttpTransport:
    if cfg.server_health_path is None or cfg.server_startup_timeout_s is None:
        raise ValueError(
            f"model {cfg.name!r}: http transport requires server.health_path and "
            "server.startup_timeout_s"
        )
    act_timeout_s = _require_extra(cfg, "act_timeout_s")
    reset_path = _require_extra(cfg, "reset_path")
    act_path = _require_extra(cfg, "act_path")
    if isinstance(act_timeout_s, bool) or not isinstance(act_timeout_s, int):
        raise ValueError(f"extra.act_timeout_s must be int, got {act_timeout_s!r}")
    if not isinstance(reset_path, str) or not reset_path.strip():
        raise ValueError(f"extra.reset_path must be non-empty str, got {reset_path!r}")
    if not isinstance(act_path, str) or not act_path.strip():
        raise ValueError(f"extra.act_path must be non-empty str, got {act_path!r}")
    base_url = f"http://{host}:{port}"
    transport = HttpTransport(
        base_url=base_url,
        health_path=cfg.server_health_path,
        startup_timeout_s=cfg.server_startup_timeout_s,
        act_timeout_s=act_timeout_s,
        reset_path=reset_path,
        act_path=act_path,
    )
    transport.wait_until_ready()
    return transport


def _build_websocket(cfg: ModelConfig, *, host: str, port: int) -> WebSocketTransport:
    # Lazy codec imports so optional deps (websockets/msgpack/openpi) stay optional.
    from dynisafety.policies.transports.codecs.lingbot import (  # noqa: PLC0415
        LingBotCodec,
    )
    from dynisafety.policies.transports.codecs.openpi import (  # noqa: PLC0415
        OpenPiCodec,
    )
    from dynisafety.policies.transports.codecs.starvla import (  # noqa: PLC0415
        StarVLACodec,
    )
    from dynisafety.policies.transports.codecs.xiaomi import (  # noqa: PLC0415
        XiaomiCodec,
    )

    codec_name = _require_extra(cfg, "ws_codec")
    if not isinstance(codec_name, str) or not codec_name.strip():
        raise ValueError(f"extra.ws_codec must be a non-empty str, got {codec_name!r}")
    codec: WebSocketCodec
    if codec_name == "openpi":
        client_src = _require_extra(cfg, "openpi_client_src")
        if not isinstance(client_src, str) or not client_src.strip():
            raise ValueError(
                f"extra.openpi_client_src must be non-empty str, got {client_src!r}"
            )
        if cfg.server_startup_timeout_s is None:
            raise ValueError(
                f"model {cfg.name!r}: openpi ws_codec requires server.startup_timeout_s"
            )
        codec = OpenPiCodec(
            client_src=client_src,
            connect_timeout_s=int(cfg.server_startup_timeout_s),
        )
    elif codec_name == "starvla":
        if cfg.server_root is None or not str(cfg.server_root).strip():
            raise ValueError(
                f"model {cfg.name!r}: starvla ws_codec requires server.root"
            )
        codec = StarVLACodec(repo_root=str(cfg.server_root))
    elif codec_name == "lingbot":
        if "api_key" not in cfg.extra:
            raise ValueError(
                f"model {cfg.name!r}: lingbot ws_codec requires extra.api_key "
                "(use null/None when unused)"
            )
        api_key = cfg.extra["api_key"]
        connect_timeout_s = _require_extra(cfg, "connect_timeout_s")
        if isinstance(connect_timeout_s, bool) or not isinstance(connect_timeout_s, int):
            raise ValueError(
                f"extra.connect_timeout_s must be int, got {connect_timeout_s!r}"
            )
        codec = LingBotCodec(connect_timeout_s=connect_timeout_s, api_key=api_key)
    elif codec_name == "xiaomi":
        if cfg.server_startup_timeout_s is None:
            raise ValueError(
                f"model {cfg.name!r}: xiaomi ws_codec requires server.startup_timeout_s"
            )
        codec = XiaomiCodec(connect_timeout_s=int(cfg.server_startup_timeout_s))
    else:
        raise ValueError(
            f"unknown extra.ws_codec {codec_name!r}; "
            "expected one of: openpi, starvla, lingbot, xiaomi"
        )
    return WebSocketTransport(host=host, port=port, codec=codec)


def _build_zmq(cfg: ModelConfig, *, host: str, port: int) -> ZmqTransport:
    if cfg.server_startup_timeout_s is None:
        raise ValueError(
            f"model {cfg.name!r}: zmq transport requires server.startup_timeout_s"
        )
    if cfg.server_root is None or not str(cfg.server_root).strip():
        raise ValueError(f"model {cfg.name!r}: zmq transport requires server.root")
    return ZmqTransport(
        host=host,
        port=port,
        startup_timeout_s=cfg.server_startup_timeout_s,
        gr00t_root=str(cfg.server_root),
    )


def build_transport(cfg: ModelConfig, *, host: str, port: int) -> Transport:
    """Construct a transport for *cfg*.

    ``host`` / ``port`` are injected by serve/sweep — never taken from code
    defaults. ``transport == "local"`` is rejected here (no remote session).
    """
    if not isinstance(host, str) or not host.strip():
        raise ValueError(f"host must be a non-empty str, got {host!r}")
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError(f"port must be int, got {port!r}")

    kind = cfg.transport
    if kind == "local":
        raise ValueError(
            f"build_transport: model {cfg.name!r} has transport='local' "
            "(no remote transport to build)"
        )
    if kind == "http":
        return _build_http(cfg, host=host, port=port)
    if kind == "websocket":
        return _build_websocket(cfg, host=host, port=port)
    if kind == "zmq":
        return _build_zmq(cfg, host=host, port=port)
    raise ValueError(
        f"unknown transport {kind!r}; expected local|http|websocket|zmq"
    )


def __getattr__(name: str) -> Any:
    # Lazy re-exports for codecs (optional heavy deps).
    if name in {"LingBotCodec", "OpenPiCodec", "StarVLACodec"}:
        import dynisafety.policies.transports.codecs as codecs  # noqa: PLC0415

        return getattr(codecs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HttpTransport",
    "LingBotCodec",
    "OpenPiCodec",
    "StarVLACodec",
    "Transport",
    "WebSocketCodec",
    "WebSocketTransport",
    "ZmqTransport",
    "build_transport",
]
