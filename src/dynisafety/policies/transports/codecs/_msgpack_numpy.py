"""msgpack + numpy helpers for websocket codecs (adapted from openpi/lingbot clients)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

_msgpack: Any | None = None


def _mp() -> Any:
    global _msgpack
    if _msgpack is None:
        try:
            import msgpack as _m  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "msgpack is required for LingBot/openpi-style websocket codecs"
            ) from exc
        _msgpack = _m
    return _msgpack


def pack_array(obj: object) -> object:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj: object) -> object:
    if isinstance(obj, dict):
        if b"__ndarray__" in obj:
            return np.ndarray(
                buffer=obj[b"data"],
                dtype=np.dtype(obj[b"dtype"]),
                shape=obj[b"shape"],
            )
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class _LazyPacker:
    """Packer proxy so importing codecs does not require msgpack until used."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _mp().Packer(default=pack_array, *args, **kwargs)

    def pack(self, obj: Any) -> bytes:
        return _mp().Packer(default=pack_array).pack(obj)


def packb(obj: Any, **kwargs: Any) -> bytes:
    return _mp().packb(obj, default=pack_array, **kwargs)


def unpackb(data: bytes, **kwargs: Any) -> Any:
    return _mp().unpackb(data, object_hook=unpack_array, **kwargs)


Packer: Callable[..., Any] = _LazyPacker()  # type: ignore[assignment]
