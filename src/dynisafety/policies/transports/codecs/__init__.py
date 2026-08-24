"""WebSocket framing codecs (openpi / starvla / lingbot)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynisafety.policies.transports.codecs.lingbot import LingBotCodec
    from dynisafety.policies.transports.codecs.openpi import OpenPiCodec
    from dynisafety.policies.transports.codecs.starvla import StarVLACodec

__all__ = ["LingBotCodec", "OpenPiCodec", "StarVLACodec"]


def __getattr__(name: str) -> type:
    if name == "LingBotCodec":
        from dynisafety.policies.transports.codecs.lingbot import (  # noqa: PLC0415
            LingBotCodec,
        )

        return LingBotCodec
    if name == "OpenPiCodec":
        from dynisafety.policies.transports.codecs.openpi import (  # noqa: PLC0415
            OpenPiCodec,
        )

        return OpenPiCodec
    if name == "StarVLACodec":
        from dynisafety.policies.transports.codecs.starvla import (  # noqa: PLC0415
            StarVLACodec,
        )

        return StarVLACodec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
