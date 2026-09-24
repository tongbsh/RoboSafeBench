"""Naming convention for safety-layer arms.

A safety layer is evaluated as ``{base}_oracle_{layer}`` — ``pi05_oracle_swr_aegis``
is π0.5 with the SWR/AEGIS shield bolted on. It is a filter on top of a
policy, not a policy: ranking it against bare VLAs compares different kinds
of object, and its own pre-onset is not a null for divergence thresholds (a
shield that stops on sight of a parked L2/L3 obstacle inflates the fitted
scale). Both ``report`` and ``analysis`` need the same two questions answered,
so they live here rather than in either package.
"""

from __future__ import annotations

SAFETY_LAYER_MARKER = "_oracle_"


def is_safety_layer_model(model: str) -> bool:
    return SAFETY_LAYER_MARKER in str(model)


def null_parent_model(model: str) -> str | None:
    """Bare VLA a safety-layer arm sits on top of.

    ``pi05_oracle_swr_aegis`` → ``pi05``. ``None`` when ``model`` is not a
    safety-layer name.
    """
    name = str(model)
    if SAFETY_LAYER_MARKER not in name:
        return None
    parent = name.split(SAFETY_LAYER_MARKER, 1)[0]
    return parent or None


__all__ = ["SAFETY_LAYER_MARKER", "is_safety_layer_model", "null_parent_model"]
