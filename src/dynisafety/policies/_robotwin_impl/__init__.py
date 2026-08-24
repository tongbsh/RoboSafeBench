"""RoboTwin-side policy implementations relocated from ``dynamic_RoboTwin/policies``.

Adapters import concrete modules from this package. The old live registry
entrypoints (``*_inference.py`` / shell) are removed in M7 S3j.
"""

from __future__ import annotations

from collections.abc import Callable

from .base_policy import RoboTwinPolicy

_REGISTRY: dict[str, type[RoboTwinPolicy]] = {}


def register_policy(
    name: str,
    cls: type[RoboTwinPolicy] | None = None,
) -> Callable | type[RoboTwinPolicy]:
    if cls is None:

        def _decorator(klass: type[RoboTwinPolicy]) -> type[RoboTwinPolicy]:
            register_policy(name, klass)
            return klass

        return _decorator

    if not (isinstance(cls, type) and issubclass(cls, RoboTwinPolicy)):
        raise TypeError(
            f"register_policy expects a RoboTwinPolicy subclass, got {cls!r}"
        )
    if name in _REGISTRY:
        import warnings

        warnings.warn(
            f"Policy '{name}' is already registered; overwriting with {cls}.",
            stacklevel=2,
        )
    _REGISTRY[name] = cls
    return cls


def get_policy_class(name: str) -> type[RoboTwinPolicy]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown policy '{name}'. Available policies: {list_policies()}."
        )
    return _REGISTRY[name]


def list_policies() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = [
    "RoboTwinPolicy",
    "register_policy",
    "get_policy_class",
    "list_policies",
]
