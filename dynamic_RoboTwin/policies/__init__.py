"""Policy registry for DyniSafety-RoboTwin.

Mirrors the pattern from ``dynamic_RLbench/policies``: each VLA model lives in
its own module and self-registers via :func:`register_policy`. The
:mod:`dynamic_RoboTwin.main_inference` driver looks the chosen model up by
``--policy_name`` and delegates the per-episode inference loop to the
returned :class:`~policies.base_policy.RoboTwinPolicy` instance.

Adding a new model
------------------
1. Create ``policies/my_model_policy.py`` implementing
   :class:`~policies.base_policy.RoboTwinPolicy`.
2. Either decorate the class with ``@register_policy("my_model")`` or call
   ``register_policy("my_model", MyModelPolicy)`` at module load time.
3. Import the module from :func:`_load_builtin_policies` below so it
   self-registers when the package is first imported.
"""

from __future__ import annotations

from typing import Callable, List, Type

from policies.base_policy import RoboTwinPolicy

_REGISTRY: dict[str, Type[RoboTwinPolicy]] = {}


def register_policy(
    name: str,
    cls: Type[RoboTwinPolicy] | None = None,
) -> Callable | Type[RoboTwinPolicy]:
    """Register a :class:`RoboTwinPolicy` subclass under *name*.

    Usable as plain function or class decorator (matches dynamic_RLbench)."""
    if cls is None:
        def _decorator(klass: Type[RoboTwinPolicy]) -> Type[RoboTwinPolicy]:
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


def get_policy_class(name: str) -> Type[RoboTwinPolicy]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown policy '{name}'. Available policies: {list_policies()}. "
            "Make sure the policy module is imported (e.g. via "
            "_load_builtin_policies) before calling get_policy_class()."
        )
    return _REGISTRY[name]


def list_policies() -> List[str]:
    return sorted(_REGISTRY.keys())


def _load_builtin_policies() -> None:
    """Import built-in policy modules so they self-register.

    Each import is wrapped in try/except so that a missing optional
    dependency for one model (e.g. ``lerobot`` for InternVLA-A1) does not
    block users that only want to evaluate the other model.
    """
    import importlib

    _BUILTIN = [
        "policies.internvla_a1_policy",
        "policies.fastwam_policy",
        "policies.lingbot_va_policy",
        "policies.starvla_policy",
        "policies.xvla_policy",
    ]
    for module_name in _BUILTIN:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            import warnings
            warnings.warn(
                f"Skipping built-in policy module '{module_name}': {exc}",
                stacklevel=2,
            )


_load_builtin_policies()


__all__ = [
    "RoboTwinPolicy",
    "register_policy",
    "get_policy_class",
    "list_policies",
]
