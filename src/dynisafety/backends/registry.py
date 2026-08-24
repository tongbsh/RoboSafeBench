"""Backend registry populate helpers (mirrors ``policies.registry.populate_policies``).

Backends must not import ``cli.eval``; eval owns the singleton and calls
:func:`populate_backends` after import.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dynisafety.core.registry import Registry

# Import failures for optional simulator deps (e.g. mujoco in internvla_a1).
# Cleared/updated on each populate_backends call; inspect for diagnostics.
BACKEND_IMPORT_ERRORS: dict[str, BaseException] = {}


def _try_register(
    registry: Registry[Any],
    name: str,
    loader: Callable[[], type],
) -> None:
    """Register one backend; isolate ImportError so other backends still load."""
    if name in registry.available():
        return
    try:
        cls = loader()
    except ImportError as exc:
        BACKEND_IMPORT_ERRORS[name] = exc
        return
    BACKEND_IMPORT_ERRORS.pop(name, None)
    registry.register(name)(cls)


def populate_backends(registry: Registry[Any]) -> None:
    """Register production backends independently (one ImportError must not block others)."""
    BACKEND_IMPORT_ERRORS.clear()

    def _libero() -> type:
        from dynisafety.backends.libero.backend import LiberoBackend

        return LiberoBackend

    def _simplerenv() -> type:
        from dynisafety.backends.simplerenv.backend import SimplerEnvBackend

        return SimplerEnvBackend

    def _robotwin() -> type:
        from dynisafety.backends.robotwin.backend import RoboTwinBackend

        return RoboTwinBackend

    def _robocasa() -> type:
        from dynisafety.backends.robocasa.backend import RoboCasaBackend

        return RoboCasaBackend

    _try_register(registry, "libero", _libero)
    _try_register(registry, "simplerenv", _simplerenv)
    _try_register(registry, "robotwin", _robotwin)
    _try_register(registry, "robocasa", _robocasa)


def register_into(registry: Registry[Any]) -> None:
    """Alias for :func:`populate_backends`."""
    populate_backends(registry)
