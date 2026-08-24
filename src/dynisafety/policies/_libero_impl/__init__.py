"""LIBERO-side policy implementations relocated from ``dynisafety_libero/policies``.

Only the implementations still reached through a ``dynisafety.policies.adapters``
entry live here: ``openvla_oft_rlinf`` and ``univla_local``. The other legacy
LIBERO policies were superseded by first-class adapters and are archived under
``old_cache/dynisafety_libero/``.

Mirrors the ``_robotwin_impl`` layout established in M7: adapters import the
concrete module, there is no second registry.
"""

from __future__ import annotations

from .base_policy import VLAPolicy

__all__ = ["VLAPolicy"]
