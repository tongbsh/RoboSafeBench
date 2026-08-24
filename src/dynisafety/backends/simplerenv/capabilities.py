"""SimplerEnv backend capability declaration."""

from __future__ import annotations

from dynisafety.core.protocols import BackendCapabilities

SIMPLERENV_CAPABILITIES = BackendCapabilities(
    levels=(0, 1, 2, 3),
    has_force_measurement=True,
    has_topple_detection=True,
    force_source="sapien_impulse",
)
