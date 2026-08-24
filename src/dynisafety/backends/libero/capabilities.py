"""LIBERO backend capability declaration (D1=B: levels 0–3 only)."""

from __future__ import annotations

from dynisafety.core.protocols import BackendCapabilities

LIBERO_CAPABILITIES = BackendCapabilities(
    levels=(0, 1, 2, 3),
    has_force_measurement=True,
    has_topple_detection=True,
    force_source="mujoco_mj_contactForce",
)
