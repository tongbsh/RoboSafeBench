"""RoboCasa backend capability declaration (kitchen wheeled mobile)."""

from __future__ import annotations

from dynisafety.core.protocols import BackendCapabilities

ROBOCASA_CAPABILITIES = BackendCapabilities(
    levels=(0, 1, 2, 3),
    has_force_measurement=True,
    has_topple_detection=True,
    force_source="mujoco_mj_contactForce",
)
