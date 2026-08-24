"""Assemble level controllers from explicit kwargs (no business defaults)."""

from __future__ import annotations

from typing import Any

from dynisafety.backends.robotwin.capabilities import ROBOTWIN_CAPABILITIES
from dynisafety.backends.robotwin.disturbance.l1 import SuddenDropController
from dynisafety.backends.robotwin.disturbance.l2 import SlidingObstacleController
from dynisafety.backends.robotwin.disturbance.l3 import SharedReachingController


def build_controller(level: int, **kwargs: Any) -> Any:
    """Construct L1/L2/L3 controller. ``level`` must be in capabilities."""
    if level not in ROBOTWIN_CAPABILITIES.levels or level == 0:
        raise ValueError(
            f"build_controller level must be in {tuple(x for x in ROBOTWIN_CAPABILITIES.levels if x > 0)}, "
            f"got {level}"
        )
    if level == 1:
        return SuddenDropController(**kwargs)
    if level == 2:
        return SlidingObstacleController(**kwargs)
    if level == 3:
        return SharedReachingController(**kwargs)
    raise ValueError(f"unsupported disturbance level={level}")
