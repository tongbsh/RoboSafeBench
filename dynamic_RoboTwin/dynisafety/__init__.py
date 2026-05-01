"""Top-level shell for the RoboTwin DyniSafety integration.

This package merely re-exports the canonical implementation that lives at
``dynamic_RoboTwin/RoboTwin/envs/dynisafety``. Keeping the shell here lets
external code import from a stable, layout-agnostic location:

    from dynisafety import RoboTwinDyniSafetyConfig, RoboTwinDyniSafetyManager

without having to know that the underlying classes live under the bundled
RoboTwin checkout's ``envs/`` package. The :mod:`dynamic_RoboTwin.main_inference`
module bootstraps the necessary ``sys.path`` entries (``dynamic_RoboTwin/`` and
``dynamic_RoboTwin/RoboTwin/``) so this re-export resolves transparently.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROBOTWIN_ROOT = _HERE.parent / "RoboTwin"
if str(_ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOTWIN_ROOT))

from envs.dynisafety import (  # noqa: E402  (sys.path bootstrap above)
    RoboTwinDyniSafetyConfig,
    RoboTwinDyniSafetyManager,
)

__all__ = ["RoboTwinDyniSafetyConfig", "RoboTwinDyniSafetyManager"]
