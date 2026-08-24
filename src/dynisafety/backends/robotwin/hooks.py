"""Thin mount points for vendored ``Base_Task`` (patch surface).

All disturbance logic lives in ``disturbance.DisturbanceManager``; this module
only wires enable / disable / before_step / after_step.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _idle_info(task: Any) -> dict[str, Any]:
    return {
        "step_collided": False,
        "collision_steps": 0,
        "collision_events": 0,
        "dynamic_level": int(getattr(task, "dynamic_level", 0)),
    }


def snapshot_info(task: Any, *, step_collided: bool = False) -> dict[str, Any]:
    mgr = getattr(task, "dynisafety_manager", None)
    if mgr is None:
        return _idle_info(task)
    return mgr.build_info(step_collided=step_collided)


def update_info(task: Any, step_collided: bool = False) -> None:
    """Write dynisafety info for the finished take_action window, then clear peaks."""
    if hasattr(task, "info") and isinstance(task.info, dict):
        task.info["dynisafety"] = snapshot_info(task, step_collided=step_collided)
    mgr = getattr(task, "dynisafety_manager", None)
    if mgr is not None and hasattr(mgr, "reset_action_contact_acc"):
        mgr.reset_action_contact_acc()


def _ensure_table_surface_z(config: Any) -> None:
    """Fill ``table_surface_z`` from YAML when legacy dataclass omits it (#25)."""
    if hasattr(config, "table_surface_z"):
        return
    import yaml

    yaml_path = Path(__file__).resolve().parents[4] / "configs" / "backends" / "robotwin.yaml"
    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    try:
        z = data["scene"]["table_surface_z"]
    except (TypeError, KeyError) as exc:
        raise ValueError(
            f"config missing table_surface_z and {yaml_path} has no scene.table_surface_z"
        ) from exc
    setattr(config, "table_surface_z", float(z))


def disable_dynamic_disturbance(task: Any) -> None:
    mgr = getattr(task, "dynisafety_manager", None)
    if mgr is not None:
        mgr.destroy()
    task.dynisafety_manager = None
    task.dynamic_level = 0
    task.dynisafety_sim_step = 0
    update_info(task, False)


def enable_dynamic_disturbance(task: Any, config: Any) -> None:
    """Arm disturbance for ``task``. ``config`` must expose required attributes."""
    from dynisafety.backends.robotwin.disturbance.manager import DisturbanceManager

    disable_dynamic_disturbance(task)
    if config is None:
        raise ValueError(
            "enable_dynamic_disturbance requires an explicit config object (no Python defaults)"
        )
    if isinstance(config, Mapping):
        # Attribute access for DisturbanceManager; callers must supply every
        # required field (YAML / config_bridge). No legacy dataclass defaults.
        from types import SimpleNamespace

        config = SimpleNamespace(**dict(config))
    # ``SimpleNamespace`` and other attribute objects from ``config_bridge`` pass
    # through unchanged (no Mapping conversion).

    _ensure_table_surface_z(config)

    level = int(config.dynamic_level)
    task.dynamic_level = level
    task.dynisafety_sim_step = 0
    if level <= 0:
        update_info(task, False)
        return

    task.dynisafety_manager = DisturbanceManager(task, config)
    task.dynisafety_manager.arm()
    update_info(task, False)


def before_step(task: Any, sim_step: int) -> None:
    mgr = getattr(task, "dynisafety_manager", None)
    if mgr is not None:
        mgr.before_step(sim_step)


def after_step(task: Any, sim_step: int) -> bool:
    """Run manager after_step; advance ``task.dynisafety_sim_step`` when armed."""
    mgr = getattr(task, "dynisafety_manager", None)
    if mgr is None:
        return False
    hit = bool(mgr.after_step(sim_step))
    task.dynisafety_sim_step = int(sim_step) + 1
    return hit
