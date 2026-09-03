"""Assemble RoboCasa disturbance controllers from YAML (no Python defaults)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dynisafety.backends.robocasa.capabilities import ROBOCASA_CAPABILITIES
from dynisafety.backends.robocasa.disturbance._support import (
    l3_meta,
    set_actor_contact_enabled,
    snapshot_geom_contact,
    teleport_actor_hidden,
    teleport_obstacle_hidden,
)
from dynisafety.backends.robocasa.disturbance.l1 import FrontalSuddenAppearController
from dynisafety.backends.robocasa.disturbance.l2 import BaseDirectedApproachController
from dynisafety.backends.robocasa.disturbance.l3 import ApproachingHumanController
from dynisafety.backends.robocasa.disturbance.pnp import PnPHazardController, build_pnp_controller
from dynisafety.backends.robocasa.env import RoboCasaEnvHandle, hide_l0_obstacle_actor
from dynisafety.config.schema import MISSING, RunConfig
from dynisafety.core.disturbance import require_configured
from dynisafety.core.disturbance import require_map_key as _require_map_key

_require = require_configured

RoboCasaController = (
    FrontalSuddenAppearController
    | BaseDirectedApproachController
    | ApproachingHumanController
    | PnPHazardController
)


def _require_l1_key(l1: Mapping[str, Any], key: str) -> Any:
    return _require_map_key(l1, key, where="extra.robocasa.l1")


def _require_l2_key(l2: Mapping[str, Any], key: str) -> Any:
    return _require_map_key(l2, key, where="extra.robocasa.l2")


def _require_l3_key(l3: Mapping[str, Any], key: str) -> Any:
    return _require_map_key(l3, key, where="extra.robocasa.l3")


def capture_obstacle_contact_template(handle: RoboCasaEnvHandle) -> None:
    """Snapshot geom contype before L0 hide, so L1/L2 spawn can restore physics."""
    env = handle.env
    meta = getattr(env, "_dynisafety_l12", None)
    if not isinstance(meta, dict):
        return
    if meta.get("geom_contact"):
        return
    meta["geom_contact"] = snapshot_geom_contact(env)


def capture_l3_contact_template(handle: RoboCasaEnvHandle) -> None:
    from dynisafety.backends.robocasa.disturbance._support import (
        snapshot_actor_geom_contact,
    )

    env = handle.env
    meta = getattr(env, "_dynisafety_l3", None)
    if not isinstance(meta, dict):
        return
    if meta.get("geom_contact"):
        return
    meta["geom_contact"] = snapshot_actor_geom_contact(env, meta)


def _l12_pool_and_mode(cfg: RunConfig) -> tuple[str, list[str]]:
    disturbance = _require(cfg.disturbance, "cfg.disturbance")
    obstacle = _require(disturbance.obstacle, "disturbance.obstacle")
    mode = obstacle.selection_mode
    if mode is MISSING:
        raise ValueError("disturbance.obstacle.selection_mode required for RoboCasa L1/L2")
    mode_text = str(mode).strip().lower()
    if mode_text not in ("fixed", "trajectory_order"):
        raise ValueError(
            "disturbance.obstacle.selection_mode must be 'fixed' or "
            f"'trajectory_order' for RoboCasa L1/L2, got {mode!r}"
        )
    pool = obstacle.pool
    if pool is MISSING or not pool:
        raise ValueError("disturbance.obstacle.pool required for RoboCasa L1/L2")
    names = [str(item).strip() for item in pool if str(item).strip()]
    if not names:
        raise ValueError("disturbance.obstacle.pool must contain at least one name")
    return mode_text, names


def _yaml_bool(raw: Any, *, where: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{where} must be a YAML bool, got {raw!r}")
    return raw


def _l3_visibility_cameras(extra_robocasa: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    extra_cams = extra_robocasa.get("extra_cameras")
    if extra_cams is not None:
        if not isinstance(extra_cams, Sequence) or isinstance(extra_cams, (str, bytes)):
            raise ValueError("extra.robocasa.extra_cameras must be a list of camera names")
        for i, name in enumerate(extra_cams):
            text = str(name).strip()
            if not text:
                raise ValueError(
                    f"extra.robocasa.extra_cameras[{i}] must be a non-empty str"
                )
            if text not in names:
                names.append(text)
    primary = extra_robocasa.get("camera")
    if not isinstance(primary, str) or not primary.strip():
        raise ValueError("extra.robocasa.camera missing for L3 visibility cameras")
    primary = primary.strip()
    if primary not in names:
        names.append(primary)
    return names


def _l3_camera_aspect(extra_robocasa: Mapping[str, Any]) -> float:
    if "policy_camera_width" in extra_robocasa or "policy_camera_height" in extra_robocasa:
        width = float(
            _require_map_key(
                extra_robocasa, "policy_camera_width", where="extra.robocasa"
            )
        )
        height = float(
            _require_map_key(
                extra_robocasa, "policy_camera_height", where="extra.robocasa"
            )
        )
    else:
        width = float(
            _require_map_key(extra_robocasa, "camera_width", where="extra.robocasa")
        )
        height = float(
            _require_map_key(extra_robocasa, "camera_height", where="extra.robocasa")
        )
    if height <= 0.0:
        raise ValueError(f"L3 camera height must be > 0, got {height}")
    return width / height


def build_controller(
    level: int,
    handle: RoboCasaEnvHandle,
    cfg: RunConfig,
    *,
    extra_robocasa: Mapping[str, Any],
    episode_index: int = 0,
) -> RoboCasaController | None:
    if str(handle.suite) == "robocasa_pnp":
        return build_pnp_controller(
            level, handle, extra_robocasa, episode_index=episode_index
        )
    if level not in ROBOCASA_CAPABILITIES.levels:
        raise ValueError(
            f"level={level} not in RoboCasa capabilities {ROBOCASA_CAPABILITIES.levels}"
        )
    if level == 0:
        hide_l0_obstacle_actor(handle.env)
        return None

    if level == 1:
        l1_extra = extra_robocasa.get("l1")
        if not isinstance(l1_extra, Mapping):
            raise ValueError("extra.robocasa.l1 missing from resolved YAML")
        half_raw = _require_l1_key(l1_extra, "half_size")
        half_list = [float(x) for x in half_raw]
        if len(half_list) != 3:
            raise ValueError(f"extra.robocasa.l1.half_size must be length 3, got {half_list!r}")
        capture_obstacle_contact_template(handle)
        mode, pool = _l12_pool_and_mode(cfg)
        ctrl = FrontalSuddenAppearController(
            handle,
            goal_clearance_m=float(_require_l1_key(l1_extra, "goal_clearance_m")),
            trigger_dist=float(_require_l1_key(l1_extra, "trigger_dist")),
            retry_step=float(_require_l1_key(l1_extra, "retry_step")),
            n_attempts=int(_require_l1_key(l1_extra, "n_attempts")),
            half_size=half_list,
            selection_mode=mode,
            pool=pool,
        )
        ctrl.reset(int(episode_index))
        teleport_obstacle_hidden(handle.env)
        return ctrl

    if level == 2:
        l2_extra = extra_robocasa.get("l2")
        if not isinstance(l2_extra, Mapping):
            raise ValueError("extra.robocasa.l2 missing from resolved YAML")
        mode, pool = _l12_pool_and_mode(cfg)
        half_raw = _require_l2_key(l2_extra, "half_size")
        half_list = [float(x) for x in half_raw]
        if len(half_list) != 3:
            raise ValueError(f"extra.robocasa.l2.half_size must be length 3, got {half_list!r}")
        disturbance = _require(cfg.disturbance, "cfg.disturbance")
        l2_clock = _require(disturbance.l2, "disturbance.l2")
        capture_obstacle_contact_template(handle)
        ctrl2 = BaseDirectedApproachController(
            handle,
            goal_clearance_m=float(_require_l2_key(l2_extra, "goal_clearance_m")),
            start_lateral=float(_require_l2_key(l2_extra, "start_lateral")),
            move_start_step=int(
                _require(l2_clock.move_start_step, "disturbance.l2.move_start_step")
            ),
            move_end_step=int(_require(l2_clock.move_end_step, "disturbance.l2.move_end_step")),
            retry_step=float(_require_l2_key(l2_extra, "retry_step")),
            n_attempts=int(_require_l2_key(l2_extra, "n_attempts")),
            min_half_span=float(_require_l2_key(l2_extra, "min_half_span")),
            trigger_dist=float(_require_l2_key(l2_extra, "trigger_dist")),
            flip_side=_yaml_bool(
                _require_l2_key(l2_extra, "flip_side"),
                where="extra.robocasa.l2.flip_side",
            ),
            half_size=half_list,
            selection_mode=mode,
            pool=pool,
        )
        ctrl2.reset(int(episode_index))
        teleport_obstacle_hidden(handle.env)
        return ctrl2

    if level == 3:
        hide_l0_obstacle_actor(handle.env)
        l3_extra = extra_robocasa.get("l3")
        if not isinstance(l3_extra, Mapping):
            raise ValueError("extra.robocasa.l3 missing from resolved YAML")
        disturbance = _require(cfg.disturbance, "cfg.disturbance")
        l3_clock = _require(disturbance.l3, "disturbance.l3")
        capture_l3_contact_template(handle)
        half_raw = _require_l3_key(l3_extra, "half_size")
        half_list = [float(x) for x in half_raw]
        if len(half_list) != 3:
            raise ValueError(f"extra.robocasa.l3.half_size must be length 3, got {half_list!r}")
        ctrl3 = ApproachingHumanController(
            handle,
            goal_clearance_m=float(_require_l3_key(l3_extra, "goal_clearance_m")),
            human_start_lateral=float(_require_l3_key(l3_extra, "human_start_lateral")),
            trigger_dist=float(_require_l3_key(l3_extra, "trigger_dist")),
            entry_steps=int(_require(l3_clock.entry_steps, "disturbance.l3.entry_steps")),
            dwell_steps=int(_require(l3_clock.dwell_steps, "disturbance.l3.dwell_steps")),
            approach_steps=int(_require_l3_key(l3_extra, "approach_steps")),
            withdraw_steps=int(_require_l3_key(l3_extra, "withdraw_steps")),
            flinch_dist=float(_require_l3_key(l3_extra, "flinch_dist")),
            flinch_steps=int(_require_l3_key(l3_extra, "flinch_steps")),
            retry_step=float(_require_l3_key(l3_extra, "retry_step")),
            n_attempts=int(_require_l3_key(l3_extra, "n_attempts")),
            flip_side=_yaml_bool(
                _require_l3_key(l3_extra, "flip_side"),
                where="extra.robocasa.l3.flip_side",
            ),
            half_size=half_list,
            min_preview_steps=int(_require_l3_key(l3_extra, "min_preview_steps")),
            close_approach_m=float(_require_l3_key(l3_extra, "close_approach_m")),
            visibility_cameras=_l3_visibility_cameras(extra_robocasa),
            camera_aspect=_l3_camera_aspect(extra_robocasa),
        )
        ctrl3.reset(int(episode_index))
        teleport_actor_hidden(handle.env, l3_meta(handle.env))
        set_actor_contact_enabled(handle.env, l3_meta(handle.env), enabled=False)
        return ctrl3

    raise ValueError(f"unsupported RoboCasa disturbance level={level}")
