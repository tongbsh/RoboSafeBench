"""Load AEGIS CBF-QP config from YAML (iron law 1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dynisafety.policies.safety.aegis.types import MotionMode, SolverFailurePolicy

_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_AEGIS_CONFIG_PATH = _REPO_ROOT / "configs" / "safety" / "aegis.yaml"


@dataclass(frozen=True)
class AegisConfig:
    enabled: bool
    alpha: float
    max_vel: float
    cbf_h_gain: float
    dt: float
    action_to_u_scale: float
    u_to_action_scale: float
    cost_weight_motion: float
    cost_weight_z: float
    solver: str
    solver_fallback: tuple[str, ...]
    on_solver_failure: SolverFailurePolicy
    gripper_index: int
    slack_active_eps: float
    intervene_atol: float
    eef_axes_diag_default: tuple[float, float, float]
    motion_mode: MotionMode

    def thresholds_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "max_vel": self.max_vel,
            "cbf_h_gain": self.cbf_h_gain,
            "dt": self.dt,
            "action_to_u_scale": self.action_to_u_scale,
            "u_to_action_scale": self.u_to_action_scale,
            "cost_weight_motion": self.cost_weight_motion,
            "cost_weight_z": self.cost_weight_z,
            "solver": self.solver,
            "solver_fallback": list(self.solver_fallback),
            "on_solver_failure": self.on_solver_failure.value,
            "gripper_index": self.gripper_index,
            "slack_active_eps": self.slack_active_eps,
            "intervene_atol": self.intervene_atol,
            "eef_axes_diag_default": list(self.eef_axes_diag_default),
            "motion_mode": self.motion_mode.value,
        }


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"AEGIS config missing required key {key!r}")
    return mapping[key]


def _require_bool(mapping: Mapping[str, Any], key: str) -> bool:
    value = _require(mapping, key)
    if not isinstance(value, bool):
        raise ValueError(f"AEGIS config {key!r} must be bool, got {value!r}")
    return value


def _require_float(mapping: Mapping[str, Any], key: str) -> float:
    value = _require(mapping, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"AEGIS config {key!r} must be float, got {value!r}")
    return float(value)


def _require_positive_float(mapping: Mapping[str, Any], key: str) -> float:
    value = _require_float(mapping, key)
    if value <= 0.0:
        raise ValueError(f"AEGIS config {key!r} must be > 0, got {value!r}")
    return value


def _require_nonempty_str(mapping: Mapping[str, Any], key: str) -> str:
    value = _require(mapping, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"AEGIS config {key!r} must be non-empty str, got {value!r}")
    return value.strip()


def _require_int_ge0(mapping: Mapping[str, Any], key: str) -> int:
    value = _require(mapping, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"AEGIS config {key!r} must be int >= 0, got {value!r}")
    return value


def _require_str_list(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = _require(mapping, key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"AEGIS config {key!r} must be a list of str")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"AEGIS config {key!r} entries must be non-empty str")
        out.append(item.strip())
    return tuple(out)


def _require_axes3(mapping: Mapping[str, Any], key: str) -> tuple[float, float, float]:
    raw = _require(mapping, key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        raise ValueError(f"AEGIS config {key!r} must be a length-3 list of floats")
    vals: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"AEGIS config {key!r} entries must be float")
        f = float(item)
        if f <= 0.0:
            raise ValueError(f"AEGIS config {key!r} entries must be > 0")
        vals.append(f)
    return vals[0], vals[1], vals[2]


def aegis_config_from_mapping(data: Mapping[str, Any]) -> AegisConfig:
    """Build :class:`AegisConfig` from a loaded mapping (no implicit defaults)."""
    if not isinstance(data, Mapping):
        raise TypeError(f"expected mapping, got {type(data)!r}")
    root = data.get("aegis", data)
    if not isinstance(root, Mapping):
        raise ValueError(f"aegis config root must be a mapping, got {type(root)!r}")

    fail_raw = _require_nonempty_str(root, "on_solver_failure").lower()
    try:
        fail_policy = SolverFailurePolicy(fail_raw)
    except ValueError as exc:
        raise ValueError(
            f"aegis.on_solver_failure must be one of "
            f"{[p.value for p in SolverFailurePolicy]}, got {fail_raw!r}"
        ) from exc

    motion_raw = _require_nonempty_str(root, "motion_mode").lower()
    try:
        motion_mode = MotionMode(motion_raw)
    except ValueError as exc:
        raise ValueError(
            f"aegis.motion_mode must be one of "
            f"{[p.value for p in MotionMode]}, got {motion_raw!r}"
        ) from exc

    return AegisConfig(
        enabled=_require_bool(root, "enabled"),
        alpha=_require_float(root, "alpha"),
        max_vel=_require_positive_float(root, "max_vel"),
        cbf_h_gain=_require_positive_float(root, "cbf_h_gain"),
        dt=_require_positive_float(root, "dt"),
        action_to_u_scale=_require_positive_float(root, "action_to_u_scale"),
        u_to_action_scale=_require_positive_float(root, "u_to_action_scale"),
        cost_weight_motion=_require_positive_float(root, "cost_weight_motion"),
        cost_weight_z=_require_positive_float(root, "cost_weight_z"),
        solver=_require_nonempty_str(root, "solver"),
        solver_fallback=_require_str_list(root, "solver_fallback"),
        on_solver_failure=fail_policy,
        gripper_index=_require_int_ge0(root, "gripper_index"),
        slack_active_eps=_require_positive_float(root, "slack_active_eps"),
        intervene_atol=_require_positive_float(root, "intervene_atol"),
        eef_axes_diag_default=_require_axes3(root, "eef_axes_diag_default"),
        motion_mode=motion_mode,
    )


def load_aegis_config(path: str | Path | None = None) -> AegisConfig:
    """Load AEGIS config from YAML. Defaults to ``configs/safety/aegis.yaml``."""
    cfg_path = Path(path) if path is not None else DEFAULT_AEGIS_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"AEGIS config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"AEGIS config root must be a mapping, got {type(raw)!r}")
    return aegis_config_from_mapping(raw)


def aegis_config_with_enabled(config: AegisConfig, enabled: bool) -> AegisConfig:
    """Return a copy of ``config`` with ``enabled`` overridden."""
    return AegisConfig(
        enabled=bool(enabled),
        alpha=config.alpha,
        max_vel=config.max_vel,
        cbf_h_gain=config.cbf_h_gain,
        dt=config.dt,
        action_to_u_scale=config.action_to_u_scale,
        u_to_action_scale=config.u_to_action_scale,
        cost_weight_motion=config.cost_weight_motion,
        cost_weight_z=config.cost_weight_z,
        solver=config.solver,
        solver_fallback=config.solver_fallback,
        on_solver_failure=config.on_solver_failure,
        gripper_index=config.gripper_index,
        slack_active_eps=config.slack_active_eps,
        intervene_atol=config.intervene_atol,
        eef_axes_diag_default=config.eef_axes_diag_default,
        motion_mode=config.motion_mode,
    )
