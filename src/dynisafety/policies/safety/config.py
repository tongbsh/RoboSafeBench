"""Load Oracle Stop-Wait-Resume config from YAML (iron law 1)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dynisafety.policies.safety.types import OracleMode

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ORACLE_CONFIG_PATH = (
    _REPO_ROOT / "configs" / "safety" / "oracle_stop_wait_resume.yaml"
)


@dataclass(frozen=True)
class OracleStopActionConfig:
    strategy: str
    chunk_size: int
    default_gripper_hold: float
    gripper_indices_by_space: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class OracleAccountingConfig:
    policy_infer_steps_key: str
    oracle_pause_steps_key: str


@dataclass(frozen=True)
class OracleConfig:
    enabled: bool
    mode: OracleMode
    observation_extra_key: str
    fail_closed_on_missing_observation: bool
    active_phases: frozenset[str]
    resume_phases: frozenset[str]
    distance_stop_m: float
    distance_resume_m: float
    closing_speed_stop_mps: float
    stop_action: OracleStopActionConfig
    accounting: OracleAccountingConfig

    def thresholds_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "distance_stop_m": self.distance_stop_m,
            "distance_resume_m": self.distance_resume_m,
            "closing_speed_stop_mps": self.closing_speed_stop_mps,
            "active_phases": sorted(self.active_phases),
            "resume_phases": sorted(self.resume_phases),
            "fail_closed_on_missing_observation": self.fail_closed_on_missing_observation,
        }


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Oracle config missing required key {key!r}")
    return mapping[key]


def _require_bool(mapping: Mapping[str, Any], key: str) -> bool:
    value = _require(mapping, key)
    if not isinstance(value, bool):
        raise ValueError(f"Oracle config {key!r} must be bool, got {value!r}")
    return value


def _require_float(mapping: Mapping[str, Any], key: str) -> float:
    value = _require(mapping, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Oracle config {key!r} must be float, got {value!r}")
    return float(value)


def _require_nonempty_str(mapping: Mapping[str, Any], key: str) -> str:
    value = _require(mapping, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Oracle config {key!r} must be non-empty str, got {value!r}")
    return value


def _require_phase_set(mapping: Mapping[str, Any], key: str) -> frozenset[str]:
    raw = _require(mapping, key)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"Oracle config {key!r} must be a non-empty list")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Oracle config {key!r} entries must be non-empty str")
        out.append(item.strip().lower())
    return frozenset(out)


def _parse_stop_action(raw: Mapping[str, Any]) -> OracleStopActionConfig:
    strategy = _require_nonempty_str(raw, "strategy")
    chunk_size = _require(raw, "chunk_size")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"stop_action.chunk_size must be int >= 1, got {chunk_size!r}")
    default_gripper = _require_float(raw, "default_gripper_hold")
    indices_raw = _require(raw, "gripper_indices_by_space")
    if not isinstance(indices_raw, Mapping) or not indices_raw:
        raise ValueError("stop_action.gripper_indices_by_space must be a non-empty mapping")
    indices: dict[str, tuple[int, ...]] = {}
    for space, vals in indices_raw.items():
        if not isinstance(space, str) or not space.strip():
            raise ValueError("gripper_indices_by_space keys must be non-empty str")
        if not isinstance(vals, (list, tuple)) or not vals:
            raise ValueError(
                f"gripper_indices_by_space[{space!r}] must be a non-empty list of ints"
            )
        parsed: list[int] = []
        for v in vals:
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(
                    f"gripper_indices_by_space[{space!r}] entries must be int >= 0"
                )
            parsed.append(v)
        indices[space] = tuple(parsed)
    return OracleStopActionConfig(
        strategy=strategy,
        chunk_size=int(chunk_size),
        default_gripper_hold=default_gripper,
        gripper_indices_by_space=indices,
    )


def _parse_accounting(raw: Mapping[str, Any]) -> OracleAccountingConfig:
    return OracleAccountingConfig(
        policy_infer_steps_key=_require_nonempty_str(raw, "policy_infer_steps_key"),
        oracle_pause_steps_key=_require_nonempty_str(raw, "oracle_pause_steps_key"),
    )


def oracle_config_from_mapping(data: Mapping[str, Any]) -> OracleConfig:
    """Build :class:`OracleConfig` from a loaded mapping (no implicit defaults)."""
    if not isinstance(data, Mapping):
        raise TypeError(f"expected mapping, got {type(data)!r}")
    root = data.get("oracle", data)
    if not isinstance(root, Mapping):
        raise ValueError(f"oracle config root must be a mapping, got {type(root)!r}")

    mode_raw = _require_nonempty_str(root, "mode").strip().lower()
    try:
        mode = OracleMode(mode_raw)
    except ValueError as exc:
        raise ValueError(
            f"oracle.mode must be one of {[m.value for m in OracleMode]}, got {mode_raw!r}"
        ) from exc

    stop_raw = _require(root, "stop_action")
    if not isinstance(stop_raw, Mapping):
        raise ValueError("oracle.stop_action must be a mapping")
    accounting_raw = _require(root, "accounting")
    if not isinstance(accounting_raw, Mapping):
        raise ValueError("oracle.accounting must be a mapping")

    return OracleConfig(
        enabled=_require_bool(root, "enabled"),
        mode=mode,
        observation_extra_key=_require_nonempty_str(root, "observation_extra_key"),
        fail_closed_on_missing_observation=_require_bool(
            root, "fail_closed_on_missing_observation"
        ),
        active_phases=_require_phase_set(root, "active_phases"),
        resume_phases=_require_phase_set(root, "resume_phases"),
        distance_stop_m=_require_float(root, "distance_stop_m"),
        distance_resume_m=_require_float(root, "distance_resume_m"),
        closing_speed_stop_mps=_require_float(root, "closing_speed_stop_mps"),
        stop_action=_parse_stop_action(stop_raw),
        accounting=_parse_accounting(accounting_raw),
    )


def load_oracle_config(path: str | Path | None = None) -> OracleConfig:
    """Load oracle config from YAML. Defaults to ``configs/safety/oracle_stop_wait_resume.yaml``."""
    cfg_path = Path(path) if path is not None else DEFAULT_ORACLE_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Oracle config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Oracle config root must be a mapping, got {type(raw)!r}")
    return oracle_config_from_mapping(raw)
