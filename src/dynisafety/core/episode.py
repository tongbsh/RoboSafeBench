"""Unified episode-level record (§3.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

SCHEMA_VERSION = "2.0"

# 2.0 dropped the reactivity trio (reaction_latency_steps / velocity_ratio /
# behavior_class). Their replacements live under ``diagnostics`` and are not
# name-compatible: the old ``behavior_class`` vocabulary
# (collide/freeze/cautious/ignore) means something different from the new
# 9-class one, so silently reading a 1.x file would misreport rather than fail.
# See docs/user/behavior_diagnostics.md §3.1 and docs/user/migration.md.
_READABLE_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

# Defaulted on read so a record can be hand-built without diagnostics.
_OPTIONAL_FIELD_NAMES = frozenset({"diagnostics"})

_FIELD_NAMES = frozenset(
    {
        "schema_version",
        "run_id",
        "config_hash",
        "backend",
        "suite",
        "task_id",
        "task_name",
        "instruction",
        "model",
        "checkpoint",
        "level",
        "seed",
        "episode_index",
        "steps",
        "max_steps",
        "termination",
        "wall_time_s",
        "success",
        "success_step",
        "contact_steps",
        "contact_events",
        "first_contact_step",
        "contact_partners",
        "peak_force_n",
        "peak_impulse_ns",
        "cumulative_impulse_ns",
        "force_equiv_dt_s",
        "max_tilt_deg",
        "toppled",
        "video_path",
        "extra",
        "diagnostics",
    }
)


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-serializable Python builtins (no numpy scalars)."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # numpy scalar / array-like fallback without hard numpy dependency at import
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (ValueError, TypeError):
            pass
    raise TypeError(f"value is not JSON-serializable: {type(value)!r}")


@dataclass(frozen=True)
class EpisodeRecord:
    # --- 身份与溯源 ---
    schema_version: str
    run_id: str
    config_hash: str
    backend: str
    suite: str
    task_id: int
    task_name: str
    instruction: str
    model: str
    checkpoint: str
    level: int
    seed: int
    episode_index: int

    # --- rollout ---
    steps: int
    max_steps: int
    termination: str
    wall_time_s: float

    # --- 任务结果 ---
    success: bool
    success_step: int | None

    # --- 接触（扰动体 <-> 机器人/在手物体）---
    contact_steps: int
    contact_events: int
    first_contact_step: int | None
    contact_partners: dict[str, int]

    # --- 力 ---
    peak_force_n: float
    peak_impulse_ns: float
    cumulative_impulse_ns: float
    force_equiv_dt_s: float

    # --- 障碍物稳定性（仅 L1/L2 有意义）---
    max_tilt_deg: float | None
    toppled: bool | None

    # --- 产物 ---
    video_path: str | None

    # --- backend 私有字段 ---
    extra: dict[str, Any]

    # --- 行为诊断（反应/停止/规避/恢复）---
    # ``BehaviorDiagnostics.to_json_dict()`` 的固定 schema，或 None（诊断关闭 /
    # backend 未接线）。与自由格式的 ``extra`` 不同，本字段进正式报表；None 时
    # 表格渲染 n/a，绝不填 0。定义见 docs/user/behavior_diagnostics.md。
    diagnostics: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        return {key: _json_safe(value) for key, value in raw.items()}

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> EpisodeRecord:
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data)!r}")
        version = data.get("schema_version")
        if version is not None and str(version) not in _READABLE_SCHEMA_VERSIONS:
            raise ValueError(
                f"EpisodeRecord schema_version {str(version)!r} is not readable by this "
                f"build (expected {SCHEMA_VERSION!r}). Behavior diagnostics replaced the "
                "reactivity fields and the behavior_class vocabulary changed meaning, so "
                "old runs must be re-generated rather than reinterpreted. "
                "See docs/user/migration.md."
            )
        keys = set(data.keys())
        missing = (_FIELD_NAMES - _OPTIONAL_FIELD_NAMES) - keys
        unknown = keys - _FIELD_NAMES
        if missing:
            raise KeyError(f"EpisodeRecord missing keys: {sorted(missing)}")
        if unknown:
            raise KeyError(f"EpisodeRecord unknown keys: {sorted(unknown)}")
        kwargs = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        # Ensure mapping fields are plain dicts (and copy for frozen safety)
        kwargs["contact_partners"] = dict(kwargs["contact_partners"])
        kwargs["extra"] = dict(kwargs["extra"])
        diagnostics = kwargs.get("diagnostics")
        kwargs["diagnostics"] = dict(diagnostics) if diagnostics is not None else None
        return cls(**kwargs)
