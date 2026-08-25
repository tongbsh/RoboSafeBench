"""Load ``extra.parallel_eval.*`` from an already-resolved DyniSafety config.

``extra`` is already an open map in the schema (``_OPEN_PREFIXES``), so this
block is legal without a new ``configs/runner/`` layer and still participates
in ``config_hash``. Every field is required: missing keys raise with the key
name. No Python defaults — numbers live only in YAML (iron law 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dynisafety.config.loader import ResolvedConfig
from dynisafety.policies.base import ModelConfig

_PREFIX = "extra.parallel_eval"

_BOOL_KEYS = ("enabled", "deterministic_rng")
_INT_KEYS = (
    "num_envs",
    "pipeline_stage_num",
    "batch_window_ms",
    "channel_port",
    "rng_seed",
)
_NUMBER_KEYS = ("channel_timeout_s",)
_STR_KEYS = (
    "infer_conda_env",
    "env_conda_env",
    "infer_device",
    "env_cuda_visible_devices",
    "channel_host",
    "torch_dtype",
)

REQUIRED_KEYS: tuple[str, ...] = (
    *_BOOL_KEYS,
    *_INT_KEYS,
    *_NUMBER_KEYS,
    *_STR_KEYS,
)


@dataclass(frozen=True)
class ParallelEvalConfig:
    """Resolved ``extra.parallel_eval`` block. All fields caller-supplied."""

    enabled: bool
    num_envs: int
    pipeline_stage_num: int
    batch_window_ms: int
    infer_conda_env: str
    env_conda_env: str
    infer_device: str
    env_cuda_visible_devices: str
    channel_host: str
    channel_port: int
    channel_timeout_s: float
    torch_dtype: str
    deterministic_rng: bool
    rng_seed: int

    @classmethod
    def from_extra(cls, extra: Mapping[str, Any]) -> ParallelEvalConfig:
        if not isinstance(extra, Mapping):
            raise TypeError(f"extra must be a mapping, got {type(extra)!r}")
        if "parallel_eval" not in extra:
            raise ValueError("extra missing required key 'parallel_eval'")
        block = extra["parallel_eval"]
        if not isinstance(block, Mapping):
            raise TypeError(
                f"{_PREFIX} must be a mapping, got {type(block)!r}"
            )
        missing = [key for key in REQUIRED_KEYS if key not in block]
        if missing:
            raise ValueError(f"{_PREFIX} missing required key {missing[0]!r}")
        return cls(
            enabled=_require_bool(block, "enabled"),
            num_envs=_require_int(block, "num_envs"),
            pipeline_stage_num=_require_int(block, "pipeline_stage_num"),
            batch_window_ms=_require_int(block, "batch_window_ms"),
            infer_conda_env=_require_nonempty_str(block, "infer_conda_env"),
            env_conda_env=_require_nonempty_str(block, "env_conda_env"),
            infer_device=_require_nonempty_str(block, "infer_device"),
            env_cuda_visible_devices=_require_nonempty_str(
                block, "env_cuda_visible_devices"
            ),
            channel_host=_require_nonempty_str(block, "channel_host"),
            channel_port=_require_int(block, "channel_port"),
            channel_timeout_s=_require_number(block, "channel_timeout_s"),
            torch_dtype=_require_nonempty_str(block, "torch_dtype"),
            deterministic_rng=_require_bool(block, "deterministic_rng"),
            rng_seed=_require_int(block, "rng_seed"),
        )

    @classmethod
    def from_resolved(cls, data: Mapping[str, Any]) -> ParallelEvalConfig:
        if not isinstance(data, Mapping):
            raise TypeError(f"resolved config must be a mapping, got {type(data)!r}")
        if "extra" not in data:
            raise ValueError("resolved config missing required key 'extra'")
        extra = data["extra"]
        if not isinstance(extra, Mapping):
            raise TypeError(
                f"resolved config key 'extra' must be a mapping, got {type(extra)!r}"
            )
        return cls.from_extra(extra)

    @classmethod
    def from_model_config(cls, cfg: ModelConfig) -> ParallelEvalConfig:
        return cls.from_extra(cfg.extra)


def load_parallel_eval_config(
    source: Mapping[str, Any] | ModelConfig | ResolvedConfig,
) -> ParallelEvalConfig:
    """Read ``extra.parallel_eval`` from resolved data, ModelConfig, or both."""
    if isinstance(source, ModelConfig):
        return ParallelEvalConfig.from_model_config(source)
    if isinstance(source, ResolvedConfig):
        return ParallelEvalConfig.from_resolved(source.data)
    if isinstance(source, Mapping):
        return ParallelEvalConfig.from_resolved(source)
    raise TypeError(
        "parallel_eval config source must be a mapping, ModelConfig, or "
        f"ResolvedConfig, got {type(source)!r}"
    )


def _require_key(block: Mapping[str, Any], key: str) -> Any:
    if key not in block:
        raise ValueError(f"{_PREFIX} missing required key {key!r}")
    value = block[key]
    if value is None:
        raise ValueError(f"{_PREFIX}.{key} is None")
    return value


def _require_bool(block: Mapping[str, Any], key: str) -> bool:
    value = _require_key(block, key)
    if not isinstance(value, bool):
        raise TypeError(f"{_PREFIX}.{key} must be bool, got {value!r}")
    return value


def _require_int(block: Mapping[str, Any], key: str) -> int:
    value = _require_key(block, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{_PREFIX}.{key} must be int, got {value!r}")
    return value


def _require_number(block: Mapping[str, Any], key: str) -> float:
    value = _require_key(block, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{_PREFIX}.{key} must be int or float, got {value!r}")
    return float(value)


def _require_nonempty_str(block: Mapping[str, Any], key: str) -> str:
    value = _require_key(block, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{_PREFIX}.{key} must be a non-empty str, got {value!r}")
    return value
