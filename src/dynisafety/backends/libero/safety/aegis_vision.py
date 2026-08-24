"""LIBERO AEGIS vision obstacle source (Prompt A3a/A3c).

A3a: YAML contract, estimator protocol, settle cache, fixture / unavailable.
A3c: GroundingDINO estimator when checkpoint+config exist (lazy import).

Does not replace the CBF-QP core. ``sim_gt`` remains the fallback source.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dynisafety.backends.libero.safety.aegis_vision_geometry import XyzBounds

_REPO_ROOT = Path(__file__).resolve().parents[5]
_OBSTACLE_SUFFIX = re.compile(r"_obstacle_\d+$", re.IGNORECASE)
_FROM_LLM_MODES = frozenset({"from_llm", "from_task_or_llm"})
_PROMPT_MODES = frozenset({"from_controller_name", "fixed", *_FROM_LLM_MODES})
_REFRESH_MODES = frozenset({"on_settle", "always", "every_n_steps"})
_FAILURE_POLICIES = frozenset({"fallback_sim_gt", "skip_filter", "raise", "hold"})


class VisionEstimateFailure(str, Enum):
    """Structured reasons. Never pretend a failed estimate is a vision ellipsoid."""

    NO_BOX = "no_box"
    EMPTY_CLOUD = "empty_cloud"
    FIT_FAILED = "fit_failed"
    MISSING_DEPTH = "missing_depth"
    MISSING_WEIGHTS = "missing_weights"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    FROM_LLM_UNAVAILABLE = "from_llm_unavailable"
    MISSING_PROMPT = "missing_prompt"
    RENDER_FAILED = "render_failed"
    FIXTURE_FAILURE = "fixture_failure"


class VisionFailurePolicy(str, Enum):
    FALLBACK_SIM_GT = "fallback_sim_gt"
    SKIP_FILTER = "skip_filter"
    RAISE = "raise"


class VisionRefreshMode(str, Enum):
    ON_SETTLE = "on_settle"
    ALWAYS = "always"
    EVERY_N_STEPS = "every_n_steps"


class VisionPromptMode(str, Enum):
    FROM_CONTROLLER_NAME = "from_controller_name"
    FIXED = "fixed"
    FROM_LLM = "from_llm"


@dataclass(frozen=True)
class VisionConfig:
    """All vision knobs from YAML (iron law 1: no implicit numeric defaults)."""

    text_prompt_mode: VisionPromptMode
    views: tuple[str, ...]
    on_failure: VisionFailurePolicy
    refresh: VisionRefreshMode
    grounding_dino_config: str
    grounding_dino_checkpoint: str
    box_threshold: float
    text_threshold: float
    device: str
    device_fallback: str
    flip_rgb_for_dino: bool
    unproject_v_flip: bool
    keep_nearest_frac: float
    min_cloud_points: int
    cluster: str
    dbscan_eps: float
    dbscan_min_points: int
    table_bounds: dict[str, XyzBounds]
    refresh_every_n: int | None = None
    text_prompt: str | None = None
    prompt_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionEstimateRequest:
    obs: Mapping[str, Any]
    env: Any
    controller: Any | None
    task_step: int
    obstacle_name: str | None
    text_prompt: str
    prompt_mode: VisionPromptMode
    views: tuple[str, ...]
    suite: str = ""


@dataclass(frozen=True)
class VisionEstimate:
    ellipsoid: Any | None
    failed: bool
    reason: str | None
    prompt: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls, ellipsoid: Any, *, prompt: str | None = None, extra: dict[str, Any] | None = None
    ) -> VisionEstimate:
        return cls(
            ellipsoid=ellipsoid,
            failed=False,
            reason=None,
            prompt=prompt,
            extra=dict(extra or {}),
        )

    @classmethod
    def failure(
        cls,
        reason: VisionEstimateFailure | str,
        *,
        prompt: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> VisionEstimate:
        value = reason.value if isinstance(reason, VisionEstimateFailure) else str(reason)
        return cls(
            ellipsoid=None,
            failed=True,
            reason=value,
            prompt=prompt,
            extra=dict(extra or {}),
        )


class ObstacleEllipsoidEstimator(Protocol):
    """RGB-D (+ optional LLM) → ``EllipsoidState``. A3c fills the real DINO impl."""

    def estimate(self, request: VisionEstimateRequest) -> VisionEstimate: ...


def prompt_from_obstacle_name(
    name: str | None,
    prompt_map: Mapping[str, str] | None = None,
) -> str | None:
    """Map ``moka_pot_obstacle_1`` → ``moka pot`` (or YAML ``prompt_map``)."""
    if name is None or not str(name).strip():
        return None
    key = str(name).strip()
    if prompt_map and key in prompt_map:
        mapped = str(prompt_map[key]).strip()
        return mapped or None
    stem = _OBSTACLE_SUFFIX.sub("", key)
    prompt = stem.replace("_", " ").strip()
    return prompt or None


def resolve_vision_text_prompt(
    cfg: VisionConfig,
    *,
    obstacle_name: str | None,
) -> str | None:
    if cfg.text_prompt_mode is VisionPromptMode.FIXED:
        raw = cfg.text_prompt
        if raw is None or not str(raw).strip():
            return None
        return str(raw).strip()
    return prompt_from_obstacle_name(obstacle_name, cfg.prompt_map)


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"extra.libero.aegis.vision missing required key {key!r}")
    return mapping[key]


def _require_nonempty_str(mapping: Mapping[str, Any], key: str) -> str:
    value = _require(mapping, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"extra.libero.aegis.vision.{key} must be a non-empty str")
    return value.strip()


def _require_float(mapping: Mapping[str, Any], key: str) -> float:
    value = _require(mapping, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"extra.libero.aegis.vision.{key} must be float")
    return float(value)


def _require_positive_float(mapping: Mapping[str, Any], key: str) -> float:
    value = _require_float(mapping, key)
    if value <= 0.0:
        raise ValueError(f"extra.libero.aegis.vision.{key} must be > 0")
    return value


def _require_str_tuple(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = _require(mapping, key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError(f"extra.libero.aegis.vision.{key} must be a non-empty list of str")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"extra.libero.aegis.vision.{key} entries must be non-empty str")
        out.append(item.strip())
    return tuple(out)


def _require_bool(mapping: Mapping[str, Any], key: str) -> bool:
    value = _require(mapping, key)
    if not isinstance(value, bool):
        raise ValueError(f"extra.libero.aegis.vision.{key} must be bool")
    return value


def _require_int_ge1(mapping: Mapping[str, Any], key: str) -> int:
    value = _require(mapping, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"extra.libero.aegis.vision.{key} must be int >= 1")
    return value


def _parse_table_bounds(mapping: Mapping[str, Any]) -> dict[str, XyzBounds]:
    raw = _require(mapping, "table_bounds")
    if not isinstance(raw, Mapping) or "default" not in raw:
        raise ValueError("extra.libero.aegis.vision.table_bounds must include a 'default' mapping")
    out: dict[str, XyzBounds] = {}
    keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    for name, block in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("table_bounds keys must be non-empty str")
        if not isinstance(block, Mapping):
            raise ValueError(f"table_bounds[{name!r}] must be a mapping")
        vals: dict[str, float] = {}
        for key in keys:
            if key not in block:
                raise ValueError(f"table_bounds[{name!r}] missing {key}")
            item = block[key]
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"table_bounds[{name!r}].{key} must be float")
            vals[key] = float(item)
        if not (
            vals["x_min"] < vals["x_max"]
            and vals["y_min"] < vals["y_max"]
            and vals["z_min"] < vals["z_max"]
        ):
            raise ValueError(f"table_bounds[{name!r}] min must be < max")
        out[name.strip()] = XyzBounds(**vals)
    return out


def bounds_for_suite(cfg: VisionConfig, suite: str | None) -> XyzBounds:
    name = (suite or "").strip()
    if name and name in cfg.table_bounds:
        return cfg.table_bounds[name]
    lowered = name.lower()
    for key, bounds in cfg.table_bounds.items():
        if key != "default" and key.lower() in lowered:
            return bounds
    return cfg.table_bounds["default"]


def _optional_prompt_map(mapping: Mapping[str, Any]) -> dict[str, str]:
    if "prompt_map" not in mapping:
        return {}
    raw = mapping["prompt_map"]
    if not isinstance(raw, Mapping):
        raise ValueError("extra.libero.aegis.vision.prompt_map must be a mapping")
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("extra.libero.aegis.vision.prompt_map keys must be non-empty str")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("extra.libero.aegis.vision.prompt_map values must be non-empty str")
        out[key.strip()] = value.strip()
    return out


def _parse_prompt_mode(raw: str) -> VisionPromptMode:
    mode = raw.strip().lower()
    if mode not in _PROMPT_MODES:
        raise ValueError(
            "extra.libero.aegis.vision.text_prompt_mode must be one of "
            f"{sorted(_PROMPT_MODES)}, got {raw!r}"
        )
    if mode in _FROM_LLM_MODES:
        return VisionPromptMode.FROM_LLM
    return VisionPromptMode(mode)


def _parse_failure_policy(raw: str) -> VisionFailurePolicy:
    policy = raw.strip().lower()
    if policy not in _FAILURE_POLICIES:
        raise ValueError(
            "extra.libero.aegis.vision.on_failure must be one of "
            "['fallback_sim_gt', 'skip_filter', 'raise'] "
            f"(hold is an alias of skip_filter), got {raw!r}"
        )
    if policy == "hold":
        return VisionFailurePolicy.SKIP_FILTER
    return VisionFailurePolicy(policy)


def vision_config_from_mapping(data: Mapping[str, Any]) -> VisionConfig:
    """Parse ``extra.libero.aegis.vision`` with no implicit numeric defaults."""
    if not isinstance(data, Mapping):
        raise TypeError(f"extra.libero.aegis.vision must be a mapping, got {type(data)!r}")

    mode = _parse_prompt_mode(_require_nonempty_str(data, "text_prompt_mode"))
    refresh = _require_nonempty_str(data, "refresh").strip().lower()
    if refresh not in _REFRESH_MODES:
        raise ValueError(
            "extra.libero.aegis.vision.refresh must be one of "
            f"{sorted(_REFRESH_MODES)}, got {refresh!r}"
        )
    refresh_mode = VisionRefreshMode(refresh)

    refresh_every_n: int | None = None
    if refresh_mode is VisionRefreshMode.EVERY_N_STEPS:
        raw_n = _require(data, "refresh_every_n")
        if isinstance(raw_n, bool) or not isinstance(raw_n, int) or raw_n < 1:
            raise ValueError("extra.libero.aegis.vision.refresh_every_n must be int >= 1")
        refresh_every_n = raw_n

    text_prompt: str | None = None
    if mode is VisionPromptMode.FIXED:
        text_prompt = _require_nonempty_str(data, "text_prompt")
    elif "text_prompt" in data and data["text_prompt"] not in (None, ""):
        text_prompt = str(data["text_prompt"]).strip()

    cluster = _require_nonempty_str(data, "cluster").lower()
    if cluster not in {"none", "dbscan"}:
        raise ValueError("extra.libero.aegis.vision.cluster must be 'none' or 'dbscan'")
    keep_frac = _require_positive_float(data, "keep_nearest_frac")
    if keep_frac > 1.0:
        raise ValueError("extra.libero.aegis.vision.keep_nearest_frac must be in (0, 1]")
    device_fallback = _require_nonempty_str(data, "device_fallback").lower()
    if device_fallback not in {"cpu", "none"}:
        raise ValueError("extra.libero.aegis.vision.device_fallback must be 'cpu' or 'none'")

    return VisionConfig(
        text_prompt_mode=mode,
        views=_require_str_tuple(data, "views"),
        on_failure=_parse_failure_policy(_require_nonempty_str(data, "on_failure")),
        refresh=refresh_mode,
        grounding_dino_config=_require_nonempty_str(data, "grounding_dino_config"),
        grounding_dino_checkpoint=_require_nonempty_str(data, "grounding_dino_checkpoint"),
        box_threshold=_require_positive_float(data, "box_threshold"),
        text_threshold=_require_positive_float(data, "text_threshold"),
        device=_require_nonempty_str(data, "device").lower(),
        device_fallback=device_fallback,
        flip_rgb_for_dino=_require_bool(data, "flip_rgb_for_dino"),
        unproject_v_flip=_require_bool(data, "unproject_v_flip"),
        keep_nearest_frac=keep_frac,
        min_cloud_points=_require_int_ge1(data, "min_cloud_points"),
        cluster=cluster,
        dbscan_eps=_require_positive_float(data, "dbscan_eps"),
        dbscan_min_points=_require_int_ge1(data, "dbscan_min_points"),
        table_bounds=_parse_table_bounds(data),
        refresh_every_n=refresh_every_n,
        text_prompt=text_prompt,
        prompt_map=_optional_prompt_map(data),
    )


def resolve_repo_path(raw: str, *, repo_root: Path | None = None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    root = repo_root if repo_root is not None else _REPO_ROOT
    return root / path


class UnavailableVisionEstimator:
    """Used when checkpoint/config files are missing, or as an explicit test double."""

    def __init__(
        self,
        cfg: VisionConfig,
        *,
        repo_root: Path | None = None,
    ) -> None:
        self._cfg = cfg
        self._repo_root = repo_root if repo_root is not None else _REPO_ROOT

    def estimate(self, request: VisionEstimateRequest) -> VisionEstimate:
        prompt = request.text_prompt
        if request.prompt_mode is VisionPromptMode.FROM_LLM:
            return VisionEstimate.failure(
                VisionEstimateFailure.FROM_LLM_UNAVAILABLE,
                prompt=prompt,
                extra={"backend": "a3a_placeholder"},
            )
        ckpt = resolve_repo_path(self._cfg.grounding_dino_checkpoint, repo_root=self._repo_root)
        cfg_path = resolve_repo_path(self._cfg.grounding_dino_config, repo_root=self._repo_root)
        if not ckpt.is_file() or not cfg_path.is_file():
            return VisionEstimate.failure(
                VisionEstimateFailure.MISSING_WEIGHTS,
                prompt=prompt,
                extra={
                    "checkpoint": str(ckpt),
                    "config": str(cfg_path),
                    "checkpoint_exists": ckpt.is_file(),
                    "config_exists": cfg_path.is_file(),
                },
            )
        return VisionEstimate.failure(
            VisionEstimateFailure.BACKEND_UNAVAILABLE,
            prompt=prompt,
            extra={"note": "GroundingDINO files present but A3c estimator is not wired"},
        )


class FixtureVisionEstimator:
    """CPU test double. Does not touch GroundingDINO."""

    def __init__(
        self,
        *,
        ellipsoid: Any | None = None,
        fail_reason: VisionEstimateFailure | str | None = None,
    ) -> None:
        if ellipsoid is None and fail_reason is None:
            fail_reason = VisionEstimateFailure.FIXTURE_FAILURE
        self.ellipsoid = ellipsoid
        self.fail_reason = fail_reason
        self.calls = 0
        self.last_request: VisionEstimateRequest | None = None

    def estimate(self, request: VisionEstimateRequest) -> VisionEstimate:
        self.calls += 1
        self.last_request = request
        if self.fail_reason is not None:
            return VisionEstimate.failure(self.fail_reason, prompt=request.text_prompt)
        assert self.ellipsoid is not None
        return VisionEstimate.ok(self.ellipsoid, prompt=request.text_prompt)


def fixture_ellipsoid(
    *,
    position: Sequence[float] = (0.12, 0.0, 1.0),
    axes_diag: Sequence[float] = (0.06, 0.06, 0.08),
    ellipsoid_cls: Any | None = None,
) -> Any:
    """Axis-aligned ellipsoid for CPU tests."""
    cls = ellipsoid_cls
    if cls is None:
        from dynisafety.policies.safety.aegis import EllipsoidState

        cls = EllipsoidState
    return cls(
        position=np.asarray(position, dtype=np.float64),
        axes_diag=np.asarray(axes_diag, dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
    )


def make_default_vision_estimator(
    cfg: VisionConfig,
    *,
    repo_root: Path | None = None,
) -> ObstacleEllipsoidEstimator:
    """Missing files → unavailable; otherwise the A3c GroundingDINO estimator."""
    ckpt = resolve_repo_path(cfg.grounding_dino_checkpoint, repo_root=repo_root)
    cfg_path = resolve_repo_path(cfg.grounding_dino_config, repo_root=repo_root)
    if not ckpt.is_file() or not cfg_path.is_file():
        return UnavailableVisionEstimator(cfg, repo_root=repo_root)
    from dynisafety.backends.libero.safety.aegis_vision_dino import GroundingDinoVisionEstimator

    return GroundingDinoVisionEstimator(cfg, repo_root=repo_root)
