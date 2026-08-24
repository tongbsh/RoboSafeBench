"""Policy observation/action contract (§3.4) — sole VLA integration surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

_VALID_TRANSPORTS = frozenset({"local", "http", "websocket", "zmq"})
_VALID_ACTION_SPACES = frozenset(
    {"delta_ee_7d", "abs_ee_7d", "abs_ee_16d", "qpos_14d", "hybrid_mobile_12d"}
)


@dataclass(frozen=True)
class ModelConfig:
    """Identity / transport / action-space declaration for a registered policy.

    No hyperparameter defaults (iron law 1); callers must supply every field.
    Remote ``server_*`` fields are ``None`` when ``transport == "local"``.
    """

    name: str
    checkpoint: str
    action_space: str
    transport: str
    adapter: str
    chunk_size: int
    observation_images: tuple[str, ...]
    observation_needs_state: bool
    backends: tuple[str, ...]
    max_jobs_per_gpu: int
    server_env: str | None
    server_root: str | None
    server_launch: str | None
    server_default_port: int | None
    server_health_path: str | None
    server_startup_timeout_s: int | None
    extra: dict[str, Any]


@dataclass(frozen=True)
class EpisodeContext:
    instruction: str
    task_id: int
    task_name: str
    episode_index: int
    seed: int
    level: int
    backend: str
    suite: str
    extra: dict[str, Any]


@dataclass
class Observation:
    images: dict[str, np.ndarray]  # "primary" / "wrist" / "left" / "right"
    state: np.ndarray | None
    instruction: str
    step: int
    extra: dict[str, Any]


@dataclass
class ActionChunk:
    actions: np.ndarray  # (chunk, action_dim)
    space: str  # "delta_ee_7d" | "abs_ee_7d" | "abs_ee_16d" | "qpos_14d" | "hybrid_mobile_12d"


class Policy(Protocol):
    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Policy: ...

    def reset(self, ctx: EpisodeContext) -> None: ...

    def act(self, obs: Observation) -> ActionChunk: ...

    def close(self) -> None: ...


def _require(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"ModelConfig missing required key {key!r}")
    return data[key]


def _require_nonempty_str(data: Mapping[str, Any], key: str) -> str:
    value = _require(data, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ModelConfig.{key} must be a non-empty str, got {value!r}")
    return value


def _parse_per_suite(raw: Mapping[str, Any]) -> dict[str, str]:
    """Validate the optional ``checkpoint.per_suite`` mapping."""
    per_suite = raw.get("per_suite")
    if per_suite is None:
        return {}
    if not isinstance(per_suite, Mapping):
        raise ValueError(
            f"ModelConfig.checkpoint.per_suite must be a mapping, got {type(per_suite)!r}"
        )
    out: dict[str, str] = {}
    for suite, value in per_suite.items():
        if not isinstance(suite, str) or not suite.strip():
            raise ValueError(
                f"ModelConfig.checkpoint.per_suite keys must be non-empty str, got {suite!r}"
            )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "ModelConfig.checkpoint.per_suite values must be non-empty str, "
                f"got {value!r} for suite {suite!r}"
            )
        out[suite] = value
    if not out:
        raise ValueError("ModelConfig.checkpoint.per_suite must not be empty")
    return out


_PER_BACKEND_OVERLAY_KEYS = frozenset(
    {"action_space", "observation", "extra", "chunk_size"}
)


def apply_per_backend(
    data: Mapping[str, Any],
    backend: str | None,
) -> dict[str, Any]:
    """Return a copy of *data* with optional top-level ``per_backend`` overlays.

    Shared models (e.g. ``starvla``) keep LIBERO-oriented top-level defaults and
    override ``action_space`` / ``observation`` / ``extra`` per backend — same
    idea as ``checkpoint.per_suite``, but keyed by eval backend.

    * ``action_space``: replaced wholesale when present in the overlay.
    * ``chunk_size``: replaced wholesale when present (e.g. StarVLA open-loop
      horizon differs by backend).
    * ``observation`` / ``extra``: shallow-merged (overlay keys win).
    * Other overlay keys are rejected (keep the surface small).

    When *backend* is ``None`` or there is no ``per_backend`` block / no entry
    for that backend, returns a shallow copy of *data* (``per_backend`` kept).
    """
    out = dict(data)
    if backend is None:
        return out
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError(f"backend must be a non-empty str, got {backend!r}")

    block = data.get("per_backend")
    if block is None:
        return out
    if not isinstance(block, Mapping):
        raise ValueError(
            f"ModelConfig.per_backend must be a mapping, got {type(block)!r}"
        )
    if not block:
        raise ValueError("ModelConfig.per_backend must not be empty when present")

    for key in block:
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                f"ModelConfig.per_backend keys must be non-empty str, got {key!r}"
            )

    overlay = block.get(backend)
    if overlay is None:
        return out
    if not isinstance(overlay, Mapping):
        raise ValueError(
            f"ModelConfig.per_backend[{backend!r}] must be a mapping, got {type(overlay)!r}"
        )

    unknown = sorted(set(overlay) - _PER_BACKEND_OVERLAY_KEYS)
    if unknown:
        raise ValueError(
            f"ModelConfig.per_backend[{backend!r}] has unsupported keys {unknown}; "
            f"allowed: {sorted(_PER_BACKEND_OVERLAY_KEYS)}"
        )

    if "action_space" in overlay:
        space = overlay["action_space"]
        if not isinstance(space, str) or not space.strip():
            raise ValueError(
                f"ModelConfig.per_backend[{backend!r}].action_space must be "
                f"a non-empty str, got {space!r}"
            )
        out["action_space"] = space

    if "chunk_size" in overlay:
        chunk = overlay["chunk_size"]
        if isinstance(chunk, bool) or not isinstance(chunk, int):
            raise ValueError(
                f"ModelConfig.per_backend[{backend!r}].chunk_size must be int, "
                f"got {chunk!r}"
            )
        if chunk < 1:
            raise ValueError(
                f"ModelConfig.per_backend[{backend!r}].chunk_size must be >= 1, "
                f"got {chunk}"
            )
        out["chunk_size"] = chunk

    if "observation" in overlay:
        obs = overlay["observation"]
        if not isinstance(obs, Mapping):
            raise ValueError(
                f"ModelConfig.per_backend[{backend!r}].observation must be a mapping, "
                f"got {type(obs)!r}"
            )
        base_obs = data.get("observation")
        merged_obs = dict(base_obs) if isinstance(base_obs, Mapping) else {}
        merged_obs.update(dict(obs))
        out["observation"] = merged_obs

    if "extra" in overlay:
        extra_ov = overlay["extra"]
        if not isinstance(extra_ov, Mapping):
            raise ValueError(
                f"ModelConfig.per_backend[{backend!r}].extra must be a mapping, "
                f"got {type(extra_ov)!r}"
            )
        base_extra = data.get("extra")
        merged_extra = dict(base_extra) if isinstance(base_extra, Mapping) else {}
        merged_extra.update(dict(extra_ov))
        out["extra"] = merged_extra

    return out



def _resolve_checkpoint(data: Mapping[str, Any]) -> str:
    """Pick the checkpoint for this run.

    ``checkpoint`` may be a plain string, or a mapping with ``default`` and an
    optional ``per_suite`` map. Suite-specific weights matter because several
    models ship one fine-tuned checkpoint per task suite (e.g. openvla-oft has
    separate LIBERO spatial/object/goal/10 checkpoints) — running the wrong one
    silently produces bad numbers rather than an error.

    When the run declares a suite and ``per_suite`` exists but does not cover it,
    this raises instead of falling back, so a mismatch can never pass silently.
    Contexts without a suite (e.g. the ``dynisafety models`` listing) use
    ``default``.
    """
    if "checkpoint" not in data:
        raise ValueError("ModelConfig missing required key 'checkpoint'")
    raw = data["checkpoint"]
    if isinstance(raw, str):
        if not raw.strip():
            raise ValueError("ModelConfig.checkpoint must be a non-empty str")
        return raw
    if isinstance(raw, Mapping):
        if "default" not in raw:
            raise ValueError("ModelConfig.checkpoint mapping requires 'default'")
        default = raw["default"]
        if not isinstance(default, str) or not default.strip():
            raise ValueError(
                "ModelConfig.checkpoint.default must be a non-empty str, "
                f"got {default!r}"
            )
        per_suite = _parse_per_suite(raw)
        if not per_suite:
            return default

        suite = data.get("suite")
        if not isinstance(suite, str) or not suite.strip():
            # No suite in scope (registry listing / validation): default is fine.
            return default
        if suite in per_suite:
            return per_suite[suite]
        raise ValueError(
            f"ModelConfig.checkpoint.per_suite has no entry for suite {suite!r}; "
            f"known suites: {sorted(per_suite)}. Add the suite (or remove "
            "per_suite to use 'default' everywhere) — falling back would run the "
            "wrong fine-tuned weights."
        )
    raise ValueError(
        f"ModelConfig.checkpoint must be str or mapping with 'default', got {type(raw)!r}"
    )


def resolve_checkpoint(data: Mapping[str, Any], *, suite: str | None = None) -> str:
    """Resolve ``checkpoint`` for an explicit suite (see :func:`_resolve_checkpoint`).

    The sweep scheduler and ``dynisafety serve`` need the same suite-aware
    selection as the eval path, but carry the suite outside the config mapping.
    """
    if suite is None:
        return _resolve_checkpoint(data)
    return _resolve_checkpoint({**data, "suite": suite})


def _parse_observation(data: Mapping[str, Any]) -> tuple[tuple[str, ...], bool]:
    obs = _require(data, "observation")
    if not isinstance(obs, Mapping):
        raise ValueError(f"ModelConfig.observation must be a mapping, got {type(obs)!r}")
    images = _require(obs, "images")
    if not isinstance(images, (list, tuple)) or not images:
        raise ValueError("ModelConfig.observation.images must be a non-empty list")
    out_images: list[str] = []
    for item in images:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"ModelConfig.observation.images entries must be non-empty str, got {item!r}"
            )
        out_images.append(item)
    needs_state = _require(obs, "needs_state")
    if not isinstance(needs_state, bool):
        raise ValueError(
            f"ModelConfig.observation.needs_state must be bool, got {needs_state!r}"
        )
    return tuple(out_images), needs_state


def _parse_backends(data: Mapping[str, Any]) -> tuple[str, ...]:
    backends = _require(data, "backends")
    if not isinstance(backends, (list, tuple)) or not backends:
        raise ValueError("ModelConfig.backends must be a non-empty list")
    out: list[str] = []
    for item in backends:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"ModelConfig.backends entries must be non-empty str, got {item!r}"
            )
        out.append(item)
    return tuple(out)


def _parse_max_jobs_per_gpu(data: Mapping[str, Any]) -> int:
    resources = data.get("resources")
    if isinstance(resources, Mapping) and "max_jobs_per_gpu" in resources:
        raw = resources["max_jobs_per_gpu"]
    elif isinstance(data.get("extra"), Mapping) and "max_jobs_per_gpu" in data["extra"]:
        # M3 transitional placement; no numeric default invented here.
        raw = data["extra"]["max_jobs_per_gpu"]
    else:
        raise ValueError(
            "ModelConfig missing resources.max_jobs_per_gpu "
            "(or transitional extra.max_jobs_per_gpu)"
        )
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"max_jobs_per_gpu must be int, got {raw!r}")
    if raw < 1:
        raise ValueError(f"max_jobs_per_gpu must be >= 1, got {raw}")
    return raw


def _parse_server(
    data: Mapping[str, Any], *, transport: str
) -> tuple[str | None, str | None, str | None, int | None, str | None, int | None]:
    if transport == "local":
        return None, None, None, None, None, None
    server = _require(data, "server")
    if not isinstance(server, Mapping):
        raise ValueError(f"ModelConfig.server must be a mapping, got {type(server)!r}")
    env = server.get("env")
    root = server.get("root")
    launch = server.get("launch")
    port = server.get("default_port")
    health = server.get("health_path")
    timeout = server.get("startup_timeout_s")
    for key, val in (
        ("env", env),
        ("root", root),
        ("launch", launch),
        ("health_path", health),
    ):
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"ModelConfig.server.{key} must be a non-empty str for "
                f"transport={transport!r}, got {val!r}"
            )
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError(
            f"ModelConfig.server.default_port must be int, got {port!r}"
        )
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        raise ValueError(
            f"ModelConfig.server.startup_timeout_s must be int, got {timeout!r}"
        )
    if timeout < 1:
        raise ValueError(f"server.startup_timeout_s must be >= 1, got {timeout}")
    return str(env), str(root), str(launch), int(port), str(health), int(timeout)


def model_config_from_resolved(
    data: Mapping[str, Any],
    *,
    backend: str | None = None,
) -> ModelConfig:
    """Build :class:`ModelConfig` from a loader-merged mapping. No implicit defaults.

    When *backend* is set, apply top-level ``per_backend`` overlays first
    (shared models such as ``starvla``).
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"expected mapping, got {type(data)!r}")

    data = apply_per_backend(data, backend)

    name = _require_nonempty_str(data, "name")
    checkpoint = _resolve_checkpoint(data)
    action_space = _require_nonempty_str(data, "action_space")
    if action_space not in _VALID_ACTION_SPACES:
        raise ValueError(
            f"ModelConfig.action_space must be one of {sorted(_VALID_ACTION_SPACES)}, "
            f"got {action_space!r}"
        )
    transport = _require_nonempty_str(data, "transport")
    if transport not in _VALID_TRANSPORTS:
        raise ValueError(
            f"ModelConfig.transport must be one of {sorted(_VALID_TRANSPORTS)}, "
            f"got {transport!r}"
        )
    adapter = _require_nonempty_str(data, "adapter")
    chunk_size = _require(data, "chunk_size")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError(f"ModelConfig.chunk_size must be int, got {chunk_size!r}")
    if chunk_size < 1:
        raise ValueError(f"ModelConfig.chunk_size must be >= 1, got {chunk_size}")

    observation_images, observation_needs_state = _parse_observation(data)
    backends = _parse_backends(data)
    max_jobs_per_gpu = _parse_max_jobs_per_gpu(data)
    (
        server_env,
        server_root,
        server_launch,
        server_default_port,
        server_health_path,
        server_startup_timeout_s,
    ) = _parse_server(data, transport=transport)

    extra_raw = data.get("extra", {})
    if extra_raw is None:
        extra_raw = {}
    if not isinstance(extra_raw, Mapping):
        raise ValueError(f"ModelConfig.extra must be a mapping, got {type(extra_raw)!r}")
    extra = dict(extra_raw)

    return ModelConfig(
        name=name,
        checkpoint=checkpoint,
        action_space=action_space,
        transport=transport,
        adapter=adapter,
        chunk_size=chunk_size,
        observation_images=observation_images,
        observation_needs_state=observation_needs_state,
        backends=backends,
        max_jobs_per_gpu=max_jobs_per_gpu,
        server_env=server_env,
        server_root=server_root,
        server_launch=server_launch,
        server_default_port=server_default_port,
        server_health_path=server_health_path,
        server_startup_timeout_s=server_startup_timeout_s,
        extra=extra,
    )
