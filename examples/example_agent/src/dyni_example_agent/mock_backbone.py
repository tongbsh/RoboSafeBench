"""Deterministic toy "VLA/WAM" backbone for integration tests.

***************************************************************************
THIS IS NOT A REAL POLICY. It exists only to verify that an external agent
can plug into DyniSafety through the public adapter contract (Observation,
ActionChunk, YAML extras, HTTP transport). Its numbers must NEVER be
reported as DyniSafety benchmark results.
***************************************************************************

The module is framework-agnostic: it does not import ``dynisafety``. The
adapter resolves ``dim_for_space`` and passes the integer action dimension.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np

_FORMAT_VERSION = 1
_IMAGE_KEY_ORDER = ("primary", "wrist", "left", "right")
_TOKEN_BYTES = 16


def _stable_rng(seed: int, instruction: str, step: int, episode_index: int) -> np.random.Generator:
    """Process-stable RNG. Do not use Python's ``hash()`` — it is salted."""
    payload = hashlib.sha256()
    payload.update(str(int(seed)).encode("utf-8"))
    payload.update(b"\x00")
    payload.update(instruction.encode("utf-8"))
    payload.update(b"\x00")
    payload.update(str(int(step)).encode("utf-8"))
    payload.update(b"\x00")
    payload.update(str(int(episode_index)).encode("utf-8"))
    digest = payload.digest()
    spawn = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 32, 4)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(spawn)))


def _pool_image(image: np.ndarray, grid: int) -> np.ndarray:
    """Mean-pool an image to a ``grid x grid`` spatial feature (channel-mean)."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"image must be (H, W) or (H, W, C), got shape {arr.shape}")
    spatial = arr.mean(axis=2)
    height, width = spatial.shape
    if height < 1 or width < 1:
        return np.zeros((grid * grid,), dtype=np.float32)
    pad_h = (grid - (height % grid)) % grid
    pad_w = (grid - (width % grid)) % grid
    if pad_h or pad_w:
        spatial = np.pad(spatial, ((0, pad_h), (0, pad_w)), mode="edge")
        height, width = spatial.shape
    gh, gw = height // grid, width // grid
    return spatial.reshape(grid, gh, grid, gw).mean(axis=(1, 3)).astype(np.float32).reshape(-1)


def _instruction_tokens(instruction: str) -> np.ndarray:
    digest = hashlib.sha256(instruction.encode("utf-8")).digest()
    raw = np.frombuffer(digest[:_TOKEN_BYTES], dtype=np.uint8)
    return raw.astype(np.float32) / np.float32(255.0)


def _fit_features(raw: np.ndarray, dim: int) -> np.ndarray:
    vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    if vec.size == dim:
        return vec
    if vec.size < dim:
        out = np.zeros(dim, dtype=np.float32)
        out[: vec.size] = vec
        return out
    n_blocks = int(np.ceil(vec.size / dim))
    padded = np.zeros(n_blocks * dim, dtype=np.float32)
    padded[: vec.size] = vec
    return padded.reshape(n_blocks, dim).sum(axis=0)


def _image_keys(images: Mapping[str, np.ndarray]) -> list[str]:
    known = [key for key in _IMAGE_KEY_ORDER if key in images]
    extra = sorted(key for key in images if key not in _IMAGE_KEY_ORDER)
    return known + extra


class MockBackbone:
    """Loaded toy weights + a deterministic feature → action projection."""

    def __init__(
        self,
        *,
        proj: np.ndarray,
        scale: np.ndarray,
        action_dim: int,
        feature_grid: int,
        action_scale: float,
        seed: int,
    ) -> None:
        self._proj = np.asarray(proj, dtype=np.float32)
        self._scale = np.asarray(scale, dtype=np.float32)
        self._action_dim = int(action_dim)
        self._feature_grid = int(feature_grid)
        self._action_scale = float(action_scale)
        self._seed = int(seed)
        self.plan_calls = 0

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        action_dim: int,
        feature_grid: int,
        action_scale: float,
        seed: int,
    ) -> MockBackbone:
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"mock weights not found: {checkpoint}. "
                "Run `python examples/example_agent/assets/make_mock_weights.py`."
            )
        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim < 1:
            raise ValueError(f"action_dim must be int >= 1, got {action_dim!r}")
        if isinstance(feature_grid, bool) or not isinstance(feature_grid, int) or feature_grid < 1:
            raise ValueError(f"feature_grid must be int >= 1, got {feature_grid!r}")
        loaded = np.load(checkpoint)
        version = int(np.asarray(loaded["format_version"]).reshape(-1)[0])
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"unsupported mock weight format_version={version}; expected {_FORMAT_VERSION}"
            )
        proj = np.asarray(loaded["proj"], dtype=np.float32)
        scale = np.asarray(loaded["scale"], dtype=np.float32)
        if proj.ndim != 2 or scale.ndim != 1 or scale.shape[0] != proj.shape[1]:
            raise ValueError(f"malformed mock weights: proj={proj.shape} scale={scale.shape}")
        if action_dim > proj.shape[1]:
            raise ValueError(f"action_dim {action_dim} exceeds weight columns {proj.shape[1]}")
        return cls(
            proj=proj,
            scale=scale,
            action_dim=action_dim,
            feature_grid=feature_grid,
            action_scale=action_scale,
            seed=seed,
        )

    def plan(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray | None,
        instruction: str,
        step: int,
        horizon: int,
        episode_index: int,
    ) -> np.ndarray:
        """Return ``(horizon, action_dim)`` float32 actions."""
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError(f"horizon must be int >= 1, got {horizon!r}")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise ValueError(f"episode_index must be a non-negative int, got {episode_index!r}")
        self.plan_calls += 1

        parts: list[np.ndarray] = [
            _pool_image(images[key], self._feature_grid) for key in _image_keys(images)
        ]
        if state is not None:
            parts.append(np.asarray(state, dtype=np.float32).reshape(-1))
        parts.append(_instruction_tokens(instruction))
        parts.append(np.asarray([float(step)], dtype=np.float32))
        features = _fit_features(np.concatenate(parts), self._proj.shape[0])

        base = features @ self._proj[:, : self._action_dim]
        rng = _stable_rng(self._seed, instruction, step, episode_index)
        noise = rng.standard_normal((horizon, self._action_dim)).astype(np.float32)
        scale = self._scale[: self._action_dim] * np.float32(self._action_scale)
        actions = (base[None, :] + noise) * scale[None, :]
        return np.asarray(actions, dtype=np.float32)
