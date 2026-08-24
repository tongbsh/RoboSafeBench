"""Obstacle mesh candidate specs for RoboTwin DyniSafety (backend adaptation).

Catalog tables live in YAML (``disturbance.obstacle.pool`` / ``rescale`` /
``upright_quats`` / ``semantic_families``). Callers must pass them explicitly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def get_available_model_ids(modelname: str, *, assets_root: Path) -> list[int]:
    """Return ids ``i`` such that ``assets_root/<model>/model_data{i}.json`` exists."""
    asset_path = Path(assets_root) / modelname
    if not asset_path.is_dir():
        return []
    available: list[int] = []
    for file in sorted(asset_path.glob("model_data*.json")):
        suffix = file.stem.replace("model_data", "")
        if suffix == "":
            continue
        try:
            available.append(int(suffix))
        except ValueError:
            continue
    return available


def get_semantic_family(
    modelname: str | None,
    *,
    families: Mapping[str, str],
) -> str | None:
    name = str(modelname or "").strip()
    if not name:
        return None
    return families.get(name)


def build_obstacle_spec(
    modelname: str,
    model_id: int,
    *,
    assets_root: Path,
    upright_quats: Mapping[str, Sequence[float]],
    model_scales: Mapping[str, float],
    semantic_families: Mapping[str, str],
    default_upright_quat: Sequence[float],
) -> dict[str, Any] | None:
    json_path = Path(assets_root) / modelname / f"model_data{model_id}.json"
    if not json_path.is_file():
        return None

    with open(json_path, encoding="utf-8") as f:
        model_data = json.load(f)

    if model_data.get("stable", False) is False:
        return None

    center = np.asarray(model_data.get("center", []), dtype=np.float64)
    extents = np.asarray(model_data.get("extents", []), dtype=np.float64)
    scale = np.asarray(model_data.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if center.shape[0] < 2 or extents.shape[0] < 3 or scale.shape[0] < 3:
        return None

    extra_scale = float(model_scales[modelname]) if modelname in model_scales else 1.0
    if extra_scale <= 0.0:
        return None

    height = (
        float(max((center[1] + extents[1] * 0.5) * scale[1], extents[1] * scale[1])) * extra_scale
    )
    support_offset = float(max((extents[1] * 0.5 - center[1]) * scale[1], 0.0)) * extra_scale
    half_x = float(extents[0] * scale[0] * 0.5) * extra_scale
    half_y = float(extents[2] * scale[2] * 0.5) * extra_scale
    radius_xy = float(max(half_x, half_y))

    if modelname in upright_quats:
        spawn_quat = list(upright_quats[modelname])
    else:
        spawn_quat = [float(v) for v in default_upright_quat]
    if len(spawn_quat) != 4:
        raise ValueError(f"upright quat for {modelname!r} must have 4 values")

    return {
        "modelname": modelname,
        "model_id": int(model_id),
        "semantic_family": get_semantic_family(modelname, families=semantic_families),
        "height": height,
        "support_offset": support_offset,
        "radius_xy": radius_xy,
        "half_xy": (half_x, half_y),
        "spawn_quat": spawn_quat,
        "convex": True,
        "extra_scale": extra_scale,
    }


def build_obstacle_candidates(
    scene: Any,
    *,
    candidate_modelnames: Iterable[str],
    upright_quats: Mapping[str, Sequence[float]],
    model_scales: Mapping[str, float],
    assets_root: Path,
    semantic_families: Mapping[str, str],
    default_upright_quat: Sequence[float],
) -> list[dict[str, Any]]:
    """Enumerate viable obstacle specs, sorted tallest-first."""
    del scene  # reserved for future scene-aware filtering
    candidates: list[dict[str, Any]] = []
    for modelname in candidate_modelnames:
        best_spec: dict[str, Any] | None = None
        for model_id in get_available_model_ids(modelname, assets_root=assets_root):
            spec = build_obstacle_spec(
                modelname,
                model_id,
                assets_root=assets_root,
                upright_quats=upright_quats,
                model_scales=model_scales,
                semantic_families=semantic_families,
                default_upright_quat=default_upright_quat,
            )
            if spec is None:
                continue
            if best_spec is None or spec["height"] > best_spec["height"]:
                best_spec = spec
        if best_spec is not None:
            candidates.append(best_spec)

    candidates.sort(key=lambda item: item["height"], reverse=True)
    return candidates


def select_obstacle(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selection_mode: str,
    episode_index: int,
    rng: np.random.Generator,
) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("select_obstacle requires a non-empty candidates list")
    mode = str(selection_mode).strip().lower()
    if mode == "random":
        idx = int(rng.integers(0, len(candidates)))
        return candidates[idx]
    if mode == "trajectory_order":
        idx = int(episode_index) % len(candidates)
        return candidates[idx]
    raise ValueError(
        f"selection_mode must be 'random' or 'trajectory_order', got {selection_mode!r}"
    )
