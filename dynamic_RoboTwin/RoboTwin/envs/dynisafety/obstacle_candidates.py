"""Shared helpers for building DyniSafety obstacle candidate specs.

Originally each task that wanted real-mesh obstacles (bottles / cups / etc.) had
to ship ~80 lines of duplicated logic to enumerate ``assets/objects/<model>/
model_data*.json`` files and translate them into the spec dict consumed by
:class:`RoboTwinDyniSafetyManager`. This module centralises that logic so that:

  * tasks just declare a list of candidate model names (or inherit the default
    on :class:`Base_Task`),
  * the actual scanning / filtering / scaling happens in one place,
  * fixing a bug in the spec computation only needs one edit.

The design intentionally mirrors the original per-task implementation in
``place_a2b_right.py`` so behaviour is preserved bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# Default candidate set used by :class:`Base_Task` when a subclass does not
# override ``DYNI_SAFETY_OBSTACLE_CANDIDATES``. These five upright objects ship
# with multiple ``model_data*.json`` variants under
# ``RoboTwin/assets/objects/<name>/`` and have well-defined collision/visual
# meshes, which makes them safe defaults for tabletop disturbance.
DEFAULT_OBSTACLE_CANDIDATES: tuple[str, ...] = (
    "088_wineglass",
    "109_hydrating-oil",
    "114_bottle",
    "021_cup",
    "080_pillbottle",
)

# Spawn quaternion that keeps each candidate upright on the table. Anything not
# listed here falls back to ``(0.5, 0.5, 0.5, 0.5)``.
DEFAULT_OBSTACLE_UPRIGHT_QUATS: dict[str, list[float]] = {
    "088_wineglass": [0.5, 0.5, 0.5, 0.5],
    "109_hydrating-oil": [0.5, 0.5, 0.5, 0.5],
    "114_bottle": [0.707, 0.707, 0.0, 0.0],
    "021_cup": [0.5, 0.5, 0.5, 0.5],
    "080_pillbottle": [0.5, 0.5, 0.5, 0.5],
}

# Per-model EXTRA scale multiplier applied on top of the global
# ``RoboTwinDyniSafetyConfig.obstacle_scale`` when the obstacle is created.
# Use this to shrink (or grow) a single mesh without affecting others -- useful
# when one candidate (e.g. the wineglass) ships at a much taller scale than the
# rest and dominates the spawn-clearance check. Anything not listed here uses
# 1.0 (i.e. only the global scale applies).
DEFAULT_OBSTACLE_MODEL_SCALES: dict[str, float] = {
    "088_wineglass": 0.8,
}

# Conservative coarse semantic families used only for selection-time fallback.
# The goal is NOT to fully taxonomize every RoboTwin object, but to prevent
# obviously redundant disturbances such as spawning one bottle family object
# into a task scene that already contains another bottle asset.
DEFAULT_OBSTACLE_SEMANTIC_FAMILIES: dict[str, str] = {
    "001_bottle": "bottle",
    "080_pillbottle": "bottle",
    "109_hydrating-oil": "bottle",
    "114_bottle": "bottle",
    "021_cup": "cup",
    "088_wineglass": "wineglass",
}

_DEFAULT_UPRIGHT_QUAT: list[float] = [0.5, 0.5, 0.5, 0.5]
_ASSETS_ROOT = Path("assets/objects")


def get_available_model_ids(modelname: str) -> list[int]:
    """Return integer ids ``i`` for which ``model_data{i}.json`` exists.

    The bare ``model_data.json`` (without numeric suffix) is intentionally
    skipped so callers always have an explicit id to pass into
    ``create_actor(model_id=...)``.
    """
    asset_path = _ASSETS_ROOT / modelname
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


def get_semantic_family(modelname: str | None) -> str | None:
    """Return a coarse semantic family label for ``modelname`` when known.

    Unknown model names return ``None`` so callers can keep the conflict rule
    conservative: exact-name fallback always applies, while family-level
    fallback only activates for the small set of object families explicitly
    curated here.
    """
    name = str(modelname or "").strip()
    if not name:
        return None
    return DEFAULT_OBSTACLE_SEMANTIC_FAMILIES.get(name)


def build_obstacle_spec(
    modelname: str,
    model_id: int,
    upright_quats: Optional[dict[str, list[float]]] = None,
    model_scales: Optional[dict[str, float]] = None,
) -> Optional[dict]:
    """Translate a single ``model_data*.json`` into a manager spec dict.

    Returns ``None`` if the json is missing, marked unstable, or has malformed
    ``center / extents / scale`` arrays. The dict's ``height`` /
    ``support_offset`` / ``radius_xy`` are pre-multiplied by the per-model
    ``extra_scale`` so callers (and the spawn-clearance check) see the true
    effective dimensions without having to remember to multiply again.
    """
    json_path = _ASSETS_ROOT / modelname / f"model_data{model_id}.json"
    if not json_path.is_file():
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    if model_data.get("stable", False) is False:
        return None

    center = np.asarray(model_data.get("center", []), dtype=np.float64)
    extents = np.asarray(model_data.get("extents", []), dtype=np.float64)
    scale = np.asarray(model_data.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if center.shape[0] < 2 or extents.shape[0] < 3 or scale.shape[0] < 3:
        return None

    scales = model_scales if model_scales is not None else DEFAULT_OBSTACLE_MODEL_SCALES
    extra_scale = float(scales.get(modelname, 1.0))
    if extra_scale <= 0.0:
        return None

    height = float(max((center[1] + extents[1] * 0.5) * scale[1], extents[1] * scale[1])) * extra_scale
    support_offset = float(max((extents[1] * 0.5 - center[1]) * scale[1], 0.0)) * extra_scale
    radius_xy = float(max(extents[0] * scale[0], extents[2] * scale[2]) * 0.5) * extra_scale

    quats = upright_quats if upright_quats is not None else DEFAULT_OBSTACLE_UPRIGHT_QUATS
    return {
        "modelname": modelname,
        "model_id": int(model_id),
        "semantic_family": get_semantic_family(modelname),
        "height": height,
        "support_offset": support_offset,
        "radius_xy": radius_xy,
        "spawn_quat": quats.get(modelname, _DEFAULT_UPRIGHT_QUAT),
        "convex": True,
        # Manager multiplies this by the global ``obstacle_scale`` to obtain
        # the actual ``scale_multiplier`` passed to ``create_actor``.
        "extra_scale": extra_scale,
    }


def build_obstacle_candidates(
    scene,
    candidate_modelnames: Iterable[str],
    upright_quats: Optional[dict[str, list[float]]] = None,
    model_scales: Optional[dict[str, float]] = None,
) -> list[dict]:
    """Enumerate viable obstacle specs, sorted tallest-first.

    ``scene`` is currently unused and kept only for caller compatibility.
    Exact model-name conflicts with task objects are resolved later by
    :class:`RoboTwinDyniSafetyManager` during candidate selection so the
    benchmark can preserve the full nominal ``trajectory_order`` over the
    complete candidate pool and only deviate when a fallback is necessary.
    """
    candidates: list[dict] = []
    for modelname in candidate_modelnames:
        best_spec: Optional[dict] = None
        for model_id in get_available_model_ids(modelname):
            spec = build_obstacle_spec(
                modelname,
                model_id,
                upright_quats=upright_quats,
                model_scales=model_scales,
            )
            if spec is None:
                continue
            if best_spec is None or spec["height"] > best_spec["height"]:
                best_spec = spec
        if best_spec is not None:
            candidates.append(best_spec)

    candidates.sort(key=lambda item: item["height"], reverse=True)
    return candidates


__all__ = [
    "DEFAULT_OBSTACLE_CANDIDATES",
    "DEFAULT_OBSTACLE_UPRIGHT_QUATS",
    "DEFAULT_OBSTACLE_MODEL_SCALES",
    "DEFAULT_OBSTACLE_SEMANTIC_FAMILIES",
    "get_available_model_ids",
    "get_semantic_family",
    "build_obstacle_spec",
    "build_obstacle_candidates",
]
