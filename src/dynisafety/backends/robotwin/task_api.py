"""Task-facing helpers formerly inlined on vendored ``Base_Task``.

``DisturbanceManager`` and per-task overrides call into this module so
``_base_task.py`` only keeps the three hook mount points.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.backends.robotwin.obstacles import build_obstacle_candidates

AVOID_EXCLUDE_NAMES = frozenset({"wall", "table", "ground", ""})
AVOID_EXCLUDE_PREFIXES = ("dynisafety_",)


def _concrete_override(task: Any, method_name: str):
    """Return unbound method if defined on a concrete task class (not Base_Task)."""
    for cls in type(task).__mro__:
        if cls.__name__ in {"Base_Task", "object"}:
            break
        if method_name in cls.__dict__:
            return cls.__dict__[method_name]
    return None


def robotwin_assets_objects_root() -> Path:
    """``assets/objects`` of the resolved RoboTwin tree.

    Honours ``ROBOTWIN_ROOT`` / the ``resolve_robotwin_root`` candidate order so
    it stays consistent with where the env is actually loaded from.
    """
    here = Path(__file__).resolve()
    # .../src/dynisafety/backends/robotwin/task_api.py → workspace = parents[4]
    workspace = here.parents[4]
    from .env import resolve_robotwin_root  # noqa: PLC0415  (avoid import cycle)

    return resolve_robotwin_root(workspace=workspace) / "assets" / "objects"


def get_target_pose(task: Any) -> list[float]:
    override = _concrete_override(task, "get_dynisafety_target_pose")
    if override is not None:
        return override(task)
    if hasattr(task, "get_place_target_pose"):
        return task.get_place_target_pose()
    raise NotImplementedError(
        f"{type(task).__name__} must implement get_dynisafety_target_pose() "
        "(or get_place_target_pose()) before dynamic disturbance can be enabled."
    )


def get_target_object_pose(task: Any) -> list[float]:
    override = _concrete_override(task, "get_dynisafety_target_object_pose")
    if override is not None:
        return override(task)
    target_object = getattr(task, "target_object", None)
    if target_object is None or not hasattr(target_object, "get_pose"):
        raise NotImplementedError(
            f"{type(task).__name__} must implement get_dynisafety_target_object_pose() "
            "before dynamic disturbance with obstacle_target_mode='object' can be enabled."
        )
    pos = np.asarray(target_object.get_pose().p, dtype=np.float64).reshape(-1).tolist()
    return pos + [1, 0, 0, 0]


def get_source_object_pose(task: Any) -> list[float]:
    override = _concrete_override(task, "get_dynisafety_source_object_pose")
    if override is not None:
        return override(task)
    source_object = getattr(task, "object", None)
    if source_object is None or not hasattr(source_object, "get_pose"):
        raise NotImplementedError(
            f"{type(task).__name__} must implement get_dynisafety_source_object_pose() "
            "before dynamic disturbance with obstacle_target_mode='source_object' "
            "can be enabled."
        )
    pos = np.asarray(source_object.get_pose().p, dtype=np.float64).reshape(-1).tolist()
    return pos + [1, 0, 0, 0]


def get_held_objects(task: Any) -> list[Any]:
    override = _concrete_override(task, "get_dynisafety_held_objects")
    if override is not None:
        return list(override(task) or [])
    obj = getattr(task, "object", None)
    if obj is not None and hasattr(obj, "get_name"):
        return [obj]
    return []


def get_obstacle_avoid_actors(task: Any) -> list[Any]:
    override = _concrete_override(task, "get_dynisafety_obstacle_avoid_actors")
    if override is not None:
        return list(override(task) or [])

    try:
        from envs.utils.actor_utils import Actor
    except ImportError:
        Actor = None  # type: ignore[misc, assignment]

    def _is_task_actor(item: Any) -> bool:
        if Actor is not None and not isinstance(item, Actor):
            return False
        try:
            name = item.get_name() or ""
        except Exception:
            name = ""
        if name in AVOID_EXCLUDE_NAMES:
            return False
        if any(name.startswith(prefix) for prefix in AVOID_EXCLUDE_PREFIXES):
            return False
        return True

    avoid: list[Any] = []
    seen_ids: set[int] = set()

    def _add(item: Any) -> None:
        if _is_task_actor(item) and id(item) not in seen_ids:
            seen_ids.add(id(item))
            avoid.append(item)

    for value in task.__dict__.values():
        if Actor is not None and isinstance(value, Actor):
            _add(value)
        elif isinstance(value, (list, tuple)):
            for sub in value:
                _add(sub)
        elif isinstance(value, dict):
            for sub in value.values():
                _add(sub)
    return avoid


def get_obstacle_candidates(
    task: Any,
    *,
    assets_root: Path | None = None,
    candidate_modelnames: Iterable[str] | None = None,
    upright_quats: Mapping[str, Sequence[float]] | None = None,
    model_scales: Mapping[str, float] | None = None,
    semantic_families: Mapping[str, str] | None = None,
    default_upright_quat: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    override = _concrete_override(task, "get_dynisafety_obstacle_candidates")
    if override is not None:
        return list(override(task) or [])

    root = assets_root if assets_root is not None else robotwin_assets_objects_root()
    names = candidate_modelnames
    if names is None:
        names = getattr(task, "DYNI_SAFETY_OBSTACLE_CANDIDATES", None)
    if names is None:
        raise ValueError("disturbance.obstacle.pool is required to build obstacle candidates")
    quats = upright_quats
    if quats is None:
        quats = getattr(task, "DYNI_SAFETY_OBSTACLE_UPRIGHT_QUATS", None)
    if quats is None:
        raise ValueError("disturbance.obstacle.upright_quats is required")
    scales = model_scales
    if scales is None:
        scales = getattr(task, "DYNI_SAFETY_OBSTACLE_MODEL_SCALES", None)
    if scales is None:
        raise ValueError("disturbance.obstacle.rescale is required")
    families = semantic_families
    if families is None:
        families = getattr(task, "DYNI_SAFETY_OBSTACLE_SEMANTIC_FAMILIES", None)
    if families is None:
        raise ValueError("disturbance.obstacle.semantic_families is required")
    default_quat = default_upright_quat
    if default_quat is None:
        default_quat = getattr(task, "DYNI_SAFETY_OBSTACLE_DEFAULT_UPRIGHT_QUAT", None)
    if default_quat is None:
        raise ValueError("disturbance.obstacle.default_upright_quat is required")

    return build_obstacle_candidates(
        getattr(task, "scene", None),
        candidate_modelnames=names,
        upright_quats=quats,
        model_scales=scales,
        assets_root=root,
        semantic_families=families,
        default_upright_quat=default_quat,
    )


def resolve_target_pose_xyz(task: Any, *, obstacle_target_mode: str) -> np.ndarray:
    mode = str(obstacle_target_mode).strip().lower()
    if mode == "object":
        pose = get_target_object_pose(task)
    elif mode == "source_object":
        pose = get_source_object_pose(task)
    else:
        pose = get_target_pose(task)
    # Concrete tasks may return a sapien Pose; prefer ``.p``.
    p = getattr(pose, "p", None)
    if p is not None:
        return np.asarray(p, dtype=np.float64)[:3].copy()
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape[0] < 3:
        raise ValueError("Dynamic safety target pose must have at least 3 values.")
    return arr[:3].copy()
