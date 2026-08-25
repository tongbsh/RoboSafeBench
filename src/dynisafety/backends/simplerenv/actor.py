"""SAPIEN dynamic-actor lifecycle for SimplerEnv disturbances.

Hyperparameters (scale, model names, hand half-size, …) are caller-injected;
this module has no business defaults (iron law 1).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dynisafety.core.disturbance import quat_about_axis, quat_mul

logger = logging.getLogger(__name__)

# Off-screen park pose (scene convention, not a tunable eval hyperparam).
HIDDEN_POS = np.array([10.0, 10.0, 0.0], dtype=np.float64)
HAND_SKIN_COLOR = [0.72, 0.52, 0.40, 1.0]


def current_obstacle_model(
    *,
    models: Sequence[str],
    fallback_model: str,
    episode_index: int,
) -> str:
    """Cycle ``models`` by episode; fall back to a single model name."""
    if models:
        if episode_index < 0:
            raise ValueError(f"episode_index must be >= 0, got {episode_index}")
        model = models[episode_index % len(models)]
        logger.debug("Episode %d → obstacle model: %s", episode_index, model)
        return str(model)
    if not fallback_model:
        raise ValueError("fallback_model is required when models is empty")
    return str(fallback_model)


def require_obstacle_collision_file(model_dir: Path) -> Path:
    """Return ``collision.obj`` or fail. A missing mesh must not become a 3 cm box."""
    collision_file = Path(model_dir) / "collision.obj"
    if not collision_file.exists():
        raise FileNotFoundError(
            f"obstacle collision mesh missing: {collision_file}; "
            "refusing box fallback (would change CR/force)"
        )
    return collision_file


def compute_obstacle_quat(yaw: float) -> np.ndarray:
    """Orient object-model obstacles upright, then spin in-plane by yaw (wxyz).

    Matches legacy ``controllers.compute_obstacle_quat`` (Rx(90) then Rz(yaw)).
    ``yaw`` is radians.
    """
    q = quat_mul(quat_about_axis("z", math.degrees(yaw)), quat_about_axis("x", 90))
    return q / np.linalg.norm(q)


def _hand_skin_material(scene: Any) -> Any | None:
    try:
        renderer = scene.get_renderer()
        if renderer is None:
            return None
        mat = renderer.create_material()
        mat.set_base_color(HAND_SKIN_COLOR)
        return mat
    except Exception:
        return None


def build_dynamic_actor(
    scene: Any,
    *,
    model_name: str,
    scale: float,
    kinematic: bool,
    use_visual_mesh: bool,
    models_dir: Path,
    asset_dir: Path,
    actor_type: str,
    hand_half_size: Sequence[float],
    hand_collision_offset: Sequence[float],
    hand_scale: float,
    name: str,
    density: float,
) -> Any:
    """Build a SAPIEN kinematic/dynamic actor for L1–L3 disturbances.

    All sizes/scales/paths are required — no embedded eval defaults.
    """
    import sapien.core as sapien

    if actor_type not in ("obstacle", "hand"):
        raise ValueError(f"actor_type must be 'obstacle' or 'hand', got {actor_type!r}")
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if hand_scale <= 0.0:
        raise ValueError(f"hand_scale must be > 0, got {hand_scale}")
    if density <= 0.0:
        raise ValueError(f"density must be > 0, got {density}")

    builder = scene.create_actor_builder()

    if actor_type == "obstacle":
        if not model_name:
            raise ValueError("model_name required for obstacle actors")
        model_dir = Path(models_dir) / model_name
        collision_file = require_obstacle_collision_file(model_dir)
        visual_file = model_dir / "textured.dae"
        if not visual_file.exists():
            visual_file = model_dir / "textured.obj"
        if not visual_file.exists():
            visual_file = model_dir / "textured.glb"

        col_kwargs: dict[str, Any] = dict(scale=[float(scale)] * 3)
        if not kinematic:
            col_kwargs["density"] = float(density)

        builder.add_multiple_collisions_from_file(filename=str(collision_file), **col_kwargs)

        if visual_file.exists():
            builder.add_visual_from_file(filename=str(visual_file), scale=[float(scale)] * 3)
        else:
            half = np.array([0.03, 0.03, 0.06], dtype=np.float64) * float(scale)
            builder.add_box_visual(half_size=half, color=[0.8, 0.2, 0.2, 1.0])
    else:
        half = np.asarray(hand_half_size, dtype=np.float64) * float(hand_scale)
        if half.shape != (3,):
            raise ValueError(f"hand_half_size must have length 3, got {hand_half_size!r}")
        offset = np.asarray(hand_collision_offset, dtype=np.float64) * float(hand_scale)
        if offset.shape != (3,):
            raise ValueError(
                f"hand_collision_offset must have length 3, got {hand_collision_offset!r}"
            )
        collision_pose = sapien.Pose(offset)
        col_kwargs = dict(half_size=half, pose=collision_pose)
        if not kinematic:
            col_kwargs["density"] = float(density)
        builder.add_box_collision(**col_kwargs)

        vis_stl = Path(asset_dir) / "hand_pusher_vis.stl"
        rend_mtl = _hand_skin_material(scene)
        if use_visual_mesh and vis_stl.is_file():
            if rend_mtl is not None:
                builder.add_visual_from_file(
                    filename=str(vis_stl),
                    scale=[float(hand_scale)] * 3,
                    material=rend_mtl,
                )
            else:
                builder.add_visual_from_file(str(vis_stl), scale=[float(hand_scale)] * 3)
        else:
            if rend_mtl is not None:
                builder.add_box_visual(half_size=half, pose=collision_pose, material=rend_mtl)
            else:
                builder.add_box_visual(half_size=half, pose=collision_pose, color=HAND_SKIN_COLOR)

    actor = builder.build_kinematic(name) if kinematic else builder.build(name)
    actor.set_pose(sapien.Pose(HIDDEN_POS))
    return actor


class ActorLifecycle:
    """Create / swap / destroy the DyniSafety dynamic actor on a SAPIEN scene."""

    def __init__(
        self,
        scene: Any,
        *,
        models_dir: Path,
        asset_dir: Path,
        hand_half_size: Sequence[float],
        hand_collision_offset: Sequence[float],
        density: float,
    ) -> None:
        self._scene = scene
        self._models_dir = Path(models_dir)
        self._asset_dir = Path(asset_dir)
        self._hand_half_size = list(hand_half_size)
        self._hand_collision_offset = list(hand_collision_offset)
        self._density = float(density)
        self._actor: Any | None = None
        self._actor_is_kinematic = True

    @property
    def actor(self) -> Any | None:
        return self._actor

    @property
    def is_kinematic(self) -> bool:
        return self._actor_is_kinematic

    def destroy(self, actor: Any | None = None) -> None:
        target = self._actor if actor is None else actor
        if target is None:
            return
        try:
            self._scene.remove_actor(target)
        except Exception:
            pass
        if target is self._actor:
            self._actor = None

    def create(
        self,
        *,
        model_name: str,
        scale: float,
        kinematic: bool,
        use_visual_mesh: bool,
        actor_type: str,
        hand_scale: float,
        name: str = "dynisafety_obstacle",
    ) -> Any:
        self.destroy()
        self._actor_is_kinematic = bool(kinematic)
        self._actor = build_dynamic_actor(
            self._scene,
            model_name=model_name,
            scale=scale,
            kinematic=kinematic,
            use_visual_mesh=use_visual_mesh,
            models_dir=self._models_dir,
            asset_dir=self._asset_dir,
            actor_type=actor_type,
            hand_half_size=self._hand_half_size,
            hand_collision_offset=self._hand_collision_offset,
            hand_scale=hand_scale,
            name=name,
            density=self._density,
        )
        return self._actor

    def release_to_physics(
        self,
        *,
        model_name: str,
        scale: float,
        use_visual_mesh: bool,
        actor_type: str,
        hand_scale: float,
        name: str = "dynisafety_obstacle",
    ) -> Any:
        """Swap a kinematic actor for a dynamic one at the same pose."""
        if self._actor is None:
            raise RuntimeError("no actor to release")
        if not self._actor_is_kinematic:
            return self._actor
        pose = self._actor.get_pose()
        actor = self.create(
            model_name=model_name,
            scale=scale,
            kinematic=False,
            use_visual_mesh=use_visual_mesh,
            actor_type=actor_type,
            hand_scale=hand_scale,
            name=name,
        )
        actor.set_pose(pose)
        actor.set_velocity(np.zeros(3))
        actor.set_angular_velocity(np.zeros(3))
        return actor
