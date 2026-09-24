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
# Same RGB as extra.robotwin.hand_color / extra.simplerenv.hand_color.
HAND_SKIN_COLOR = [0.78, 0.60, 0.48, 1.0]


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


def hand_rgba(color: Sequence[float]) -> list[float]:
    """RGB or RGBA in [0, 1] → RGBA list (visual only)."""
    raw = [float(x) for x in color]
    if len(raw) == 3:
        raw.append(1.0)
    if len(raw) != 4:
        raise ValueError(f"hand_color must be RGB or RGBA, got {color!r}")
    if any(c < 0.0 or c > 1.0 for c in raw):
        raise ValueError(f"hand_color channels must be in [0, 1], got {color!r}")
    return raw


def _scene_renderer(scene: Any) -> Any | None:
    """SAPIEN 2.2 ``Scene`` has no ``get_renderer``; use ``scene.engine``."""
    getter = getattr(scene, "get_renderer", None)
    if callable(getter):
        try:
            rend = getter()
            if rend is not None:
                return rend
        except Exception:
            pass
    engine = getattr(scene, "engine", None)
    if engine is None:
        return None
    eg = getattr(engine, "get_renderer", None)
    if not callable(eg):
        return None
    try:
        return eg()
    except Exception:
        return None


def _finish_skin_material(mat: Any, rgba: Sequence[float]) -> Any:
    """Matte flesh: unset metallic/specular so a tinted STL is not blown to white."""
    if hasattr(mat, "set_base_color"):
        mat.set_base_color(list(rgba))
    else:
        mat.base_color = list(rgba)
    if hasattr(mat, "set_roughness"):
        mat.set_roughness(0.85)
    else:
        try:
            mat.roughness = 0.85
        except Exception:
            pass
    if hasattr(mat, "set_metallic"):
        mat.set_metallic(0.0)
    else:
        try:
            mat.metallic = 0.0
        except Exception:
            pass
    if hasattr(mat, "set_specular"):
        mat.set_specular(0.0)
    return mat


def _hand_skin_material(scene: Any, color: Sequence[float]) -> Any | None:
    """Build a renderer-owned material. ``sapien.RenderMaterial()`` has no ctor on 2.2."""
    rgba = hand_rgba(color)
    renderer = _scene_renderer(scene)
    if renderer is not None:
        create = getattr(renderer, "create_material", None)
        if callable(create):
            try:
                return _finish_skin_material(create(), rgba)
            except Exception:
                pass
    try:
        import sapien.core as sapien

        try:
            mat = sapien.RenderMaterial(base_color=list(rgba))
        except TypeError:
            mat = sapien.RenderMaterial()
            mat.set_base_color(list(rgba))
        return _finish_skin_material(mat, rgba)
    except Exception:
        return None


def apply_hand_skin_color(
    actor: Any,
    color: Sequence[float],
    scene: Any | None = None,
) -> None:
    """Bind a flesh material after build.

    SAPIEN 2.2 STL load often ignores the constructor ``material=`` argument
    and keeps the default white. ``shape.material.set_base_color`` updates the
    Python object but not the GPU slot — ``set_material`` is required.
    Visual only; collision is the palm box.
    """
    rgba = hand_rgba(color)
    bound = _hand_skin_material(scene, color) if scene is not None else None
    getter = getattr(actor, "get_visual_bodies", None)
    if not callable(getter):
        return
    try:
        bodies = getter()
    except Exception:
        return
    for body in bodies or ():
        shapes_fn = getattr(body, "get_render_shapes", None)
        if not callable(shapes_fn):
            continue
        try:
            shapes = shapes_fn()
        except Exception:
            continue
        for shape in shapes or ():
            if bound is not None:
                setter = getattr(shape, "set_material", None)
                if callable(setter):
                    try:
                        setter(bound)
                        continue
                    except Exception:
                        pass
            mat = getattr(shape, "material", None)
            if mat is None:
                continue
            try:
                mat.set_base_color(rgba)
            except Exception:
                try:
                    mat.base_color = rgba
                except Exception:
                    continue


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
    hand_color: Sequence[float],
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
        skin = hand_rgba(hand_color)
        rend_mtl = _hand_skin_material(scene, skin)
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
                builder.add_box_visual(half_size=half, pose=collision_pose, color=skin)

    actor = builder.build_kinematic(name) if kinematic else builder.build(name)
    actor.set_pose(sapien.Pose(HIDDEN_POS))
    if actor_type == "hand":
        apply_hand_skin_color(actor, hand_color, scene=scene)
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
        hand_color: Sequence[float],
    ) -> None:
        self._scene = scene
        self._models_dir = Path(models_dir)
        self._asset_dir = Path(asset_dir)
        self._hand_half_size = list(hand_half_size)
        self._hand_collision_offset = list(hand_collision_offset)
        self._density = float(density)
        self._hand_color = hand_rgba(hand_color)
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
            hand_color=self._hand_color,
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
        mass_kg: float | None = None,
        linear_velocity: np.ndarray | None = None,
    ) -> Any:
        """Swap a kinematic actor for a dynamic one at the same pose."""
        if self._actor is None:
            raise RuntimeError("no actor to release")
        if not self._actor_is_kinematic:
            return self._actor
        pose = self._actor.get_pose()
        if mass_kg is not None and actor_type == "hand":
            half = np.asarray(self._hand_half_size, dtype=np.float64) * float(hand_scale)
            volume = 8.0 * float(np.prod(half))
            if volume > 1e-12:
                self._density = float(mass_kg) / volume
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
        if mass_kg is not None:
            setter = getattr(actor, "set_mass", None)
            if callable(setter):
                setter(float(mass_kg))
        vel = (
            np.zeros(3, dtype=np.float64)
            if linear_velocity is None
            else np.asarray(linear_velocity, dtype=np.float64).reshape(3)
        )
        actor.set_velocity(vel)
        actor.set_angular_velocity(np.zeros(3))
        return actor
