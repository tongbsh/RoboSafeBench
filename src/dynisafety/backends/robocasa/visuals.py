"""RoboCasa visual-only assets: L1/L2 pool rotation and clothed L3 parts.

Collision half-sizes stay in YAML. This module never swaps the free-joint box.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynisafety.core.disturbance import require_map_key as _require_map_key

L12_AUTHOR_HALF_XY = 0.12


@dataclass(frozen=True)
class VisualSpec:
    name: str
    mesh_path: Path
    rgba: tuple[float, float, float, float]


def select_visual_id(
    pool: Sequence[str],
    episode_index: int,
    selection_mode: str,
) -> str:
    """Pick a pool entry. ``trajectory_order`` is ``episode_index % N``."""
    names = [str(item).strip() for item in pool if str(item).strip()]
    if not names:
        raise ValueError("RoboCasa obstacle pool must contain at least one name")
    mode = str(selection_mode).strip().lower()
    if mode == "fixed":
        return names[0]
    if mode == "trajectory_order":
        return names[int(episode_index) % len(names)]
    raise ValueError(
        "disturbance.obstacle.selection_mode must be 'fixed' or "
        f"'trajectory_order' for RoboCasa, got {selection_mode!r}"
    )


def parse_rgba(raw: Any, *, where: str) -> tuple[float, float, float, float]:
    from dynisafety.backends.robocasa.env import parse_obstacle_rgba

    color = parse_obstacle_rgba(raw, where=where)
    return (float(color[0]), float(color[1]), float(color[2]), float(color[3]))


def parse_l12_visual_catalog(
    extra_robocasa: Mapping[str, Any],
    pool: Sequence[str],
) -> list[VisualSpec]:
    l12 = extra_robocasa.get("l12")
    if not isinstance(l12, Mapping):
        raise ValueError("extra.robocasa.l12 missing from resolved YAML")
    visuals = l12.get("visuals")
    if not isinstance(visuals, Mapping):
        raise ValueError("extra.robocasa.l12.visuals missing from resolved YAML")
    catalog: list[VisualSpec] = []
    for name in pool:
        key = str(name).strip()
        spec = visuals.get(key)
        if not isinstance(spec, Mapping):
            raise ValueError(f"extra.robocasa.l12.visuals.{key} missing from resolved YAML")
        mesh = str(_require_map_key(spec, "mesh_path", where=f"extra.robocasa.l12.visuals.{key}")).strip()
        if not mesh:
            raise ValueError(f"extra.robocasa.l12.visuals.{key}.mesh_path must be non-empty")
        rgba = parse_rgba(spec.get("rgba"), where=f"extra.robocasa.l12.visuals.{key}.rgba")
        catalog.append(VisualSpec(name=key, mesh_path=Path(mesh), rgba=rgba))
    return catalog


def parse_l3_visual_parts(l3: Mapping[str, Any]) -> list[VisualSpec]:
    parts_raw = l3.get("visual_parts")
    parts: list[VisualSpec] = []
    if parts_raw is not None:
        if not isinstance(parts_raw, Sequence) or isinstance(parts_raw, (str, bytes)):
            raise ValueError("extra.robocasa.l3.visual_parts must be a list")
        for i, raw in enumerate(parts_raw):
            if not isinstance(raw, Mapping):
                raise ValueError(f"extra.robocasa.l3.visual_parts[{i}] must be a map")
            name = str(_require_map_key(raw, "name", where=f"extra.robocasa.l3.visual_parts[{i}]")).strip()
            mesh = str(_require_map_key(raw, "mesh_path", where=f"extra.robocasa.l3.visual_parts[{i}]")).strip()
            if not name or not mesh:
                raise ValueError(f"extra.robocasa.l3.visual_parts[{i}] needs name and mesh_path")
            rgba = parse_rgba(raw.get("rgba"), where=f"extra.robocasa.l3.visual_parts[{i}].rgba")
            parts.append(VisualSpec(name=name, mesh_path=Path(mesh), rgba=rgba))
    main = str(_require_map_key(l3, "mesh_path", where="extra.robocasa.l3")).strip()
    if not main:
        raise ValueError("extra.robocasa.l3.mesh_path must be non-empty")
    main_rgba = parse_rgba(l3.get("mesh_rgba"), where="extra.robocasa.l3.mesh_rgba")
    if not any(part.mesh_path.resolve() == Path(main).resolve() for part in parts):
        parts.insert(0, VisualSpec(name="body", mesh_path=Path(main), rgba=main_rgba))
    return parts


def attach_named_visual(
    box: Any,
    *,
    name: str,
    mesh_path: Path,
    rgba: Sequence[float],
    half_z: float,
    xy_scale: float = 1.0,
) -> str:
    """Add a non-colliding visual mesh; hide duplicate box visuals."""
    from robosuite.utils.mjcf_utils import array_to_string, new_geom

    color = [float(x) for x in rgba]
    if len(color) != 4:
        raise ValueError(f"mesh rgba must be length 4, got {color!r}")
    mesh_path = Path(mesh_path).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"RoboCasa visual mesh missing: {mesh_path}")
    slug = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    prefix = str(box.naming_prefix)
    mesh_name = f"{prefix}{slug}"
    geom_name = f"{prefix}{slug}_visual"
    asset = box.asset
    already = any(child.tag == "mesh" and child.get("name") == mesh_name for child in list(asset))
    if not already:
        scale = f"{float(xy_scale):.6f} {float(xy_scale):.6f} 1.000000"
        ET.SubElement(
            asset,
            "mesh",
            {"name": mesh_name, "file": str(mesh_path), "scale": scale},
        )
    obj = box.get_obj()
    for geom in obj.iter("geom"):
        if geom.get("type") == "mesh" or geom.get("mesh"):
            continue
        geom.set("rgba", "0 0 0 0")
    if any(geom.get("name") == geom_name for geom in obj.iter("geom")):
        return geom_name
    obj.append(
        new_geom(
            name=geom_name,
            type="mesh",
            size=(1, 1, 1),
            pos=(0.0, 0.0, -float(half_z)),
            group=1,
            mesh=mesh_name,
            contype="0",
            conaffinity="0",
            mass="1e-8",
            rgba=array_to_string(color),
        )
    )
    visual = getattr(box, "_visual_geoms", None)
    if isinstance(visual, list) and geom_name not in visual:
        visual.append(geom_name)
    return geom_name


def attach_visual_specs(
    box: Any,
    specs: Sequence[VisualSpec],
    *,
    half_z: float,
    xy_scale: float = 1.0,
    visible_names: Sequence[str] | None = None,
) -> dict[str, str]:
    allowed = None if visible_names is None else {str(name) for name in visible_names}
    mapping: dict[str, str] = {}
    for spec in specs:
        rgba = spec.rgba if allowed is None or spec.name in allowed else (0.0, 0.0, 0.0, 0.0)
        mapping[spec.name] = attach_named_visual(
            box,
            name=spec.name,
            mesh_path=spec.mesh_path,
            rgba=rgba,
            half_z=half_z,
            xy_scale=xy_scale,
        )
    return mapping


def apply_named_geom_rgba(env: Any, geom_name: str, rgba: Sequence[float]) -> bool:
    model = getattr(getattr(env, "sim", None), "model", None)
    if model is None:
        return False
    raw = getattr(model, "_model", model)
    if not hasattr(raw, "geom_rgba"):
        return False
    try:
        gid = int(model.geom_name2id(str(geom_name)))
    except Exception:
        return False
    raw.geom_rgba[gid, 0] = float(rgba[0])
    raw.geom_rgba[gid, 1] = float(rgba[1])
    raw.geom_rgba[gid, 2] = float(rgba[2])
    raw.geom_rgba[gid, 3] = float(rgba[3])
    return True


def apply_l12_visual(env: Any, visual_id: str) -> str | None:
    """Show one pool visual; hide the others. No-op when the catalog is absent."""
    meta = getattr(env, "_dynisafety_l12", None)
    if not isinstance(meta, dict):
        return None
    geoms = meta.get("visual_geoms")
    colors = meta.get("visual_rgba")
    if not isinstance(geoms, Mapping) or not geoms:
        return None
    chosen = str(visual_id)
    if chosen not in geoms:
        chosen = next(iter(geoms))
    for name, geom_name in geoms.items():
        rgba = (0.0, 0.0, 0.0, 0.0)
        if str(name) == chosen and isinstance(colors, Mapping):
            raw = colors.get(name)
            if isinstance(raw, Sequence) and len(raw) == 4:
                rgba = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        apply_named_geom_rgba(env, str(geom_name), rgba)
    meta["active_visual"] = chosen
    return chosen


def reapply_l12_visual(env: Any) -> str | None:
    """Restore the active pool mesh after contact hide/show zeros geom alpha.

    Rollout snapshots ``geom_contact`` before the episode picks a visual, so
    every mesh alpha is 0. Replaying that snapshot on appear would leave the
    obstacle invisible. Collision flags stay on the box; this only fixes paint.
    """
    meta = getattr(env, "_dynisafety_l12", None)
    if not isinstance(meta, dict):
        return None
    geoms = meta.get("visual_geoms")
    if not isinstance(geoms, Mapping) or not geoms:
        return None
    chosen = meta.get("active_visual")
    return apply_l12_visual(env, str(chosen or next(iter(geoms))))
