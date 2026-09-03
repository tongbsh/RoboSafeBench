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


@dataclass(frozen=True)
class VisualSpec:
    name: str
    mesh_path: Path
    rgba: tuple[float, float, float, float]
    texture_path: Path | None = None


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
        texture = _optional_texture_path(
            spec.get("texture_path"),
            where=f"extra.robocasa.l12.visuals.{key}.texture_path",
        )
        catalog.append(
            VisualSpec(name=key, mesh_path=Path(mesh), rgba=rgba, texture_path=texture)
        )
    return catalog


def _optional_texture_path(raw: Any, *, where: str) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{where} must be a non-empty path when set")
    return Path(text)


def obj_axis_bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Axis-aligned vertex bounds from a Wavefront OBJ (``v x y z`` only)."""
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    count = 0
    with Path(path).open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            mins[0] = min(mins[0], x)
            maxs[0] = max(maxs[0], x)
            mins[1] = min(mins[1], y)
            maxs[1] = max(maxs[1], y)
            mins[2] = min(mins[2], z)
            maxs[2] = max(maxs[2], z)
            count += 1
    if count < 1:
        raise ValueError(f"mesh has no vertices: {path}")
    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])


def robocasa_mesh_scale(mesh_path: Path) -> tuple[float, float, float]:
    """Use the object's ``model.xml`` mesh scale when present (Objaverse is not 1 m)."""
    model_xml = Path(mesh_path).resolve().parent.parent / "model.xml"
    if not model_xml.is_file():
        return (1.0, 1.0, 1.0)
    target = Path(mesh_path).name
    tree = ET.parse(model_xml)
    for mesh in tree.iter("mesh"):
        file_attr = str(mesh.get("file") or "")
        if Path(file_attr).name != target:
            continue
        raw = mesh.get("scale")
        if raw is None or not str(raw).strip():
            return (1.0, 1.0, 1.0)
        parts = [float(x) for x in str(raw).split()]
        if len(parts) == 1:
            s = parts[0]
            if s <= 0.0:
                raise ValueError(f"mesh scale must be > 0 in {model_xml}, got {s}")
            return (s, s, s)
        if len(parts) == 3:
            if min(parts) <= 0.0:
                raise ValueError(f"mesh scale must be > 0 in {model_xml}, got {parts}")
            return (parts[0], parts[1], parts[2])
        raise ValueError(f"mesh scale in {model_xml} must be 1 or 3 floats, got {raw!r}")
    return (1.0, 1.0, 1.0)


def union_obj_axis_bounds(
    paths: Sequence[Path],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Axis-aligned union of several Wavefront OBJs."""
    umins = [float("inf"), float("inf"), float("inf")]
    umaxs = [float("-inf"), float("-inf"), float("-inf")]
    any_mesh = False
    for path in paths:
        mins, maxs = obj_axis_bounds(path)
        any_mesh = True
        for i in range(3):
            umins[i] = min(umins[i], mins[i])
            umaxs[i] = max(umaxs[i], maxs[i])
    if not any_mesh:
        raise ValueError("union_obj_axis_bounds needs at least one mesh")
    return (umins[0], umins[1], umins[2]), (umaxs[0], umaxs[1], umaxs[2])


def fit_from_bounds(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    *,
    half_z: float,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Center XY and sit ``mins[2]`` on the collision-box floor."""
    height = maxs[2] - mins[2]
    if height <= 1e-6:
        raise ValueError(f"mesh height must be > 0, got {height}")
    sx, sy, sz = (float(scale[0]), float(scale[1]), float(scale[2]))
    if min(sx, sy, sz) <= 0.0:
        raise ValueError(f"mesh scale must be > 0, got {scale}")
    cx = 0.5 * (mins[0] + maxs[0])
    cy = 0.5 * (mins[1] + maxs[1])
    pos = (-cx * sx, -cy * sy, -float(half_z) - mins[2] * sz)
    return (sx, sy, sz), pos


def fit_visual_scale_pos(
    mesh_path: Path,
    *,
    half_z: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Keep RoboCasa's authored meters. Center XY and sit the bottom on the box floor."""
    mins, maxs = obj_axis_bounds(mesh_path)
    return fit_from_bounds(mins, maxs, half_z=half_z, scale=robocasa_mesh_scale(mesh_path))


def _visual_files_ready(specs: Sequence[VisualSpec]) -> bool:
    if not specs:
        return False
    for spec in specs:
        if not spec.mesh_path.is_file():
            return False
        if spec.texture_path is not None and not spec.texture_path.is_file():
            return False
    return True


def _parse_named_visual_list(raw_parts: Any, *, key: str) -> list[VisualSpec]:
    if not isinstance(raw_parts, Sequence) or isinstance(raw_parts, (str, bytes)):
        raise ValueError(f"extra.robocasa.l3.{key} must be a list")
    parts: list[VisualSpec] = []
    for i, raw in enumerate(raw_parts):
        if not isinstance(raw, Mapping):
            raise ValueError(f"extra.robocasa.l3.{key}[{i}] must be a map")
        where = f"extra.robocasa.l3.{key}[{i}]"
        name = str(_require_map_key(raw, "name", where=where)).strip()
        mesh = str(_require_map_key(raw, "mesh_path", where=where)).strip()
        if not name or not mesh:
            raise ValueError(f"{where} needs name and mesh_path")
        rgba = parse_rgba(raw.get("rgba"), where=f"{where}.rgba")
        texture = _optional_texture_path(raw.get("texture_path"), where=f"{where}.texture_path")
        parts.append(VisualSpec(name=name, mesh_path=Path(mesh), rgba=rgba, texture_path=texture))
    return parts


def parse_l3_visual_parts(l3: Mapping[str, Any]) -> list[VisualSpec]:
    """Prefer Rocketbox (clothed) → local SMPL → in-tree clothed boxes.

    Missing optional files fall through. Collision half_size is not read here.
    """
    rocket_raw = l3.get("rocketbox_visuals")
    if rocket_raw is not None:
        rocket = _parse_named_visual_list(rocket_raw, key="rocketbox_visuals")
        if _visual_files_ready(rocket):
            return rocket
    smpl_raw = l3.get("smpl_mesh_path")
    if smpl_raw is not None and str(smpl_raw).strip():
        smpl_path = Path(str(smpl_raw).strip())
        if smpl_path.is_file():
            rgba = parse_rgba(
                l3.get("smpl_mesh_rgba", l3.get("mesh_rgba")),
                where="extra.robocasa.l3.smpl_mesh_rgba",
            )
            return [VisualSpec(name="smpl_body", mesh_path=smpl_path, rgba=rgba)]
    parts_raw = l3.get("visual_parts")
    parts: list[VisualSpec] = []
    if parts_raw is not None:
        parts = _parse_named_visual_list(parts_raw, key="visual_parts")
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
    texture_path: Path | None = None,
    scale_xyz: tuple[float, float, float] | None = None,
    pos: tuple[float, float, float] | None = None,
) -> str:
    """Add a non-colliding visual mesh; hide duplicate box visuals."""
    from robosuite.utils.mjcf_utils import array_to_string, new_geom

    color = [float(x) for x in rgba]
    if len(color) != 4:
        raise ValueError(f"mesh rgba must be length 4, got {color!r}")
    mesh_path = Path(mesh_path).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"RoboCasa visual mesh missing: {mesh_path}")
    tex_path: Path | None = None
    if texture_path is not None:
        tex_path = Path(texture_path).resolve()
        if not tex_path.is_file():
            raise FileNotFoundError(f"RoboCasa visual texture missing: {tex_path}")
    slug = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    prefix = str(box.naming_prefix)
    mesh_name = f"{prefix}{slug}"
    geom_name = f"{prefix}{slug}_visual"
    tex_name = f"{prefix}{slug}_tex"
    mat_name = f"{prefix}{slug}_mat"
    if scale_xyz is None or pos is None:
        scale_xyz, pos = fit_visual_scale_pos(mesh_path, half_z=float(half_z))
    asset = box.asset
    already = any(child.tag == "mesh" and child.get("name") == mesh_name for child in list(asset))
    if not already:
        scale = f"{scale_xyz[0]:.6f} {scale_xyz[1]:.6f} {scale_xyz[2]:.6f}"
        ET.SubElement(
            asset,
            "mesh",
            {"name": mesh_name, "file": str(mesh_path), "scale": scale},
        )
        if tex_path is not None:
            if not any(child.tag == "texture" and child.get("name") == tex_name for child in list(asset)):
                ET.SubElement(
                    asset,
                    "texture",
                    {"name": tex_name, "type": "2d", "file": str(tex_path)},
                )
            if not any(child.tag == "material" and child.get("name") == mat_name for child in list(asset)):
                ET.SubElement(
                    asset,
                    "material",
                    {
                        "name": mat_name,
                        "texture": tex_name,
                        "rgba": "1 1 1 1",
                    },
                )
    obj = box.get_obj()
    for geom in obj.iter("geom"):
        if geom.get("type") == "mesh" or geom.get("mesh"):
            continue
        geom.set("rgba", "0 0 0 0")
    if any(geom.get("name") == geom_name for geom in obj.iter("geom")):
        return geom_name
    geom = new_geom(
        name=geom_name,
        type="mesh",
        size=(1, 1, 1),
        pos=pos,
        group=1,
        mesh=mesh_name,
        contype="0",
        conaffinity="0",
        mass="1e-8",
        rgba=array_to_string(color),
    )
    if tex_path is not None:
        geom.set("material", mat_name)
    obj.append(geom)
    visual = getattr(box, "_visual_geoms", None)
    if isinstance(visual, list) and geom_name not in visual:
        visual.append(geom_name)
    return geom_name


def attach_visual_specs(
    box: Any,
    specs: Sequence[VisualSpec],
    *,
    half_z: float,
    visible_names: Sequence[str] | None = None,
    shared_frame: bool = False,
) -> dict[str, str]:
    allowed = None if visible_names is None else {str(name) for name in visible_names}
    shared: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    if shared_frame and specs:
        mins, maxs = union_obj_axis_bounds([spec.mesh_path for spec in specs])
        shared = fit_from_bounds(mins, maxs, half_z=float(half_z))
    mapping: dict[str, str] = {}
    for spec in specs:
        rgba = spec.rgba if allowed is None or spec.name in allowed else (0.0, 0.0, 0.0, 0.0)
        extra: dict[str, Any] = {}
        if shared is not None:
            extra["scale_xyz"] = shared[0]
            extra["pos"] = shared[1]
        mapping[spec.name] = attach_named_visual(
            box,
            name=spec.name,
            mesh_path=spec.mesh_path,
            rgba=rgba,
            half_z=half_z,
            texture_path=spec.texture_path,
            **extra,
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
