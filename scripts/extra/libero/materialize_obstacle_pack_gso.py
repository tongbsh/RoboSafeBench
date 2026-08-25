#!/usr/bin/env python3
"""Copy selected GSO scans into the reversible LIBERO obstacle pack.

Reads a sparse/full clone of kevinzakka/mujoco_scanned_objects and writes
meshes under ``src/dynisafety/backends/libero/assets/obstacle_pack_gso``.

Runtime remesh is per unused carrier slot (see obstacle_pack._ensure_carrier_xml).
Official on-stage types keep stock visual + collision.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# GSO folder name inside models/
GSO_SOURCES = {
    "gso_weston_tonic_22": "Weston_No_22_Cajun_Jerky_Tonic_12_fl_oz_nLj64ZnGwDh",
    "gso_weston_tonic_33": "Weston_No_33_Signature_Sausage_Tonic_12_fl_oz",
    "gso_ace_mug": "ACE_Coffee_Mug_Kristen_16_oz_cup",
    "gso_threshold_mug": "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White",
}

# Bake a Z-up, zmin≈0 GSO mesh into each donor body frame.
# pos/quat applied on a wrapper body around the geoms.
# quat is MuJoCo wxyz.
ALIGN = {
    "moka_pot_obstacle": {
        "asset": "gso_weston_tonic_22",
        "pos": (0.0, 0.0, -0.1074),
        "quat_wxyz": (1.0, 0.0, 0.0, 0.0),
    },
    "moka_pot_small_obstacle": {
        "asset": "gso_weston_tonic_22",
        "pos": (0.0, 0.0, -0.0671),
        "quat_wxyz": (1.0, 0.0, 0.0, 0.0),
    },
    "milk_obstacle": {
        "asset": "gso_weston_tonic_33",
        "pos": (0.0, -0.1428, 0.0),
        "quat_wxyz": (0.70711, -0.70711, 0.0, 0.0),
    },
    "milk_small_obstacle": {
        "asset": "gso_weston_tonic_33",
        "pos": (0.0, -0.0714, 0.0),
        "quat_wxyz": (0.70711, -0.70711, 0.0, 0.0),
    },
    "white_storage_box_obstacle": {
        "asset": "gso_threshold_mug",
        "pos": (0.0, 0.0, 0.0),
        "quat_wxyz": (1.0, 0.0, 0.0, 0.0),
    },
    "yellow_book_obstacle": {
        "asset": "gso_ace_mug",
        "pos": (0.0, 0.0, 0.0),
        "quat_wxyz": (1.0, 0.0, 0.0, 0.0),
    },
}

_MTL = """newmtl material_0
Ns 250
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.500 0.500 0.500
map_Kd texture.png
"""

_README = """# LIBERO GSO obstacle pack

Adds similar-sized extras on **unused** SafeLIBERO obstacle slots.
Official types keep their stock visual and collision. Does not change
BDDL or `pruned_init`. Tasks that already use all 6 slots are unchanged.

- Enable: `--set extra.libero.obstacle_pack.enabled=true`
- Disable (official types only): `enabled=false` (default)

`official_identities.json` lists each task's BDDL slots and the Level-I
instance that is on-stage in each of the 50 init states. Pack planning
steals surplus episodes from those official types so extras also appear.

Meshes are Google Scanned Objects via
[kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects)
(CC BY 4.0). See `manifest.yaml` and `THIRD_PARTY_NOTICES.md`.

Regenerate meshes:

```
python scripts/extra/libero/materialize_obstacle_pack_gso.py \\
  --gso-root /tmp/gso_full/mujoco_scanned_objects
```
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_pack_root() -> Path:
    return (
        _repo_root()
        / "src"
        / "dynisafety"
        / "backends"
        / "libero"
        / "assets"
        / "obstacle_pack_gso"
    )


def _load_obj_verts(path: Path) -> list[tuple[float, float, float]]:
    verts: list[tuple[float, float, float]] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise ValueError(f"no vertices in {path}")
    return verts


def _aabb(verts: list[tuple[float, float, float]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    xs, ys, zs = zip(*verts)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _rotate(quat_wxyz: tuple[float, float, float, float], vec: tuple[float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = quat_wxyz
    vx, vy, vz = vec
    # q * v * q^{-1}
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _copy_gso_meshes(src_dir: Path, dest_dir: Path) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / "model.obj", dest_dir / "model.obj")
    shutil.copy2(src_dir / "texture.png", dest_dir / "texture.png")
    (dest_dir / "material_0.mtl").write_text(_MTL)
    collisions: list[str] = []
    for col in sorted(src_dir.glob("model_collision_*.obj"), key=lambda p: int(p.stem.split("_")[-1])):
        shutil.copy2(col, dest_dir / col.name)
        collisions.append(col.name)
    if not collisions:
        raise FileNotFoundError(f"no collision objs in {src_dir}")
    return collisions


def _write_xml(
    *,
    xml_path: Path,
    mesh_rel: str,
    collisions: list[str],
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    verts: list[tuple[float, float, float]],
) -> None:
    mn, mx = _aabb(verts)
    corners = [
        (x, y, z)
        for x in (mn[0], mx[0])
        for y in (mn[1], mx[1])
        for z in (mn[2], mx[2])
    ]
    world = []
    for corner in corners:
        rx, ry, rz = _rotate(quat, corner)
        world.append((rx + pos[0], ry + pos[1], rz + pos[2]))
    wmin, wmax = _aabb(world)
    r_xy = 0.5 * max(wmax[0] - wmin[0], wmax[1] - wmin[1])
    col_xml = []
    for name in collisions:
        col_xml.append(
            f'        <geom type="mesh" mesh="{name[:-4]}" group="0" '
            f'rgba="0.8 0.8 0.8 0.3" density="100" friction="0.95 0.3 0.1" '
            f'solimp="0.998 0.998 0.001" solref="0.001 1"/>'
        )
    xml = f"""<mujoco model="{xml_path.stem}">
  <asset>
    <texture type="2d" name="tex-gso" file="{mesh_rel}/texture.png"/>
    <material name="mat-gso" texture="tex-gso" reflectance="0.5" specular="0.5"/>
    <mesh name="visual" file="{mesh_rel}/model.obj"/>
{chr(10).join(f'    <mesh name="{c[:-4]}" file="{mesh_rel}/{c}"/>' for c in collisions)}
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <body name="gso_align" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}">
        <geom type="mesh" mesh="visual" material="mat-gso" group="1"
              conaffinity="0" contype="0" density="100" friction="0.95 0.3 0.1"
              solimp="0.998 0.998 0.001" solref="0.001 1"/>
{chr(10).join(col_xml)}
        </body>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {wmin[2] - 0.001:.5f}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {wmax[2]:.5f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{r_xy:.5f} {r_xy:.5f} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml)


def _write_manifest(pack_root: Path) -> None:
    lines = [
        "license: CC-BY-4.0",
        "upstream: https://github.com/kevinzakka/mujoco_scanned_objects",
        "google_scanned_objects: https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research",
        "assets:",
    ]
    for asset_id, folder in GSO_SOURCES.items():
        lines.append(f"  {asset_id}: {folder}")
    lines.append("slots:")
    for slot, spec in ALIGN.items():
        lines.append(f"  {slot}: {spec['asset']}")
    (pack_root / "manifest.yaml").write_text("\n".join(lines) + "\n")
    (pack_root / "README.md").write_text(_README)


def materialize(gso_root: Path, pack_root: Path) -> None:
    models = gso_root / "models"
    if not models.is_dir():
        raise FileNotFoundError(f"GSO models/ not found: {models}")

    meshes_root = pack_root / "meshes"
    collision_names: dict[str, list[str]] = {}
    verts_by_asset: dict[str, list[tuple[float, float, float]]] = {}
    for asset_id, folder in GSO_SOURCES.items():
        src = models / folder
        if not (src / "model.obj").is_file():
            raise FileNotFoundError(f"missing {src / 'model.obj'}")
        dest = meshes_root / asset_id
        collision_names[asset_id] = _copy_gso_meshes(src, dest)
        verts_by_asset[asset_id] = _load_obj_verts(dest / "model.obj")
        print(f"[pack] meshes {asset_id} <- {folder}")

    for slot, spec in ALIGN.items():
        asset_id = str(spec["asset"])
        rel = f"../meshes/{asset_id}"
        _write_xml(
            xml_path=pack_root / slot / f"{slot}.xml",
            mesh_rel=rel,
            collisions=collision_names[asset_id],
            pos=tuple(spec["pos"]),  # type: ignore[arg-type]
            quat=tuple(spec["quat_wxyz"]),  # type: ignore[arg-type]
            verts=verts_by_asset[asset_id],
        )
        print(f"[pack] xml {slot} -> {asset_id}")

    _write_manifest(pack_root)
    print(f"[pack] wrote {pack_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gso-root",
        type=Path,
        default=Path("/tmp/gso_full/mujoco_scanned_objects"),
        help="Clone root that contains models/<GSO name>/",
    )
    parser.add_argument(
        "--pack-root",
        type=Path,
        default=_default_pack_root(),
        help="Output pack directory",
    )
    args = parser.parse_args()
    materialize(args.gso_root.resolve(), args.pack_root.resolve())


if __name__ == "__main__":
    main()
