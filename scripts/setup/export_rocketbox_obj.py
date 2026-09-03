#!/usr/bin/env python3
"""Convert one Microsoft Rocketbox FBX avatar to MuJoCo OBJ + PNG.

Does not touch SMPL weights. Collision boxes stay in YAML.

Pipeline:
  1. assimp export FBX → Wavefront OBJ (one mesh per material)
  2. TGA color maps → PNG (MuJoCo rejects JPEG/TGA)
  3. Unity Y-up centimeters → MuJoCo Z-up meters, feet on z=0, XY centered

Default pose is the authored T-pose (arms out). Size is not stretched to
the L3 1.1 m collision column.

Requires: assimp CLI, Pillow.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# Unity (X right, Y up, Z forward) cm → MuJoCo (X forward, Y left, Z up) m.
_CM_TO_M = 0.01
# Bind pose is T/A-pose (arms out). Fold around approximate shoulders.
_ARM_Y = 0.20
_ARM_Z0 = 0.72
_ARM_Z1 = 1.40
_SHOULDER_Y = 0.155
_SHOULDER_Z = 1.30
_FOLD_RAD = 0.70  # ~40° extra toward -Z


def _unity_cm_to_mujoco_m(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (z * _CM_TO_M, x * _CM_TO_M, y * _CM_TO_M)


def _fold_tpose_arm(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Rotate T/A-pose arm verts down around a fixed shoulder (Y = left/right)."""
    if abs(y) < _ARM_Y or z < _ARM_Z0 or z > _ARM_Z1:
        return (x, y, z)
    if y > 0.0:
        pivot_y = _SHOULDER_Y
        angle = -_FOLD_RAD
    else:
        pivot_y = -_SHOULDER_Y
        angle = _FOLD_RAD
    dy = y - pivot_y
    dz = z - _SHOULDER_Z
    c, s = math.cos(angle), math.sin(angle)
    return (x, pivot_y + dy * c - dz * s, _SHOULDER_Z + dy * s + dz * c)


def _plant(verts: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    ox = 0.5 * (min(xs) + max(xs))
    oy = 0.5 * (min(ys) + max(ys))
    oz = min(zs)
    return [(x - ox, y - oy, z - oz) for x, y, z in verts]


def _parse_face_corner(token: str) -> tuple[int, int | None, int | None]:
    parts = token.split("/")
    vi = int(parts[0])
    vti = int(parts[1]) if len(parts) > 1 and parts[1] else None
    vni = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return vi, vti, vni


def _relabel(index: int, mapping: dict[int, int], nxt: list[int]) -> int:
    if index not in mapping:
        mapping[index] = nxt[0]
        nxt[0] += 1
    return mapping[index]


def convert_tga_to_png(src: Path, dest: Path) -> None:
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGBA").save(dest, format="PNG")


def export_fbx_obj(fbx: Path, obj: Path) -> None:
    obj.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["assimp", "export", str(fbx), str(obj)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not obj.is_file():
        raise SystemExit(
            "assimp export failed. Install assimp-utils.\n"
            f"cmd: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
        )


def split_and_plant(src_obj: Path, out_dir: Path) -> dict[str, Path]:
    """Split by usemtl, plant once so body/head stay registered."""
    verts: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, ...]] = []
    normals: list[tuple[float, float, float]] = []
    faces_by_mtl: dict[str, list[list[tuple[int, int | None, int | None]]]] = defaultdict(list)
    current = "default"

    with src_obj.open(encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("v "):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                verts.append(_unity_cm_to_mujoco_m(x, y, z))
            elif line.startswith("vt "):
                parts = line.split()
                texcoords.append(tuple(float(p) for p in parts[1:]))
            elif line.startswith("vn "):
                parts = line.split()
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("usemtl "):
                current = line.split(None, 1)[1].strip() or "default"
            elif line.startswith("f "):
                corners = [_parse_face_corner(tok) for tok in line.split()[1:]]
                if len(corners) >= 3:
                    faces_by_mtl[current].append(corners)

    if not verts:
        raise SystemExit(f"no vertices in {src_obj}")

    planted = _plant(verts)
    body_ids = {
        (vi if vi > 0 else len(planted) + 1 + vi) - 1
        for face in faces_by_mtl.get("f110_body", [])
        for vi, _vti, _vni in face
    }
    folded = list(planted)
    for i in body_ids:
        folded[i] = _fold_tpose_arm(*planted[i])
    planted = _plant(folded)

    name_map = {
        "f110_body": "body",
        "f110_A_head": "head",
    }
    written: dict[str, Path] = {}
    for mtl, faces in faces_by_mtl.items():
        slug = name_map.get(mtl, mtl.replace(" ", "_"))
        dest = out_dir / f"{slug}.obj"
        v_map: dict[int, int] = {}
        vt_map: dict[int, int] = {}
        vn_map: dict[int, int] = {}
        v_next = [1]
        vt_next = [1]
        vn_next = [1]
        out_v: list[tuple[float, float, float]] = []
        out_vt: list[tuple[float, ...]] = []
        out_vn: list[tuple[float, float, float]] = []
        out_f: list[str] = []
        for face in faces:
            bits: list[str] = []
            for vi, vti, vni in face:
                if vi < 0:
                    vi = len(planted) + 1 + vi
                new_v = _relabel(vi, v_map, v_next)
                if new_v > len(out_v):
                    out_v.append(planted[vi - 1])
                token = str(new_v)
                if vti is not None:
                    if vti < 0:
                        vti = len(texcoords) + 1 + vti
                    new_vt = _relabel(vti, vt_map, vt_next)
                    if new_vt > len(out_vt):
                        out_vt.append(texcoords[vti - 1])
                    token += f"/{new_vt}"
                    if vni is not None:
                        if vni < 0:
                            vni = len(normals) + 1 + vni
                        new_vn = _relabel(vni, vn_map, vn_next)
                        if new_vn > len(out_vn):
                            out_vn.append(normals[vni - 1])
                        token += f"/{new_vn}"
                elif vni is not None:
                    if vni < 0:
                        vni = len(normals) + 1 + vni
                    new_vn = _relabel(vni, vn_map, vn_next)
                    if new_vn > len(out_vn):
                        out_vn.append(normals[vni - 1])
                    token += f"//{new_vn}"
                bits.append(token)
            out_f.append("f " + " ".join(bits))
        lines = [
            f"# Rocketbox {slug} — Unity Y-up cm → MuJoCo Z-up m, planted",
            f"o {slug}",
            f"g {slug}",
        ]
        lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in out_v)
        lines.extend("vt " + " ".join(f"{c:.6f}" for c in vt) for vt in out_vt)
        lines.extend(f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in out_vn)
        lines.extend(out_f)
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[slug] = dest
        height = max(z for _, _, z in out_v) - min(z for _, _, z in out_v)
        width = max(x for x, _, _ in out_v) - min(x for x, _, _ in out_v)
        depth = max(y for _, y, _ in out_v) - min(y for _, y, _ in out_v)
        print(f"{slug}: verts={len(out_v)} size_m≈{width:.3f} x {depth:.3f} x {height:.3f} → {dest}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        help="Directory with Chef_Female_01.fbx and f110_*_color*.tga",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write body.obj / head.obj / *.png here (default: parent of --src-dir)",
    )
    args = parser.parse_args(argv)

    workspace = Path(__file__).resolve().parents[2]
    src_dir = (
        args.src_dir
        if args.src_dir is not None
        else workspace / "external/assets/rocketbox/Chef_Female_01/src"
    )
    src_dir = src_dir.resolve()
    out_dir = (args.out_dir if args.out_dir is not None else src_dir.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fbx = src_dir / "Chef_Female_01.fbx"
    if not fbx.is_file():
        raise SystemExit(f"missing FBX: {fbx}")
    raw_obj = src_dir / "Chef_Female_01.obj"
    if not raw_obj.is_file():
        export_fbx_obj(fbx, raw_obj)

    body_tga = src_dir / "f110_body_color.tga"
    head_tga = src_dir / "f110_head_color_a.tga"
    for tex in (body_tga, head_tga):
        if not tex.is_file() or tex.stat().st_size < 8_000_000:
            raise SystemExit(f"incomplete color TGA (need full ~12 MiB): {tex}")
    convert_tga_to_png(body_tga, out_dir / "body.png")
    convert_tga_to_png(head_tga, out_dir / "head.png")
    print(f"textures → {out_dir / 'body.png'} {out_dir / 'head.png'}")

    written = split_and_plant(raw_obj, out_dir)
    if "body" not in written or "head" not in written:
        raise SystemExit(f"expected body+head materials, got {sorted(written)}")

    license_src = src_dir.parent / "LICENSE.md"
    license_dest = out_dir / "LICENSE.md"
    if license_src.is_file() and license_src.resolve() != license_dest.resolve():
        shutil.copy2(license_src, license_dest)
    print(f"wrote Rocketbox visuals under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
