"""Write original fallback meshes (L1/L2 primitives + clothed L3 pedestrian).

L1/L2 default visuals now point at RoboCasa Lightwheel / Objaverse textured
meshes in configs/backends/robocasa.yaml. These primitive OBJs stay as a
license-clean fallback. Collision half-sizes stay in YAML. L3 stays in-tree
so the open-source tree does not ship a nude SMPL body.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent
OBSTACLE_DIR = ASSETS / "obstacles"
PEDESTRIAN_DIR = ASSETS / "pedestrian"

# Authoring frame for L1/L2 visuals: bottom at z=0, XY centered, height 1.30 m.
# Fits extra.robocasa.l1.half_size [0.12, 0.12, 0.65]. L2 scales XY at attach.
OBSTACLE_HALF_XY = 0.12
OBSTACLE_HEIGHT = 1.30
PEDESTRIAN_HEIGHT = 1.70

POOL = (
    "cardboard_stack",
    "kitchen_bin",
    "delivery_crate",
    "laundry_hamper",
    "water_cooler",
)


class MeshBuf:
    def __init__(self) -> None:
        self.verts: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []

    def add_box(self, center: tuple[float, float, float], half: tuple[float, float, float]) -> None:
        cx, cy, cz = center
        hx, hy, hz = half
        base = len(self.verts)
        corners = (
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        )
        self.verts.extend(corners)
        quads = (
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (2, 6, 7, 3),
            (0, 3, 7, 4),
            (1, 5, 6, 2),
        )
        for a, b, c, d in quads:
            self.faces.append((base + a, base + b, base + c))
            self.faces.append((base + a, base + c, base + d))

    def add_cylinder(
        self,
        *,
        x: float,
        y: float,
        z0: float,
        z1: float,
        radius: float,
        n: int = 16,
    ) -> None:
        base = len(self.verts)
        self.verts.append((x, y, z0))
        self.verts.append((x, y, z1))
        for i in range(n):
            ang = 2.0 * np.pi * i / n
            px = x + radius * float(np.cos(ang))
            py = y + radius * float(np.sin(ang))
            self.verts.append((px, py, z0))
            self.verts.append((px, py, z1))
        for i in range(n):
            j = (i + 1) % n
            b0 = base + 2 + 2 * i
            b1 = base + 2 + 2 * j
            t0 = b0 + 1
            t1 = b1 + 1
            self.faces.append((b0, b1, t1))
            self.faces.append((b0, t1, t0))
            self.faces.append((base, b1, b0))
            self.faces.append((base + 1, t0, t1))

    def add_frustum(
        self,
        *,
        z0: float,
        z1: float,
        r0: float,
        r1: float,
        n: int = 12,
    ) -> None:
        base = len(self.verts)
        self.verts.append((0.0, 0.0, z0))
        self.verts.append((0.0, 0.0, z1))
        for i in range(n):
            ang = 2.0 * np.pi * i / n
            c = float(np.cos(ang))
            s = float(np.sin(ang))
            self.verts.append((r0 * c, r0 * s, z0))
            self.verts.append((r1 * c, r1 * s, z1))
        for i in range(n):
            j = (i + 1) % n
            b0 = base + 2 + 2 * i
            b1 = base + 2 + 2 * j
            t0 = b0 + 1
            t1 = b1 + 1
            self.faces.append((b0, b1, t1))
            self.faces.append((b0, t1, t0))
            self.faces.append((base, b1, b0))
            self.faces.append((base + 1, t0, t1))

    def write(self, path: Path, comment: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"# {comment}\n")
            for vx, vy, vz in self.verts:
                fh.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for a, b, c in self.faces:
                fh.write(f"f {a + 1} {b + 1} {c + 1}\n")


def _cardboard_stack() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_box((0.00, 0.00, 0.215), (0.115, 0.110, 0.215))
    mesh.add_box((0.006, -0.006, 0.650), (0.108, 0.114, 0.215))
    mesh.add_box((-0.005, 0.007, 1.085), (0.114, 0.106, 0.215))
    return mesh


def _kitchen_bin() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_frustum(z0=0.0, z1=1.12, r0=0.090, r1=0.116, n=12)
    mesh.add_box((0.0, 0.0, 1.145), (0.118, 0.118, 0.025))
    mesh.add_box((0.0, 0.0, 1.245), (0.110, 0.110, 0.055))
    return mesh


def _delivery_crate() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_box((0.0, 0.0, 0.65), (0.118, 0.118, 0.65))
    for z in (0.22, 0.55, 0.88, 1.18):
        mesh.add_box((0.0, 0.0, z), (0.120, 0.120, 0.028))
    return mesh


def _laundry_hamper() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_frustum(z0=0.0, z1=1.18, r0=0.088, r1=0.118, n=12)
    mesh.add_box((0.0, 0.0, 1.22), (0.120, 0.120, 0.025))
    mesh.add_box((0.0, 0.0, 1.275), (0.108, 0.108, 0.025))
    return mesh


def _water_cooler() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_box((0.0, 0.0, 0.40), (0.110, 0.110, 0.40))
    mesh.add_box((0.0, 0.08, 0.62), (0.035, 0.018, 0.04))
    mesh.add_cylinder(x=0.0, y=0.0, z0=0.80, z1=1.26, radius=0.090, n=16)
    mesh.add_cylinder(x=0.0, y=0.0, z0=1.26, z1=1.30, radius=0.040, n=12)
    return mesh


def _ped_shoes() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_box((-0.09, 0.02, 0.035), (0.055, 0.085, 0.035))
    mesh.add_box((0.09, 0.02, 0.035), (0.055, 0.085, 0.035))
    return mesh


def _ped_legs() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_box((-0.09, 0.00, 0.44), (0.050, 0.055, 0.38))
    mesh.add_box((0.09, 0.00, 0.44), (0.050, 0.055, 0.38))
    mesh.add_box((0.00, 0.00, 0.88), (0.145, 0.075, 0.08))
    return mesh


def _ped_torso() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_box((0.00, 0.00, 1.16), (0.16, 0.09, 0.22))
    mesh.add_box((0.00, 0.00, 1.36), (0.15, 0.08, 0.04))
    return mesh


def _ped_arms() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_box((-0.22, 0.00, 1.10), (0.045, 0.045, 0.20))
    mesh.add_box((0.22, 0.00, 1.10), (0.045, 0.045, 0.20))
    return mesh


def _ped_hands() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_box((-0.22, 0.00, 0.86), (0.040, 0.040, 0.055))
    mesh.add_box((0.22, 0.00, 0.86), (0.040, 0.040, 0.055))
    return mesh


def _ped_head() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_box((0.00, 0.00, 1.44), (0.055, 0.055, 0.04))
    mesh.add_cylinder(x=0.0, y=0.0, z0=1.46, z1=1.64, radius=0.095, n=14)
    return mesh


def _ped_hair() -> MeshBuf:
    mesh = MeshBuf()
    mesh.add_cylinder(x=0.0, y=0.0, z0=1.58, z1=1.70, radius=0.100, n=14)
    mesh.add_box((0.00, 0.07, 1.60), (0.090, 0.025, 0.055))
    return mesh


def generate_all(root: Path | None = None) -> dict[str, Path]:
    root = ASSETS if root is None else Path(root)
    obstacles = root / "obstacles"
    pedestrian = root / "pedestrian"
    written: dict[str, Path] = {}
    builders = {
        "cardboard_stack": (_cardboard_stack, "DyniSafety kitchen cardboard stack (original)"),
        "kitchen_bin": (_kitchen_bin, "DyniSafety tall kitchen bin (original)"),
        "delivery_crate": (_delivery_crate, "DyniSafety delivery crate (original)"),
        "laundry_hamper": (_laundry_hamper, "DyniSafety laundry hamper (original)"),
        "water_cooler": (_water_cooler, "DyniSafety water cooler (original)"),
    }
    for name, (builder, comment) in builders.items():
        path = obstacles / f"{name}.obj"
        builder().write(path, comment)
        written[name] = path
    ped = {
        "shoes": (_ped_shoes, "clothed pedestrian shoes"),
        "legs": (_ped_legs, "clothed pedestrian pants"),
        "torso": (_ped_torso, "clothed pedestrian shirt"),
        "arms": (_ped_arms, "clothed pedestrian sleeves"),
        "hands": (_ped_hands, "clothed pedestrian hands"),
        "head": (_ped_head, "clothed pedestrian head"),
        "hair": (_ped_hair, "clothed pedestrian hair"),
    }
    for name, (builder, comment) in ped.items():
        path = pedestrian / f"{name}.obj"
        builder().write(path, f"DyniSafety {comment} (original)")
        written[f"pedestrian_{name}"] = path
    return written


if __name__ == "__main__":
    paths = generate_all()
    for name, path in paths.items():
        print(f"{name}\t{path}")
