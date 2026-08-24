"""Optional SMPL-family standing mesh export (not the default RoboCasa L3 visual).

Default L3 visuals are original clothed parts under
``backends/robocasa/assets/pedestrian/``. This module remains for local
research rebuilds: it skins a user-provided SMPL / SMPL+H / SMPL-X ``*.npz``,
drops T-pose arms, and writes a Z-up / +X-facing OBJ.

SMPL-family weights are not in the tree (MPI registration) and must not be
committed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_L_SHOULDER = 16
_R_SHOULDER = 17
_ARM_DOWN_L = np.array([0.0, 0.0, -0.5 * np.pi], dtype=np.float64)
_ARM_DOWN_R = np.array([0.0, 0.0, 0.5 * np.pi], dtype=np.float64)
_UINT32_ROOT = 4294967295


@dataclass(frozen=True)
class StandingMeshStats:
    n_verts: int
    n_faces: int
    n_joints: int
    height_m: float
    width_m: float
    depth_m: float
    min_z: float
    center_xy: tuple[float, float]
    source: str


def _as_dense(arr: Any) -> np.ndarray:
    if hasattr(arr, "toarray"):
        return np.asarray(arr.toarray(), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    out = np.zeros((rotvec.shape[0], 3, 3), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for i, vec in enumerate(rotvec):
        theta = float(np.linalg.norm(vec))
        if theta < 1e-8:
            out[i] = eye
            continue
        axis = vec / theta
        kx, ky, kz = axis
        skew = np.array(
            [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
            dtype=np.float64,
        )
        out[i] = eye + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
    return out


def _parents(kintree_table: np.ndarray, n_joints: int) -> np.ndarray:
    raw = np.asarray(kintree_table)
    if raw.ndim != 2 or raw.shape[1] < n_joints:
        raise ValueError(f"kintree_table must be (2, {n_joints}), got {tuple(raw.shape)}")
    parents = np.asarray(raw[0, :n_joints], dtype=np.int64)
    parents[parents == _UINT32_ROOT] = -1
    parents[0] = -1
    return parents


def lbs_posed_vertices(
    model: Mapping[str, Any],
    *,
    pose: np.ndarray,
    betas: np.ndarray | None = None,
) -> np.ndarray:
    """Linear-blend-skin ``v_template`` with axis-angle ``pose`` (J, 3)."""
    v_template = np.asarray(model["v_template"], dtype=np.float64).reshape(-1, 3)
    shapedirs = _as_dense(model["shapedirs"])
    posedirs = _as_dense(model["posedirs"])
    regressor = _as_dense(model["J_regressor"])
    weights = _as_dense(model["weights"])
    n_verts = int(v_template.shape[0])
    n_joints = int(regressor.shape[0])
    if shapedirs.shape[:2] != (n_verts, 3):
        raise ValueError(f"shapedirs shape {shapedirs.shape} does not match v_template {v_template.shape}")
    if weights.shape != (n_verts, n_joints):
        raise ValueError(f"weights shape {weights.shape} != ({n_verts}, {n_joints})")
    if regressor.shape[1] != n_verts:
        raise ValueError(f"J_regressor shape {regressor.shape} does not match {n_verts} verts")

    pose = np.asarray(pose, dtype=np.float64).reshape(n_joints, 3)
    n_shape = int(shapedirs.shape[2])
    if betas is None:
        beta_vec = np.zeros((n_shape,), dtype=np.float64)
    else:
        beta_vec = np.zeros((n_shape,), dtype=np.float64)
        src = np.asarray(betas, dtype=np.float64).reshape(-1)
        take = min(src.size, n_shape)
        beta_vec[:take] = src[:take]
    v_shaped = v_template + np.tensordot(shapedirs, beta_vec, axes=([2], [0]))
    joints = regressor @ v_shaped

    rot = _rodrigues(pose)
    pose_feat = (rot[1:] - np.eye(3, dtype=np.float64)).reshape(-1)
    if posedirs.shape != (n_verts, 3, pose_feat.size):
        raise ValueError(
            f"posedirs shape {posedirs.shape} does not match pose feature {pose_feat.size} "
            f"(expected (V, 3, 9*(J-1)) for J={n_joints})"
        )
    v_posed = v_shaped + np.tensordot(posedirs, pose_feat, axes=([2], [0]))

    parents = _parents(model["kintree_table"], n_joints)
    global_tf = np.zeros((n_joints, 4, 4), dtype=np.float64)
    for i in range(n_joints):
        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = rot[i]
        parent = int(parents[i])
        if parent < 0:
            local[:3, 3] = joints[i]
            global_tf[i] = local
        else:
            local[:3, 3] = joints[i] - joints[parent]
            global_tf[i] = global_tf[parent] @ local

    rest = np.eye(4, dtype=np.float64)[None, :, :].repeat(n_joints, axis=0)
    rest[:, :3, 3] = joints
    skin = global_tf @ np.linalg.inv(rest)
    blended = np.einsum("vj,jab->vab", weights, skin)
    homo = np.concatenate([v_posed, np.ones((n_verts, 1), dtype=np.float64)], axis=1)
    return np.einsum("vab,vb->va", blended, homo)[:, :3]


def standing_pose(n_joints: int) -> np.ndarray:
    """T-pose → arms hanging; extra SMPL+H / SMPL-X hand joints stay at rest."""
    if n_joints < 18:
        raise ValueError(f"SMPL-family models need >= 18 joints, got {n_joints}")
    pose = np.zeros((n_joints, 3), dtype=np.float64)
    pose[_L_SHOULDER] = _ARM_DOWN_L
    pose[_R_SHOULDER] = _ARM_DOWN_R
    return pose


def _to_mujoco_standing(verts: np.ndarray) -> np.ndarray:
    """Y-up SMPL (faces +Z) → Z-up MuJoCo (faces +X), feet on z=0, XY centered."""
    y_up = np.column_stack([verts[:, 0], -verts[:, 2], verts[:, 1]])
    z_up_face_x = np.column_stack([-y_up[:, 1], y_up[:, 0], y_up[:, 2]])
    z_up_face_x = z_up_face_x - np.array(
        [
            float(z_up_face_x[:, 0].mean()),
            float(z_up_face_x[:, 1].mean()),
            float(z_up_face_x[:, 2].min()),
        ],
        dtype=np.float64,
    )
    return z_up_face_x


def _faces(model: Mapping[str, Any]) -> np.ndarray:
    raw = model["f"] if "f" in model else model["faces"]
    faces = np.asarray(raw, dtype=np.int64).reshape(-1, 3)
    if int(faces.min()) < 0:
        raise ValueError("mesh faces contain negative indices")
    return faces


def load_smpl_npz(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"SMPL-family npz missing: {path}. "
            "Place a local SMPL+H model.npz or SMPLX_*.npz (not shipped)."
        )
    with np.load(path, allow_pickle=True) as packed:
        return {key: packed[key] for key in packed.files}


def export_standing_mesh(model_path: Path, dest: Path) -> StandingMeshStats:
    """Skin a standing pedestrian and write Wavefront OBJ (meters)."""
    model = load_smpl_npz(Path(model_path))
    n_joints = int(_as_dense(model["J_regressor"]).shape[0])
    verts = _to_mujoco_standing(lbs_posed_vertices(model, pose=standing_pose(n_joints)))
    faces = _faces(model)
    if int(faces.max()) >= verts.shape[0]:
        raise ValueError(f"face index {int(faces.max())} >= n_verts {verts.shape[0]}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write("# DyniSafety L3 standing pedestrian (SMPL-family LBS)\n")
        fh.write(f"# source {Path(model_path).name}; arms down; Z-up; +X forward; MPI weights not redistributed\n")
        for vx, vy, vz in verts:
            fh.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for a, b, c in faces:
            fh.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    return StandingMeshStats(
        n_verts=int(verts.shape[0]),
        n_faces=int(faces.shape[0]),
        n_joints=n_joints,
        height_m=float(maxs[2] - mins[2]),
        width_m=float(maxs[1] - mins[1]),
        depth_m=float(maxs[0] - mins[0]),
        min_z=float(mins[2]),
        center_xy=(float(verts[:, 0].mean()), float(verts[:, 1].mean())),
        source=str(Path(model_path)),
    )


def ensure_standing_mesh(*, model_path: Path, mesh_path: Path) -> Path:
    """Use an existing OBJ, or export one. Never silently fall back to a box."""
    mesh_path = Path(mesh_path)
    if mesh_path.is_file() and mesh_path.stat().st_size > 0:
        return mesh_path
    stats = export_standing_mesh(Path(model_path), mesh_path)
    if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
        raise RuntimeError(f"failed to export standing mesh to {mesh_path} from {model_path}")
    del stats
    return mesh_path


def attach_visual_mesh(
    box: Any,
    *,
    mesh_path: Path,
    rgba: Sequence[float],
    half_z: float,
    name: str = "smpl",
) -> str:
    """Add a non-colliding visual mesh; hide the duplicate box visual."""
    from dynisafety.backends.robocasa.visuals import attach_named_visual

    return attach_named_visual(
        box,
        name=name,
        mesh_path=Path(mesh_path),
        rgba=rgba,
        half_z=half_z,
    )
