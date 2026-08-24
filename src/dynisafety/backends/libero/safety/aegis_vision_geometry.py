"""RGB-D → point cloud → ellipsoid (A3b/A3c). No GroundingDINO / torch imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class XyzBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    T_cam_to_world: np.ndarray
    camera_name: str


def cxcywh_to_xyxy_norm(boxes: np.ndarray) -> np.ndarray:
    """Normalized cxcywh (N,4) → normalized xyxy (N,4)."""
    arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    return np.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], axis=1)


def dino_box_to_native_xyxy(
    box_cxcywh_norm: np.ndarray,
    *,
    width: int,
    height: int,
    flip_180: bool,
) -> tuple[int, int, int, int] | None:
    """Map one DINO box to native pixel crop ``(xmin, ymin, xmax, ymax)``.

    Pixel conversion matches ``vlsa-aegis/main/utils.py`` ``get_point_cloud``:
    cxcywh→xyxy, scale by ``(w,h,w,h)``, ``astype(int)`` truncation, then
    ``xmin = w-1-x2`` when the DINO image was 180°-flipped.
    """
    xyxy = cxcywh_to_xyxy_norm(np.asarray(box_cxcywh_norm, dtype=np.float64).reshape(1, 4))[0]
    pixel = xyxy * np.array([width, height, width, height], dtype=np.float64)
    x1, y1, x2, y2 = pixel.astype(int).tolist()
    if flip_180:
        xmin, xmax = width - 1 - x2, width - 1 - x1
        ymin, ymax = height - 1 - y2, height - 1 - y1
    else:
        xmin, xmax, ymin, ymax = x1, x2, y1, y2
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(width, int(xmax))
    ymax = min(height, int(ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def unproject_depth_crop(
    depth_hw: np.ndarray,
    K: np.ndarray,
    t_cam_to_world: np.ndarray,
    *,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    v_flip: bool,
    min_depth: float = 1e-4,
) -> np.ndarray:
    """Unproject a depth crop to world ``(N, 3)`` points."""
    depth = np.asarray(depth_hw, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    height, width = depth.shape
    k = np.asarray(K, dtype=np.float64).reshape(3, 3)
    t_cw = np.asarray(t_cam_to_world, dtype=np.float64).reshape(4, 4)
    k_inv = np.linalg.inv(k)

    v_full, u_full = np.indices((height, width))
    if v_flip:
        v_full = (height - 1) - v_full
    u = u_full[ymin:ymax, xmin:xmax].reshape(-1)
    v = v_full[ymin:ymax, xmin:xmax].reshape(-1)
    z = depth[ymin:ymax, xmin:xmax].reshape(-1)
    valid = np.isfinite(z) & (z > float(min_depth))
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    pixels = np.stack([u, v, np.ones_like(u)], axis=0)
    points_cam = k_inv @ pixels * z
    ones = np.ones((1, points_cam.shape[1]), dtype=np.float64)
    points_h = np.vstack([points_cam, ones])
    world = t_cw @ points_h
    return world[:3, :].T


def _largest_dbscan_cluster(
    pts: np.ndarray,
    *,
    eps: float,
    min_points: int,
) -> np.ndarray:
    """Keep the largest DBSCAN cluster (upstream ``filtering_points`` / Open3D).

    If every point is noise (``labels.max() < 0``), return *pts* unchanged.
    """
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if arr.shape[0] == 0:
        return arr
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return arr
    labels = DBSCAN(eps=float(eps), min_samples=int(min_points)).fit_predict(arr)
    if labels.size == 0 or int(np.max(labels)) < 0:
        return arr
    valid = labels[labels >= 0]
    largest = int(np.bincount(valid).argmax())
    return arr[labels == largest]


def filter_points(
    pts: np.ndarray,
    bounds: XyzBounds,
    *,
    keep_nearest_frac: float,
    cluster: str = "none",
    dbscan_eps: float | None = None,
    dbscan_min_points: int | None = None,
) -> np.ndarray:
    """Workspace AABB + nearest-fraction + optional DBSCAN (official ``filtering_points``)."""
    arr = np.asarray(pts, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    arr = arr.reshape(-1, 3)
    keep = (
        (arr[:, 0] > bounds.x_min)
        & (arr[:, 0] < bounds.x_max)
        & (arr[:, 1] > bounds.y_min)
        & (arr[:, 1] < bounds.y_max)
        & (arr[:, 2] > bounds.z_min)
        & (arr[:, 2] < bounds.z_max)
    )
    arr = arr[keep]
    if arr.shape[0] == 0:
        return arr
    frac = float(keep_nearest_frac)
    if 0.0 < frac < 1.0 and arr.shape[0] >= 2:
        center = arr.mean(axis=0)
        dist = np.linalg.norm(arr - center, axis=1)
        # Upstream: ``keep_count = int(len(pts) * 0.8)`` (truncates toward 0).
        keep_count = int(arr.shape[0] * frac)
        if keep_count >= 1:
            idx = np.argsort(dist)[:keep_count]
            arr = arr[idx]
    if arr.shape[0] == 0:
        return arr
    mode = str(cluster).strip().lower()
    if mode == "dbscan":
        if dbscan_eps is None or dbscan_min_points is None:
            raise ValueError("dbscan clustering requires dbscan_eps and dbscan_min_points")
        arr = _largest_dbscan_cluster(arr, eps=float(dbscan_eps), min_points=int(dbscan_min_points))
    elif mode not in {"none", ""}:
        raise ValueError(f"unknown cluster mode {cluster!r}")
    return arr


def fit_mvee_ellipsoid(pts: np.ndarray, ellipsoid_cls: Any) -> Any:
    """Minimum-volume enclosing ellipsoid from points (hull vertices)."""
    import cvxpy as cp
    from scipy.spatial import ConvexHull

    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if arr.shape[0] < 4:
        raise ValueError(f"need >= 4 points to fit an ellipsoid, got {arr.shape[0]}")
    hull = ConvexHull(arr)
    hull_pts = arr[hull.vertices]
    n, d = hull_pts.shape
    m_var = cp.Variable((d, d), PSD=True)
    g_var = cp.Variable(d)
    objective = cp.Minimize(-cp.log_det(m_var))
    constraints = [cp.norm(m_var @ hull_pts[i] - g_var) <= 1 for i in range(n)]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except cp.SolverError:
        prob.solve(verbose=False)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or m_var.value is None:
        raise RuntimeError(f"MVEE failed with status={prob.status!r}")
    center = np.linalg.solve(m_var.value, g_var.value)
    a_mat = m_var.value.T @ m_var.value
    eigvals, eigvecs = np.linalg.eigh(a_mat)
    eigvals = np.clip(eigvals, 1e-15, None)
    axes = 1.0 / np.sqrt(eigvals)
    order = np.argsort(axes)[::-1]
    axes = axes[order]
    rotation = eigvecs[:, order]
    return ellipsoid_cls(position=center, axes_diag=axes, rotation=rotation)


def gl_depth_to_meters(depth_map: np.ndarray, sim: Any) -> np.ndarray:
    """Convert OpenGL [0,1] depth to meters (official ``get_real_depth_map``)."""
    depth = np.asarray(depth_map, dtype=np.float64)
    try:
        from robosuite.utils.camera_utils import get_real_depth_map

        return np.asarray(get_real_depth_map(sim, depth), dtype=np.float64)
    except Exception:
        pass
    try:
        extent = float(sim.model.stat.extent)
        near = float(sim.model.vis.map.znear) * extent
        far = float(sim.model.vis.map.zfar) * extent
    except Exception:
        return depth
    denom = 1.0 - depth * (1.0 - near / far)
    denom = np.clip(denom, 1e-12, None)
    return near / denom


def camera_intrinsics(sim: Any, camera_name: str, height: int, width: int) -> np.ndarray:
    try:
        from robosuite.utils.camera_utils import get_camera_intrinsic_matrix

        return np.asarray(
            get_camera_intrinsic_matrix(sim, camera_name, height, width),
            dtype=np.float64,
        )
    except Exception:
        cam_id = int(sim.model.camera_name2id(camera_name))
        fovy = float(np.asarray(sim.model.cam_fovy[cam_id]).reshape(-1)[0])
        f = 0.5 * float(height) / np.tan(np.deg2rad(fovy) / 2.0)
        return np.array(
            [[f, 0.0, width * 0.5], [0.0, f, height * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def camera_extrinsics(sim: Any, camera_name: str) -> np.ndarray:
    try:
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix

        return np.asarray(get_camera_extrinsic_matrix(sim, camera_name), dtype=np.float64)
    except Exception:
        cam_id = int(sim.model.camera_name2id(camera_name))
        pos = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64).reshape(3)
        xmat = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = xmat
        t[:3, 3] = pos
        return t


def unwrap_mujoco_sim(env: Any) -> Any | None:
    cur = env
    for _ in range(6):
        if cur is None:
            return None
        sim = getattr(cur, "sim", None)
        if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
            return sim
        cur = getattr(cur, "env", None)
    return None


def _render_rgb_depth(
    sim: Any, camera_name: str, height: int, width: int
) -> tuple[np.ndarray, np.ndarray] | None:
    render = getattr(sim, "render", None)
    if not callable(render):
        return None
    try:
        out = render(camera_name=camera_name, height=int(height), width=int(width), depth=True)
    except TypeError:
        try:
            out = render(width=int(width), height=int(height), camera_name=camera_name, depth=True)
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(out, tuple) or len(out) != 2:
        return None
    rgb, depth = out
    rgb_arr = np.asarray(rgb)
    depth_arr = np.asarray(depth)
    if rgb_arr.ndim == 3 and rgb_arr.shape[2] > 3:
        rgb_arr = rgb_arr[:, :, :3]
    return rgb_arr, depth_arr


def load_camera_frame(
    *,
    env: Any,
    obs: Any,
    camera_name: str,
) -> CameraFrame | None:
    """Prefer ``obs`` RGB-D; otherwise on-demand ``sim.render(..., depth=True)``."""
    rgb = None
    depth = None
    if isinstance(obs, dict):
        rgb = obs.get(f"{camera_name}_image")
        depth = obs.get(f"{camera_name}_depth")
    sim = unwrap_mujoco_sim(env)
    if rgb is None and sim is not None:
        rendered = _render_rgb_depth(sim, camera_name, 256, 256)
        if rendered is None:
            return None
        rgb, depth = rendered
    if rgb is None:
        return None
    rgb_arr = np.asarray(rgb)
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] < 3:
        return None
    height, width = int(rgb_arr.shape[0]), int(rgb_arr.shape[1])
    if depth is None:
        if sim is None:
            return None
        rendered = _render_rgb_depth(sim, camera_name, height, width)
        if rendered is None:
            return None
        depth = rendered[1]
    depth_arr = np.asarray(depth, dtype=np.float64)
    if depth_arr.ndim == 3:
        depth_arr = depth_arr.squeeze()
    if depth_arr.ndim != 2:
        return None
    if sim is not None:
        # Official get_point_cloud always runs get_real_depth_map on the depth map.
        depth_arr = gl_depth_to_meters(depth_arr, sim)
        k = camera_intrinsics(sim, camera_name, height, width)
        t_cw = camera_extrinsics(sim, camera_name)
    else:
        return None
    rgb_u8 = rgb_arr[:, :, :3]
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = np.clip(rgb_u8, 0, 255).astype(np.uint8)
    return CameraFrame(
        rgb=rgb_u8,
        depth=depth_arr,
        K=k,
        T_cam_to_world=t_cw,
        camera_name=str(camera_name),
    )
