"""Rigid body groups and robot collision geometry of a LIBERO (MuJoCo) scene (task H3, decision D15 a).

**Rigid groups.** A MuJoCo body without joints is welded to its parent. Every body is assigned to the group of the
nearest ancestor-or-self that has a joint; a body whose chain to the world has no joint at all is static and forms
its own group (welding static bodies together would merge, e.g., the robot pedestal with link 0). A group is named
after its root body, or, when the root has no collision geom, after the first member (in body order) that has one:
the Panda hand is ``gripper0_right_gripper`` (root ``robot0_link7``, which has visual geoms only), a finger and its
pad are ``gripper0_leftfinger``. A group's frame is the frame of the body it is named after; contacts, positions,
velocities, orientations and collision points of the group all use that name and frame (01 §3.3 items 5, 9).

**Collision points** (``robot_collision_points``). For every collision geom of a robot group, surface samples in the
group frame, at most ``spacing`` apart: MuJoCo collides a mesh through its convex hull, so a mesh contributes its
hull (scipy ``ConvexHull`` of the compiled vertices; triangles subdivided until every edge is <= spacing, then the
barycentric grid of each triangle); a box its six faces on a grid; a cylinder its two caps and its side; a sphere a
latitude / longitude grid. Grid steps are chosen so that neighbouring samples are at most ``spacing`` apart along
both directions of a face, which bounds the distance from any surface point to its nearest sample by
``spacing / sqrt(2)``.
"""
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_ELLIPSOID, GEOM_CYLINDER, GEOM_BOX, GEOM_MESH = 0, 2, 3, 4, 5, 6, 7


def quat_to_mat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """``wxyz`` of a rotation matrix (w >= 0)."""
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def is_collision_geom(model: object, g: int) -> bool:
    return bool(model.geom_contype[g]) or bool(model.geom_conaffinity[g])


class RigidGroups(object):
    """Body -> group of the scene (see the module docstring). ``group_of[b]`` is the group name of body ``b``;
    ``frame_body[name]`` the body whose frame the group uses; ``members[name]`` its bodies; ``has_joint_root``
    whether the group moves (is not static)."""

    def __init__(self, model: object, body_name: Sequence[str]) -> None:
        nbody = int(model.nbody)
        jointed = set(int(model.jnt_bodyid[j]) for j in range(int(model.njnt)))
        collides = set(int(model.geom_bodyid[g]) for g in range(int(model.ngeom)) if is_collision_geom(model, g))
        root = {}  # type: Dict[int, int]
        for b in range(nbody):
            r = b
            while r != 0 and r not in jointed and int(model.body_parentid[r]) != 0:
                parent = int(model.body_parentid[r])
                if not self._moves(model, parent, jointed):
                    break
                r = parent
            root[b] = r
        groups = {}  # type: Dict[int, List[int]]
        for b in range(1, nbody):
            groups.setdefault(root[b], []).append(b)
        self.group_of = {}  # type: Dict[int, str]
        self.frame_body = {}  # type: Dict[str, int]
        self.members = {}  # type: Dict[str, List[int]]
        for r, members in sorted(groups.items()):
            named = r if r in collides else next((b for b in sorted(members) if b in collides), r)
            name = body_name[named]
            self.frame_body[name] = named
            self.members[name] = sorted(members)
            for b in members:
                self.group_of[b] = name
        self.group_of[0] = body_name[0]

    @staticmethod
    def _moves(model: object, b: int, jointed: set) -> bool:
        while b != 0:
            if b in jointed:
                return True
            b = int(model.body_parentid[b])
        return False


def _body_to_frame(model: object, body: int, frame_body: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation and translation of ``body``'s frame in ``frame_body``'s frame (welded chain, model constants)."""
    def to_root(b: int) -> Tuple[np.ndarray, np.ndarray]:
        rot, pos = np.eye(3), np.zeros(3)
        chain = []
        while b != 0:
            chain.append(b)
            b = int(model.body_parentid[b])
        for c in reversed(chain):
            r = quat_to_mat(model.body_quat[c])
            pos = pos + rot.dot(np.asarray(model.body_pos[c], dtype=np.float64))
            rot = rot.dot(r)
        return rot, pos
    rb, pb = to_root(body)
    rf, pf = to_root(frame_body)
    return rf.T.dot(rb), rf.T.dot(pb - pf)


def _grid(n_u: int, n_v: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n_u + 1)
    v = np.linspace(0.0, 1.0, n_v + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


def _steps(length: float, spacing: float) -> int:
    return max(1, int(math.ceil(length / spacing - 1e-9)))


def box_points(half: Sequence[float], spacing: float) -> np.ndarray:
    h = np.asarray(half, dtype=np.float64)
    out = []
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        uv = _grid(_steps(2 * h[a], spacing), _steps(2 * h[b], spacing))
        for sign in (-1.0, 1.0):
            p = np.zeros((len(uv), 3))
            p[:, axis] = sign * h[axis]
            p[:, a] = -h[a] + 2 * h[a] * uv[:, 0]
            p[:, b] = -h[b] + 2 * h[b] * uv[:, 1]
            out.append(p)
    return np.unique(np.round(np.concatenate(out), 12), axis=0)


def cylinder_points(radius: float, half_length: float, spacing: float) -> np.ndarray:
    n_ang = _steps(2 * math.pi * radius, spacing)
    ang = np.linspace(0.0, 2 * math.pi, n_ang, endpoint=False)
    out = []
    for z in np.linspace(-half_length, half_length, _steps(2 * half_length, spacing) + 1):
        out.append(np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full_like(ang, z)], axis=1))
    for z in (-half_length, half_length):
        for r in np.linspace(0.0, radius, _steps(radius, spacing) + 1)[:-1]:
            n = max(1, _steps(2 * math.pi * r, spacing))
            a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
            out.append(np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)], axis=1))
    return np.concatenate(out)


def sphere_points(radius: float, spacing: float) -> np.ndarray:
    n_lat = _steps(math.pi * radius, spacing)
    out = [np.array([[0.0, 0.0, radius], [0.0, 0.0, -radius]])]
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat
        r = radius * math.sin(theta)
        n = max(1, _steps(2 * math.pi * r, spacing))
        a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        out.append(np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, radius * math.cos(theta))], axis=1))
    return np.concatenate(out)


def hull_points(vertices: np.ndarray, spacing: float) -> np.ndarray:
    """Samples of the convex hull surface of ``vertices`` (what MuJoCo collides for a mesh geom)."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    out = [vertices[hull.vertices]]
    for simplex in hull.simplices:
        a, b, c = vertices[simplex]
        longest = max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c))
        n = _steps(longest, spacing)
        if n == 1:
            continue
        ij = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
        w = np.array(ij, dtype=np.float64) / n
        out.append(a + np.outer(w[:, 0], b - a) + np.outer(w[:, 1], c - a))
    return np.unique(np.round(np.concatenate(out), 12), axis=0)


def geom_points(model: object, g: int, spacing: float) -> np.ndarray:
    """Surface samples of collision geom ``g`` in its own frame."""
    kind = int(model.geom_type[g])
    size = np.asarray(model.geom_size[g], dtype=np.float64)
    if kind == GEOM_BOX:
        return box_points(size, spacing)
    if kind == GEOM_CYLINDER:
        return cylinder_points(size[0], size[1], spacing)
    if kind == GEOM_SPHERE:
        return sphere_points(size[0], spacing)
    if kind == GEOM_MESH:
        mesh = int(model.geom_dataid[g])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        return hull_points(np.asarray(model.mesh_vert[start:start + count], dtype=np.float64), spacing)
    raise ValueError("geom %d has type %d: no surface sampler (box, cylinder, sphere, mesh only)" % (g, kind))


def group_points(model: object, groups: RigidGroups, name: str, spacing: float) -> np.ndarray:
    """Collision surface samples of group ``name`` in the frame of its frame body."""
    frame = groups.frame_body[name]
    out = []
    for b in groups.members[name]:
        rot_b, pos_b = _body_to_frame(model, b, frame)
        for g in range(int(model.ngeom)):
            if int(model.geom_bodyid[g]) != b or not is_collision_geom(model, g):
                continue
            pts = geom_points(model, g, spacing / 2)
            rot_g = quat_to_mat(model.geom_quat[g])
            in_body = pts.dot(rot_g.T) + np.asarray(model.geom_pos[g], dtype=np.float64)
            out.append(in_body.dot(rot_b.T) + pos_b)
    if not out:
        raise ValueError("group %r has no collision geom" % name)
    points = np.concatenate(out)
    # Hull triangulations can oversample the same surface by orders of magnitude.
    # A half-spacing surface grid covers each face within spacing/(2*sqrt(2));
    # retaining one actual surface point per voxel adds at most spacing/2.
    # Thus every surface location remains within 0.854*spacing of a sample.
    cell = spacing / (2 * math.sqrt(3))
    _, keep = np.unique(np.floor(points / cell).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(keep)]
