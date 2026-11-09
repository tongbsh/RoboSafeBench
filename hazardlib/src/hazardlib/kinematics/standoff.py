"""Geometric standoff from sampled robot geometry (decision D15 a). Pure functions, no simulator.

The scalar standoff ``r_hazard(n) + eef_contact_radius_m`` (D1) models the robot as a ball around the EEF. With
robot geometry (``HazardHost.robot_collision_points``: surface sample points of every robot collision body in the
body's own frame) the standoff is solved instead:

* **Hazard shape.** ``agent`` is the hazard body with the identity orientation, the same shapes the support radius
  of :func:`hazardlib.instantiate.support_radius_m` accepts: a sphere (equal half extents) or a box (3 half
  extents). Any other shape is refused (``ValueError``).
* **First contact along the normal.** The hazard centre is at ``p_ref + s n`` (``n`` a unit vector). Coming from
  far away (``s`` decreasing from ``+inf``), the first ``s`` at which the hazard touches any sample point is the
  standoff ``s*``; the touched point names the first contact body. Per point ``q`` with ``a = q - p_ref``:

  - sphere of radius ``r``: the point lies on the hazard's path iff its distance to the axis
    ``d = |a - (a.n) n| <= r``; then ``s(q) = a.n + sqrt(r^2 - d^2)``;
  - box with half extents ``h`` (axis aligned): ``|a_i - s n_i| <= h_i`` for every axis gives an interval of ``s``
    per axis (an axis with ``n_i = 0`` either always or never holds); their intersection is the interval in which
    ``q`` is inside the box, and ``s(q)`` is its upper end.

  ``s* = max_q s(q)``; no point on the path means "no contact" (``None``). ``s*`` may be smaller than the hazard's
  support radius, or negative, when the reference point itself is not on the robot surface.
* **Penetration depth** (gate G0): how far a point is inside the hazard placed at ``c``: sphere ``r - |q - c|``,
  box ``min_i (h_i - |q_i - c_i|)``; a positive value means inside. It is the same test as the distance check
  ``|q - c| < r_hazard + r_eef`` of the scalar model, with the robot reduced to its sample points.
* **Body poses at an instant.** A body's points go to the world as ``R(q_body) p + x_body``. Between samples of a
  recorded body path the position is interpolated linearly and the orientation by normalised linear
  interpolation after flipping to the shorter arc (``SampledPath.quat``), the same interpolation that
  ``probe.context._median_quats`` uses for the EEF orientation.

Sampling error: if every point of the true surface lies within ``delta`` of a sample, ``s*`` is never above the
true value and, for a sphere hazard of radius ``r`` against a spherical body of radius ``r_e`` whose contact point
is at angle ``theta`` from the nearest sample (``delta = 2 r_e sin(theta / 2)``), it is below by
``r + r_e - r_e cos(theta) - sqrt(r^2 - r_e^2 sin(theta)^2) ~ (delta^2 / 2) (1 / r_e + 1 / r)`` (see
:func:`sphere_sampling_error_bound`).
"""
import dataclasses
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .._numeric import finite_array, finite_float, positive_float, vec3
from .path import SampledPath

GEOMETRY_SHAPES = ("sphere", "box")


# ------------------------------------------------------------------------------------------------ inputs

def hazard_shape(agent: Any) -> Tuple[str, np.ndarray]:
    """``("sphere", [r])`` or ``("box", [hx, hy, hz])`` of an agent with ``shape`` and ``half_extents_m``."""
    extents = [positive_float("agent.half_extents_m[%d]" % i, h) for i, h in enumerate(agent.half_extents_m)]
    if agent.shape == "sphere":
        if not extents or len(set(extents)) != 1:
            raise ValueError("a sphere agent needs equal half extents, got %r" % (extents,))
        return "sphere", np.array(extents[:1])
    if agent.shape == "box":
        if len(extents) != 3:
            raise ValueError("a box agent needs 3 half extents, got %r" % (extents,))
        return "box", np.array(extents)
    raise ValueError("the geometric standoff supports the agent shapes %s, not %r"
                     % (list(GEOMETRY_SHAPES), agent.shape))


def unit_normal(name: str, value: Any) -> np.ndarray:
    """``value`` as a float vector of unit length; anything else is an error (no silent normalisation). Unit
    means equal to 1 within the square root of the float64 machine epsilon."""
    arr = vec3(name, value)
    norm = float(np.linalg.norm(arr))
    if not abs(norm - 1.0) <= math.sqrt(np.finfo(np.float64).eps):
        raise ValueError("%s must be a unit vector, got %s (norm %r)" % (name, arr.tolist(), norm))
    return arr


def points_array(name: str, value: Any) -> np.ndarray:
    arr = finite_array(name, value)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("%s: expected an (M, 3) array, got shape %s" % (name, arr.shape))
    return arr


# ------------------------------------------------------------------------------- first contact along n

def point_standoffs(agent: Any, p_ref: Any, normal: Any, points: Any) -> np.ndarray:
    """``s(q)`` of every point (see the module docstring); ``-inf`` for a point the hazard never touches."""
    kind, h = hazard_shape(agent)
    n = unit_normal("normal", normal)
    pts = points_array("points", points)
    a = pts - vec3("p_ref", p_ref)[None, :]
    out = np.full(pts.shape[0], -np.inf)
    if kind == "sphere":
        r = float(h[0])
        along = a @ n
        perp2 = np.maximum(np.einsum("ij,ij->i", a, a) - along * along, 0.0)
        hit = perp2 <= r * r
        out[hit] = along[hit] + np.sqrt(r * r - perp2[hit])
        return out
    lo = np.full(pts.shape[0], -np.inf)
    hi = np.full(pts.shape[0], np.inf)
    ok = np.ones(pts.shape[0], dtype=bool)
    for i in range(3):
        if n[i] == 0.0:
            ok &= np.abs(a[:, i]) <= h[i]
            continue
        e1 = (a[:, i] - h[i]) / n[i]
        e2 = (a[:, i] + h[i]) / n[i]
        lo = np.maximum(lo, np.minimum(e1, e2))
        hi = np.minimum(hi, np.maximum(e1, e2))
    ok &= lo <= hi
    out[ok] = hi[ok]
    return out


@dataclasses.dataclass(frozen=True)
class FirstContact:
    """``standoff_m`` = ``s*``; ``body`` and ``point`` (world) of the first touched sample; ``index`` into the
    point array."""
    standoff_m: float
    body: str
    point: np.ndarray
    index: int


def first_contact_along_normal(agent: Any, p_ref: Any, normal: Any, points: Any,
                               bodies: Sequence[str]) -> Optional[FirstContact]:
    """The standoff ``s*`` and first contact body (module docstring); ``None`` when no point is on the path.
    Ties go to the first point in array order."""
    pts = points_array("points", points)
    if len(bodies) != pts.shape[0]:
        raise ValueError("bodies: %d names for %d points" % (len(bodies), pts.shape[0]))
    values = point_standoffs(agent, p_ref, normal, pts)
    if pts.shape[0] == 0 or not np.isfinite(values).any():
        return None
    index = int(np.argmax(values))
    return FirstContact(float(values[index]), str(bodies[index]), pts[index].copy(), index)


def penetration_depths(agent: Any, center: Any, points: Any) -> np.ndarray:
    """Depth of every point inside the hazard centred at ``center`` (positive inside; module docstring)."""
    kind, h = hazard_shape(agent)
    d = points_array("points", points) - vec3("center", center)[None, :]
    if kind == "sphere":
        return float(h[0]) - np.linalg.norm(d, axis=1)
    return np.min(h[None, :] - np.abs(d), axis=1)


def sphere_sampling_error_bound(r_hazard_m: float, r_body_m: float, delta_m: float) -> float:
    """Largest underestimate of ``s*`` for a sphere hazard against a sampled sphere body whose every surface point
    lies within ``delta_m`` (chord) of a sample: ``r + r_e - r_e cos(theta) - sqrt(r^2 - r_e^2 sin^2 theta)`` with
    ``2 r_e sin(theta / 2) = delta``."""
    r = positive_float("r_hazard_m", r_hazard_m)
    r_e = positive_float("r_body_m", r_body_m)
    delta = positive_float("delta_m", delta_m)
    if delta >= 2.0 * r_e:
        raise ValueError("delta_m=%r must be below the body diameter %r" % (delta, 2.0 * r_e))
    theta = 2.0 * math.asin(delta / (2.0 * r_e))
    side = r_e * math.sin(theta)
    if side > r:
        raise ValueError("the sample gap is wider than the hazard (r_e sin(theta)=%r > r=%r)" % (side, r))
    return r + r_e - r_e * math.cos(theta) - math.sqrt(r * r - side * side)


# ------------------------------------------------------------------------------------- body frames

def quat_matrix(quat: Any) -> np.ndarray:
    """Rotation matrix of a ``wxyz`` quaternion (normalised first; zero is an error)."""
    q = finite_array("quat", quat)
    if q.shape != (4,):
        raise ValueError("quat: expected shape (4,), got %s" % (q.shape,))
    norm = float(np.linalg.norm(q))
    if not norm > 0:
        raise ValueError("quat must be non-zero")
    w, x, y, z = q / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class RobotGeometry:
    """Body-frame surface samples of the robot's own collision bodies and their declared maximum spacing
    (BD ``robot_point_spacing_m``). ``points``: body name -> (M, 3), M >= 1."""

    def __init__(self, points: Mapping[str, Any], spacing_m: float) -> None:
        self.spacing_m = positive_float("spacing_m", spacing_m)
        self.points = {}  # type: Dict[str, np.ndarray]
        for name in sorted(points):
            if not isinstance(name, str) or not name:
                raise ValueError("robot geometry body names must be non-empty strings, got %r" % (name,))
            arr = points_array("points[%r]" % name, points[name])
            if arr.shape[0] == 0:
                raise ValueError("points[%r] is empty" % name)
            self.points[name] = arr

    def to_dict(self) -> Dict[str, Any]:
        return {"spacing_m": self.spacing_m, "points": {k: v.tolist() for k, v in sorted(self.points.items())}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RobotGeometry":
        if not isinstance(data, Mapping) or set(data) != {"spacing_m", "points"}:
            raise ValueError("robot geometry must be a mapping with exactly 'spacing_m' and 'points', got %r"
                             % (sorted(data) if isinstance(data, Mapping) else data,))
        return cls(data["points"], data["spacing_m"])

    def restricted(self, names: Sequence[str]) -> "RobotGeometry":
        missing = sorted(set(names) - set(self.points))
        if missing:
            raise ValueError("the robot geometry has no points for the body(ies) %s (has: %s)"
                             % (missing, sorted(self.points)))
        return RobotGeometry({n: self.points[n] for n in names}, self.spacing_m)

    def world_points(self, body_pos: Mapping[str, Any], body_quat: Mapping[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """All sample points in the world, ``(K, 3)``, and the body name of each, for the bodies of ``body_pos``
        (sorted by name); every one needs points and an orientation."""
        chunks, names = [], []  # type: List[np.ndarray], List[str]
        for name in sorted(body_pos):
            if name not in self.points:
                raise ValueError("no robot geometry for body %r (has: %s)" % (name, sorted(self.points)))
            if name not in body_quat:
                raise ValueError("no orientation for body %r" % (name,))
            local = self.points[name]
            chunks.append(local @ quat_matrix(body_quat[name]).T + vec3("body_pos[%r]" % name, body_pos[name])[None, :])
            names.extend([name] * local.shape[0])
        if not chunks:
            return np.zeros((0, 3)), []
        return np.concatenate(chunks), names


class BodyPosePaths:
    """Recorded poses of named bodies at common sample times; :meth:`at` interpolates them (module docstring)."""

    def __init__(self, times_s: Any, pos: Mapping[str, Any], quat: Mapping[str, Any]) -> None:
        if set(pos) != set(quat):
            raise ValueError("body position and orientation paths name different bodies: %s vs %s"
                             % (sorted(pos), sorted(quat)))
        self.times_s = finite_array("times_s", times_s)
        self._paths = {name: SampledPath(self.times_s, pos[name], quat[name]) for name in sorted(pos)}

    @property
    def names(self) -> List[str]:
        return list(self._paths)

    def at(self, t: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        t = finite_float("t", t)
        pos = {name: path.pos(t) for name, path in self._paths.items()}
        quat = {name: path.quat(t) for name, path in self._paths.items()}
        return pos, quat


def robot_points_at(geometry: RobotGeometry, paths: BodyPosePaths, t: float) -> Tuple[np.ndarray, List[str]]:
    """World sample points of the bodies of ``paths`` at time ``t``."""
    pos, quat = paths.at(t)
    return geometry.world_points(pos, quat)
