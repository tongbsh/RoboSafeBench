"""HazardSpec + TaskContext -> HazardInstance (paradigm §5.2; decisions D1, D2, D4, D6, D11 b).

``instantiate(spec, task_context, seed, bd, settings)`` is a pure function of ``(spec, TC, task_id, layout_id,
seed)`` plus the BD and the settings. Its only randomness (a list-valued intensity axis, a direction the spec
leaves open) comes from a private stream: sha256 of the spec hash, the TC hash, the task, the layout, the seed,
a label and a counter. No global random state is read or changed, and the stream does not depend on the numpy
version.

Steps:

1. **Hit time.** ``exposure.anchor`` must be ``nominal_ee_path``; ``t* = start + frac * duration`` of
   ``exposure.anchor_phase``. The reference point is ``p_ref = nominal_ee_path(t_hit) + exposure.offset_m``
   (recorded as ``provenance.anchor_point``, the ``p*`` of §5.2 used by the exposure check).
2. **Standoff (D1, D15 a).** The contact normal ``n`` points from the EEF towards the hazard (fall: ``+z``; reach:
   ``-direction``; slide: see 4). The centroid target handed to ``build_hit_path`` is ``p_ref + standoff * n``.
   With robot geometry in the TaskContext (``provenance.robot_geometry`` with the robot body position and
   orientation paths) the standoff is the first contact along ``n`` of the agent (identity orientation) with the
   robot's sample points placed at ``t_hit`` (:mod:`hazardlib.kinematics.standoff`: body poses interpolated
   linearly / by normalised lerp); a hazard that misses every point is rejected
   ``no_robot_contact_along_normal``. Without geometry ``standoff = r_hazard(n) + eef_contact_radius_m``, where
   ``r_hazard`` is the support radius of the agent along ``n`` (sphere: its radius; box with the fixed identity
   orientation: ``sum |n_i| h_i``; other shapes are refused). ``standoff_m``, ``standoff_source``
   (``robot_geometry`` / ``eef_contact_radius``), ``standoff_body`` (first contact body, ``None`` without
   geometry) and ``robot_point_spacing_m`` go into the provenance.
3. **Primitive parameters** from the intensity vector: see :mod:`hazardlib.instantiate.mapping`. The release
   window is ``runtime.yaml: release_velocity_window_s`` (D4), never an engine hook step.
4. **Slide support.** The hazard slides horizontally on a support surface (the host's
   ``support_surfaces`` recorded in the TaskContext, else the TC's estimate; the source is recorded): centroid
   height = surface top + the agent's z half-size. The hit time is the time closest to ``t*`` inside the anchor
   phase (``t*`` or a nominal sample) at which ``p_ref`` lies over a surface with
   ``0 <= p_ref_z - top <= 2 h_z + eef_contact_radius_m``; none -> rejected ``no_support_surface``. The normal
   is then tilted out of the horizontal plane so that ``p_ref + standoff(n) n`` has the required height:
   ``standoff(phi) sin(phi) = dz`` solved by bisection over ``phi`` in ``[-pi/2, pi/2]`` after checking that the
   end points bracket ``dz`` (else ``no_support_surface``). The scalar standoff is monotone in ``phi``; the
   geometric one is not, and can jump where a sample point enters or leaves the hazard's path, so with geometry a
   tilt at which the hazard misses the robot is rejected ``no_robot_contact_along_normal`` and a bracket that
   closes on such a jump (the first contact point at one end is off the path at the other end) is rejected
   ``slide_normal_unsolved``; no approximate root is taken.
5. **Keyframes** sample the analytic path every ``keyframe_dt_s`` (``hook_times``), plus ``t_hit`` and the
   release time (``insert_instants``: a grid point within round-off of either gives way); the largest linear-interpolation error at interior points must be <= ``keyframe_interp_tol_m``
   (else rejected ``keyframe_interp_error``). Orientation: the identity for every keyframe (``AgentSpec``
   has no orientation field yet; recorded as ``hazard_quat_source``).
6. **Follow-through (D19, optional).** ``motion.params.follow_through`` is ``"pass"`` or ``"retract"`` (slide and
   reach only; absent = no exit path, the instance is then exactly the pre-D19 one). The instance gets
   ``exit_keyframes`` and its ``keyframes`` are cut at ``release_time_s`` (the part of a reach path beyond the
   hit moves into the exit path), so the playback path is ``keyframes + exit_keyframes`` with one shared junction
   sample. Nothing else changes: onset, nominal hit, release time and velocity, standoff and the random stream
   (the key leaves ``follow_through`` out of the spec hash) are those of the same spec without it; only
   ``provenance.spec_hash`` and ``provenance.follow_through`` (and the keyframe ``count``) differ.
   ``"pass"``: from the release pose on, a straight line at the (constant) release velocity with a constant
   orientation, until the centre is strictly outside the TaskContext workspace box widened by the hazard's
   bounding-sphere radius (no ``g0`` margin); the end is the first sample of the keyframe grid (every
   ``keyframe_dt_s`` from the release) that is outside. ``"retract"``: the analytic primitive path is followed on
   to its own end if it continues past the hit (a reach with ``overshoot_m > 0``), then retraced time-reversed back
   to its start pose (a path without continuation is retraced from the hit pose). The exit is sampled every
   ``keyframe_dt_s`` (plus the turning instant) and held to ``keyframe_interp_tol_m`` like the approach; both
   modes record their duration in ``provenance.follow_through``.
7. **Rejections** raise :class:`InstanceRejected` with a reason in :data:`REJECT_REASONS`; a malformed spec or
   context raises ``ValueError`` instead. Geometric validity (workspace, penetration, first contact, R1) is
   gate G0's job.
"""
import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .._numeric import finite_float, unit_vec3
from .._records import is_unknown
from ..hashing import sha256_hex
from ..kinematics.exit import RetractPath, bounding_radius_m, straight_exit_steps
from ..kinematics.hit import FAMILY_FALL, FAMILY_REACH, FAMILY_SLIDE, build_hit_path
from ..kinematics.path import SampledPath
from ..kinematics.standoff import (BodyPosePaths, RobotGeometry, first_contact_along_normal, point_standoffs,
                                   robot_points_at)
from ..kinematics.subdivide import hook_times, insert_instants
from ..spec.benchmark_descriptor import BenchmarkDescriptor
from ..spec.hazard_spec import AgentSpec, HazardSpec
from ..spec.instance import HazardInstance
from ..spec.task_context import TaskContext
from .mapping import ZeroOvershootError, fall_release_height_m, reach_geometry, slide_start_distance_m
from .settings import FAMILY_AXES, InstantiateSettings

FOLLOW_THROUGH_MODES = ("pass", "retract")
REJECT_REASONS = ("no_support_surface", "reach_zero_overshoot", "hit_path_invalid", "keyframe_interp_error",
                  "no_robot_contact_along_normal", "slide_normal_unsolved", "exit_path_invalid")
PRIMITIVES = {FAMILY_FALL: ("free_fall", (), ()),
              FAMILY_SLIDE: ("slide", (), ("direction", "follow_through")),
              FAMILY_REACH: ("reach", ("speed_profile",), ("direction", "follow_through"))}
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
_UP = (0.0, 0.0, 1.0)


class InstanceRejected(Exception):
    """The spec cannot be realised in this TaskContext; ``reason`` is one of :data:`REJECT_REASONS`."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__("%s: %s" % (reason, detail))
        self.reason = reason
        self.detail = detail


class _Stream:
    """Deterministic uniforms in [0, 1): sha256(key | label | counter), 52 bits."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.counter = 0

    def uniform(self, label: str) -> float:
        text = "%s|%s|%d" % (self.key, label, self.counter)
        self.counter += 1
        return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:13], 16) / float(16 ** 13)

    def choice(self, label: str, options: Sequence[Any]) -> Any:
        return options[min(int(self.uniform(label) * len(options)), len(options) - 1)]


def support_radius_m(agent: AgentSpec, normal: Any) -> float:
    """Distance from the agent's centre to its supporting plane with outward normal ``normal``."""
    n = unit_vec3("normal", normal)
    h = list(agent.half_extents_m)
    if agent.shape == "sphere":
        if len(set(h)) != 1:
            raise ValueError("a sphere agent needs equal half extents, got %r" % (h,))
        return float(h[0])
    if agent.shape == "box":
        if len(h) != 3:
            raise ValueError("a box agent needs 3 half extents, got %r" % (h,))
        return float(sum(abs(n[i]) * h[i] for i in range(3)))
    raise ValueError("the standoff needs the support radius of the agent; shape %r is not supported "
                     "(sphere, box)" % (agent.shape,))


def half_height_m(agent: AgentSpec) -> float:
    return support_radius_m(agent, _UP)


def _known(bd: BenchmarkDescriptor, name: str) -> float:
    value = getattr(bd, name)
    if is_unknown(value):
        raise ValueError("BD %r leaves %s unknown; the builder needs it" % (bd.name, name))
    return float(value)


def _intensity(spec: HazardSpec, settings: InstantiateSettings, stream: _Stream) -> Dict[str, float]:
    required, optional = FAMILY_AXES[spec.family]
    unknown = sorted(set(spec.intensity) - set(required + optional))
    missing = [a for a in required if a not in spec.intensity]
    if unknown or missing:
        raise ValueError("spec %r: intensity axes of %s are %s (+ optional %s); missing %s, unexpected %s"
                         % (spec.id, spec.family, list(required), list(optional), missing, unknown))
    chosen = {}
    for axis in sorted(spec.intensity):
        value = spec.intensity[axis]
        if isinstance(value, list):
            if not value:
                raise ValueError("spec %r: intensity.%s is an empty list" % (spec.id, axis))
            value = stream.choice("intensity." + axis, value)
        number = finite_float("spec %r intensity.%s" % (spec.id, axis), value)
        lo, hi = settings.intensity_ranges[spec.family][axis]
        if not lo <= number <= hi:
            raise ValueError("spec %r: intensity.%s=%r is outside the registered range [%r, %r] (intensity.yaml)"
                             % (spec.id, axis, number, lo, hi))
        chosen[axis] = number
    return chosen


def _motion_params(spec: HazardSpec) -> Mapping[str, Any]:
    primitive, required, optional = PRIMITIVES[spec.family]
    if spec.motion.primitive != primitive:
        raise ValueError("spec %r: family %s needs motion.primitive %r, got %r"
                         % (spec.id, spec.family, primitive, spec.motion.primitive))
    params = spec.motion.params
    if "follow_through" in params:
        if spec.family == FAMILY_FALL:
            raise ValueError("spec %r: motion.params.follow_through is not supported for %s (a fall is dynamic "
                             "from its onset and has no exit path)" % (spec.id, FAMILY_FALL))
        if params["follow_through"] not in FOLLOW_THROUGH_MODES:
            raise ValueError("spec %r: motion.params.follow_through must be one of %s, got %r"
                             % (spec.id, list(FOLLOW_THROUGH_MODES), params["follow_through"]))
    missing = [k for k in required if k not in params]
    unknown = sorted(set(params) - set(required + optional))
    if missing or unknown:
        raise ValueError("spec %r: motion.params of %r take %s (+ optional %s); missing %s, unexpected %s"
                         % (spec.id, primitive, list(required), list(optional), missing, unknown))
    return params


def _direction(params: Mapping[str, Any], stream: _Stream, horizontal: bool) -> Tuple[np.ndarray, str]:
    if "direction" in params:
        d = unit_vec3("motion.params.direction", params["direction"])
        if horizontal and d[2] != 0.0:
            raise ValueError("a slide direction must be horizontal (z = 0), got %r" % (params["direction"],))
        return d, "spec"
    azimuth = 2.0 * math.pi * stream.uniform("direction")
    return np.array([math.cos(azimuth), math.sin(azimuth), 0.0]), "rng"


def _surface_under(surfaces: List[Any], point: np.ndarray, max_height: float) -> Optional[Any]:
    best = None
    for surface in surfaces:
        top = surface.max_m[2]
        inside = all(surface.min_m[i] <= point[i] <= surface.max_m[i] for i in range(2))
        if inside and 0.0 <= point[2] - top <= max_height and (best is None or top > best.max_m[2]):
            best = surface
    return best


def _slide_normal(agent: AgentSpec, direction: np.ndarray, dz: float, r_eef: float) -> Tuple[np.ndarray, float]:
    """Normal ``-direction cos(phi) + z sin(phi)`` with ``standoff(phi) sin(phi) = dz`` (bisection)."""
    def at(phi: float) -> Tuple[np.ndarray, float]:
        normal = -direction * math.cos(phi) + np.array(_UP) * math.sin(phi)
        return normal, support_radius_m(agent, normal) + r_eef

    lo, hi = -0.5 * math.pi, 0.5 * math.pi
    if at(lo)[1] * math.sin(lo) > dz or at(hi)[1] * math.sin(hi) < dz:
        raise InstanceRejected("no_support_surface", "the EEF is too far above the surface to be touched")
    while True:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        _, standoff = at(mid)
        if standoff * math.sin(mid) < dz:
            lo = mid
        else:
            hi = mid
    return at(0.5 * (lo + hi))


def _stream_spec_hash(spec: HazardSpec) -> str:
    """The spec hash that keys the random stream: ``follow_through`` only adds an exit path, it must not
    change the directions or intensity choices drawn for the instance."""
    if "follow_through" not in spec.motion.params:
        return spec.hash()
    data = spec.to_dict()
    del data["motion"]["params"]["follow_through"]
    return HazardSpec.from_dict(data).hash()


def _interp_error_m(path: Any, times: np.ndarray, positions: np.ndarray) -> float:
    """Largest linear-interpolation error of the keyframes ``(times, positions)`` against ``path`` at the
    interior points of every interval."""
    replay = SampledPath(times, positions, None)
    fractions = np.linspace(0.0, 1.0, 9)[1:-1]
    probes = (times[:-1, None] + fractions[None, :] * np.diff(times)[:, None]).ravel()
    return float(np.max(np.linalg.norm(replay.pos(probes) - path.pos(probes), axis=1)))


class _RobotAt:
    """The robot geometry of a TaskContext placed at one instant (D15 a)."""

    def __init__(self, geometry: RobotGeometry, paths: BodyPosePaths, t: float) -> None:
        self.geometry = geometry
        self.points, self.bodies = robot_points_at(geometry, paths, t)

    def contact(self, agent: AgentSpec, p_ref: np.ndarray, normal: np.ndarray) -> Any:
        return first_contact_along_normal(agent, p_ref, normal, self.points, self.bodies)


def _robot_geometry(task_context: TaskContext, bd: BenchmarkDescriptor) -> Optional[Tuple[RobotGeometry, BodyPosePaths]]:
    """Geometry and body pose paths of the TaskContext, or ``None`` without geometry."""
    prov = task_context.provenance
    data = prov.get("robot_geometry")
    if data is None:
        return None
    paths, quats = prov.get("robot_body_paths"), prov.get("robot_body_quat_paths")
    if not paths or not quats:
        raise ValueError("TaskContext %s/%s has robot_geometry but no robot_body_paths / robot_body_quat_paths"
                         % (task_context.task_id, task_context.layout_id))
    geometry = RobotGeometry.from_dict(data)
    spacing = bd.robot_point_spacing_m
    if is_unknown(spacing) or float(spacing) != geometry.spacing_m:
        raise ValueError("BD %r says robot_point_spacing_m = %r but TaskContext %s/%s was probed with robot geometry "
                         "spaced %r; use the BD the probe ran with" % (bd.name, spacing, task_context.task_id,
                                                                      task_context.layout_id, geometry.spacing_m))
    times = task_context.nominal_ee_path.time_s
    return geometry.restricted(sorted(paths)), BodyPosePaths(times, paths, quats)


def _geometric_slide_normal(agent: AgentSpec, direction: np.ndarray, dz: float, p_ref: np.ndarray,
                            robot: _RobotAt) -> Tuple[np.ndarray, Any]:
    """:func:`_slide_normal` with the geometric standoff (module docstring, step 4)."""
    def normal_at(phi: float) -> np.ndarray:
        return -direction * math.cos(phi) + np.array(_UP) * math.sin(phi)

    def contact_at(phi: float) -> Any:
        contact = robot.contact(agent, p_ref, normal_at(phi))
        if contact is None:
            raise InstanceRejected("no_robot_contact_along_normal",
                                   "a slide tilted by %r rad toward the EEF misses every robot sample point" % phi)
        return contact

    lo, hi = -0.5 * math.pi, 0.5 * math.pi
    if contact_at(lo).standoff_m * math.sin(lo) > dz or contact_at(hi).standoff_m * math.sin(hi) < dz:
        raise InstanceRejected("no_support_surface", "the EEF is too far above the surface to be touched")
    while True:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if contact_at(mid).standoff_m * math.sin(mid) < dz:
            lo = mid
        else:
            hi = mid
    values = {phi: point_standoffs(agent, p_ref, normal_at(phi), robot.points) for phi in (lo, hi)}
    first = {phi: int(np.argmax(v)) for phi, v in values.items()}
    if not (np.isfinite(values[lo][first[hi]]) and np.isfinite(values[hi][first[lo]])):
        raise InstanceRejected("slide_normal_unsolved",
                               "standoff(phi) sin(phi) = %r has no root: the bisection closes at phi = %r on a jump "
                               "of the geometric standoff (%r -> %r m)"
                               % (dz, hi, float(values[lo].max()), float(values[hi].max())))
    phi = 0.5 * (lo + hi)
    return normal_at(phi), contact_at(phi)


def _follow_through(mode: str, plan: Any, path: Any, times: np.ndarray, positions: np.ndarray,
                    task_context: TaskContext, agent: AgentSpec,
                    settings: InstantiateSettings) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Module docstring, step 6: ``(keyframe times, positions)`` cut at the release, the ``exit_keyframes`` data
    and the ``provenance.follow_through`` record."""
    t_release = plan.release_time_s
    kept = times <= t_release
    times, positions = times[kept], positions[kept]
    if times[-1] != t_release:
        raise ValueError("internal: the release time %r is not a keyframe time (last kept: %r)"
                         % (t_release, times[-1]))
    p_release = positions[-1].copy()
    dt = settings.keyframe_dt_s
    record = {"mode": mode, "keyframe_dt_s": dt}  # type: Dict[str, Any]
    if mode == "pass":
        velocity = np.array(plan.release_velocity, dtype=np.float64)
        ws = task_context.workspace_bounds
        radius = bounding_radius_m(agent.shape, agent.half_extents_m)
        lo, hi = np.array(ws.min_m) - radius, np.array(ws.max_m) + radius
        try:
            steps = straight_exit_steps(p_release, velocity, lo, hi, dt)
        except ValueError as exc:
            raise InstanceRejected("exit_path_invalid", str(exc)) from None
        while True:
            exit_times = hook_times(t_release, t_release + steps * dt, dt)
            exit_pos = p_release + np.outer(exit_times - t_release, velocity)
            if np.any(exit_pos[-1] < lo) or np.any(exit_pos[-1] > hi):
                break
            steps += 1
        record.update(release_velocity_mps=velocity.tolist(), workspace_widened_by_m=radius,
                      max_interp_error_m=0.0)
    else:
        retract = RetractPath(path, t_release)
        exit_times = insert_instants(hook_times(retract.t_start_s, retract.t_end_s, dt), [retract.t_turn_s], dt)
        exit_pos = retract.pos(exit_times)
        exit_pos[0] = p_release
        error = _interp_error_m(retract, exit_times, exit_pos)
        if error > settings.keyframe_interp_tol_m:
            raise InstanceRejected("keyframe_interp_error",
                                   "exit keyframes every %r s miss the analytic retraction by %r m > "
                                   "keyframe_interp_tol_m=%r" % (dt, error, settings.keyframe_interp_tol_m))
        record.update(continuation_s=retract.t_turn_s - t_release, retrace_s=retract.t_turn_s - path.t_start_s,
                      max_interp_error_m=error)
    record.update(exit_duration_s=float(exit_times[-1] - t_release), exit_end_s=float(exit_times[-1]),
                  exit_keyframe_count=int(exit_times.size), keyframe_count=int(times.size))
    exit_data = {"time_s": exit_times.tolist(), "pos": exit_pos.tolist(),
                 "quat": [list(IDENTITY_QUAT)] * exit_times.size}
    return times, positions, exit_data, record


def instantiate(spec: HazardSpec, task_context: TaskContext, seed: int, bd: BenchmarkDescriptor,
                settings: InstantiateSettings) -> HazardInstance:
    """See the module docstring."""
    if not isinstance(spec, HazardSpec) or not isinstance(task_context, TaskContext):
        raise TypeError("instantiate needs a HazardSpec and a TaskContext")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int, got %r" % (seed,))
    family = spec.family
    if family not in FAMILY_AXES:
        raise ValueError("spec %r: family %r is not supported by the builder (supported: %s)"
                         % (spec.id, family, sorted(FAMILY_AXES)))
    if spec.timing.mode != "nominal_time":
        raise ValueError("spec %r: timing.mode %r is not supported; the runtime follows keyframes in time "
                         "(nominal_time)" % (spec.id, spec.timing.mode))
    if spec.release.mode != "dynamic_release":
        raise ValueError("spec %r: release.mode %r is not supported (dynamic_release)" % (spec.id, spec.release.mode))
    exposure = spec.exposure
    if exposure.anchor != "nominal_ee_path" or exposure.anchor_phase is None:
        raise ValueError("spec %r: exposure must be anchored to 'nominal_ee_path' with an anchor_phase, got %r"
                         % (spec.id, exposure.anchor))
    phases = {p.name: p for p in task_context.phases}
    ref = exposure.anchor_phase
    if ref.phase not in phases:
        raise ValueError("spec %r: anchor phase %r is not in the TaskContext (phases: %s)"
                         % (spec.id, ref.phase, [p.name for p in task_context.phases]))
    phase = phases[ref.phase]
    t_star = phase.t_start_s + ref.frac * (phase.t_end_s - phase.t_start_s)
    nominal = SampledPath.from_pose_sequence(task_context.nominal_ee_path)
    offset = np.array(exposure.offset_m, dtype=np.float64)
    r_eef = _known(bd, "eef_contact_radius_m")
    robot_geo = _robot_geometry(task_context, bd)
    agent = spec.agent
    window = settings.release_velocity_window_s
    tc_hash = task_context.hash()
    stream = _Stream(sha256_hex({"spec": _stream_spec_hash(spec), "task_context": tc_hash, "task_id": task_context.task_id,
                                 "layout_id": task_context.layout_id, "seed": seed}))
    intensity = _intensity(spec, settings, stream)
    motion = _motion_params(spec)
    tau = intensity["tau_s"]
    extra = {}  # type: Dict[str, Any]

    if family == FAMILY_FALL:
        gravity = _known(bd, "gravity_mps2")
        if "mass_kg" in intensity and intensity["mass_kg"] != agent.mass_kg:
            raise ValueError("spec %r: intensity.mass_kg=%r differs from agent.mass_kg=%r"
                             % (spec.id, intensity["mass_kg"], agent.mass_kg))
        t_hit = t_star
        normal = np.array(_UP)
        standoff = support_radius_m(agent, normal) + r_eef
        p_ref = nominal.pos(t_hit) + offset
        params = {"g": gravity}  # type: Dict[str, Any]
        contact = None if robot_geo is None else _RobotAt(robot_geo[0], robot_geo[1], t_hit).contact(agent, p_ref, normal)
        extra["mapping"] = {"release_height_m": fall_release_height_m(tau, gravity), "gravity_mps2": gravity}
    elif family == FAMILY_REACH:
        direction, extra["direction_source"] = _direction(motion, stream, horizontal=False)
        profile = motion["speed_profile"]
        try:
            geometry = reach_geometry(intensity["approach_speed_mps"], tau, settings.reach_overshoot_fraction,
                                      profile)
        except ZeroOvershootError as exc:
            raise InstanceRejected("reach_zero_overshoot", str(exc)) from None
        t_hit = t_star
        normal = -direction
        standoff = support_radius_m(agent, normal) + r_eef
        p_ref = nominal.pos(t_hit) + offset
        contact = None if robot_geo is None else _RobotAt(robot_geo[0], robot_geo[1], t_hit).contact(agent, p_ref, normal)
        params = {"direction": direction.tolist(), "reach_distance_m": geometry.reach_distance_m,
                  "overshoot_m": geometry.overshoot_m, "speed_profile": profile, "release_window_s": window}
        extra["mapping"] = {"reach_distance_m": geometry.reach_distance_m, "overshoot_m": geometry.overshoot_m,
                            "overshoot_fraction": settings.reach_overshoot_fraction, "stroke_m": geometry.stroke_m,
                            "u_hit": geometry.u_hit, "stroke_duration_s": geometry.stroke_duration_s,
                            "speed_profile": profile, "limb_size_m": intensity.get("limb_size_m")}
    else:
        direction, extra["direction_source"] = _direction(motion, stream, horizontal=True)
        h_z = half_height_m(agent)
        candidates = [t_star] + [float(t) for t in nominal.times_s if phase.t_start_s <= t <= phase.t_end_s]
        chosen = None
        for t in sorted(set(candidates), key=lambda value: (abs(value - t_star), value)):
            point = nominal.pos(t) + offset
            surface = _surface_under(task_context.support_surfaces, point, 2.0 * h_z + r_eef)
            if surface is not None:
                chosen = (t, point, surface)
                break
        if chosen is None:
            raise InstanceRejected("no_support_surface",
                                   "no time in phase %r where the nominal EEF is within %r m above a support "
                                   "surface (%d surface(s))" % (phase.name, 2.0 * h_z + r_eef,
                                                                len(task_context.support_surfaces)))
        t_hit, p_ref, surface = chosen
        top = surface.max_m[2]
        if robot_geo is None:
            normal, standoff = _slide_normal(agent, direction, top + h_z - p_ref[2], r_eef)
            contact = None
        else:
            normal, contact = _geometric_slide_normal(agent, direction, top + h_z - p_ref[2], p_ref,
                                                      _RobotAt(robot_geo[0], robot_geo[1], t_hit))
        params = {"direction": direction.tolist(), "speed_mps": intensity["approach_speed_mps"],
                  "release_window_s": window}
        extra["mapping"] = {"start_distance_m": slide_start_distance_m(intensity["approach_speed_mps"], tau),
                            "speed_mps": intensity["approach_speed_mps"]}
        extra["support_surface"] = dict(surface.to_dict(), top_m=top, hazard_half_height_m=h_z)

    if robot_geo is not None:
        if contact is None:
            raise InstanceRejected("no_robot_contact_along_normal",
                                   "a hazard approaching p_ref = %s along %s at t_hit = %r misses every robot sample "
                                   "point" % (p_ref.tolist(), normal.tolist(), t_hit))
        standoff = contact.standoff_m
        geo = {"standoff_source": "robot_geometry", "standoff_body": contact.body,
               "robot_point_spacing_m": robot_geo[0].spacing_m, "contact_point": contact.point.tolist(),
               "hazard_support_radius_m": support_radius_m(agent, normal)}
    else:
        geo = {"standoff_source": "eef_contact_radius", "standoff_body": None, "robot_point_spacing_m": None,
               "contact_point": (p_ref + r_eef * normal).tolist(), "hazard_support_radius_m": standoff - r_eef}
    centroid = p_ref + standoff * normal
    if family == FAMILY_SLIDE:
        centroid[2] = extra["support_surface"]["top_m"] + extra["support_surface"]["hazard_half_height_m"]
    try:
        plan = build_hit_path(family, nominal, t_hit, centroid, tau, params)
    except ValueError as exc:
        raise InstanceRejected("hit_path_invalid", str(exc)) from None

    path = plan.path
    grid = hook_times(path.t_start_s, path.t_end_s, settings.keyframe_dt_s)
    times = insert_instants(grid, [plan.nominal_hit_time_s, plan.release_time_s], settings.keyframe_dt_s)
    positions = path.pos(times)
    replay = SampledPath(times, positions, None)
    interp_error = _interp_error_m(path, times, positions)
    if interp_error > settings.keyframe_interp_tol_m:
        raise InstanceRejected("keyframe_interp_error",
                               "keyframes every %r s miss the analytic path by %r m > keyframe_interp_tol_m=%r"
                               % (settings.keyframe_dt_s, interp_error, settings.keyframe_interp_tol_m))
    if family != FAMILY_FALL:
        extra["keyframe_release_velocity_mps"] = replay.finite_difference_velocity(plan.release_time_s, window).tolist()
        extra["release_window_s"] = window

    exit_keyframes = None  # type: Optional[Dict[str, Any]]
    if "follow_through" in motion:
        times, positions, exit_keyframes, extra["follow_through"] = _follow_through(
            motion["follow_through"], plan, path, times, positions, task_context, agent, settings)

    instance_id = "%s.%s.%s.s%d" % (spec.id, task_context.task_id, task_context.layout_id, seed)
    provenance = {
        "builder": "hazardlib.instantiate",
        "task_id": task_context.task_id, "layout_id": task_context.layout_id, "instance_seed": seed,
        "spec_hash": spec.hash(), "task_context_hash": tc_hash, "bd_hash": bd.hash(),
        "settings": settings.to_dict(),
        "anchor": {"phase": ref.phase, "frac": ref.frac, "t_star_s": t_star, "offset_m": list(exposure.offset_m)},
        "anchor_point": p_ref.tolist(),
        "contact_normal": normal.tolist(),
        "contact_point": geo["contact_point"],
        "standoff_m": standoff,
        "standoff_source": geo["standoff_source"],
        "standoff_body": geo["standoff_body"],
        "robot_point_spacing_m": geo["robot_point_spacing_m"],
        "hazard_support_radius_m": geo["hazard_support_radius_m"],
        "eef_contact_radius_m": r_eef,
        "hazard_quat_wxyz": list(IDENTITY_QUAT),
        "hazard_quat_source": "identity (HazardSpec.agent has no orientation field)",
        "intensity": intensity,
        "tau_s": tau,
        "hit_diagnostics": dict(plan.diagnostics),
        "hit_velocity_mps": plan.hit_velocity.tolist(),
        "keyframes": {"dt_s": settings.keyframe_dt_s, "count": int(times.size), "max_interp_error_m": interp_error,
                      "tol_m": settings.keyframe_interp_tol_m},
    }
    provenance.update(extra)
    data = {
        "instance_id": instance_id, "spec_id": spec.id, "family": family, "agent": agent.to_dict(),
        "keyframes": {"time_s": times.tolist(), "pos": positions.tolist(), "quat": [list(IDENTITY_QUAT)] * times.size},
        "nominal_hit_point": plan.nominal_hit_point.tolist(), "nominal_hit_time_s": plan.nominal_hit_time_s,
        "onset_time_s": plan.onset_time_s, "release_time_s": plan.release_time_s,
        "release_velocity": {"lin_mps": plan.release_velocity.tolist(), "ang_radps": [0.0, 0.0, 0.0]},
        "certificate": None, "provenance": provenance,
    }
    if exit_keyframes is not None:
        data["exit_keyframes"] = exit_keyframes
    return HazardInstance.from_dict(data)


def slide_support_rule(task_context: TaskContext, bd: BenchmarkDescriptor, agent: AgentSpec):
    """The builder's own slide-support test (module docstring, step 4) as ``(p_ref, t_hit_s) -> bool``: can a
    slide be built with the reference point ``p_ref`` at ``t_hit_s``?

    Two builder conditions hold the support rule: (1) ``p_ref`` lies over a TaskContext support surface with
    ``0 <= p_ref_z - top <= 2 h_z + eef_contact_radius_m``; (2) the tilt bracket of the normal contains the
    required height step ``dz = top + h_z - p_ref_z`` (with robot geometry the standoff along the vertical is
    the geometric one, so this is tighter than (1): the box riding on the table must reach the lowest finger
    point). Both are evaluated by the builder's own functions (``_surface_under``, ``_geometric_slide_normal``
    / ``_slide_normal``, ``half_height_m``) on the TaskContext and BD, not by a copy of their formulas. A
    rejection other than ``no_support_surface`` (``no_robot_contact_along_normal``, ``slide_normal_unsolved``)
    depends on the slide direction, not on the support, and does not make the point unsupported. Generators
    use it to place slide anchors only where :func:`instantiate` can find a support."""
    h_z = half_height_m(agent)
    r_eef = _known(bd, "eef_contact_radius_m")
    robot = _robot_geometry(task_context, bd)
    direction = np.array([1., 0., 0.])    # the bracket (phi = +-pi/2) does not depend on the direction

    def rule(point: Any, t_s: float) -> bool:
        point = np.asarray(point, dtype=float)
        surface = _surface_under(task_context.support_surfaces, point, 2.0 * h_z + r_eef)
        if surface is None:
            return False
        dz = surface.max_m[2] + h_z - point[2]
        try:
            if robot is None:
                _slide_normal(agent, direction, dz, r_eef)
            else:
                _geometric_slide_normal(agent, direction, dz, point, _RobotAt(robot[0], robot[1], float(t_s)))
        except InstanceRejected as exc:
            return exc.reason != "no_support_surface"
        return True
    return rule
