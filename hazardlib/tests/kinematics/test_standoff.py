"""Geometric standoff (T07, D15 a): first contact along the normal, penetration depth, body frames, sampling error.

Analytic cases: a sphere / an axis-aligned box hazard against a single point and against a planar point lattice;
misses; non-unit normals; and a sampled sphere body, whose result must stay within the sampling-error bound of the
scalar formula ``r_hazard + r_e`` (module docstring of ``hazardlib.kinematics.standoff``).
"""
import math

import numpy as np
import pytest

from hazardlib.conformance.fakehost import sphere_surface_points
from hazardlib.kinematics.standoff import (BodyPosePaths, RobotGeometry, first_contact_along_normal, hazard_shape,
                                           penetration_depths, point_standoffs, quat_matrix, robot_points_at,
                                           sphere_sampling_error_bound, unit_normal)
from hazardlib.spec import AgentSpec

UP = np.array([0.0, 0.0, 1.0])


def sphere(r):
    return AgentSpec(kind="object", shape="sphere", half_extents_m=[r], mass_kg=0.1)


def box(hx, hy, hz):
    return AgentSpec(kind="object", shape="box", half_extents_m=[hx, hy, hz], mass_kg=0.1)


def lattice(spacing, half_width, z=0.0, center=(0.0, 0.0)):
    k = int(round(half_width / spacing))
    xs = center[0] + spacing * np.arange(-k, k + 1)
    ys = center[1] + spacing * np.arange(-k, k + 1)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)


# ---------------------------------------------------------------------------------------------- one point

@pytest.mark.parametrize("offset", [0.0, 0.01, 0.019])
def test_sphere_against_one_point_is_the_chord_formula(offset):
    r = 0.02
    p_ref = np.array([0.1, -0.2, 0.3])
    q = p_ref + np.array([offset, 0.0, -0.05])
    contact = first_contact_along_normal(sphere(r), p_ref, UP, [q], ["b"])
    assert contact.standoff_m == pytest.approx(-0.05 + math.sqrt(r * r - offset * offset), abs=1e-15)
    assert contact.body == "b" and contact.index == 0
    np.testing.assert_array_equal(contact.point, q)
    centre = p_ref + contact.standoff_m * UP
    assert np.linalg.norm(centre - q) == pytest.approx(r, abs=1e-15)  # touching, not inside


def test_box_against_one_point_is_the_upper_end_of_the_interval():
    agent = box(0.03, 0.02, 0.01)
    p_ref = np.zeros(3)
    # straight above the point: the bottom face touches it at s = h_z
    assert first_contact_along_normal(agent, p_ref, UP, [[0.0, 0.0, 0.0]], ["b"]).standoff_m == pytest.approx(0.01)
    # off the axis but under the face
    assert first_contact_along_normal(agent, p_ref, UP, [[0.029, -0.019, 0.0]], ["b"]).standoff_m == \
        pytest.approx(0.01)
    # oblique normal (1, 0, 1)/sqrt2: the leading corner reaches the origin when s/sqrt2 = h_x and s/sqrt2 = h_z
    n = np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0)
    s = point_standoffs(agent, p_ref, n, [[0.0, 0.0, 0.0]])[0]
    assert s == pytest.approx(math.sqrt(2.0) * 0.01)  # the z face is the binding one (h_z < h_x)


def test_a_point_missed_from_the_side_is_no_contact():
    assert first_contact_along_normal(sphere(0.02), np.zeros(3), UP, [[0.0201, 0.0, -0.1]], ["b"]) is None
    assert first_contact_along_normal(box(0.03, 0.02, 0.01), np.zeros(3), UP, [[0.0, 0.021, -0.1]], ["b"]) is None
    values = point_standoffs(sphere(0.02), np.zeros(3), UP, [[0.0201, 0.0, -0.1], [0.0, 0.0, -0.1]])
    assert values[0] == -np.inf and values[1] == pytest.approx(-0.08)
    assert first_contact_along_normal(sphere(0.02), np.zeros(3), UP, np.zeros((0, 3)), []) is None


@pytest.mark.parametrize("normal", [[0.0, 0.0, 2.0], [0.0, 0.0, 0.999], [0.6, 0.0, 0.6], [0.0, 0.0, 0.0]])
def test_non_unit_normals_are_refused(normal):
    with pytest.raises(ValueError, match="unit vector"):
        first_contact_along_normal(sphere(0.02), np.zeros(3), normal, [[0.0, 0.0, 0.0]], ["b"])
    with pytest.raises(ValueError, match="unit vector"):
        unit_normal("n", normal)


def test_unit_normals_from_normalisation_are_accepted():
    v = np.array([0.3, -0.7, 0.2])
    unit_normal("n", v / np.linalg.norm(v))


@pytest.mark.parametrize("agent", [AgentSpec(kind="o", shape="cylinder", half_extents_m=[0.02, 0.05], mass_kg=0.1),
                                   AgentSpec(kind="o", shape="sphere", half_extents_m=[0.02, 0.03], mass_kg=0.1),
                                   AgentSpec(kind="o", shape="box", half_extents_m=[0.02, 0.03], mass_kg=0.1)])
def test_other_shapes_are_refused(agent):
    with pytest.raises(ValueError):
        hazard_shape(agent)
    with pytest.raises(ValueError):
        first_contact_along_normal(agent, np.zeros(3), UP, [[0.0, 0.0, 0.0]], ["b"])


def test_bodies_must_match_the_points():
    with pytest.raises(ValueError, match="2 names for 1 points"):
        first_contact_along_normal(sphere(0.02), np.zeros(3), UP, [[0.0, 0.0, 0.0]], ["a", "b"])


# ------------------------------------------------------------------------------------------- point lattice

def test_sphere_lowered_onto_a_lattice_rests_at_its_radius():
    r, g = 0.02, 0.002
    contact = first_contact_along_normal(sphere(r), [0.0, 0.0, 0.0], UP, lattice(g, 0.05), ["plane"] * 2601)
    assert contact.standoff_m == pytest.approx(r, abs=1e-15)  # the lattice point under the centre


@pytest.mark.parametrize("tilt_deg", [10.0, 30.0, 50.0])
def test_tilted_approach_onto_a_lattice_matches_the_plane_within_the_sampling_bound(tilt_deg):
    """A sphere moving along n = (sin a, 0, cos a) touches the plane z = 0 when its centre is r high: s = r / cos a.
    With samples g apart, a point lies within g / sqrt(2) of the true contact point, so s is low by at most
    r - sqrt(r^2 - g^2 / 2) (divided by cos a)."""
    r, g = 0.02, 0.002
    a = math.radians(tilt_deg)
    n = np.array([math.sin(a), 0.0, math.cos(a)])
    p_ref = np.array([0.0003, 0.0007, 0.0])  # off the lattice nodes
    s = first_contact_along_normal(sphere(r), p_ref, n, lattice(g, 0.08), ["plane"] * 6561).standoff_m
    exact = r / math.cos(a)
    bound = (r - math.sqrt(r * r - g * g / 2.0)) / math.cos(a)
    assert exact - bound - 1e-12 <= s <= exact + 1e-12


@pytest.mark.parametrize("tilt_deg", [0.0, 20.0, 45.0])
def test_box_onto_a_lattice_rests_on_its_bottom_face(tilt_deg):
    """An axis-aligned box touches the plane when its centre is h_z high, whatever the tilt of the approach: s = h_z /
    cos a, exactly, as long as a lattice point lies under the bottom face."""
    agent = box(0.03, 0.02, 0.01)
    a = math.radians(tilt_deg)
    n = np.array([0.0, math.sin(a), math.cos(a)])
    s = first_contact_along_normal(agent, [0.0011, 0.0, 0.0], n, lattice(0.002, 0.08), ["plane"] * 6561).standoff_m
    assert s == pytest.approx(0.01 / math.cos(a), abs=1e-12)


# ------------------------------------------------------------------------------- sampled sphere body

@pytest.mark.parametrize("spacing", [0.002, 0.005, 0.01])
def test_sampled_sphere_body_matches_the_scalar_formula_within_the_bound(spacing):
    """Point cloud on a sphere of radius r_e around p_ref: s* <= r + r_e always (all samples are on the surface) and
    r + r_e - s* <= the bound of a chord gap equal to the spacing (every surface point is within one spacing of a
    sample: neighbouring rings and neighbouring points of a ring are at most one spacing apart)."""
    r, r_e = 0.02, 0.03
    points = sphere_surface_points(r_e, spacing)
    rng = np.random.default_rng(7)
    bound = sphere_sampling_error_bound(r, r_e, spacing)
    worst = 0.0
    for _ in range(200):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        s = first_contact_along_normal(sphere(r), np.zeros(3), n, points, ["hand"] * len(points)).standoff_m
        assert s <= r + r_e + 1e-15
        worst = max(worst, r + r_e - s)
    assert worst <= bound
    print("spacing %.3f m: worst s* error %.3e m over 200 directions, bound %.3e m" % (spacing, worst, bound))


def test_axis_directions_are_sampled_exactly():
    r, r_e = 0.02, 0.03
    points = sphere_surface_points(r_e, 0.005)
    for n in np.vstack([np.eye(3), -np.eye(3)]):
        s = first_contact_along_normal(sphere(r), np.zeros(3), n, points, ["hand"] * len(points)).standoff_m
        assert s == pytest.approx(r + r_e, abs=1e-15)


def test_sampling_error_bound_matches_its_small_gap_approximation():
    r, r_e, delta = 0.02, 0.03, 0.001
    exact = sphere_sampling_error_bound(r, r_e, delta)
    assert exact == pytest.approx(delta * delta / 2.0 * (1.0 / r_e + 1.0 / r), rel=1e-3)
    with pytest.raises(ValueError):
        sphere_sampling_error_bound(r, r_e, 2.5 * r_e)


def test_sphere_surface_points_respect_the_spacing():
    points = sphere_surface_points(0.03, 0.004)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 0.03, atol=1e-15)
    # nearest-neighbour distance of every sample is at most the spacing
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min(axis=1).max() <= 0.004


# ---------------------------------------------------------------------------------- penetration depth

def test_penetration_depths_of_sphere_and_box():
    np.testing.assert_allclose(penetration_depths(sphere(0.02), [0.0, 0.0, 0.0], [[0.0, 0.0, 0.015], [0.03, 0, 0]]),
                               [0.005, -0.01])
    np.testing.assert_allclose(penetration_depths(box(0.03, 0.02, 0.01), [0.0, 0.0, 0.0],
                                                  [[0.0, 0.0, 0.0], [0.025, 0.0, 0.0], [0.0, 0.0, 0.02]]),
                               [0.01, 0.005, -0.01])


def test_first_contact_has_zero_depth_and_nothing_deeper():
    n = np.array([0.0, 0.6, 0.8])
    points = sphere_surface_points(0.03, 0.005) + 0.15 * n + np.array([0.01, 0.0, 0.0])
    contact = first_contact_along_normal(sphere(0.02), np.zeros(3), n, points, ["w"] * len(points))
    depths = penetration_depths(sphere(0.02), contact.standoff_m * n, points)
    assert abs(depths[contact.index]) <= 1e-15 and depths.max() <= 1e-15


# ------------------------------------------------------------------------------------ frames and paths

def test_quat_matrix_rotates_wxyz():
    half = math.sqrt(0.5)
    np.testing.assert_allclose(quat_matrix([half, 0.0, 0.0, half]) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(quat_matrix([2.0, 0.0, 0.0, 0.0]), np.eye(3))
    with pytest.raises(ValueError):
        quat_matrix([0.0, 0.0, 0.0, 0.0])


def test_world_points_place_each_body_and_name_its_points():
    half = math.sqrt(0.5)
    geometry = RobotGeometry({"a": [[1.0, 0.0, 0.0]], "b": [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]}, 0.5)
    points, names = geometry.world_points({"b": [1.0, 1.0, 1.0], "a": [0.0, 0.0, 0.0]},
                                          {"a": [half, 0.0, 0.0, half], "b": [1.0, 0.0, 0.0, 0.0]})
    assert names == ["a", "b", "b"]
    np.testing.assert_allclose(points, [[0.0, 1.0, 0.0], [1.0, 1.0, 2.0], [1.0, 1.0, 3.0]], atol=1e-15)
    with pytest.raises(ValueError, match="no robot geometry"):
        geometry.world_points({"c": [0.0, 0.0, 0.0]}, {"c": [1.0, 0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="no orientation"):
        geometry.world_points({"a": [0.0, 0.0, 0.0]}, {})
    assert RobotGeometry.from_dict(geometry.to_dict()).to_dict() == geometry.to_dict()
    with pytest.raises(ValueError, match="no points for the body"):
        geometry.restricted(["a", "z"])


def test_body_poses_interpolate_linearly_and_by_nlerp():
    half = math.sqrt(0.5)
    paths = BodyPosePaths([0.0, 1.0], {"a": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]},
                          {"a": [[1.0, 0.0, 0.0, 0.0], [half, 0.0, 0.0, half]]})
    pos, quat = paths.at(0.5)
    np.testing.assert_allclose(pos["a"], [1.0, 0.0, 0.0])
    # nlerp halfway between 0 and 90 deg about z is exactly 45 deg (symmetric endpoints)
    np.testing.assert_allclose(quat["a"], [math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)], atol=1e-15)
    geometry = RobotGeometry({"a": [[1.0, 0.0, 0.0]]}, 0.1)
    points, names = robot_points_at(geometry, paths, 1.0)
    np.testing.assert_allclose(points, [[2.0, 1.0, 0.0]], atol=1e-15)
    # the shorter arc: q and -q are the same rotation
    flipped = BodyPosePaths([0.0, 1.0], {"a": [[0.0, 0.0, 0.0]] * 2},
                            {"a": [[1.0, 0.0, 0.0, 0.0], [-half, 0.0, 0.0, -half]]})
    np.testing.assert_allclose(flipped.at(0.5)[1]["a"], quat["a"], atol=1e-15)
    with pytest.raises(ValueError, match="different bodies"):
        BodyPosePaths([0.0, 1.0], {"a": [[0.0] * 3] * 2}, {})
