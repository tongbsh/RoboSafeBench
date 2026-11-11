"""Oriented-box geometry of the generator, against brute-force sampling (no simulator)."""
import numpy as np
import pytest

from hazardlib.generate.geometry import (box_corners, box_overlap_depth, points_box_distance, points_boxes_distance,
                                         quat_to_mat, world_boxes)


def random_rotation(rng):
    q = rng.normal(size=4)
    return quat_to_mat(q / np.linalg.norm(q))


def random_box(rng, centre_scale):
    return rng.uniform(-centre_scale, centre_scale, 3), random_rotation(rng), rng.uniform(0.2, 1.0, 3)


def sample_inside(rng, box, n):
    c, r, h = box
    return c + (rng.uniform(-1, 1, (n, 3)) * h).dot(r.T)


def inside(points, box):
    c, r, h = box
    return np.all(np.abs((points - c).dot(r)) <= h, axis=-1)


def test_quat_to_mat_is_a_rotation_and_matches_known_cases():
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = random_rotation(rng)
        assert np.allclose(r.dot(r.T), np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1.0)
    quarter = quat_to_mat([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])      # 90 deg about z
    assert np.allclose(quarter.dot([1, 0, 0]), [0, 1, 0], atol=1e-12)
    stack = quat_to_mat(np.tile([1., 0, 0, 0], (4, 5, 1)))
    assert stack.shape == (4, 5, 3, 3) and np.allclose(stack, np.eye(3))


def test_sat_agrees_with_brute_force_sampling_on_random_pairs():
    rng = np.random.default_rng(1)
    overlapping = separated = 0
    for _ in range(300):
        a, b = random_box(rng, 0.5), random_box(rng, 1.8)
        depth = float(box_overlap_depth(*a, *b))
        assert np.isclose(depth, float(box_overlap_depth(*b, *a)), atol=1e-12)        # symmetric
        points_a, points_b = sample_inside(rng, a, 6000), sample_inside(rng, b, 6000)
        sampled = bool(np.any(inside(points_a, b)) or np.any(inside(points_b, a)))
        if depth < 0:
            separated += 1
            assert not sampled, "SAT says separated but a sampled point is shared"
        else:
            overlapping += 1
        if sampled:
            assert depth > 0
    assert overlapping > 30 and separated > 30


def test_sat_depth_is_exact_for_axis_aligned_and_for_face_separation():
    eye = np.eye(3)
    a = (np.zeros(3), eye, np.array([1., 1., 1.]))
    assert np.isclose(box_overlap_depth(*a, np.array([1.5, 0.2, 0.0]), eye, np.array([1., 1., 1.])), 0.5)
    assert np.isclose(box_overlap_depth(*a, np.array([0.0, 0.0, -1.7]), eye, np.array([1., 1., 1.])), 0.3)
    assert np.isclose(box_overlap_depth(*a, np.array([3.0, 0.0, 0.0]), eye, np.array([1., 1., 1.])), -1.0)  # gap 1
    # fully inside: the shortest way out
    assert np.isclose(box_overlap_depth(*a, np.zeros(3), eye, np.array([.1, .1, .1])), 1.1)


def test_sat_needs_the_edge_cross_axes():
    """Two cubes tilted 45 degrees about different axes can only be separated by an edge-edge axis."""
    s = np.sqrt(0.5)
    about_x = np.array([[1, 0, 0], [0, s, -s], [0, s, s]])
    about_y = np.array([[s, 0, s], [0, 1, 0], [-s, 0, s]])
    h = np.array([1., 1., 1.])
    rng = np.random.default_rng(2)
    found = 0
    for _ in range(400):
        offset = rng.uniform(-2.6, 2.6, 3)
        depth = float(box_overlap_depth(np.zeros(3), about_x, h, offset, about_y, h))
        sampled = np.any(inside(sample_inside(rng, (np.zeros(3), about_x, h), 4000), (offset, about_y, h)))
        if depth < 0:
            assert not sampled
            found += 1
        if sampled:
            assert depth > 0
    assert found > 20


def test_sat_is_vectorised_over_leading_dimensions():
    rng = np.random.default_rng(3)
    boxes = [random_box(rng, 1.0) for _ in range(6)]
    a = random_box(rng, 0.5)
    many = box_overlap_depth(a[0], a[1], a[2], np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes]),
                             np.array([b[2] for b in boxes]))
    assert many.shape == (6,)
    assert np.allclose(many, [float(box_overlap_depth(*a, *b)) for b in boxes])


def test_point_box_distance_against_dense_surface_sampling():
    rng = np.random.default_rng(4)
    for _ in range(10):
        box = random_box(rng, 0.2)
        c, r, h = box
        # dense samples of the surface
        faces = []
        for axis in range(3):
            for sign in (-1, 1):
                uv = rng.uniform(-1, 1, (4000, 3)) * h
                uv[:, axis] = sign * h[axis]
                faces.append(c + uv.dot(r.T))
        surface = np.concatenate(faces)
        points = rng.uniform(-2, 2, (40, 3))
        exact = points_box_distance(points, c, r, h)
        brute = np.min(np.linalg.norm(points[:, None, :] - surface[None], axis=-1), axis=1)
        outside = ~inside(points, box)
        assert np.all(exact[outside] <= brute[outside] + 1e-9)
        assert np.allclose(exact[outside], brute[outside], atol=0.06)
        assert np.all(exact[~outside] == 0.0)
    many = points_boxes_distance(points, np.array([b[0] for b in [box, box]]), np.array([box[1], box[1]]),
                                 np.array([box[2], box[2]]))
    assert many.shape == (40, 2) and np.allclose(many[:, 0], exact)


def test_world_boxes_move_the_corners_rigidly():
    rng = np.random.default_rng(5)
    body_rot, body_pos = random_rotation(rng), rng.uniform(-1, 1, 3)
    local = random_box(rng, 0.3)
    centre, rot, half = world_boxes(local[0][None], local[1][None], local[2][None], body_pos, body_rot)
    moved = box_corners(centre[0], rot[0], half[0])
    expected = box_corners(*local).dot(body_rot.T) + body_pos
    assert np.allclose(moved, expected)
