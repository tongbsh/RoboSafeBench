import numpy as np
import pytest

from hazardlib.kinematics import Path, SampledPath, straight_path
from hazardlib.spec import PoseSequence

TIMES = [0.0, 1.0, 3.0]
POINTS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]


def sampled():
    return SampledPath(TIMES, POINTS, None)


def test_base_path_is_abstract_and_validates_its_span():
    path = Path(0.0, 1.0)
    with pytest.raises(NotImplementedError):
        path.pos(0.5)
    for bad in ((1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0), (0.0, float("inf"))):
        with pytest.raises(ValueError):
            Path(*bad)


def test_sampled_path_interpolates_and_clamps():
    path = sampled()
    np.testing.assert_allclose(path.pos(0.0), [0, 0, 0])
    np.testing.assert_allclose(path.pos(0.5), [0.5, 0, 0])
    np.testing.assert_allclose(path.pos(1.0), [1, 0, 0])
    np.testing.assert_allclose(path.pos(2.0), [1, 1, 0])
    np.testing.assert_allclose(path.pos(3.0), [1, 2, 0])
    np.testing.assert_allclose(path.pos(-5.0), [0, 0, 0])   # clamped before the start
    np.testing.assert_allclose(path.pos(99.0), [1, 2, 0])   # clamped after the end
    assert (path.t_start_s, path.t_end_s, path.duration_s) == (0.0, 3.0, 3.0)


def test_sampled_path_velocity_is_piecewise_constant_and_zero_outside():
    path = sampled()
    np.testing.assert_allclose(path.vel(0.5), [1, 0, 0])
    np.testing.assert_allclose(path.vel(2.0), [0, 1, 0])
    np.testing.assert_allclose(path.vel(0.0), [1, 0, 0])   # start: first segment
    np.testing.assert_allclose(path.vel(1.0), [0, 1, 0])   # interior sample: segment to its right
    np.testing.assert_allclose(path.vel(3.0), [0, 1, 0])   # last sample: last segment
    np.testing.assert_array_equal(path.vel(-0.001), [0, 0, 0])
    np.testing.assert_array_equal(path.vel(3.001), [0, 0, 0])


def test_scalar_and_array_shapes():
    path = sampled()
    assert path.pos(0.5).shape == (3,) and path.vel(0.5).shape == (3,)
    assert path.pos(np.linspace(0, 3, 5)).shape == (5, 3)
    assert path.pos(np.zeros((2, 4))).shape == (2, 4, 3)
    assert path.vel(np.zeros((2, 4))).shape == (2, 4, 3)
    assert path.pos([]).shape == (0, 3) and path.vel([]).shape == (0, 3)
    np.testing.assert_allclose(path.pos([0.5, 2.0]), [[0.5, 0, 0], [1, 1, 0]])


def test_non_finite_times_are_rejected():
    path = sampled()
    for bad in (float("nan"), float("inf"), [0.0, float("nan")]):
        with pytest.raises(ValueError):
            path.pos(bad)
        with pytest.raises(ValueError):
            path.vel(bad)


@pytest.mark.parametrize("times,points,quat,match", [
    ([0.0], [[0, 0, 0]], None, "at least 2"),
    ([0.0, 1.0, 1.0], POINTS, None, "strictly increasing"),
    ([0.0, 2.0, 1.0], POINTS, None, "strictly increasing"),
    ([0.0, 1.0], POINTS, None, "pos must have shape"),
    (TIMES, [[0, 0], [1, 0], [1, 2]], None, "pos must have shape"),
    (TIMES, POINTS, [[1, 0, 0, 0]] * 2, "quat must have shape"),
    (TIMES, POINTS, [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], "zero quaternion"),
    (TIMES, [[0, 0, 0], [float("nan"), 0, 0], [1, 2, 0]], None, "non-finite"),
])
def test_sampled_path_validation(times, points, quat, match):
    with pytest.raises(ValueError, match=match):
        SampledPath(times, points, quat)


def test_sampled_path_does_not_alias_its_inputs():
    points = np.array(POINTS)
    path = SampledPath(TIMES, points, None)
    points[:] = 99.0
    np.testing.assert_allclose(path.pos(1.0), [1, 0, 0])


def test_from_pose_sequence_uses_the_spec_object():
    seq = PoseSequence.from_dict({"time_s": [0.0, 2.0], "pos": [[0, 0, 0], [2, 0, 0]],
                                  "quat": [[1, 0, 0, 0], [1, 0, 0, 0]]})
    path = SampledPath.from_pose_sequence(seq)
    np.testing.assert_allclose(path.pos(1.0), [1, 0, 0])
    assert path.has_orientation


def test_orientation_is_optional_and_reported():
    plain = sampled()
    assert not plain.has_orientation and plain.quat(0.5) is None
    assert plain.sample([0.0, 1.0])[1] is None
    assert straight_path([0, 0, 0], [1, 0, 0], 0.0, 1.0).quat(0.5) is None


def test_quaternion_interpolation():
    ident = [1.0, 0.0, 0.0, 0.0]
    quarter = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]   # 90 degrees about z
    path = SampledPath([0.0, 1.0], [[0, 0, 0], [1, 0, 0]], [ident, quarter])
    np.testing.assert_allclose(path.quat(0.0), ident, atol=1e-15)
    np.testing.assert_allclose(path.quat(1.0), quarter, atol=1e-15)
    half = path.quat(0.5)   # normalised lerp at the middle equals the 45 degree rotation
    np.testing.assert_allclose(half, [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)], atol=1e-12)
    ts = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(np.linalg.norm(path.quat(ts), axis=1), 1.0, atol=1e-12)
    pos, quat = path.sample(ts)
    assert pos.shape == (11, 3) and quat.shape == (11, 4)


def test_quaternion_takes_the_short_arc_for_antipodal_signs():
    ident = [1.0, 0.0, 0.0, 0.0]
    path = SampledPath([0.0, 1.0], [[0, 0, 0], [1, 0, 0]], [ident, [-1.0, 0.0, 0.0, 0.0]])
    mid = path.quat(0.5)
    assert np.all(np.isfinite(mid)) and abs(np.linalg.norm(mid) - 1.0) < 1e-12
    assert abs(abs(mid[0]) - 1.0) < 1e-12   # same rotation, not a collapsed zero quaternion


# ---------------------------------------------------------- finite-difference velocity

def test_finite_difference_is_the_backward_difference():
    path = sampled()
    dt = 0.1
    expected = (path.pos(2.0) - path.pos(2.0 - dt)) / dt
    np.testing.assert_array_equal(path.finite_difference_velocity(2.0, dt), expected)
    np.testing.assert_allclose(path.finite_difference_velocity(0.5, 0.25), [1, 0, 0])


def test_finite_difference_is_exact_for_constant_velocity():
    path = straight_path([0.1, 0.2, 0.3], [0.1, 0.2 + 1.5, 0.3], 1.0, 2.0)
    np.testing.assert_allclose(path.finite_difference_velocity(2.0, 0.004), [0, 1.5, 0], atol=1e-11)


def test_finite_difference_applies_no_speed_floor():
    slow = straight_path([0, 0, 0], [1e-12, 0, 0], 0.0, 1.0)
    vel = slow.finite_difference_velocity(1.0, 0.5)
    assert 0 < np.linalg.norm(vel) < 1e-11
    still = straight_path([0.3, 0.3, 0.3], [0.3, 0.3, 0.3], 0.0, 1.0)
    np.testing.assert_array_equal(still.finite_difference_velocity(1.0, 0.1), [0, 0, 0])


def test_finite_difference_refuses_a_step_that_leaves_the_span():
    path = sampled()
    with pytest.raises(ValueError, match="not inside"):
        path.finite_difference_velocity(0.05, 0.1)    # step starts before the path
    with pytest.raises(ValueError, match="not inside"):
        path.finite_difference_velocity(3.5, 0.1)     # ends after the path
    for bad_dt in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            path.finite_difference_velocity(1.0, bad_dt)
    with pytest.raises(ValueError):
        path.finite_difference_velocity(np.array([1.0, 2.0]), 0.1)
