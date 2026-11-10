"""Exit-path kinematics (decision D19): the retracting path and the straight exit of a passing hazard."""
import numpy as np
import pytest

from hazardlib.kinematics import reach_path, straight_path
from hazardlib.kinematics.exit import RetractPath, bounding_radius_m, straight_exit_steps

LO, HI = np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0])


def test_retract_path_follows_the_continuation_then_retraces_the_whole_path():
    base = reach_path([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 3.0, "min_jerk")
    path = RetractPath(base, 2.0)                                 # the hit is at t = 2, the primitive ends at 3
    assert (path.t_start_s, path.t_end_s) == (2.0, 5.0)           # 1 s of continuation + 2 s of retrace
    t = np.array([2.0, 2.5, 3.0])
    np.testing.assert_allclose(path.pos(t), base.pos(t), atol=1e-15)
    for s in (0.0, 0.3, 1.0, 1.7, 2.0):
        np.testing.assert_allclose(path.pos(3.0 + s), base.pos(3.0 - s), atol=1e-15)
    np.testing.assert_allclose(path.pos(5.0), [0.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(path.vel(4.0), -base.vel(2.0), atol=1e-15)
    np.testing.assert_allclose(path.vel(2.5), base.vel(2.5), atol=1e-15)


def test_retract_path_without_continuation_retraces_from_the_hit_pose():
    base = straight_path([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], 1.0, 2.0)
    path = RetractPath(base, 2.0)
    assert (path.t_start_s, path.t_end_s) == (2.0, 3.0)
    np.testing.assert_allclose(path.pos(2.0), [0.3, 0.0, 0.0])
    np.testing.assert_allclose(path.pos(2.5), [0.15, 0.0, 0.0])
    np.testing.assert_allclose(path.pos(3.0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(path.vel(2.5), [-0.3, 0.0, 0.0])


@pytest.mark.parametrize("t_from", [1.0, 0.5, 3.5])
def test_retract_path_needs_a_hit_time_inside_the_path(t_from):
    with pytest.raises(ValueError):
        RetractPath(reach_path([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 3.0, "linear"), t_from)


def test_bounding_radius():
    assert bounding_radius_m("sphere", [0.03]) == 0.03
    assert bounding_radius_m("sphere", [0.03, 0.03, 0.03]) == 0.03
    assert bounding_radius_m("box", [0.03, 0.04, 0.0]) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        bounding_radius_m("capsule", [0.03, 0.04])


@pytest.mark.parametrize("p0,v", [([0.0, 0.0, 0.5], [0.5, 0.0, 0.0]), ([0.0, 0.0, 0.5], [-0.3, 0.4, 0.0]),
                                  ([0.2, 0.9, 0.5], [0.0, 0.0, 1.0]), ([0.0, 0.0, 0.5], [0.0, 0.0, -2.0]),
                                  ([2.0, 0.0, 0.5], [0.5, 0.0, 0.0])])
def test_straight_exit_ends_on_the_first_grid_sample_strictly_outside(p0, v):
    dt = 0.01
    n = straight_exit_steps(p0, v, LO, HI, dt)
    p0, v = np.array(p0), np.array(v)
    end = p0 + v * (n * dt)
    assert np.any(end < LO) or np.any(end > HI)
    if n > 1:
        before = p0 + v * ((n - 1) * dt)
        assert not (np.any(before < LO) or np.any(before > HI))


def test_straight_exit_on_the_boundary_is_not_outside():
    # the box edge at x = 1 is reached exactly at n = 100 steps of 0.01 s at 1 m/s: on the edge is not outside
    assert straight_exit_steps([0.0, 0.0, 0.5], [1.0, 0.0, 0.0], LO, HI, 0.01) >= 101


def test_straight_exit_needs_a_moving_hazard():
    with pytest.raises(ValueError, match="zero release velocity"):
        straight_exit_steps([0.0, 0.0, 0.5], [0.0, 0.0, 0.0], LO, HI, 0.01)
