import numpy as np
import pytest

from hazardlib.kinematics import (BallisticPath, LinePath, arc_path, fall_height_for_tau,
                                  fall_time_for_height, free_fall_path, min_jerk,
                                  min_jerk_derivative, min_jerk_inverse, profile_inverse,
                                  reach_path, straight_path)


def central_difference(path, t, h):
    return (path.pos(t + h) - path.pos(t - h)) / (2.0 * h)


# ----------------------------------------------------------------------- min-jerk

def test_min_jerk_end_points_and_midpoint():
    assert min_jerk(0.0) == 0.0
    assert min_jerk(1.0) == 1.0
    assert min_jerk(0.5) == 0.5
    assert min_jerk_derivative(0.0) == 0.0
    assert min_jerk_derivative(1.0) == 0.0
    assert min_jerk_derivative(0.5) == pytest.approx(1.875, abs=1e-15)   # 30 * (1/4)^2


def test_min_jerk_matches_the_closed_form_and_is_symmetric_and_monotone():
    u = np.linspace(0.0, 1.0, 1001)
    expected = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    # Two algebraically equal forms of a polynomial whose terms (10, 15, 6) cancel down to <= 1 differ by
    # rounding of order (10 + 15 + 6) * eps ~ 7e-15; 1e-14 is that bound, not a tuned value (a first
    # draft used 1e-15 and failed with a max difference of 3.2e-15, see report T02).
    np.testing.assert_allclose(min_jerk(u), expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(min_jerk(u) + min_jerk(1.0 - u), 1.0, atol=1e-14)
    assert np.all(np.diff(min_jerk(u)) > 0)
    assert np.all(min_jerk_derivative(u) >= 0)


def test_min_jerk_derivative_is_the_derivative():
    rng = np.random.RandomState(0)
    u = rng.uniform(0.01, 0.99, 50)
    h = 1e-6
    numeric = (min_jerk(u + h) - min_jerk(u - h)) / (2 * h)
    np.testing.assert_allclose(min_jerk_derivative(u), numeric, atol=1e-8)


def test_min_jerk_has_zero_acceleration_at_both_ends():
    h = 1e-5
    accel0 = (min_jerk_derivative(h) - min_jerk_derivative(0.0)) / h
    accel1 = (min_jerk_derivative(1.0) - min_jerk_derivative(1.0 - h)) / h
    assert abs(accel0) < 1e-3 and abs(accel1) < 1e-3   # ~ 90 h^2 = 1e-8 analytically


def test_min_jerk_is_clamped_outside_the_unit_interval():
    for u in (-3.0, -1e-9):
        assert min_jerk(u) == 0.0 and min_jerk_derivative(u) == 0.0
    for u in (1.0 + 1e-9, 7.0):
        assert min_jerk(u) == 1.0 and min_jerk_derivative(u) == 0.0


def test_min_jerk_scalar_in_float_out_array_in_array_out():
    assert isinstance(min_jerk(0.3), float) and isinstance(min_jerk_derivative(0.3), float)
    arr = min_jerk(np.array([[0.0, 0.5], [1.0, 2.0]]))
    assert arr.shape == (2, 2)
    np.testing.assert_array_equal(arr, [[0.0, 0.5], [1.0, 1.0]])
    assert min_jerk(np.array([0.2, 0.4]))[1] == min_jerk(0.4)   # vectorised == scalar


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", None])
def test_min_jerk_rejects_non_finite_or_non_numeric(bad):
    with pytest.raises((ValueError, TypeError)):
        min_jerk(bad)


def test_min_jerk_inverse_round_trips_to_float_resolution():
    rng = np.random.RandomState(1)
    for s in list(rng.uniform(0.0, 1.0, 200)) + [1e-12, 1e-6, 0.5, 1 - 1e-9]:
        u = min_jerk_inverse(s)
        assert 0.0 <= u <= 1.0
        assert abs(min_jerk(u) - s) <= 4e-16 + 1e-12 * s
    assert min_jerk_inverse(0.0) == 0.0 and min_jerk_inverse(1.0) == 1.0
    assert min_jerk_inverse(0.5) == pytest.approx(0.5, abs=1e-15)
    values = [min_jerk_inverse(s) for s in np.linspace(0, 1, 21)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_min_jerk_inverse_and_profile_inverse_validation():
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            min_jerk_inverse(bad)
    assert profile_inverse("linear", 0.3) == 0.3
    assert profile_inverse("min_jerk", 0.5) == pytest.approx(0.5, abs=1e-15)
    with pytest.raises(ValueError, match="speed_profile"):
        profile_inverse("cubic", 0.3)
    with pytest.raises(ValueError):
        profile_inverse("linear", 1.5)


# --------------------------------------------------------------------- free fall

def test_fall_round_trip_has_relative_error_below_1e_9():
    rng = np.random.RandomState(2)
    for _ in range(500):
        tau = rng.uniform(1e-3, 5.0)
        g = rng.uniform(0.5, 25.0)
        back = fall_time_for_height(fall_height_for_tau(tau, g), g)
        assert abs(back - tau) / tau < 1e-9


def test_fall_known_values():
    assert fall_height_for_tau(1.0, 9.81) == pytest.approx(4.905, rel=1e-15)
    assert fall_time_for_height(4.905, 9.81) == pytest.approx(1.0, rel=1e-15)
    assert fall_height_for_tau(0.0, 9.81) == 0.0
    assert fall_time_for_height(0.0, 9.81) == 0.0
    assert fall_height_for_tau(2.0, 1.62) == pytest.approx(0.5 * 1.62 * 4.0, rel=1e-15)


def test_fall_time_agrees_with_numerical_integration_of_gravity():
    g, height, dt = 9.81, 0.8, 1e-5
    z, v, t = height, 0.0, 0.0
    while z > 0.0:                      # velocity Verlet for z'' = -g
        v_half = v - 0.5 * g * dt
        z_next = z + v_half * dt
        v = v_half - 0.5 * g * dt
        if z_next <= 0.0:               # linear interpolation of the crossing inside the step
            t += dt * z / (z - z_next)
            break
        z, t = z_next, t + dt
    assert abs(t - fall_time_for_height(height, g)) < 1e-9


def test_fall_functions_accept_arrays_and_return_floats_for_scalars():
    taus = np.array([0.1, 0.2, 0.4])
    heights = fall_height_for_tau(taus, 9.81)
    assert heights.shape == (3,)
    np.testing.assert_allclose(fall_time_for_height(heights, 9.81), taus, rtol=1e-12)
    assert isinstance(fall_height_for_tau(0.3, 9.81), float)
    assert isinstance(fall_time_for_height(0.3, 9.81), float)


@pytest.mark.parametrize("call", [
    lambda: fall_height_for_tau(-0.1, 9.81),
    lambda: fall_height_for_tau(0.1, 0.0),
    lambda: fall_height_for_tau(0.1, -9.81),
    lambda: fall_height_for_tau(float("nan"), 9.81),
    lambda: fall_time_for_height(-0.1, 9.81),
    lambda: fall_time_for_height(0.1, 0.0),
    lambda: fall_time_for_height(0.1, float("inf")),
    lambda: fall_time_for_height(np.array([0.1, -0.2]), 9.81),
])
def test_fall_functions_reject_invalid_arguments(call):
    with pytest.raises(ValueError):
        call()


def test_free_fall_path_is_gravity_from_rest():
    path = free_fall_path([0.1, 0.2, 1.0], 2.0, 3.0, 9.81)
    assert isinstance(path, BallisticPath)
    np.testing.assert_allclose(path.pos(2.0), [0.1, 0.2, 1.0])
    np.testing.assert_allclose(path.pos(2.5), [0.1, 0.2, 1.0 - 0.5 * 9.81 * 0.25])
    np.testing.assert_allclose(path.vel(2.5), [0, 0, -9.81 * 0.5])
    np.testing.assert_array_equal(path.vel(2.0), [0, 0, 0])
    with pytest.raises(ValueError):
        free_fall_path([0, 0, 1], 2.0, 3.0, 0.0)


# ------------------------------------------------------------------------- paths

def test_straight_path_is_constant_speed_between_its_end_points():
    path = straight_path([0, 0, 0], [3, 0, 4], 1.0, 3.0)
    assert isinstance(path, LinePath)
    np.testing.assert_allclose(path.pos(1.0), [0, 0, 0])
    np.testing.assert_allclose(path.pos(2.0), [1.5, 0, 2.0])
    np.testing.assert_allclose(path.pos(3.0), [3, 0, 4])
    for t in (1.0, 1.7, 3.0):
        np.testing.assert_allclose(path.vel(t), [1.5, 0, 2.0])      # |v| = 5 / 2 = 2.5
    np.testing.assert_array_equal(path.vel(0.99), [0, 0, 0])
    np.testing.assert_array_equal(path.vel(3.01), [0, 0, 0])
    np.testing.assert_allclose(path.pos(0.0), [0, 0, 0])             # clamped
    np.testing.assert_allclose(path.pos(9.0), [3, 0, 4])


def test_arc_path_lifts_by_a_parabola_of_the_given_height():
    path = arc_path([0, 0, 0.2], [1, 0, 0.2], 0.0, 2.0, 0.3)
    np.testing.assert_allclose(path.pos(0.0), [0, 0, 0.2])
    np.testing.assert_allclose(path.pos(2.0), [1, 0, 0.2], atol=1e-15)
    np.testing.assert_allclose(path.pos(1.0), [0.5, 0, 0.5])           # peak = base + arc_height
    ts = np.linspace(0, 2, 41)
    z = path.pos(ts)[:, 2]
    assert np.max(z) == pytest.approx(0.5, abs=1e-15)
    np.testing.assert_allclose(z, 0.2 + 4 * 0.3 * (ts / 2) * (1 - ts / 2), atol=1e-15)
    np.testing.assert_allclose(path.vel(1.0), [0.5, 0, 0], atol=1e-15)   # flat at the apex
    assert path.vel(0.0)[2] > 0 > path.vel(2.0)[2]
    flat = arc_path([0, 0, 0], [1, 0, 0], 0.0, 1.0, 0.0)
    np.testing.assert_allclose(flat.pos(0.5), [0.5, 0, 0])
    with pytest.raises(ValueError):
        arc_path([0, 0, 0], [1, 0, 0], 0.0, 1.0, -0.1)


def test_reach_path_min_jerk_rests_at_both_ends_and_is_half_way_at_mid_time():
    path = reach_path([0, 0, 0], [0, 0.8, 0], 1.0, 2.0, "min_jerk")
    np.testing.assert_array_equal(path.vel(1.0), [0, 0, 0])
    np.testing.assert_array_equal(path.vel(2.0), [0, 0, 0])
    np.testing.assert_allclose(path.pos(1.5), [0, 0.4, 0], atol=1e-15)
    np.testing.assert_allclose(path.pos(2.0), [0, 0.8, 0])
    np.testing.assert_allclose(path.vel(1.5), [0, 0.8 * 1.875, 0], atol=1e-14)
    ts = np.linspace(1.0, 2.0, 201)
    assert np.all(np.diff(path.pos(ts)[:, 1]) > 0)


def test_reach_path_linear_profile_and_invalid_profile():
    path = reach_path([0, 0, 0], [0, 1, 0], 0.0, 2.0, "linear")
    np.testing.assert_allclose(path.vel(0.0), [0, 0.5, 0])
    with pytest.raises(ValueError, match="speed_profile"):
        reach_path([0, 0, 0], [0, 1, 0], 0.0, 2.0, "smoothstep")


@pytest.mark.parametrize("make", [
    lambda: straight_path([0, 0, 0], [1, 2, 3], 0.5, 2.5),
    lambda: arc_path([0, 0, 0], [1, 2, 3], 0.5, 2.5, 0.4),
    lambda: reach_path([0, 0, 0], [1, 2, 3], 0.5, 2.5, "min_jerk"),
    lambda: free_fall_path([0, 0, 2], 0.5, 2.5, 9.81),
    lambda: BallisticPath([0, 0, 0], [1, 0.5, 2], [0, 0, -3], 0.5, 2.5),
])
def test_velocity_is_the_derivative_of_position_for_every_path_type(make):
    path = make()
    ts = np.linspace(0.6, 2.4, 30)
    h = 1e-5
    numeric = np.array([central_difference(path, t, h) for t in ts])
    np.testing.assert_allclose(path.vel(ts), numeric, atol=1e-7)
    assert path.pos(ts).shape == (30, 3)


def test_path_inputs_are_validated_and_copied():
    start = np.array([0.0, 0.0, 0.0])
    path = straight_path(start, [1, 0, 0], 0.0, 1.0)
    start[:] = 50.0
    np.testing.assert_allclose(path.pos(0.0), [0, 0, 0])
    for args in (([0, 0], [1, 0, 0], 0.0, 1.0), ([0, 0, 0], [1, 0, 0], 1.0, 1.0),
                 ([0, 0, 0], [1, 0, float("nan")], 0.0, 1.0), ([0, 0, 0], [1, 0, 0], 2.0, 1.0)):
        with pytest.raises(ValueError):
            straight_path(*args)
