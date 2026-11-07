import numpy as np
import pytest

from hazardlib.semantics import is_exposure_low, min_distance_to_hit_point

HIT = [0.5, 0.0, 0.3]
T = np.linspace(0.0, 2.0, 21)


def trajectory(offset):
    """An EEF moving along +x at 0.5 m/s, passing the hit point's x at t = 1, at a fixed lateral offset."""
    pos = np.zeros((T.size, 3))
    pos[:, 0] = 0.0 + 0.5 * T
    pos[:, 1] = offset
    pos[:, 2] = 0.3
    return pos


def test_a_robot_that_passes_the_hit_point_is_exposed():
    assert not is_exposure_low(trajectory(0.0), T, HIT, 0.0, 2.0, 0.05)
    assert min_distance_to_hit_point(trajectory(0.0), T, HIT, 0.0, 2.0) == pytest.approx(0.0, abs=1e-12)


def test_a_robot_that_stays_away_is_low_exposure():
    assert is_exposure_low(trajectory(0.8), T, HIT, 0.0, 2.0, 0.3)
    assert min_distance_to_hit_point(trajectory(0.8), T, HIT, 0.0, 2.0) == pytest.approx(0.8, abs=1e-12)


def test_the_radius_comparison_is_strict():
    pos = trajectory(0.25)                               # closest sample is exactly 0.25 m away
    distance = min_distance_to_hit_point(pos, T, HIT, 0.0, 2.0)
    assert distance == pytest.approx(0.25)
    assert not is_exposure_low(pos, T, HIT, 0.0, 2.0, distance)          # distance == radius: not low
    assert is_exposure_low(pos, T, HIT, 0.0, 2.0, distance * 0.999)


def test_only_samples_inside_the_window_count_and_the_window_is_inclusive():
    pos = trajectory(0.0)                                # at the hit point around t = 1
    assert is_exposure_low(pos, T, HIT, 1.5, 2.0, 0.1)    # the close approach is before the window
    assert not is_exposure_low(pos, T, HIT, 0.0, 1.0, 0.1)
    # window ends exactly on the sample at t = 1.0 (inclusive)
    assert min_distance_to_hit_point(pos, T, HIT, 1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert min_distance_to_hit_point(pos, T, HIT, 0.0, 0.5) == pytest.approx(0.5 - 0.5 * 0.5, abs=1e-12)


def test_a_window_without_samples_is_an_error():
    with pytest.raises(ValueError, match="no end-effector sample inside the window"):
        is_exposure_low(trajectory(0.0), T, HIT, 5.0, 6.0, 0.1)
    with pytest.raises(ValueError, match="no end-effector sample"):
        is_exposure_low(trajectory(0.0), T, HIT, 0.051, 0.099, 0.1)     # between two samples


@pytest.mark.parametrize("kwargs,match", [
    (dict(t_start_s=1.0, t_end_s=0.5), "before t_start_s"),
    (dict(radius_m=-0.1), "radius_m"),
    (dict(radius_m=float("nan")), "radius_m"),
    (dict(hit_point=[0, 0]), "hit_point"),
    (dict(times_s=T[:-1]), "eef_positions must have shape"),
    (dict(eef_positions=np.zeros((T.size, 2))), "eef_positions must have shape"),
    (dict(times_s=T[::-1].copy()), "strictly increasing"),
    (dict(t_start_s=float("nan")), "t_start_s"),
])
def test_invalid_arguments_are_rejected(kwargs, match):
    args = dict(eef_positions=trajectory(0.0), times_s=T, hit_point=HIT, t_start_s=0.0, t_end_s=2.0, radius_m=0.1)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        is_exposure_low(**args)
