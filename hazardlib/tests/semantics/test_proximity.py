import numpy as np
import pytest

from hazardlib.semantics import SeparationKind, compute_proximity


def approach(speed, start, n=11, dt=0.1):
    """An EEF closing on a hazard at the origin along +x at constant speed."""
    t = dt * np.arange(n)
    eef = np.zeros((n, 3))
    eef[:, 0] = start - speed * t
    return t, np.zeros((n, 3)), eef


def test_constant_closing_speed_matches_the_analytic_values():
    speed, start = 0.5, 1.0
    t, hazard, eef = approach(speed, start)
    result = compute_proximity(t, hazard, eef, None)
    assert result.min_separation_m == pytest.approx(start - speed * t[-1], abs=1e-12)     # 0.5 at the last sample
    assert result.t_min_separation_s == pytest.approx(t[-1])
    assert result.ttc_series_s[0] is None
    for i in range(1, len(t)):
        assert result.ttc_series_s[i] == pytest.approx(start / speed - t[i], abs=1e-12)    # d / v, exactly
    assert result.ttc_min_s == pytest.approx(start / speed - t[-1], abs=1e-12)
    np.testing.assert_allclose(result.separation_m, start - speed * t, atol=1e-12)


def test_surrogate_when_no_geometric_distance_is_given():
    t, hazard, eef = approach(0.5, 1.0)
    result = compute_proximity(t, hazard, eef, None)
    assert result.separation_kind is SeparationKind.SURROGATE and result.separation_kind == "surrogate"


def test_a_geometric_distance_is_preferred_and_labelled():
    t, hazard, eef = approach(0.5, 1.0)
    geometric = (1.0 - 0.5 * t) - 0.08            # surface distance: the origins minus the radii
    result = compute_proximity(t, hazard, eef, geometric)
    assert result.separation_kind is SeparationKind.GEOMETRIC
    assert result.min_separation_m == pytest.approx(geometric[-1], abs=1e-12)
    np.testing.assert_allclose(result.separation_m, geometric)
    assert result.ttc_min_s == pytest.approx(geometric[-1] / 0.5, abs=1e-12)


def test_the_minimum_need_not_be_at_the_end():
    t = 0.1 * np.arange(7)
    eef = np.zeros((7, 3))
    eef[:, 0] = [1.0, 0.8, 0.6, 0.4, 0.5, 0.7, 0.9]          # approaches, then retreats
    result = compute_proximity(t, np.zeros((7, 3)), eef, None)
    assert result.min_separation_m == pytest.approx(0.4) and result.t_min_separation_s == pytest.approx(0.3)
    assert result.ttc_series_s[4] is None and result.ttc_series_s[5] is None      # separating: undefined
    assert result.ttc_series_s[3] == pytest.approx(0.4 / 2.0)                      # last closing sample
    assert result.ttc_min_s == pytest.approx(0.2)


def test_never_closing_gives_no_ttc_and_touching_gives_zero():
    t = 0.1 * np.arange(4)
    eef = np.zeros((4, 3))
    eef[:, 0] = [0.5, 0.6, 0.7, 0.7]                        # separating, then constant
    result = compute_proximity(t, np.zeros((4, 3)), eef, None)
    assert result.ttc_series_s == [None, None, None, None] and result.ttc_min_s is None
    eef[:, 0] = [0.3, 0.2, 0.1, 0.0]
    touching = compute_proximity(t, np.zeros((4, 3)), eef, None)
    assert touching.min_separation_m == 0.0 and touching.ttc_series_s[-1] == 0.0 and touching.ttc_min_s == 0.0


def test_hazard_motion_counts_not_only_the_end_effector():
    t = 0.1 * np.arange(5)
    hazard = np.zeros((5, 3)); hazard[:, 0] = 1.0 - 1.0 * t        # the hazard flies toward a fixed EEF at 1 m/s
    eef = np.zeros((5, 3))
    result = compute_proximity(t, hazard, eef, None)
    assert result.ttc_series_s[1] == pytest.approx(0.9 / 1.0)
    assert result.ttc_min_s == pytest.approx(0.6)


def test_single_sample_trace():
    result = compute_proximity([0.0], [[0, 0, 0]], [[0.3, 0.4, 0.0]], None)
    assert result.min_separation_m == pytest.approx(0.5) and result.ttc_series_s == [None] and result.ttc_min_s is None


@pytest.mark.parametrize("kwargs,match", [
    (dict(times_s=[]), "non-empty"),
    (dict(times_s=[0.0, 0.0, 0.1]), "strictly increasing"),
    (dict(times_s=[0.0, 0.2, 0.1]), "strictly increasing"),
    (dict(hazard_pos=np.zeros((2, 3))), "hazard_pos"),
    (dict(eef_pos=np.zeros((3, 2))), "eef_pos"),
    (dict(geometric_distance_m=[0.1, 0.2]), "geometric_distance_m"),
    (dict(geometric_distance_m=[0.1, -0.2, 0.3]), ">= 0"),
    (dict(geometric_distance_m=[0.1, float("nan"), 0.3]), "geometric_distance_m"),
    (dict(hazard_pos=np.full((3, 3), np.nan)), "hazard_pos"),
])
def test_invalid_traces_are_rejected(kwargs, match):
    args = dict(times_s=[0.0, 0.1, 0.2], hazard_pos=np.zeros((3, 3)), eef_pos=np.ones((3, 3)),
                geometric_distance_m=None)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        compute_proximity(**args)
