import numpy as np
import pytest

from hazardlib.host import ContactObs
from hazardlib.semantics import analytic_impulse_ns, closing_normal_speed


def table_obs():
    """The partner is a stationary table; it pushes the hazard up, so the normal is +z (partner -> hazard)."""
    return ContactObs(partner_name="table", partner_class="scene", position=[0, 0, 0],
                      normal=[0, 0, 1], time_s=0.3)


def test_hazard_falling_onto_a_stationary_table_has_normal_speed_equal_to_its_speed():
    obs = table_obs()
    speed = 1.7
    assert closing_normal_speed([0, 0, -speed], [0, 0, 0], obs.normal) == pytest.approx(speed, rel=1e-15)


def test_a_hazard_moving_away_has_zero_normal_speed():
    obs = table_obs()
    assert closing_normal_speed([0, 0, +1.7], [0, 0, 0], obs.normal) == 0.0
    assert closing_normal_speed([0, 0, 0], [0, 0, 0], obs.normal) == 0.0       # at rest: nothing closes


def test_the_normal_points_from_the_partner_to_the_hazard():
    n = [0.0, 0.0, 1.0]
    # partner moving up into a resting hazard closes; partner moving down away from it does not
    assert closing_normal_speed([0, 0, 0], [0, 0, 2.0], n) == pytest.approx(2.0)
    assert closing_normal_speed([0, 0, 0], [0, 0, -2.0], n) == 0.0
    # flipping the normal flips the answer: the convention matters
    assert closing_normal_speed([0, 0, -1.0], [0, 0, 0], [0, 0, -1.0]) == 0.0


def test_only_the_normal_component_counts_and_the_normal_is_normalised():
    # 45 degree approach: v_n = |v| cos 45
    assert closing_normal_speed([1.0, 0, -1.0], [0, 0, 0], [0, 0, 1]) == pytest.approx(1.0)
    assert closing_normal_speed([1.0, 0, 0], [0, 0, 0], [0, 0, 1]) == 0.0            # purely tangential
    assert closing_normal_speed([0, 0, -2.0], [0, 0, 0], [0, 0, 7.5]) == pytest.approx(2.0)   # non-unit normal
    # relative velocity: both moving
    assert closing_normal_speed([0, 0, -1.0], [0, 0, 0.5], [0, 0, 1]) == pytest.approx(1.5)


def test_a_zero_normal_is_an_error_not_a_fallback():
    with pytest.raises(ValueError, match="non-zero"):
        closing_normal_speed([0, 0, -1.0], [0, 0, 0], [0, 0, 0])


@pytest.mark.parametrize("args", [
    ([0, 0], [0, 0, 0], [0, 0, 1]), ([0, 0, 0], [0, 0, float("nan")], [0, 0, 1]),
    ([0, 0, 0], [0, 0, 0], [0, 0, float("inf")]), ([0, 0, 0], [0, 0, 0], [0, 1]),
])
def test_closing_speed_rejects_malformed_vectors(args):
    with pytest.raises(ValueError):
        closing_normal_speed(*args)


def test_analytic_impulse_is_m_one_plus_e_vn():
    assert analytic_impulse_ns(0.5, 2.0, 0.0) == pytest.approx(1.0)
    assert analytic_impulse_ns(0.5, 2.0, 1.0) == pytest.approx(2.0)          # elastic: twice the momentum
    assert analytic_impulse_ns(0.5, 2.0, 0.25) == pytest.approx(0.5 * 1.25 * 2.0)
    assert analytic_impulse_ns(0.5, 0.0, 0.3) == 0.0


def test_restitution_is_required_and_must_not_be_negative():
    with pytest.raises(TypeError):
        analytic_impulse_ns(0.5, 2.0)                                           # no default for e
    with pytest.raises(ValueError, match="restitution"):
        analytic_impulse_ns(0.5, 2.0, -0.01)
    for bad in (float("nan"), float("inf"), "0.0", None):
        with pytest.raises(ValueError):
            analytic_impulse_ns(0.5, 2.0, bad)


@pytest.mark.parametrize("args", [(0.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (float("nan"), 1.0, 0.0),
                                  (1.0, -0.1, 0.0), (1.0, float("inf"), 0.0), (True, 1.0, 0.0)])
def test_impulse_rejects_invalid_mass_or_speed(args):
    with pytest.raises(ValueError):
        analytic_impulse_ns(*args)


def test_the_impulse_follows_the_closing_speed_end_to_end():
    obs = table_obs()
    vn = closing_normal_speed([0.2, 0.0, -1.2], [0, 0, 0], obs.normal)
    assert analytic_impulse_ns(0.3, vn, 0.0) == pytest.approx(0.3 * 1.2)
