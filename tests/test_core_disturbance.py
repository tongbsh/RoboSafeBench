"""Pure-logic tests for ``dynisafety.core.disturbance`` (no simulator)."""

from __future__ import annotations

import numpy as np
import pytest

from dynisafety.config.schema import MISSING
from dynisafety.core.contact import quat_to_matrix
from dynisafety.core.disturbance import (
    MIN_JERK_PEAK_COEFF,
    Phase,
    arc_lift,
    arm_quat_from_yaw,
    compute_approach_steps,
    interpolate_toward,
    lerp,
    minimum_jerk,
    perpendicular_2d,
    quat_about_axis,
    quat_mul,
    require_configured,
    require_map_key,
)


def test_minimum_jerk_endpoints_and_midpoint() -> None:
    assert minimum_jerk(0.0) == pytest.approx(0.0)
    assert minimum_jerk(1.0) == pytest.approx(1.0)
    assert minimum_jerk(0.5) == pytest.approx(0.5)


def test_minimum_jerk_clips_outside_unit_interval() -> None:
    assert minimum_jerk(-0.25) == pytest.approx(0.0)
    assert minimum_jerk(1.5) == pytest.approx(1.0)


def test_minimum_jerk_is_monotonic() -> None:
    taus = np.linspace(0.0, 1.0, 21)
    vals = [minimum_jerk(float(t)) for t in taus]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def test_compute_approach_steps_nominal() -> None:
    # duration = 1.875 * 0.4 / 0.5 = 1.5 → ceil(1.5 * 20) = 30
    assert compute_approach_steps(0.4, 0.5, control_freq=20) == 30
    assert MIN_JERK_PEAK_COEFF == pytest.approx(15.0 / 8.0)


def test_compute_approach_steps_floor_is_two() -> None:
    assert compute_approach_steps(1e-6, 10.0, control_freq=20) == 2


def test_compute_approach_steps_rejects_non_positive_freq() -> None:
    with pytest.raises(ValueError, match="control_freq"):
        compute_approach_steps(0.4, 0.5, control_freq=0)
    with pytest.raises(ValueError, match="control_freq"):
        compute_approach_steps(0.4, 0.5, control_freq=-10)


def test_compute_approach_steps_tiny_peak_uses_floor() -> None:
    expected = compute_approach_steps(0.4, 0.01, control_freq=20)
    assert compute_approach_steps(0.4, 1e-9, control_freq=20) == expected
    assert expected > 2


def test_arc_lift_endpoints_and_midpoint() -> None:
    assert arc_lift(0.0, 0.12) == pytest.approx(0.0)
    assert arc_lift(1.0, 0.12) == pytest.approx(0.0)
    assert arc_lift(0.5, 0.12) == pytest.approx(0.12)


def test_arc_lift_clips_alpha() -> None:
    assert arc_lift(-0.3, 0.2) == pytest.approx(0.0)
    assert arc_lift(1.4, 0.2) == pytest.approx(0.0)


def test_lerp_endpoints_midpoint_2d_3d_and_clip() -> None:
    start2 = np.array([0.0, 0.0])
    end2 = np.array([2.0, 4.0])
    assert lerp(start2, end2, 0.0) == pytest.approx(start2)
    assert lerp(start2, end2, 1.0) == pytest.approx(end2)
    assert lerp(start2, end2, 0.5) == pytest.approx(np.array([1.0, 2.0]))
    assert lerp(start2, end2, -1.0) == pytest.approx(start2)
    assert lerp(start2, end2, 2.0) == pytest.approx(end2)

    start3 = np.array([0.0, 1.0, 2.0])
    end3 = np.array([3.0, 4.0, 8.0])
    assert lerp(start3, end3, 0.5) == pytest.approx(np.array([1.5, 2.5, 5.0]))


def test_perpendicular_2d_unit_orthogonal_and_flip() -> None:
    n = perpendicular_2d(np.array([1.0, 0.0]))
    assert float(np.dot(n, [1.0, 0.0])) == pytest.approx(0.0)
    assert float(np.linalg.norm(n)) == pytest.approx(1.0)
    assert n == pytest.approx(np.array([0.0, 1.0]))
    assert perpendicular_2d(np.array([1.0, 0.0]), flip=True) == pytest.approx(-n)
    scaled = perpendicular_2d(np.array([3.0, 4.0]))
    assert float(np.dot(scaled, [3.0, 4.0])) == pytest.approx(0.0)
    assert float(np.linalg.norm(scaled)) == pytest.approx(1.0)


def test_perpendicular_2d_zero_raises() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        perpendicular_2d(np.array([0.0, 0.0]))


def test_perpendicular_2d_preserves_input_dimension() -> None:
    """3-vector callers (SimplerEnv / RoboTwin) get ``(-uy, ux, 0)`` back."""
    n3 = perpendicular_2d(np.array([1.0, 0.0, 0.7]))
    assert n3.shape == (3,)
    assert n3 == pytest.approx(np.array([0.0, 1.0, 0.0]))
    assert perpendicular_2d(np.array([3.0, 4.0, -2.0])) == pytest.approx(
        np.append(perpendicular_2d(np.array([3.0, 4.0])), 0.0)
    )
    assert perpendicular_2d(np.array([1.0, 0.0, 0.0]), flip=True) == pytest.approx(
        np.array([0.0, -1.0, 0.0])
    )


def test_perpendicular_2d_zero_xy_projection_raises() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        perpendicular_2d(np.array([0.0, 0.0, 5.0]))


def test_perpendicular_2d_rejects_other_dimensions() -> None:
    with pytest.raises(ValueError, match="2 or 3 elements"):
        perpendicular_2d(np.array([1.0, 2.0, 3.0, 4.0]))


def test_interpolate_toward_endpoints_and_midpoint() -> None:
    origin = np.array([0.0, 0.0, 1.0])
    target = np.array([2.0, 4.0, 5.0])
    assert interpolate_toward(origin, target, 0.0) == pytest.approx(origin)
    assert interpolate_toward(origin, target, 1.0) == pytest.approx(target)
    assert interpolate_toward(origin, target, 0.5) == pytest.approx(np.array([1.0, 2.0, 3.0]))


def test_quat_mul_identity_and_associativity() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.5, 0.5, 0.5, 0.5])
    q = q / np.linalg.norm(q)
    assert quat_mul(identity, q) == pytest.approx(q)
    assert quat_mul(q, identity) == pytest.approx(q)

    q1 = quat_about_axis("x", 30.0)
    q2 = quat_about_axis("y", 45.0)
    q3 = quat_about_axis("z", 60.0)
    left = quat_mul(quat_mul(q1, q2), q3)
    right = quat_mul(q1, quat_mul(q2, q3))
    assert left == pytest.approx(right, abs=1e-12)


def test_quat_mul_matches_matrix_product() -> None:
    q1 = quat_about_axis("z", 25.0)
    q2 = quat_about_axis("x", -40.0)
    r_mul = quat_to_matrix(quat_mul(q1, q2))
    r_mat = quat_to_matrix(q1) @ quat_to_matrix(q2)
    assert r_mul == pytest.approx(r_mat, abs=1e-12)


def test_quat_about_axis_z90_sends_x_to_y() -> None:
    rot = quat_to_matrix(quat_about_axis("z", 90.0))
    x_axis = np.array([1.0, 0.0, 0.0])
    assert rot @ x_axis == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-12)


def test_quat_about_axis_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown axis"):
        quat_about_axis("w", 10.0)


def test_arm_quat_from_yaw_unit_and_yaw_sensitive() -> None:
    q0 = arm_quat_from_yaw(0.0)
    q1 = arm_quat_from_yaw(0.4)
    assert float(np.linalg.norm(q0)) == pytest.approx(1.0)
    assert float(np.linalg.norm(q1)) == pytest.approx(1.0)
    assert not np.allclose(q0, q1)


def test_phase_union_and_str_enum() -> None:
    names = {member.name for member in Phase}
    assert names == {
        "HIDDEN",
        "ENTER",
        "APPROACH",
        "DWELL",
        "WITHDRAW",
        "POST_HIT_HOLD",
        "FLINCH",
        "FLINCH_EXIT",
        "EXIT",
        "WAIT",
        "DONE",
    }
    assert Phase.HIDDEN == "hidden"
    assert Phase.POST_HIT_HOLD == "post_hit_hold"
    assert isinstance(Phase.HIDDEN, str)


def test_require_configured_missing_and_none_raise() -> None:
    with pytest.raises(ValueError, match="iron law 1"):
        require_configured(MISSING, "hand.mass_kg")
    with pytest.raises(ValueError, match="iron law 1"):
        require_configured(None, "hand.mass_kg")


def test_require_configured_returns_value_including_falsey() -> None:
    assert require_configured(0, "n") == 0
    assert require_configured("", "s") == ""
    assert require_configured(False, "flag") is False
    assert require_configured(1.5, "v") == 1.5


def test_require_map_key_present_and_absent() -> None:
    block = {"alpha": 0.4}
    assert require_map_key(block, "alpha", where="extra.robocasa.l2") == 0.4
    with pytest.raises(ValueError, match="extra.robocasa.l2.beta"):
        require_map_key(block, "beta", where="extra.robocasa.l2")
