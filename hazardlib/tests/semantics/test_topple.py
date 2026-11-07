import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardlib.semantics import is_toppled, tilt_deg_up_axis

CONTACT_PY = Path(__file__).resolve().parents[3] / "src" / "dynisafety" / "core" / "contact.py"
IDENTITY = [1.0, 0.0, 0.0, 0.0]


def axis_angle(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = np.radians(degrees) / 2.0
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def legacy_relative_quaternion_angle(ref, cur):
    """The legacy LIBERO / RoboCasa definition (core/contact.py::tilt_deg_from_body_quat with ref_quat):
    the total angle of the relative rotation, yaw included. Reimplemented here as the comparison."""
    ref_inv = np.array([ref[0], -ref[1], -ref[2], -ref[3]])
    rel_w = abs(qmul(ref_inv, cur)[0])
    return float(np.degrees(2.0 * np.arccos(min(1.0, rel_w))))


@pytest.mark.parametrize("axis,degrees,expected", [
    ([0, 0, 1], 40.0, 0.0),        # P0 appendix: pure yaw 40 degrees
    ([0, 0, 1], 90.0, 0.0),        # pure yaw 90 degrees
    ([1, 0, 0], 40.0, 40.0),       # real tilt of 40 degrees about x
    ([1, 0, 0], 20.0, 20.0),       # tilt of 20 degrees about x
])
def test_reproduces_the_p0_appendix_table(axis, degrees, expected):
    assert tilt_deg_up_axis(IDENTITY, axis_angle(axis, degrees)) == pytest.approx(expected, abs=1e-9)


def test_pure_yaw_is_not_tilt_but_the_legacy_total_angle_calls_it_one():
    for degrees in (40.0, 90.0):
        cur = axis_angle([0, 0, 1], degrees)
        assert tilt_deg_up_axis(IDENTITY, cur) == pytest.approx(0.0, abs=1e-9)
        assert legacy_relative_quaternion_angle(IDENTITY, cur) == pytest.approx(degrees, abs=1e-9)
        assert is_toppled(legacy_relative_quaternion_angle(IDENTITY, cur), 30.0)        # legacy: toppled
        assert not is_toppled(tilt_deg_up_axis(IDENTITY, cur), 30.0)                    # new: not toppled


def test_the_legacy_and_new_definitions_agree_when_there_is_no_yaw():
    for degrees in (10.0, 20.0, 40.0, 100.0):
        cur = axis_angle([1, 0, 0], degrees)
        assert tilt_deg_up_axis(IDENTITY, cur) == pytest.approx(
            legacy_relative_quaternion_angle(IDENTITY, cur), abs=1e-9)


def test_yaw_added_to_a_tilt_does_not_change_the_tilt():
    tilt = axis_angle([1, 0, 0], 25.0)
    for yaw in (10.0, 77.0, 180.0, -130.0):
        assert tilt_deg_up_axis(IDENTITY, qmul(axis_angle([0, 0, 1], yaw), tilt)) == pytest.approx(25.0, abs=1e-9)


def test_a_reference_that_is_not_upright_defines_its_own_up_axis():
    ref = axis_angle([1, 0, 0], 90.0)                  # body lying on its side
    assert tilt_deg_up_axis(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert tilt_deg_up_axis(ref, qmul(axis_angle([0, 0, 1], 55.0), ref)) == pytest.approx(0.0, abs=1e-9)
    assert tilt_deg_up_axis(ref, qmul(axis_angle([1, 0, 0], 30.0), ref)) == pytest.approx(30.0, abs=1e-9)


def test_range_unnormalised_quaternions_and_tiny_tilts():
    assert tilt_deg_up_axis(IDENTITY, axis_angle([1, 0, 0], 180.0)) == pytest.approx(180.0, abs=1e-9)
    assert tilt_deg_up_axis(IDENTITY, 3.0 * axis_angle([1, 0, 0], 40.0)) == pytest.approx(40.0, abs=1e-9)
    assert tilt_deg_up_axis(IDENTITY, axis_angle([0, 1, 0], 1e-4)) == pytest.approx(1e-4, rel=1e-6)   # keeps precision
    assert tilt_deg_up_axis(IDENTITY, IDENTITY) == 0.0


def test_matches_the_legacy_sapien_style_function_on_random_rotations():
    name = "t03_legacy_contact_topple"
    spec = importlib.util.spec_from_file_location(name, str(CONTACT_PY))
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[name] = legacy              # @dataclass in that file needs its module registered
    spec.loader.exec_module(legacy)
    rng = np.random.RandomState(5)
    for _ in range(200):
        ref, cur = rng.normal(size=4), rng.normal(size=4)
        assert tilt_deg_up_axis(ref, cur) == pytest.approx(
            legacy.tilt_deg_from_reference(ref_quat=ref, current_quat=cur), abs=1e-6)


def test_is_toppled_is_a_strict_comparison():
    assert is_toppled(30.0001, 30.0)
    assert not is_toppled(30.0, 30.0)
    assert not is_toppled(29.9, 30.0)
    assert not is_toppled(0.0, 0.0)                    # an upright object with threshold 0 is not toppled
    assert is_toppled(1e-9, 0.0)


@pytest.mark.parametrize("call", [
    lambda: is_toppled(10.0, -1.0), lambda: is_toppled(float("nan"), 30.0), lambda: is_toppled(10.0, float("nan")),
    lambda: is_toppled(True, 30.0), lambda: tilt_deg_up_axis([0, 0, 0, 0], IDENTITY),
    lambda: tilt_deg_up_axis(IDENTITY, [1, 0, 0]), lambda: tilt_deg_up_axis(IDENTITY, [1, 0, 0, float("nan")]),
    lambda: tilt_deg_up_axis("wxyz", IDENTITY),
])
def test_invalid_input_is_rejected(call):
    with pytest.raises(ValueError):
        call()
