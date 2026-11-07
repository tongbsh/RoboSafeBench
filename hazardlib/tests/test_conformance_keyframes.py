import numpy as np
import pytest

from hazardlib.conformance.harness import build_nominal_hit_instance, path_instance
from hazardlib.kinematics import straight_path
from hazardlib.kinematics.hit import FAMILY_FALL, FAMILY_SLIDE
from hazardlib.spec.hazard_spec import AgentSpec


def sphere(radius=0.03):
    return AgentSpec.from_dict({"kind": "object", "shape": "sphere", "half_extents_m": [radius], "mass_kg": 0.2})


def keyframe_times(instance):
    return np.asarray(instance.to_dict()["keyframes"]["time_s"], dtype=np.float64)


def test_path_instance_builds_when_the_span_is_not_a_float_multiple_of_the_keyframe_dt():
    instance = path_instance("p", sphere(), [0.4, 0.0, 0.8], [0.0, -0.5, 0.0], 0.5, 0.8, [0.0, -0.5, 0.0], 0.01)
    times = keyframe_times(instance)
    assert times[0] == 0.5 and times[-1] == 0.8
    assert np.all(np.diff(times) > 0)


@pytest.mark.parametrize("tau", [0.1, 0.15, 0.2, 0.3, 0.6])
def test_nominal_hit_instance_builds_on_the_suite_grid(tau):
    nominal = straight_path([0.3, 0.0, 0.6], [0.5, 0.1, 0.6], 0.0, 6.0)
    for k in range(30, 120):
        t_hit = k / 30.0
        fall = build_nominal_hit_instance("f", nominal, FAMILY_FALL, sphere(), t_hit, tau, {"g": 9.81}, 0.02, 0.01)
        assert np.all(np.diff(keyframe_times(fall)) > 0)
        slide = build_nominal_hit_instance(
            "s", nominal, FAMILY_SLIDE, sphere(), t_hit, tau,
            {"direction": [0.0, -1.0, 0.0], "speed_mps": 1.0, "release_window_s": 0.05}, 0.02, 0.01)
        assert np.all(np.diff(keyframe_times(slide)) > 0)
