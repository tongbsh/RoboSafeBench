import inspect

import numpy as np
import pytest

from hazardlib import (CapabilityError, ContactObs, HazardHandle, HazardHost, OptionalHazardHost,
                       PartnerClass, Pose, RobotState, SpecError, Surface, Tier, TimeState,
                       missing_methods, require_methods)
from hazardlib import host as host_module

# Method name -> parameter names (after self), per 01_integration_rules.md §3.1/§3.2.
REQUIRED_SIGNATURES = {
    "describe": [],
    "alloc_hazard": ["env", "agent"],
    "free_hazard": ["h"],
    "set_hazard_pose": ["h", "pos", "quat"],
    "set_hazard_visible": ["h", "visible"],
    "release_to_physics": ["h", "lin_vel", "ang_vel"],
    "robot_state": ["env"],
    "object_poses": ["env"],
    "hazard_contacts": ["h"],
    "time_state": ["env"],
    "task_success": ["env"],
    "hold_action": ["env"],
    "install_hooks": ["runtime"],
}
OPTIONAL_SIGNATURES = {
    "min_distance": ["h"],
    "snapshot": ["env"],
    "restore": ["env", "snap"],
    "support_surfaces": ["env"],
    "render_check": ["h"],
}


def _protocol_methods(proto):
    return {n: f for n, f in vars(proto).items()
            if inspect.isfunction(f) and not n.startswith("_")}


def _params(func):
    return [p for p in inspect.signature(func).parameters if p != "self"]


def test_required_protocol_matches_the_integration_rules_exactly():
    methods = _protocol_methods(HazardHost)
    assert set(methods) == set(REQUIRED_SIGNATURES)
    for name, expected in REQUIRED_SIGNATURES.items():
        assert _params(methods[name]) == expected, name
    assert set(host_module.REQUIRED_METHODS) == set(REQUIRED_SIGNATURES)


def test_optional_protocol_matches_the_integration_rules_exactly():
    methods = _protocol_methods(OptionalHazardHost)
    assert set(methods) == set(OPTIONAL_SIGNATURES)
    for name, expected in OPTIONAL_SIGNATURES.items():
        assert _params(methods[name]) == expected, name
    assert set(host_module.OPTIONAL_METHODS) == set(OPTIONAL_SIGNATURES)
    assert not set(OPTIONAL_SIGNATURES) & set(REQUIRED_SIGNATURES)


def _make_host(skip=()):
    """A fresh class per call (isinstance results on Protocols are cached per class)."""
    namespace = {name: (lambda self, *a, **k: None)
                 for name in REQUIRED_SIGNATURES if name not in skip}
    return type("FakeHost", (), namespace)()


def test_runtime_protocol_check_distinguishes_complete_from_incomplete_hosts():
    complete = _make_host()
    assert isinstance(complete, HazardHost)
    assert missing_methods(complete) == []
    require_methods(complete)
    for name in REQUIRED_SIGNATURES:
        incomplete = _make_host(skip=(name,))
        assert not isinstance(incomplete, HazardHost), name
        assert missing_methods(incomplete) == [name]
        with pytest.raises(CapabilityError, match=name):
            require_methods(incomplete)


def test_non_callable_attribute_does_not_count_as_a_method():
    host = _make_host()
    type(host).task_success = 42
    assert "task_success" in missing_methods(host)


def test_optional_methods_are_not_required_for_a_host():
    host = _make_host()
    assert not isinstance(host, OptionalHazardHost)
    assert missing_methods(host, host_module.OPTIONAL_METHODS) == list(host_module.OPTIONAL_METHODS)
    require_methods(host)  # does not raise: optional capabilities are not required


def test_tier_and_capability_error_basics():
    assert [t.value for t in Tier] == ["A", "B", "C"]
    assert Tier("A") is Tier.A
    assert issubclass(CapabilityError, RuntimeError)
    assert [p.value for p in PartnerClass] == ["robot_link", "held_object", "scene", "hazard"]


# ---------------------------------------------------------------- data types

def test_pose_coerces_to_float_arrays_and_checks_shape():
    pose = Pose([0, 0, 1], (1, 0, 0, 0))
    assert pose.pos.dtype == np.float64 and pose.pos.shape == (3,)
    assert pose.quat.shape == (4,)
    for bad_pos, bad_quat in (([0, 0], [1, 0, 0, 0]), ([0, 0, 0], [1, 0, 0]),
                              ([0, 0, float("nan")], [1, 0, 0, 0]), (["a", 0, 0], [1, 0, 0, 0])):
        with pytest.raises(SpecError):
            Pose(bad_pos, bad_quat)


def test_hazard_handle_identity_ignores_the_native_payload():
    a = HazardHandle(env=0, key="h0", native=object())
    b = HazardHandle(env=0, key="h0", native=object())
    c = HazardHandle(env=1, key="h0")
    assert a == b and hash(a) == hash(b) and a != c
    assert "native" not in repr(a)
    with pytest.raises(Exception):
        a.key = "other"  # frozen


def _robot_state(**overrides):
    kwargs = dict(eef_pose=Pose([0, 0, 0], [1, 0, 0, 0]), eef_lin_vel_mps=[0, 0, 0],
                  collision_body_pos=[[0, 0, 0], [0, 0, 1]])
    kwargs.update(overrides)
    return RobotState(**kwargs)


def test_robot_state_defaults_and_validation():
    state = _robot_state()
    assert state.collision_body_pos.shape == (2, 3)
    assert state.gripper_opening_m is None and state.base_pose is None and state.held_object is None
    state = _robot_state(gripper_opening_m=0.04, held_object="cup",
                         collision_body_names=["a", "b"],
                         collision_body_lin_vel_mps=[[0, 0, 0], [1, 0, 0]],
                         base_pose=Pose([0, 0, 0], [1, 0, 0, 0]))
    assert state.gripper_opening_m == 0.04 and state.collision_body_lin_vel_mps.shape == (2, 3)
    for bad in ({"collision_body_pos": [0, 0, 0]}, {"collision_body_pos": [[0, 0]]},
                {"eef_lin_vel_mps": [0, 0]}, {"gripper_opening_m": float("nan")},
                {"collision_body_names": ["only_one"]},
                {"collision_body_lin_vel_mps": [[0, 0, 0]]}):
        with pytest.raises(SpecError):
            _robot_state(**bad)


def test_contact_obs_coerces_partner_class_and_validates():
    obs = ContactObs(partner_name="finger", partner_class="robot_link", position=[0, 0, 0],
                     normal=[0, 0, 1], time_s=0.5, impulse_ns=0.1)
    assert obs.partner_class is PartnerClass.ROBOT_LINK and obs.rel_normal_speed_mps is None
    for bad in ({"partner_class": "ghost"}, {"partner_name": ""}, {"position": [0, 0]},
                {"normal": [0, 0]}, {"time_s": float("inf")}, {"impulse_ns": float("nan")}):
        kwargs = dict(partner_name="p", partner_class="scene", position=[0, 0, 0], normal=[0, 0, 1],
                      time_s=0.0)
        kwargs.update(bad)
        with pytest.raises(SpecError):
            ContactObs(**kwargs)


def test_time_state_requires_positive_steps():
    TimeState(sim_time_s=0.0, physics_dt_s=0.004, control_dt_s=0.032)
    for bad in ({"physics_dt_s": 0.0}, {"control_dt_s": -1.0}, {"sim_time_s": float("nan")}):
        kwargs = dict(sim_time_s=0.0, physics_dt_s=0.004, control_dt_s=0.032)
        kwargs.update(bad)
        with pytest.raises(SpecError):
            TimeState(**kwargs)


def test_surface_is_a_validated_record():
    d = {"name": "table", "min_m": [0, 0, 0], "max_m": [1, 1, 0.7], "source": "scene_yaml"}
    surface = Surface.from_dict(d)
    assert surface.to_dict()["max_m"] == [1.0, 1.0, 0.7]
    for bad in ({"min_m": [2, 0, 0]}, {"name": ""}, {"source": ""}, {"max_m": [1, 1]}):
        with pytest.raises(SpecError):
            Surface.from_dict(dict(d, **bad))
