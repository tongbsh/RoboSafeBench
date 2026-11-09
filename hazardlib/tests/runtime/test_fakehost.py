"""FakeHost on its own: exact CCD physics, capabilities, and the effect of every mutation (used by T05)."""
import math

import numpy as np
import pytest

import runtime_world as W
from hazardlib.conformance import MUTATIONS, FakeAction, FakeHost, FakeSphere, FakeWorldConfig
from hazardlib.host import CapabilityError, PartnerClass
from hazardlib.spec import AgentSpec


def _host(mutations=(), **world):
    host = FakeHost(W.world_config(**world), mutations=mutations)
    host.reset_world(0, "static", "objects", 0)
    return host


def _dynamic(host, pos, vel=(0.0, 0.0, 0.0), env=0):
    h = host.alloc_hazard(env, W.sphere_agent())
    host.set_hazard_pose(h, pos, (1.0, 0.0, 0.0, 0.0))
    host.set_hazard_visible(h, True)
    host.release_to_physics(h, vel, (0.0, 0.0, 0.0))
    return h


def _run_until_contact(host, h, steps, env=0):
    for _ in range(steps):
        host.advance_physics(env, FakeAction.hold())
        contacts = host.hazard_contacts(h)
        if contacts:
            return contacts
    return []


# ------------------------------------------------------------------------------------------- mechanics

def test_ccd_landing_time_is_exact():
    host = _host()
    height = 0.5
    h = _dynamic(host, (0.6, 0.6, W.HAZARD_R + height))
    contact = _run_until_contact(host, h, 1000)[0]
    t_land = math.sqrt(2 * height / 9.81)
    assert contact.partner_name == "table" and contact.partner_class is PartnerClass.SCENE
    assert contact.time_s == pytest.approx(t_land, abs=1e-12)
    np.testing.assert_allclose(contact.normal, [0.0, 0.0, 1.0])
    # bounced with e = 0.5, then decelerated by gravity for the rest of the physics step
    rest = host.time_state(0).sim_time_s - contact.time_s
    assert host.hazard_state(h).lin_vel_mps[2] == pytest.approx(0.5 * 9.81 * t_land - 9.81 * rest, abs=1e-12)


def test_sphere_contact_reflects_with_restitution_and_reports_the_normal():
    host = _host()
    h = _dynamic(host, (-0.06, 0.0, 0.3), vel=(2.0, 0.0, 0.0))  # 5 ms to the hand: gravity barely acts
    contact = _run_until_contact(host, h, 200)[0]
    assert contact.partner_name == "robot0_hand" and contact.partner_class is PartnerClass.ROBOT_LINK
    assert contact.time_s == pytest.approx(0.005, abs=1e-5)
    assert contact.normal[0] < -0.9999
    before = np.array([2.0, 0.0, -9.81 * contact.time_s])
    after = before - (1.0 + 0.5) * (before @ contact.normal) * contact.normal  # e = 0.5, immovable hand
    assert host.hazard_state(h).lin_vel_mps[0] == pytest.approx(after[0], abs=1e-12)


def test_hazard_dropped_on_a_sphere_ends_up_carried_by_it():
    host = _host()
    h = _dynamic(host, (0.0, 0.0, 0.3 + W.HAND_R + W.HAZARD_R + 0.2))
    for _ in range(500):  # the bounces on top of the still hand die out after ~0.4 s
        host.advance_physics(0, FakeAction.hold())
    host.hazard_contacts(h)
    for _ in range(500):
        host.advance_physics(0, FakeAction.velocity((0.0, 0.1, 0.0)))
    contacts = host.hazard_contacts(h)
    assert len(contacts) == 500 and {c.partner_name for c in contacts} == {"robot0_hand"}
    assert contacts and contacts[-1].partner_name == "robot0_hand"
    state = host.hazard_state(h)
    eef = host.robot_state(0).eef_pose.pos
    assert np.linalg.norm(state.pos - eef) == pytest.approx(W.HAND_R + W.HAZARD_R)
    np.testing.assert_allclose(state.lin_vel_mps, [0.0, 0.1, 0.0], atol=1e-12)


def test_kinematic_hazard_reads_back_the_pose_with_zero_velocity_and_overlaps_are_reported():
    host = _host()
    h = host.alloc_hazard(0, W.sphere_agent())
    host.set_hazard_visible(h, True)
    host.set_hazard_pose(h, (0.0, 0.0, 0.31), (1.0, 0.0, 0.0, 0.0))
    state = host.hazard_state(h)
    np.testing.assert_allclose(state.pos, [0.0, 0.0, 0.31])
    assert not np.any(state.lin_vel_mps)
    host.advance_physics(0, FakeAction.hold())
    assert [c.partner_name for c in host.hazard_contacts(h)] == ["robot0_hand"]
    np.testing.assert_allclose(host.hazard_state(h).pos, [0.0, 0.0, 0.31])


def test_parked_hazard_touches_nothing_and_is_not_rendered():
    host = _host()
    before = host.policy_observation(0)["rgb"].copy()
    h = host.alloc_hazard(0, W.sphere_agent())
    host.set_hazard_pose(h, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))  # posed but still hidden
    host.set_hazard_visible(h, False)
    host.advance_physics(0, FakeAction.hold())
    np.testing.assert_allclose(host.hazard_state(h).pos, W.PARK)
    assert host.hazard_contacts(h) == [] and host.render_check(h) is False
    assert np.array_equal(host.policy_observation(0)["rgb"], before)
    host.set_hazard_pose(h, (0.0, -0.2, 0.3), (1.0, 0.0, 0.0, 0.0))
    host.set_hazard_visible(h, True)
    assert host.render_check(h) is True
    assert not np.array_equal(host.policy_observation(0)["rgb"], before)


def test_only_spheres_and_known_options():
    host = _host()
    box = AgentSpec.from_dict({"kind": "object", "shape": "box", "half_extents_m": [0.02, 0.02, 0.03],
                               "mass_kg": 0.2})
    with pytest.raises(CapabilityError, match="spheres only"):
        host.alloc_hazard(0, box)
    with pytest.raises(ValueError, match="unknown mutation"):
        FakeHost(W.world_config(), mutations=("gravity_off",))
    with pytest.raises(ValueError, match="only optional methods"):
        FakeHost(W.world_config(), disable=("hazard_state",))
    with pytest.raises(ValueError, match="reserved"):
        FakeSphere("table", (0.0, 0.0, 0.0), 0.1)
    with pytest.raises(RuntimeError, match="has not been reset"):
        FakeHost(W.world_config()).robot_state(0)
    assert isinstance(W.world_config(), FakeWorldConfig)


def test_snapshot_restore_replays_exactly():
    host = _host()
    h = _dynamic(host, (0.6, 0.6, 0.6), vel=(0.3, 0.0, 0.0))
    snap = host.snapshot(0)
    trace = []
    for _ in range(300):
        host.advance_physics(0, FakeAction.hold())
        trace.append(host.hazard_state(h).pos.copy())
    host.restore(0, snap)
    for expected in trace:
        host.advance_physics(0, FakeAction.hold())
        assert np.array_equal(host.hazard_state(h).pos, expected)


def test_bd_is_complete_and_tier_a():
    bd = FakeHost(W.world_config()).describe()
    assert bd.unknown_fields() == [] and bd.tier().value == "A"
    assert bd.gravity_mps2 == 9.81 and bd.hazard_ccd is True and bd.eef_contact_radius_m == W.HAND_R
    assert bd.timebase.physics_dt_s == 0.002 and bd.timebase.substeps_per_action == 10
    assert list(bd.robot_contact_bodies.include) == ["prefix:robot0_", "held_object"]


# ------------------------------------------------------------------------------------------- mutations

def test_mutation_list_is_the_t04_list():
    # the T04 list plus T07's bad_geometry (D15 a)
    assert set(MUTATIONS) == {"tunneling", "residual_velocity", "state_leak", "wrong_dt", "no_rollback",
                              "park_interacts", "oracle_depends_on_hazard", "bad_standoff",
                              "hook_dependent_release", "post_impact_vn", "bad_geometry"}


def test_tunneling_steps_over_the_hand_but_declares_ccd():
    host = _host(("tunneling",), physics_dt_s=0.01)
    assert host.describe().hazard_ccd is True
    h = _dynamic(host, (-0.11, 0.0, 0.3), vel=(12.0, 0.0, 0.0))
    assert _run_until_contact(host, h, 3) == []
    clean = _host(physics_dt_s=0.01)
    h = _dynamic(clean, (-0.11, 0.0, 0.3), vel=(12.0, 0.0, 0.0))
    assert _run_until_contact(clean, h, 3)[0].partner_name == "robot0_hand"


def test_residual_velocity_stays_on_the_body():
    host = _host(("residual_velocity",))
    h = host.alloc_hazard(0, W.sphere_agent())
    host.set_hazard_pose(h, (0.0, -0.2, 0.3), (1.0, 0.0, 0.0, 0.0))
    host.set_hazard_pose(h, (0.0, -0.198, 0.3), (1.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(host.hazard_state(h).lin_vel_mps, [0.0, 1.0, 0.0], atol=1e-9)
    host.release_to_physics(h, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    np.testing.assert_allclose(host.hazard_state(h).lin_vel_mps, [1.0, 1.0, 0.0], atol=1e-9)


def test_state_leak_appends_the_hazard_position():
    clean, leaky = _host(), _host(("state_leak",))
    for host in (clean, leaky):
        h = host.alloc_hazard(0, W.sphere_agent())
        host.set_hazard_pose(h, (0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0))
        host.set_hazard_visible(h, True)
    state = leaky.policy_observation(0)["state"]
    assert state.size == clean.policy_observation(0)["state"].size + 3
    np.testing.assert_allclose(state[-3:], [0.1, 0.2, 0.3])


def test_wrong_dt_declares_twice_the_real_step():
    host = _host(("wrong_dt",))
    assert host.describe().timebase.physics_dt_s == 2 * host.time_state(0).physics_dt_s


def test_no_rollback_ignores_the_pose_after_an_overlap():
    host = _host(("no_rollback",))
    h = host.alloc_hazard(0, W.sphere_agent())
    host.set_hazard_visible(h, True)
    host.set_hazard_pose(h, (0.0, 0.0, 0.31), (1.0, 0.0, 0.0, 0.0))
    host.advance_physics(0, FakeAction.hold())
    assert host.hazard_contacts(h)
    host.set_hazard_pose(h, (0.0, -0.2, 0.3), (1.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(host.hazard_state(h).pos, [0.0, 0.0, 0.31])
    host.advance_physics(0, FakeAction.hold())
    host.hazard_contacts(h)
    host.advance_physics(0, FakeAction.hold())
    assert host.hazard_contacts(h)  # still overlapping: the rollback never happened


def test_park_interacts_collides_and_renders_at_the_parking_pose():
    host = _host(("park_interacts",), park_pos_m=(0.3, 0.3, 0.0))
    before = host.policy_observation(0)["rgb"].copy()
    h = host.alloc_hazard(0, W.sphere_agent())
    host.set_hazard_visible(h, False)
    host.advance_physics(0, FakeAction.hold())
    assert [c.partner_name for c in host.hazard_contacts(h)] == ["table"]
    assert host.render_check(h) is True
    assert not np.array_equal(host.policy_observation(0)["rgb"], before)


def test_oracle_depends_on_the_visible_hazard():
    host = FakeHost(W.world_config(), mutations=("oracle_depends_on_hazard",))
    host.reset_world(0, "static", "empty", 0)
    assert host.task_success(0) is True
    h = host.alloc_hazard(0, W.sphere_agent())
    host.set_hazard_visible(h, True)
    assert host.task_success(0) is False


def test_bad_standoff_aims_at_the_eef_reference_point():
    clean, bad = _host(), _host(("bad_standoff",))
    np.testing.assert_allclose(W.slide_instance(bad).nominal_hit_point, W.EEF_START)
    np.testing.assert_allclose(W.slide_instance(clean).nominal_hit_point,
                               np.array(W.EEF_START) - [W.HAZARD_R + W.HAND_R, 0.0, 0.0])


def test_hook_dependent_release_uses_the_last_two_poses():
    host = _host(("hook_dependent_release",))
    h = host.alloc_hazard(0, W.sphere_agent())
    host.set_hazard_pose(h, (0.0, -0.2, 0.3), (1.0, 0.0, 0.0, 0.0))
    host.set_hazard_pose(h, (0.0, -0.204, 0.3), (1.0, 0.0, 0.0, 0.0))
    host.release_to_physics(h, (5.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    np.testing.assert_allclose(host.hazard_state(h).lin_vel_mps, [0.0, -2.0, 0.0], atol=1e-9)


def test_post_impact_vn_reports_contacts_one_hook_late():
    host = _host(("post_impact_vn",))
    h = _dynamic(host, (-0.2, 0.0, 0.3), vel=(2.0, 0.0, 0.0))
    steps = 0
    while True:
        host.advance_physics(0, FakeAction.hold())
        steps += 1
        contacts = host.hazard_contacts(h)
        if contacts:
            break
    t_contact = contacts[0].time_s
    assert host.time_state(0).sim_time_s - t_contact > host.config.physics_dt_s - 1e-12
