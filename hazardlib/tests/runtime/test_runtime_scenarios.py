"""Acceptance scenarios of T04 on FakeHost: nominal hit (C13 on FakeHost), robot reaction, R1 re-check,
clock, fall gravity, pre-contact v_n, scene-before-robot contacts."""
import warnings

import numpy as np
import pytest

import runtime_world as W
from hazardlib.conformance import FakeAction
from hazardlib.host import PartnerClass
from hazardlib.runtime import R1ViolationWarning
from hazardlib.semantics import EventKind

HIT_TOL = W.TOL["c13_nominal_hit.hit_time_abs_tol_s"]
VN_TOL = W.TOL["c6_relative_velocity.normal_speed_rel_tol"]
FALL_TOL = W.TOL["c5_release_gravity.free_fall_time_rel_tol"]


def _dynamic(runtime, env=0):
    return [e for e in runtime.contact_events(env) if e.kind is EventKind.DYNAMIC]


# ------------------------------------------------------------------------------------- nominal hit (C13)

@pytest.mark.parametrize("family", ["slide", "fall"])
def test_nominal_instance_hits_at_the_nominal_time(family):
    host, runtime = W.make()
    inst = W.slide_instance(host) if family == "slide" else W.fall_instance(host)
    log = W.run_episode(host, runtime, inst, 1.0)
    assert log.first_official_contact_t_s is not None
    assert abs(log.first_official_contact_t_s - inst.nominal_hit_time_s) < HIT_TOL
    assert log.partner_class is PartnerClass.ROBOT_LINK
    assert log.kinematic_overlap_count == 0
    first = [e for e in _dynamic(runtime) if e.partner_class is PartnerClass.ROBOT_LINK][0]
    assert first.partner_name == "robot0_hand"


def test_nominal_reach_hits_within_one_physics_step():
    """A reach is released one physics step before the release pose with the R7 window velocity, which is
    not the min-jerk path's instantaneous velocity: the hit time is off by O(dt), not by float noise."""
    host, runtime = W.make()
    inst = W.reach_instance(host)
    log = W.run_episode(host, runtime, inst, 1.0)
    assert log.partner_class is PartnerClass.ROBOT_LINK and log.kinematic_overlap_count == 0
    assert abs(log.first_official_contact_t_s - inst.nominal_hit_time_s) < host.config.physics_dt_s


def test_ignoring_the_standoff_breaks_the_nominal_hit():
    host, runtime = W.make(mutations=("bad_standoff",))
    inst = W.slide_instance(host)
    assert inst.provenance["standoff_m"] == 0.0
    log = W.run_episode(host, runtime, inst, 1.0)
    # aimed at the EEF reference point: the hazard overlaps the hand ~ (r + eef radius) / speed early
    assert log.kinematic_overlap_count > 0
    assert inst.nominal_hit_time_s - log.first_official_contact_t_s > 0.8 * (W.HAZARD_R + W.HAND_R) / 1.0


# ----------------------------------------------------------------------------------------- robot reaction

def test_robot_that_steps_aside_within_tau_is_not_hit():
    host, runtime = W.make()
    inst = W.slide_instance(host)  # onset 0.2, hit 0.5
    log = W.run_episode(host, runtime, inst, 1.0,
                        actions=lambda t: FakeAction.follow() if t < 0.3 else W.away((0.0, 1.0, 0.0)))
    assert log.first_official_contact_t_s is None and log.partner_class is None
    assert log.v_rel_normal_mps is None and log.impulse_analytic_ns is None
    assert log.released_t_s is not None and log.min_separation_m > 0.0


def test_robot_that_stops_short_is_not_hit():
    host, runtime = W.make()
    inst = W.slide_instance(host, direction=(0.0, 1.0, 0.0), task="to_goal")  # hits the moving hand at 0.5 s
    log = W.run_episode(host, runtime, inst, 1.0, task="to_goal",
                        actions=lambda t: FakeAction.follow() if t < 0.2 else FakeAction.hold())
    assert log.first_official_contact_t_s is None
    assert log.task_success is False


# ------------------------------------------------------------------------------------- R1 re-check (D11 d)

def _fast_world(**kwargs):
    # 12 m/s with a 0.01 s physics step moves 0.12 m per hook: more than the 0.1 m the hand + hazard spheres
    # span along the path, so a discrete check can step over the hand.
    return W.make(physics_dt_s=0.01, substeps_per_action=2, **kwargs)


def _run_fast(host, runtime):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = W.run_episode(host, runtime, W.slide_instance(host, speed=12.0, tau=0.1), 1.0)
    return log, [w.category for w in caught]


def test_fast_hazard_is_caught_by_ccd_and_counted():
    host, runtime = _fast_world()
    log, warned = _run_fast(host, runtime)
    assert log.first_official_contact_t_s is not None
    assert log.r1_violation_count > 0
    assert R1ViolationWarning not in warned  # hazard_ccd: true -> counted, not an error


def test_tunneling_is_exposed_by_the_r1_count():
    host, runtime = _fast_world(mutations=("tunneling",))
    assert host.describe().hazard_ccd is True
    log, warned = _run_fast(host, runtime)
    assert log.first_official_contact_t_s is None  # stepped over the hand
    assert log.r1_violation_count > 0
    assert R1ViolationWarning not in warned


def test_honest_host_without_ccd_gets_a_warning():
    host, runtime = _fast_world(host_kwargs={"ccd": False})
    assert host.describe().hazard_ccd is False
    log, warned = _run_fast(host, runtime)
    assert log.r1_violation_count > 0
    assert R1ViolationWarning in warned


@pytest.mark.parametrize("family", ["slide", "fall", "reach"])
def test_slow_hazards_have_no_r1_violation(family):
    host, runtime = W.make()
    inst = {"slide": W.slide_instance, "fall": W.fall_instance, "reach": W.reach_instance}[family](host)
    log = W.run_episode(host, runtime, inst, 1.0)
    assert log.r1_violation_count == 0


# --------------------------------------------------------------------------------------------------- clock

def test_coarse_control_step_keeps_event_times_in_seconds():
    host, runtime = W.make(physics_dt_s=0.002, substeps_per_action=100)  # control step 0.2 s
    assert host.describe().timebase.control_dt_s == pytest.approx(0.2)
    first = W.slide_instance(host)
    log = W.run_episode(host, runtime, first, 1.0)
    assert abs(log.onset_t_s - first.onset_time_s) < 1e-9
    assert abs(log.first_official_contact_t_s - first.nominal_hit_time_s) < HIT_TOL
    # second episode on the same host: the host clock is at 1.0 s, instance times stay episode-relative
    off_grid = W.slide_instance(host, tau=0.2913, instance_id="slide_1")
    log = W.run_episode(host, runtime, off_grid, 1.0)
    assert runtime.episode_info(0)["t_start_s"] == pytest.approx(1.0)
    assert abs(log.onset_t_s - off_grid.onset_time_s) <= 0.5 * host.config.physics_dt_s + 1e-12
    assert abs(log.first_official_contact_t_s - off_grid.nominal_hit_time_s) < HIT_TOL


# -------------------------------------------------------------------------------------------------- fall

@pytest.mark.parametrize("host_g", [9.8, 9.81])
def test_fall_lands_with_the_hosts_gravity(host_g):
    host, runtime = W.make(gravity_mps2=host_g)
    assert host.describe().gravity_mps2 == host_g
    point = W.table_height_centroid(0.3)
    landing = {}
    for solver_g in (9.8, 9.81):
        inst = W.fall_onto(point, t_hit=0.7, tau=0.5, g=solver_g, instance_id="fall_g%s" % solver_g)
        log = W.run_episode(host, runtime, inst, 1.0)
        info = runtime.episode_info(0)["release"]
        assert info["kind"] == "onset" and not np.any(info["lin_vel_mps"]) and not np.any(info["readback_lin_vel_mps"])
        assert log.kinematic_overlap_count == 0 and log.released_t_s == log.onset_t_s
        assert not [e for e in runtime.contact_events(0) if e.kind is EventKind.KINEMATIC_OVERLAP]
        landing[solver_g] = log.first_scene_contact_t_s
    fall_time = 0.5
    assert abs(landing[host_g] - 0.7) < FALL_TOL * fall_time
    other = 9.81 if host_g == 9.8 else 9.8
    assert abs(landing[other] - 0.7) > 100 * FALL_TOL * fall_time


# --------------------------------------------------------------------------------- pre-contact v_n (D10 c)

def test_vn_is_the_known_approach_speed():
    host, runtime = W.make()
    log = W.run_episode(host, runtime, W.slide_instance(host, speed=1.5), 1.0)
    assert log.v_rel_normal_mps == pytest.approx(1.5, rel=VN_TOL)
    assert log.impulse_analytic_ns == pytest.approx(0.2 * (1.0 + 0.0) * log.v_rel_normal_mps)
    assert log.hazard_mass_kg == 0.2


def test_vn_includes_the_robot_bodys_own_velocity():
    host, runtime = W.make()
    inst = W.slide_instance(host, speed=1.0)
    log = W.run_episode(host, runtime, inst, 1.0,
                        actions=lambda t: FakeAction.hold() if t < 0.4 else W.away((-0.5, 0.0, 0.0)))
    assert log.v_rel_normal_mps == pytest.approx(1.5, rel=VN_TOL)
    assert log.first_official_contact_t_s < inst.nominal_hit_time_s


def test_post_impact_velocity_breaks_vn():
    host, runtime = W.make(mutations=("post_impact_vn",))
    log = W.run_episode(host, runtime, W.slide_instance(host, speed=1.5), 1.0)
    assert log.first_official_contact_t_s is not None
    assert log.v_rel_normal_mps != pytest.approx(1.5, rel=VN_TOL)


def test_vn_is_none_without_body_velocities():
    host, runtime = W.make(host_kwargs={"report_body_velocities": False})
    log = W.run_episode(host, runtime, W.slide_instance(host), 1.0)
    assert log.first_official_contact_t_s is not None
    assert log.v_rel_normal_mps is None and log.impulse_analytic_ns is None
    event = [e for e in _dynamic(runtime) if e.partner_class is PartnerClass.ROBOT_LINK][0]
    assert event.v_normal_mps is None


# ------------------------------------------------------------------------- scene before robot (D10 a)

def test_table_bounce_then_robot():
    host, runtime = W.make()
    inst = W.fall_onto(W.table_height_centroid(0.1))  # lands at 0.5 s, the hand sweeps in afterwards
    log = W.run_episode(host, runtime, inst, 1.0, task="sweep_in")
    assert abs(log.first_scene_contact_t_s - 0.5) < HIT_TOL
    assert log.first_official_contact_t_s > log.first_scene_contact_t_s
    robot = [e for e in _dynamic(runtime) if e.partner_class is PartnerClass.ROBOT_LINK]
    assert robot[0].partner_name == "robot0_hand"
    assert robot[0].time_s == log.first_official_contact_t_s
    assert log.partner_class is PartnerClass.ROBOT_LINK and log.v_rel_normal_mps > 0


def test_table_only_contact_has_no_official_contact():
    host, runtime = W.make()
    log = W.run_episode(host, runtime, W.fall_onto(W.table_height_centroid(0.1)), 1.0, task="static")
    assert abs(log.first_scene_contact_t_s - 0.5) < HIT_TOL
    scene = [e for e in _dynamic(runtime) if e.partner_class is PartnerClass.SCENE]
    assert scene and all(e.partner_name == "table" for e in scene)
    for field in ("first_official_contact_t_s", "partner_class", "v_rel_normal_mps", "impulse_analytic_ns",
                  "contact_duration_s", "constraint_state"):
        assert getattr(log, field) is None, field


def test_excluded_robot_body_counts_as_scene():
    host, runtime = W.make()
    camera = np.array(W.EEF_START) + np.array([0.0, 0.15, 0.1])  # robot0_camera, excluded by substr:camera
    inst = W.fall_onto(camera + np.array([0.0, 0.0, 0.02 + W.HAZARD_R]))
    log = W.run_episode(host, runtime, inst, 1.0)
    assert abs(log.first_scene_contact_t_s - 0.5) < HIT_TOL
    assert [e.partner_name for e in _dynamic(runtime)][0] == "robot0_camera"
    assert log.first_official_contact_t_s is None


def test_multi_body_held_object_needs_held_object_bodies():
    held = dict(held_object="tray", held_bodies=(W.FakeSphere("tray_a", (0.0, 0.0, -0.06), 0.02),))
    tray_top = np.array(W.EEF_START) + np.array([0.0, 0.0, -0.06])
    point = tray_top + np.array([0.06, 0.0, 0.0])  # beside the tray, below the hand
    for report, expected in ((True, PartnerClass.HELD_OBJECT), (False, PartnerClass.SCENE)):
        host, runtime = W.make(report_held_object_bodies=report, **held)
        params = {"direction": [-1.0, 0.0, 0.0], "speed_mps": 1.0, "release_window_s": W.WINDOW_S}
        nominal = W.SampledPath([0.0, 20.0], [point - [0.02, 0.0, 0.0]] * 2, None)
        plan = W.build_hit_path("intrusion.slide", nominal, 0.5, point, 0.3, params)
        inst = W.instance_from_plan(plan, "tray_hit", "slide_spec", W.sphere_agent())
        log = W.run_episode(host, runtime, inst, 1.0)
        first = _dynamic(runtime)[0]
        assert first.partner_name == "tray_a" and first.partner_class is expected
        if expected is PartnerClass.HELD_OBJECT:
            assert log.partner_class is PartnerClass.HELD_OBJECT
        else:
            assert log.first_scene_contact_t_s == first.time_s
