import numpy as np
import pytest

from hazardlib.kinematics import (FAMILIES, HitPlan, SampledPath, build_hit_path,
                                  fall_height_for_tau, straight_path)

G = 9.81
SAMPLES_PER_FAMILY = 40


def simpson(func, t0, t1, n):
    """Composite Simpson integral of a vector function over [t0, t1] with n (even) panels."""
    ts = np.linspace(t0, t1, n + 1)
    values = np.array([func(t) for t in ts])
    weights = np.ones(n + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return (t1 - t0) / (3.0 * n) * np.tensordot(weights, values, axes=1)


def verlet_fall(p0, v0, g, t0, t1, n):
    """Velocity Verlet for r'' = (0, 0, -g)."""
    dt = (t1 - t0) / n
    a = np.array([0.0, 0.0, -g])
    p, v = np.array(p0, dtype=float), np.array(v0, dtype=float)
    for _ in range(n):
        v_half = v + 0.5 * a * dt
        p = p + v_half * dt
        v = v_half + 0.5 * a * dt
    return p


def random_case(rng, family):
    t_hit = rng.uniform(1.0, 5.0)
    tau = rng.uniform(0.1, 1.0)
    hit_point = rng.uniform(-0.5, 0.5, 3) + np.array([0.0, 0.0, 0.5])
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    nominal = straight_path(rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3), 0.0, 6.0)
    hook_dt = rng.uniform(0.001, 0.05)
    if family == "intrusion.fall":
        params = {"g": G}
    elif family == "intrusion.slide":
        params = {"direction": direction, "speed_mps": rng.uniform(0.2, 2.0), "hook_dt_s": hook_dt}
    else:
        params = {"direction": direction, "reach_distance_m": rng.uniform(0.2, 1.0),
                  "overshoot_m": rng.uniform(0.05, 0.4),
                  "speed_profile": ["min_jerk", "linear"][rng.randint(2)], "hook_dt_s": hook_dt}
    return nominal, t_hit, hit_point, tau, params


@pytest.mark.parametrize("family", FAMILIES)
def test_hazard_reaches_the_hit_point_at_the_hit_time_by_numerical_integration(family):
    rng = np.random.RandomState({"intrusion.fall": 11, "intrusion.slide": 12, "intrusion.reach": 13}[family])
    for _ in range(SAMPLES_PER_FAMILY):
        nominal, t_hit, hit_point, tau, params = random_case(rng, family)
        plan = build_hit_path(family, nominal, t_hit, hit_point, tau, params)
        assert isinstance(plan, HitPlan) and plan.family == family
        assert plan.onset_time_s == pytest.approx(t_hit - tau, abs=1e-12)
        assert plan.path.t_start_s == plan.onset_time_s
        start = plan.path.pos(plan.onset_time_s)
        if family == "intrusion.fall":
            # the engine takes over at the release: integrate gravity from the release state
            reached = verlet_fall(start, plan.release_velocity, G, plan.release_time_s, t_hit, 2000)
        else:
            reached = start + simpson(plan.path.vel, plan.onset_time_s, t_hit, 2000)
        assert np.linalg.norm(reached - hit_point) < 1e-6          # acceptance criterion
        assert np.linalg.norm(plan.path.pos(t_hit) - hit_point) < 1e-9   # analytic position, tighter
        assert plan.nominal_hit_time_s == t_hit
        np.testing.assert_array_equal(plan.nominal_hit_point, hit_point)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_non_reacting_nominal_robot_meets_the_hazard_at_the_hit_time(family):
    rng = np.random.RandomState(21)
    for _ in range(SAMPLES_PER_FAMILY):
        nominal, t_hit, _, tau, params = random_case(rng, family)
        hit_point = nominal.pos(t_hit)          # the hit point lies on the nominal EEF path
        plan = build_hit_path(family, nominal, t_hit, hit_point, tau, params)
        assert np.linalg.norm(plan.path.pos(t_hit) - nominal.pos(t_hit)) < 1e-6
        assert plan.diagnostics["nominal_ee_offset_m"] < 1e-12
        assert np.linalg.norm(plan.path.pos(plan.onset_time_s) - nominal.pos(t_hit)) > 1e-3


def test_fall_is_released_at_rest_at_the_onset_from_the_right_height():
    nominal = straight_path([0, 0, 0], [1, 0, 0], 0.0, 4.0)
    plan = build_hit_path("intrusion.fall", nominal, 3.0, [0.2, -0.1, 0.4], 0.6, {"g": G})
    assert plan.onset_time_s == pytest.approx(2.4, abs=1e-12)
    assert plan.release_time_s == plan.onset_time_s
    np.testing.assert_array_equal(plan.release_velocity, [0, 0, 0])
    height = fall_height_for_tau(0.6, G)
    np.testing.assert_allclose(plan.path.pos(plan.onset_time_s), [0.2, -0.1, 0.4 + height])
    np.testing.assert_allclose(plan.hit_velocity, [0, 0, -G * 0.6])      # speed g tau at the hit
    assert plan.diagnostics["travel_distance_m"] == pytest.approx(height, rel=1e-12)
    assert plan.diagnostics["hit_speed_mps"] == pytest.approx(G * 0.6, rel=1e-12)


def test_slide_runs_at_constant_speed_along_the_direction_and_releases_at_the_hit():
    nominal = straight_path([0, 0, 0.3], [0, 0, 0.3], 0.0, 5.0)        # a stationary nominal EEF
    params = {"direction": [0.0, 2.0, 0.0], "speed_mps": 0.8, "hook_dt_s": 0.004}
    plan = build_hit_path("intrusion.slide", nominal, 3.0, [0.0, 0.0, 0.3], 0.5, params)
    np.testing.assert_allclose(plan.path.pos(plan.onset_time_s), [0.0, -0.4, 0.3])   # hit - v tau d
    assert plan.release_time_s == 3.0
    np.testing.assert_allclose(plan.release_velocity, [0, 0.8, 0], atol=1e-10)
    np.testing.assert_allclose(plan.hit_velocity, [0, 0.8, 0], atol=1e-12)
    for t in (2.5, 2.75, 3.0):
        np.testing.assert_allclose(plan.path.vel(t), [0, 0.8, 0], atol=1e-12)
    assert plan.diagnostics["travel_distance_m"] == pytest.approx(0.4, rel=1e-12)
    assert plan.diagnostics["relative_speed_at_hit_mps"] == pytest.approx(0.8, rel=1e-12)   # EEF at rest


def test_relative_speed_uses_the_nominal_end_effector_velocity():
    nominal = straight_path([0, 0, 0], [0, 4, 0], 0.0, 4.0)           # moving at 1 m/s along +y
    params = {"direction": [0, -1, 0], "speed_mps": 0.5, "hook_dt_s": 0.01}   # head-on slide
    plan = build_hit_path("intrusion.slide", nominal, 2.0, nominal.pos(2.0), 0.4, params)
    assert plan.diagnostics["relative_speed_at_hit_mps"] == pytest.approx(1.5, rel=1e-9)


def reach_reference(reach, overshoot, tau, profile):
    """Independent solution: hit fraction f, normalised hit time u (polynomial root), duration and speed."""
    f = reach / (reach + overshoot)
    if profile == "linear":
        u = f
        ds = 1.0
    else:
        roots = np.roots([6.0, -15.0, 10.0, 0.0, 0.0, -f])
        u = [r.real for r in roots if abs(r.imag) < 1e-9 and -1e-9 <= r.real <= 1 + 1e-9][0]
        ds = 30.0 * u * u * (1 - u) ** 2
    duration = tau / u
    return u, duration, (reach + overshoot) * ds / duration


@pytest.mark.parametrize("profile", ["min_jerk", "linear"])
def test_reach_duration_and_speed_match_an_independent_solution(profile):
    rng = np.random.RandomState(31)
    for _ in range(SAMPLES_PER_FAMILY):
        reach, overshoot = rng.uniform(0.2, 1.0), rng.uniform(0.05, 0.5)
        tau = rng.uniform(0.2, 1.0)
        nominal = straight_path([0, 0, 0], [1, 0, 0], 0.0, 6.0)
        params = {"direction": [1, 1, 0], "reach_distance_m": reach, "overshoot_m": overshoot,
                  "speed_profile": profile, "hook_dt_s": 0.002}
        plan = build_hit_path("intrusion.reach", nominal, 4.0, [0.5, 0.5, 0.3], tau, params)
        u, duration, speed = reach_reference(reach, overshoot, tau, profile)
        assert plan.path.duration_s == pytest.approx(duration, rel=1e-9)
        assert plan.diagnostics["hit_speed_mps"] == pytest.approx(speed, rel=1e-8)
        np.testing.assert_allclose(plan.hit_velocity / np.linalg.norm(plan.hit_velocity),
                                   np.array([1, 1, 0]) / np.sqrt(2), atol=1e-12)
        assert plan.diagnostics["travel_distance_m"] == pytest.approx(reach, rel=1e-12)
        assert plan.release_time_s == 4.0


def test_reach_release_velocity_is_the_backward_finite_difference_without_a_floor():
    nominal = straight_path([0, 0, 0], [1, 0, 0], 0.0, 6.0)
    hook_dt = 0.02
    params = {"direction": [0, 0, 1], "reach_distance_m": 0.6, "overshoot_m": 0.2,
              "speed_profile": "min_jerk", "hook_dt_s": hook_dt}
    plan = build_hit_path("intrusion.reach", nominal, 4.0, [0.0, 0.0, 0.5], 0.7, params)
    np.testing.assert_array_equal(plan.release_velocity,
                                  plan.path.finite_difference_velocity(4.0, hook_dt))   # R7, exact
    total = 0.8
    max_accel = (10.0 / np.sqrt(3.0)) * 6.0 * total / plan.path.duration_s ** 2   # |s''| <= 10/sqrt(3)
    gap = np.linalg.norm(plan.release_velocity - plan.hit_velocity)
    assert gap <= 0.5 * max_accel * hook_dt * (1 + 1e-9)    # first-order accuracy of the difference


def test_reach_without_overshoot_arrives_at_rest_for_min_jerk():
    nominal = straight_path([0, 0, 0], [1, 0, 0], 0.0, 6.0)
    params = {"direction": [1, 0, 0], "reach_distance_m": 0.5, "overshoot_m": 0.0,
              "speed_profile": "min_jerk", "hook_dt_s": 0.01}
    plan = build_hit_path("intrusion.reach", nominal, 3.0, [0.5, 0, 0], 0.5, params)
    assert plan.path.duration_s == pytest.approx(0.5, rel=1e-12)
    assert plan.diagnostics["hit_speed_mps"] < 1e-12
    np.testing.assert_allclose(plan.path.pos(3.0), [0.5, 0, 0], atol=1e-12)


def test_onset_margin_flags_an_onset_before_the_nominal_path_starts():
    nominal = straight_path([0, 0, 0], [1, 0, 0], 0.0, 6.0)
    early = build_hit_path("intrusion.fall", nominal, 0.3, [0.1, 0, 0.2], 0.5, {"g": G})
    assert early.diagnostics["onset_margin_s"] == pytest.approx(-0.2, abs=1e-12)
    ok = build_hit_path("intrusion.fall", nominal, 2.0, [0.1, 0, 0.2], 0.5, {"g": G})
    assert ok.diagnostics["onset_margin_s"] == pytest.approx(1.5, abs=1e-12)


def test_offset_hit_point_is_reported_not_moved():
    nominal = straight_path([0, 0, 0], [1, 0, 0], 0.0, 4.0)
    offset = np.array([0.0, 0.05, 0.0])
    plan = build_hit_path("intrusion.fall", nominal, 2.0, nominal.pos(2.0) + offset, 0.4, {"g": G})
    assert plan.diagnostics["nominal_ee_offset_m"] == pytest.approx(0.05, rel=1e-12)
    np.testing.assert_allclose(plan.path.pos(2.0), nominal.pos(2.0) + offset, atol=1e-12)


def test_sampled_nominal_path_is_accepted():
    nominal = SampledPath([0.0, 1.0, 2.0], [[0, 0, 0.3], [0.2, 0, 0.3], [0.2, 0.2, 0.3]], None)
    plan = build_hit_path("intrusion.fall", nominal, 1.5, nominal.pos(1.5), 0.4, {"g": G})
    assert plan.diagnostics["nominal_ee_offset_m"] < 1e-12


def valid_slide():
    return ("intrusion.slide", straight_path([0, 0, 0], [1, 0, 0], 0.0, 4.0), 2.0, [0, 0, 0.3], 0.5,
            {"direction": [1, 0, 0], "speed_mps": 0.5, "hook_dt_s": 0.01})


def call(overrides):
    args = list(valid_slide())
    for index, value in overrides.items():
        args[index] = value
    return build_hit_path(*args)


def test_baseline_slide_call_is_valid():
    assert call({}).family == "intrusion.slide"


@pytest.mark.parametrize("overrides,match", [
    ({0: "occupancy.dwell"}, "unsupported family"),
    ({0: "nonsense"}, "unsupported family"),
    ({1: "not a path"}, "must be a hazardlib.kinematics.Path"),
    ({2: 4.5}, "outside the nominal EEF path span"),
    ({2: -0.1}, "outside the nominal EEF path span"),
    ({2: float("nan")}, "t_hit"),
    ({3: [0, 0]}, "hit_point"),
    ({4: 0.0}, "tau_s"),
    ({4: -0.5}, "tau_s"),
    ({4: float("inf")}, "tau_s"),
    ({5: {"direction": [1, 0, 0], "speed_mps": 0.5}}, "missing"),
    ({5: {"direction": [1, 0, 0], "speed_mps": 0.5, "hook_dt_s": 0.01, "extra": 1}}, "unexpected"),
    ({5: {"direction": [0, 0, 0], "speed_mps": 0.5, "hook_dt_s": 0.01}}, "non-zero"),
    ({5: {"direction": [1, 0, 0], "speed_mps": 0.0, "hook_dt_s": 0.01}}, "speed_mps"),
    ({5: {"direction": [1, 0, 0], "speed_mps": 0.5, "hook_dt_s": 0.0}}, "hook_dt_s"),
    ({5: {"direction": [1, 0, 0], "speed_mps": 0.5, "hook_dt_s": 0.6}}, "exceeds the kinematic phase"),
    ({5: [1, 2, 3]}, "params must be a mapping"),
])
def test_invalid_calls_are_rejected_with_a_clear_message(overrides, match):
    with pytest.raises(ValueError, match=match):
        call(overrides)


def test_fall_and_reach_parameter_validation():
    nominal = straight_path([0, 0, 0], [1, 0, 0], 0.0, 4.0)
    with pytest.raises(ValueError, match="params.g"):
        build_hit_path("intrusion.fall", nominal, 2.0, [0, 0, 0.1], 0.4, {"g": 0.0})
    with pytest.raises(ValueError, match="missing"):
        build_hit_path("intrusion.fall", nominal, 2.0, [0, 0, 0.1], 0.4, {})
    good = {"direction": [1, 0, 0], "reach_distance_m": 0.5, "overshoot_m": 0.1,
            "speed_profile": "min_jerk", "hook_dt_s": 0.01}
    for key, bad, match in (("overshoot_m", -0.1, "overshoot_m"), ("reach_distance_m", 0.0, "reach"),
                            ("speed_profile", "cubic", "speed_profile")):
        with pytest.raises(ValueError, match=match):
            build_hit_path("intrusion.reach", nominal, 2.0, [0, 0, 0.1], 0.4, dict(good, **{key: bad}))


def test_supported_families_are_exactly_the_three_intrusion_families():
    assert FAMILIES == ("intrusion.fall", "intrusion.slide", "intrusion.reach")


def test_results_do_not_alias_the_caller_inputs():
    nominal = straight_path([0, 0, 0], [1, 0, 0], 0.0, 4.0)
    hit_point = np.array([0.0, 0.0, 0.3])
    direction = np.array([1.0, 0.0, 0.0])
    plan = build_hit_path("intrusion.slide", nominal, 2.0, hit_point,  0.5,
                          {"direction": direction, "speed_mps": 0.5, "hook_dt_s": 0.01})
    hit_point[:] = 9.0
    direction[:] = 9.0
    np.testing.assert_allclose(plan.nominal_hit_point, [0, 0, 0.3])
    np.testing.assert_allclose(plan.path.pos(2.0), [0, 0, 0.3])
