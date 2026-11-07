"""Episode binding (D11 a), control episodes, environment independence, parking (R6), provenance (D11 c),
instability attribution (D10 b), topple (D10 h) and missing-is-None logging."""
import numpy as np
import pytest

import runtime_world as W
from hazardlib.conformance import FakeAction
from hazardlib.semantics import EventKind, HazardEventLog, InvalidClass, SeparationKind

HAZARD_FIELDS = ("onset_t_s", "released_t_s", "first_official_contact_t_s", "first_scene_contact_t_s",
                 "partner_class", "v_rel_normal_mps", "hazard_mass_kg", "impulse_analytic_ns",
                 "engine_peak_force_n", "force_source", "contact_duration_s", "constraint_state",
                 "min_separation_m", "separation_kind", "ttc_min_s", "exposure_low", "kinematic_overlap_count",
                 "r1_violation_count")
PROVENANCE_FIELDS = ("bd_hash", "adapter_version", "manifest_hash", "benchmark_commits")


def _events(runtime, env=0):
    return [(e.kind, e.time_s, e.partner_name, e.partner_class) for e in runtime.contact_events(env)]


# ---------------------------------------------------------------------------------- binding (D11 a)

def _hooks_by_hand(host, runtime, inst, duration_s):
    runtime.arm(0, inst)
    host.reset_world(0, "static", "empty", 0)
    runtime.begin_episode(0)
    for _ in range(int(round(duration_s / host.config.physics_dt_s))):
        runtime.pre_step(0)
        host.advance_physics(0, FakeAction.follow())
        runtime.post_step(0)
    return runtime.end_episode(0)


@pytest.mark.parametrize("family", ["slide", "fall"])
def test_driver_and_direct_hooks_give_the_same_log(family):
    make_inst = W.slide_instance if family == "slide" else W.fall_instance
    host_a, runtime_a = W.make()
    log_a = W.run_episode(host_a, runtime_a, make_inst(host_a), 1.0)
    host_b, runtime_b = W.make()
    log_b = _hooks_by_hand(host_b, runtime_b, make_inst(host_b), 1.0)
    assert log_a.to_dict() == log_b.to_dict()
    assert _events(runtime_a) == _events(runtime_b)
    assert log_a.first_official_contact_t_s is not None


def test_binding_errors_are_loud():
    host, runtime = W.make()
    with pytest.raises(RuntimeError, match="without arm"):
        host.reset_episode(0, "static", "empty", 0)
    with pytest.raises(RuntimeError, match="without a running episode"):
        runtime.end_episode(0)
    with pytest.raises(RuntimeError, match="outside an episode"):
        runtime.report_instability(0, "nan")
    inst = W.slide_instance(host)
    runtime.arm(0, inst)
    with pytest.raises(RuntimeError, match="already armed"):
        runtime.arm(0, inst)
    host.reset_episode(0, "static", "empty", 0)
    runtime.arm(0, inst)
    with pytest.raises(RuntimeError, match="while an episode is running"):
        host.reset_episode(0, "static", "empty", 0)
    runtime.end_episode(0)


@pytest.mark.parametrize("kwargs, message", [
    (dict(instance=None), "control_id"),
    (dict(instance=None, control_id="ctl", task_entities=["cup"]), "control_onset_s"),
    (dict(instance="inst", control_id="ctl"), "control episodes"),
    (dict(instance="inst", task_entities="cup"), "list of names"),
    (dict(instance="inst", task_entities=["cup", "cup"]), "duplicates"),
    (dict(instance="inst", task_entities=[]), "non-empty"),
])
def test_arm_checks_its_arguments(kwargs, message):
    host, runtime = W.make()
    if kwargs["instance"] == "inst":
        kwargs["instance"] = W.slide_instance(host)
    with pytest.raises(ValueError, match=message):
        runtime.arm(0, **kwargs)


def test_unknown_task_entity_fails_at_the_reset():
    host, runtime = W.make()
    runtime.arm(0, W.slide_instance(host), task_entities=["vase"])
    with pytest.raises(ValueError, match="vase"):
        host.reset_episode(0, "static", "objects", 0)


def test_control_episode_logs_no_hazard_quantity():
    host, runtime = W.make(provenance=W.provenance())
    log = W.run_episode(host, runtime, None, 1.0, task="to_goal", layout="objects", control_id="ctl_7",
                        task_entities=["cup", "bowl"], control_onset_s=0.2)
    assert log.instance_id == "ctl_7"
    for field in HAZARD_FIELDS:
        assert getattr(log, field) is None, field
    assert log.task_success is False  # the goal is reached at 2 s
    assert log.toppled is False and log.max_tilt_deg == 0.0
    assert log.invalid_class is InvalidClass.NONE and log.attribution_ambiguous is False
    assert log.bd_hash == "a" * 64
    assert runtime.contact_events(0) == []
    log = W.run_episode(host, runtime, None, 2.5, task="to_goal", control_id="ctl_8")
    assert log.task_success is True and log.toppled is None and log.max_tilt_deg is None


# -------------------------------------------------------------------------------- independence, parking

def _observe(host, env):
    obs = host.policy_observation(env)
    return obs["rgb"].copy(), obs["state"].copy()


def _interleaved(host, runtime, plans, duration_s):
    for env, (inst, task) in plans.items():
        runtime.arm(env, inst)
        host.reset_episode(env, task, "empty", env)
    seen = {env: [] for env in plans}
    for _ in range(int(round(duration_s / 0.02))):
        for env in plans:
            host.step(env, FakeAction.follow())
            seen[env].append(_observe(host, env))
    return {env: runtime.end_episode(env) for env in plans}, seen


def test_environments_are_independent():
    host, runtime = W.make()
    plans = {0: (W.slide_instance(host), "static"), 1: (W.fall_onto(W.table_height_centroid(0.1)), "static")}
    logs, seen = _interleaved(host, runtime, plans, 1.0)
    events = {env: _events(runtime, env) for env in plans}
    for env in plans:
        solo_host, solo_runtime = W.make()
        solo_logs, solo_seen = _interleaved(solo_host, solo_runtime, {env: plans[env]}, 1.0)
        assert solo_logs[env].to_dict() == logs[env].to_dict()
        assert _events(solo_runtime, env) == events[env]
        for (rgb, state), (solo_rgb, solo_state) in zip(seen[env], solo_seen[env]):
            assert np.array_equal(rgb, solo_rgb) and np.array_equal(state, solo_state)
    assert logs[0].first_official_contact_t_s is not None and logs[1].first_official_contact_t_s is None


def test_hazard_is_parked_before_and_after_and_invisible_while_parked():
    host, runtime = W.make()
    control_host, control_runtime = W.make()
    inst = W.slide_instance(host)   # first keyframe (placement) at the onset, 0.2 s
    runtime.arm(0, inst)
    control_runtime.arm(0, None, control_id="ctl")
    host.reset_episode(0, "static", "empty", 0)
    control_host.reset_episode(0, "static", "empty", 0)
    assert all(np.array_equal(a, b) for a, b in zip(_observe(host, 0), _observe(control_host, 0)))
    differed = False
    for k in range(50):
        t = k * 0.02
        host.step(0, FakeAction.follow())
        control_host.step(0, FakeAction.follow())
        (rgb, state), (control_rgb, control_state) = _observe(host, 0), _observe(control_host, 0)
        assert np.array_equal(state, control_state)  # R5: the hazard is never in the state vector
        if t + 0.02 < inst.onset_time_s - 1e-9:
            assert np.array_equal(rgb, control_rgb), t
        differed = differed or not np.array_equal(rgb, control_rgb)
    assert differed  # the placed hazard is visible
    runtime.end_episode(0)
    control_runtime.end_episode(0)
    assert all(np.array_equal(a, b) for a, b in zip(_observe(host, 0), _observe(control_host, 0)))
    first = min(e.time_s for e in runtime.contact_events(0))
    assert first >= inst.onset_time_s


def test_contacts_of_a_parked_hazard_are_ignored():
    """``park_interacts`` makes the parked hazard touch the table (it is parked below it); the runtime drains
    those contacts. Catching the host defect is C2's job, not the runtime's."""
    host, runtime = W.make(mutations=("park_interacts",))
    clean_host, clean_runtime = W.make()
    log = W.run_episode(host, runtime, W.slide_instance(host), 1.0)
    clean = W.run_episode(clean_host, clean_runtime, W.slide_instance(clean_host), 1.0)
    assert log.to_dict() == clean.to_dict()
    assert _events(runtime) == _events(clean_runtime)


# ---------------------------------------------------------------------------------- provenance (D11 c)

def test_provenance_is_copied_into_every_log():
    host, runtime = W.make(provenance=W.provenance())
    log = W.run_episode(host, runtime, W.slide_instance(host), 1.0)
    assert (log.bd_hash, log.adapter_version, log.manifest_hash, log.benchmark_commits) == \
        ("a" * 64, "fakehost-test", "b" * 64, {"fakehost": "builtin"})
    assert log.task_success is True
    host, runtime = W.make()
    log = W.run_episode(host, runtime, W.slide_instance(host), 1.0, layout="objects", task_entities=["cup"])
    for field in PROVENANCE_FIELDS:
        assert getattr(log, field) is None, field
    assert log.max_tilt_deg == 0.0 and log.toppled is False


def test_task_success_is_read_before_the_hazard_is_parked():
    host, runtime = W.make(mutations=("oracle_depends_on_hazard",))
    log = W.run_episode(host, runtime, W.slide_instance(host), 1.0)
    assert log.task_success is False  # the hazard was still visible when the oracle was read


# ----------------------------------------------------------------------------- instability (D10 b)

def _unstable(at_s, duration_s=2.1, control=False, entity_pos=None, **report):
    host, runtime = W.make()

    def actions(t):
        if abs(t - at_s) < 1e-9:
            pos = host.robot_state(0).eef_pose.pos if entity_pos == "hand" else entity_pos
            runtime.report_instability(0, "nan_check", entity_pos=pos, **report)
            runtime.report_instability(0, "second_report_is_ignored", entity_is_hazard=True)
        return FakeAction.follow()

    if control:
        log = W.run_episode(host, runtime, None, duration_s, actions=actions, control_id="ctl")
    else:
        log = W.run_episode(host, runtime, W.slide_instance(host), duration_s, actions=actions)
    return log, runtime.episode_info(0)["instability"]


@pytest.mark.parametrize("at_s, kwargs, expected, ambiguous", [
    (0.52, dict(entity_pos="hand"), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),     # in window, near
    (0.52, dict(entity_pos=(1.0, 1.0, 0.0)), InvalidClass.SIM_UNSTABLE, False),       # in window, far
    (2.0, dict(entity_pos="hand"), InvalidClass.SIM_UNSTABLE, False),                 # long after the window
    (1.5, dict(entity_is_hazard=True), InvalidClass.HAZARD_INDUCED_UNSTABLE, False),  # the hazard itself
    (0.6, dict(), InvalidClass.HAZARD_INDUCED_UNSTABLE, True),                        # position unknown
    (0.1, dict(entity_pos="hand"), InvalidClass.SIM_UNSTABLE, False),                 # before the onset
])
def test_instability_attribution(at_s, kwargs, expected, ambiguous):
    log, report = _unstable(at_s, **kwargs)
    assert log.invalid_class is expected and log.attribution_ambiguous is ambiguous
    assert report["kind"] == "nan_check" and report["time_s"] == pytest.approx(at_s)
    if at_s < 0.2:
        assert report["distance_m"] is None  # parked: no hazard position to measure from


def test_instability_in_a_control_episode_is_sim_unstable():
    log, _ = _unstable(0.52, control=True, entity_pos="hand")
    assert log.invalid_class is InvalidClass.SIM_UNSTABLE


# ----------------------------------------------------------------------------------- topple (D10 h)

def _fall_on(top, entities, instance_id):
    host, runtime = W.make()
    inst = W.fall_onto(np.array(top) + np.array([0.0, 0.0, W.HAZARD_R]), instance_id=instance_id)
    log = W.run_episode(host, runtime, inst, 1.0, layout="objects", task_entities=entities)
    return log


def test_knocked_over_object_is_toppled():
    log = _fall_on((0.25, 0.0, 0.1), ["cup", "bowl"], "cup_hit")
    assert log.toppled is True and log.max_tilt_deg == pytest.approx(90.0)
    assert log.first_scene_contact_t_s == pytest.approx(0.5)


def test_pure_yaw_is_not_toppled():
    log = _fall_on((-0.25, 0.0, 0.1), ["bowl"], "bowl_hit")
    assert log.first_scene_contact_t_s == pytest.approx(0.5)
    assert log.toppled is False and log.max_tilt_deg == pytest.approx(0.0, abs=1e-9)


def test_unmonitored_topple_is_none():
    log = _fall_on((0.25, 0.0, 0.1), None, "cup_hit")
    assert log.toppled is None and log.max_tilt_deg is None


def test_tilt_is_measured_from_the_onset_pose():
    host, runtime = W.make()

    def actions(t):
        if abs(t - 0.1) < 1e-9:  # white box: a disturbance tips the cup over before the onset (0.2 s)
            cup = host._worlds[0].objects["cup"]
            cup.quat = np.array(W.TOPPLE_X)
        return FakeAction.follow()

    log = W.run_episode(host, runtime, W.slide_instance(host), 1.0, layout="objects", task_entities=["cup"],
                        actions=actions)
    assert log.max_tilt_deg == pytest.approx(0.0, abs=1e-9) and log.toppled is False


# ------------------------------------------------------------------------------- missing is None, not 0

def test_missing_quantities_are_none_not_zero():
    host, runtime = W.make()
    log = W.run_episode(host, runtime, W.slide_instance(host), 1.0, action=W.away((0.0, 1.0, 0.0)))
    for field in ("first_official_contact_t_s", "partner_class", "v_rel_normal_mps", "impulse_analytic_ns",
                  "engine_peak_force_n", "force_source", "contact_duration_s", "constraint_state", "exposure_low",
                  "toppled", "max_tilt_deg") + PROVENANCE_FIELDS:
        assert getattr(log, field) is None, field
    assert log.kinematic_overlap_count == 0 and log.r1_violation_count == 0
    assert log.hazard_mass_kg == 0.2 and log.released_t_s is not None
    assert log.separation_kind is SeparationKind.GEOMETRIC and log.min_separation_m > 0
    assert HazardEventLog.from_dict(log.to_dict()).to_dict() == log.to_dict()


def test_surrogate_separation_without_min_distance():
    host, runtime = W.make(host_kwargs={"disable": ("min_distance",)})
    log = W.run_episode(host, runtime, W.slide_instance(host), 1.0)
    assert log.separation_kind is SeparationKind.SURROGATE
    assert log.min_separation_m == pytest.approx(W.HAZARD_R + W.HAND_R, abs=1e-3)


def test_kinematic_overlap_events_are_kept():
    host, runtime = W.make(mutations=("bad_standoff",))
    W.run_episode(host, runtime, W.slide_instance(host), 1.0)
    kinds = [e.kind for e in runtime.contact_events(0)]
    assert kinds[0] is EventKind.KINEMATIC_OVERLAP and EventKind.DYNAMIC in kinds
