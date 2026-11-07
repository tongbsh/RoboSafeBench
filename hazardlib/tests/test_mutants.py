"""Mutation tests of the conformance suite (T05).

The suite is only as good as the defects it catches. Each FakeHost mutation (``fakehost.MUTATIONS``) must fail
exactly the checks pinned in ``CAUGHT_BY``, each with status ``fail`` (a crash is not a catch), and the clean FakeHost
must pass all of C1-C13. The test-local defects below (FakeHost subclasses and targets) cover host contract items
the ten mutations do not touch; each must fail the checks named for it. Honest variants (no CCD, other gravity,
other time step, one environment, no held object) must pass. On hosts that declare statistical repro, C2 / C7 / C8 /
C12 are judged against C8's per-kind bounds (D13 h): unverified while they are PLACEHOLDER, and the defects in
``STATISTICAL_DEFECTS`` must still fail under bounds the honest noisy host passes.
"""
import copy
import dataclasses
import functools
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardlib.conformance import (MUTATIONS, Check, ConformanceConfig, FakeAction, FakeHost, ReportError,
                                   SuiteSession, load_checks, make_fake_target, read_report, register_host,
                                   registered_hosts, resolve_target, run_suite, validate_report)
from hazardlib.conformance.report import CHECK_KEYS, REPORT_SCHEMA_VERSION, TOP_LEVEL_KEYS, write_report
from hazardlib.host import CapabilityError, PartnerClass
from hazardlib.spec.benchmark_descriptor import BenchmarkDescriptor

CHECK_IDS = ["C%d" % i for i in range(1, 14)]
SRC = Path(__file__).resolve().parents[1] / "src"
CHECKS_DIR = SRC / "hazardlib" / "conformance" / "tests"

# mutation -> every check it fails (pinned; all other checks pass)
CAUGHT_BY = {
    "tunneling": ("C3",),
    "residual_velocity": ("C3", "C4", "C5", "C6", "C13"),
    "state_leak": ("C2", "C7"),
    "wrong_dt": ("C1",),
    "no_rollback": ("C3",),
    "park_interacts": ("C2", "C9", "C11"),
    "oracle_depends_on_hazard": ("C7",),
    "bad_standoff": ("C13",),
    "hook_dependent_release": ("C3", "C4", "C5", "C6"),
    "post_impact_vn": ("C3", "C4", "C6", "C11"),
}
# mutation -> the check of 01 §5 whose claim the defect breaks (must be among the catches)
INTENDED = {
    "tunneling": "C3", "residual_velocity": "C3", "state_leak": "C2", "wrong_dt": "C1", "no_rollback": "C3",
    "park_interacts": "C2", "oracle_depends_on_hazard": "C7", "bad_standoff": "C13",
    "hook_dependent_release": "C5", "post_impact_vn": "C6",
}


@functools.lru_cache(maxsize=None)
def _report(mutation):
    target = make_fake_target(() if mutation == "clean" else (mutation,))
    return run_suite(SuiteSession(target))


def _statuses(report):
    return {c["id"]: c["status"] for c in report["checks"]}


def _run(target, select=None, config=None):
    return run_suite(SuiteSession(target, config), select)


def _config_with(section, key, value):
    base = ConformanceConfig.load()
    files = copy.deepcopy(base.files)
    files["conformance"][section][key] = value
    return ConformanceConfig(files, base.source)


# --------------------------------------------------------------------------------------------- clean and table

def test_the_clean_fakehost_passes_every_check():
    report = _report("clean")
    assert [c["id"] for c in report["checks"]] == CHECK_IDS
    assert _statuses(report) == {cid: "pass" for cid in CHECK_IDS}, [
        (c["id"], c["reason"]) for c in report["checks"] if c["status"] != "pass"]
    assert report["verdict"] == "pass"


def test_the_mutation_table_covers_every_fakehost_mutation():
    assert set(CAUGHT_BY) == set(MUTATIONS) == set(INTENDED)
    for mutation, checks in CAUGHT_BY.items():
        assert checks and INTENDED[mutation] in checks, mutation


@pytest.mark.parametrize("mutation", sorted(CAUGHT_BY))
def test_each_mutation_fails_exactly_its_pinned_checks(mutation):
    statuses = _statuses(_report(mutation))
    failing = {cid for cid, status in statuses.items() if status != "pass"}
    assert failing == set(CAUGHT_BY[mutation]), {cid: statuses[cid] for cid in failing}
    assert all(statuses[cid] == "fail" for cid in failing), statuses
    assert _report(mutation)["verdict"] == "fail"


def test_every_check_catches_at_least_one_defect_or_is_covered_below():
    caught = {cid for checks in CAUGHT_BY.values() for cid in checks}
    local = {cid for _, _, checks in LOCAL_DEFECTS for cid in checks}
    assert caught | local == set(CHECK_IDS)


# --------------------------------------------------------------------------------------- test-local defects

class BeginBeforeReset(FakeHost):
    """The reset hook fires before the scene is reset (01 §3.3 item 2)."""

    def reset_episode(self, env, task_id, layout_id, seed):
        if self._worlds[env].eef is None:
            self.reset_world(env, task_id, layout_id, seed)
        if self._runtime is not None:
            self._runtime.begin_episode(env)
        self.reset_world(env, task_id, layout_id, seed)


class ClockOffset(FakeHost):
    """The reset steps the physics once more after the reset hook (episode zero, item 3)."""

    def reset_episode(self, env, task_id, layout_id, seed):
        super().reset_episode(env, task_id, layout_id, seed)
        self.advance_physics(env, FakeAction.hold())


class XyzwQuaternion(FakeHost):
    """hazard_state returns xyzw instead of wxyz (item 15)."""

    def hazard_state(self, h):
        state = super().hazard_state(h)
        return dataclasses.replace(state, quat=np.roll(state.quat, -1))


class NoKinematicOverlap(FakeHost):
    """Kinematic hazards overlapping the robot are not reported (item 7)."""

    def _overlaps(self, world, hz, t):
        return [c for c in super()._overlaps(world, hz, t)
                if c.partner_class not in (PartnerClass.ROBOT_LINK, PartnerClass.HELD_OBJECT)]


class FlippedNormal(FakeHost):
    """Contact normals point from the hazard to the partner (item 6)."""

    def _obs(self, hz, name, position, normal, t):
        obs = super()._obs(hz, name, position, normal, t)
        return dataclasses.replace(obs, normal=-np.asarray(obs.normal, dtype=float))


class EpisodeRelativeContactClock(FakeHost):
    """Contact times on the episode clock instead of the simulation clock (item 4)."""

    def hazard_contacts(self, h):
        offset = self._world(h.env).reset_steps * self.config.physics_dt_s
        return [dataclasses.replace(c, time_s=c.time_s - offset) for c in super().hazard_contacts(h)]


def _setting(section, key):
    return ConformanceConfig.load().number(section, key)


class GravityMismatch(FakeHost):
    """The BD declares a gravity off the simulated one by twice c5 gravity_rel_tol."""

    def describe(self):
        data = super().describe().to_dict()
        data["gravity_mps2"] = self.config.gravity_mps2 * (1 + 2 * _setting("c5_release_gravity", "gravity_rel_tol"))
        return BenchmarkDescriptor.from_dict(data, strict=True)


class NoisyReset(FakeHost):
    """Same-seed resets differ by sub-millimetre EEF noise while the BD declares exact repro."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(7)

    def reset_world(self, env, task_id, layout_id, seed):
        super().reset_world(env, task_id, layout_id, seed)
        world = self._world(env)
        world.eef = world.eef + self._rng.normal(0.0, 1e-4, 3)


def _declared_statistical(bd):
    data = bd.to_dict()
    data["determinism"]["same_seed_repro"] = "statistical"
    return BenchmarkDescriptor.from_dict(data, strict=True)


class StatisticalNoisyReset(NoisyReset):
    def describe(self):
        return _declared_statistical(super().describe())


class FlakyOracle(FakeHost):
    """task_success is inverted in every other episode: same-seed repeats do not reproduce the oracle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resets = 0

    def reset_world(self, env, task_id, layout_id, seed):
        super().reset_world(env, task_id, layout_id, seed)
        self._resets += 1

    def task_success(self, env):
        return super().task_success(env) != (self._resets % 2 == 1)


class StatisticalFlakyOracle(FlakyOracle):
    def describe(self):
        return _declared_statistical(super().describe())


class StatisticalTwoCameras(StatisticalNoisyReset):
    """A second policy camera, ``wrist_rgb``, brighter by WRIST_OFFSET grey levels in every other episode."""
    WRIST_OFFSET = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resets = 0

    def reset_world(self, env, task_id, layout_id, seed):
        super().reset_world(env, task_id, layout_id, seed)
        self._resets += 1

    def policy_observation(self, env):
        obs = super().policy_observation(env)
        offset = self.WRIST_OFFSET * (self._resets % 2)
        obs["wrist_rgb"] = np.clip(obs["rgb"].astype(np.int64) + offset, 0, 255).astype(np.uint8)
        return obs


class DriftingHold(FakeHost):
    """hold_action creeps at twice the speed that would exhaust c10 drift_m_tol over hold_duration_s."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._speed = 2 * _setting("c10_hold_t_stop", "drift_m_tol") / _setting("c10_hold_t_stop", "hold_duration_s")

    def hold_action(self, env):
        self._world(env)
        return FakeAction.velocity((self._speed, 0.0, 0.0))


class JitteryStop(FakeHost):
    """The hold takes effect after a delay that changes every episode (T_stop not repeatable)."""
    DELAYS_PHYSICS_STEPS = (0, 5, 10, 15, 20)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._episodes = 0
        self._held = {}

    def reset_world(self, env, task_id, layout_id, seed):
        super().reset_world(env, task_id, layout_id, seed)
        self._episodes += 1
        self._held[env] = 0

    def advance_physics(self, env, action):
        if action.mode == "hold":
            self._held[env] += 1
            if self._held[env] <= self.DELAYS_PHYSICS_STEPS[self._episodes % len(self.DELAYS_PHYSICS_STEPS)]:
                action = FakeAction.follow()
        super().advance_physics(env, action)


class SharedRender(FakeHost):
    """Every environment's camera also draws the hazards of the other environments (batch leak)."""

    def _render(self, world):
        image = super()._render(world)
        cfg = self.config
        (x0, y0), (x1, y1) = cfg.image_min_xy_m, cfg.image_max_xy_m
        xs = x0 + (np.arange(cfg.image_px) + 0.5) * (x1 - x0) / cfg.image_px
        ys = y0 + (np.arange(cfg.image_px) + 0.5) * (y1 - y0) / cfg.image_px
        gx, gy = np.meshgrid(xs, ys)
        for other in self._worlds:
            if other is not world:
                for hz in other.hazards.values():
                    if hz.visible:
                        image[(gx - hz.pos[0]) ** 2 + (gy - hz.pos[1]) ** 2 <= hz.radius ** 2] = (220, 0, 0)
        return image


class InvisibleHazard(FakeHost):
    """The policy cameras never show the hazard (render_check still says it is in view)."""

    def _render(self, world):
        saved = {key: hz.visible for key, hz in world.hazards.items()}
        for hz in world.hazards.values():
            hz.visible = False
        try:
            return super()._render(world)
        finally:
            for key, visible in saved.items():
                world.hazards[key].visible = visible


class DampedFall(FakeHost):
    """Free hazards lose velocity at 0.5 /s (undeclared damping, 01 §5 C5)."""
    DAMPING_PER_S = 0.5

    def advance_physics(self, env, action):
        super().advance_physics(env, action)
        for hz in self._world(env).hazards.values():
            if hz.dynamic and not hz.resting and hz.carrier is None:
                hz.vel = hz.vel * (1.0 - self.DAMPING_PER_S * self.config.physics_dt_s)


def _without_fault_injection():
    target = make_fake_target()
    target.inject_engine_fault = None
    return target


def _silent_fault_injection():
    target = make_fake_target()
    target.inject_engine_fault = lambda env, hazard: None
    return target


# name, target factory, checks that must fail (status "fail")
LOCAL_DEFECTS = [
    ("begin_hook_before_reset", lambda: make_fake_target(host_cls=BeginBeforeReset), ("C1",)),
    ("clock_advanced_after_begin", lambda: make_fake_target(host_cls=ClockOffset), ("C1",)),
    ("xyzw_quaternion", lambda: make_fake_target(host_cls=XyzwQuaternion), ("C3",)),
    ("kinematic_overlap_unreported", lambda: make_fake_target(host_cls=NoKinematicOverlap), ("C3", "C11")),
    ("held_bodies_unreported",
     lambda: make_fake_target(world_overrides={"report_held_object_bodies": False}), ("C4",)),
    ("flipped_contact_normal", lambda: make_fake_target(host_cls=FlippedNormal), ("C4", "C6")),
    ("episode_relative_contact_clock", lambda: make_fake_target(host_cls=EpisodeRelativeContactClock), ("C4",)),
    ("no_body_velocities", lambda: make_fake_target(host_kwargs={"report_body_velocities": False}), ("C6",)),
    ("bd_gravity_mismatch", lambda: make_fake_target(host_cls=GravityMismatch), ("C5",)),
    ("undeclared_damping", lambda: make_fake_target(host_cls=DampedFall), ("C5",)),
    ("noisy_reset_declared_exact", lambda: make_fake_target(host_cls=NoisyReset), ("C8",)),
    ("invisible_hazard", lambda: make_fake_target(host_cls=InvisibleHazard), ("C9",)),
    ("drifting_hold", lambda: make_fake_target(host_cls=DriftingHold), ("C10",)),
    ("t_stop_not_repeatable", lambda: make_fake_target(host_cls=JitteryStop), ("C10",)),
    ("no_fault_injection", _without_fault_injection, ("C11",)),
    ("faults_not_reported", _silent_fault_injection, ("C11",)),
    ("render_leaks_across_envs", lambda: make_fake_target(host_cls=SharedRender), ("C12",)),
]


@pytest.mark.parametrize("name,factory,checks", LOCAL_DEFECTS, ids=[d[0] for d in LOCAL_DEFECTS])
def test_each_local_defect_fails_its_checks(name, factory, checks):
    report = _run(factory(), select=list(checks))
    statuses = _statuses(report)
    assert statuses == {cid: "fail" for cid in checks}, [(c["id"], c["status"], c["reason"])
                                                         for c in report["checks"]]
    assert all(c["reason"] for c in report["checks"])


def _config_with_bounds(vector, image):
    """conformance.yaml with both statistical C8 bounds set (state vectors, images in grey levels)."""
    base = ConformanceConfig.load()
    files = copy.deepcopy(base.files)
    files["conformance"]["c8_determinism"].update(statistical_state_vector_max_abs_diff=vector,
                                                  statistical_image_max_abs_diff=image)
    return ConformanceConfig(files, base.source)


def test_c8_statistical_host_is_measured_then_judged_against_the_bound():
    target = make_fake_target(host_cls=StatisticalNoisyReset)
    check = _run(target, select=["C8"])["checks"][0]
    assert check["status"] == "unverified"
    assert "statistical_state_vector_max_abs_diff is not set" in check["reason"]
    magnitude = max(check["measured"]["repeat_max_abs_diff"]["vectors"],
                    check["measured"]["repeat_max_abs_diff"]["images"])
    assert magnitude > 0
    loose = _config_with_bounds(2 * magnitude, 2 * magnitude)
    strict = _config_with_bounds(0.5 * magnitude, 0.5 * magnitude)
    assert _run(make_fake_target(host_cls=StatisticalNoisyReset), ["C8"], loose)["checks"][0]["status"] == "pass"
    assert _run(make_fake_target(host_cls=StatisticalNoisyReset), ["C8"], strict)["checks"][0]["status"] == "fail"


# ------------------------------------------------------- statistical hosts: C2 / C7 / C12 against C8 (D13 h)

STATISTICAL_CHECKS = ["C2", "C7", "C8", "C12"]


def _check(target, check_id, config=None):
    return _run(target, [check_id], config)["checks"][0]


@functools.lru_cache(maxsize=None)
def _statistical_magnitudes(check_id):
    """(state-vector, largest per-camera image) difference a check measures on StatisticalNoisyReset. The noise is
    seeded per host, so every fresh host measures the same values."""
    measured = _check(make_fake_target(host_cls=StatisticalNoisyReset), check_id)["measured"]["statistical"]
    return measured["state_vector_max_abs_diff"], max((measured["images"] or {}).values(), default=0.0)


@pytest.mark.parametrize("check_id", STATISTICAL_CHECKS)
def test_statistical_host_with_placeholder_bounds_is_measured_not_judged(check_id):
    check = _check(make_fake_target(host_cls=StatisticalNoisyReset), check_id)
    vector, _ = _statistical_magnitudes(check_id)
    assert check["status"] == "unverified", check["reason"]
    assert "measured, not judged" in check["reason"] and "statistical_state_vector_max_abs_diff" in check["reason"]
    assert vector > 0 and ("%.3g" % vector) in check["reason"]


@pytest.mark.parametrize("check_id", STATISTICAL_CHECKS)
def test_statistical_host_passes_bounds_at_twice_the_measured_magnitude(check_id):
    vector, image = _statistical_magnitudes(check_id)
    check = _check(make_fake_target(host_cls=StatisticalNoisyReset), check_id, _config_with_bounds(2 * vector,
                                                                                                    2 * image))
    assert check["status"] == "pass", check["reason"]
    assert check["measured"]["statistical"]["state_vector_max_abs_diff"] == vector


@pytest.mark.parametrize("check_id", STATISTICAL_CHECKS)
def test_statistical_host_fails_bounds_at_half_the_measured_magnitude(check_id):
    vector, image = _statistical_magnitudes(check_id)
    check = _check(make_fake_target(host_cls=StatisticalNoisyReset), check_id, _config_with_bounds(0.5 * vector,
                                                                                                    0.5 * image))
    assert check["status"] == "fail" and "above the statistical bound" in check["reason"], check["reason"]


class StatisticalSharedRender(SharedRender, StatisticalNoisyReset):
    pass


# name, target factory, checks that must fail on a statistical host whose bounds the honest StatisticalNoisyReset
# passes (twice its largest magnitude over C2 / C7 / C8 / C12)
STATISTICAL_DEFECTS = [
    ("statistical_state_leak", lambda: make_fake_target(("state_leak",), host_cls=StatisticalNoisyReset),
     ("C2", "C7")),
    ("statistical_oracle_depends_on_hazard",
     lambda: make_fake_target(("oracle_depends_on_hazard",), host_cls=StatisticalNoisyReset), ("C7",)),
    ("statistical_render_leaks_across_envs", lambda: make_fake_target(host_cls=StatisticalSharedRender), ("C12",)),
]


def _honest_statistical_bounds():
    vector = max(_statistical_magnitudes(c)[0] for c in STATISTICAL_CHECKS)
    image = max(_statistical_magnitudes(c)[1] for c in STATISTICAL_CHECKS)
    return 2 * vector, 2 * image


@pytest.mark.parametrize("name,factory,checks", STATISTICAL_DEFECTS, ids=[d[0] for d in STATISTICAL_DEFECTS])
def test_each_statistical_defect_fails_its_checks(name, factory, checks):
    vector_bound, image_bound = _honest_statistical_bounds()
    config = _config_with_bounds(vector_bound, image_bound)
    honest = _run(make_fake_target(host_cls=StatisticalNoisyReset), list(checks), config)
    assert set(_statuses(honest).values()) == {"pass"}, [(c["id"], c["reason"]) for c in honest["checks"]]
    report = _run(factory(), list(checks), config)
    assert _statuses(report) == {cid: "fail" for cid in checks}, [(c["id"], c["status"], c["reason"])
                                                                  for c in report["checks"]]
    if name == "statistical_state_leak":
        for check in report["checks"]:
            assert check["measured"]["statistical"]["state_vector_max_abs_diff"] > 100 * vector_bound


@pytest.mark.parametrize("check_id", ["C2", "C7", "C12"])
def test_statistical_host_does_not_judge_an_oracle_its_repeats_do_not_reproduce(check_id):
    check = _check(make_fake_target(host_cls=StatisticalFlakyOracle), check_id, _config_with_bounds(0.0, 0.0))
    assert check["status"] == "unverified", check["reason"]
    assert "task_success not judged: same-seed repeats (C8) disagree" in check["reason"]
    assert check["measured"]["statistical_oracle"]["repeats_oracle_differs_at_s"]
    assert _check(make_fake_target(host_cls=FlakyOracle), check_id)["status"] == "fail"


@pytest.mark.parametrize("check_id", ["C2", "C8", "C12"])
def test_statistical_host_compares_every_camera_on_its_own(check_id):
    check = _check(make_fake_target(host_cls=StatisticalTwoCameras), check_id,
                   _config_with_bounds(1.0, 0.5 * StatisticalTwoCameras.WRIST_OFFSET))
    images = check["measured"]["statistical"]["images"]
    assert set(images) == {"rgb", "wrist_rgb"} and images["wrist_rgb"] == StatisticalTwoCameras.WRIST_OFFSET
    over = sorted(camera for camera, d in images.items() if d > 0.5 * StatisticalTwoCameras.WRIST_OFFSET)
    assert check["status"] == "fail" and "camera(s) %s differ" % over in check["reason"], check["reason"]


def test_c5_landing_time_alone_catches_undeclared_damping():
    """The landing-time criterion, not only the gravity fit, separates DampedFall with a margin of one tolerance."""
    check = _run(make_fake_target(host_cls=DampedFall), ["C5"])["checks"][0]
    steps = _setting("c5_release_gravity", "free_fall_time_tol_physics_steps")
    assert check["status"] == "fail" and "physics steps" in check["reason"], check["reason"]
    assert abs(check["measured"]["free_fall"]["fall_time_error_physics_steps"]) >= 2 * steps
    step_end_detection = _run(make_fake_target(ccd=False), ["C5"])["checks"][0]["measured"]["free_fall"]
    assert 0 < step_end_detection["fall_time_error_physics_steps"] <= steps


def test_c13_unrepeatable_motion_is_unverified_not_passed():
    target = make_fake_target()
    episodes = []

    def motion(t):
        if t == 0.0:
            episodes.append(t)
        return FakeAction.velocity((0.5 + 0.01 * len(episodes), 0.0, 0.0))

    target.motion_action = motion
    target.nominal_eef_path = None
    report = _run(target, select=["C13"])
    assert report["checks"][0]["status"] == "unverified", report["checks"][0]["reason"]
    assert report["verdict"] == "unverified"


def test_c13_tolerance_below_one_physics_step_fails():
    config = _config_with("c13_nominal_hit", "hit_time_abs_tol_s", 0.001)
    check = _run(make_fake_target(), ["C13"], config)["checks"][0]
    assert check["status"] == "fail" and "below one physics step" in check["reason"]


def test_a_placeholder_tolerance_fails_its_check_and_names_the_key():
    config = _config_with("c3_pose_fidelity", "pose_error_m_tol", "PLACEHOLDER")
    check = _run(make_fake_target(), ["C3"], config)["checks"][0]
    assert check["status"] == "fail"
    assert check["reason"].startswith("settings:") and "c3_pose_fidelity.pose_error_m_tol" in check["reason"]


# ------------------------------------------------------------------------------------------- honest variants

HONEST_VARIANTS = [
    ("no_ccd", lambda: make_fake_target(ccd=False)),
    ("gravity_3", lambda: make_fake_target(world_overrides={"gravity_mps2": 3.0})),
    ("gravity_15", lambda: make_fake_target(world_overrides={"gravity_mps2": 15.0})),
    ("physics_dt_1ms", lambda: make_fake_target(world_overrides={"physics_dt_s": 0.001, "substeps_per_action": 10})),
]


@pytest.mark.parametrize("name,factory", HONEST_VARIANTS, ids=[v[0] for v in HONEST_VARIANTS])
def test_honest_variants_pass_every_check(name, factory):
    report = _run(factory())
    assert _statuses(report) == {cid: "pass" for cid in CHECK_IDS}, [
        (c["id"], c["reason"]) for c in report["checks"] if c["status"] != "pass"]


def test_c5_measures_the_gravity_of_the_host():
    report = _run(make_fake_target(world_overrides={"gravity_mps2": 3.0}), ["C5"])
    fall = report["checks"][0]["measured"]["free_fall"]
    assert fall["g_measured_mps2"] == pytest.approx(3.0, rel=1e-9)
    assert fall["fall_time_measured_s"] == pytest.approx(np.sqrt(2 * fall["height_m"] / 3.0), rel=1e-9)


def test_c12_is_skipped_only_on_a_single_environment_host():
    report = _run(make_fake_target(world_overrides={"num_envs": 1}))
    statuses = _statuses(report)
    assert statuses.pop("C12") == "skip"
    assert set(statuses.values()) == {"pass"}
    c12 = [c for c in report["checks"] if c["id"] == "C12"][0]
    assert "num_envs = 1" in c12["reason"]


class TierBFakeHost(FakeHost):
    def describe(self):
        data = super().describe().to_dict()
        data["contact_tier"] = "B"
        return BenchmarkDescriptor.from_dict(data, strict=True)


class UndeterminedTierFakeHost(FakeHost):
    def describe(self):
        data = super().describe().to_dict()
        data["contact_tier"] = "unknown"
        return BenchmarkDescriptor.from_dict(data)


def test_c6_is_skipped_on_tier_b_with_a_reason():
    report = _run(make_fake_target(host_cls=TierBFakeHost), ["C6"])
    assert report["tier"] == "B"
    assert report["checks"][0]["status"] == "skip" and "Tier B" in report["checks"][0]["reason"]


def test_c6_is_not_skipped_when_the_tier_is_undetermined():
    report = _run(make_fake_target(host_cls=UndeterminedTierFakeHost), ["C6"])
    assert report["tier"] == "undetermined"
    check = report["checks"][0]
    assert check["status"] == "fail" and "capability:" in check["reason"] and "contact_tier" in check["reason"]


def test_only_c6_and_c12_can_skip():
    skippers = {check.check_id for check in load_checks() if check.skip_when is not None}
    assert skippers == {"C6", "C12"}


def test_c4_without_a_held_object_notes_it_and_passes():
    target = make_fake_target()
    target.scenario = dataclasses.replace(target.scenario, held_strike=None)
    check = _run(target, ["C4"])["checks"][0]
    assert check["status"] == "pass"
    assert any("held" in note for note in check["measured"]["notes"])


# ------------------------------------------------------------------------------------------------- settings

def test_every_conformance_setting_is_used_by_a_check():
    report = _report("clean")
    used = {key for c in report["checks"] for key in c["tolerances"]}
    # read only on hosts without CCD and on statistical hosts
    used |= set(_run(make_fake_target(ccd=False), ["C3"])["checks"][0]["tolerances"])
    used |= set(_run(make_fake_target(host_cls=StatisticalNoisyReset), ["C8"])["checks"][0]["tolerances"])
    config = ConformanceConfig.load()
    declared = {"%s.%s" % (section, key) for section, block in config.files["conformance"].items()
                if section not in ("probe_runtime", "probe_calibration") for key in block}
    unused = declared - used
    assert not unused, sorted(unused)
    assert used <= declared, sorted(used - declared)
    probe_runtime = {v["source"].split(":")[1] for v in report["runtime_config"].values()
                     if v["source"].startswith("conformance.yaml:")}
    assert probe_runtime <= {"probe_runtime.%s" % k for k in config.raw("probe_runtime")}


def test_c13_tolerance_is_between_one_physics_step_and_the_bad_standoff_effect():
    tol = ConformanceConfig.load().number("c13_nominal_hit", "hit_time_abs_tol_s")
    clean = _report("clean")
    dt = [c for c in clean["checks"] if c["id"] == "C13"][0]["measured"]["physics_dt_s"]
    bad = [c for c in _report("bad_standoff")["checks"] if c["id"] == "C13"][0]["measured"]["hits"]
    smallest_effect = min(abs(h["residual_s"]) for label, h in bad.items() if label.startswith("target/"))
    worst_clean = max(abs(h["residual_s"]) for h in
                      [c for c in clean["checks"] if c["id"] == "C13"][0]["measured"]["hits"].values())
    assert dt <= tol < smallest_effect and worst_clean < tol


# ----------------------------------------------------------------------------------------------- registry

def _registry_probe_factory():
    return make_fake_target(name="t05-registry-probe")


def _other_factory():
    return make_fake_target()


def test_registry_registers_resolves_and_rejects_conflicts():
    assert "fake" in registered_hosts()
    register_host("t05-registry-probe", _registry_probe_factory)
    register_host("t05-registry-probe", _registry_probe_factory)
    with pytest.raises(ValueError, match="already registered"):
        register_host("t05-registry-probe", _other_factory)
    assert resolve_target("t05-registry-probe").name == "t05-registry-probe"
    with pytest.raises(CapabilityError, match="no-such-host"):
        resolve_target("no-such-host")


def test_registry_imports_host_modules(tmp_path, monkeypatch):
    (tmp_path / "t05_tmp_host.py").write_text(
        "from hazardlib.conformance import make_fake_target, register_host\n"
        "register_host('t05-from-module', lambda: make_fake_target(('wrong_dt',), name='t05-from-module'))\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    target = resolve_target("t05-from-module", modules=["t05_tmp_host"])
    assert target.name == "t05-from-module" and target.adapter_version.endswith("/wrong_dt")


# ------------------------------------------------------------------------------------------------- report

def test_report_layout_is_complete_and_validates():
    report = _report("clean")
    assert tuple(sorted(report)) == tuple(sorted(TOP_LEVEL_KEYS))
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert len(report["bd_hash"]) == 64 and report["adapter_version"] == "fakehost-builtin-1/clean"
    for check in report["checks"]:
        assert tuple(sorted(check)) == tuple(sorted(CHECK_KEYS))
        assert check["tolerances"], check["id"]
    validate_report(report)
    validate_report(_report("park_interacts"))


def test_report_round_trips_through_json(tmp_path):
    path = tmp_path / "report.json"
    write_report(_report("post_impact_vn"), path)
    assert read_report(path) == json.loads(json.dumps(_report("post_impact_vn")))


def _strip_durations(report):
    data = json.loads(json.dumps(report))
    for check in data["checks"]:
        check.pop("duration_s")
    return data


def test_report_is_stable_across_runs():
    again = run_suite(SuiteSession(make_fake_target()))
    assert _strip_durations(again) == _strip_durations(_report("clean"))


@pytest.mark.parametrize("tamper,message", [
    (lambda r: r.update(schema_version="0"), "schema_version"),
    (lambda r: r["summary"].update(fail=3), "summary"),
    (lambda r: r.update(verdict="fail"), "verdict"),
    (lambda r: r["checks"][0].update(status="maybe"), "status"),
    (lambda r: r["checks"][0].update(status="fail", reason=None), "reason"),
    (lambda r: r.pop("bd_hash"), "bd_hash"),
    (lambda r: r.update(config_source="/srv/checkout/configs/hazard"), "config_source"),
    (lambda r: r.update(config_source="~/RoboSafeBench/configs/hazard"), "config_source"),
    (lambda r: r.update(config_source="C:\\RoboSafeBench\\configs\\hazard"), "config_source"),
    (lambda r: r.update(config_source="../elsewhere/configs"), "config_source"),
    (lambda r: r.update(config_source="external:/tmp/configs"), "config_source"),
    (lambda r: r.update(config_source="external:"), "config_source"),
])
def test_report_validation_rejects_inconsistent_reports(tamper, message):
    report = json.loads(json.dumps(_report("clean")))
    tamper(report)
    with pytest.raises(ReportError, match=message):
        validate_report(report)


def _machine_paths():
    repo = Path(__file__).parents[2]
    return {os.path.expanduser("~"), str(repo.absolute()), str(repo.resolve())}


def test_report_json_contains_no_machine_path(tmp_path):
    out = tmp_path / "report.json"
    assert _cli()(["--host", "fake", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert [p for p in _machine_paths() if p in text] == []
    assert read_report(out)["config_source"] == "configs/hazard"


def test_an_external_config_dir_is_named_not_located(tmp_path):
    config_dir = tmp_path / "frozen_configs"
    config_dir.mkdir()
    for name in ("conformance", "runtime", "semantics", "intensity", "fakehost"):
        shutil.copy(str(SRC.parents[1] / "configs" / "hazard" / (name + ".yaml")), str(config_dir))
    session = SuiteSession(make_fake_target(config_dir=config_dir), ConformanceConfig.load(config_dir))
    out = tmp_path / "report.json"
    write_report(run_suite(session, ["C1"]), out)
    text = out.read_text(encoding="utf-8")
    assert read_report(out)["config_source"] == "external:frozen_configs"
    assert [p for p in _machine_paths() | {str(tmp_path)} if p in text] == []


# ---------------------------------------------------------------------------------------------------- CLI

def _cli():
    return importlib.import_module("hazardlib.conformance.__main__").main


def _wrong_dt_factory():
    return make_fake_target(("wrong_dt",), name="t05-cli-wrong-dt")


def _unrepeatable_factory():
    target = make_fake_target(name="t05-cli-unrepeatable")
    target.nominal_eef_path = None
    episodes = []

    def motion(t):
        if t == 0.0:
            episodes.append(t)
        return FakeAction.velocity((0.5 + 0.01 * len(episodes), 0.0, 0.0))

    target.motion_action = motion
    return target


def test_cli_writes_a_valid_report_and_exits_zero_on_the_clean_host(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert _cli()(["--host", "fake", "--out", str(out)]) == 0
    report = read_report(out)
    assert report["verdict"] == "pass" and [c["id"] for c in report["checks"]] == CHECK_IDS
    printed = capsys.readouterr().out
    assert "verdict=PASS" in printed and printed.count(" PASS ") >= 13


def test_cli_exit_codes_for_fail_and_unverified(tmp_path):
    register_host("t05-cli-wrong-dt", _wrong_dt_factory)
    register_host("t05-cli-unrepeatable", _unrepeatable_factory)
    assert _cli()(["--host", "t05-cli-wrong-dt", "--select", "C1,C2"]) == 1
    out = tmp_path / "u.json"
    assert _cli()(["--host", "t05-cli-unrepeatable", "--select", "C13", "--out", str(out)]) == 2
    assert [c["id"] for c in read_report(out)["checks"]] == ["C13"]


def test_load_checks_selects_by_id_and_rejects_unknown_ids():
    assert [c.check_id for c in load_checks(["C5", "C1"])] == ["C1", "C5"]
    assert all(isinstance(c, Check) for c in load_checks())
    with pytest.raises(ValueError, match="C99"):
        load_checks(["C99"])


# ----------------------------------------------------------------------------------- pytest plugin, imports

def _python(args, extra_path=()):
    env_path = [str(SRC)] + [str(p) for p in extra_path]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(env_path))
    return subprocess.run([sys.executable] + args, capture_output=True, text=True, env=env,
                          cwd=str(SRC.parent), timeout=600)


def test_pytest_plugin_runs_the_suite_against_a_named_host(tmp_path):
    out = tmp_path / "pytest_report.json"
    result = _python(["-m", "pytest", str(CHECKS_DIR), "--host=fake", "-q", "-p", "no:cacheprovider",
                      "--conformance-report=%s" % out])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "13 passed" in result.stdout
    assert read_report(out)["verdict"] == "pass"


def test_pytest_plugin_reports_failures_of_a_host_from_a_module(tmp_path):
    (tmp_path / "t05_plugin_host.py").write_text(
        "from hazardlib.conformance import make_fake_target, register_host\n"
        "register_host('t05-plugin-wrong-dt', lambda: make_fake_target(('wrong_dt',)))\n")
    result = _python(["-m", "pytest", str(CHECKS_DIR), "--host=t05-plugin-wrong-dt",
                      "--host-module=t05_plugin_host", "-q", "-p", "no:cacheprovider", "-k", "c01 or c02"],
                     extra_path=[tmp_path])
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 1 passed" in result.stdout and "physics_dt_s=0.004" in result.stdout


def test_running_the_suite_loads_no_simulator_or_dynisafety_module():
    code = ("import sys\n"
            "from hazardlib.conformance import SuiteSession, make_fake_target, run_suite\n"
            "run_suite(SuiteSession(make_fake_target()), ['C1', 'C13'])\n"
            "bad = ('torch', 'scipy', 'mujoco', 'pybullet', 'sapien', 'robosuite', 'dynisafety', 'hazardhosts',\n"
            "       'isaacgym', 'isaacsim', 'omni', 'pytest')\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in bad))\n")
    result = _python(["-c", code])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
