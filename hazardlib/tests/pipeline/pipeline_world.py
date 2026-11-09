"""FakeHost world, hosts, policies and helpers for the T06 pipeline tests.

Every number here is test data for a toy world (table at z = 0, sphere hand of radius 0.03 m, sphere hazards
of radius 0.03 m, so the standoff is 0.06 m), not a pre-registered threshold. The resolved configuration the
pipeline reads lives in ``fixture_configs/`` next to this file; the repository's ``configs/hazard`` stays
PLACEHOLDER.

Tasks (one layout ``L0`` each, no task objects):

* ``pick``: EEF at z = 0.3 m, +x at 0.5 m/s to the grasp point x = 0.3 (t = 0.6 s), then +y at 0.5 m/s to the
  goal (t = 1.2 s). The gripper closes (and reports ``held_object = cube``) once x >= 0.299.
* ``low``: EEF at z = 0.05 m (low enough for slides on the table), +x at 0.5 m/s from x = -0.2 to the goal
  x = 0.3 (t = 1.0 s); the gripper stays open.

Hosts are built by :func:`make_host` (also the ``hazardctl --host`` factory): ``GripperHost`` stops
instantly on ``hold``; with ``decel_mps2`` the hold brakes at that deceleration (finite T_stop), and
``censor`` makes the hold be ignored always or on odd seeds (censored T_stop repetitions).

Freezing needs a real conformance report whose ``bd_hash`` is the pipeline host's (D13 a):
:func:`conformance_target` runs the suite on the same host class and timebase in the suite's own world.
"""
import copy
import dataclasses
import functools
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from hazardlib.calibrate import TStopProtocolSettings, run_t_stop_protocol, write_calibration
from hazardlib.conformance import (ConformanceConfig, FakeAction, FakeHost, FakeSphere, FakeTask, FakeWorldConfig,
                                   SuiteSession, make_fake_target, run_suite)
from hazardlib.driving import EpisodeSession
from hazardlib.instantiate import InstantiateSettings, instantiate
from hazardlib.gates import GateSettings
from hazardlib.kinematics import SampledPath, build_hit_path, hook_times
from hazardlib.probe import ProbeSettings, build_task_context, record_probe_trace
from hazardlib.runtime import RuntimeConfig
from hazardlib.spec import BenchmarkDescriptor, HazardInstance, HazardSpec

HERE = Path(__file__).resolve().parent
FIXTURE_CONFIG_DIR = HERE / "fixture_configs"
REPO_CONFIG_DIR = HERE.parents[2] / "configs" / "hazard"

HAND_R = 0.03
HAZARD_R = 0.03
STANDOFF = HAND_R + HAZARD_R
PHYSICS_DT = 0.002
SUBSTEPS = 10
CONTROL_DT = PHYSICS_DT * SUBSTEPS
GRAVITY = 9.81
GRASP_X = 0.3
OPEN_M, CLOSED_M = 0.08, 0.0
HELD = "cube"
WORKSPACE = {"min_m": [-0.6, -0.6, 0.0], "max_m": [0.8, 0.8, 1.0]}
TABLE_SURFACE = {"name": "table", "min_m": [-1.0, -1.0, 0.0], "max_m": [1.0, 1.0, 0.0], "source": "fakehost_config"}
STATIC_ENTITIES = {"goal": {"name": "goal_marker", "source": "oracle_config"}}
PROBE_HORIZON_S = 3.0
LAYOUT = "L0"
DECEL_MPS2 = 2.5           # the braking host of the end-to-end chain: T_stop = 0.5 m/s / 2.5 m/s^2 = 0.2 s


def tasks():
    pick = FakeTask((0.0, 0.6, 1.2, 30.0),
                    ((0.0, 0.0, 0.3), (GRASP_X, 0.0, 0.3), (GRASP_X, 0.3, 0.3), (GRASP_X, 0.3, 0.3)),
                    (GRASP_X, 0.3, 0.3), 0.002, {LAYOUT: ()})
    low = FakeTask((0.0, 1.0, 30.0), ((-0.2, 0.0, 0.05), (0.3, 0.0, 0.05), (0.3, 0.0, 0.05)),
                   (0.3, 0.0, 0.05), 0.002, {LAYOUT: ()})
    return {"pick": pick, "low": low}


def world_config(**overrides):
    values = dict(
        num_envs=2, physics_dt_s=PHYSICS_DT, substeps_per_action=SUBSTEPS, gravity_mps2=GRAVITY, restitution=0.5,
        table_height_m=0.0, table_min_xy_m=(-1.0, -1.0), table_max_xy_m=(1.0, 1.0), park_pos_m=(5.0, 5.0, -5.0),
        robot_bodies=(FakeSphere("robot0_hand", (0.0, 0.0, 0.0), HAND_R),
                      FakeSphere("robot0_link", (0.0, -0.15, 0.1), 0.03),
                      FakeSphere("robot0_camera", (0.0, 0.15, 0.1), 0.02)),
        held_object=None, held_bodies=(), report_held_object_bodies=True,
        robot_include=("prefix:robot0_", "held_object"), robot_exclude=("substr:camera",),
        eef_contact_radius_m=HAND_R, robot_point_spacing_m=None, image_px=32, image_min_xy_m=(-0.5, -0.5), image_max_xy_m=(0.5, 0.5),
        max_contacts_per_step=8, tasks=tasks())
    values.update(overrides)
    return FakeWorldConfig(**values)


# ------------------------------------------------------------------------------------------------ hosts

class GripperHost(FakeHost):
    """FakeHost that reports a gripper opening and the held object on the ``pick`` task."""

    def describe(self):
        data = super().describe().to_dict()
        data["state_access"]["gripper"] = True
        return BenchmarkDescriptor.from_dict(data, strict=True)

    def robot_state(self, env):
        state = super().robot_state(env)
        world = self._ready_world(env)
        closed = world.task_id == "pick" and float(state.eef_pose.pos[0]) >= GRASP_X - 1e-3
        return dataclasses.replace(state, gripper_opening_m=CLOSED_M if closed else OPEN_M,
                                   held_object=HELD if closed else None)


class BrakingHost(GripperHost):
    """``hold`` decelerates the EEF at ``decel_mps2`` instead of stopping it; ``censor`` = ``"always"`` /
    ``"odd_seeds"`` ignores the hold (keeps the velocity) always / on odd environment seeds."""

    def __init__(self, config, decel_mps2, censor=None, **kwargs):
        super().__init__(config, **kwargs)
        self.decel_mps2 = float(decel_mps2)
        self.censor = censor

    def _ignores_hold(self, world):
        return self.censor == "always" or (self.censor == "odd_seeds" and world.seed % 2 == 1)

    def advance_physics(self, env, action):
        if action.mode == "hold":
            world = self._ready_world(env)
            velocity = np.array(world.eef_vel)
            speed = float(np.linalg.norm(velocity))
            if self._ignores_hold(world):
                action = FakeAction.velocity(velocity) if speed > 0 else action
            elif speed > 0:
                slower = max(0.0, speed - self.decel_mps2 * self.config.physics_dt_s)
                action = FakeAction.velocity(velocity * (slower / speed)) if slower > 0 else action
        super().advance_physics(env, action)


def make_host(decel_mps2=None, censor=None, ccd=True, disable=(), **world):
    config = world_config(**world)
    if decel_mps2 is None:
        return GripperHost(config, ccd=ccd, disable=tuple(disable))
    return BrakingHost(config, decel_mps2, censor=censor, ccd=ccd, disable=tuple(disable))


# --------------------------------------------------------------------------------------------- policies

class PathFollower:
    """Scripted reference: velocity commands that put the EEF on the task's nominal path at its own progress
    clock (advanced only when ``act`` is called, so a held run resumes where it stopped)."""

    def __init__(self, host, name="follower", kind="scripted", time_scale=1.0, under_test=False):
        self.host, self.name, self.kind = host, name, kind
        self.time_scale, self.under_test = float(time_scale), bool(under_test)

    def reset(self, task_id, layout_id, seed):
        self.path = self.host.nominal_eef_path(task_id)
        self.progress = 0.0

    def act(self, observation):
        self.progress += CONTROL_DT * self.time_scale
        target = self.path.pos(min(self.progress, self.path.t_end_s))
        return FakeAction.velocity((target - observation["state"][0:3]) / CONTROL_DT)


class DemoReplay:
    """Replays a demonstration recorded at the control rate (one waypoint per control step)."""

    kind = "demo_replay"

    def __init__(self, host, name="demo", under_test=False):
        self.host, self.name, self.under_test = host, name, bool(under_test)

    def reset(self, task_id, layout_id, seed):
        path = self.host.nominal_eef_path(task_id)
        n = int(round(path.t_end_s / CONTROL_DT))
        self.frames = path.pos(np.arange(1, n + 1) * CONTROL_DT)
        self.index = 0

    def act(self, observation):
        target = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return FakeAction.velocity((target - observation["state"][0:3]) / CONTROL_DT)


class Stayer:
    """Never moves (a reference that fails the baseline, or a robot far from the hit point)."""

    def __init__(self, host, name="stayer", kind="scripted", under_test=False):
        self.host, self.name, self.kind, self.under_test = host, name, kind, bool(under_test)

    def reset(self, task_id, layout_id, seed):
        pass

    def act(self, observation):
        return FakeAction.hold()


class PassThroughAvoider:
    """A G2 'avoider' that changes nothing (G2 then fails wherever the nominal run is hit)."""

    def __init__(self, host):
        self.host = host

    def filter_action(self, action, robot_state, hazard_state):
        return action


def make_follower(host, **kwargs):
    return PathFollower(host, **kwargs)


def make_demo(host, **kwargs):
    return DemoReplay(host, **kwargs)


def make_stayer(host, **kwargs):
    return Stayer(host, **kwargs)


def make_pass_avoider(host):
    return PassThroughAvoider(host)


# --------------------------------------------------------------------------------------------- settings

def runtime_cfg(config_dir=FIXTURE_CONFIG_DIR):
    return RuntimeConfig.load(config_dir)


def probe_settings(config_dir=FIXTURE_CONFIG_DIR):
    return ProbeSettings.load(config_dir)


def instantiate_settings(config_dir=FIXTURE_CONFIG_DIR):
    return InstantiateSettings.load(config_dir)


def gate_settings(config_dir=FIXTURE_CONFIG_DIR):
    return GateSettings.load(config_dir)


def session(host):
    return EpisodeSession(host, runtime_cfg())


def copy_fixture_configs(target):
    target = Path(target)
    shutil.copytree(str(FIXTURE_CONFIG_DIR), str(target))
    return target


def edit_yaml(path, edit):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    edit(data)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return Path(path)


# ------------------------------------------------------------------- conformance report, calibration (D13 a, b)

def assemble_conformance_config_dir(target):
    """One directory for the conformance suite and the pipeline: the resolved fixture configs plus the
    repository's ``conformance.yaml`` and ``fakehost.yaml``, copied at test time (T05 owns their keys, so they
    are never stored next to the fixtures)."""
    target = copy_fixture_configs(target)
    for name in ("conformance", "fakehost"):
        shutil.copy(str(REPO_CONFIG_DIR / ("%s.yaml" % name)), str(target))
    return target


def conformance_target(config_dir, decel_mps2=None, ccd=True, robot_point_spacing_m=None):
    """The suite's FakeHost world (``fakehost.yaml``: the suite's scenario needs its tasks and geometry) on this
    module's host class and timebase, so that its BD, hence ``bd_hash``, equals that of
    ``make_host(decel_mps2, ccd=ccd, robot_point_spacing_m=...)``. The BD records the timebase, gravity, CCD,
    environments, contact-body rules, EEF contact radius, robot point spacing (D15 a), state access and optional
    methods, not the tasks or the scene geometry."""
    cls, kwargs = (GripperHost, {}) if decel_mps2 is None else (BrakingHost, {"decel_mps2": decel_mps2})
    overrides = {"substeps_per_action": SUBSTEPS, "robot_point_spacing_m": robot_point_spacing_m}
    return make_fake_target(ccd=ccd, world_overrides=overrides, host_cls=cls, host_kwargs=kwargs,
                            config_dir=config_dir)


def conformance_report(config_dir, **target):
    """A real report: C1-C13 run on :func:`conformance_target` with the settings in ``config_dir``."""
    return run_suite(SuiteSession(conformance_target(config_dir, **target), ConformanceConfig.load(config_dir)))


def write_calibration_file(path, task="pick", decel_mps2=DECEL_MPS2):
    """``hazardctl calibrate`` on the braking host, as a function."""
    host = make_host(decel_mps2=decel_mps2)
    result = run_t_stop_protocol(session(host), PathFollower(host), task_context(task),
                                 TStopProtocolSettings.load(FIXTURE_CONFIG_DIR), 0, 0)
    write_calibration(str(path), result)
    return Path(path)


# Session fixtures, imported by the test modules that freeze (a second conftest.py would shadow the
# ``conftest`` module that tests/test_spec_objects.py imports). Tests get copies and write under their tmp_path.

@pytest.fixture(scope="session")
def conformance_dir(tmp_path_factory):
    return assemble_conformance_config_dir(tmp_path_factory.mktemp("conformance") / "configs")


@pytest.fixture(scope="session")
def conformance_reports(conformance_dir):
    """``get(**target)``: a fresh copy of the report of ``conformance_target(conformance_dir, **target)``."""
    cache = {}

    def get(**target):
        key = tuple(sorted(target.items()))
        if key not in cache:
            cache[key] = conformance_report(conformance_dir, **target)
        return copy.deepcopy(cache[key])

    return get


@pytest.fixture(scope="session")
def calibration_file(tmp_path_factory):
    return write_calibration_file(tmp_path_factory.mktemp("calibration") / "calibration.yaml")


# ------------------------------------------------------------------------------------------------ probes

def probe_policies(host):
    """Two source kinds; the time-warped third run checks the per-phase median (it is the outlier)."""
    return [(PathFollower(host, name="follower"), 0), (DemoReplay(host, name="demo"), 1),
            (PathFollower(host, name="follower_fast", time_scale=1.25), 2)]


def record_traces(host, task, policies=None, layout=LAYOUT):
    sess = session(host)
    return [record_probe_trace(sess, policy, task, layout, seed, PROBE_HORIZON_S, 0)
            for policy, seed in (policies or probe_policies(host))]


@functools.lru_cache(maxsize=None)
def task_context(task):
    host = make_host()
    return build_task_context(record_traces(host, task), probe_settings(), static_entities=STATIC_ENTITIES,
                              declared_workspace=WORKSPACE)


# ------------------------------------------------------------------------------------------------- specs

def _agent(radius):
    return {"kind": "object", "shape": "sphere", "half_extents_m": [radius], "mass_kg": 0.2}


def _spec(spec_id, family, primitive, params, intensity, phase, frac, radius, offset):
    return HazardSpec.from_dict({
        "spec_version": "1.0", "id": spec_id, "family": family, "agent": _agent(radius),
        "motion": {"primitive": primitive, "params": params},
        "exposure": {"anchor": "nominal_ee_path", "offset_m": list(offset),
                     "anchor_phase": {"phase": phase, "frac": frac}},
        "timing": {"mode": "nominal_time"}, "intensity": intensity, "release": {"mode": "dynamic_release"},
    })


def fall_spec(tau=0.2, phase="approach", frac=0.5, spec_id="fall", radius=HAZARD_R, offset=(0.0, 0.0, 0.0)):
    return _spec(spec_id, "intrusion.fall", "free_fall", {}, {"tau_s": tau}, phase, frac, radius, offset)


def slide_spec(tau=0.3, speed=0.5, direction=(0.0, 1.0, 0.0), phase="approach", frac=0.5, spec_id="slide",
               radius=HAZARD_R, offset=(0.0, 0.0, 0.0)):
    params = {} if direction is None else {"direction": list(direction)}
    return _spec(spec_id, "intrusion.slide", "slide", params, {"tau_s": tau, "approach_speed_mps": speed}, phase,
                 frac, radius, offset)


def reach_spec(tau=0.3, speed=0.6, direction=(0.0, 1.0, 0.0), phase="approach", frac=0.75, spec_id="reach",
               radius=HAZARD_R, profile="min_jerk", offset=(0.0, 0.0, 0.0)):
    params = {"speed_profile": profile}
    if direction is not None:
        params["direction"] = list(direction)
    return _spec(spec_id, "intrusion.reach", "reach", params, {"tau_s": tau, "approach_speed_mps": speed}, phase,
                 frac, radius, offset)


def build(spec, task, seed=0, host=None, settings=None, tc=None):
    host = host or make_host()
    return instantiate(spec, tc or task_context(task), seed, host.describe(), settings or instantiate_settings())


def tc_map(*tasks_):
    return {(task_context(t).task_id, task_context(t).layout_id): task_context(t) for t in tasks_}


def hit_params(instance):
    """The ``build_hit_path`` parameters of a builder instance, recovered from what it records."""
    prov = instance.provenance
    if instance.family == "intrusion.fall":
        return {"g": prov["mapping"]["gravity_mps2"]}
    direction = (-np.array(prov["contact_normal"])).tolist()
    if instance.family == "intrusion.slide":
        delta = np.array(instance.keyframes.pos[-1]) - np.array(instance.keyframes.pos[0])
        return {"direction": (delta / np.linalg.norm(delta)).tolist(), "speed_mps": prov["mapping"]["speed_mps"],
                "release_window_s": prov["release_window_s"]}
    m = prov["mapping"]
    return {"direction": direction, "reach_distance_m": m["reach_distance_m"], "overshoot_m": m["overshoot_m"],
            "speed_profile": m["speed_profile"], "release_window_s": prov["release_window_s"]}


def analytic_plan(instance, tc, hit_point=None):
    nominal = SampledPath.from_pose_sequence(tc.nominal_ee_path)
    point = instance.nominal_hit_point if hit_point is None else hit_point
    return build_hit_path(instance.family, nominal, instance.nominal_hit_time_s, point, instance.provenance["tau_s"],
                          hit_params(instance))


def without_standoff(instance, tc):
    """The same instance aimed at the EEF reference point itself (the standoff ignored, D1 violated)."""
    plan = analytic_plan(instance, tc, hit_point=instance.provenance["anchor_point"])
    dt = instance.provenance["keyframes"]["dt_s"]
    times = np.unique(np.concatenate([hook_times(plan.path.t_start_s, plan.path.t_end_s, dt),
                                      [plan.nominal_hit_time_s]]))
    data = instance.to_dict()
    data.update({"instance_id": instance.instance_id + ".no_standoff",
                 "keyframes": {"time_s": times.tolist(), "pos": plan.path.pos(times).tolist(),
                               "quat": [[1.0, 0.0, 0.0, 0.0]] * times.size},
                 "nominal_hit_point": plan.nominal_hit_point.tolist(), "onset_time_s": plan.onset_time_s,
                 "release_time_s": plan.release_time_s,
                 "release_velocity": {"lin_mps": plan.release_velocity.tolist(), "ang_radps": [0.0, 0.0, 0.0]},
                 "provenance": dict(data["provenance"], standoff_m=0.0)})
    return HazardInstance.from_dict(data)


def fall_height(tau, g=GRAVITY):
    return 0.5 * g * tau * tau


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))))))
