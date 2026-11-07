"""Shared FakeHost worlds, settings and drivers for the runtime tests.

Every number here is test data (the YAML thresholds are still PLACEHOLDER), chosen so the analytic answers
are easy to write down: table at z = 0, hand sphere at the EEF, hazards are spheres.
"""
import math

import numpy as np

from hazardlib.conformance import FakeAction, FakeHost, FakeSphere, FakeTask, FakeTaskObject, FakeWorldConfig
from hazardlib.kinematics import FAMILY_FALL, SampledPath, build_hit_path, hook_times
from hazardlib.runtime import HazardRuntime, Provenance, RuntimeConfig
from hazardlib.spec import AgentSpec, HazardInstance

HAND_R = 0.03          # radius of the hand sphere = the BD's eef_contact_radius_m
HAZARD_R = 0.02
EEF_START = (0.0, 0.0, 0.3)
PARK = (5.0, 5.0, -5.0)
WINDOW_S = 0.05        # release_velocity_window_s
TOPPLE_X = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)    # 90 deg about world x
YAW_Z = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))       # 90 deg about world z
SLIDE_START = (-0.35, 0.0, 0.3)

# Test tolerances. The conformance keys they stand for are still PLACEHOLDER (T05 / pilot); the prompt
# forbids reading PLACEHOLDER in tests, so the literals live here, named after their YAML key.
TOL = {
    "c13_nominal_hit.hit_time_abs_tol_s": 1e-6,          # exact for FakeHost slide / fall; the reach test uses dt
    "c5_release_gravity.release_velocity_abs_tol_mps": 1e-9,
    "c5_release_gravity.free_fall_time_rel_tol": 1e-6,
    "c6_relative_velocity.normal_speed_rel_tol": 1e-4,   # gravity acts during the last physics step
}


def runtime_config(**overrides):
    values = dict(max_step_ratio=0.5, release_velocity_window_s=WINDOW_S, restitution_e=0.0, gap_tolerance_s=0.02,
                  t_q_s=0.1, opposing_normals_tol_rad=0.3, tilt_threshold_deg=30.0,
                  hazard_induced_max_distance_m=0.3, attribution_window_s=0.5)
    values.update(overrides)
    return RuntimeConfig(**values)


def sphere_agent(radius=HAZARD_R, mass=0.2):
    return AgentSpec.from_dict({"kind": "object", "shape": "sphere", "half_extents_m": [radius], "mass_kg": mass})


def _objects():
    cup = FakeTaskObject(FakeSphere("cup", (0.25, 0.0, 0.05), 0.05), TOPPLE_X)
    bowl = FakeTaskObject(FakeSphere("bowl", (-0.25, 0.0, 0.05), 0.05), YAW_Z)
    return (cup, bowl)


def tasks():
    hold_still = ((0.0, 20.0), (EEF_START, EEF_START))
    to_goal = ((0.0, 2.0, 20.0), (EEF_START, (0.4, 0.0, 0.3), (0.4, 0.0, 0.3)))
    # low hand that sweeps in over x = 0.1 just after a hazard dropped there has bounced off the table
    sweep_in = ((0.0, 0.4, 0.55, 20.0), ((0.3, 0.0, 0.11), (0.3, 0.0, 0.11), (0.1, 0.0, 0.11), (0.1, 0.0, 0.11)))
    # hand parked where the default slide instance (hit 0.5 s, tau 0.3 s, 1 m/s along +x) starts
    blocker = ((0.0, 20.0), (SLIDE_START, SLIDE_START))
    layouts = {"empty": (), "objects": _objects()}
    return {
        "static": FakeTask(hold_still[0], hold_still[1], EEF_START, 0.01, layouts),
        "blocker": FakeTask(blocker[0], blocker[1], SLIDE_START, 0.01, layouts),
        "to_goal": FakeTask(to_goal[0], to_goal[1], (0.4, 0.0, 0.3), 0.01, layouts),
        "sweep_in": FakeTask(sweep_in[0], sweep_in[1], (0.1, 0.0, 0.11), 0.01, layouts),
    }


def world_config(**overrides):
    values = dict(
        num_envs=2, physics_dt_s=0.002, substeps_per_action=10, gravity_mps2=9.81, restitution=0.5,
        table_height_m=0.0, table_min_xy_m=(-1.0, -1.0), table_max_xy_m=(1.0, 1.0), park_pos_m=PARK,
        robot_bodies=(FakeSphere("robot0_hand", (0.0, 0.0, 0.0), HAND_R),
                      FakeSphere("robot0_link", (0.0, -0.15, 0.1), 0.03),
                      FakeSphere("robot0_camera", (0.0, 0.15, 0.1), 0.02)),
        held_object=None, held_bodies=(), report_held_object_bodies=True,
        robot_include=("prefix:robot0_", "held_object"), robot_exclude=("substr:camera",),
        eef_contact_radius_m=HAND_R, image_px=32, image_min_xy_m=(-0.5, -0.5), image_max_xy_m=(0.5, 0.5),
        max_contacts_per_step=8, tasks=tasks())
    values.update(overrides)
    return FakeWorldConfig(**values)


def make(mutations=(), cfg=None, provenance=None, host_kwargs=None, **world):
    host = FakeHost(world_config(**world), mutations=mutations, **(host_kwargs or {}))
    runtime = HazardRuntime(host, cfg or runtime_config(), provenance)
    return host, runtime


def slide_instance(host, t_hit=0.5, tau=0.3, speed=1.0, direction=(1.0, 0.0, 0.0), task="static",
                   instance_id="slide_0", keyframe_dt=0.01, agent=None):
    params = {"direction": list(direction), "speed_mps": speed, "release_window_s": WINDOW_S}
    return host.nominal_hit_instance(task, instance_id, "slide_spec", "intrusion.slide", agent or sphere_agent(),
                                     t_hit, tau, params, keyframe_dt)


def fall_instance(host, t_hit=0.5, tau=0.3, task="static", instance_id="fall_0", g=None, keyframe_dt=0.01):
    params = {"g": host.describe().gravity_mps2 if g is None else g}
    return host.nominal_hit_instance(task, instance_id, "fall_spec", "intrusion.fall", sphere_agent(), t_hit,
                                     tau, params, keyframe_dt)


def reach_instance(host, t_hit=0.5, tau=0.3, direction=(1.0, 0.0, 0.0), task="static", instance_id="reach_0",
                   keyframe_dt=0.01):
    params = {"direction": list(direction), "reach_distance_m": 0.3, "overshoot_m": 0.1,
              "speed_profile": "min_jerk", "release_window_s": WINDOW_S}
    return host.nominal_hit_instance(task, instance_id, "reach_spec", "intrusion.reach", sphere_agent(), t_hit,
                                     tau, params, keyframe_dt)


def instance_from_plan(plan, instance_id, spec_id, agent, keyframe_dt=0.01):
    """The same layout as ``FakeHost.nominal_hit_instance`` for a plan solved by the test itself."""
    times = hook_times(plan.path.t_start_s, plan.path.t_end_s, keyframe_dt)
    return HazardInstance.from_dict({
        "instance_id": instance_id, "spec_id": spec_id, "family": plan.family, "agent": agent.to_dict(),
        "keyframes": {"time_s": times.tolist(), "pos": plan.path.pos(times).tolist(),
                      "quat": [[1.0, 0.0, 0.0, 0.0]] * times.size},
        "nominal_hit_point": plan.nominal_hit_point.tolist(), "nominal_hit_time_s": plan.nominal_hit_time_s,
        "onset_time_s": plan.onset_time_s, "release_time_s": plan.release_time_s,
        "release_velocity": {"lin_mps": plan.release_velocity.tolist(), "ang_radps": [0.0, 0.0, 0.0]},
        "certificate": None, "provenance": {"builder": "runtime_world.instance_from_plan"},
    })


def fall_onto(centroid_at_hit, t_hit=0.5, tau=0.3, g=9.81, instance_id="fall_point"):
    """A fall whose centroid reaches ``centroid_at_hit`` at ``t_hit`` (e.g. resting height on the table)."""
    point = np.array(centroid_at_hit, dtype=float)
    nominal = SampledPath([0.0, 20.0], [point, point], None)
    plan = build_hit_path(FAMILY_FALL, nominal, t_hit, point, tau, {"g": g})
    return instance_from_plan(plan, instance_id, "fall_spec", sphere_agent())


def table_height_centroid(x, y=0.0):
    return (x, y, HAZARD_R)


def run_episode(host, runtime, instance, duration_s, env=0, action=None, actions=None, task="static",
                layout="empty", task_entities=None, control_id=None, control_onset_s=None, seed=0):
    """arm -> reset_episode -> control steps -> end_episode. ``actions(t)`` may switch the action by the
    episode time at the start of each control step."""
    runtime.arm(env, instance, task_entities=task_entities, control_id=control_id, control_onset_s=control_onset_s)
    host.reset_episode(env, task, layout, seed)
    control_dt = host.config.physics_dt_s * host.config.substeps_per_action
    for k in range(int(round(duration_s / control_dt))):
        step_action = actions(k * control_dt) if actions is not None else (action or FakeAction.follow())
        host.step(env, step_action)
    return runtime.end_episode(env)


def provenance():
    return Provenance(bd_hash="a" * 64, adapter_version="fakehost-test", manifest_hash="b" * 64,
                      benchmark_commits={"fakehost": "builtin"})


def away(velocity=(0.0, 1.0, 0.0)):
    return FakeAction.velocity(velocity)


def arrays_equal(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
