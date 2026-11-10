"""The LIBERO G2 avoider's proposal logic on a synthetic straight-line nominal path (no simulator)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hazardhosts.libero.avoider import NominalPath, NominalPathAvoider, offset_candidates
from hazardlib.config import load_config
from hazardlib.host import HazardState
from hazardlib.runtime import KINEMATIC, RELEASED
from hazardlib.spec.instance import HazardInstance

SETTINGS = load_config("avoider", strict=True, config_dir=Path(__file__).parents[2] / "configs/hazard/pilot/libero")
DT, STEP, N = 0.05, 0.01, 60


def nominal_path():
    """A point robot ("hand", one surface sample at its origin) moving along +x, 1 cm per control step."""
    eye = np.eye(3)
    states = [{"eef_pos": np.array([k * STEP, 0., 1.]), "eef_rot": eye, "held": None,
               "bodies": {"hand": (np.array([k * STEP, 0., 1.]), eye)}} for k in range(N + 1)]
    goals = [(np.array([(k + 1) * STEP, 0., 1.]), eye) for k in range(N)]
    actions = [np.array([STEP / 0.05, 0., 0., 0., 0., 0., -1. if k < 30 else 1.]) for k in range(N)]
    recorder = SimpleNamespace(states=states, goals=goals, taken=actions)
    return NominalPath(recorder, {"hand": np.zeros((1, 3))}, 1)


class Host(object):
    """The controller reaches every goal within one step (a perfect tracker)."""
    control_dt = DT

    def __init__(self):
        self.m = SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0., 0., -9.81])))
        self._gripper_cmd = -1.0
        controller = SimpleNamespace(output_max=np.array([.05, .05, .05, .5, .5, .5]), ee_pos=np.array([0., 0., 1.]),
                                     ee_ori_mat=np.eye(3), update=lambda force=False: None)
        self.env = SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(controller=controller)]))
        self.controller = controller

    def apply(self, action):
        self.controller.ee_pos = self.controller.ee_pos + action[:3] * self.controller.output_max[:3]


def instance(path_x, path_y, release_time):
    """A kinematic hazard that stands still at (path_x, path_y, 1) until its release."""
    return HazardInstance.from_dict({
        "instance_id": "i", "spec_id": "s", "family": "intrusion.reach",
        "agent": {"kind": "object", "shape": "box", "half_extents_m": [.02, .02, .02], "mass_kg": .05},
        "keyframes": {"time_s": [0., release_time], "pos": [[path_x, path_y, 1.]] * 2, "quat": [[1., 0., 0., 0.]] * 2},
        "nominal_hit_point": [path_x, path_y, 1.], "nominal_hit_time_s": release_time, "onset_time_s": 0.,
        "release_time_s": release_time, "release_velocity": {"lin_mps": [0., 0., 0.], "ang_radps": [0., 0., 0.]},
        "certificate": {"g0": True, "g1": None, "g2": None, "label": "uncertified", "notes": ""},
        "provenance": {}})


def state(x, y, z=1., vel=(0., 0., 0.)):
    return HazardState(pos=[x, y, z], quat=[1., 0., 0., 0.], lin_vel_mps=list(vel), ang_vel_radps=[0., 0., 0.])


def make(phase):
    host = Host()
    return host, NominalPathAvoider(host, nominal_path(), SETTINGS, lambda: phase[0])


def test_offset_candidates_start_with_zero_and_prefer_the_shortest_then_upward():
    candidates = offset_candidates(SETTINGS)
    lengths = [float(np.linalg.norm(c)) for c in candidates]
    assert not np.any(candidates[0]) and lengths == sorted(lengths)
    assert candidates[1][2] > 0 and not np.any(candidates[1][:2])


def test_without_a_hazard_the_action_is_the_recorded_one():
    host, avoider = make([None])
    for k in range(N):
        action = avoider.filter_action(None, None, None)
        np.testing.assert_allclose(action[:6], [STEP / 0.05, 0., 0., 0., 0., 0.], atol=1e-12)
        assert action[6] == (-1. if k < 30 else 1.)
        host.apply(action)
    assert avoider.s == N and avoider.summary()["held_steps"] == 0


def test_a_moving_hazard_ahead_stops_the_robot_outside_the_keep_out_and_one_on_top_makes_it_retreat():
    phase = [KINEMATIC]
    host, avoider = make(phase)
    avoider.arm(instance(0.30, 0., 10.))
    for _ in range(40):
        host.apply(avoider.filter_action(None, None, state(0.30, 0.)))
    stop = avoider.s
    # Stopped, and no state within the stop lookahead is inside the keep-out sphere.
    assert stop < 30 and (0.30 - (stop + SETTINGS["stop_lookahead_steps"]) * STEP) >= avoider.radius - STEP
    assert avoider.summary()["retreat_steps"] == 0 and avoider.summary()["held_steps"] > 0
    # The hazard now sits on the robot's current state: go back along the path.
    avoider.arm(instance(stop * STEP, 0., 10.))
    host.apply(avoider.filter_action(None, None, state(stop * STEP, 0.)))
    assert avoider.s == stop - 1 and avoider.summary()["retreat_steps"] == 1


def test_resting_debris_on_the_path_is_passed_with_a_bounded_ramped_offset_that_returns_to_zero():
    phase = [RELEASED]
    host, avoider = make(phase)
    avoider.arm(instance(0.30, 0., 0.5))
    largest, previous = 0.0, np.zeros(3)
    for _ in range(N + 20):
        action = avoider.filter_action(None, None, state(0.30, 0.))
        assert np.linalg.norm(avoider.offset - previous) <= SETTINGS["max_offset_step_m"] + 1e-12
        previous = avoider.offset.copy()
        largest = max(largest, float(np.linalg.norm(avoider.offset)))
        host.apply(action)
        # The tracked configuration never enters the keep-out sphere.
        point = avoider.nominal.states[avoider.s]["bodies"]["hand"][0] + avoider.offset
        assert np.linalg.norm(point - [0.30, 0., 1.]) >= avoider.radius - 1e-9
    assert avoider.s == N and largest >= avoider.radius - 1e-9
    assert not np.any(avoider.offset) and avoider.summary()["blocked_steps"] == 0
    # The gripper command never returns to an earlier one.
    assert action[6] == 1.


def test_debris_that_no_offset_clears_blocks_instead_of_advancing():
    phase = [RELEASED]
    host, avoider = make(phase)
    avoider.arm(instance(0.30, 0., 0.5))
    avoider.radius = 1.0  # larger than every offset candidate
    for _ in range(N):
        host.apply(avoider.filter_action(None, None, state(0.30, 0.)))
    assert avoider.s == 0 and avoider.summary()["blocked_steps"] > 0


def test_the_predicted_path_is_the_keyframes_plus_the_exit_path_and_stays_kinematic_to_its_end():
    from hazardhosts.libero.scene_clearance import playback_path
    exit_instance = HazardInstance.from_dict(dict(
        instance(0.30, 0., 1.).to_dict(),
        exit_keyframes={"time_s": [1., 2.], "pos": [[.3, 0., 1.], [.3, .5, 1.]], "quat": [[1., 0., 0., 0.]] * 2}))
    phase = [KINEMATIC]
    host, avoider = make(phase)
    avoider.arm(exit_instance)
    assert avoider.path.t_end_s == 2.
    np.testing.assert_allclose(avoider.path.pos(1.5), [.3, .25, 1.])
    np.testing.assert_allclose(playback_path(exit_instance).pos(1.5), avoider.path.pos(1.5))
    # At t=0 the keep-out reaches along the exit path (to y=.5 at t=2), not only to the release at t=1.
    centres, moving = avoider._keep_out(0., state(.3, 0.))
    assert moving and np.max(centres[:, 1]) >= .45 and np.any(np.isclose(centres[:, 1], 0.))
    # Without an exit path the old prediction (to the release, then ballistic) is unchanged.
    avoider.arm(instance(0.30, 0., 1.))
    assert avoider.path.t_end_s == 1.
    centres, _ = avoider._keep_out(0., state(.3, 0.))
    assert np.max(np.abs(centres[:, 1])) < 1e-9
