"""A synthetic V1 / V2 for the generator tests: one finger body moving along x above a table."""
from pathlib import Path

import numpy as np

from hazardlib.config import load_config
from hazardlib.generate import (GenerateSettings, LayoutBody, SceneLayout, Segment, Trajectory, TrajectorySet)

CONFIG = Path(__file__).parents[3] / "configs/hazard/pilot/libero"
AGENT = dict(kind="object", shape="box", half_extents_m=[.02] * 3, mass_kg=.05)
TOP = .9
DT = .05
N = 61
WORKSPACE = dict(min_m=[-1., -1., .9], max_m=[1., 1., 1.5])


def raw_settings():
    return load_config("trajectory_analysis", strict=True, config_dir=CONFIG)


def settings(**changes):
    data = dict(raw_settings(), **changes)
    return GenerateSettings.from_mapping(data)


def support(point, t_s):
    """Stand-in of ``slide_support_rule``: the finger point within ``2 h_z + r_eef`` above the table."""
    return 0. <= point[2] - TOP <= .06


def box(centre, half, rot=None):
    return (np.array([centre], dtype=float), np.array([np.eye(3) if rot is None else rot]),
            np.array([half], dtype=float))


def layout(demo, neighbour=((1.0, 1.0, .95),)):
    c, r, h = box([0., 0., TOP - .02], [1., 1., .02])
    bodies = [LayoutBody("table", "support", c, r, h, np.zeros(3), np.array([1., 0, 0, 0]))]
    for i, centre in enumerate(neighbour):
        c, r, h = box([0., 0., 0.], [.05, .05, .05])
        bodies.append(LayoutBody("cup_%d" % i, "neighbour", c, r, h, np.array(centre), np.array([1., 0, 0, 0])))
    c, r, h = box([0., 0., 0.], [.05, .05, .03])
    bodies.append(LayoutBody("bowl", "task_entity", c, r, h, np.array([.3, 0., TOP + .03]), np.array([1., 0, 0, 0])))
    return SceneLayout(demo, DT, N, bodies, [dict(name="table", min_m=[-1., -1., TOP], max_m=[1., 1., TOP])],
                       dict(min_m=[-.1, -.1, .9], max_m=[.7, .1, 1.1]))


def trajectory(demo, y_offset=0.0, y_until=0, yaw=0.0):
    k = np.arange(N)
    eef = np.stack([.01 * k, np.where(k < y_until, y_offset, 0.0), np.full(N, .95)], axis=1)
    quat = np.tile([1., 0, 0, 0], (N, 1))
    body_pos = eef[:, None, :]
    body_quat = quat[:, None, :]
    entity_quat = np.tile([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], (N, 1, 1))
    entity_pos = np.tile([.3, 0., TOP + .03], (N, 1, 1))
    segments = [Segment("approach_0", "approach", "bowl", 0, 30, 0., 30 * DT),
                Segment("manipulate_0", "manipulate", "bowl", 30, 60, 30 * DT, 60 * DT)]
    return Trajectory(demo, DT, k * DT, eef, quat, ["gripper0_leftfinger"], body_pos, body_quat, ["bowl"],
                      entity_pos, entity_quat, [None] * 30 + ["bowl"] * 31, segments)


POINTS = {"gripper0_leftfinger": np.array([[0., 0., -.02], [0., 0., 0.], [0., 0., .02]])}


def task_set(n_demos=5, deviating=None, yaw=0.0, y_until=20):
    """``deviating``: index of the demo whose first ``y_until`` samples run 0.5 m beside the common path."""
    trajectories = [trajectory("demo_%d" % i, .5 if i == deviating else 0.0, y_until, yaw) for i in range(n_demos)]
    return TrajectorySet("synthetic/task", dict(POINTS), 0.02, trajectories,
                         {t.demo: layout(t.demo) for t in trajectories})
