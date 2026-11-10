"""Conservative pruning must agree with exhaustive sample evaluation."""
import numpy as np
import pytest

from hazardlib.gates.g0 import _max_body_penetration
from hazardlib.kinematics.standoff import RobotGeometry, penetration_depths
from hazardlib.spec.hazard_spec import AgentSpec


@pytest.mark.parametrize("shape", ["box", "sphere"])
def test_pruned_penetration_matches_all_samples_with_rotated_bodies(shape):
    rng = np.random.RandomState(137)
    geometry = RobotGeometry({"a": rng.normal(size=(100, 3))*.1,
                              "b": rng.normal(size=(200, 3))*.2,
                              "distant": rng.normal(size=(300, 3))*.1}, .005)
    pos = {"a": np.array([0., 0., 0.]), "b": np.array([.4, -.1, .2]), "distant": np.array([20., 10., 5.])}
    quats = {k: rng.normal(size=4) for k in pos}
    bounds = {k: ((p.min(axis=0)+p.max(axis=0))/2, (p.max(axis=0)-p.min(axis=0))/2)
              for k, p in geometry.points.items()}
    agent = AgentSpec(kind="object", shape=shape, half_extents_m=[.1, .1, .1], mass_kg=.05)
    points, names = geometry.world_points(pos, quats)
    for centre in rng.uniform(-.6, .6, (100, 3)):
        exhaustive = penetration_depths(agent, centre, points)
        index = int(np.argmax(exhaustive))
        depth, body = _max_body_penetration(agent, centre, geometry, (pos, quats), bounds)
        assert depth == pytest.approx(exhaustive[index], abs=1e-14)
        assert body == names[index]


def test_identical_body_ties_keep_the_original_sorted_name():
    geometry = RobotGeometry({"z": [[0., 0., 0.]], "a": [[0., 0., 0.]]}, .005)
    pos = {name: np.zeros(3) for name in geometry.points}
    quats = {name: np.array([1., 0., 0., 0.]) for name in geometry.points}
    bounds = {name: (np.zeros(3), np.zeros(3)) for name in geometry.points}
    agent = AgentSpec(kind="object", shape="box", half_extents_m=[.1]*3, mass_kg=.05)
    depth, body = _max_body_penetration(agent, np.zeros(3), geometry, (pos, quats), bounds)
    assert depth == .1 and body == "a"
