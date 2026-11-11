"""V1 / V2 containers: IO round trip, content hash, validation (no simulator)."""
import json

import numpy as np
import pytest

import generate_synthetic as S
from hazardlib.generate import LayoutBody, SceneLayout, Trajectory, TrajectorySet


def test_trajectory_round_trip_and_hash(tmp_path):
    t = S.trajectory("demo_0")
    digest = t.save(tmp_path)
    assert digest == t.content_hash()
    back = Trajectory.load(tmp_path)
    assert back.content_hash() == digest and back.demo == "demo_0"
    assert np.array_equal(back.eef_pos, t.eef_pos) and back.held == t.held
    assert [s.to_dict() for s in back.segments] == [s.to_dict() for s in t.segments]
    assert back.first_entity == "bowl"
    changed = S.trajectory("demo_0", y_offset=.1, y_until=3)
    assert changed.content_hash() != digest


def test_trajectory_load_detects_tampering(tmp_path):
    t = S.trajectory("demo_0")
    t.save(tmp_path)
    meta = json.loads((tmp_path / "v1.json").read_text())
    meta["held"][0] = "bowl"
    (tmp_path / "v1.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="hash"):
        Trajectory.load(tmp_path)


def test_trajectory_shape_and_segment_checks():
    t = S.trajectory("demo_0")
    with pytest.raises(ValueError, match="eef_pos"):
        Trajectory(t.demo, t.control_dt_s, t.time_s, t.eef_pos[:-1], t.eef_quat, t.body_names, t.body_pos,
                   t.body_quat, t.entity_names, t.entity_pos, t.entity_quat, t.held, t.segments)
    bad = [type(t.segments[0])("a", "approach", "nobody", 0, 5, 0., .25)]
    with pytest.raises(ValueError, match="entity"):
        Trajectory(t.demo, t.control_dt_s, t.time_s, t.eef_pos, t.eef_quat, t.body_names, t.body_pos, t.body_quat,
                   t.entity_names, t.entity_pos, t.entity_quat, t.held, bad)


def test_layout_round_trip_static_and_moving_bodies(tmp_path):
    layout = S.layout("demo_0")
    c, r, h = S.box([0., 0., 0.], [.04, .04, .04])
    moving_pos = np.stack([np.linspace(0, .3, S.N), np.zeros(S.N), np.full(S.N, 1.0)], axis=1)
    layout.bodies.append(LayoutBody("drawer", "fixture", c, r, h, moving_pos, np.tile([1., 0, 0, 0], (S.N, 1))))
    digest = layout.save(tmp_path)
    back = SceneLayout.load(tmp_path)
    assert back.content_hash() == digest == layout.content_hash()
    assert [b.static for b in back.bodies] == [True, True, True, False]
    assert np.array_equal(back.body("drawer").pos, moving_pos)
    # world boxes follow the track, are clamped past the end, and static ones do not move
    c0, _, _, labels = back.boxes_at(0, ("fixture",))
    c30, _, _, _ = back.boxes_at(30, ("fixture",))
    cend, _, _, _ = back.boxes_at(10 * S.N, ("fixture",))
    assert labels == ["drawer"] and np.allclose(c30 - c0, [.15, 0, 0]) and np.allclose(cend[0], [.3, 0, 1.0])
    assert np.allclose(back.boxes_at(0, ("support",))[0], back.boxes_at(50, ("support",))[0])
    # exclusion by label
    assert "bowl" not in back.boxes_at(0, ("task_entity",), exclude=("bowl",))[3]


def test_layout_validation():
    with pytest.raises(ValueError, match="role"):
        c, r, h = S.box([0, 0, 0], [1, 1, 1])
        LayoutBody("x", "wall", c, r, h, np.zeros(3), np.array([1., 0, 0, 0]))
    layout = S.layout("d")
    with pytest.raises(ValueError, match="duplicate"):
        SceneLayout("d", S.DT, S.N, layout.bodies + layout.bodies[:1], [], dict(min_m=[0, 0, 0], max_m=[1, 1, 1]))


def test_task_set_round_trip_first_n_and_hash(tmp_path):
    ts = S.task_set(n_demos=4)
    ts.save(tmp_path)
    back = TrajectorySet.load(tmp_path)
    assert back.content_hash() == ts.content_hash()
    assert [t.demo for t in back.trajectories] == ["demo_0", "demo_1", "demo_2", "demo_3"]
    assert all(np.array_equal(back.robot_points[k], ts.robot_points[k]) for k in ts.robot_points)
    two = TrajectorySet.load(tmp_path, max_demos=2)
    assert [t.demo for t in two.trajectories] == ["demo_0", "demo_1"]
    assert two.content_hash() != back.content_hash()
    # a layout file edited behind the index is refused
    other = S.layout("demo_1", neighbour=((.5, .5, .95),))
    other.save(tmp_path / "demos" / "demo_1")
    with pytest.raises(ValueError, match="demo_1"):
        TrajectorySet.load(tmp_path)
