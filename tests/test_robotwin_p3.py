"""P3: official topple after release, AABB avoid, YAML catalog / trajectory_order."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dynisafety.backends.robotwin.avoid import (
    nudge_xy_aabb,
    nudge_xy_disk,
    push_xy_away,
    world_xy_aabb_from_obb,
)
from dynisafety.backends.robotwin.contact import classify_topple
from dynisafety.backends.robotwin.disturbance.manager import DisturbanceManager
from dynisafety.backends.robotwin.obstacles import select_obstacle
from dynisafety.config.loader import LoadRequest, load_config


def test_classify_topple_official_only_after_release() -> None:
    assert classify_topple(tilt_exceeds=True, released_to_physics=False) == (False, True)
    assert classify_topple(tilt_exceeds=True, released_to_physics=True) == (True, False)
    assert classify_topple(tilt_exceeds=False, released_to_physics=True) == (False, False)
    assert classify_topple(tilt_exceeds=False, released_to_physics=False) == (False, False)


def test_nudge_xy_disk_and_aabb() -> None:
    center = np.array([0.0, 0.0])
    # Inside the keep-out circle → pushed to the rim.
    out = nudge_xy_disk(np.array([0.01, 0.0]), center, 0.10)
    assert abs(float(np.linalg.norm(out)) - 0.10) < 1e-9
    # AABB: obstacle center inside expanded laptop lid.
    pushed = nudge_xy_aabb(
        np.array([0.0, 0.0]),
        (0.03, 0.03),
        np.array([-0.15, -0.10]),
        np.array([0.15, 0.10]),
        0.02,
    )
    assert abs(float(pushed[0])) >= 0.15 + 0.03 + 0.02 - 1e-9 or abs(float(pushed[1])) >= 0.10 + 0.03 + 0.02 - 1e-9


def test_push_xy_away_and_world_aabb() -> None:
    moved = push_xy_away(np.array([0.2, 0.0]), np.array([0.0, 0.0]), 0.10)
    assert abs(float(moved[0]) - 0.30) < 1e-9
    assert push_xy_away(np.array([0.2, 0.0]), np.array([0.0, 0.0]), 0.0)[0] == 0.2
    mins, maxs = world_xy_aabb_from_obb(
        center=np.array([0.0, 0.0, 0.0]),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        half_xyz=np.array([0.2, 0.05, 0.1]),
    )
    assert abs(float(mins[0]) + 0.2) < 1e-9
    assert abs(float(maxs[1]) - 0.05) < 1e-9


def test_select_obstacle_trajectory_order_is_deterministic() -> None:
    cands = [{"modelname": f"m{i}"} for i in range(3)]
    rng = np.random.default_rng(0)
    assert select_obstacle(cands, selection_mode="trajectory_order", episode_index=4, rng=rng)[
        "modelname"
    ] == "m1"


def test_manager_topple_stays_landing_while_kinematic() -> None:
    mgr = object.__new__(DisturbanceManager)
    mgr.config = SimpleNamespace(dynamic_level=2, topple_threshold_deg=30.0)
    mgr.actor = SimpleNamespace(get_pose=lambda: SimpleNamespace(q=np.array([0.0, 1.0, 0.0, 0.0])))
    mgr._obstacle_quat = np.array([1.0, 0.0, 0.0, 0.0])
    mgr._actor_is_kinematic = True
    mgr._max_tilt_deg = 0.0
    mgr._obstacle_toppled = False
    mgr._landing_toppled = False
    mgr._update_topple_metrics()
    assert mgr._landing_toppled is True
    assert mgr._obstacle_toppled is False
    mgr._actor_is_kinematic = False
    mgr._update_topple_metrics()
    assert mgr._obstacle_toppled is True


def test_robotwin_yaml_catalog_and_open_laptop_overlay() -> None:
    base = load_config(LoadRequest(backend="robotwin", level=2, suite="main20", model="random"))
    obstacle = base.data["disturbance"]["obstacle"]
    assert obstacle["selection_mode"] == "trajectory_order"
    assert obstacle["pool"] == [
        "088_wineglass",
        "109_hydrating-oil",
        "114_bottle",
        "021_cup",
        "080_pillbottle",
    ]
    assert obstacle["rescale"]["088_wineglass"] == 0.8
    assert base.data["scene"]["obstacle_avoid_shape"] == "disk"
    assert float(base.data["scene"]["l2_landing_push_m"]) == 0.0

    laptop = load_config(
        LoadRequest(
            backend="robotwin",
            level=2,
            suite="main20",
            task="16",
            model="random",
        )
    )
    assert laptop.provenance["scene.obstacle_avoid_shape"].endswith("tasks/main20_16.yaml")
    lo = laptop.data["disturbance"]["obstacle"]
    assert lo["pool"] == ["021_cup", "080_pillbottle", "114_bottle"]
    assert laptop.data["scene"]["obstacle_avoid_shape"] == "aabb"
    assert float(laptop.data["scene"]["obstacle_avoid_margin"]) == 0.08
    assert float(laptop.data["scene"]["l2_landing_push_m"]) == 0.10
    # Parent catalog metadata still present after overlaying the pool.
    assert "114_bottle" in lo["upright_quats"]
    assert lo["semantic_families"]["021_cup"] == "cup"
