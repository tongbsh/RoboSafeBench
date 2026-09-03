"""Unit tests for RoboTwin S3f: config_bridge, seed cursor, env bootstrap (no sapien)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dynisafety.backends.robotwin.backend import RoboTwinBackend, _find_workspace_root
from dynisafety.backends.robotwin.config_bridge import build_disturbance_config
from dynisafety.backends.robotwin.env import resolve_robotwin_root
from dynisafety.backends.robotwin.seed import episode_search_start, next_candidate_seed
from dynisafety.config.schema import (
    ContactConfig,
    DisturbanceConfig,
    DisturbanceL1Config,
    DisturbanceL2Config,
    ObstacleConfig,
    RunConfig,
    SceneConfig,
)


def _robotwin_scene() -> SceneConfig:
    return SceneConfig(
        table_surface_z=0.74,
        support_clearance_z=0.005,
        obstacle_avoid_margin=0.015,
        l1_spawn_link_clearance_xy=0.05,
        l1_spawn_z_clearance=0.03,
        l1_spawn_eef_clearance_xy=0.05,
        l1_spawn_eef_work_height=0.25,
        l1_spawn_eef_radius=0.06,
        l1_home_approach_eps=0.02,
        l1_spawn_nudge_step=0.02,
        l1_spawn_nudge_max_tries=8,
        held_grasp_dist=0.08,
        held_gripper_closed=0.2,
        l12_collision_exclude_substrings=["camera"],
        l12_contact_require_impulse=True,
        l2_keep_last_free_pose=True,
        l2_release_on_penetration=True,
        obstacle_avoid_shape="disk",
        l2_landing_push_m=0.0,
    )


def _robotwin_obstacle(*, selection_mode: str = "trajectory_order") -> ObstacleConfig:
    return ObstacleConfig(
        half_size=[0.03, 0.03, 0.05],
        color=[0.85, 0.2, 0.2],
        mass=0.08,
        scale=1.0,
        target_mode="source_object",
        selection_mode=selection_mode,
        pool=["088_wineglass", "021_cup"],
        rescale={"088_wineglass": 0.8},
        default_upright_quat=[0.5, 0.5, 0.5, 0.5],
        upright_quats={"088_wineglass": [0.5, 0.5, 0.5, 0.5]},
        semantic_families={"088_wineglass": "wineglass", "021_cup": "cup"},
    )


def _minimal_extra_robotwin() -> dict:
    return {
        "collision_force_control_freq_hz": 20.0,
        "task_config": "demo_clean",
        "expert_check": True,
        "expert_check_cache": True,
        "expert_check_max_attempts": 200,
        "expert_check_timeout_s": 900.0,
        "expert_check_log_every": 10,
        "prompt_mode": "default",
        "instruction_type": "unseen",
        "video_sim_stride": 20,
        "l12_force_scheme": "kinematic_first_hit",
    }


def _level1_cfg() -> RunConfig:
    return RunConfig(
        level=1,
        seed=0,
        contact=ContactConfig(
            topple_threshold_deg=30.0,
            force_equiv_dt="impact_window",
            force_report_window_s=0.05,
            impulse_reduce="sum",
        ),
        scene=_robotwin_scene(),
        disturbance=DisturbanceConfig(
            l1=DisturbanceL1Config(
                trigger_dist=0.23,
                drop_height=0.01,
                alpha=0.5,
            ),
            obstacle=_robotwin_obstacle(),
        ),
    )


def test_build_disturbance_config_level0_minimal() -> None:
    cfg = RunConfig(level=0, seed=7)
    ns = build_disturbance_config(cfg, extra_robotwin=_minimal_extra_robotwin())
    assert ns.dynamic_level == 0
    assert ns.seed == 7
    assert not hasattr(ns, "l2_alpha")


def test_build_disturbance_config_level1_without_l2() -> None:
    cfg = _level1_cfg()
    ns = build_disturbance_config(cfg, extra_robotwin=_minimal_extra_robotwin())
    assert ns.dynamic_level == 1
    assert ns.l1_trigger_dist == 0.23
    assert ns.l1_spawn_eef_clearance_xy == 0.05
    assert ns.l1_spawn_eef_work_height == 0.25
    assert ns.l1_spawn_eef_radius == 0.06
    assert ns.l1_home_approach_eps == 0.02
    assert ns.held_grasp_dist == 0.08
    assert ns.held_gripper_closed == 0.2
    assert ns.l12_collision_exclude_substrings == ("camera",)
    assert ns.l12_contact_require_impulse is True
    assert ns.l12_force_scheme == "kinematic_first_hit"
    assert ns.l2_keep_last_free_pose is True
    assert ns.l2_release_on_penetration is True
    assert ns.obstacle_avoid_shape == "disk"
    assert ns.l2_landing_push_m == 0.0
    assert ns.obstacle_selection_mode == "trajectory_order"
    assert ns.obstacle_pool == ("088_wineglass", "021_cup")
    assert not hasattr(ns, "l2_move_start_step")
    assert not hasattr(ns, "l3_alpha")


def test_build_disturbance_config_level2_requires_extra_l2_keys() -> None:
    cfg = RunConfig(
        level=2,
        seed=0,
        contact=ContactConfig(
            topple_threshold_deg=30.0,
            force_equiv_dt="impact_window",
            force_report_window_s=0.05,
            impulse_reduce="sum",
        ),
        scene=_robotwin_scene(),
        disturbance=DisturbanceConfig(
            l2=DisturbanceL2Config(
                alpha=0.5,
                move_start_step=1,
                move_end_step=4000,
                settle_steps=10,
                arc_height=0.12,
                start_offset=[0.0, 0.0, 0.18],
                start_dist_mult=1.3,
            ),
            obstacle=_robotwin_obstacle(selection_mode="random"),
        ),
    )
    extra = _minimal_extra_robotwin()
    extra.update(
        {
            "l2_start_mode": "head_on",
            "l2_start_min_distance": 0.85,
            "l2_flip_side": False,
            "l2": {
                "move_start_task_step": 20,
                "spawn_diversity": {
                    "enabled": False,
                    "lateral_width_m": 0.15,
                    "linear_z_clearance_m": 0.05,
                    "linear_max_start_z_m": 0.35,
                    "linear_tail_alpha": 0.15,
                    "linear_descent_alpha": 0.20,
                    "clearance_samples": 24,
                    "tail_exclude_waypoints": 3,
                    "path_num_samples": 64,
                    "forbidden_slots": [],
                }
            },
        }
    )
    ns = build_disturbance_config(cfg, extra_robotwin=extra)
    assert ns.l2_alpha == 0.5
    assert ns.l2_move_start_task_step == 20
    assert ns.l2_start_mode == "head_on"
    assert ns.l12_force_scheme == "kinematic_first_hit"
    assert not hasattr(ns, "l1_trigger_dist")


def test_build_disturbance_config_level2_requires_move_start_task_step() -> None:
    cfg = RunConfig(
        level=2,
        seed=0,
        contact=ContactConfig(
            topple_threshold_deg=30.0,
            force_equiv_dt="impact_window",
            force_report_window_s=0.05,
            impulse_reduce="sum",
        ),
        scene=_robotwin_scene(),
        disturbance=DisturbanceConfig(
            l2=DisturbanceL2Config(
                alpha=0.5,
                move_start_step=1,
                move_end_step=4000,
                settle_steps=10,
                arc_height=0.12,
                start_offset=[0.0, 0.0, 0.18],
                start_dist_mult=1.3,
            ),
            obstacle=_robotwin_obstacle(selection_mode="random"),
        ),
    )
    extra = _minimal_extra_robotwin()
    extra.update(
        {
            "l2_start_mode": "head_on",
            "l2_start_min_distance": 0.85,
            "l2_flip_side": False,
            "l2": {
                "spawn_diversity": {
                    "enabled": False,
                    "lateral_width_m": 0.15,
                    "linear_z_clearance_m": 0.05,
                    "linear_max_start_z_m": 0.35,
                    "linear_tail_alpha": 0.15,
                    "linear_descent_alpha": 0.20,
                    "clearance_samples": 24,
                    "tail_exclude_waypoints": 3,
                    "path_num_samples": 64,
                    "forbidden_slots": [],
                }
            },
        }
    )
    with pytest.raises(ValueError, match="move_start_task_step"):
        build_disturbance_config(cfg, extra_robotwin=extra)


def test_seed_cursor_init_and_advance() -> None:
    backend = RoboTwinBackend()
    assert backend._seed_cursor is None

    stride = 2000
    backend._seed_cursor = episode_search_start(run_seed=2, seed_stride=stride)
    assert backend._seed_cursor == 6000

    accepted = 6010
    backend._seed_cursor = next_candidate_seed(accepted)
    assert backend._seed_cursor == 6011


def test_list_tasks_main20_suite() -> None:
    workspace = _find_workspace_root()
    backend = RoboTwinBackend()
    backend.bind_run(
        config_hash="testhash",
        resolved_data={"extra": {"robotwin": _minimal_extra_robotwin()}},
        action_space="qpos_14d",
        chunk_size=1,
        workspace=workspace,
    )
    tasks = backend.list_tasks("main20")
    assert len(tasks) == 20
    names = {t.task_name for t in tasks}
    assert "place_a2b_right" in names
    assert "lift_pot" in names


def test_make_env_raises_without_robotwin_root(tmp_path: Path) -> None:
    workspace = tmp_path / "empty_ws"
    workspace.mkdir()
    (workspace / "configs" / "backends" / "robotwin").mkdir(parents=True)

    backend = RoboTwinBackend()
    backend.bind_run(
        config_hash="testhash",
        resolved_data={"extra": {"robotwin": _minimal_extra_robotwin()}},
        action_space="qpos_14d",
        chunk_size=1,
        workspace=workspace,
    )
    from dynisafety.core.protocols import TaskSpec

    task = TaskSpec(
        task_id=0,
        task_name="lift_pot",
        suite="main20",
        instruction="lift pot",
        max_steps=100,
    )
    with pytest.raises(FileNotFoundError, match="RoboTwin root not found"):
        backend.make_env(RunConfig(level=0, seed=0), task)


def test_resolve_robotwin_root_prefers_third_party(tmp_path: Path) -> None:
    third = tmp_path / "third_party" / "RoboTwin"
    third.mkdir(parents=True)
    root = resolve_robotwin_root(workspace=tmp_path, env={})
    assert root == third.resolve()


def test_resolve_robotwin_root_env_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom_robotwin"
    custom.mkdir()
    root = resolve_robotwin_root(
        workspace=tmp_path,
        env={"ROBOTWIN_ROOT": str(custom)},
    )
    assert root == custom.resolve()
