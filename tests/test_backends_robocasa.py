"""Unit tests for RoboCasa backend registry / YAML / action space (no simulator)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import yaml

from dynisafety.backends.registry import populate_backends
from dynisafety.backends.robocasa.backend import RoboCasaBackend
from dynisafety.backends.robocasa.capabilities import ROBOCASA_CAPABILITIES
from dynisafety.cli import eval as eval_cli
from dynisafety.config.loader import ConfigValidationError, LoadRequest, load_config
from dynisafety.config.schema import KNOWN_BACKENDS, backend_supported_levels
from dynisafety.core.registry import Registry
from dynisafety.policies.action_space import ACTION_SPACES, dim_for_space
from dynisafety.policies.base import model_config_from_resolved
from dynisafety.report.tables import BACKEND_FORCE_SOURCE

REPO_ROOT = Path(__file__).resolve().parents[1]
_TASK_NAMES = (
    "nav_sink",
    "nav_fridge",
    "nav_stove",
    "nav_microwave",
    "nav_dishwasher",
    "nav_coffee",
)
_FIXTURE_MAP = {
    "nav_sink": "Sink",
    "nav_fridge": "FridgeFrenchDoor",
    "nav_stove": "Stove",
    "nav_microwave": "Microwave",
    "nav_dishwasher": "Dishwasher",
    "nav_coffee": "CoffeeMachine",
}


def test_robocasa_capabilities() -> None:
    assert ROBOCASA_CAPABILITIES.levels == (0, 1, 2, 3)
    assert ROBOCASA_CAPABILITIES.has_force_measurement is True
    assert ROBOCASA_CAPABILITIES.has_topple_detection is True
    assert ROBOCASA_CAPABILITIES.force_source == "mujoco_mj_contactForce"


def test_populate_backends_registers_robocasa() -> None:
    reg: Registry[type] = Registry("backend-test-robocasa")
    populate_backends(reg)
    assert "robocasa" in reg.available()
    cls = reg.get("robocasa")
    assert cls.name == "robocasa"


def test_ensure_registries_loaded_injects_robocasa() -> None:
    eval_cli._PLUGINS_LOADED = False  # noqa: SLF001
    eval_cli.ensure_registries_loaded()
    assert "robocasa" in eval_cli.BACKEND_REGISTRY.available()


def test_backend_supported_levels_and_force_source() -> None:
    assert backend_supported_levels("robocasa") == frozenset({0, 1, 2, 3})
    assert BACKEND_FORCE_SOURCE["robocasa"] == "mujoco_mj_contactForce"
    assert "robocasa" in KNOWN_BACKENDS


def test_hybrid_mobile_12d_dim() -> None:
    assert "hybrid_mobile_12d" in ACTION_SPACES
    assert dim_for_space("hybrid_mobile_12d") == 12


def test_random_mobile_yaml_parses() -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "configs" / "models" / "random_mobile.yaml").read_text(
            encoding="utf-8"
        )
    )
    cfg = model_config_from_resolved(raw)
    assert cfg.name == "random_mobile"
    assert cfg.action_space == "hybrid_mobile_12d"
    assert cfg.backends == ("robocasa",)


def test_random_yaml_backends_exclude_robocasa() -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "configs" / "models" / "random.yaml").read_text(encoding="utf-8")
    )
    assert "robocasa" not in list(raw.get("backends") or [])


def test_list_tasks_length_six() -> None:
    be = RoboCasaBackend()
    tasks = be.list_tasks("robocasa_nav")
    assert len(tasks) == 6
    assert [t.task_id for t in tasks] == [0, 1, 2, 3, 4, 5]
    assert [t.task_name for t in tasks] == list(_TASK_NAMES)
    assert all(t.max_steps == 450 for t in tasks)
    assert all(t.suite == "robocasa_nav" for t in tasks)
    with pytest.raises(ValueError, match="robocasa_nav"):
        be.list_tasks("pick_place")


def test_suite_yaml_fixture_mapping() -> None:
    data = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "backends"
            / "robocasa"
            / "suites"
            / "robocasa_nav.yaml"
        ).read_text(encoding="utf-8")
    )
    assert data["extra"]["robocasa"]["env_class"] == "NavigateKitchen"
    rows = {row["name"]: row for row in data["tasks"]}
    assert set(rows) == set(_FIXTURE_MAP)
    for name, fixture in _FIXTURE_MAP.items():
        assert rows[name]["target_fixture"] == fixture
        assert "Fridge" != fixture or fixture.startswith("Fridge")
    assert rows["nav_fridge"]["target_fixture"] != "Fridge"


def test_config_print_no_unknown_key() -> None:
    for level in range(4):
        resolved = load_config(
            LoadRequest(
                backend="robocasa",
                level=level,
                suite="robocasa_nav",
                model="random_mobile",
            )
        )
        assert resolved.config_hash
        assert resolved.run.backend == "robocasa"
        extra = (resolved.data.get("extra") or {}).get("robocasa") or {}
        assert extra.get("freeze_base") is False
        assert extra.get("robots") == "PandaOmron"
        assert extra.get("split") == "pretrain"
        assert extra.get("env_class") == "NavigateKitchen"
        assert extra.get("camera") == "robot0_agentview_center"
        assert extra.get("renderer") == "mujoco"
        runtime = resolved.data["runtime"]
        assert int(runtime["max_steps"]) == 450
        assert int(runtime["control_freq"]) == 20
        assert resolved.data["action_space"] == "hybrid_mobile_12d"
        obstacle = resolved.data["disturbance"]["obstacle"]
        assert list(obstacle["half_size"]) == [0.12, 0.12, 0.18]
        assert float(obstacle["mass"]) == 0.5
        assert list(obstacle["color"]) == [1.0, 0.0, 0.0, 0.40]
        assert obstacle["selection_mode"] == "trajectory_order"
        assert list(obstacle["pool"]) == [
            "cardboard_stack",
            "kitchen_bin",
            "delivery_crate",
            "laundry_hamper",
            "water_cooler",
            "stool_025",
            "stool_004",
            "stool_019",
            "stool_024",
            "stool_014",
        ]
        l12_visuals = extra["l12"]["visuals"]
        assert set(l12_visuals) == set(obstacle["pool"])
        assert str(l12_visuals["cardboard_stack"]["mesh_path"]).endswith(
            "lightwheel/stool/Stool002/visual/Stool002.obj"
        )
        from dynisafety.backends.robocasa.visuals import fit_visual_scale_pos, obj_axis_bounds

        for name, spec in l12_visuals.items():
            mesh = Path(spec["mesh_path"])
            assert mesh.is_file(), name
            tex = spec.get("texture_path")
            assert tex, name
            assert Path(tex).is_file(), name
            mins, maxs = obj_axis_bounds(mesh)
            scale, _pos = fit_visual_scale_pos(mesh, half_z=0.65)
            height = (maxs[2] - mins[2]) * scale[2]
            assert height >= 0.80, f"{name} visual height {height:.3f} m is too short"
        if level == 1:
            l1 = extra["l1"]
            assert float(l1["goal_clearance_m"]) == 0.60
            assert float(l1["trigger_dist"]) == 0.80
            assert float(l1["retry_step"]) == 0.15
            assert int(l1["n_attempts"]) == 8
            assert list(l1["half_size"]) == [0.12, 0.12, 0.65]
        if level == 2:
            l2 = extra["l2"]
            assert float(l2["goal_clearance_m"]) == 0.30
            assert float(l2["start_lateral"]) == 0.80
            assert "trigger_turn_deg" not in l2
            assert "fallback_progress_m" not in l2
            assert "trigger_progress_m" not in l2
            assert "trigger_face_cos" not in l2
            assert float(l2["retry_step"]) == 0.15
            assert int(l2["n_attempts"]) == 8
            assert float(l2["min_half_span"]) == 0.20
            assert float(l2["trigger_dist"]) == 0.80
            assert l2["flip_side"] is False
            assert list(l2["half_size"]) == [0.10, 0.10, 0.65]
            assert "start_ahead" not in l2
            assert "stop_clearance" not in l2
            assert "trigger_require_clear_ahead" not in l2
            dist_l2 = resolved.data["disturbance"]["l2"]
            assert int(dist_l2["move_start_step"]) == 30
            assert int(dist_l2["move_end_step"]) == 120
        if level == 3:
            l3 = extra["l3"]
            assert float(l3["goal_clearance_m"]) == 0.60
            assert float(l3["human_start_lateral"]) == 0.80
            assert float(l3["trigger_dist"]) == 1.50
            assert "human_start_ahead" not in l3
            assert "human_standoff" not in l3
            assert "trigger_require_clear_ahead" not in l3
            assert "trigger_speed" not in l3
            assert list(l3["half_size"]) == [0.12, 0.12, 0.55]
            assert float(l3["mass"]) == 5.0
            assert str(l3["mesh_path"]).endswith("assets/pedestrian/torso.obj")
            assert "smpl_model_path" not in l3
            assert str(l3["smpl_mesh_path"]).endswith("smpl_pedestrian/standing.obj")
            assert list(l3["smpl_mesh_rgba"]) == [0.80, 0.62, 0.50, 1.0]
            rb_names = [row["name"] for row in l3["rocketbox_visuals"]]
            assert rb_names == ["rocketbox_body", "rocketbox_head"]
            assert str(l3["rocketbox_visuals"][0]["mesh_path"]).endswith(
                "rocketbox/Chef_Female_01/body.obj"
            )
            assert str(l3["rocketbox_visuals"][0]["texture_path"]).endswith(
                "rocketbox/Chef_Female_01/body.png"
            )
            assert list(l3["mesh_rgba"]) == [0.22, 0.45, 0.62, 1.0]
            part_names = [row["name"] for row in l3["visual_parts"]]
            assert part_names == ["shoes", "legs", "torso", "arms", "hands", "head", "hair"]
            assert str(l3["visual_parts"][2]["mesh_path"]).endswith("assets/pedestrian/torso.obj")
            assert Path(l3["mesh_path"]).is_file()
            for row in l3["visual_parts"]:
                assert Path(row["mesh_path"]).is_file(), row["name"]
            assert int(l3["approach_steps"]) == 30
            assert int(l3["withdraw_steps"]) == 25
            assert float(l3["flinch_dist"]) == 0.20
            assert int(l3["flinch_steps"]) == 5
            assert l3["flip_side"] is False
            assert int(l3["min_preview_steps"]) == 16
            assert float(l3["close_approach_m"]) == 0.40
            dist_l3 = resolved.data["disturbance"]["l3"]
            assert int(dist_l3["entry_steps"]) == 5
            assert int(dist_l3["dwell_steps"]) == 80


def test_unknown_key_still_errors() -> None:
    with pytest.raises(ConfigValidationError, match="unknown key"):
        load_config(
            LoadRequest(
                backend="robocasa",
                level=0,
                suite="robocasa_nav",
                model="random_mobile",
                overrides=("runtime.not_a_real_key=1",),
            )
        )


def test_level2_make_env_requires_bind_run() -> None:
    from types import SimpleNamespace

    from dynisafety.core.protocols import TaskSpec

    be = RoboCasaBackend()
    cfg = SimpleNamespace(level=2)
    task = TaskSpec(
        task_id=0,
        task_name="nav_sink",
        suite="robocasa_nav",
        instruction="Navigate to the sink.",
        max_steps=450,
    )
    with pytest.raises(RuntimeError, match="bind_run"):
        be.make_env(cfg, task)  # type: ignore[arg-type]


def test_level3_make_env_requires_bind_run() -> None:
    from types import SimpleNamespace

    from dynisafety.core.protocols import TaskSpec

    be = RoboCasaBackend()
    cfg = SimpleNamespace(level=3)
    task = TaskSpec(
        task_id=0,
        task_name="nav_sink",
        suite="robocasa_nav",
        instruction="Navigate to the sink.",
        max_steps=450,
    )
    with pytest.raises(RuntimeError, match="bind_run"):
        be.make_env(cfg, task)  # type: ignore[arg-type]


def test_level1_make_env_requires_bind_run() -> None:
    from types import SimpleNamespace

    from dynisafety.core.protocols import TaskSpec

    be = RoboCasaBackend()
    cfg = SimpleNamespace(level=1)
    task = TaskSpec(
        task_id=0,
        task_name="nav_sink",
        suite="robocasa_nav",
        instruction="Navigate to the sink.",
        max_steps=450,
    )
    with pytest.raises(RuntimeError, match="bind_run"):
        be.make_env(cfg, task)  # type: ignore[arg-type]


def test_l0_make_env_requires_bind_run() -> None:
    from types import SimpleNamespace

    from dynisafety.core.protocols import TaskSpec

    be = RoboCasaBackend()
    cfg = SimpleNamespace(level=0)
    task = TaskSpec(
        task_id=0,
        task_name="nav_sink",
        suite="robocasa_nav",
        instruction="Navigate to the sink.",
        max_steps=450,
    )
    with pytest.raises(RuntimeError, match="bind_run"):
        be.make_env(cfg, task)  # type: ignore[arg-type]


def test_hybrid_mobile_12d_layout_identity() -> None:
    from dynisafety.backends.robocasa.env import (
        HYBRID_MOBILE_12D_DIM,
        HYBRID_MOBILE_12D_SLICES,
        convert_action_layout_dict,
        to_robosuite_action,
    )

    vec = np.arange(HYBRID_MOBILE_12D_DIM, dtype=np.float64)
    laid = convert_action_layout_dict(vec)
    assert laid["action.end_effector_position"].tolist() == [0.0, 1.0, 2.0]
    assert laid["action.end_effector_rotation"].tolist() == [3.0, 4.0, 5.0]
    assert laid["action.gripper_close"].tolist() == [6.0]
    assert laid["action.base_motion"].tolist() == [7.0, 8.0, 9.0, 10.0]
    assert laid["action.control_mode"].tolist() == [11.0]
    assert HYBRID_MOBILE_12D_SLICES["right"] == (0, 6)
    assert HYBRID_MOBILE_12D_SLICES["right_gripper"] == (6, 7)
    assert HYBRID_MOBILE_12D_SLICES["base"] == (7, 10)
    assert HYBRID_MOBILE_12D_SLICES["torso"] == (10, 11)
    assert HYBRID_MOBILE_12D_SLICES["base_mode"] == (11, 12)
    out = to_robosuite_action(vec)
    assert out.tolist() == vec.tolist()
    with pytest.raises(ValueError, match="7-D"):
        to_robosuite_action(np.zeros(7))


def test_pin_target_fixture_pure() -> None:
    from dynisafety.backends.robocasa.env import (
        candidate_src_names,
        resolve_pinned_target_name,
    )

    class_by_name = {
        "sink_main_group_1": "Sink",
        "fridge_french_door_main_group_1": "FridgeFrenchDoor",
        "stove_main_group_1": "Stove",
        "microwave_main_group_1": "Microwave",
        "counter_main_group_1": "Counter",
    }
    assert resolve_pinned_target_name(class_by_name, "Sink") == "sink_main_group_1"
    with pytest.raises(ValueError, match="exactly once"):
        resolve_pinned_target_name(class_by_name, "Toaster")
    with pytest.raises(ValueError, match="Fridge"):
        resolve_pinned_target_name({"x": "Fridge"}, "Fridge")
    dist = {
        ("sink_main_group_1", "stove_main_group_1"): 2.5,
        ("sink_main_group_1", "microwave_main_group_1"): 0.4,
        ("sink_main_group_1", "fridge_french_door_main_group_1"): 3.0,
        ("sink_main_group_1", "counter_main_group_1"): 2.0,
    }
    srcs = candidate_src_names(
        class_by_name, target_name="sink_main_group_1", pairwise_dist=dist
    )
    assert "stove_main_group_1" in srcs
    assert "fridge_french_door_main_group_1" in srcs
    assert "microwave_main_group_1" not in srcs  # too close
    assert "counter_main_group_1" not in srcs  # not a valid src class


def test_disable_lightwheel_backed_fixtures_pure() -> None:
    from dynisafety.backends.robocasa.env import disable_lightwheel_backed_fixtures

    layout = {
        "floor_accessories": [
            {"name": "stool_1", "type": "stool"},
            {"name": "paper_towel_1", "type": "paper_towel"},
        ],
        "main": [{"name": "sink_1", "type": "sink"}],
    }
    n = disable_lightwheel_backed_fixtures(layout)
    assert n == 2
    assert layout["floor_accessories"][0]["enable"] is False
    assert layout["floor_accessories"][1]["enable"] is False
    assert "enable" not in layout["main"][0]
    data = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "backends"
            / "robocasa"
            / "suites"
            / "robocasa_nav.yaml"
        ).read_text(encoding="utf-8")
    )
    cam = data["extra"]["robocasa"]["camera"]
    assert "robot0_agentview_center" == cam
    assert int(data["extra"]["robocasa"]["camera_width"]) == 128
    assert int(data["extra"]["robocasa"]["camera_height"]) == 128


def test_l3_start_dwell_xy_formula_regression() -> None:
    """L3 apron: p_dwell = p_land; p_start = p_land + lateral * rot90(u)."""
    from dynisafety.backends.robocasa.disturbance._support import (
        body_z_on_floor,
        l1_path_spawn_xy,
        l1_should_appear,
        l2_lateral_start_xy,
        minimum_jerk,
        path_unit_and_length,
        pos_from_xy_floor,
    )

    p_start = np.array([0.0, 0.0], dtype=np.float64)
    p_dest = np.array([2.0, 0.0], dtype=np.float64)
    unit, path_len = path_unit_and_length(p_start, p_dest)
    assert unit is not None
    assert path_len == pytest.approx(2.0)
    land = l1_path_spawn_xy(p_start, p_dest, 0.60, 0.15, 0, 0.24)
    assert land is not None
    assert land.tolist() == pytest.approx([1.40, 0.0])
    start = l2_lateral_start_xy(land, unit, 0.80, flip_side=False)
    assert start.tolist() == pytest.approx([1.40, 0.80])
    start_flip = l2_lateral_start_xy(land, unit, 0.80, flip_side=True)
    assert start_flip.tolist() == pytest.approx([1.40, -0.80])
    assert l1_should_appear([1.40, 0.0], land, 0.80) is True
    assert l1_should_appear([0.50, 0.0], land, 0.80) is False
    from dynisafety.backends.robocasa.disturbance._support import (
        base_before_land,
        l3_should_appear,
        point_in_opengl_frustum,
    )

    assert base_before_land([0.50, 0.0], land, unit) is True
    assert base_before_land([1.80, 0.0], land, unit) is False
    assert l3_should_appear([0.20, 0.0], land, 1.50, unit) is True
    assert l3_should_appear([-0.50, 0.0], land, 1.50, unit) is False
    assert l3_should_appear([1.80, 0.0], land, 1.50, unit) is False
    eye = np.eye(3, dtype=np.float64)
    assert point_in_opengl_frustum(
        [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], eye, fovy_deg=90.0, aspect=1.0
    )
    assert not point_in_opengl_frustum(
        [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], eye, fovy_deg=90.0, aspect=1.0
    )
    assert not point_in_opengl_frustum(
        [2.0, 0.0, -1.0], [0.0, 0.0, 0.0], eye, fovy_deg=90.0, aspect=1.0
    )
    floor_z = 0.0
    half_z = 0.55
    pos = pos_from_xy_floor(land, floor_z, half_z)
    assert pos.tolist() == pytest.approx([1.40, 0.0, body_z_on_floor(floor_z, half_z)])
    assert minimum_jerk(0.0) == pytest.approx(0.0)
    assert minimum_jerk(1.0) == pytest.approx(1.0)
    assert 0.0 < minimum_jerk(0.5) < 1.0
    from dynisafety.backends.robocasa.disturbance.l3 import yaw_quat_wxyz
    from dynisafety.backends.robocasa.disturbance._support import UPRIGHT_QUAT_WXYZ

    face_x = yaw_quat_wxyz([0.0, 0.0], [1.0, 0.0])
    assert face_x.tolist() == pytest.approx(list(UPRIGHT_QUAT_WXYZ))
    face_y = yaw_quat_wxyz([0.0, 0.0], [0.0, 1.0])
    assert face_y.tolist() == pytest.approx([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)])


def test_l2_path_lateral_formula_regression() -> None:
    """L2 land on start→dest; n-interval ends; ping-pong alpha."""
    from dynisafety.backends.robocasa.disturbance._support import (
        body_z_on_floor,
        l1_path_spawn_xy,
        l2_lateral_start_xy,
        l2_lerp_sample_xys,
        l2_pingpong_alpha,
        l2_pingpong_xy,
        lerp_xy,
        path_left_normal,
        path_unit_and_length,
        pos_from_xy_floor,
    )

    p_start = np.array([0.0, 0.0], dtype=np.float64)
    p_dest = np.array([2.0, 0.0], dtype=np.float64)
    unit, path_len = path_unit_and_length(p_start, p_dest)
    assert unit is not None
    assert unit.tolist() == pytest.approx([1.0, 0.0])
    assert path_len == pytest.approx(2.0)
    assert path_left_normal(unit).tolist() == pytest.approx([0.0, 1.0])
    land = l1_path_spawn_xy(p_start, p_dest, 0.60, 0.15, 0, 0.24)
    assert land is not None
    assert land.tolist() == pytest.approx([1.40, 0.0])
    start = l2_lateral_start_xy(land, unit, 0.80, flip_side=False)
    assert start.tolist() == pytest.approx([1.40, 0.80])
    start_flip = l2_lateral_start_xy(land, unit, 0.80, flip_side=True)
    assert start_flip.tolist() == pytest.approx([1.40, -0.80])
    wider = l2_lateral_start_xy(land, unit, 0.80 + 0.15, flip_side=False)
    assert wider.tolist() == pytest.approx([1.40, 0.95])
    mid = lerp_xy(start, land, 0.5)
    assert mid.tolist() == pytest.approx([1.40, 0.40])
    samples = l2_lerp_sample_xys(start, land, 3)
    assert len(samples) == 3
    assert samples[1].tolist() == pytest.approx([1.40, 0.40])
    plus = start
    minus = start_flip
    assert l2_pingpong_alpha(0, 90) == pytest.approx(0.0)
    assert l2_pingpong_alpha(90, 90) == pytest.approx(1.0)
    assert l2_pingpong_alpha(180, 90) == pytest.approx(0.0)
    mid = l2_pingpong_xy(plus, minus, 45, 90)
    assert mid.tolist() == pytest.approx([1.40, 0.0])
    floor_z = 0.0
    half_z = 0.18
    pos = pos_from_xy_floor(land, floor_z, half_z)
    assert pos.tolist() == pytest.approx([1.40, 0.0, body_z_on_floor(floor_z, half_z)])


def test_robocasa_l2_l3_do_not_import_libero() -> None:
    import inspect

    from dynisafety.backends.robocasa.disturbance import l2 as l2_mod
    from dynisafety.backends.robocasa.disturbance import l3 as l3_mod

    for mod in (l2_mod, l3_mod):
        src = inspect.getsource(mod)
        assert "dynisafety.backends.libero" not in src
        assert "ENTER_HEIGHT =" not in src
        assert "DWELL_HEIGHT =" not in src
        assert "ARC_PEAK =" not in src


def test_l1_path_spawn_formula_regression() -> None:
    """L1 spawn sits on start→dest, goal_clearance before dest; retries walk back."""
    from dynisafety.backends.robocasa.disturbance._support import (
        l1_path_spawn_xy,
        l1_should_appear,
        path_unit_and_length,
    )

    p_start = np.array([0.0, 0.0], dtype=np.float64)
    p_dest = np.array([2.0, 0.0], dtype=np.float64)
    goal_clearance = 0.60
    retry_step = 0.15
    min_start = 0.24
    spawn0 = l1_path_spawn_xy(
        p_start, p_dest, goal_clearance, retry_step, 0, min_start
    )
    assert spawn0 is not None
    assert spawn0.tolist() == pytest.approx([1.40, 0.0])
    unit, path_len = path_unit_and_length(p_start, p_dest)
    assert unit is not None
    assert unit.tolist() == pytest.approx([1.0, 0.0])
    assert path_len == pytest.approx(2.0)

    spawn1 = l1_path_spawn_xy(
        p_start, p_dest, goal_clearance, retry_step, 1, min_start
    )
    assert spawn1 is not None
    assert spawn1.tolist() == pytest.approx([1.25, 0.0])

    short_dest = np.array([0.50, 0.0], dtype=np.float64)
    clamped = l1_path_spawn_xy(
        p_start, short_dest, goal_clearance, retry_step, 0, min_start
    )
    assert clamped is not None
    assert clamped.tolist() == pytest.approx([0.24, 0.0])

    assert l1_should_appear([1.40, 0.0], [1.40, 0.0], 0.80) is True
    assert l1_should_appear([0.50, 0.0], [1.40, 0.0], 0.80) is False
    assert l1_should_appear([0.70, 0.0], [1.40, 0.0], 0.80) is True


def test_l1_clear_ahead_trigger_contract() -> None:
    """Clear-ahead wait/block helper stays available for spawn-failure flags."""
    from dynisafety.backends.robocasa.disturbance._support import (
        l1_probe_failure_action,
        l1_spawn_failed_flag,
    )

    assert l1_probe_failure_action(True) == "wait"
    assert l1_probe_failure_action(False) == "block"
    assert l1_spawn_failed_flag(dropped=True, motion_armed=True, blocked=False) is False
    assert l1_spawn_failed_flag(dropped=False, motion_armed=True, blocked=False) is True
    assert l1_spawn_failed_flag(dropped=False, motion_armed=False, blocked=False) is False
    assert l1_spawn_failed_flag(dropped=False, motion_armed=False, blocked=True) is True


def test_box_density_from_mass() -> None:
    from dynisafety.backends.robocasa.env import box_density_from_mass

    half = [0.12, 0.12, 0.18]
    mass = 0.5
    volume = 8.0 * 0.12 * 0.12 * 0.18
    assert box_density_from_mass(half, mass) == pytest.approx(mass / volume)


def test_parse_obstacle_rgba() -> None:
    from dynisafety.backends.robocasa.env import parse_obstacle_rgba

    assert parse_obstacle_rgba([1.0, 0.0, 0.0, 1.0]) == [1.0, 0.0, 0.0, 1.0]
    assert parse_obstacle_rgba([1, 0, 0]) == [1.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="RGB or RGBA"):
        parse_obstacle_rgba([1.0, 0.0])
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        parse_obstacle_rgba([2.0, 0.0, 0.0, 1.0])


def test_eval_choices_include_robocasa() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    eval_cli.register_eval_parser(sub)
    eval_parser = sub.choices["eval"]
    backend_action = next(
        a for a in eval_parser._actions if "--backend" in a.option_strings
    )
    assert "robocasa" in (backend_action.choices or ())


def test_resolve_fridge_falls_back_to_present_class() -> None:
    from dynisafety.backends.robocasa.env import resolve_pinned_target_name

    class_by_name = {"fridge_1": "FridgeSideBySide", "sink_1": "Sink"}
    assert resolve_pinned_target_name(class_by_name, "FridgeFrenchDoor") == "fridge_1"


def test_pairwise_skips_box_without_ext_sites() -> None:
    from types import SimpleNamespace

    from dynisafety.backends.robocasa.env import _pairwise_dist_map

    class _Sink:
        pass

    env = SimpleNamespace(
        fixtures={"sink": _Sink(), "dynisafety_l12_obstacle": object()}
    )
    # _Sink class name is not a NavigateKitchen fixture class.
    assert _pairwise_dist_map(env) == {}


def test_xiaomi_yaml_parses() -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "configs" / "models" / "xiaomi.yaml").read_text(encoding="utf-8")
    )
    cfg = model_config_from_resolved(raw)
    assert cfg.name == "xiaomi"
    assert cfg.action_space == "hybrid_mobile_12d"
    assert cfg.backends == ("robocasa",)
    assert cfg.observation_images == ("left", "right", "wrist")
    assert cfg.observation_needs_state is True
    assert cfg.extra["ws_codec"] == "xiaomi"
    assert cfg.extra["policy_cameras"]["wrist"] == "robot0_eye_in_hand"


def test_robocasa365_state_14d() -> None:
    from dynisafety.backends.robocasa.env import robocasa365_state_14d

    obs = {
        "robot0_base_to_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_base_to_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, -0.04], dtype=np.float32),
        "robot0_base_pos": np.array([1.0, 2.0, 0.0], dtype=np.float32),
        "robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    state = robocasa365_state_14d(obs)
    assert state.shape == (14,)
    np.testing.assert_allclose(state[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(state[3:6], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(state[6:8], [0.04, -0.04])
    np.testing.assert_allclose(state[8:11], [1.0, 2.0, 0.0])


def test_bind_run_accepts_xiaomi_policy_cameras() -> None:
    be = RoboCasaBackend()
    be.bind_run(
        config_hash="testhash",
        resolved_data={
            "extra": {
                "robocasa": {
                    "camera": "robot0_agentview_center",
                    "env_class": "NavigateKitchen",
                }
            }
        },
        action_space="hybrid_mobile_12d",
        chunk_size=1,
        observation_images=("left", "right", "wrist"),
        observation_needs_state=True,
        model_extra={
            "policy_cameras": {
                "left": "robot0_agentview_left",
                "right": "robot0_agentview_right",
                "wrist": "robot0_eye_in_hand",
            },
            "policy_camera_width": 256,
            "policy_camera_height": 256,
        },
    )
    assert be._policy_cameras["wrist"] == "robot0_eye_in_hand"
    assert "robot0_eye_in_hand" in be._extra_robocasa["extra_cameras"]
    assert be._extra_robocasa["policy_camera_width"] == 256


def test_smpl_standing_export_is_not_tpose(tmp_path: Path) -> None:
    """Standing LBS must drop T-pose arms (width << 1.6 m) and plant feet at z=0."""
    model = REPO_ROOT / "external" / "assets" / "model.npz"
    if not model.is_file():
        pytest.skip("SMPL-family model.npz is not installed under external/assets")
    from dynisafety.backends.robocasa.smpl_pedestrian import export_standing_mesh

    stats = export_standing_mesh(model, tmp_path / "standing.obj")
    assert 1.50 < stats.height_m < 2.00
    assert stats.width_m < 0.80
    assert stats.depth_m < 0.50
    assert stats.min_z == pytest.approx(0.0, abs=1e-5)
    assert abs(stats.center_xy[0]) < 1e-6
    assert abs(stats.center_xy[1]) < 1e-6
    assert (tmp_path / "standing.obj").stat().st_size > 0


_L12_POOL = (
    "cardboard_stack",
    "kitchen_bin",
    "delivery_crate",
    "laundry_hamper",
    "water_cooler",
)


def _obj_vertices(path: Path) -> list[tuple[float, float, float]]:
    verts: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return verts


def test_apply_l12_visual_survives_contact_alpha_restore() -> None:
    from types import SimpleNamespace

    from dynisafety.backends.robocasa.visuals import apply_l12_visual, reapply_l12_visual

    rgba = np.zeros((3, 4), dtype=np.float64)
    names = {"box": 0, "mesh_cardboard": 1, "mesh_bin": 2}

    class _Model:
        geom_rgba = rgba

        def geom_name2id(self, name: str) -> int:
            return names[str(name)]

    env = SimpleNamespace(
        sim=SimpleNamespace(model=_Model()),
        _dynisafety_l12={
            "visual_geoms": {
                "cardboard_stack": "mesh_cardboard",
                "kitchen_bin": "mesh_bin",
            },
            "visual_rgba": {
                "cardboard_stack": [0.76, 0.60, 0.38, 1.0],
                "kitchen_bin": [0.22, 0.36, 0.30, 1.0],
            },
        },
    )
    apply_l12_visual(env, "kitchen_bin")
    rgba[:, 3] = 0.0
    assert reapply_l12_visual(env) == "kitchen_bin"
    assert rgba[0, 3] == 0.0
    assert rgba[1, 3] == 0.0
    assert rgba[2, 3] == pytest.approx(1.0)
    assert list(rgba[2, :3]) == pytest.approx([0.22, 0.36, 0.30])


def test_select_visual_id_trajectory_order() -> None:
    from dynisafety.backends.robocasa.visuals import select_visual_id

    assert select_visual_id(_L12_POOL, 0, "trajectory_order") == "cardboard_stack"
    assert select_visual_id(_L12_POOL, 1, "trajectory_order") == "kitchen_bin"
    assert select_visual_id(_L12_POOL, 5, "trajectory_order") == "cardboard_stack"
    assert select_visual_id(_L12_POOL, 4, "trajectory_order") == "water_cooler"
    assert select_visual_id(_L12_POOL, 9, "fixed") == "cardboard_stack"
    with pytest.raises(ValueError, match="selection_mode"):
        select_visual_id(_L12_POOL, 0, "closest_visible")


def test_obj_axis_bounds_and_fit_visual_scale_pos(tmp_path: Path) -> None:
    from dynisafety.backends.robocasa.visuals import fit_visual_scale_pos, obj_axis_bounds

    obj = tmp_path / "unit.obj"
    obj.write_text(
        "v -1 0 0\nv 1 0 0\nv 0 -2 0\nv 0 2 0\nv 0 0 2\nv 0 0 4\n",
        encoding="utf-8",
    )
    mins, maxs = obj_axis_bounds(obj)
    assert mins == (-1.0, -2.0, 0.0)
    assert maxs == (1.0, 2.0, 4.0)
    scale, pos = fit_visual_scale_pos(obj, half_z=0.65)
    assert scale == pytest.approx((1.0, 1.0, 1.0))
    assert pos == pytest.approx((0.0, 0.0, -0.65))

    model_xml = tmp_path / "model.xml"
    visual_dir = tmp_path / "visual"
    visual_dir.mkdir()
    nested = visual_dir / "unit.obj"
    nested.write_text(obj.read_text(encoding="utf-8"), encoding="utf-8")
    model_xml.write_text(
        '<mujoco><asset><mesh file="visual/unit.obj" scale="0.12 0.12 0.12"/></asset></mujoco>\n',
        encoding="utf-8",
    )
    scaled, scaled_pos = fit_visual_scale_pos(nested, half_z=0.65)
    assert scaled == pytest.approx((0.12, 0.12, 0.12))
    assert scaled_pos == pytest.approx((0.0, 0.0, -0.65))


def test_l3_shared_frame_keeps_head_above_body(tmp_path: Path) -> None:
    from dynisafety.backends.robocasa.visuals import (
        fit_from_bounds,
        fit_visual_scale_pos,
        union_obj_axis_bounds,
    )

    body = tmp_path / "body.obj"
    head = tmp_path / "head.obj"
    body.write_text("v -0.1 -0.1 0\nv 0.1 0.1 1.4\n", encoding="utf-8")
    head.write_text("v -0.05 -0.05 1.4\nv 0.05 0.05 1.7\n", encoding="utf-8")
    _, pos_body = fit_visual_scale_pos(body, half_z=0.55)
    _, pos_head = fit_visual_scale_pos(head, half_z=0.55)
    assert pos_head[2] < pos_body[2] - 1.0
    mins, maxs = union_obj_axis_bounds([body, head])
    _, pos_shared = fit_from_bounds(mins, maxs, half_z=0.55)
    assert pos_shared == pytest.approx(pos_body)
    assert 1.4 + pos_shared[2] == pytest.approx(0.85)


def test_parse_l12_visual_catalog_reads_texture(tmp_path: Path) -> None:
    from dynisafety.backends.robocasa.visuals import parse_l12_visual_catalog

    mesh = tmp_path / "box.obj"
    tex = tmp_path / "box.png"
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    tex.write_bytes(b"\x89PNG")
    catalog = parse_l12_visual_catalog(
        {
            "l12": {
                "visuals": {
                    "cardboard_stack": {
                        "mesh_path": str(mesh),
                        "texture_path": str(tex),
                        "rgba": [1.0, 1.0, 1.0, 1.0],
                    }
                }
            }
        },
        ["cardboard_stack"],
    )
    assert catalog[0].texture_path == tex
    assert catalog[0].rgba == (1.0, 1.0, 1.0, 1.0)


def test_parse_l3_visual_parts_prefers_local_smpl(tmp_path: Path) -> None:
    from dynisafety.backends.robocasa.visuals import parse_l3_visual_parts

    smpl = tmp_path / "standing.obj"
    smpl.write_text("v 0 0 0\nv 0 0 1\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    fallback = tmp_path / "torso.obj"
    fallback.write_text("v 0 0 0\n", encoding="utf-8")
    parts = parse_l3_visual_parts(
        {
            "mesh_path": str(fallback),
            "mesh_rgba": [0.22, 0.45, 0.62, 1.0],
            "smpl_mesh_path": str(smpl),
            "smpl_mesh_rgba": [0.80, 0.62, 0.50, 1.0],
            "visual_parts": [
                {"name": "torso", "mesh_path": str(fallback), "rgba": [0.22, 0.45, 0.62, 1.0]}
            ],
        }
    )
    assert [p.name for p in parts] == ["smpl_body"]
    assert parts[0].mesh_path == smpl
    missing = parse_l3_visual_parts(
        {
            "mesh_path": str(fallback),
            "mesh_rgba": [0.22, 0.45, 0.62, 1.0],
            "smpl_mesh_path": str(tmp_path / "absent.obj"),
            "visual_parts": [
                {"name": "torso", "mesh_path": str(fallback), "rgba": [0.22, 0.45, 0.62, 1.0]}
            ],
        }
    )
    assert [p.name for p in missing] == ["torso"]


def test_parse_l3_visual_parts_prefers_rocketbox_over_smpl(tmp_path: Path) -> None:
    from dynisafety.backends.robocasa.visuals import parse_l3_visual_parts

    body = tmp_path / "body.obj"
    head = tmp_path / "head.obj"
    body_png = tmp_path / "body.png"
    head_png = tmp_path / "head.png"
    smpl = tmp_path / "standing.obj"
    fallback = tmp_path / "torso.obj"
    for path in (body, head, smpl, fallback):
        path.write_text("v 0 0 0\n", encoding="utf-8")
    body_png.write_bytes(b"\x89PNG\r\n\x1a\n")
    head_png.write_bytes(b"\x89PNG\r\n\x1a\n")
    cfg = {
        "mesh_path": str(fallback),
        "mesh_rgba": [0.22, 0.45, 0.62, 1.0],
        "smpl_mesh_path": str(smpl),
        "smpl_mesh_rgba": [0.80, 0.62, 0.50, 1.0],
        "rocketbox_visuals": [
            {
                "name": "rocketbox_body",
                "mesh_path": str(body),
                "texture_path": str(body_png),
                "rgba": [1.0, 1.0, 1.0, 1.0],
            },
            {
                "name": "rocketbox_head",
                "mesh_path": str(head),
                "texture_path": str(head_png),
                "rgba": [1.0, 1.0, 1.0, 1.0],
            },
        ],
        "visual_parts": [
            {"name": "torso", "mesh_path": str(fallback), "rgba": [0.22, 0.45, 0.62, 1.0]}
        ],
    }
    parts = parse_l3_visual_parts(cfg)
    assert [p.name for p in parts] == ["rocketbox_body", "rocketbox_head"]
    assert parts[0].texture_path == body_png
    missing_tex = parse_l3_visual_parts(
        {
            **cfg,
            "rocketbox_visuals": [
                {
                    "name": "rocketbox_body",
                    "mesh_path": str(body),
                    "texture_path": str(tmp_path / "absent.png"),
                    "rgba": [1.0, 1.0, 1.0, 1.0],
                }
            ],
        }
    )
    assert [p.name for p in missing_tex] == ["smpl_body"]


def test_l12_obstacle_objs_fit_l1_aabb() -> None:
    root = (
        REPO_ROOT
        / "src"
        / "dynisafety"
        / "backends"
        / "robocasa"
        / "assets"
        / "obstacles"
    )
    for name in _L12_POOL:
        verts = _obj_vertices(root / f"{name}.obj")
        assert verts, name
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        zs = [v[2] for v in verts]
        assert min(zs) == pytest.approx(0.0, abs=2e-3)
        assert max(zs) <= 1.30 + 1e-6
        assert max(abs(x) for x in xs) <= 0.12 + 1e-6
        assert max(abs(y) for y in ys) <= 0.12 + 1e-6


def test_l3_clothed_parts_stand_on_floor() -> None:
    root = (
        REPO_ROOT
        / "src"
        / "dynisafety"
        / "backends"
        / "robocasa"
        / "assets"
        / "pedestrian"
    )
    names = ("shoes", "legs", "torso", "arms", "hands", "head", "hair")
    zmin = 1.0
    zmax = -1.0
    for name in names:
        verts = _obj_vertices(root / f"{name}.obj")
        assert verts, name
        zs = [v[2] for v in verts]
        zmin = min(zmin, min(zs))
        zmax = max(zmax, max(zs))
    assert zmin == pytest.approx(0.0, abs=2e-3)
    assert zmax <= 1.70 + 1e-6
