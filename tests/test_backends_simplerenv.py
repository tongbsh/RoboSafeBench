"""Unit tests for SimplerEnv backend (no simulator required)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dynisafety.backends.registry import populate_backends
from dynisafety.backends.simplerenv.capabilities import SIMPLERENV_CAPABILITIES
from dynisafety.backends.simplerenv.contact import force_equiv_dt_s
from dynisafety.backends.simplerenv.disturbance.assemble import build_controller
from dynisafety.backends.simplerenv.env import (
    build_reset_options,
    default_asset_dir,
    resolve_simplerenv_root,
)
from dynisafety.cli import eval as eval_cli
from dynisafety.config.loader import LoadRequest, load_config
from dynisafety.core.registry import Registry


def test_simplerenv_capabilities() -> None:
    assert SIMPLERENV_CAPABILITIES.levels == (0, 1, 2, 3)
    assert SIMPLERENV_CAPABILITIES.force_source == "sapien_impulse"


def test_populate_backends_registers_simplerenv() -> None:
    reg: Registry[type] = Registry("backend-test-se")
    populate_backends(reg)
    assert "simplerenv" in reg.available()
    assert reg.get("simplerenv").name == "simplerenv"


def test_ensure_registries_loaded_injects_simplerenv() -> None:
    eval_cli._PLUGINS_LOADED = False  # noqa: SLF001
    eval_cli.ensure_registries_loaded()
    assert "simplerenv" in eval_cli.BACKEND_REGISTRY.available()


def test_list_tasks_bridge_max_steps() -> None:
    from dynisafety.backends.simplerenv.backend import SimplerEnvBackend

    be = SimplerEnvBackend()
    tasks = be.list_tasks("bridge")
    assert [t.task_name for t in tasks] == [
        "bridge_carrot",
        "bridge_stack",
        "bridge_spoon",
        "bridge_eggplant",
    ]
    assert all(t.max_steps == 120 for t in tasks)
    with pytest.raises(ValueError, match="bridge"):
        be.list_tasks("pick_coke_can")


def test_yaml_false_ac4() -> None:
    resolved = load_config(
        LoadRequest(backend="simplerenv", level=2, suite="bridge", model="random")
    )
    assert resolved.run.disturbance.l2.release_after_settle is False
    resolved3 = load_config(
        LoadRequest(backend="simplerenv", level=3, suite="bridge", model="random")
    )
    assert resolved3.run.disturbance.l3.flip_side is False
    assert resolved3.run.disturbance.l3.enable_pre_physics_collision_guard is False


def test_extra_density_and_obb() -> None:
    resolved = load_config(
        LoadRequest(backend="simplerenv", level=3, suite="bridge", model="random")
    )
    extra = resolved.data.get("extra") or {}
    assert float(extra["obstacle_density"]) == 500.0
    assert bool(extra["use_visual_mesh"]) is True
    assert list(extra["obb_half"]) == [0.25, 0.08, 0.12]
    assert list(extra["simplerenv"]["hand_color"]) == [0.78, 0.60, 0.48]


def test_build_controller_rejects_level_4() -> None:
    resolved = load_config(
        LoadRequest(backend="simplerenv", level=3, suite="bridge", model="random")
    )
    with pytest.raises(ValueError, match="not in SimplerEnv capabilities"):
        build_controller(
            4,
            actor=None,
            cfg=resolved.run,
            access=None,  # type: ignore[arg-type]
            reset_options={},
            get_eef_pos_fn=lambda: None,  # type: ignore[arg-type, return-value]
            seed=0,
            resolved_extra={},
        )


def test_build_controller_level0_none() -> None:
    resolved = load_config(
        LoadRequest(backend="simplerenv", level=0, suite="bridge", model="random")
    )
    assert (
        build_controller(
            0,
            actor=None,
            cfg=resolved.run,
            access=None,  # type: ignore[arg-type]
            reset_options={},
            get_eef_pos_fn=lambda: None,  # type: ignore[arg-type, return-value]
            seed=0,
            resolved_extra={},
        )
        is None
    )


def test_force_equiv_dt_impact_window() -> None:
    l3 = load_config(
        LoadRequest(backend="simplerenv", level=3, suite="bridge", model="random")
    )
    dt_l3 = force_equiv_dt_s(
        l3.run, scene_timestep_s=0.002, control_freq=5
    )
    assert dt_l3 == pytest.approx(0.01)
    l2 = load_config(
        LoadRequest(backend="simplerenv", level=2, suite="bridge", model="random")
    )
    dt_l2 = force_equiv_dt_s(
        l2.run, scene_timestep_s=0.002, control_freq=5
    )
    assert dt_l2 == pytest.approx(0.05)


def test_build_reset_options_episode_cycle() -> None:
    resolved = load_config(
        LoadRequest(backend="simplerenv", level=0, suite="bridge", model="random")
    )
    task_extra = {
        "robot_init_x": 0.147,
        "robot_init_y": 0.028,
        "robot_init_quat": [0.0, 0.0, 0.0, 1.0],
    }
    opts = build_reset_options(resolved.run, task_extra, episode_index=25)
    # range [0, 24) → 25 % 24 = 1
    assert opts["obj_init_options"]["episode_id"] == 1
    assert list(opts["robot_init_options"]["init_xy"]) == pytest.approx([0.147, 0.028])


def test_resolve_simplerenv_root_and_assets() -> None:
    ws = Path(__file__).resolve().parents[1]
    root = resolve_simplerenv_root(workspace=ws)
    assert root.is_dir()
    assert (root / "simpler_env").is_dir() or root.name == "SimplerEnv-OpenVLA"
    assets = default_asset_dir()
    assert (assets / "hand_pusher_vis.stl").is_file()
    assert (assets / "hand_pusher_collision.stl").is_file()


def test_wrapper_rejects_level0() -> None:
    from dynisafety.backends.simplerenv.wrapper import DyniSafetyWrapper

    resolved = load_config(
        LoadRequest(backend="simplerenv", level=0, suite="bridge", model="random")
    )
    with pytest.raises(ValueError, match="rejects level=0"):
        DyniSafetyWrapper(
            env=None,  # type: ignore[arg-type]
            level=0,
            run_cfg=resolved.run,
            episode_index=0,
            seed=0,
            simplerenv_root=Path("."),
            workspace=Path("."),
            resolved_extra={},
            task_extra={},
        )


def test_start_inward_just_outside_view_stops_at_frame_edge() -> None:
    from dynisafety.backends.simplerenv.targets import start_inward_just_outside_view

    K = np.array([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]])
    T_cw = np.eye(4)
    shift = start_inward_just_outside_view(
        anchor_xy=np.array([-1.5, 0.0]),
        inward_dir=np.array([1.0, 0.0]),
        start_z=1.0,
        max_inward_m=1.0,
        palm_radius_m=0.05,
        K=K,
        T_cw=T_cw,
        width=200,
        height=200,
        margin_px=12,
    )
    # Left edge of the padded frame is u=12 → X ≈ -0.88 at Z=1; palm front
    # (+0.05) must stay outside, so origin stays ~0.05 further left.
    assert 0.50 < shift < 0.65
    assert start_inward_just_outside_view(
        anchor_xy=np.array([0.0, 0.0]),
        inward_dir=np.array([1.0, 0.0]),
        start_z=1.0,
        max_inward_m=1.0,
        palm_radius_m=0.05,
        K=K,
        T_cw=T_cw,
        width=200,
        height=200,
        margin_px=12,
    ) == pytest.approx(0.0)


def test_no_bridge_max_or_type_bool_in_src() -> None:
    import re

    root = Path(__file__).resolve().parents[1] / "src" / "dynisafety"
    text = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.py"))
    assert "BRIDGE_MAX_EPISODE_STEPS" not in text
    # argparse ``type=bool`` trap (defect #2); ignore numpy ``dtype=bool``.
    assert re.search(r"(?<![a-z_])type=bool\b", text) is None


class _FakeSkinMat:
    def __init__(self) -> None:
        self.base_color: list[float] | None = None
        self.roughness: float | None = None
        self.metallic: float | None = None
        self.specular: float | None = None

    def set_base_color(self, color: list[float]) -> None:
        self.base_color = list(color)

    def set_roughness(self, value: float) -> None:
        self.roughness = float(value)

    def set_metallic(self, value: float) -> None:
        self.metallic = float(value)

    def set_specular(self, value: float) -> None:
        self.specular = float(value)


def test_hand_skin_material_uses_scene_engine_renderer() -> None:
    from dynisafety.backends.simplerenv.actor import _hand_skin_material, _scene_renderer

    created: list[_FakeSkinMat] = []

    class _Renderer:
        def create_material(self) -> _FakeSkinMat:
            mat = _FakeSkinMat()
            created.append(mat)
            return mat

    scene = SimpleNamespace(engine=SimpleNamespace(get_renderer=lambda: _Renderer()))
    assert _scene_renderer(scene) is not None
    mat = _hand_skin_material(scene, [0.78, 0.60, 0.48])
    assert mat is created[0]
    assert mat.base_color == [0.78, 0.60, 0.48, 1.0]
    assert mat.metallic == 0.0
    assert mat.specular == 0.0
    assert mat.roughness == 0.85


def test_apply_hand_skin_color_set_material_not_just_base_color() -> None:
    from dynisafety.backends.simplerenv.actor import apply_hand_skin_color

    bound: list[object] = []
    default = _FakeSkinMat()
    default.set_base_color([1.0, 1.0, 1.0, 1.0])

    class _Shape:
        material = default

        def set_material(self, mat: object) -> None:
            bound.append(mat)

    actor = SimpleNamespace(
        get_visual_bodies=lambda: [SimpleNamespace(get_render_shapes=lambda: [_Shape()])]
    )
    scene = SimpleNamespace(
        engine=SimpleNamespace(get_renderer=lambda: SimpleNamespace(create_material=_FakeSkinMat))
    )
    apply_hand_skin_color(actor, [0.78, 0.60, 0.48], scene=scene)
    assert len(bound) == 1
    assert isinstance(bound[0], _FakeSkinMat)
    assert bound[0].base_color == [0.78, 0.60, 0.48, 1.0]
    assert default.base_color == [1.0, 1.0, 1.0, 1.0]
