"""Unit tests for SimplerEnv backend (no simulator required)."""

from __future__ import annotations

from pathlib import Path

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
    resolved = load_config(
        LoadRequest(backend="simplerenv", level=3, suite="bridge", model="random")
    )
    dt = force_equiv_dt_s(
        resolved.run, scene_timestep_s=0.002, control_freq=5
    )
    assert dt == pytest.approx(0.05)


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


def test_no_bridge_max_or_type_bool_in_src() -> None:
    import re

    root = Path(__file__).resolve().parents[1] / "src" / "dynisafety"
    text = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.py"))
    assert "BRIDGE_MAX_EPISODE_STEPS" not in text
    # argparse ``type=bool`` trap (defect #2); ignore numpy ``dtype=bool``.
    assert re.search(r"(?<![a-z_])type=bool\b", text) is None
