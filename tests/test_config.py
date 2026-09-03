"""Config loader tests (M1 / M12 S3a) — no simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from dynisafety.config.loader import (
    ConfigValidationError,
    LoadRequest,
    ResolvedConfig,
    compute_config_hash,
    layer_paths,
    load_config,
    load_resolved_yaml,
    task_layer_stems,
    task_yaml_stem,
)
from dynisafety.config.schema import assert_no_embedded_defaults

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(**kwargs: Any) -> ResolvedConfig:
    return load_config(LoadRequest(**kwargs))


def _matrix() -> list[tuple[str, int, str, str]]:
    """Primary suite per backend × levels 0–3."""
    specs = {
        "libero": ("safelibero_spatial", "random"),
        "simplerenv": ("bridge", "random"),
        "robotwin": ("main20", "random"),
        "robocasa": ("robocasa_nav", "random_mobile"),
    }
    return [
        (backend, level, suite, model)
        for backend, (suite, model) in specs.items()
        for level in range(4)
    ]


def test_schema_no_embedded_defaults_smoke() -> None:
    """Schema MISSING sentinel — authoritative coverage remains in test_contracts."""
    assert_no_embedded_defaults()


def test_merge_priority_model_overrides_base() -> None:
    resolved = _load(
        backend="libero",
        level=3,
        suite="safelibero_spatial",
        model="random",
    )
    assert resolved.provenance["action_space"].endswith("models/random.yaml")
    assert resolved.data["action_space"] == "delta_ee_7d"
    assert resolved.provenance["runtime.seed_stride"].endswith("base.yaml")
    assert int(resolved.data["runtime"]["seed_stride"]) == 2000
    # Higher layers beat base: backend level YAML owns L3 disturbance alphas.
    assert resolved.provenance["disturbance.l3.alpha"].endswith(
        "backends/libero/levels/l3.yaml"
    )


def test_cli_set_override_wins() -> None:
    resolved = _load(
        backend="libero",
        level=0,
        suite="safelibero_spatial",
        model="random",
        overrides=("runtime.episodes=3",),
    )
    assert int(resolved.data["runtime"]["episodes"]) == 3
    assert resolved.provenance["runtime.episodes"] == "cli:--set"


def test_validate_unknown_key_suggests() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(
            backend="libero",
            level=0,
            suite="safelibero_spatial",
            model="random",
            overrides=("runtime.seed_strid=1",),
        )
    msg = str(exc_info.value)
    assert "runtime.seed_strid" in msg
    assert "runtime.seed_stride" in msg


def test_config_hash_idempotent_roundtrip(tmp_path: Path) -> None:
    resolved = _load(
        backend="robotwin",
        level=1,
        suite="main20",
        model="random",
    )
    h1 = compute_config_hash(resolved.data)
    h2 = compute_config_hash(resolved.data)
    assert h1 == h2 == resolved.config_hash

    dump = dict(resolved.data)
    dump["config_hash"] = resolved.config_hash
    path = tmp_path / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(dump, sort_keys=True), encoding="utf-8")
    reloaded = load_resolved_yaml(path)
    assert reloaded.config_hash == resolved.config_hash


def test_yaml_bool_false_not_coerced_true() -> None:
    """Regression for argparse type=bool / YAML false (defect #2 / M6)."""
    resolved = _load(
        backend="simplerenv",
        level=2,
        suite="bridge",
        model="random",
    )
    release = resolved.data["disturbance"]["l2"]["release_after_settle"]
    assert release is False
    assert release is not True
    # CLI override must also preserve YAML false / Python False.
    overridden = _load(
        backend="simplerenv",
        level=2,
        suite="bridge",
        model="random",
        overrides=("disturbance.l2.release_after_settle=false",),
    )
    assert overridden.data["disturbance"]["l2"]["release_after_settle"] is False


@pytest.mark.parametrize(("backend", "level", "suite", "model"), _matrix())
def test_backend_level_suite_loads(
    backend: str, level: int, suite: str, model: str
) -> None:
    resolved = _load(backend=backend, level=level, suite=suite, model=model)
    assert resolved.config_hash
    assert len(resolved.config_hash) == 12
    assert resolved.run.backend == backend
    assert int(resolved.run.level) == level
    assert int(resolved.data["runtime"]["seed_stride"]) == 2000


def test_task_yaml_stem_and_numeric_id_compose() -> None:
    assert task_yaml_stem("safelibero_spatial", 1) == "safelibero_spatial_1"
    assert task_layer_stems("safelibero_spatial", "1") == (
        "1",
        "safelibero_spatial_1",
    )
    assert task_layer_stems("safelibero_spatial", "safelibero_spatial_1") == (
        "safelibero_spatial_1",
    )


def test_spatial_task0_rescale_loads_from_task_yaml() -> None:
    for level in (0, 1, 2, 3):
        resolved = _load(
            backend="libero",
            level=level,
            suite="safelibero_spatial",
            task="0",
            model="random",
        )
        rescale = resolved.data["disturbance"]["obstacle"]["rescale"]
        assert rescale["milk_obstacle_1"] == 0.85
        assert "moka_pot_obstacle_1" not in rescale
        assert resolved.provenance["disturbance.obstacle.rescale.milk_obstacle_1"].endswith(
            "tasks/safelibero_spatial_0.yaml"
        )
        if level == 2:
            assert resolved.data["extra"]["libero"]["l2"]["nudge_landing_xy"] is False
            assert resolved.provenance["extra.libero.l2.nudge_landing_xy"].endswith(
                "tasks/safelibero_spatial_0.yaml"
            )
        else:
            assert resolved.data["extra"]["libero"]["task_object_clearance_xy"] == pytest.approx(
                0.05
            )
    other_task = _load(
        backend="libero",
        level=2,
        suite="safelibero_spatial",
        task="2",
        model="random",
    )
    assert "milk_obstacle_1" not in (other_task.data["disturbance"]["obstacle"].get("rescale") or {})
    assert other_task.data["extra"]["libero"]["l2"]["nudge_landing_xy"] is True
    other_suite = _load(
        backend="libero",
        level=2,
        suite="safelibero_object",
        task="0",
        model="random",
    )
    assert "milk_obstacle_1" not in (other_suite.data["disturbance"]["obstacle"].get("rescale") or {})
    assert other_suite.data["extra"]["libero"]["l2"]["nudge_landing_xy"] is True


def test_object_task1_l2_nudge_isolated_from_other_levels_and_tasks() -> None:
    l2 = _load(
        backend="libero",
        level=2,
        suite="safelibero_object",
        task="1",
        model="random",
    )
    assert l2.data["extra"]["libero"]["l2"]["nudge_landing_xy"] is True
    assert l2.provenance["extra.libero.l2.nudge_landing_xy"].endswith(
        "tasks/safelibero_object_1.yaml"
    )
    assert l2.data["extra"]["libero"]["l2"]["start_z_offset"] == pytest.approx(0.03)
    assert l2.provenance["extra.libero.l2.start_z_offset"].endswith(
        "tasks/safelibero_object_1.yaml"
    )
    assert l2.data["extra"]["libero"]["l2"]["arc_height"] == pytest.approx(0.24)
    assert l2.provenance["extra.libero.l2.arc_height"].endswith(
        "tasks/safelibero_object_1.yaml"
    )
    assert l2.data["disturbance"]["l2"]["arc_height"] == pytest.approx(0.2)
    assert l2.data["extra"]["libero"]["task_object_clearance_xy"] == pytest.approx(0.05)

    l1 = _load(
        backend="libero",
        level=1,
        suite="safelibero_object",
        task="1",
        model="random",
    )
    assert l1.data["extra"]["libero"]["task_object_clearance_xy"] == pytest.approx(0.05)
    assert l1.data["extra"]["libero"].get("l1_nudge_on_penetration_only") is True
    assert l1.data["disturbance"]["l1"]["drop_height"] == pytest.approx(0.01)
    assert "l2" not in l1.data.get("disturbance", {}) or "arc_height" not in (
        l1.data.get("disturbance", {}).get("l2") or {}
    )

    neighbor = _load(
        backend="libero",
        level=2,
        suite="safelibero_object",
        task="3",
        model="random",
    )
    assert neighbor.data["extra"]["libero"]["l2"]["nudge_landing_xy"] is True
    assert neighbor.provenance["extra.libero.l2.nudge_landing_xy"].endswith(
        "levels/l2.yaml"
    )
    assert neighbor.data["disturbance"]["l2"]["arc_height"] == pytest.approx(0.2)
    assert "arc_height" not in neighbor.data["extra"]["libero"]["l2"]
    assert "start_z_offset" not in neighbor.data["extra"]["libero"]["l2"]


def test_object_task2_l2_nudge_matches_l1_and_stays_isolated() -> None:
    l2 = _load(
        backend="libero",
        level=2,
        suite="safelibero_object",
        task="2",
        model="random",
    )
    assert l2.data["extra"]["libero"]["l2"]["nudge_landing_xy"] is True
    assert l2.provenance["extra.libero.l2.nudge_landing_xy"].endswith(
        "tasks/safelibero_object_2.yaml"
    )
    assert l2.data["extra"]["libero"]["task_object_clearance_xy"] == pytest.approx(0.05)
    assert "start_z_offset" not in l2.data["extra"]["libero"]["l2"]
    assert "arc_height" not in l2.data["extra"]["libero"]["l2"]
    assert l2.data["disturbance"]["l2"]["arc_height"] == pytest.approx(0.2)

    l1 = _load(
        backend="libero",
        level=1,
        suite="safelibero_object",
        task="2",
        model="random",
    )
    assert l1.data["extra"]["libero"]["task_object_clearance_xy"] == pytest.approx(0.05)
    assert l1.data["extra"]["libero"].get("l1_nudge_on_penetration_only") is True
    assert l1.data["disturbance"]["l1"]["drop_height"] == pytest.approx(0.01)

    spatial0 = _load(
        backend="libero",
        level=2,
        suite="safelibero_spatial",
        task="0",
        model="random",
    )
    assert spatial0.data["extra"]["libero"]["l2"]["nudge_landing_xy"] is False
    assert spatial0.data["disturbance"]["l2"]["arc_height"] == pytest.approx(0.2)
    assert "arc_height" not in spatial0.data["extra"]["libero"]["l2"]
    assert "start_z_offset" not in spatial0.data["extra"]["libero"]["l2"]


def test_spatial_task1_rescale_loads_from_task_yaml() -> None:
    resolved = _load(
        backend="libero",
        level=1,
        suite="safelibero_spatial",
        task="1",
        model="random",
    )
    rescale = resolved.data["disturbance"]["obstacle"]["rescale"]
    assert rescale["moka_pot_obstacle_1"] == 0.85
    assert rescale["milk_obstacle_1"] == 0.85
    assert rescale["wine_bottle_obstacle_1"] == 0.85
    assert resolved.provenance["disturbance.obstacle.rescale.moka_pot_obstacle_1"].endswith(
        "tasks/safelibero_spatial_1.yaml"
    )
    assert float(resolved.data["disturbance"]["l1"]["trigger_dist_bias"]) == 0.05
    configs_root = REPO_ROOT / "configs"
    paths = layer_paths(
        LoadRequest(
            backend="libero",
            level=1,
            suite="safelibero_spatial",
            task="1",
        ),
        configs_root,
    )
    assert any(p.name == "safelibero_spatial_1.yaml" for p in paths)


def test_libero_long_tasks_enable_home_workspace_onset() -> None:
    for task, stem in (
        ("0", "safelibero_long_0.yaml"),
        ("1", "safelibero_long_1.yaml"),
        ("2", "safelibero_long_2.yaml"),
        ("3", "safelibero_long_3.yaml"),
    ):
        long = _load(
            backend="libero",
            level=1,
            suite="safelibero_long",
            task=task,
            model="random",
        )
        assert long.data["extra"]["libero"]["l1_home_workspace_onset"] is True
        assert long.provenance["extra.libero.l1_home_workspace_onset"].endswith(stem)
        want_pen = stem in ("safelibero_long_2.yaml", "safelibero_long_3.yaml")
        assert long.data["extra"]["libero"]["l1_nudge_on_penetration_only"] is want_pen
        pen_src = long.provenance["extra.libero.l1_nudge_on_penetration_only"]
        assert pen_src.endswith(stem)
        if stem == "safelibero_long_2.yaml":
            assert long.data["extra"]["libero"]["l1_workspace_onset_dist"] == pytest.approx(
                0.20
            )
            assert long.provenance["extra.libero.l1_workspace_onset_dist"].endswith(stem)
        else:
            assert long.data["extra"]["libero"]["l1_workspace_onset_dist"] is None
            assert long.provenance["extra.libero.l1_workspace_onset_dist"].endswith(
                "backends/libero.yaml"
            )
    for suite, task in (
        ("safelibero_spatial", "0"),
        ("safelibero_object", "0"),
        ("safelibero_goal", "0"),
    ):
        other = _load(backend="libero", level=1, suite=suite, task=task, model="random")
        assert other.data["extra"]["libero"]["l1_home_workspace_onset"] is False
        assert other.provenance["extra.libero.l1_home_workspace_onset"].endswith(
            "backends/libero.yaml"
        )
        assert other.data["extra"]["libero"]["l1_nudge_on_penetration_only"] is True
        assert other.provenance["extra.libero.l1_nudge_on_penetration_only"].endswith(
            "backends/libero.yaml"
        )
        assert other.data["extra"]["libero"]["l1_workspace_onset_dist"] is None
        assert other.provenance["extra.libero.l1_workspace_onset_dist"].endswith(
            "backends/libero.yaml"
        )


def test_libero_goal_task2_l1_workspace_onset_dist_25cm() -> None:
    stove = _load(
        backend="libero",
        level=1,
        suite="safelibero_goal",
        task="2",
        model="random",
    )
    assert stove.data["extra"]["libero"]["l1_workspace_onset_dist"] == pytest.approx(0.25)
    assert stove.provenance["extra.libero.l1_workspace_onset_dist"].endswith(
        "safelibero_goal_2.yaml"
    )
    neighbor = _load(
        backend="libero",
        level=1,
        suite="safelibero_goal",
        task="0",
        model="random",
    )
    assert neighbor.data["extra"]["libero"]["l1_workspace_onset_dist"] is None
    assert neighbor.provenance["extra.libero.l1_workspace_onset_dist"].endswith(
        "backends/libero.yaml"
    )


def test_libero_long_l1_trigger_dist_matches_other_suites_15cm() -> None:
    long = _load(
        backend="libero",
        level=1,
        suite="safelibero_long",
        task="0",
        model="random",
    )
    assert float(long.data["disturbance"]["l1"]["trigger_dist"]) == 0.15
    assert long.provenance["disturbance.l1.trigger_dist"].endswith(
        "suites/safelibero_long.yaml"
    )
    for suite in ("safelibero_spatial", "safelibero_object", "safelibero_goal"):
        other = _load(backend="libero", level=1, suite=suite, task="0", model="random")
        assert float(other.data["disturbance"]["l1"]["trigger_dist"]) == 0.15
        assert other.provenance["disturbance.l1.trigger_dist"].endswith(
            "backends/libero/levels/l1.yaml"
        )


def test_obstacle_rescale_rejects_non_positive() -> None:
    with pytest.raises(ConfigValidationError, match="positive number"):
        _load(
            backend="libero",
            level=1,
            suite="safelibero_spatial",
            model="random",
            overrides=("disturbance.obstacle.rescale.moka_pot_obstacle_1=0",),
        )


def test_libero_l2_spawn_diversity_default_on() -> None:
    resolved = _load(
        backend="libero",
        level=2,
        suite="safelibero_spatial",
        model="random",
    )
    spawn = resolved.data["extra"]["libero"]["l2"]["spawn_diversity"]
    assert spawn["enabled"] is True
    assert spawn["linear_z_clearance_m"] == pytest.approx(0.05)
    assert spawn["tail_exclude_waypoints"] == 3
    assert spawn["forbidden_slots"] == []
    assert resolved.provenance["extra.libero.l2.spawn_diversity.enabled"].endswith(
        "backends/libero/levels/l2.yaml"
    )


def test_libero_obstacle_pack_default_on() -> None:
    resolved = _load(
        backend="libero",
        level=0,
        suite="safelibero_spatial",
        model="random",
    )
    pack = resolved.data["extra"]["libero"]["obstacle_pack"]
    assert pack["enabled"] is True
    assert resolved.provenance["extra.libero.obstacle_pack.enabled"].endswith(
        "backends/libero.yaml"
    )


def test_spatial_task1_l2_forbidden_slots_isolated() -> None:
    spatial1 = _load(
        backend="libero",
        level=2,
        suite="safelibero_spatial",
        task="1",
        model="random",
    )
    spawn1 = spatial1.data["extra"]["libero"]["l2"]["spawn_diversity"]
    assert spawn1["forbidden_slots"] == ["right"]
    assert spatial1.provenance["extra.libero.l2.spawn_diversity.forbidden_slots"].endswith(
        "tasks/safelibero_spatial_1.yaml"
    )
    assert spawn1["enabled"] is True

    spatial0 = _load(
        backend="libero",
        level=2,
        suite="safelibero_spatial",
        task="0",
        model="random",
    )
    spawn0 = spatial0.data["extra"]["libero"]["l2"]["spawn_diversity"]
    assert spawn0["forbidden_slots"] == []
    assert spatial0.provenance["extra.libero.l2.spawn_diversity.forbidden_slots"].endswith(
        "backends/libero/levels/l2.yaml"
    )


def test_libero_l3_reach_diversity_default_on() -> None:
    resolved = _load(
        backend="libero",
        level=3,
        suite="safelibero_spatial",
        model="random",
    )
    reach = resolved.data["extra"]["libero"]["l3"]["reach_diversity"]
    assert reach["enabled"] is True
    assert reach["near_alpha"] == pytest.approx(0.30)
    assert reach["mid_alpha"] == pytest.approx(0.50)
    assert reach["forbidden_slots"] == []
    assert reach["forbidden_depths"] == []
    assert resolved.provenance["extra.libero.l3.reach_diversity.enabled"].endswith(
        "backends/libero/levels/l3.yaml"
    )
    assert float(resolved.data["hand"]["mass_kg"]) == pytest.approx(4.5)


def test_libero_l3_reach_diversity_overlay_isolated() -> None:
    disabled = _load(
        backend="libero",
        level=3,
        suite="safelibero_goal",
        task="3",
        model="random",
        overrides=("extra.libero.l3.reach_diversity.enabled=false",),
    )
    assert disabled.data["extra"]["libero"]["l3"]["reach_diversity"]["enabled"] is False
    assert disabled.provenance["extra.libero.l3.reach_diversity.enabled"] == "cli:--set"

    neighbor = _load(
        backend="libero",
        level=3,
        suite="safelibero_goal",
        task="0",
        model="random",
    )
    assert neighbor.data["extra"]["libero"]["l3"]["reach_diversity"]["enabled"] is True

    level2 = _load(
        backend="libero",
        level=2,
        suite="safelibero_spatial",
        task="0",
        model="random",
    )
    assert "reach_diversity" not in ((level2.data.get("extra") or {}).get("libero") or {}).get(
        "l3", {}
    )

    level0 = _load(
        backend="libero",
        level=0,
        suite="safelibero_spatial",
        task="0",
        model="random",
    )
    assert "reach_diversity" not in ((level0.data.get("extra") or {}).get("libero") or {}).get(
        "l3", {}
    )


def test_libero_l2_spawn_diversity_overlay_isolated() -> None:
    disabled = _load(
        backend="libero",
        level=2,
        suite="safelibero_goal",
        task="3",
        model="random",
        overrides=("extra.libero.l2.spawn_diversity.enabled=false",),
    )
    assert disabled.data["extra"]["libero"]["l2"]["spawn_diversity"]["enabled"] is False
    assert disabled.provenance["extra.libero.l2.spawn_diversity.enabled"] == "cli:--set"

    neighbor_task = _load(
        backend="libero",
        level=2,
        suite="safelibero_goal",
        task="0",
        model="random",
    )
    assert neighbor_task.data["extra"]["libero"]["l2"]["spawn_diversity"]["enabled"] is True
    assert neighbor_task.provenance["extra.libero.l2.spawn_diversity.enabled"].endswith(
        "backends/libero/levels/l2.yaml"
    )

    spatial0 = _load(
        backend="libero",
        level=2,
        suite="safelibero_spatial",
        task="0",
        model="random",
    )
    assert spatial0.data["extra"]["libero"]["l2"]["spawn_diversity"]["enabled"] is True
    assert spatial0.data["extra"]["libero"]["l2"]["nudge_landing_xy"] is False


def test_simplerenv_l2_spawn_diversity_default_on() -> None:
    resolved = _load(backend="simplerenv", level=2, suite="bridge", model="random")
    spawn = resolved.data["extra"]["simplerenv"]["l2"]["spawn_diversity"]
    assert spawn["enabled"] is True
    assert spawn["lateral_width_m"] == pytest.approx(0.15)
    assert resolved.provenance["extra.simplerenv.l2.spawn_diversity.enabled"].endswith(
        "backends/simplerenv/levels/l2.yaml"
    )
    assert resolved.data["disturbance"]["l2"]["align_start_y_to_eef"] is False
    assert float(resolved.data["contact"]["force_report_window_s"]) == pytest.approx(0.05)


def test_simplerenv_l3_reach_diversity_default_on() -> None:
    resolved = _load(backend="simplerenv", level=3, suite="bridge", model="random")
    l3 = resolved.data["extra"]["simplerenv"]["l3"]
    assert l3["target_mode"] == "current"
    assert l3["reach_diversity"]["enabled"] is True
    assert resolved.provenance["extra.simplerenv.l3.reach_diversity.enabled"].endswith(
        "backends/simplerenv/levels/l3.yaml"
    )
    assert resolved.data["extra"]["simplerenv"]["l3_force_scheme"] == "dynamic_release"
    assert float(resolved.data["disturbance"]["l3"]["enter_height"]) == pytest.approx(0.08)
    assert float(resolved.data["disturbance"]["l3"]["corner_inward_shift"]) == pytest.approx(0.25)
    assert list(resolved.data["extra"]["simplerenv"]["hand_color"]) == [0.78, 0.60, 0.48]
    assert float(resolved.data["hand"]["mass_kg"]) == pytest.approx(4.5)
    assert float(resolved.data["contact"]["force_report_window_s"]) == pytest.approx(0.01)


def test_robotwin_l2_spawn_diversity_default_on() -> None:
    resolved = _load(backend="robotwin", level=2, suite="main20", model="random")
    spawn = resolved.data["extra"]["robotwin"]["l2"]["spawn_diversity"]
    assert spawn["enabled"] is True
    assert spawn["path_num_samples"] == 64
    assert resolved.data["extra"]["robotwin"]["l2_start_mode"] == "head_on"
    assert resolved.data["extra"]["robotwin"]["l12_force_scheme"] == "kinematic_first_hit"
    assert resolved.data["disturbance"]["l2"]["alpha"] == pytest.approx(0.5)
    assert resolved.data["extra"]["robotwin"]["l2"]["move_start_task_step"] == 20


def test_robotwin_l1_alpha_is_half() -> None:
    resolved = _load(backend="robotwin", level=1, suite="main20", model="random")
    assert resolved.data["disturbance"]["l1"]["alpha"] == pytest.approx(0.5)


def test_robotwin_l3_reach_diversity_default_on() -> None:
    resolved = _load(backend="robotwin", level=3, suite="main20", model="random")
    l3 = resolved.data["extra"]["robotwin"]["l3"]
    assert l3["target_mode"] == "current"
    assert l3["move_start_task_step"] == 20
    assert l3["reach_diversity"]["enabled"] is True
    assert resolved.data["extra"]["robotwin"]["l3_approach_steps"] == 400
    assert resolved.data["extra"]["robotwin"]["video_sim_stride"] == 20
    assert resolved.data["disturbance"]["l3"]["entry_steps"] == 6000
    assert resolved.data["disturbance"]["l3"]["dwell_steps"] == 1
    assert list(resolved.data["extra"]["robotwin"]["l3_spawn_world_xy"]) == [0.0, 0.32]
    assert resolved.data["extra"]["robotwin"]["l3_force_scheme"] == "dynamic_release"
    assert float(resolved.data["hand"]["mass_kg"]) == pytest.approx(4.5)


def test_simplerenv_diversity_overlay_isolated() -> None:
    disabled = _load(
        backend="simplerenv",
        level=2,
        suite="bridge",
        model="random",
        overrides=("extra.simplerenv.l2.spawn_diversity.enabled=false",),
    )
    assert disabled.data["extra"]["simplerenv"]["l2"]["spawn_diversity"]["enabled"] is False
    neighbor = _load(backend="simplerenv", level=2, suite="bridge", model="random")
    assert neighbor.data["extra"]["simplerenv"]["l2"]["spawn_diversity"]["enabled"] is True
    level3 = _load(backend="simplerenv", level=3, suite="bridge", model="random")
    assert "spawn_diversity" not in ((level3.data.get("extra") or {}).get("simplerenv") or {}).get(
        "l2", {}
    )


def test_robotwin_diversity_overlay_isolated() -> None:
    disabled = _load(
        backend="robotwin",
        level=3,
        suite="main20",
        model="random",
        overrides=(
            "extra.robotwin.l3.target_mode=path_intercept",
            "extra.robotwin.l3.reach_diversity.enabled=false",
        ),
    )
    assert disabled.data["extra"]["robotwin"]["l3"]["target_mode"] == "path_intercept"
    assert disabled.data["extra"]["robotwin"]["l3"]["reach_diversity"]["enabled"] is False
    neighbor = _load(backend="robotwin", level=3, suite="main20", model="random")
    assert neighbor.data["extra"]["robotwin"]["l3"]["target_mode"] == "current"
    assert neighbor.data["extra"]["robotwin"]["l3"]["reach_diversity"]["enabled"] is True
    level2 = _load(backend="robotwin", level=2, suite="main20", model="random")
    assert "reach_diversity" not in ((level2.data.get("extra") or {}).get("robotwin") or {}).get(
        "l3", {}
    )


def test_runtime_seed_stride_is_2000_from_yaml() -> None:
    """§0.3 / iron law 1: seed_stride lives only in YAML (base)."""
    base = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    assert int(base["runtime"]["seed_stride"]) == 2000
    resolved = _load(backend="libero", level=0, suite="safelibero_spatial", model="random")
    assert int(resolved.data["runtime"]["seed_stride"]) == 2000
    assert resolved.provenance["runtime.seed_stride"].endswith("base.yaml")


def test_pi05_oracle_swr_aegis_overlays_without_clobbering_backend() -> None:
    """God's-eye SWR+AEGIS model keeps paper_main LIBERO knobs (diversity / pack)."""
    resolved = _load(
        backend="libero",
        level=2,
        suite="safelibero_spatial",
        task=task_yaml_stem("safelibero_spatial", 0),
        model="pi05_oracle_swr_aegis",
    )
    extra = resolved.data["extra"]
    libero = extra["libero"]
    assert extra["feasibility_condition"] == "oracle_swr_aegis"
    assert libero["oracle"]["enabled"] is True
    assert libero["aegis"]["enabled"] is True
    assert libero["aegis"]["obstacle_source"] == "sim_gt"
    assert libero["aegis"]["activate_after"] == "settled_or_done"
    assert libero["l2"]["spawn_diversity"]["enabled"] is True
    assert libero["obstacle_pack"]["enabled"] is True
    assert resolved.run.model == "pi05_oracle_swr_aegis"
    assert resolved.provenance["extra.feasibility_condition"].endswith(
        "pi05_oracle_swr_aegis.yaml"
    )
    assert resolved.provenance["extra.libero.oracle.enabled"].endswith(
        "pi05_oracle_swr_aegis.yaml"
    )
    assert resolved.provenance["extra.libero.l2.spawn_diversity.enabled"].endswith(
        "backends/libero/levels/l2.yaml"
    )
    assert resolved.provenance["extra.libero.obstacle_pack.enabled"].endswith(
        "backends/libero.yaml"
    )


def test_capture_disabled_by_default_from_base_yaml() -> None:
    """WRM sidecar is off unless explicitly enabled; default lives only in YAML."""
    base = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    assert base["capture"]["enabled"] is False
    resolved = _load(backend="libero", level=0, suite="safelibero_spatial", model="random")
    assert resolved.data["capture"]["enabled"] is False
    assert resolved.run.capture.enabled is False
    assert resolved.provenance["capture.enabled"].endswith("base.yaml")
    on = _load(
        backend="libero",
        level=0,
        suite="safelibero_spatial",
        model="random",
        overrides=("capture.enabled=true",),
    )
    assert on.data["capture"]["enabled"] is True
    assert on.run.capture.enabled is True
    assert on.provenance["capture.enabled"] == "cli:--set"
