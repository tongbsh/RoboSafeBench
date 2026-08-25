"""Unit tests for the additive LIBERO GSO obstacle pack (no simulator)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from dynisafety.backends.libero.obstacle_pack import (
    align_carrier_quat,
    default_pack_root,
    parse_obstacle_pack,
    plan_for_task,
    planned_instance,
    resolve_obstacle_xml,
    rewrite_obstacle_xml_path,
    similar_standing,
    steal_episode_names,
    with_remesh,
)
from dynisafety.config.loader import LoadRequest, load_config

_ASSETS = (
    "gso_weston_tonic_22",
    "gso_weston_tonic_33",
    "gso_ace_mug",
    "gso_threshold_mug",
)
_RATIOS = {
    "height_ratio_min": 0.70,
    "height_ratio_max": 1.43,
    "footprint_ratio_min": 0.70,
    "footprint_ratio_max": 1.43,
}


def _pack_cfg(*, enabled: bool, assets: list[str] | None = None) -> dict:
    return {
        "obstacle_pack": {
            "enabled": enabled,
            "assets": list(assets if assets is not None else _ASSETS),
            **_RATIOS,
        }
    }


def _touch_meshes(root: Path, assets: tuple[str, ...] = _ASSETS) -> None:
    for asset in assets:
        mesh = root / "meshes" / asset
        mesh.mkdir(parents=True)
        (mesh / "model.obj").write_text("v 0 0 0\n")


def test_parse_missing_block_is_disabled() -> None:
    assert parse_obstacle_pack({}) is None
    assert parse_obstacle_pack(None) is None


def test_parse_requires_bool_and_assets() -> None:
    with pytest.raises(TypeError, match="enabled"):
        parse_obstacle_pack(
            {"obstacle_pack": {"enabled": "true", "assets": list(_ASSETS), **_RATIOS}}
        )
    with pytest.raises(ValueError, match="missing keys"):
        parse_obstacle_pack({"obstacle_pack": {"enabled": False}})
    with pytest.raises(TypeError, match="assets"):
        parse_obstacle_pack(
            {
                "obstacle_pack": {
                    "enabled": False,
                    "assets": "gso_weston_tonic_22",
                    **_RATIOS,
                }
            }
        )


def test_disabled_pack_does_not_require_meshes(tmp_path: Path) -> None:
    pack = parse_obstacle_pack(_pack_cfg(enabled=False), root=tmp_path)
    assert pack is not None
    assert pack.enabled is False
    assert resolve_obstacle_xml("moka_pot_obstacle", pack) is None
    assert resolve_obstacle_xml("wine_bottle_obstacle", pack) is None


def test_enabled_pack_without_remesh_keeps_stock(tmp_path: Path) -> None:
    _touch_meshes(tmp_path)
    pack = parse_obstacle_pack(_pack_cfg(enabled=True), root=tmp_path)
    assert pack is not None
    assert resolve_obstacle_xml("moka_pot_obstacle", pack) is None
    assert resolve_obstacle_xml("wine_bottle_obstacle", pack) is None


def test_enabled_pack_missing_mesh_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="GSO meshes missing"):
        parse_obstacle_pack(_pack_cfg(enabled=True), root=tmp_path)


def test_rewrite_only_remeshed_obstacle_objects_paths(tmp_path: Path) -> None:
    _touch_meshes(tmp_path)
    pack = with_remesh(
        parse_obstacle_pack(_pack_cfg(enabled=True), root=tmp_path),
        {"moka_pot_obstacle": "gso_weston_tonic_22"},
    )
    stock = (
        "/sim/libero/libero/assets/obstacle_objects/"
        "moka_pot_obstacle/moka_pot_obstacle.xml"
    )
    other = (
        "/sim/libero/libero/assets/stable_scanned_objects/"
        "white_bowl/white_bowl.xml"
    )
    wine = (
        "/sim/libero/libero/assets/obstacle_objects/"
        "wine_bottle_obstacle/wine_bottle_obstacle.xml"
    )
    rewritten = rewrite_obstacle_xml_path(stock, pack)
    assert rewritten != stock
    assert rewritten.endswith("moka_pot_obstacle/gso_weston_tonic_22.xml")
    assert Path(rewritten).is_file()
    assert rewrite_obstacle_xml_path(wine, pack) == wine
    assert rewrite_obstacle_xml_path(other, pack) == other
    assert rewrite_obstacle_xml_path(stock, None) == stock


def test_align_carrier_quat_stands_y_up_milk_on_z_up_official() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    same = align_carrier_quat(
        "wine_bottle_obstacle_1", "moka_pot_obstacle_1", identity
    )
    assert same == pytest.approx(identity)
    q = align_carrier_quat(
        "yellow_book_obstacle_1", "milk_obstacle_1", identity
    )
    # +90° about X: milk +Y → book +Z
    assert q == pytest.approx(np.array([0.70710678, 0.70710678, 0.0, 0.0]), abs=1e-5)


def test_similar_standing_window_is_symmetric() -> None:
    wine = (6.9, 6.9, 25.4)
    weston = (6.3, 6.4, 19.9)
    mug = (9.1, 12.6, 13.5)
    kwargs = dict(
        height_lo=0.70,
        height_hi=1.43,
        footprint_lo=0.70,
        footprint_hi=1.43,
    )
    assert similar_standing(weston, wine, **kwargs)
    assert similar_standing(wine, weston, **kwargs)
    assert not similar_standing(mug, wine, **kwargs)


def test_steal_keeps_every_official_type() -> None:
    official = ("wine_bottle_obstacle_1",) * 50
    extras = ("moka_pot_obstacle_1", "white_storage_box_obstacle_1")
    planned = steal_episode_names(official, extras)
    counts = Counter(planned)
    assert set(counts) == {"wine_bottle_obstacle_1", *extras}
    assert counts["wine_bottle_obstacle_1"] >= counts[extras[0]]
    assert sum(counts.values()) == 50


def test_planned_instance_only_swaps_extras() -> None:
    from dynisafety.backends.libero.obstacle_pack import TaskPackPlan

    plan = TaskPackPlan(
        remesh={},
        extras=("moka_pot_obstacle_1",),
        planned=("wine_bottle_obstacle_1", "moka_pot_obstacle_1"),
    )
    assert planned_instance(plan, 0, "wine_bottle_obstacle_1") == "wine_bottle_obstacle_1"
    assert planned_instance(plan, 1, "wine_bottle_obstacle_1") == "moka_pot_obstacle_1"
    assert planned_instance(plan, 0, "red_coffee_mug_obstacle_1") == "red_coffee_mug_obstacle_1"
    assert planned_instance(None, 0, "wine_bottle_obstacle_1") == "wine_bottle_obstacle_1"


def test_object_suite_adds_wine_like_extras_only() -> None:
    pack = parse_obstacle_pack(_pack_cfg(enabled=True), root=default_pack_root())
    plan = plan_for_task(
        pack,
        "safelibero_object",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    )
    assert plan is not None
    assert "wine_bottle_obstacle" not in plan.remesh
    assert plan.remesh == {
        "moka_pot_obstacle": "gso_weston_tonic_22",
        "white_storage_box_obstacle": "gso_weston_tonic_33",
    }
    assert set(plan.extras) == {
        "moka_pot_obstacle_1",
        "white_storage_box_obstacle_1",
    }
    counts = Counter(plan.planned)
    assert "wine_bottle_obstacle_1" in counts
    assert set(plan.extras).issubset(counts)


def test_spatial_full_slots_add_nothing() -> None:
    pack = parse_obstacle_pack(_pack_cfg(enabled=True), root=default_pack_root())
    plan = plan_for_task(
        pack,
        "safelibero_spatial",
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    )
    assert plan is not None
    assert plan.extras == ()
    assert plan.remesh == {}
    assert "wine_bottle_obstacle_1" in plan.planned


def test_shipped_meshes_and_identities_exist() -> None:
    root = default_pack_root()
    for asset in _ASSETS:
        assert (root / "meshes" / asset / "model.obj").is_file(), asset
    assert (root / "meshes" / "gso_ace_mug" / "texture.png").is_file()
    identities = root / "official_identities.json"
    assert identities.is_file()


def test_cli_can_enable_shipped_pack() -> None:
    resolved = load_config(
        LoadRequest(
            backend="libero",
            level=1,
            suite="safelibero_object",
            model="random",
            overrides=("extra.libero.obstacle_pack.enabled=true",),
        )
    )
    libero = (resolved.data.get("extra") or {}).get("libero") or {}
    pack = parse_obstacle_pack(libero)
    assert pack is not None
    assert pack.enabled is True
    plan = plan_for_task(
        pack,
        "safelibero_object",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    )
    assert plan is not None
    assert plan.extras
    assert resolve_obstacle_xml("wine_bottle_obstacle", pack) is None
    remeshed = with_remesh(pack, plan.remesh)
    assert resolve_obstacle_xml("moka_pot_obstacle", remeshed) is not None
    assert resolve_obstacle_xml("wine_bottle_obstacle", remeshed) is None
