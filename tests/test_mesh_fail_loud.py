"""Missing disturbance meshes must fail, not silently become a box."""

from __future__ import annotations

from pathlib import Path

import pytest

from dynisafety.backends.robotwin.disturbance.manager import require_hand_mesh_path
from dynisafety.backends.simplerenv.actor import require_obstacle_collision_file
from dynisafety.core.rollout import SETUP_INVALID_KEY, setup_invalid_extra


def test_require_hand_mesh_path_empty() -> None:
    with pytest.raises(FileNotFoundError, match="empty"):
        require_hand_mesh_path("")


def test_require_hand_mesh_path_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="hand mesh not found"):
        require_hand_mesh_path("no_such_hand.stl", repo_root=tmp_path)


def test_require_hand_mesh_path_ok(tmp_path: Path) -> None:
    mesh = tmp_path / "hand.stl"
    mesh.write_bytes(b"solid")
    assert require_hand_mesh_path(str(mesh)) == mesh


def test_require_obstacle_collision_missing(tmp_path: Path) -> None:
    model_dir = tmp_path / "coke_can"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="collision mesh missing"):
        require_obstacle_collision_file(model_dir)


def test_require_obstacle_collision_ok(tmp_path: Path) -> None:
    model_dir = tmp_path / "coke_can"
    model_dir.mkdir()
    col = model_dir / "collision.obj"
    col.write_text("v 0 0 0\n")
    assert require_obstacle_collision_file(model_dir) == col


def test_setup_invalid_extra_shape() -> None:
    extra = setup_invalid_extra("no_visible_obstacle", extra={"active_obstacle": None})
    assert extra[SETUP_INVALID_KEY] is True
    assert extra["setup_invalid_reason"] == "no_visible_obstacle"
    assert extra["active_obstacle"] is None
