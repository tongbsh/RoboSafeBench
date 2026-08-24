"""P0/P1: L1/L2 camera exclude, rising-edge events, spawn record, jsonl recount."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dynisafety.backends.robotwin.contact import (
    filter_collision_names,
    partner_name_excluded,
    rising_edge_contact,
    xyz_list,
)
from dynisafety.backends.robotwin.disturbance.manager import DisturbanceManager


def _load_recount_mod():
    path = Path(__file__).resolve().parents[1] / "tools" / "recount_robotwin_contact.py"
    spec = importlib.util.spec_from_file_location("recount_robotwin_contact", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_partner_name_excluded_camera() -> None:
    assert partner_name_excluded("left_camera", ["camera"])
    assert partner_name_excluded("right_camera", ["camera"])
    assert not partner_name_excluded("fl_link7", ["camera"])
    assert not partner_name_excluded("001_bottle", ["camera"])


def test_filter_collision_names_drops_camera_only() -> None:
    names = {"left_camera", "fl_link7", "001_bottle", "right_camera"}
    kept = filter_collision_names(names, ["camera"])
    assert kept == {"fl_link7", "001_bottle"}
    assert filter_collision_names(names, []) == names


def test_rising_edge_contact_counts_once_per_run() -> None:
    prev = False
    events = 0
    steps = 0
    for hit in (False, True, True, True, False, True):
        is_event, prev = rising_edge_contact(prev, hit)
        if hit:
            steps += 1
        if is_event:
            events += 1
    assert steps == 4
    assert events == 2


def test_xyz_list_json_safe() -> None:
    assert xyz_list(np.array([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        xyz_list([1.0, 2.0])


def test_official_targets_exclude_camera_on_l1_l2_not_via_l12_on_l3() -> None:
    cfg = SimpleNamespace(
        dynamic_level=2,
        l12_collision_exclude_substrings=("camera",),
        l3_ignore_collision_link_substrings=(),
        obstacle_half_size=(0.03, 0.03, 0.05),
        obstacle_scale=1.0,
    )
    task = SimpleNamespace(
        robot=SimpleNamespace(left_entity=SimpleNamespace(get_links=lambda: []), right_entity=SimpleNamespace(get_links=lambda: []))
    )
    # Avoid constructor robot walk: inject after a stub init path.
    mgr = object.__new__(DisturbanceManager)
    mgr.task = task
    mgr.config = cfg
    mgr._collision_target_names = {"left_camera", "fl_link6", "box"}
    assert mgr._official_collision_target_names() == {"fl_link6", "box"}
    cfg.dynamic_level = 3
    # L3 does not reuse the L12 list; empty L3 ignore keeps cameras.
    assert mgr._official_collision_target_names() == {"left_camera", "fl_link6", "box"}
    cfg.l3_ignore_collision_link_substrings = ("camera",)
    assert mgr._official_collision_target_names() == {"fl_link6", "box"}


def test_recount_no_cam_and_force_flags() -> None:
    recount = _load_recount_mod()
    contact_flags = recount.contact_flags
    summarize = recount.summarize
    camera_only = {
        "success": True,
        "contact_steps": 40,
        "peak_force_n": 0.0,
        "contact_partners": {"right_camera": 40},
    }
    arm_hit = {
        "success": False,
        "contact_steps": 10,
        "peak_force_n": 8.0,
        "contact_partners": {"fl_link7": 10, "left_camera": 3},
    }
    flags = contact_flags(camera_only, exclude_substrings=("camera",))
    assert flags["official"] is True
    assert flags["no_cam"] is False
    assert flags["force_eps"] is False
    flags2 = contact_flags(arm_hit, exclude_substrings=("camera",))
    assert flags2["no_cam"] is True
    assert flags2["force_1n"] is True

    summary = summarize([camera_only, arm_hit], exclude_substrings=("camera",))
    assert summary["n"] == 2
    assert summary["cr_official"] == 1.0
    assert summary["cr_no_cam"] == 0.5
    assert summary["ssr_official"] == 0.0
    assert summary["ssr_no_cam"] == 0.5
