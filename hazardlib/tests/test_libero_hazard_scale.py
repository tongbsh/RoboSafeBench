"""D21 hazard scale axis: level selection by the assembled config directory, the root preparation and the
physical-consequence diagnostics of the generation report (no simulator)."""
import copy
import json
import math
import os

import pytest

from hazardhosts.libero import hazard_scale as hs
from hazardhosts.libero.pilot import assemble_config
from hazardhosts.libero.trajectory_candidates import (consequence_by_family, consequence_record,
                                                      consequence_statistics, consequence_table)
from hazardlib.config import default_config_dir, load_config
from hazardlib.semantics.event_log import HazardEventLog
from hazardlib.spec.hazard_spec import AgentSpec

QUANTILES = [0.1, 0.5, 0.9]


def test_levels_are_the_declared_pilot_cubes_with_plausible_density():
    default, levels = hs.load_levels()
    assert default == "S" and sorted(levels) == ["L", "M", "S"]
    assert (levels["S"]["half_extent_m"], levels["S"]["mass_kg"]) == (0.02, 0.05)
    assert (levels["M"]["half_extent_m"], levels["M"]["mass_kg"]) == (0.035, 0.25)
    assert (levels["L"]["half_extent_m"], levels["L"]["mass_kg"]) == (0.05, 0.8)
    for spec in levels.values():
        density = spec["mass_kg"] / (2 * spec["half_extent_m"]) ** 3
        assert 500 < density < 1000                                  # a plausible solid, not a dense small block
        # D20 b scoring radius: bounding radius of the box + the host's eef contact radius, rounded up to 5 mm
        bound = math.sqrt(3) * spec["half_extent_m"] + 0.02
        assert bound <= spec["coverage_radius_m"] < bound + 0.005 + 1e-9


def test_default_level_is_the_unchanged_host_config_and_the_trajectory_settings():
    host = load_config("libero")
    member = host["pool"]["members"][0]
    spec = hs.level_spec("S")[1]
    assert member["half_extents_m"] == [spec["half_extent_m"]] * 3 and member["mass_kg"] == spec["mass_kg"]
    assert host["scenario"]["probe_agent"]["half_extents_m"] == member["half_extents_m"]
    assert host["scenario"]["strike"]["radius_m"] == spec["strike_radius_m"]
    settings = load_config("trajectory_analysis", config_dir=default_config_dir() / "pilot/libero")
    assert hs.settings_for(settings, "S") == settings          # same generation hash as before D21


def test_scaled_config_only_changes_the_pool_member_and_the_probe_agent():
    host = load_config("libero")
    original = copy.deepcopy(host)
    scaled = hs.scaled_host_config(host, hs.level_spec("L")[1])
    assert host == original                                    # the input is not modified
    assert scaled["pool"]["members"][0]["half_extents_m"] == [0.05] * 3
    assert scaled["pool"]["members"][0]["mass_kg"] == 0.8
    assert scaled["scenario"]["probe_agent"]["half_extents_m"] == [0.05] * 3
    assert scaled["scenario"]["probe_agent"]["mass_kg"] == 0.8
    assert scaled["scenario"]["strike"]["radius_m"] == hs.level_spec("L")[1]["strike_radius_m"] != \
        host["scenario"]["strike"]["radius_m"]
    for config in (host, scaled):                              # everything else (the strike body and direction)
        config["scenario"]["strike"]["radius_m"] = None
        config["pool"]["members"][0].update(half_extents_m=None, mass_kg=None)
        config["scenario"]["probe_agent"].update(half_extents_m=None, mass_kg=None)
    assert scaled == host
    host["pool"]["members"].append(dict(host["pool"]["members"][0]))
    with pytest.raises(ValueError):
        hs.scaled_host_config(host, hs.level_spec("M")[1])


def test_assembled_config_selects_the_level(tmp_path):
    plain = assemble_config(tmp_path / "plain")
    assert not (plain / hs.MARKER).exists() and hs.config_level(plain) == "S"
    explicit = assemble_config(tmp_path / "s", "S")
    assert explicit.joinpath("libero.yaml").read_bytes() == plain.joinpath("libero.yaml").read_bytes()
    assert explicit.joinpath("libero.yaml").read_bytes() == default_config_dir().joinpath("libero.yaml").read_bytes()
    assert not (explicit / hs.MARKER).exists()
    for level, half, mass in (("M", 0.035, 0.25), ("L", 0.05, 0.8)):
        directory = assemble_config(tmp_path / level, level)
        assert hs.config_level(directory) == level
        cfg = load_config("libero", config_dir=directory)
        member = cfg["pool"]["members"][0]
        assert member["name"] == "hazardlib_obstacle_0"        # same body and joint: same state vector layout
        assert member["half_extents_m"] == [half] * 3 and member["mass_kg"] == mass
        assert cfg["scenario"]["strike"]["radius_m"] == hs.level_spec(level)[1]["strike_radius_m"]
        agent = AgentSpec.from_dict(cfg["scenario"]["probe_agent"])
        assert list(agent.half_extents_m) == [half] * 3 and agent.mass_kg == mass
        for name in ("runtime.yaml", "gates.yaml", "trajectory_analysis.yaml"):   # nothing else differs
            assert (directory / name).read_bytes() == (plain / name).read_bytes()
    with pytest.raises(ValueError):
        assemble_config(tmp_path / "M", None)                  # a level directory is never silently reset
    with pytest.raises(ValueError):
        hs.level_spec("XL")


def test_settings_of_a_level_differ_only_in_the_coverage_radius():
    settings = load_config("trajectory_analysis", config_dir=default_config_dir() / "pilot/libero")
    for level in ("M", "L"):
        changed = hs.settings_for(settings, level)
        assert changed["coverage_radius_m"] == hs.level_spec(level)[1]["coverage_radius_m"]
        assert {k: v for k, v in changed.items() if k != "coverage_radius_m"} == \
            {k: v for k, v in settings.items() if k != "coverage_radius_m"}


def _linked_root(tmp_path, name, scene):
    source, root = tmp_path / "source", tmp_path / name
    source.mkdir(exist_ok=True)
    root.mkdir()
    if not (source / "scene.json").exists():
        (source / "scene.json").write_text(json.dumps(scene))
    os.link(str(source / "scene.json"), str(root / "scene.json"))      # cp -al, as the batch launcher does
    return source, root


def test_prepare_root_gives_the_root_its_level_without_touching_the_linked_source(tmp_path):
    agent = {"kind": "object", "shape": "box", "half_extents_m": [0.02, 0.02, 0.02], "mass_kg": 0.05}
    source, root_s = _linked_root(tmp_path, "root_S", {"task_id": "t", "agent": agent})
    before = (source / "scene.json").read_text()
    assert hs.prepare_root(root_s, "S") == "S"
    assert (root_s / "scene.json").read_text() == before and not (root_s / "config" / hs.MARKER).exists()
    _, root = _linked_root(tmp_path, "root_M", None)
    for _ in range(2):                                                  # idempotent
        assert hs.prepare_root(root, "M") == "M"
    scene = json.loads((root / "scene.json").read_text())
    assert scene["agent"]["half_extents_m"] == [0.035] * 3 and scene["agent"]["mass_kg"] == 0.25
    assert scene["hazard_scale"] == "M" and scene["agent"]["kind"] == "object" and scene["task_id"] == "t"
    assert hs.config_level(root / "config") == "M"
    assert (source / "scene.json").read_text() == before                # the source is a separate inode now
    with pytest.raises(ValueError):
        hs.prepare_root(root, "L")                                      # one root, one level


def _item(failed, impulse=None, scene_contact=None, selected=True, with_consequence=True):
    item = {"selected": selected, "nominal_hit": {"success": not failed, "contact": True}}
    if with_consequence:
        item["consequence"] = {"task_success": not failed, "impulse_analytic_ns": impulse,
                               "first_scene_contact_t_s": scene_contact, "v_rel_normal_mps": None}
    return item


def test_consequence_record_reads_the_nominal_hit_log():
    log = HazardEventLog.new("i", task_success=False, impulse_analytic_ns=0.42, v_rel_normal_mps=0.7,
                             hazard_mass_kg=0.8, first_scene_contact_t_s=2.5)
    record = consequence_record(log)
    assert record["task_success"] is False and record["impulse_analytic_ns"] == 0.42
    assert record["v_rel_normal_mps"] == 0.7 and record["hazard_mass_kg"] == 0.8
    assert record["first_scene_contact_t_s"] == 2.5 and record["contact_duration_s"] is None
    json.dumps(record, allow_nan=False)


def test_consequence_statistics_count_task_failures_of_admitted_instances_only():
    entries = [({}, [_item(True, 0.5, 1.0, selected=False), _item(False, 0.2, selected=True)]),
               ({}, [_item(True, 0.6, 3.0)]),
               ({}, [_item(True, 0.4), {"rejection": "scene_audit:x"}]),
               ({}, [{"rejection": "visibility:x"}]),                    # nothing admitted: not counted
               ({}, [_item(False, with_consequence=False)])]            # a ledger from before D21
    stats = consequence_statistics(entries, QUANTILES)
    assert stats["n_admitted"] == 4 and stats["reference_task_failed"] == 2
    assert stats["failure_rate"] == 0.5 and stats["n_scene_contact"] == 1
    assert stats["impulse_ns"]["n"] == 3 and stats["impulse_ns"]["quantiles"]["0.5"] == pytest.approx(0.4)
    empty = consequence_statistics([({}, [])], QUANTILES)
    assert empty["n_admitted"] == 0 and empty["failure_rate"] is None and empty["impulse_ns"]["n"] == 0


def test_consequence_rollup_and_table():
    cells = {"intrusion.reach@el00": {"1.5": consequence_statistics([({}, [_item(True, 0.3)])], QUANTILES),
                                      "3": consequence_statistics([({}, [_item(False, 0.1)])], QUANTILES)},
             "intrusion.reach@el30": {"1.5": consequence_statistics([({}, [_item(True)])], QUANTILES)},
             "intrusion.fall@vertical": {"1.5": consequence_statistics([({}, [])], QUANTILES)}}
    by_family = consequence_by_family(cells)
    assert by_family["intrusion.reach"] == {"n_admitted": 3, "reference_task_failed": 2, "failure_rate": 2 / 3}
    assert by_family["intrusion.fall"]["failure_rate"] is None
    assert by_family["all"]["n_admitted"] == 3
    table = consequence_table(cells, by_family)
    assert "2/3 (67%)" in table and "intrusion.fall@vertical" in table and "| all | 3 |" in table
