"""Argument handling and bookkeeping of the batch launcher (no simulator)."""
import json

import pytest

from hazardhosts.libero.trajectory_batch import exported, parse_slots, parse_tasks, usable_demos


def test_tasks_and_slots_parse():
    assert parse_tasks("libero_spatial:0,libero_10:2") == [("libero_spatial", 0), ("libero_10", 2)]
    assert len(parse_tasks("all")) == 40
    assert parse_slots("0:2,2:3") == [0, 0, 2, 2, 2]
    with pytest.raises(SystemExit):
        parse_tasks("libero_nope:1")
    with pytest.raises(SystemExit):
        parse_tasks("libero_goal:x")


def test_usable_demos_and_export_state(tmp_path):
    rows = [{"demo": "demo_%d" % i, "replay_success": i != 1} for i in range(5)]
    (tmp_path / "coverage.json").write_text(json.dumps({"demonstrations": rows}))
    assert usable_demos(tmp_path, None) == ["demo_0", "demo_2", "demo_3", "demo_4"]
    assert usable_demos(tmp_path, 2) == ["demo_0", "demo_2"]
    assert not exported(tmp_path, ["demo_0", "demo_2"])
    (tmp_path / "generation_inputs").mkdir()
    (tmp_path / "generation_inputs" / "index.json").write_text(json.dumps({"demos": ["demo_0", "demo_2", "demo_3"]}))
    assert exported(tmp_path, ["demo_0", "demo_2"])
    assert not exported(tmp_path, ["demo_0", "demo_2", "demo_3", "demo_4"])
