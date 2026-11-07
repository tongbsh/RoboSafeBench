"""Shared fixtures. Sample values are arbitrary test data, not thresholds."""
import copy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "hazard"


def _hex(ch):
    return ch * 64


def hazard_spec_dict():
    return {
        "spec_version": "1.0",
        "id": "fall_basic",
        "family": "intrusion.fall",
        "agent": {"kind": "object", "shape": "box", "half_extents_m": [0.03, 0.03, 0.03], "mass_kg": 0.2},
        "motion": {"primitive": "free_fall", "params": {"height_m": 0.4, "tags": ["a", "b"]}},
        "exposure": {"anchor": "nominal_ee_path", "offset_m": [0.0, 0.1, 0.0],
                     "anchor_phase": {"phase": "approach", "frac": 0.6}},
        "timing": {"mode": "nominal_time"},
        "intensity": {"tau_s": 0.8, "levels": [1, 2, 3]},
        "release": {"mode": "dynamic_release"},
    }


def pose_sequence_dict():
    return {
        "time_s": [0.0, 0.5, 1.0],
        "pos": [[0.0, 0.0, 0.5], [0.1, 0.0, 0.4], [0.2, 0.0, 0.3]],
        "quat": [[1.0, 0.0, 0.0, 0.0]] * 3,
    }


def task_context_dict():
    return {
        "task_id": "task_a",
        "layout_id": "layout_0",
        "entities": {"target": {"name": "block_red", "source": "oracle_config"},
                     "container": {"name": "drawer", "source": "probe_inferred"}},
        "phases": [{"name": "approach", "t_start_s": 0.0, "t_end_s": 1.0},
                   {"name": "grasp", "t_start_s": 1.0, "t_end_s": 2.0}],
        "nominal_ee_path": pose_sequence_dict(),
        "occupancy": {"points": [[0.0, 0.0, 0.1], [0.1, 0.0, 0.1]], "time_s": [0.0, 0.5]},
        "workspace_bounds": {"min_m": [-0.5, -0.5, 0.0], "max_m": [0.5, 0.5, 0.8]},
        "support_surfaces": [{"name": "table", "min_m": [-0.4, -0.4, 0.0],
                              "max_m": [0.4, 0.4, 0.0], "source": "scene_yaml"}],
        "provenance": {"probe_sources": ["demo_replay", "scripted"], "n": 2},
    }


def certificate_dict():
    return {"g0": True, "g1": True, "g2": None, "label": "wait_solvable", "notes": "g2 not run"}


def hazard_instance_dict():
    return {
        "instance_id": "inst_0",
        "keyframes": pose_sequence_dict(),
        "nominal_hit_point": [0.2, 0.0, 0.3],
        "nominal_hit_time_s": 1.0,
        "onset_time_s": 0.2,
        "release_time_s": 0.5,
        "release_velocity": {"lin_mps": [0.0, 0.0, -1.0], "ang_radps": [0.0, 0.0, 0.0]},
        "certificate": certificate_dict(),
        "provenance": {"instance_seed": 7, "spec_id": "fall_basic"},
    }


def manifest_dict():
    return {
        "manifest_version": "1.0",
        "benchmark": "calvin",
        "bd_hash": _hex("a"),
        "probe_set_hash": _hex("b"),
        "spec_library_hash": _hex("c"),
        "adapter_version": "0.0-test",
        "benchmark_commits": {"calvin": "fa03f01", "calvin_env": "797142c"},
        "seed": 123,
        "entries": [
            {"instance_id": "inst_0", "spec_id": "fall_basic", "task_id": "task_a",
             "layout_id": "layout_0", "instance_seed": 7, "instance_hash": _hex("d")},
            {"instance_id": "inst_1", "spec_id": "fall_basic", "task_id": "task_a",
             "layout_id": "layout_1", "instance_seed": 8, "instance_hash": _hex("e")},
        ],
    }


def complete_bd_dict():
    """A BD with no `unknown` at all, shaped like the 01 §2 example (test data)."""
    return {
        "name": "toybench",
        "bd_version": "1.0",
        "benchmark": {"repo": "example/toy", "commits": {"toy": "abc1234"}, "license": "MIT"},
        "python": {"required": "3.8", "env_name": "toy_env"},
        "engine": {"name": "toyengine", "version": "1.2"},
        "timebase": {"physics_dt_s": 0.004, "control_dt_s": 0.032, "substeps_per_action": 8},
        "gravity_mps2": 9.81,
        "hazard_ccd": False,
        "parallel": {"num_envs": 1, "batched": False},
        "loop_owner": "user_script",
        "hook_point": "wrap env.step",
        "spawn_mode": "runtime",
        "kinematic_pose_set": True,
        "release_with_velocity": True,
        "contact_tier": "A",
        "distance": "geometric",
        "robot_contact_bodies": {"include": ["arm", "gripper", "held_object"], "exclude": ["camera"]},
        "eef_contact_radius_m": 0.02,
        "state_access": {"eef": True, "gripper": True, "base": False, "objects": True,
                        "fixtures": True, "support_surfaces": True},
        "task_entity_source": "oracle_config",
        "success_oracle": {"kind": "state_diff", "readonly": True},
        "snapshot_restore": True,
        "determinism": {"same_seed_repro": "exact"},
        "invalid_sample_mapping": {"setup_invalid": "layout_valid", "sim_unstable": "nan_check"},
        "observation_isolation": {"hazard_visible_in_rgb": True, "hazard_in_state_vectors": False},
        "policy_interface": {"kind": "toy_policy", "notes": ""},
    }


@pytest.fixture
def config_dir():
    return CONFIG_DIR


@pytest.fixture
def hazard_spec_d():
    return hazard_spec_dict()


@pytest.fixture
def task_context_d():
    return task_context_dict()


@pytest.fixture
def instance_d():
    return hazard_instance_dict()


@pytest.fixture
def manifest_d():
    return manifest_dict()


@pytest.fixture
def bd_d():
    return complete_bd_dict()


@pytest.fixture
def make_bd_dict():
    def _make(**overrides):
        d = complete_bd_dict()
        for dotted, value in overrides.items():
            parts = dotted.split("__")
            cur = d
            for part in parts[:-1]:
                cur = cur[part]
            cur[parts[-1]] = value
        return d
    return _make
