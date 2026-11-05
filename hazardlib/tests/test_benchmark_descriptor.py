import copy

import pytest

from hazardlib import SpecError, Tier, load_bd
from hazardlib.spec import (BenchmarkDescriptor, TierUndeterminedError, TIER_A_REQUIREMENTS)

import conftest

# The five fields that together make Tier A (01 §2), as BD override keys.
A_FIELDS = {
    "contact_tier": "A",
    "kinematic_pose_set": True,
    "release_with_velocity": True,
    "state_access__eef": True,
    "state_access__objects": True,
}


def test_requirements_constant_matches_the_integration_rules():
    assert [d for d, _ in TIER_A_REQUIREMENTS] == [
        "contact_tier", "kinematic_pose_set", "release_with_velocity",
        "state_access.eef", "state_access.objects"]


# --------------------------------------------------------------- calvin draft

@pytest.fixture
def calvin(config_dir):
    return load_bd("calvin", config_dir=config_dir)


def test_calvin_draft_loads_non_strict_and_keeps_verified_facts(calvin):
    assert calvin.name == "calvin"
    assert calvin.benchmark.repo == "mees/calvin"
    assert calvin.benchmark.commits == {"calvin": "fa03f01", "calvin_env": "797142c"}
    assert calvin.python.required == "3.8" and calvin.python.env_name == "calvin_venv"
    assert calvin.engine.name == "pybullet" and calvin.engine.version == "unknown"
    # 02 §1: bullet_time_step=240, control_freq=30, action_repeat=int(240 // 30)
    assert calvin.timebase.physics_dt_s == pytest.approx(1.0 / 240.0, rel=1e-12)
    assert calvin.timebase.control_dt_s == pytest.approx(1.0 / 30.0, rel=1e-12)
    assert calvin.timebase.substeps_per_action == 8
    assert calvin.timebase.substeps_per_action * calvin.timebase.physics_dt_s == pytest.approx(
        calvin.timebase.control_dt_s, rel=1e-12)
    assert calvin.loop_owner == "user_script"
    assert calvin.task_entity_source == "oracle_config"


def test_calvin_draft_unknown_fields_are_exactly_the_probe_checklist(calvin):
    assert calvin.unknown_fields() == [
        "engine.version", "parallel.num_envs", "parallel.batched", "kinematic_pose_set",
        "release_with_velocity", "contact_tier", "distance", "robot_contact_bodies.include",
        "robot_contact_bodies.exclude", "state_access.base", "success_oracle.readonly",
        "determinism.same_seed_repro", "invalid_sample_mapping.setup_invalid",
        "invalid_sample_mapping.sim_unstable", "observation_isolation.hazard_visible_in_rgb",
        "observation_isolation.hazard_in_state_vectors", "policy_interface.kind"]


def test_calvin_draft_tier_is_undetermined_and_names_the_blocking_fields(calvin):
    """Derivation: eef/objects are verified true, but contact_tier,
    kinematic_pose_set and release_with_velocity are [待探针] in 02 §2, so
    neither A, B nor C can be asserted yet."""
    with pytest.raises(TierUndeterminedError) as err:
        calvin.tier()
    assert err.value.fields == ["contact_tier", "kinematic_pose_set", "release_with_velocity"]
    assert "contact_tier" in str(err.value)


def test_calvin_draft_becomes_tier_a_once_probes_confirm_the_01_example_values(calvin, config_dir):
    from hazardlib.config import load_yaml
    data = load_yaml(config_dir / "bd" / "calvin.yaml")
    data["contact_tier"] = "A"
    data["kinematic_pose_set"] = True
    data["release_with_velocity"] = True
    assert BenchmarkDescriptor.from_dict(data).tier() is Tier.A


def test_calvin_draft_strict_lists_every_unknown(config_dir):
    with pytest.raises(SpecError) as err:
        load_bd("calvin", strict=True, config_dir=config_dir)
    for path in ("kinematic_pose_set", "determinism.same_seed_repro", "engine.version"):
        assert path in str(err.value)


# ----------------------------------------------------------------- tier rules

def test_complete_bd_is_tier_a(bd_d):
    assert BenchmarkDescriptor.from_dict(bd_d).tier() is Tier.A


@pytest.mark.parametrize("override", [
    {"contact_tier": "B"},                 # boolean/partner-only contacts
    {"contact_tier": "C"},                 # no contact information at all
    {"release_with_velocity": False},      # cannot release to dynamics
    {"state_access__eef": False},
    {"state_access__objects": False},
    {"contact_tier": "B", "release_with_velocity": False},
])
def test_missing_one_capability_degrades_tier_a_to_b(make_bd_dict, override):
    assert BenchmarkDescriptor.from_dict(make_bd_dict(**override)).tier() is Tier.B


def test_each_tier_a_requirement_is_individually_necessary(make_bd_dict):
    failing = {"contact_tier": "B", "kinematic_pose_set": False, "release_with_velocity": False,
               "state_access__eef": False, "state_access__objects": False}
    for key, bad in failing.items():
        tier = BenchmarkDescriptor.from_dict(make_bd_dict(**{key: bad})).tier()
        assert tier is not Tier.A, key


def test_cannot_place_a_hazard_means_tier_c(make_bd_dict):
    assert BenchmarkDescriptor.from_dict(make_bd_dict(kinematic_pose_set=False)).tier() is Tier.C


def test_tier_c_is_decisive_even_if_other_fields_are_unknown(make_bd_dict):
    bd = BenchmarkDescriptor.from_dict(make_bd_dict(
        kinematic_pose_set=False, contact_tier="unknown", release_with_velocity="unknown"))
    assert bd.tier() is Tier.C


def test_known_failure_with_kinematic_true_is_b_even_if_other_fields_are_unknown(make_bd_dict):
    bd = BenchmarkDescriptor.from_dict(make_bd_dict(
        release_with_velocity=False, contact_tier="unknown"))
    assert bd.tier() is Tier.B


def test_unknown_is_never_a_capability(make_bd_dict):
    for key in A_FIELDS:
        bd = BenchmarkDescriptor.from_dict(make_bd_dict(**{key: "unknown"}))
        with pytest.raises(TierUndeterminedError) as err:
            bd.tier()
        assert err.value.fields == [key.replace("__", ".")]


def test_unknown_kinematic_flag_with_a_known_failure_is_undetermined_not_b(make_bd_dict):
    """Could still be Tier C if kinematic_pose_set turns out False."""
    bd = BenchmarkDescriptor.from_dict(make_bd_dict(
        kinematic_pose_set="unknown", release_with_velocity=False))
    with pytest.raises(TierUndeterminedError) as err:
        bd.tier()
    assert err.value.fields == ["kinematic_pose_set"]


def test_whole_section_unknown_is_handled(make_bd_dict):
    bd = BenchmarkDescriptor.from_dict(make_bd_dict(state_access="unknown"))
    with pytest.raises(TierUndeterminedError) as err:
        bd.tier()
    assert err.value.fields == ["state_access.eef", "state_access.objects"]
    assert bd.unknown_fields() == ["state_access"]


def test_tier_undetermined_is_a_capability_error(bd_d):
    from hazardlib import CapabilityError
    assert issubclass(TierUndeterminedError, CapabilityError)


# ------------------------------------------------------------ strict / schema

def test_strict_accepts_a_complete_bd_and_rejects_any_single_unknown(make_bd_dict):
    BenchmarkDescriptor.from_dict(make_bd_dict(), strict=True)
    with pytest.raises(SpecError, match="engine.version"):
        BenchmarkDescriptor.from_dict(make_bd_dict(engine__version="unknown"), strict=True)
    # the same dict loads without strict
    assert BenchmarkDescriptor.from_dict(make_bd_dict(engine__version="unknown")).engine.version == "unknown"


def test_strict_error_names_every_unknown_field(make_bd_dict):
    d = make_bd_dict(engine__version="unknown", parallel__num_envs="unknown", distance="unknown")
    with pytest.raises(SpecError) as err:
        BenchmarkDescriptor.from_dict(d, strict=True)
    message = str(err.value)
    assert "3 unknown" in message
    for path in ("engine.version", "parallel.num_envs", "distance"):
        assert path in message


def test_unknown_is_allowed_in_numeric_bool_and_enum_positions_when_not_strict(make_bd_dict):
    bd = BenchmarkDescriptor.from_dict(make_bd_dict(
        timebase__physics_dt_s="unknown", parallel__batched="unknown",
        loop_owner="unknown", contact_tier="unknown"))
    assert bd.timebase.physics_dt_s == "unknown" and bd.loop_owner == "unknown"
    assert bd.to_dict()["parallel"]["batched"] == "unknown"


def test_roundtrip_of_a_bd_with_unknowns_is_lossless(config_dir):
    from hazardlib.config import load_yaml
    data = load_yaml(config_dir / "bd" / "calvin.yaml")
    assert BenchmarkDescriptor.from_dict(data).to_dict() == data


@pytest.mark.parametrize("override,match", [
    ({"loop_owner": "nobody"}, "loop_owner"),
    ({"spawn_mode": "magic"}, "spawn_mode"),
    ({"contact_tier": "D"}, "contact_tier"),
    ({"distance": "psychic"}, "distance"),
    ({"task_entity_source": "vibes"}, "task_entity_source"),
    ({"determinism__same_seed_repro": "sometimes"}, "same_seed_repro"),
    ({"timebase__physics_dt_s": -0.1}, "physics_dt_s"),
    ({"timebase__control_dt_s": 0.0}, "control_dt_s"),
    ({"timebase__substeps_per_action": 0}, "substeps_per_action"),
    ({"timebase__substeps_per_action": 2.5}, "expected an int"),
    ({"parallel__num_envs": 0}, "num_envs"),
    ({"parallel__batched": "yes"}, "expected a bool"),
    ({"kinematic_pose_set": 1}, "expected a bool"),
    ({"python__required": 3.8}, "quote"),
    ({"engine__version": 3.3}, "expected a string"),
    ({"name": ""}, "name"),
    ({"benchmark__commits": ["a"]}, "expected a mapping"),
])
def test_schema_violations_are_reported(make_bd_dict, override, match):
    with pytest.raises(SpecError, match=match):
        BenchmarkDescriptor.from_dict(make_bd_dict(**override))


def test_every_top_level_and_nested_key_is_required(bd_d):
    for key in list(bd_d):
        d = copy.deepcopy(bd_d)
        d.pop(key)
        with pytest.raises(SpecError, match=key):
            BenchmarkDescriptor.from_dict(d)
    for section in ("benchmark", "python", "engine", "timebase", "parallel", "robot_contact_bodies",
                    "state_access", "success_oracle", "determinism", "invalid_sample_mapping",
                    "observation_isolation", "policy_interface"):
        for key in list(bd_d[section]):
            d = copy.deepcopy(bd_d)
            d[section].pop(key)
            with pytest.raises(SpecError, match=key):
                BenchmarkDescriptor.from_dict(d)


def test_unexpected_keys_are_rejected_at_every_level(bd_d):
    d = copy.deepcopy(bd_d)
    d["extra"] = 1
    with pytest.raises(SpecError, match="extra"):
        BenchmarkDescriptor.from_dict(d)
    d = copy.deepcopy(bd_d)
    d["state_access"]["telepathy"] = True
    with pytest.raises(SpecError, match="telepathy"):
        BenchmarkDescriptor.from_dict(d)
