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


# --------------------------------------------------------------- calvin (T01 draft, completed by H1)

# The T01 draft's `unknown` fields (02 §2 [待探针]); H1 resolved each with a probe or a conformance check.
CALVIN_PROBED_FIELDS = [
    "engine.version", "gravity_mps2", "hazard_ccd", "parallel.num_envs", "parallel.batched",
    "kinematic_pose_set", "release_with_velocity", "contact_tier", "distance",
    "robot_contact_bodies.include", "robot_contact_bodies.exclude", "eef_contact_radius_m",
    "state_access.base", "success_oracle.readonly",
    "determinism.same_seed_repro", "invalid_sample_mapping.setup_invalid",
    "invalid_sample_mapping.sim_unstable", "observation_isolation.hazard_visible_in_rgb",
    "observation_isolation.hazard_in_state_vectors", "policy_interface.kind"]


@pytest.fixture
def calvin(config_dir):
    return load_bd("calvin", strict=True, config_dir=config_dir)


def _calvin_with_unknown(config_dir, paths):
    from hazardlib.config import load_yaml
    data = load_yaml(config_dir / "bd" / "calvin.yaml")
    for path in paths:
        node = data
        *parents, leaf = path.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = "unknown"
    return data


def test_calvin_loads_strict_and_keeps_verified_facts(calvin):
    assert calvin.name == "calvin"
    assert calvin.benchmark.repo == "mees/calvin"
    assert calvin.benchmark.commits == {"calvin": "fa03f01", "calvin_env": "797142c"}
    assert calvin.python.required == "3.8" and calvin.python.env_name == "calvin_venv"
    assert calvin.engine.name == "pybullet" and calvin.engine.version == "3.2.7"
    # 02 §1: bullet_time_step=240, control_freq=30, action_repeat=int(240 // 30)
    assert calvin.timebase.physics_dt_s == pytest.approx(1.0 / 240.0, rel=1e-12)
    assert calvin.timebase.control_dt_s == pytest.approx(1.0 / 30.0, rel=1e-12)
    assert calvin.timebase.substeps_per_action == 8
    assert calvin.timebase.substeps_per_action * calvin.timebase.physics_dt_s == pytest.approx(
        calvin.timebase.control_dt_s, rel=1e-12)
    assert calvin.loop_owner == "user_script"
    assert calvin.task_entity_source == "oracle_config"


def test_calvin_has_no_unknown_field_left(calvin):
    assert calvin.unknown_fields() == []


def test_calvin_is_tier_a(calvin):
    assert calvin.tier() is Tier.A


def test_calvin_tier_rests_on_the_probed_fields(config_dir):
    """Derivation: eef/objects are verified true (02 §2); contact_tier, kinematic_pose_set and
    release_with_velocity were [待探针] in the draft. Without them the tier is undetermined again."""
    probed = ["contact_tier", "kinematic_pose_set", "release_with_velocity"]
    with pytest.raises(TierUndeterminedError) as err:
        BenchmarkDescriptor.from_dict(_calvin_with_unknown(config_dir, probed)).tier()
    assert err.value.fields == probed
    assert "contact_tier" in str(err.value)


def test_calvin_strict_would_list_every_probed_field(config_dir):
    with pytest.raises(SpecError) as err:
        BenchmarkDescriptor.from_dict(_calvin_with_unknown(config_dir, CALVIN_PROBED_FIELDS), strict=True)
    assert "%d unknown" % len(CALVIN_PROBED_FIELDS) in str(err.value)
    for path in CALVIN_PROBED_FIELDS:
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


# ----------------------------------------------- D1 / D2 / D11 d top-level fields

NEW_FIELDS = ("gravity_mps2", "eef_contact_radius_m", "hazard_ccd")


@pytest.mark.parametrize("key", NEW_FIELDS)
def test_new_top_level_fields_are_required(bd_d, key):
    d = copy.deepcopy(bd_d)
    d.pop(key)
    with pytest.raises(SpecError, match="missing required key.*%s" % key):
        BenchmarkDescriptor.from_dict(d)


@pytest.mark.parametrize("key", NEW_FIELDS)
def test_new_fields_may_be_unknown_and_strict_lists_them(make_bd_dict, key):
    bd = BenchmarkDescriptor.from_dict(make_bd_dict(**{key: "unknown"}))
    assert bd.unknown_fields() == [key]
    assert bd.tier() is Tier.A  # not tier-relevant
    with pytest.raises(SpecError, match=key):
        BenchmarkDescriptor.from_dict(make_bd_dict(**{key: "unknown"}), strict=True)


def test_all_three_unknown_are_named_by_strict(make_bd_dict):
    d = make_bd_dict(gravity_mps2="unknown", eef_contact_radius_m="unknown", hazard_ccd="unknown")
    with pytest.raises(SpecError) as err:
        BenchmarkDescriptor.from_dict(d, strict=True)
    assert "3 unknown" in str(err.value)
    for key in NEW_FIELDS:
        assert key in str(err.value)


@pytest.mark.parametrize("key,value", [
    ("gravity_mps2", 9.8), ("eef_contact_radius_m", 0.035), ("hazard_ccd", True)])
def test_bd_hash_follows_the_new_field_values(make_bd_dict, key, value):
    base = BenchmarkDescriptor.from_dict(make_bd_dict())
    changed = BenchmarkDescriptor.from_dict(make_bd_dict(**{key: value}))
    assert getattr(changed, key) == value
    assert changed.hash() != base.hash()


@pytest.mark.parametrize("override,match", [
    ({"gravity_mps2": 0.0}, "gravity_mps2"),
    ({"gravity_mps2": -9.8}, "gravity_mps2"),
    ({"gravity_mps2": "9.81"}, "expected a number"),
    ({"eef_contact_radius_m": -0.01}, "eef_contact_radius_m"),
    ({"hazard_ccd": 1}, "expected a bool"),
    ({"hazard_ccd": "yes"}, "expected a bool"),
])
def test_new_field_values_are_checked(make_bd_dict, override, match):
    with pytest.raises(SpecError, match=match):
        BenchmarkDescriptor.from_dict(make_bd_dict(**override))


def test_eef_contact_radius_may_be_zero(make_bd_dict):
    assert BenchmarkDescriptor.from_dict(make_bd_dict(eef_contact_radius_m=0.0)).eef_contact_radius_m == 0.0


# --------------------------------------------- robot_contact_bodies match kinds (D10 f)

def _with_bodies(bd_d, include=None, exclude=None):
    d = copy.deepcopy(bd_d)
    if include is not None:
        d["robot_contact_bodies"]["include"] = include
    if exclude is not None:
        d["robot_contact_bodies"]["exclude"] = exclude
    return d


def test_every_match_kind_and_the_held_object_token_load(bd_d):
    bd = BenchmarkDescriptor.from_dict(_with_bodies(
        bd_d, include=["exact:panda_hand", "prefix:panda_link", "substr:finger", "held_object"],
        exclude=["exact:wrist_cam", "prefix:cam_", "substr:sensor"]))
    assert bd.robot_contact_bodies.include[-1] == "held_object"
    assert BenchmarkDescriptor.from_dict(_with_bodies(bd_d, exclude=[])).robot_contact_bodies.exclude == []


@pytest.mark.parametrize("field,entries,bad_index", [
    ("include", ["arm"], 0),
    ("include", ["prefix:arm", "gripper"], 1),
    ("include", ["regex:arm.*"], 0),
    ("include", ["prefix:"], 0),
    ("include", ["Prefix:arm"], 0),
    ("include", [":arm"], 0),
    ("include", ["prefix:arm", "unknown"], 1),
    ("exclude", ["camera"], 0),
    ("exclude", ["substr:camera", "held_object"], 1),  # the category token is include-only
])
def test_an_entry_without_a_valid_match_kind_is_rejected_and_named(bd_d, field, entries, bad_index):
    with pytest.raises(SpecError) as err:
        BenchmarkDescriptor.from_dict(_with_bodies(bd_d, **{field: entries}))
    message = str(err.value)
    assert "robot_contact_bodies" in message
    assert "%s[%d] = %r" % (field, bad_index, entries[bad_index]) in message


@pytest.mark.parametrize("include", [[], ["held_object"]])
def test_include_needs_a_pattern_entry(bd_d, include):
    with pytest.raises(SpecError, match="include needs at least one"):
        BenchmarkDescriptor.from_dict(_with_bodies(bd_d, include=include))


def test_unknown_contact_body_lists_keep_their_strict_behaviour(bd_d):
    d = _with_bodies(bd_d, include="unknown", exclude="unknown")
    bd = BenchmarkDescriptor.from_dict(d)
    assert bd.unknown_fields() == ["robot_contact_bodies.include", "robot_contact_bodies.exclude"]
    with pytest.raises(SpecError, match="2 unknown"):
        BenchmarkDescriptor.from_dict(d, strict=True)
    complete = BenchmarkDescriptor.from_dict(bd_d, strict=True)
    assert complete.robot_contact_bodies.include == ["prefix:arm", "prefix:gripper", "held_object"]


@pytest.mark.parametrize("change", [
    {"include": ["exact:arm", "prefix:gripper", "held_object"]},
    {"include": ["prefix:arm", "prefix:gripper"]},
    {"exclude": ["prefix:camera"]},
    {"exclude": []},
])
def test_bd_hash_follows_the_contact_body_entries(bd_d, change):
    base = BenchmarkDescriptor.from_dict(bd_d)
    changed = BenchmarkDescriptor.from_dict(_with_bodies(bd_d, **change))
    assert changed.hash() != base.hash()


def test_unexpected_keys_are_rejected_at_every_level(bd_d):
    d = copy.deepcopy(bd_d)
    d["extra"] = 1
    with pytest.raises(SpecError, match="extra"):
        BenchmarkDescriptor.from_dict(d)
    d = copy.deepcopy(bd_d)
    d["state_access"]["telepathy"] = True
    with pytest.raises(SpecError, match="telepathy"):
        BenchmarkDescriptor.from_dict(d)
