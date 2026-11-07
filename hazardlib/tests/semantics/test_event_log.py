import json

import pytest

from hazardlib.host import PartnerClass
from hazardlib.semantics import (SCHEMA_VERSION, ConstraintState, HazardEventLog, InvalidClass, SeparationKind,
                                 append_jsonl, read_jsonl, write_jsonl)

SECTION6_FIELDS = ["instance_id", "onset_t_s", "released_t_s", "first_official_contact_t_s",
                   "first_scene_contact_t_s",                                                    # D10 a
                   "partner_class", "v_rel_normal_mps", "hazard_mass_kg", "impulse_analytic_ns",
                   "engine_peak_force_n", "force_source", "contact_duration_s", "constraint_state",
                   "min_separation_m", "separation_kind", "ttc_min_s", "exposure_low", "toppled",
                   "kinematic_overlap_count", "invalid_class"]
OUTCOME_FIELDS = ["task_success", "max_tilt_deg", "r1_violation_count"]                  # D11 c
PROVENANCE_FIELDS = ["bd_hash", "adapter_version", "manifest_hash", "benchmark_commits"]  # D11 c


def full_log(instance_id="inst_0"):
    return HazardEventLog.new(
        instance_id, onset_t_s=0.1 + 0.2, released_t_s=1.0 / 3.0, first_official_contact_t_s=1.25,
        first_scene_contact_t_s=1.0 + 1.0 / 7.0, partner_class="robot_link", v_rel_normal_mps=0.7, hazard_mass_kg=0.2, impulse_analytic_ns=0.14,
        engine_peak_force_n=3.5, force_source="pybullet_contact_normal_force", contact_duration_s=0.8,
        constraint_state=ConstraintState.CLAMP, min_separation_m=0.0, separation_kind="surrogate",
        ttc_min_s=0.0, exposure_low=False, toppled=True, kinematic_overlap_count=0,
        invalid_class=InvalidClass.HAZARD_INDUCED_UNSTABLE, attribution_ambiguous=True,
        task_success=False, max_tilt_deg=1.0 / 3.0, r1_violation_count=2, bd_hash="a" * 64,
        adapter_version="0.0-test", manifest_hash="b" * 64,
        benchmark_commits={"calvin": "fa03f01", "calvin_env": "797142c"})


def test_fields_are_the_section_6_set_plus_schema_version_ambiguity_outcome_and_provenance():
    names = list(full_log().to_dict())
    assert names == (["schema_version"] + SECTION6_FIELDS + ["attribution_ambiguous"] + OUTCOME_FIELDS
                     + PROVENANCE_FIELDS)


def test_round_trip_through_a_jsonl_file_is_lossless(tmp_path):
    path = tmp_path / "hazard_events.jsonl"
    logs = [full_log("a"), full_log("b")]
    write_jsonl(path, logs)
    back = read_jsonl(path)
    assert [b.to_dict() for b in back] == [l.to_dict() for l in logs]
    assert back[0].onset_t_s == 0.1 + 0.2 and back[0].released_t_s == 1.0 / 3.0       # exact floats, not rounded
    assert back[0].first_scene_contact_t_s == 1.0 + 1.0 / 7.0
    assert back[0].partner_class is PartnerClass.ROBOT_LINK
    assert back[0].constraint_state is ConstraintState.CLAMP
    assert back[0].separation_kind is SeparationKind.SURROGATE
    assert back[0].invalid_class is InvalidClass.HAZARD_INDUCED_UNSTABLE
    assert back[0].kinematic_overlap_count == 0 and isinstance(back[0].kinematic_overlap_count, int)
    assert back[0].exposure_low is False and back[0].toppled is True and back[0].attribution_ambiguous is True
    assert back[0].task_success is False and back[0].max_tilt_deg == 1.0 / 3.0
    assert back[0].r1_violation_count == 2 and isinstance(back[0].r1_violation_count, int)
    assert back[0].bd_hash == "a" * 64 and back[0].manifest_hash == "b" * 64
    assert back[0].adapter_version == "0.0-test"
    assert back[0].benchmark_commits == {"calvin": "fa03f01", "calvin_env": "797142c"}


def test_benchmark_commits_is_copied_in_and_out():
    commits = {"calvin": "fa03f01"}
    log = HazardEventLog.new("i", benchmark_commits=commits)
    commits["calvin"] = "0000000"
    assert log.benchmark_commits == {"calvin": "fa03f01"}
    out = log.to_dict()
    out["benchmark_commits"]["calvin"] = "1111111"
    assert log.benchmark_commits == {"calvin": "fa03f01"}


def test_new_outcome_and_provenance_fields_are_none_not_zero_when_absent(tmp_path):
    log = HazardEventLog.new("dev_run", task_success=True)
    for name in ["max_tilt_deg", "r1_violation_count"] + PROVENANCE_FIELDS:
        assert getattr(log, name) is None, name
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [log, HazardEventLog.new("zeros", max_tilt_deg=0.0, r1_violation_count=0,
                                                task_success=False)])
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    assert raw[0]["r1_violation_count"] is None and raw[0]["max_tilt_deg"] is None
    assert raw[0]["benchmark_commits"] is None and raw[0]["task_success"] is True
    assert raw[1]["r1_violation_count"] == 0 and raw[1]["max_tilt_deg"] == 0.0 and raw[1]["task_success"] is False


def test_the_file_has_one_json_object_per_line_with_enum_values_as_strings(tmp_path):
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [full_log("a"), full_log("b")])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["partner_class"] == "robot_link" and first["constraint_state"] == "clamp"
    assert first["separation_kind"] == "surrogate" and first["invalid_class"] == "hazard_induced_unstable"
    assert first["schema_version"] == SCHEMA_VERSION


def test_missing_values_are_none_and_never_zero(tmp_path):
    log = HazardEventLog.new("only_id")
    assert all(v is None for k, v in log.to_dict().items() if k not in ("schema_version", "instance_id"))
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [log])
    assert json.loads(path.read_text())["kinematic_overlap_count"] is None                 # null, not 0
    back = read_jsonl(path)[0]
    assert back.kinematic_overlap_count is None and back.impulse_analytic_ns is None
    assert back.exposure_low is None and back.toppled is None and back.partner_class is None


def test_keys_absent_from_a_line_read_back_as_none(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "instance_id": "i", "onset_t_s": 0.5}) + "\n")
    log = read_jsonl(path)[0]
    assert log.onset_t_s == 0.5
    for name in SECTION6_FIELDS[2:] + ["attribution_ambiguous"] + OUTCOME_FIELDS + PROVENANCE_FIELDS:
        assert getattr(log, name) is None, name


def test_zero_is_kept_distinct_from_none():
    log = HazardEventLog.new("i", impulse_analytic_ns=0.0, kinematic_overlap_count=0, min_separation_m=0.0,
                             contact_duration_s=0.0, exposure_low=False)
    d = log.to_dict()
    assert d["impulse_analytic_ns"] == 0.0 and d["kinematic_overlap_count"] == 0 and d["exposure_low"] is False
    assert d["ttc_min_s"] is None


def test_append_adds_one_episode_at_a_time(tmp_path):
    path = tmp_path / "x.jsonl"
    append_jsonl(path, full_log("a"))
    append_jsonl(path, HazardEventLog.new("b"))
    assert [l.instance_id for l in read_jsonl(path)] == ["a", "b"]


def test_blank_lines_are_skipped_and_errors_name_the_line(tmp_path):
    path = tmp_path / "x.jsonl"
    good = full_log("a").to_json_line()
    path.write_text(good + "\n\n" + good + "\n")
    assert len(read_jsonl(path)) == 2
    path.write_text(good + "\n" + json.dumps({"schema_version": "1", "instance_id": "b", "bogus": 1}) + "\n")
    with pytest.raises(ValueError, match=r"line 2: unexpected key\(s\) bogus"):
        read_jsonl(path)
    path.write_text(good + "\nnot json\n")
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(path)


def test_reader_rejects_unsupported_versions_missing_ids_and_non_objects(tmp_path):
    path = tmp_path / "x.jsonl"
    for payload, match in (({"schema_version": "999", "instance_id": "i"}, "unsupported schema_version"),
                           ({"instance_id": "i"}, "missing required key schema_version"),
                           ({"schema_version": SCHEMA_VERSION}, "missing required key instance_id")):
        path.write_text(json.dumps(payload) + "\n")
        with pytest.raises(ValueError, match=match):
            read_jsonl(path)
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="JSON object"):
        read_jsonl(path)


def test_first_scene_contact_is_optional_and_independent_of_the_official_contact(tmp_path):
    log = HazardEventLog.new("fall", first_scene_contact_t_s=0.9)            # landed, never touched the robot
    assert log.first_scene_contact_t_s == 0.9 and log.first_official_contact_t_s is None
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [log, HazardEventLog.new("none")])
    first, second = read_jsonl(path)
    assert first.first_scene_contact_t_s == 0.9 and second.first_scene_contact_t_s is None
    assert json.loads(path.read_text().splitlines()[1])["first_scene_contact_t_s"] is None
    assert SCHEMA_VERSION == "1"


@pytest.mark.parametrize("field,value", [
    ("onset_t_s", float("nan")), ("released_t_s", float("inf")), ("onset_t_s", "0.5"), ("onset_t_s", True),
    ("first_scene_contact_t_s", float("nan")), ("first_scene_contact_t_s", float("-inf")),
    ("first_scene_contact_t_s", "1.0"), ("first_scene_contact_t_s", False),
    ("v_rel_normal_mps", -0.1), ("impulse_analytic_ns", -1.0), ("engine_peak_force_n", -1.0),
    ("contact_duration_s", -0.01), ("min_separation_m", -0.01), ("ttc_min_s", -0.01),
    ("hazard_mass_kg", 0.0), ("hazard_mass_kg", -0.2),
    ("exposure_low", 1), ("toppled", "yes"), ("attribution_ambiguous", 0),
    ("partner_class", "ghost"), ("constraint_state", "squeezed"), ("separation_kind", "approximate"),
    ("invalid_class", "broken"),
    ("force_source", ""), ("force_source", 5),
    ("kinematic_overlap_count", -1), ("kinematic_overlap_count", 1.5), ("kinematic_overlap_count", True),
    ("task_success", 1), ("task_success", "true"),
    ("max_tilt_deg", -0.5), ("max_tilt_deg", float("nan")), ("max_tilt_deg", float("inf")), ("max_tilt_deg", "3"),
    ("r1_violation_count", -1), ("r1_violation_count", 2.0), ("r1_violation_count", False),
    ("bd_hash", ""), ("bd_hash", 7), ("adapter_version", ""), ("adapter_version", 1.0),
    ("manifest_hash", ""), ("manifest_hash", ["x"]),
    ("benchmark_commits", "fa03f01"), ("benchmark_commits", ["calvin"]), ("benchmark_commits", {"calvin": ""}),
    ("benchmark_commits", {"": "fa03f01"}), ("benchmark_commits", {"calvin": 1234567}),
])
def test_field_validation(field, value):
    with pytest.raises(ValueError, match=field):
        HazardEventLog.new("i", **{field: value})


def test_identity_fields_are_validated():
    for bad in ("", None, 3):
        with pytest.raises(ValueError, match="instance_id"):
            HazardEventLog(schema_version=SCHEMA_VERSION, instance_id=bad)
        with pytest.raises(ValueError, match="schema_version"):
            HazardEventLog(schema_version=bad, instance_id="i")


def test_a_non_finite_value_set_after_construction_fails_at_write_time_and_keeps_the_old_file(tmp_path):
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [full_log("keep")])
    before = path.read_text()
    broken = full_log("broken")
    broken.onset_t_s = float("nan")                       # bypasses __post_init__ on purpose
    with pytest.raises(ValueError, match="broken"):
        write_jsonl(path, [broken])
    assert path.read_text() == before                     # nothing was truncated
    with pytest.raises(ValueError):
        append_jsonl(path, broken)
    assert path.read_text() == before


def test_enum_fields_accept_members_or_values_and_integers_are_accepted_for_floats():
    a = HazardEventLog.new("i", partner_class=PartnerClass.SCENE, onset_t_s=2)
    b = HazardEventLog.new("i", partner_class="scene", onset_t_s=2.0)
    assert a.to_dict() == b.to_dict() and isinstance(a.onset_t_s, float)
