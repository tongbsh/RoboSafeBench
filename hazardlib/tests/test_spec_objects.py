import dataclasses

import numpy as np
import pytest
import yaml

from hazardlib import SpecError
from hazardlib.spec import (AgentSpec, Certificate, HazardInstance, HazardSpec, Manifest,
                            PoseSequence, TaskContext)

import conftest

CASES = [
    (HazardSpec, conftest.hazard_spec_dict),
    (PoseSequence, conftest.pose_sequence_dict),
    (TaskContext, conftest.task_context_dict),
    (Certificate, conftest.certificate_dict),
    (HazardInstance, conftest.hazard_instance_dict),
    (Manifest, conftest.manifest_dict),
]
IDS = [c.__name__ for c, _ in CASES]


@pytest.mark.parametrize("cls,factory", CASES, ids=IDS)
def test_dict_roundtrip_is_lossless(cls, factory):
    d = factory()
    obj = cls.from_dict(d)
    assert obj.to_dict() == d
    assert cls.from_dict(obj.to_dict()) == obj


@pytest.mark.parametrize("cls,factory", CASES, ids=IDS)
def test_yaml_roundtrip_is_lossless(cls, factory):
    obj = cls.from_dict(factory())
    text = yaml.safe_dump(obj.to_dict())
    assert cls.from_dict(yaml.safe_load(text)) == obj


@pytest.mark.parametrize("cls,factory", CASES, ids=IDS)
def test_unexpected_key_is_rejected_with_its_name(cls, factory):
    d = factory()
    d["surprise"] = 1
    with pytest.raises(SpecError, match="surprise"):
        cls.from_dict(d)


@pytest.mark.parametrize("cls,factory", CASES, ids=IDS)
def test_every_required_key_is_enforced_and_named(cls, factory):
    required = [f.name for f in dataclasses.fields(cls)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    assert required, "record has no required fields?"
    for key in required:
        d = factory()
        d.pop(key)
        with pytest.raises(SpecError, match=key):
            cls.from_dict(d)


@pytest.mark.parametrize("cls,factory", CASES, ids=IDS)
def test_non_mapping_is_rejected(cls, factory):
    with pytest.raises(SpecError, match="expected a mapping"):
        cls.from_dict([1, 2])


def test_error_messages_carry_a_dotted_path():
    d = conftest.hazard_spec_dict()
    d["agent"]["half_extents_m"][1] = -0.1
    with pytest.raises(SpecError, match=r"HazardSpec\.agent: half_extents_m\[1\]"):
        HazardSpec.from_dict(d)


def test_numpy_inputs_are_normalised_to_plain_python():
    d = conftest.pose_sequence_dict()
    d["time_s"] = np.array(d["time_s"])
    d["pos"] = np.array(d["pos"])
    d["quat"] = np.array(d["quat"])
    out = PoseSequence.from_dict(d).to_dict()
    assert out == conftest.pose_sequence_dict()
    assert type(out["time_s"][0]) is float and type(out["pos"][0][0]) is float


def test_non_finite_numbers_are_rejected_everywhere():
    d = conftest.hazard_spec_dict()
    d["agent"]["mass_kg"] = float("nan")
    with pytest.raises(SpecError, match="non-finite"):
        HazardSpec.from_dict(d)
    d = conftest.hazard_spec_dict()
    d["intensity"]["tau_s"] = float("inf")
    with pytest.raises(SpecError, match="non-finite"):
        HazardSpec.from_dict(d)


def test_bool_is_not_accepted_as_a_number_or_int():
    d = conftest.hazard_spec_dict()
    d["agent"]["mass_kg"] = True
    with pytest.raises(SpecError, match="expected a number"):
        HazardSpec.from_dict(d)
    m = conftest.manifest_dict()
    m["seed"] = True
    with pytest.raises(SpecError, match="expected an int"):
        Manifest.from_dict(m)


def test_int_accepted_for_float_fields_and_float_rejected_for_int_fields():
    d = conftest.hazard_spec_dict()
    d["agent"]["mass_kg"] = 2
    assert HazardSpec.from_dict(d).agent.mass_kg == 2.0
    m = conftest.manifest_dict()
    m["seed"] = 1.5
    with pytest.raises(SpecError, match="expected an int"):
        Manifest.from_dict(m)


# ---------------------------------------------------------------- HazardSpec

@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["timing"].update(mode="whenever"), "timing.mode"),
    (lambda d: d["timing"].update(mode=""), "timing.mode"),
    (lambda d: d["agent"].update(mass_kg=0.0), "mass_kg"),
    (lambda d: d["agent"].update(half_extents_m=[]), "half_extents_m"),
    (lambda d: d["agent"].update(half_extents_m=[0.1, 0.0]), r"half_extents_m\[1\]"),
    (lambda d: d["exposure"]["anchor_phase"].update(frac=1.5), "frac"),
    (lambda d: d["exposure"]["anchor_phase"].update(frac=-0.1), "frac"),
    (lambda d: d["exposure"].update(offset_m=[0.0, 0.0]), "offset_m"),
    (lambda d: d.update(id=""), "id"),
    (lambda d: d.update(family=" "), "family"),
    (lambda d: d["motion"].update(primitive=""), "primitive"),
    (lambda d: d["intensity"].update(bad=object()), "intensity.bad"),
])
def test_hazard_spec_validation(mutate, match):
    d = conftest.hazard_spec_dict()
    mutate(d)
    with pytest.raises(SpecError, match=match):
        HazardSpec.from_dict(d)


def test_both_timing_modes_are_accepted():
    for mode in ("nominal_time", "phase_event"):
        d = conftest.hazard_spec_dict()
        d["timing"]["mode"] = mode
        assert HazardSpec.from_dict(d).timing.mode == mode


def test_anchor_phase_is_optional():
    d = conftest.hazard_spec_dict()
    d["exposure"].pop("anchor_phase")
    assert HazardSpec.from_dict(d).exposure.anchor_phase is None


def test_motion_params_default_to_empty():
    d = conftest.hazard_spec_dict()
    d["motion"].pop("params")
    assert HazardSpec.from_dict(d).motion.params == {}


def test_agent_spec_standalone():
    agent = AgentSpec(kind="object", shape="box", half_extents_m=[0.1, 0.1, 0.1], mass_kg=1.0)
    agent.validate()
    agent.mass_kg = -1.0
    with pytest.raises(SpecError, match="mass_kg"):
        agent.validate()


# --------------------------------------------------------------- PoseSequence

@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(time_s=[]), "at least one"),
    (lambda d: d.update(time_s=[0.0, 0.5]), "equal length"),
    (lambda d: d.update(time_s=[0.0, 0.5, 0.5]), "strictly increasing"),
    (lambda d: d.update(time_s=[0.0, 0.5, 0.4]), "strictly increasing"),
    (lambda d: d["pos"].__setitem__(1, [0.0, 0.0]), r"pos\[1\]"),
    (lambda d: d["quat"].__setitem__(2, [1.0, 0.0, 0.0]), r"quat\[2\]"),
    (lambda d: d["quat"].__setitem__(0, [0.0, 0.0, 0.0, 0.0]), "non-zero"),
])
def test_pose_sequence_validation(mutate, match):
    d = conftest.pose_sequence_dict()
    if match == "at least one":
        d = {"time_s": [], "pos": [], "quat": []}
    else:
        mutate(d)
    with pytest.raises(SpecError, match=match):
        PoseSequence.from_dict(d)


def test_pose_sequence_len():
    assert len(PoseSequence.from_dict(conftest.pose_sequence_dict())) == 3


# ---------------------------------------------------------------- TaskContext

@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["entities"]["target"].update(source="guess"), "source"),
    (lambda d: d["entities"]["target"].update(name=""), "name"),
    (lambda d: d["phases"].append({"name": "approach", "t_start_s": 2.0, "t_end_s": 3.0}), "unique"),
    (lambda d: d["phases"].reverse(), "sorted"),
    (lambda d: d["phases"][0].update(t_end_s=-1.0), "before t_start_s"),
    (lambda d: d["occupancy"].update(time_s=[0.0]), "equal length"),
    (lambda d: d["occupancy"]["points"].__setitem__(0, [0.0]), r"points\[0\]"),
    (lambda d: d["workspace_bounds"].update(min_m=[1.0, 0.0, 0.0]), "exceeds"),
    (lambda d: d["support_surfaces"][0].update(source=""), "source"),
    (lambda d: d["support_surfaces"][0].update(max_m=[0.0, 0.0, -1.0]), "exceeds"),
    (lambda d: d.update(task_id=""), "task_id"),
    (lambda d: d.update(layout_id=0), "expected a string"),
])
def test_task_context_validation(mutate, match):
    d = conftest.task_context_dict()
    mutate(d)
    with pytest.raises(SpecError, match=match):
        TaskContext.from_dict(d)


def test_task_context_accepts_every_entity_source():
    from hazardlib.spec import ENTITY_SOURCES
    d = conftest.task_context_dict()
    for i, source in enumerate(ENTITY_SOURCES):
        d["entities"]["role%d" % i] = {"name": "n%d" % i, "source": source}
    assert len(TaskContext.from_dict(d).entities) == 2 + len(ENTITY_SOURCES)


def test_task_context_allows_empty_occupancy_and_surfaces():
    d = conftest.task_context_dict()
    d["occupancy"] = {"points": [], "time_s": []}
    d["support_surfaces"] = []
    tc = TaskContext.from_dict(d)
    assert tc.occupancy.points == [] and tc.support_surfaces == []


# ---------------------------------------------------------------- Certificate

def test_certificate_labels_and_tristate_gates():
    from hazardlib.spec import CERT_LABELS
    assert CERT_LABELS == ("wait_solvable", "evade_solvable", "unsolvable_by_probes", "uncertified")
    ok = [
        {"g0": True, "g1": True, "g2": None, "label": "wait_solvable"},
        {"g0": True, "g1": False, "g2": True, "label": "evade_solvable"},
        {"g0": True, "g1": False, "g2": False, "label": "unsolvable_by_probes"},
        {"g0": True, "g1": False, "g2": None, "label": "unsolvable_by_probes"},
        {"g0": True, "g1": True, "g2": None, "label": "uncertified"},
        {"g0": None, "g1": None, "g2": None, "label": "uncertified"},
    ]
    for item in ok:
        item.setdefault("notes", "")
        assert Certificate.from_dict(item).label == item["label"]


@pytest.mark.parametrize("bad", [
    {"g0": True, "g1": False, "g2": None, "label": "wait_solvable"},
    {"g0": False, "g1": True, "g2": None, "label": "wait_solvable"},
    {"g0": True, "g1": False, "g2": False, "label": "evade_solvable"},
    {"g0": True, "g1": True, "g2": None, "label": "unsolvable_by_probes"},
    {"g0": True, "g1": False, "g2": True, "label": "unsolvable_by_probes"},
    {"g0": True, "g1": True, "g2": True, "label": "mystery"},
])
def test_certificate_rejects_inconsistent_or_unknown_labels(bad):
    bad = dict(bad, notes="")
    with pytest.raises(SpecError):
        Certificate.from_dict(bad)


# ------------------------------------------------------------- HazardInstance

@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(onset_time_s=1.5), "after nominal_hit_time_s"),
    (lambda d: d.update(release_time_s=None), "both set or both None"),
    (lambda d: d.update(release_velocity=None), "both set or both None"),
    (lambda d: d.update(nominal_hit_point=[0.0, 0.0]), "nominal_hit_point"),
    (lambda d: d["release_velocity"].update(lin_mps=[0.0]), "lin_mps"),
    (lambda d: d.update(instance_id=""), "instance_id"),
    (lambda d: d["keyframes"].update(time_s=[1.0, 0.5, 0.0]), "strictly increasing"),
    (lambda d: d["certificate"].update(label="nope"), "certificate.label"),
])
def test_instance_validation(mutate, match):
    d = conftest.hazard_instance_dict()
    mutate(d)
    with pytest.raises(SpecError, match=match):
        HazardInstance.from_dict(d)


def test_instance_without_release_or_certificate_is_valid():
    d = conftest.hazard_instance_dict()
    d["release_time_s"] = None
    d["release_velocity"] = None
    d["certificate"] = None
    inst = HazardInstance.from_dict(d)
    assert inst.release_velocity is None and inst.certificate is None
    assert inst.to_dict()["certificate"] is None


def test_onset_equal_to_hit_is_allowed():
    d = conftest.hazard_instance_dict()
    d["onset_time_s"] = d["nominal_hit_time_s"]
    HazardInstance.from_dict(d)


# ------------------------------------------------------------------- Manifest

@pytest.mark.parametrize("field", ["bd_hash", "probe_set_hash", "spec_library_hash"])
@pytest.mark.parametrize("bad", ["", "abc", "A" * 64, "g" * 64, "a" * 63])
def test_manifest_hash_fields_must_be_sha256_hex(field, bad):
    d = conftest.manifest_dict()
    d[field] = bad
    with pytest.raises(SpecError, match=field):
        Manifest.from_dict(d)


def test_manifest_entry_hash_and_duplicate_ids():
    d = conftest.manifest_dict()
    d["entries"][0]["instance_hash"] = "xyz"
    with pytest.raises(SpecError, match="instance_hash"):
        Manifest.from_dict(d)
    d = conftest.manifest_dict()
    d["entries"][1]["instance_id"] = d["entries"][0]["instance_id"]
    with pytest.raises(SpecError, match="duplicate instance_id"):
        Manifest.from_dict(d)


def test_manifest_hash_tracks_every_component():
    base = Manifest.from_dict(conftest.manifest_dict())
    h0 = base.hash()
    for field, value in (("bd_hash", "f" * 64), ("adapter_version", "other"), ("seed", 124),
                         ("benchmark_commits", {"calvin": "0000000"})):
        d = conftest.manifest_dict()
        d[field] = value
        assert Manifest.from_dict(d).hash() != h0, field
    d = conftest.manifest_dict()
    d["entries"].reverse()
    assert Manifest.from_dict(d).hash() != h0  # entry order is part of the manifest
