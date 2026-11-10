"""``HazardInstance.exit_keyframes`` (decision D19): validation, serialisation and hash stability."""
import copy

import pytest

import conftest as C
from hazardlib._records import SpecError
from hazardlib.spec import HazardInstance

# Hash of ``HazardInstance.from_dict(conftest.hazard_instance_dict())`` computed with the sources of commit a57349c
# *before* the exit_keyframes field existed (the worktree was clean; a script printed ``inst.hash()``). It proves that an
# instance without an exit path serialises and hashes exactly as before the change.
HASH_BEFORE_EXIT_KEYFRAMES = "d3132d7968254a083af723d5f3818698d6ad291967e243beeda8d9393216354d"

ID = [1.0, 0.0, 0.0, 0.0]


def exit_instance_dict(family="intrusion.slide"):
    data = C.hazard_instance_dict()
    data["family"] = family
    data["keyframes"] = {"time_s": [0.2, 0.35, 0.5], "pos": [[0.0, 0.0, 0.3], [0.05, 0.0, 0.3], [0.1, 0.0, 0.3]],
                         "quat": [ID] * 3}
    data["release_time_s"] = 0.5
    data["release_velocity"] = {"lin_mps": [0.3, 0.0, 0.0], "ang_radps": [0.0, 0.0, 0.0]}
    data["exit_keyframes"] = {"time_s": [0.5, 0.6, 0.7], "pos": [[0.1, 0.0, 0.3], [0.13, 0.0, 0.3], [0.16, 0.0, 0.3]],
                              "quat": [ID] * 3}
    return data


def test_instance_without_exit_keyframes_serialises_and_hashes_as_before():
    inst = HazardInstance.from_dict(C.hazard_instance_dict())
    assert inst.exit_keyframes is None
    assert "exit_keyframes" not in inst.to_dict()
    assert inst.to_dict() == C.hazard_instance_dict()
    assert inst.hash() == HASH_BEFORE_EXIT_KEYFRAMES
    explicit = dict(C.hazard_instance_dict(), exit_keyframes=None)       # an explicit null is accepted and omitted
    assert HazardInstance.from_dict(explicit).hash() == HASH_BEFORE_EXIT_KEYFRAMES


@pytest.mark.parametrize("family", ["intrusion.slide", "intrusion.reach"])
def test_exit_keyframes_round_trip_and_change_the_hash(family):
    data = exit_instance_dict(family)
    inst = HazardInstance.from_dict(data)
    assert inst.exit_keyframes.time_s == [0.5, 0.6, 0.7]
    assert inst.to_dict() == data and HazardInstance.from_dict(inst.to_dict()).hash() == inst.hash()
    without = copy.deepcopy(data)
    del without["exit_keyframes"]
    assert HazardInstance.from_dict(without).hash() != inst.hash()


def _broken(edit):
    data = exit_instance_dict()
    edit(data)
    return data


@pytest.mark.parametrize("edit,message", [
    (lambda d: d.update(family="intrusion.fall"), "kinematic family"),
    (lambda d: d.update(release_time_s=None, release_velocity=None), "exit_keyframes need release_time_s"),
    (lambda d: d["exit_keyframes"].update(time_s=[0.55, 0.6, 0.7]), "time_s\\[0\\]"),
    (lambda d: d["keyframes"].update(time_s=[0.2, 0.35, 0.45]), "last keyframe time"),
    (lambda d: d["exit_keyframes"]["pos"].__setitem__(0, [0.1, 0.0, 0.3000000001]), "continuity"),
    (lambda d: d["exit_keyframes"]["quat"].__setitem__(0, [0.0, 1.0, 0.0, 0.0]), "continuity"),
    (lambda d: d["exit_keyframes"].update(time_s=[0.5], pos=[[0.1, 0.0, 0.3]], quat=[ID]), "at least 2"),
    (lambda d: d["exit_keyframes"].update(time_s=[0.5, 0.7, 0.6]), "strictly increasing"),
])
def test_exit_keyframes_validation(edit, message):
    with pytest.raises(SpecError, match=message):
        HazardInstance.from_dict(_broken(edit))


def test_continuity_is_exact_not_toleranced():
    data = exit_instance_dict()
    data["exit_keyframes"]["pos"][0][0] = 0.1 + 1e-18      # equals 0.1 in floating point: accepted
    HazardInstance.from_dict(data)
    data["exit_keyframes"]["pos"][0][0] = 0.1 + 1e-16      # a different float: rejected
    with pytest.raises(SpecError, match="continuity"):
        HazardInstance.from_dict(data)
