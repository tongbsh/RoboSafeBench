import numpy as np
import pytest

from hazardlib.host import ContactObs, PartnerClass
from hazardlib.semantics import ContactEvent, EventKind, classify_partner

INCLUDE = ["robot0_", "gripper0_", "mobilebase0_", "panda0_"]     # RoboCasa-style prefixes (02 §3)
EXCLUDE = ["camera"]


def classify(name, include=INCLUDE, exclude=EXCLUDE, held="mug_main", hazards=("hazard_0",)):
    return classify_partner(name, include, exclude, held, list(hazards))


@pytest.mark.parametrize("name,expected", [
    ("robot0_link3", PartnerClass.ROBOT_LINK),
    ("gripper0_right_finger", PartnerClass.ROBOT_LINK),
    ("mobilebase0_wheel_left", PartnerClass.ROBOT_LINK),          # the mobile base and wheels are robot side
    ("panda0_hand", PartnerClass.ROBOT_LINK),
    ("robot0_eye_camera_mount", PartnerClass.SCENE),             # excluded: cameras / sensors are not robot side
    ("mug_main", PartnerClass.HELD_OBJECT),                      # the grasped object
    ("cup_on_table", PartnerClass.SCENE),                        # an ungrasped task object is scene
    ("counter_top", PartnerClass.SCENE),
    ("hazard_0", PartnerClass.HAZARD),
])
def test_partner_classification(name, expected):
    assert classify(name) is expected


def test_rule_precedence():
    # hazard beats everything, held object beats exclude, exclude beats include
    assert classify("robot0_ghost", hazards=("robot0_ghost",)) is PartnerClass.HAZARD
    assert classify("mug_camera", held="mug_camera") is PartnerClass.HELD_OBJECT
    assert classify("robot0_camera") is PartnerClass.SCENE
    assert classify("mug_main", held=None) is PartnerClass.SCENE       # no grasp: just a scene object


def test_matching_is_by_case_sensitive_substring():
    assert classify("left_gripper0_pad") is PartnerClass.ROBOT_LINK    # not only prefixes
    assert classify("ROBOT0_link") is PartnerClass.SCENE


def test_held_object_token_is_a_category_not_a_pattern():
    include = ["arm", "gripper", "held_object"]                      # the 01 §2 example
    assert classify_partner("arm_link2", include, ["camera"], None, []) is PartnerClass.ROBOT_LINK
    assert classify_partner("held_object_thing", include, ["camera"], None, []) is PartnerClass.SCENE
    assert classify_partner("pen", include, ["camera"], "pen", []) is PartnerClass.HELD_OBJECT
    with pytest.raises(ValueError, match="no body-name pattern"):
        classify_partner("arm_link2", ["held_object"], [], None, [])


@pytest.mark.parametrize("kwargs,match", [
    (dict(partner_name=""), "partner_name"),
    (dict(partner_name=3), "partner_name"),
    (dict(robot_include=[]), "no body-name pattern"),
    (dict(robot_include=["robot0_", ""]), "invalid pattern"),
    (dict(robot_include="robot0_"), "not a single string"),
    (dict(robot_exclude=[None]), "invalid pattern"),
    (dict(hazard_names=["ok", 5]), "invalid pattern"),
])
def test_classification_rejects_misconfiguration(kwargs, match):
    args = dict(partner_name="x", robot_include=INCLUDE, robot_exclude=EXCLUDE, held_object_name=None,
                hazard_names=[])
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        classify_partner(**args)


# --------------------------------------------------------------------- ContactEvent

def event(**overrides):
    kwargs = dict(kind="official", time_s=0.5, partner_name="finger", partner_class="robot_link",
                  position=[0, 0, 0.1])
    kwargs.update(overrides)
    return ContactEvent(**kwargs)


def test_contact_event_coerces_and_defaults_to_none():
    ev = event()
    assert ev.kind is EventKind.OFFICIAL and ev.partner_class is PartnerClass.ROBOT_LINK
    assert ev.normal is None and ev.v_normal_mps is None and ev.impulse_ns is None
    assert ev.position.shape == (3,) and ev.position.dtype == np.float64
    ev = event(normal=[0, 0, 1], v_normal_mps=0.4, impulse_ns=0.0)
    assert ev.v_normal_mps == 0.4 and ev.impulse_ns == 0.0       # 0 is a real value, kept distinct from None


def test_contact_event_from_a_host_observation():
    obs = ContactObs(partner_name="table", partner_class="scene", position=[0, 0, 0],
                     normal=[0, 0, 1], time_s=0.7, impulse_ns=0.05)
    ev = ContactEvent.from_obs(obs, EventKind.OFFICIAL, 0.3)
    assert (ev.partner_name, ev.partner_class, ev.time_s, ev.impulse_ns, ev.v_normal_mps) == (
        "table", PartnerClass.SCENE, 0.7, 0.05, 0.3)
    np.testing.assert_array_equal(ev.normal, [0, 0, 1])
    assert ContactEvent.from_obs(obs, EventKind.OFFICIAL, None).v_normal_mps is None


def test_official_events_may_be_with_scene_but_overlaps_are_robot_side_only():
    assert event(partner_class="scene").partner_class is PartnerClass.SCENE      # R8: scene contacts of a fall
    for ok in ("robot_link", "held_object"):
        assert event(kind="kinematic_overlap", partner_class=ok).kind is EventKind.KINEMATIC_OVERLAP
    for bad in ("scene", "hazard"):
        with pytest.raises(ValueError, match="kinematic_overlap must be with the robot side"):
            event(kind="kinematic_overlap", partner_class=bad)


@pytest.mark.parametrize("overrides,match", [
    (dict(kind="maybe"), "kind"),
    (dict(partner_class="ghost"), "partner_class"),
    (dict(time_s=float("nan")), "time_s"),
    (dict(partner_name=""), "partner_name"),
    (dict(position=[0, 0]), "position"),
    (dict(normal=[0, 1]), "normal"),
    (dict(v_normal_mps=-0.1), "v_normal_mps"),
    (dict(impulse_ns=-1.0), "impulse_ns"),
])
def test_contact_event_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        event(**overrides)
