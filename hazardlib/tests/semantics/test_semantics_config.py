import pytest

from hazardlib import ConfigError, load_config
from hazardlib._records import find_markers

EXPECTED = {
    "contact": {"restitution_e", "gap_tolerance_s"},
    "clamp": {"t_q_s", "opposing_normals_tol_rad"},
    "toppling": {"tilt_threshold_deg"},
    "exposure": {"low_exposure_radius_m"},
    "stability": {"hazard_induced_max_distance_m", "attribution_window_s"},
}


@pytest.fixture
def semantics(config_dir):
    return load_config("semantics", config_dir=config_dir)


def test_the_skeleton_has_exactly_the_registered_keys(semantics):
    assert {section: set(body) for section, body in semantics.items()} == EXPECTED


def test_d9_key_changes_are_applied(semantics):
    assert "hazard_induced_max_distance_m" in semantics["stability"]
    assert "hazard_induced_unstable_speed_mps" not in semantics["stability"]      # replaced by the distance key
    assert "gap_tolerance_s" in semantics["contact"]
    assert "tilt_threshold_deg" in semantics["toppling"] and "tilt_threshold_rad" not in semantics["toppling"]


def test_d10_key_changes_are_applied(semantics):
    assert "impulse_gate_ns" not in semantics["contact"]                     # D10 j: no impulse gate
    assert "attribution_window_s" in semantics["stability"]                  # D10 b


def test_every_leaf_is_still_a_placeholder(semantics):
    leaves = [value for body in semantics.values() for value in body.values()]
    assert leaves and set(leaves) == {"PLACEHOLDER"}
    flagged = {path for path, marker in find_markers(semantics) if marker == "PLACEHOLDER"}
    assert flagged == {"%s.%s" % (s, k) for s, keys in EXPECTED.items() for k in keys}


def test_strict_loading_fails_naming_the_keys_and_non_strict_loads(config_dir):
    with pytest.raises(ConfigError) as err:
        load_config("semantics", strict=True, config_dir=config_dir)
    message = str(err.value)
    assert "unresolved value(s) with strict=True" in message
    assert "contact.gap_tolerance_s=PLACEHOLDER" in message
    assert "stability.hazard_induced_max_distance_m=PLACEHOLDER" in message
    assert "stability.attribution_window_s=PLACEHOLDER" in message
    assert load_config("semantics", strict=False, config_dir=config_dir)["contact"]["gap_tolerance_s"] == "PLACEHOLDER"
