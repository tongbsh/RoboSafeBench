"""The calibration block of configs/hazard/intensity.yaml (T02-F, decision D3): structure only."""
import pytest
import yaml

from hazardlib import ConfigError, load_config, load_yaml
from hazardlib._records import find_markers

CALIBRATION_KEYS = {"t_stop_speed_threshold_mps", "t_stop_min_consecutive_samples", "t_stop_repeats",
                    "sigma_bins", "staircase"}
STAIRCASE_KEYS = {"step_s", "tau_min_s", "tau_max_s"}
NEW_PATHS = ["calibration.t_stop_min_consecutive_samples", "calibration.staircase.step_s",
             "calibration.staircase.tau_min_s", "calibration.staircase.tau_max_s"]


@pytest.fixture
def calibration(config_dir):
    return load_config("intensity", config_dir=config_dir)["calibration"]


def test_calibration_block_has_exactly_the_registered_keys(calibration):
    assert set(calibration) == CALIBRATION_KEYS
    assert set(calibration["staircase"]) == STAIRCASE_KEYS
    # the key that existed since T01 is still there, exactly once (duplicate keys are a load error)
    assert calibration["t_stop_speed_threshold_mps"] == "PLACEHOLDER"
    assert calibration["t_stop_repeats"] == "PLACEHOLDER"


def test_every_new_calibration_value_is_a_placeholder_not_a_number(calibration):
    assert calibration["t_stop_min_consecutive_samples"] == "PLACEHOLDER"
    for key in STAIRCASE_KEYS:
        assert calibration["staircase"][key] == "PLACEHOLDER"


def test_the_new_keys_are_reported_as_unresolved(config_dir):
    data = load_config("intensity", config_dir=config_dir)
    unresolved = {path for path, marker in find_markers(data) if marker == "PLACEHOLDER"}
    for path in NEW_PATHS:
        assert path in unresolved


def test_strict_loading_the_whole_file_fails_and_non_strict_loads(config_dir):
    with pytest.raises(ConfigError, match="unresolved value"):
        load_config("intensity", strict=True, config_dir=config_dir)
    assert "calibration" in load_config("intensity", strict=False, config_dir=config_dir)


def test_strict_error_names_each_calibration_key(config_dir, tmp_path):
    # The full file has dozens of unresolved leaves and the message lists only the first few, so check the
    # calibration block on its own: strict=True must name every one of its keys.
    block = load_config("intensity", config_dir=config_dir)["calibration"]
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump({"calibration": block}), encoding="utf-8")
    with pytest.raises(ConfigError) as err:
        load_yaml(path, strict=True)
    message = str(err.value)
    assert "7 unresolved" in message   # 4 scalar keys + 3 staircase keys
    for dotted in ["calibration.t_stop_speed_threshold_mps", "calibration.t_stop_repeats",
                   "calibration.sigma_bins"] + NEW_PATHS:
        assert dotted + "=PLACEHOLDER" in message
    assert load_yaml(path, strict=False)["calibration"]["staircase"]["step_s"] == "PLACEHOLDER"


def test_visible_lead_is_not_an_intensity_axis(config_dir):
    # D5: the visible lead tau_vis is a diagnostic measured per episode, not an intensity input.
    axes = load_config("intensity", config_dir=config_dir)["intensity"]
    assert set(axes["intrusion.slide"]) == {"tau_s", "approach_speed_mps"}
    for family, family_axes in axes.items():
        assert not any("precursor" in key or "lead" in key for key in family_axes), family


def test_a_resolved_calibration_block_passes_strict(tmp_path):
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump({"calibration": {
        "t_stop_speed_threshold_mps": 1, "t_stop_min_consecutive_samples": 2, "t_stop_repeats": 3,
        "sigma_bins": [1, 2], "staircase": {"step_s": 4, "tau_min_s": 5, "tau_max_s": 6}}}),
        encoding="utf-8")
    assert load_yaml(path, strict=True)["calibration"]["staircase"]["tau_max_s"] == 6   # arbitrary test data
