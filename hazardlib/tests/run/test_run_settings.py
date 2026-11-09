"""``configs/hazard/run.yaml`` and :class:`RunSettings` (R1 item 3: strict loading, no defaults, hashed)."""
import copy

import pytest
import yaml

import run_world as RW
from hazardlib.config import ConfigError
from hazardlib.hashing import sha256_hex
from hazardlib.run.settings import RunSettings

KEYS = ("episode.horizon_extra_s", "episode.stop_on_success", "report.tau_bin_edges_s", "report.sigma_bin_edges",
        "report.interval.method", "report.interval.confidence")


def _leaves(data, prefix=""):
    for key, value in data.items():
        path = "%s.%s" % (prefix, key) if prefix else key
        if isinstance(value, dict):
            yield from _leaves(value, path)
        else:
            yield path, value


def test_repository_run_yaml_is_all_placeholder_and_refused():
    data = yaml.safe_load((RW.W.REPO_CONFIG_DIR / "run.yaml").read_text(encoding="utf-8"))
    leaves = dict(_leaves(data))
    assert sorted(leaves) == sorted(KEYS)
    assert set(leaves.values()) == {"PLACEHOLDER"}
    with pytest.raises(ConfigError) as err:
        RunSettings.load(RW.W.REPO_CONFIG_DIR)
    assert "6 unresolved value(s) with strict=True" in str(err.value)
    for key in KEYS:
        assert key in str(err.value)


def test_resolved_run_yaml_loads_and_is_hashed(tmp_path):
    RW.write_run_yaml(tmp_path)
    settings = RunSettings.load(tmp_path)
    assert settings.episode.horizon_extra_s == 1.5 and settings.episode.stop_on_success is False
    assert settings.report.tau_bin_edges_s == [0.2, 0.4] and settings.report.sigma_bin_edges == [1.0, 2.0]
    assert settings.report.interval_method == "clopper_pearson" and settings.report.confidence == 0.95
    assert settings.config_hash == sha256_hex(RW.RUN_YAML)
    changed = copy.deepcopy(RW.RUN_YAML)
    changed["report"]["interval"]["confidence"] = 0.9
    RW.write_run_yaml(tmp_path, changed)
    assert RunSettings.load(tmp_path).config_hash != settings.config_hash


@pytest.mark.parametrize("path, value, message", [
    (("episode", "horizon_extra_s"), -0.1, "horizon_extra_s"),
    (("episode", "stop_on_success"), "yes", "stop_on_success must be true or false"),
    (("report", "tau_bin_edges_s"), [0.4, 0.2], "strictly increasing"),
    (("report", "tau_bin_edges_s"), [], "non-empty list"),
    (("report", "sigma_bin_edges"), "1.0", "non-empty list"),
    (("report", "interval", "method"), "normal", "is not one of"),
    (("report", "interval", "confidence"), 1.0, "must lie in (0, 1)"),
])
def test_invalid_values_are_refused(tmp_path, path, value, message):
    data = copy.deepcopy(RW.RUN_YAML)
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    RW.write_run_yaml(tmp_path, data)
    with pytest.raises(ConfigError) as err:
        RunSettings.load(tmp_path)
    assert message in str(err.value)


def test_a_missing_key_is_named(tmp_path):
    data = copy.deepcopy(RW.RUN_YAML)
    del data["report"]["interval"]["method"]
    RW.write_run_yaml(tmp_path, data)
    with pytest.raises(ConfigError, match=r"run\.report\.interval\.method is missing"):
        RunSettings.load(tmp_path)
