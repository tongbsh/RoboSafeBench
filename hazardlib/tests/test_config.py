from pathlib import Path

import pytest
import yaml

import hazardlib
from hazardlib import (ConfigError, SpecError, default_config_dir, load_bd, load_config,
                       load_yaml)
from hazardlib.config import CONFIG_DIR_ENV

SKELETON_CONFIGS = ["families", "intensity", "gates", "semantics", "conformance"]


def test_default_config_dir_finds_the_repo_configs(monkeypatch, config_dir):
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    assert default_config_dir() == config_dir


def test_env_var_overrides_and_is_validated(monkeypatch, tmp_path):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    assert default_config_dir() == tmp_path
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "missing"))
    with pytest.raises(ConfigError, match="not a directory"):
        default_config_dir()


def test_directory_layout_matches_the_integration_rules(config_dir):
    assert (config_dir / "bd" / "calvin.yaml").is_file()
    assert (config_dir / "annotations" / ".gitkeep").is_file()
    for name in SKELETON_CONFIGS:
        assert (config_dir / (name + ".yaml")).is_file(), name


@pytest.mark.parametrize("name", SKELETON_CONFIGS)
def test_skeleton_loads_non_strict_but_not_strict(name, config_dir):
    data = load_config(name, config_dir=config_dir)
    assert isinstance(data, dict) and data
    with pytest.raises(ConfigError, match="PLACEHOLDER") as err:
        load_config(name, strict=True, config_dir=config_dir)
    assert "unresolved value(s) with strict=True" in str(err.value)


def _leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            for leaf in _leaves(v):
                yield leaf
    elif isinstance(obj, list):
        for v in obj:
            for leaf in _leaves(v):
                yield leaf
    else:
        yield obj


@pytest.mark.parametrize("name", SKELETON_CONFIGS)
def test_skeleton_contains_no_invented_values(name, config_dir):
    """T01 must not invent numbers: every leaf is the PLACEHOLDER marker."""
    leaves = list(_leaves(load_config(name, config_dir=config_dir)))
    assert leaves
    assert set(leaves) == {"PLACEHOLDER"}


def test_load_config_missing_file_is_a_config_error(config_dir):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config("does_not_exist", config_dir=config_dir)


def _write(tmp_path, text, name="x.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_strict_flags_placeholder_and_unknown_at_any_depth(tmp_path):
    path = _write(tmp_path, "a: 1\nb:\n  - {c: PLACEHOLDER}\n  - d: [unknown, 2]\ne: {f: {g: PLACEHOLDER}}\n")
    assert load_yaml(path)["a"] == 1
    with pytest.raises(ConfigError) as err:
        load_yaml(path, strict=True)
    message = str(err.value)
    assert "3 unresolved" in message
    for expected in ("b[0].c=PLACEHOLDER", "b[1].d[0]=unknown", "e.f.g=PLACEHOLDER"):
        assert expected in message


def test_strict_passes_a_resolved_file(tmp_path):
    path = _write(tmp_path, "a: 1\nb: [x, y]\nnote: 'PLACEHOLDERS are fine inside longer text'\n")
    assert load_yaml(path, strict=True)["a"] == 1


def test_strict_message_is_truncated_but_counts_everything(tmp_path):
    path = _write(tmp_path, "".join("k%d: PLACEHOLDER\n" % i for i in range(45)))
    with pytest.raises(ConfigError) as err:
        load_yaml(path, strict=True)
    assert "45 unresolved" in str(err.value) and "and 25 more" in str(err.value)


def test_duplicate_keys_are_rejected(tmp_path):
    path = _write(tmp_path, "a: 1\nb: 2\na: 3\n")
    with pytest.raises(ConfigError, match="duplicate mapping key 'a'"):
        load_yaml(path)
    nested = _write(tmp_path, "outer:\n  x: 1\n  x: 2\n", "n.yaml")
    with pytest.raises(ConfigError, match="duplicate mapping key 'x'"):
        load_yaml(nested)


def test_invalid_yaml_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml(_write(tmp_path, "a: [1, 2\n"))


def test_config_error_is_a_spec_error():
    assert issubclass(ConfigError, SpecError)


def test_load_bd_by_name_and_by_path_agree(config_dir):
    by_name = load_bd("calvin", config_dir=config_dir)
    by_path = load_bd(str(config_dir / "bd" / "calvin.yaml"))
    assert by_name == by_path and by_name.hash() == by_path.hash()


def test_load_bd_strict_error_names_the_file(config_dir):
    with pytest.raises(ConfigError) as err:
        load_bd("calvin", strict=True, config_dir=config_dir)
    assert "calvin.yaml" in str(err.value) and "unknown" in str(err.value)


def test_load_bd_schema_error_names_the_file(tmp_path):
    path = _write(tmp_path, "name: broken\n", "broken.yaml")
    with pytest.raises(ConfigError, match="broken.yaml"):
        load_bd(path)


def test_yaml_pitfalls_surface_as_readable_type_errors(tmp_path, config_dir):
    data = yaml.safe_load((config_dir / "bd" / "calvin.yaml").read_text())
    text = (config_dir / "bd" / "calvin.yaml").read_text().replace('required: "3.8"', "required: 3.8")
    path = _write(tmp_path, text, "unquoted.yaml")
    with pytest.raises(ConfigError, match="quote"):
        load_bd(path)
    assert data["python"]["required"] == "3.8"


def test_hazardlib_exports_the_public_api():
    for name in ("load_bd", "load_config", "load_yaml", "default_config_dir", "HazardHost", "Tier",
                 "CapabilityError", "SpecError", "ConfigError", "spec", "hashing"):
        assert hasattr(hazardlib, name), name
