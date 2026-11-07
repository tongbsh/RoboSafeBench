import dataclasses

import pytest

import runtime_world as W
from hazardlib import DRIVER_METHODS, missing_methods, require_methods
from hazardlib.config import ConfigError, load_config
from hazardlib.conformance import FakeHost
from hazardlib.host import OPTIONAL_METHODS, REQUIRED_METHODS, CapabilityError
from hazardlib.runtime import RUNTIME_KEYS, SEMANTICS_KEYS, HazardRuntime, Provenance, RuntimeConfig
from hazardlib.spec import Manifest

RESOLVED_RUNTIME = {"max_step_ratio": 0.5, "release_velocity_window_s": 0.05,
                    "keyframe_dt_s": "PLACEHOLDER", "keyframe_interp_tol_m": "PLACEHOLDER"}
RESOLVED_SEMANTICS = {
    "contact": {"restitution_e": 0.0, "gap_tolerance_s": 0.02},
    "clamp": {"t_q_s": 0.1, "opposing_normals_tol_rad": 0.3},
    "toppling": {"tilt_threshold_deg": 30.0},
    "exposure": {"low_exposure_radius_m": "PLACEHOLDER"},
    "stability": {"hazard_induced_max_distance_m": 0.3, "attribution_window_s": 0.5},
}


def test_runtime_yaml_has_the_decided_ratio_and_placeholders_only(config_dir):
    data = load_config("runtime", config_dir=config_dir)
    assert data == {"max_step_ratio": 0.5, "release_velocity_window_s": "PLACEHOLDER",
                    "keyframe_dt_s": "PLACEHOLDER", "keyframe_interp_tol_m": "PLACEHOLDER"}


def test_repo_configs_are_rejected_while_placeholders_remain(config_dir):
    with pytest.raises(ConfigError) as err:
        RuntimeConfig.load(config_dir)
    message = str(err.value)
    assert "8 unresolved value(s)" in message
    assert "runtime.release_velocity_window_s=PLACEHOLDER" in message
    for section, key in SEMANTICS_KEYS:
        assert "semantics.%s.%s=PLACEHOLDER" % (section, key) in message
    assert "max_step_ratio" not in message


def test_resolved_configs_build_the_runtime_config():
    cfg = RuntimeConfig.from_configs(RESOLVED_RUNTIME, RESOLVED_SEMANTICS)
    assert cfg == W.runtime_config()
    assert set(RUNTIME_KEYS) == {"max_step_ratio", "release_velocity_window_s"}


def test_missing_and_invalid_values_are_config_errors():
    runtime = dict(RESOLVED_RUNTIME)
    del runtime["max_step_ratio"]
    semantics = dict(RESOLVED_SEMANTICS, clamp={"t_q_s": "unknown", "opposing_normals_tol_rad": 0.3})
    with pytest.raises(ConfigError) as err:
        RuntimeConfig.from_configs(runtime, semantics)
    assert "runtime.max_step_ratio is missing" in str(err.value)
    assert "semantics.clamp.t_q_s=unknown" in str(err.value)
    with pytest.raises(ConfigError, match="release_velocity_window_s"):
        RuntimeConfig.from_configs(dict(RESOLVED_RUNTIME, release_velocity_window_s=0.0), RESOLVED_SEMANTICS)


@pytest.mark.parametrize("field, value", [
    ("max_step_ratio", 0.0), ("release_velocity_window_s", -0.1), ("restitution_e", -0.5),
    ("t_q_s", float("nan")), ("opposing_normals_tol_rad", 4.0), ("attribution_window_s", float("inf")),
])
def test_runtime_config_validates_every_field(field, value):
    with pytest.raises(ValueError, match=field):
        W.runtime_config(**{field: value})


def test_runtime_config_has_no_defaults():
    assert all(f.default is dataclasses.MISSING for f in dataclasses.fields(RuntimeConfig))
    with pytest.raises(TypeError):
        RuntimeConfig(max_step_ratio=0.5)


def test_provenance_is_validated_and_copied_from_a_manifest(manifest_d):
    with pytest.raises(ValueError):
        Provenance(bd_hash="", adapter_version="x", manifest_hash=None, benchmark_commits=None)
    with pytest.raises(ValueError):
        Provenance(bd_hash=None, adapter_version=None, manifest_hash=None, benchmark_commits={"repo": ""})
    manifest = Manifest.from_dict(manifest_d)
    prov = Provenance.from_manifest(manifest)
    assert prov.log_fields() == {"bd_hash": manifest_d["bd_hash"], "adapter_version": "0.0-test",
                                 "manifest_hash": manifest.hash(),
                                 "benchmark_commits": manifest_d["benchmark_commits"]}


# ------------------------------------------------------------------------------- runtime construction

def test_fakehost_implements_the_whole_contract():
    host = FakeHost(W.world_config())
    require_methods(host)
    require_methods(host, DRIVER_METHODS)
    assert missing_methods(host, OPTIONAL_METHODS) == []
    degraded = FakeHost(W.world_config(), disable=("min_distance", "snapshot", "support_surfaces"))
    assert missing_methods(degraded, OPTIONAL_METHODS) == ["min_distance", "snapshot", "support_surfaces"]
    bd = degraded.describe()
    assert bd.distance == "surrogate" and bd.snapshot_restore is False
    assert bd.state_access.support_surfaces is False
    assert len(REQUIRED_METHODS) == 14


class _PatchedBD(FakeHost):
    """FakeHost whose BD differs in the given top-level fields."""

    def __init__(self, changes, **kwargs):
        super().__init__(W.world_config(), **kwargs)
        self._changes = changes

    def describe(self):
        return dataclasses.replace(super().describe(), **self._changes)


@pytest.mark.parametrize("changes, message", [
    ({"contact_tier": "B"}, "Tier B"),
    ({"release_with_velocity": False}, "Tier B"),
    ({"hazard_ccd": "unknown"}, "hazard_ccd"),
    ({"distance": "unknown"}, "distance"),
])
def test_runtime_refuses_hosts_it_cannot_serve(changes, message):
    with pytest.raises(CapabilityError, match=message):
        HazardRuntime(_PatchedBD(changes), W.runtime_config())


def test_geometric_distance_needs_min_distance():
    host = _PatchedBD({"distance": "geometric"}, disable=("min_distance",))
    with pytest.raises(CapabilityError, match="min_distance"):
        HazardRuntime(host, W.runtime_config())


def test_runtime_checks_its_arguments_and_installs_its_hooks():
    host = FakeHost(W.world_config())
    with pytest.raises(TypeError, match="RuntimeConfig"):
        HazardRuntime(host, {"max_step_ratio": 0.5})
    with pytest.raises(TypeError, match="Provenance"):
        HazardRuntime(host, W.runtime_config(), provenance={"bd_hash": "x"})
    with pytest.raises(CapabilityError, match="missing required method"):
        HazardRuntime(object(), W.runtime_config())
    runtime = HazardRuntime(host, W.runtime_config())
    assert host._runtime is runtime
