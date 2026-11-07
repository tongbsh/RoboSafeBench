"""configs/hazard/conformance.yaml after T05: every value is set and derived, except the statistical C8 bound."""
import math
import re

from hazardlib import load_config
from hazardlib.conformance import ConformanceConfig
from hazardlib.runtime import RUNTIME_KEYS, SEMANTICS_KEYS

# Keys that shape a probe (lengths, counts, the C13 path families) rather than bound a measurement.
PROBE_SHAPE_KEYS = {"duration_s", "control_steps", "snapshot_replay_steps", "hold_duration_s", "tau_s", "tail_s",
                    "families", "slide_speed_mps", "reach_distance_m", "reach_overshoot_m", "reach_speed_profile"}


def _leaves(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            for leaf in _leaves(value, path + (key,)):
                yield leaf
    else:
        yield path, obj


def test_the_only_placeholder_left_is_the_statistical_c8_bound(config_dir):
    leaves = dict(_leaves(load_config("conformance", config_dir=config_dir)))
    assert [p for p, v in leaves.items() if v == "PLACEHOLDER"] == [("c8_determinism", "statistical_max_abs_diff")]
    assert "unknown" not in leaves.values()


def test_every_tolerance_carries_its_derivation(config_dir):
    """Each tolerance line has an inline comment or sits under a comment block of its own section."""
    lines = (config_dir / "conformance.yaml").read_text(encoding="utf-8").splitlines()
    section, block_comment, missing = None, False, []
    for line in lines:
        if re.match(r"^[a-z0-9_]+:\s*$", line):
            section, block_comment = line[:-1], False
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            block_comment = block_comment or section is not None
            continue
        match = re.match(r"^  ([a-z0-9_]+):", line)
        if not match or section is None or not section.startswith("c"):
            continue
        key = match.group(1)
        if key not in PROBE_SHAPE_KEYS and "#" not in line and not block_comment:
            missing.append("%s.%s" % (section, key))
    assert not missing, missing


def test_probe_runtime_covers_exactly_the_unfrozen_runtime_fields(config_dir):
    probe_runtime = set(load_config("conformance", config_dir=config_dir)["probe_runtime"])
    fields = set(RUNTIME_KEYS) | {path[-1] for path in SEMANTICS_KEYS}
    assert probe_runtime <= fields
    runtime, sources = ConformanceConfig.load(config_dir).runtime_config()
    assert set(sources) == fields
    for field, used in sources.items():
        assert used["source"].startswith(("runtime.yaml:", "semantics.yaml:", "conformance.yaml:probe_runtime."))
        assert getattr(runtime, field) == used["value"]


def test_c13_tolerance_covers_one_fakehost_physics_step(config_dir):
    config = ConformanceConfig.load(config_dir)
    world = load_config("fakehost", strict=True, config_dir=config_dir)["world"]
    assert config.number("c13_nominal_hit", "hit_time_abs_tol_s") >= world["physics_dt_s"]


def test_coupled_values_follow_their_documented_formulas(config_dir):
    c = ConformanceConfig.load(config_dir)
    hit_tol = c.number("c13_nominal_hit", "hit_time_abs_tol_s")
    slide = c.number("c13_nominal_hit", "slide_speed_mps")
    assert c.number("c10_hold_t_stop", "drift_m_tol") == hit_tol * slide
    assert c.number("c10_hold_t_stop", "t_stop_iqr_s_tol") == hit_tol
    assert c.number("c13_nominal_hit", "replay_repeat_abs_tol_m") == 0.5 * hit_tol * slide
    assert abs(c.number("probe_calibration", "t_stop_speed_threshold_mps")
               - c.number("c10_hold_t_stop", "drift_m_tol") / c.number("c10_hold_t_stop", "hold_duration_s")) < 1e-12
    fakehost = load_config("fakehost", strict=True, config_dir=config_dir)
    radius = fakehost["scenario"]["probe_agent"]["half_extents_m"][0]
    drop = fakehost["scenario"]["drop_point_m"][2] - fakehost["scenario"]["drop_surface_height_m"] - radius
    t_land = math.sqrt(2 * drop / fakehost["world"]["gravity_mps2"])
    steps = c.number("c5_release_gravity", "free_fall_time_tol_physics_steps")
    landing_tol_s = steps * fakehost["world"]["physics_dt_s"]
    # t ~ g^-1/2: the g error that moves the FakeHost landing by the landing tolerance, rounded down to 3 decimals
    assert c.number("c5_release_gravity", "gravity_rel_tol") == math.floor(2 * landing_tol_s / t_land * 1000) / 1000
    assert abs(c.number("c3_pose_fidelity", "orientation_error_rad_tol")
               - c.number("c3_pose_fidelity", "pose_error_m_tol") / radius) < 1e-12
