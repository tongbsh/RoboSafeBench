"""D21 hazard scale axis of the LIBERO host: which cube hazard (half extent, mass) a run uses.

The levels live in ``configs/hazard/pilot/hazard_scale.yaml`` (S / M / L, pilot values). A level is selected by
the **assembled config directory** of a run (``pilot.assemble_config(directory, scale)``): for the default level
(S) the directory is exactly what it always was; for another level ``libero.yaml`` gets the level's half extents
and mass on the single pool member and on ``scenario.probe_agent`` (so the host's pool agent, hence every spec's
``agent``, and the conformance probes follow) and the level's strike radius (``scenario.strike.radius_m``), and ``hazard_scale.yaml`` is written into the directory as the
marker :func:`config_level` reads. Every consumer that takes ``config_dir`` (``LiberoHost``, ``RuntimeConfig``,
the generator, the evaluators) therefore picks the level up without a parameter of its own. The pool keeps one
member of the same name and the same joint, so the MuJoCo state vector (cached ``physical_states.npz``) has the
same layout at every level.

``prepare_root`` makes a generation root level-consistent: the root's ``config/`` and the ``agent`` of its
``scene.json`` (collect measured the default one); ``settings_for`` applies the per-level generator settings.
"""
import copy
import json
from pathlib import Path

import yaml

from hazardlib.config import default_config_dir, load_yaml

MARKER = "hazard_scale.yaml"
LEVELS_FILE = "pilot/hazard_scale.yaml"
LEVEL_KEYS = ("half_extent_m", "mass_kg", "coverage_radius_m", "strike_radius_m")


def load_levels(path=None):
    """``(default_level, {level: {half_extent_m, mass_kg, coverage_radius_m}})`` of the levels file."""
    path = Path(path) if path is not None else default_config_dir() / LEVELS_FILE
    data = load_yaml(path, strict=True)
    levels = data["levels"]
    default = data["default_level"]
    if default not in levels:
        raise ValueError("%s: default_level %r is not a level" % (path, default))
    for name, spec in levels.items():
        missing = [k for k in LEVEL_KEYS if k not in spec]
        extra = [k for k in spec if k not in LEVEL_KEYS]
        if missing or extra:
            raise ValueError("%s: level %s: missing %s, unknown %s" % (path, name, missing, extra))
        if not all(float(spec[k]) > 0 for k in LEVEL_KEYS):
            raise ValueError("%s: level %s: values must be positive" % (path, name))
    return default, {str(k): dict(v) for k, v in levels.items()}


def level_spec(level, path=None):
    default, levels = load_levels(path)
    level = default if level is None else level
    if level not in levels:
        raise ValueError("unknown hazard scale %r (levels: %s)" % (level, sorted(levels)))
    return level, levels[level]


def scaled_host_config(host_config, spec):
    """A copy of the parsed ``libero.yaml`` whose single pool member and probe agent have the level's cube and whose
    strike radius is the level's survey value."""
    cfg = copy.deepcopy(host_config)
    members = cfg["pool"]["members"]
    if len(members) != 1:
        raise ValueError("hazard scale needs exactly one pool member, got %d" % len(members))
    half = [float(spec["half_extent_m"])] * 3
    members[0]["half_extents_m"] = list(half)
    members[0]["mass_kg"] = float(spec["mass_kg"])
    probe = cfg["scenario"]["probe_agent"]
    probe["half_extents_m"] = list(half)
    probe["mass_kg"] = float(spec["mass_kg"])
    cfg["scenario"]["strike"]["radius_m"] = float(spec["strike_radius_m"])
    return cfg


def apply_to_config_dir(directory, level, levels_path=None):
    """Select ``level`` in an assembled config directory. The default level changes nothing (the legacy tree)."""
    directory = Path(directory)
    level, spec = level_spec(level, levels_path)
    default, _ = load_levels(levels_path)
    if level == default:
        if (directory / MARKER).exists():
            raise ValueError("%s is a %s config: assemble the default level into a new directory"
                             % (directory, config_level(directory)))
        return level
    cfg = scaled_host_config(load_yaml(directory / "libero.yaml"), spec)
    (directory / "libero.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (directory / MARKER).write_text(yaml.safe_dump({"level": level, **spec}, sort_keys=False))
    return level


def config_level(directory, levels_path=None):
    """The level an assembled config directory selects (the default level without the marker)."""
    marker = Path(directory) / MARKER
    if marker.exists():
        return str(yaml.safe_load(marker.read_text())["level"])
    return load_levels(levels_path)[0]


def settings_for(settings, level, levels_path=None):
    """The generator settings of a level: ``coverage_radius_m`` is the level's (the bounding radius of its box)."""
    _, spec = level_spec(level, levels_path)
    return dict(settings, coverage_radius_m=float(spec["coverage_radius_m"]))


def prepare_root(root, level, levels_path=None):
    """Make ``<root>/config`` and ``<root>/scene.json`` carry ``level`` (idempotent; an existing config of another
    level is an error). ``scene.json`` is replaced by a new file, never rewritten in place, so a hard-linked
    source tree (``trajectory_batch`` links collect / derive outputs) is not modified."""
    from .pilot import assemble_config
    root = Path(root)
    level, spec = level_spec(level, levels_path)
    config = root / "config"
    if not config.exists():
        assemble_config(config, level, levels_path)
    elif config_level(config, levels_path) != level:
        raise ValueError("%s is a %s config, not %s: use another output directory" % (config, config_level(config), level))
    scene_path = root / "scene.json"
    if scene_path.exists():
        scene = json.loads(scene_path.read_text())
        half = [float(spec["half_extent_m"])] * 3
        agent = dict(scene["agent"], half_extents_m=half, mass_kg=float(spec["mass_kg"]))
        if scene["agent"] != agent:
            scene = dict(scene, agent=agent, hazard_scale=level)
            temporary = scene_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n")
            temporary.replace(scene_path)
    return level
