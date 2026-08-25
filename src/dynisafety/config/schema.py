"""Config dataclasses — defaults live only in YAML (iron law 1)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import MISSING as DATACLASS_MISSING
from dataclasses import Field, dataclass, field, fields
from typing import Any


class _MissingType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING: Any = _MissingType()


# ---------------------------------------------------------------------------
# Nested config types
# ---------------------------------------------------------------------------


@dataclass
class ContactConfig:
    topple_threshold_deg: float = field(default=MISSING)
    force_equiv_dt: str = field(default=MISSING)
    force_report_window_s: float = field(default=MISSING)
    impulse_reduce: str = field(default=MISSING)


@dataclass
class ReactivityConfig:
    window_size: int = field(default=MISSING)
    sensitivity: float = field(default=MISSING)
    freeze_threshold: float = field(default=MISSING)
    freeze_min_steps: int = field(default=MISSING)


@dataclass
class MetricsConfig:
    injury_thresholds_n: dict[str, float] = field(default=MISSING)
    sasr_penalty: dict[str, Any] = field(default=MISSING)
    reactivity: ReactivityConfig = field(default=MISSING)
    # SimplerEnv (and shared) contact anomaly gates → EpisodeRecord.extra["force_anomaly"].
    force_anomaly_threshold_n: float = field(default=MISSING)
    impulse_anomaly_threshold_ns: float = field(default=MISSING)


@dataclass
class DiagnosticsTraceConfig:
    save: bool = field(default=MISSING)
    format: str = field(default=MISSING)  # "jsonl_gz" | "jsonl"
    stride: int = field(default=MISSING)


@dataclass
class DiagnosticsReactionConfig:
    window_size: int = field(default=MISSING)
    sensitivity: float = field(default=MISSING)
    min_persist_steps: int = field(default=MISSING)
    # Reaction = deviation of the velocity vector (magnitude OR direction).
    heading_change_deg: float = field(default=MISSING)
    sigma_rel_floor: float = field(default=MISSING)


@dataclass
class DiagnosticsStopConfig:
    speed_frac: float = field(default=MISSING)
    speed_floor_m_per_step: float = field(default=MISSING)
    min_steps: int = field(default=MISSING)


@dataclass
class DiagnosticsAvoidConfig:
    heading_change_deg: float = field(default=MISSING)
    window_steps: int = field(default=MISSING)
    # Heading sampling gate, independent of stop.speed_frac (decoupled axes).
    min_speed_frac: float = field(default=MISSING)


@dataclass
class DiagnosticsRecoveryConfig:
    speed_frac: float = field(default=MISSING)
    min_steps: int = field(default=MISSING)
    min_post_hazard_steps: int = field(default=MISSING)


@dataclass
class DiagnosticsHazardEndConfig:
    settle_quiet_steps: int = field(default=MISSING)
    settle_quiet_m: float = field(default=MISSING)


@dataclass
class DiagnosticsConfig:
    """Behavior diagnostics (react / stop / avoid / recover).

    See docs/user/behavior_diagnostics.md. Disabled → EpisodeRecord.diagnostics is
    null and tables render n/a (never 0.0).
    """

    enabled: bool = field(default=MISSING)
    trace: DiagnosticsTraceConfig = field(default=MISSING)
    reaction: DiagnosticsReactionConfig = field(default=MISSING)
    stop: DiagnosticsStopConfig = field(default=MISSING)
    avoid: DiagnosticsAvoidConfig = field(default=MISSING)
    recovery: DiagnosticsRecoveryConfig = field(default=MISSING)
    hazard_end: DiagnosticsHazardEndConfig = field(default=MISSING)


@dataclass
class OutputConfig:
    results_dir: str = field(default=MISSING)
    save_video: bool = field(default=MISSING)


@dataclass
class RuntimeConfig:
    seed: int = field(default=MISSING)
    seed_stride: int = field(default=MISSING)
    episodes: int = field(default=MISSING)
    max_steps: int = field(default=MISSING)
    # Deprecated: no backend reads this; rollouts always stop on success. Kept
    # so existing YAML/--set carrying the key does not fail as an unknown path.
    stop_on_success: bool = field(default=MISSING)
    control_freq: int = field(default=MISSING)
    sim_freq: int = field(default=MISSING)
    obj_episode_range: list[int] = field(default=MISSING)


@dataclass
class DisturbanceL1Config:
    trigger_dist: float = field(default=MISSING)
    drop_height: float = field(default=MISSING)
    alpha: float = field(default=MISSING)
    settle_steps: int = field(default=MISSING)
    drop_z_bias: float = field(default=MISSING)
    trigger_dist_bias: float = field(default=MISSING)
    spawn_link_clearance_xy: float = field(default=MISSING)
    spawn_z_clearance: float = field(default=MISSING)


@dataclass
class DisturbanceL2Config:
    alpha: float = field(default=MISSING)
    move_start_step: int = field(default=MISSING)
    move_end_step: int = field(default=MISSING)
    settle_steps: int = field(default=MISSING)
    arc_height: float = field(default=MISSING)
    start_offset: list[float] = field(default=MISSING)
    align_start_y_to_eef: bool = field(default=MISSING)
    outside_table_dist: float = field(default=MISSING)
    release_after_settle: bool = field(default=MISSING)
    mirror_start: bool = field(default=MISSING)
    start_dist_mult: float = field(default=MISSING)
    enable_random_start: bool = field(default=MISSING)
    random_start_xy_range: float = field(default=MISSING)
    random_start_z_range: float = field(default=MISSING)
    random_start_max_attempts: int = field(default=MISSING)


@dataclass
class DisturbanceL3Config:
    alpha: float = field(default=MISSING)
    dwell_steps: int = field(default=MISSING)
    entry_steps: int = field(default=MISSING)
    withdraw_speed_mult: float = field(default=MISSING)
    arm_speed: str = field(default=MISSING)
    start_mode: str = field(default=MISSING)
    enter_height: float = field(default=MISSING)
    move_start_step: int = field(default=MISSING)
    flip_side: bool = field(default=MISSING)
    corner_inward_shift: float = field(default=MISSING)
    corner_entry_backoff: float = field(default=MISSING)
    start_offset: list[float] = field(default=MISSING)
    mirror_start: bool = field(default=MISSING)
    start_dist_mult: float = field(default=MISSING)
    enable_pre_physics_collision_guard: bool = field(default=MISSING)
    ignore_collision_link_substrings: list[str] = field(default=MISSING)


@dataclass
class ObstacleConfig:
    pool: list[str] = field(default=MISSING)
    rotation_pool: list[str] = field(default=MISSING)
    models: list[str] = field(default=MISSING)
    scale: float = field(default=MISSING)
    half_size: list[float] = field(default=MISSING)
    mass: float = field(default=MISSING)
    color: list[float] = field(default=MISSING)
    selection_mode: str = field(default=MISSING)
    target_mode: str = field(default=MISSING)
    rescale: dict[str, float] = field(default=MISSING)
    upright_quats: dict[str, list[float]] = field(default=MISSING)
    semantic_families: dict[str, str] = field(default=MISSING)
    default_upright_quat: list[float] = field(default=MISSING)


@dataclass
class DisturbanceConfig:
    level: int = field(default=MISSING)
    l1: DisturbanceL1Config = field(default=MISSING)
    l2: DisturbanceL2Config = field(default=MISSING)
    l3: DisturbanceL3Config = field(default=MISSING)
    obstacle: ObstacleConfig = field(default=MISSING)


@dataclass
class HandConfig:
    geometry: str = field(default=MISSING)  # "box" | "mesh"
    mesh_path: str = field(default=MISSING)
    scale: float = field(default=MISSING)
    peak_velocity: dict[str, float] = field(default=MISSING)
    mass_kg: float = field(default=MISSING)
    contact_solref: list[float] = field(default=MISSING)
    contact_solimp: list[float] = field(default=MISSING)
    contact_friction: list[float] = field(default=MISSING)
    near_eef_m: float = field(default=MISSING)
    near_eef_max_step_m: float = field(default=MISSING)
    in_contact_max_step_m: float = field(default=MISSING)
    big_jump_m: float = field(default=MISSING)
    half_size: list[float] = field(default=MISSING)
    collision_offset: list[float] = field(default=MISSING)
    enter_height: float = field(default=MISSING)
    flinch_dist: float = field(default=MISSING)
    pre_physics_obb_half_scale: float = field(default=MISSING)


@dataclass
class SceneConfig:
    table_surface_z: float = field(default=MISSING)
    table_height: float = field(default=MISSING)
    support_clearance_z: float = field(default=MISSING)
    obstacle_avoid_margin: float = field(default=MISSING)
    l1_spawn_link_clearance_xy: float = field(default=MISSING)
    l1_spawn_z_clearance: float = field(default=MISSING)
    l1_spawn_eef_clearance_xy: float = field(default=MISSING)
    l1_spawn_eef_work_height: float = field(default=MISSING)
    l1_spawn_eef_radius: float = field(default=MISSING)
    l1_home_approach_eps: float = field(default=MISSING)
    l1_spawn_nudge_step: float = field(default=MISSING)
    l1_spawn_nudge_max_tries: int = field(default=MISSING)
    held_grasp_dist: float = field(default=MISSING)
    held_gripper_closed: float = field(default=MISSING)
    l12_collision_exclude_substrings: list[str] = field(default=MISSING)
    l12_contact_require_impulse: bool = field(default=MISSING)
    l2_keep_last_free_pose: bool = field(default=MISSING)
    l2_release_on_penetration: bool = field(default=MISSING)
    obstacle_avoid_shape: str = field(default=MISSING)
    l2_landing_push_m: float = field(default=MISSING)


@dataclass
class RunConfig:
    # Flat contract fields (M0); seed/max_steps mirrored from runtime.* by loader.
    backend: str = field(default=MISSING)
    suite: str = field(default=MISSING)
    level: int = field(default=MISSING)
    seed: int = field(default=MISSING)
    model: str = field(default=MISSING)
    checkpoint: str = field(default=MISSING)
    max_steps: int = field(default=MISSING)
    run_id: str = field(default=MISSING)
    contact: ContactConfig = field(default=MISSING)
    metrics: MetricsConfig = field(default=MISSING)
    diagnostics: DiagnosticsConfig = field(default=MISSING)
    output: OutputConfig = field(default=MISSING)
    # Nested views of the YAML tree (M1).
    runtime: RuntimeConfig = field(default=MISSING)
    disturbance: DisturbanceConfig = field(default=MISSING)
    hand: HandConfig = field(default=MISSING)
    scene: SceneConfig = field(default=MISSING)


# ---------------------------------------------------------------------------
# Reflection / allowlists
# ---------------------------------------------------------------------------

_NESTED_FIELD_TYPES: dict[type, dict[str, type]] = {
    MetricsConfig: {"reactivity": ReactivityConfig},
    DiagnosticsConfig: {
        "trace": DiagnosticsTraceConfig,
        "reaction": DiagnosticsReactionConfig,
        "stop": DiagnosticsStopConfig,
        "avoid": DiagnosticsAvoidConfig,
        "recovery": DiagnosticsRecoveryConfig,
        "hazard_end": DiagnosticsHazardEndConfig,
    },
    DisturbanceConfig: {
        "l1": DisturbanceL1Config,
        "l2": DisturbanceL2Config,
        "l3": DisturbanceL3Config,
        "obstacle": ObstacleConfig,
    },
    RunConfig: {
        "contact": ContactConfig,
        "metrics": MetricsConfig,
        "diagnostics": DiagnosticsConfig,
        "output": OutputConfig,
        "runtime": RuntimeConfig,
        "disturbance": DisturbanceConfig,
        "hand": HandConfig,
        "scene": SceneConfig,
    },
}


def iter_config_dataclasses() -> Iterable[type]:
    """Yield config dataclasses that must use MISSING defaults only."""
    return (
        ContactConfig,
        ReactivityConfig,
        MetricsConfig,
        DiagnosticsTraceConfig,
        DiagnosticsReactionConfig,
        DiagnosticsStopConfig,
        DiagnosticsAvoidConfig,
        DiagnosticsRecoveryConfig,
        DiagnosticsHazardEndConfig,
        DiagnosticsConfig,
        OutputConfig,
        RuntimeConfig,
        DisturbanceL1Config,
        DisturbanceL2Config,
        DisturbanceL3Config,
        ObstacleConfig,
        DisturbanceConfig,
        HandConfig,
        SceneConfig,
        RunConfig,
    )


def assert_no_embedded_defaults(cls: type | None = None) -> None:
    """Raise AssertionError if any config field has a non-MISSING default."""
    classes: tuple[type, ...] = (cls,) if cls is not None else tuple(iter_config_dataclasses())
    for klass in classes:
        for f in fields(klass):
            default = f.default
            if isinstance(default, Field):  # pragma: no cover
                raise AssertionError(f"{klass.__name__}.{f.name} unexpected Field as default")
            if default is DATACLASS_MISSING:
                raise AssertionError(
                    f"{klass.__name__}.{f.name} has no default; expected MISSING sentinel"
                )
            if f.default_factory is not DATACLASS_MISSING:  # type: ignore[misc]
                raise AssertionError(
                    f"{klass.__name__}.{f.name} uses default_factory; expected MISSING sentinel"
                )
            if default is not MISSING:
                raise AssertionError(
                    f"{klass.__name__}.{f.name} has embedded default {default!r}; expected MISSING"
                )


def _paths_for_dataclass(cls: type, prefix: str) -> set[str]:
    out: set[str] = set()
    nested = _NESTED_FIELD_TYPES.get(cls, {})
    for f in fields(cls):
        path = f"{prefix}.{f.name}" if prefix else f.name
        out.add(path)
        child = nested.get(f.name)
        if child is not None:
            out |= _paths_for_dataclass(child, path)
    return out


# Prefixes under which arbitrary child keys are allowed (open maps / lists of maps).
_OPEN_PREFIXES: frozenset[str] = frozenset(
    {
        "metrics.injury_thresholds_n",
        "metrics.sasr_penalty",
        "hand.peak_velocity",
        "disturbance.obstacle.rescale",
        "disturbance.obstacle.upright_quats",
        "disturbance.obstacle.semantic_families",
        "extra",
        "tasks",
        "suite_tasks",
        # M4 model YAML (S2 D11=B): nested blocks under configs/models/*.yaml
        "observation",
        "server",
        "resources",
        "checkpoint",
        # Shared-model overlays keyed by eval backend (e.g. starvla×robotwin).
        "per_backend",
    }
)

# Extra top-level / model-layer keys not mirrored as RunConfig nested fields.
_EXTRA_KNOWN_PATHS: frozenset[str] = frozenset(
    {
        "name",
        "action_space",
        "extra",
        "tasks",
        "suite_tasks",
        # M4 frozen model YAML keys (S2 D11=B minimal allowlist)
        "transport",
        "adapter",
        "chunk_size",
        "observation",
        "server",
        "resources",
        "backends",
        "checkpoint",
        "per_backend",
    }
)

# Legacy names forbidden in YAML (error + suggestion).
LEGACY_KEY_SUGGESTIONS: dict[str, str] = {
    "l2_drop_trigger_dist": "disturbance.l1.trigger_dist",
    "l1_trigger_dist": "disturbance.l1.trigger_dist",
    "l2_drop_height": "disturbance.l1.drop_height",
    "l1_drop_height": "disturbance.l1.drop_height",
    "l1_alpha": "disturbance.l1.alpha",
    "l2_alpha": "disturbance.l2.alpha",
    "l3_alpha": "disturbance.l3.alpha",
    "l3_interp_alpha": "disturbance.l3.alpha",
    "l2_hold_steps": "disturbance.l2.settle_steps",
    "l1_settle_steps": "disturbance.l2.settle_steps",
    "l2_settle_steps": "disturbance.l2.settle_steps",
    "l3_hand_mesh": "hand.geometry",
    "hand_geometry": "hand.geometry",
    "l3_hand_scale": "hand.scale",
    "hand_mesh_scale": "hand.scale",
    "topple_threshold_deg": "contact.topple_threshold_deg",
    "collision_force_equiv_dt": "contact.force_equiv_dt",
    "collision_force_report_window_s": "contact.force_report_window_s",
    "collision_impulse_point_reduce": "contact.impulse_reduce",
    "dynisafety_force_equiv_dt": "contact.force_equiv_dt",
    "dynisafety_force_report_window_s": "contact.force_report_window_s",
    "dynisafety_contact_impulse_reduce": "contact.impulse_reduce",
    "l2_mirror_start": "disturbance.l2.mirror_start",
    "l2_start_dist_mult": "disturbance.l2.start_dist_mult",
    "l2_enable_random_start": "disturbance.l2.enable_random_start",
    "l2_align_start_y_to_eef": "disturbance.l2.align_start_y_to_eef",
    "l2_move_end_step": "disturbance.l2.move_end_step",
    "l3_mirror_start": "disturbance.l3.mirror_start",
    "l3_start_dist_mult": "disturbance.l3.start_dist_mult",
    "l3_dwell_steps": "disturbance.l3.dwell_steps",
    "l3_ignore_collision_link_substrings": "disturbance.l3.ignore_collision_link_substrings",
    "force_anomaly_threshold_n": "metrics.force_anomaly_threshold_n",
    "impulse_anomaly_threshold_ns": "metrics.impulse_anomaly_threshold_ns",
}


def known_config_paths() -> frozenset[str]:
    """Allowed dotted key paths (containers and leaves) for unknown-key detection."""
    paths = _paths_for_dataclass(RunConfig, "")
    # YAML stores nested runtime/disturbance under the same names; also allow
    # selecting keys that mirror LoadRequest injection.
    paths |= set(_EXTRA_KNOWN_PATHS)
    return frozenset(paths)


def open_config_prefixes() -> frozenset[str]:
    """Prefixes that accept arbitrary descendants."""
    return _OPEN_PREFIXES


def level_relevant_prefixes(level: int) -> frozenset[str]:
    """Prefixes meaningful for the active disturbance level."""
    base = frozenset({"disturbance.level", "disturbance.obstacle"})
    if level <= 0:
        return base
    return base | frozenset({f"disturbance.l{level}"})


KNOWN_BACKENDS: tuple[str, ...] = ("libero", "simplerenv", "robotwin", "robocasa")


def backend_supported_levels(backend: str) -> frozenset[int]:
    """Built-in capability table (all current backends: 0..3)."""
    table = {
        "libero": frozenset({0, 1, 2, 3}),
        "simplerenv": frozenset({0, 1, 2, 3}),
        "robotwin": frozenset({0, 1, 2, 3}),
        "robocasa": frozenset({0, 1, 2, 3}),
    }
    return table.get(backend, frozenset())
