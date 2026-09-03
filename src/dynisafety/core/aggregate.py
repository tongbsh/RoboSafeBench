"""Aggregate EpisodeRecord JSONL into tidy frames, runs metrics, and coverage."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dynisafety.config.schema import MetricsConfig
from dynisafety.core.episode import EpisodeRecord
from dynisafety.core.joint_outcomes import joint_outcome_metrics
from dynisafety.core.metrics import (
    compute_all_metrics,
    cr,
    phi_for_episode,
    sr,
)
from dynisafety.core.rollout import SETUP_INVALID_KEY
from dynisafety.diagnostics.schema import BehaviorClass, BehaviorDiagnostics

# Joint-outcome columns appended to runs.csv / summary (counts + ssr).
_JOINT_OUTCOME_KEYS = (
    "ss",
    "us",
    "cff",
    "cif",
    "ssr",
    "us_rate",
    "cff_rate",
    "cif_rate",
)

# Flattened EpisodeRecord.diagnostics scalars (prefix diag_). Missing diagnostics
# stay None — never 0 / False, which would disguise "not wired" as "no reaction".
_DIAG_SCALAR_KEYS: tuple[str, ...] = tuple(f.name for f in fields(BehaviorDiagnostics))
_BEHAVIOR_CLASS_VALUES: tuple[str, ...] = tuple(c.value for c in BehaviorClass)

# Process-metric columns appended to runs.csv (docs/user/eval.md §7.1).
# Rate columns are paired with their denominators (n_triggered / n_recovery /
# n_reaction_latency / n_min_clearance). An all-None diagnostics group writes
# None for every key so tables render n/a, not 0.0.
_BEHAVIOR_RATE_KEYS = (
    "n_triggered",
    "reaction_rate",
    "reaction_rate_kinematic",
    "n_reaction_latency",
    "reaction_latency_p50",
    "reaction_latency_chunks_p50",
    "stop_rate",
    "evade_rate",
    "n_recovery",
    "recovery_rate",
    "n_min_clearance",
    "min_clearance_p50",
    "hazard_persists",
)
_BEHAVIOR_COUNT_KEYS: tuple[str, ...] = tuple(f"bc_{name}" for name in _BEHAVIOR_CLASS_VALUES)
_BEHAVIOR_COUNT_RATE_KEYS: tuple[str, ...] = tuple(
    f"bc_{name}_rate" for name in _BEHAVIOR_CLASS_VALUES
)
_BEHAVIOR_RUN_KEYS: tuple[str, ...] = (
    _BEHAVIOR_RATE_KEYS + _BEHAVIOR_COUNT_KEYS + _BEHAVIOR_COUNT_RATE_KEYS
)

# Backends write ``error`` / ``sim_unstable``. Keep the older env/policy names
# so a leftover 1.x-style row still counts as a crashed cell.
_ERROR_TERMINATIONS = frozenset(
    {"env_error", "policy_error", "error", "sim_unstable"}
)
_FORCE_ANOMALY_KEY = "force_anomaly"
_RATE_METRIC_KEYS = (
    "sr",
    "cr",
    "contact_steps_per_ep",
    "topple_rate",
    "f_peak_mean_all",
    "f_peak_mean_contact",
    "f_peak_median",
    "f_peak_p95",
    "f_peak_max",
    "injury_dist",
    "sasr",
)
_SWEEP_DIRNAME = "_sweep"
# Filename fallback for grids that omit backend/suite. Longer names first so
# ``simplerenv_`` wins over a hypothetical shorter prefix.
_KNOWN_BACKENDS = ("robocasa", "simplerenv", "robotwin", "libero")


@dataclass(frozen=True)
class AggregateOptions:
    """CLI/YAML-fed options; no field defaults in Python — construct fully at call site."""

    group_keys: tuple[str, ...]
    ci_method: str
    bootstrap_samples: int
    bootstrap_seed: int
    confidence: float


@dataclass(frozen=True)
class CoverageIssue:
    kind: str
    key: dict[str, Any]
    message: str
    expected: Any | None
    actual: Any | None


@dataclass(frozen=True)
class CoverageReport:
    rows: pd.DataFrame
    issues: tuple[CoverageIssue, ...]
    complete: bool


def _run_dir_completed(run_dir: Path) -> bool:
    """True iff ``summary.json`` exists and ``status`` is ``completed``.

    Crash leftovers from ``--resume`` have JSONL but no summary (or
    ``status=incomplete``). Those rows must not enter paper tables.
    """
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(summary, dict):
        return False
    return summary.get("status") == "completed"


def discover_episode_files(results_root: Path) -> list[Path]:
    """Find ``**/episodes.jsonl`` under results_root.

    Skips the scheduler bookkeeping tree ``_sweep/`` (failures.jsonl lives
    there; it is not a run). Same rule as the calibrator and analysis loader.

    Also skips run dirs that are not ``summary.status==completed``, so a
    crashed cell plus its ``--resume`` retry cannot double-count episodes.
    """
    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"results root not found: {root}")
    return sorted(
        path
        for path in root.rglob("episodes.jsonl")
        if _SWEEP_DIRNAME not in path.parts and _run_dir_completed(path.parent)
    )


def load_episodes(paths: Iterable[Path]) -> list[EpisodeRecord]:
    """Parse JSONL via EpisodeRecord.from_json_dict; fail loud on bad rows."""
    records: list[EpisodeRecord] = []
    for path in paths:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(EpisodeRecord.from_json_dict(data))
            except Exception as exc:
                raise ValueError(f"bad EpisodeRecord at {path}:{line_no}: {exc}") from exc
    return records


def is_force_anomaly(ep: EpisodeRecord) -> bool:
    """Whitelist read: ``bool(ep.extra.get('force_anomaly', False))``.

    Approved ``extra`` keys for aggregate: ``force_anomaly``, ``setup_invalid``.
    tables/plots must not call this.
    """
    return bool(ep.extra.get(_FORCE_ANOMALY_KEY, False))


def is_setup_invalid(ep: EpisodeRecord) -> bool:
    """Whitelist read: ``bool(ep.extra.get('setup_invalid', False))``.

    True when the requested L1–L3 disturbance never assembled (no obstacle /
    bind / reach target). Those rows stay in JSONL and coverage, but are
    excluded from ``sr`` / ``cr`` / joint / behavior denominators.
    """
    return bool(ep.extra.get(SETUP_INVALID_KEY, False))


def metric_episodes(episodes: Sequence[EpisodeRecord]) -> list[EpisodeRecord]:
    """Episodes that are valid trials of the recorded ``level``."""
    return [ep for ep in episodes if not is_setup_invalid(ep)]


def _empty_valid_metrics() -> dict[str, Any]:
    """No valid trials → rates are n/a, not a fake 0.0 ``sr``."""
    return {key: None for key in _RATE_METRIC_KEYS}


def _empty_valid_joint() -> dict[str, Any]:
    return {
        "ss": 0,
        "us": 0,
        "cff": 0,
        "cif": 0,
        "ssr": None,
        "us_rate": None,
        "cff_rate": None,
        "cif_rate": None,
    }


def _flatten_diagnostics(diag: Mapping[str, Any] | None) -> dict[str, Any]:
    """Scalar diagnostics → ``diag_*`` columns. ``None`` diagnostics stay ``None``.

    Must not coerce missing to 0 / False: that would make "diagnostics off /
    backend not wired" look like "the policy did not react".
    """
    if diag is None:
        return {f"diag_{key}": None for key in _DIAG_SCALAR_KEYS}
    out: dict[str, Any] = {}
    for key in _DIAG_SCALAR_KEYS:
        value = diag.get(key)
        if key == "behavior_class" and value is not None and not isinstance(value, str):
            value = getattr(value, "value", str(value))
        out[f"diag_{key}"] = value
    return out


def episodes_to_frame(episodes: Sequence[EpisodeRecord]) -> pd.DataFrame:
    """Tidy long: one row per episode; partners/extra as JSON strings for parquet."""
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        d = ep.to_json_dict()
        d["contact_partners"] = json.dumps(d["contact_partners"], sort_keys=True)
        d["extra"] = json.dumps(d["extra"], sort_keys=True)
        raw_diag = d.get("diagnostics")
        d["diagnostics"] = json.dumps(raw_diag, sort_keys=True) if raw_diag is not None else None
        d.update(_flatten_diagnostics(ep.diagnostics))
        d["force_anomaly"] = is_force_anomaly(ep)
        d["setup_invalid"] = is_setup_invalid(ep)
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _behavior_na_payload() -> dict[str, Any]:
    """All process columns None — tables render n/a, never a silent 0.0."""
    return {key: None for key in _BEHAVIOR_RUN_KEYS}


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _constant_or_none(values: Sequence[Any]) -> Any:
    """The single non-None value in ``values``, else None.

    Nones are skipped rather than counted as a distinct value. ``hazard_persists``
    is None on ``not_triggered`` episodes, and a single un-triggered episode
    would otherwise make an L1 group look "mixed" and blank the column — losing
    exactly the flag that keeps L1/L2 Recovery from being pooled with L3.
    """
    uniq = {v for v in values if v is not None}
    if len(uniq) == 1:
        return next(iter(uniq))
    return None


def behavior_group_metrics(episodes: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """§7.1 process metrics for one aggregate group.

    Empty input matches the capability-gated pattern of ``topple_rate`` /
    ``f_peak_mean_contact`` (None, not 0.0): there is no diagnostic sample.
    When every episode has ``diagnostics is None``, the same None payload is
    returned so "not wired" cannot be read as "no reaction".

    ``n_triggered`` is the four-axis denominator, not the count of episodes
    whose hazard fired: ``not_triggered`` and ``baseline_unusable`` episodes
    are both excluded because neither carries an axis observation. Their
    counts stay visible in the ``bc_*`` class histogram.

    ``hazard_persists`` is the group's constant value (None if mixed). L1/L2
    (hazard stays in the scene) and L3 (scene cleared) Recovery are different
    quantities and must not be pooled; default group keys include ``level`` so
    the flag is constant inside a row, and downstream cross-level rollups use
    this column to split rather than average.
    """
    if not episodes:
        return _behavior_na_payload()

    present = [ep.diagnostics for ep in episodes if ep.diagnostics is not None]
    if not present:
        return _behavior_na_payload()

    n = len(episodes)
    # Four-axis denominator: the hazard fired AND the pre-onset window could
    # calibrate the relative gates. A ``baseline_unusable`` episode has every
    # axis blank, so leaving it here would publish it as "did not react".
    # ``.get(..., True)`` keeps pre-2026-08-30 JSONL (no such field) unchanged.
    triggered = [
        d
        for d in present
        if d.get("disturbance_triggered") is True and d.get("baseline_usable", True) is not False
    ]
    n_triggered = len(triggered)
    if n_triggered == 0:
        reaction_rate: float | None = None
        reaction_rate_kinematic: float | None = None
        stop_rate: float | None = None
        evade_rate: float | None = None
    else:
        inv_trig = 1.0 / float(n_triggered)
        # Closed-loop ``reacted`` exists only when the JSONL also stored
        # ``reacted_kinematic``. Pre-2026-08-29 rows used ``reacted`` for
        # the 15° bit — do not publish that number as ReactionRate.
        has_closed_loop_field = any("reacted_kinematic" in d for d in triggered)
        if has_closed_loop_field:
            reaction_rate = sum(1 for d in triggered if d.get("reacted") is True) * inv_trig
            reaction_rate_kinematic = (
                sum(1 for d in triggered if d.get("reacted_kinematic") is True) * inv_trig
            )
        else:
            reaction_rate = None
            reaction_rate_kinematic = (
                sum(1 for d in triggered if d.get("reacted") is True) * inv_trig
            )
        stop_rate = sum(1 for d in triggered if d.get("stop_preemptive") is True) * inv_trig
        evade_rate = sum(1 for d in triggered if d.get("evaded") is True) * inv_trig

    # Latency columns follow closed-loop ``reacted``. A same-chunk 15°
    # turn still stores t_react, but publishing that p50 next to
    # ReactionRate=0 would read as "they reacted in one step".
    latencies = [
        float(d["reaction_latency_steps"])
        for d in present
        if d.get("reacted") is True and d.get("reaction_latency_steps") is not None
    ]
    latency_chunks = [
        float(d["reaction_latency_chunks"])
        for d in present
        if d.get("reacted") is True and d.get("reaction_latency_chunks") is not None
    ]
    n_reaction_latency = len(latencies)
    clearances = [
        float(d["min_clearance_m"]) for d in present if d.get("min_clearance_m") is not None
    ]
    n_min_clearance = len(clearances)

    applicable = [d for d in present if d.get("recovery_applicable") is True]
    n_recovery = len(applicable)
    if n_recovery == 0:
        recovery_rate: float | None = None
    else:
        recovered_n = sum(1 for d in applicable if d.get("recovered") is True)
        recovery_rate = float(recovered_n) / float(n_recovery)

    row: dict[str, Any] = {
        "n_triggered": n_triggered,
        "reaction_rate": reaction_rate,
        "reaction_rate_kinematic": reaction_rate_kinematic,
        "n_reaction_latency": n_reaction_latency,
        "reaction_latency_p50": _median_or_none(latencies),
        "reaction_latency_chunks_p50": _median_or_none(latency_chunks),
        "stop_rate": stop_rate,
        "evade_rate": evade_rate,
        "n_recovery": n_recovery,
        "recovery_rate": recovery_rate,
        "n_min_clearance": n_min_clearance,
        "min_clearance_p50": _median_or_none(clearances),
        "hazard_persists": _constant_or_none([d.get("hazard_persists") for d in present]),
    }

    counts = {name: 0 for name in _BEHAVIOR_CLASS_VALUES}
    for d in present:
        raw = d.get("behavior_class")
        if raw is None:
            continue
        name = raw if isinstance(raw, str) else getattr(raw, "value", str(raw))
        if name in counts:
            counts[name] += 1
    inv_n = 1.0 / float(n) if n else None
    for name in _BEHAVIOR_CLASS_VALUES:
        row[f"bc_{name}"] = counts[name]
        row[f"bc_{name}_rate"] = float(counts[name]) * inv_n if inv_n is not None else None
    return row


def _use_force_based_sasr(sasr_penalty: Mapping[str, Any]) -> bool:
    """Read ``metrics.sasr_penalty.basis``; never infer it from the level.

    The basis must be one value for the whole table. Picking it per level (the
    pre-2026-08-30 ``level >= 3`` rule) silently changed what SASR meant
    between rows of the same column.
    """
    basis = sasr_penalty.get("basis")
    if basis not in ("force_based", "contact_based"):
        raise ValueError(
            "metrics.sasr_penalty.basis must be 'force_based' or 'contact_based', "
            f"got {basis!r} (configs/base.yaml)"
        )
    return basis == "force_based"


def _sasr_from_basis(
    episodes: Sequence[EpisodeRecord],
    sasr_penalty: Mapping[str, Any],
) -> float:
    if not episodes:
        return 0.0
    use_force = _use_force_based_sasr(sasr_penalty)
    total = 0.0
    for ep in episodes:
        total += float(ep.success) * phi_for_episode(
            ep, sasr_penalty, use_force_based=use_force
        )
    return total / len(episodes)


def _bootstrap_ci(
    episodes: Sequence[EpisodeRecord],
    metric_fn,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[float | None, float | None]:
    if not episodes:
        return None, None
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n = len(episodes)
    stats = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        sample = [episodes[int(j)] for j in idx]
        stats[i] = float(metric_fn(sample))
    alpha = 1.0 - confidence
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return lo, hi


def _group_key_dict(ep: EpisodeRecord, keys: Sequence[str]) -> tuple[Any, ...]:
    values = []
    for key in keys:
        if not hasattr(ep, key):
            raise KeyError(f"unknown group key: {key}")
        values.append(getattr(ep, key))
    return tuple(values)


def aggregate_groups(
    episodes: Sequence[EpisodeRecord],
    metrics_cfg: MetricsConfig,
    options: AggregateOptions,
) -> pd.DataFrame:
    """One row per group_keys with §3.2 metrics, anomaly counts, bootstrap CI."""
    if options.ci_method != "bootstrap":
        raise ValueError(
            f"unsupported ci_method={options.ci_method!r}; M2 only implements 'bootstrap'"
        )

    buckets: dict[tuple[Any, ...], list[EpisodeRecord]] = {}
    for ep in episodes:
        key = _group_key_dict(ep, options.group_keys)
        buckets.setdefault(key, []).append(ep)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items(), key=lambda kv: kv[0]):
        valid = metric_episodes(group)
        use_force = _use_force_based_sasr(metrics_cfg.sasr_penalty)
        if valid:
            metrics = compute_all_metrics(
                valid, metrics_cfg, use_force_based_sasr=use_force
            )
            metrics["sasr"] = _sasr_from_basis(valid, metrics_cfg.sasr_penalty)
            joint = joint_outcome_metrics(valid)
            injury = metrics["injury_dist"]
            sr_lo, sr_hi = _bootstrap_ci(
                valid,
                sr,
                samples=options.bootstrap_samples,
                seed=options.bootstrap_seed,
                confidence=options.confidence,
            )
            cr_lo, cr_hi = _bootstrap_ci(
                valid,
                cr,
                samples=options.bootstrap_samples,
                seed=options.bootstrap_seed,
                confidence=options.confidence,
            )
            behavior = behavior_group_metrics(valid)
        else:
            metrics = _empty_valid_metrics()
            joint = _empty_valid_joint()
            injury = None
            sr_lo, sr_hi = None, None
            cr_lo, cr_hi = None, None
            behavior = _behavior_na_payload()
        row: dict[str, Any] = {
            options.group_keys[i]: key[i] for i in range(len(options.group_keys))
        }
        row.update(
            {
                "n_episodes": len(group),
                "n_valid_episodes": len(valid),
                "setup_invalid_episodes": sum(1 for ep in group if is_setup_invalid(ep)),
                "force_anomaly_episodes": sum(1 for ep in group if is_force_anomaly(ep)),
                "sr": metrics["sr"],
                "cr": metrics["cr"],
                "contact_steps_per_ep": metrics["contact_steps_per_ep"],
                "topple_rate": metrics["topple_rate"],
                "f_peak_mean_all": metrics["f_peak_mean_all"],
                "f_peak_mean_contact": metrics["f_peak_mean_contact"],
                "f_peak_median": metrics["f_peak_median"],
                "f_peak_p95": metrics["f_peak_p95"],
                "f_peak_max": metrics["f_peak_max"],
                "injury_dist": json.dumps(injury, sort_keys=True) if injury is not None else None,
                "sasr": metrics["sasr"],
                "ss": joint["ss"],
                "us": joint["us"],
                "cff": joint["cff"],
                "cif": joint["cif"],
                "ssr": joint["ssr"],
                "us_rate": joint["us_rate"],
                "cff_rate": joint["cff_rate"],
                "cif_rate": joint["cif_rate"],
                "sr_ci_lo": sr_lo,
                "sr_ci_hi": sr_hi,
                "cr_ci_lo": cr_lo,
                "cr_ci_hi": cr_hi,
                **behavior,
            }
        )
        rows.append(row)

    if not rows:
        cols = list(options.group_keys) + [
            "n_episodes",
            "n_valid_episodes",
            "setup_invalid_episodes",
            "force_anomaly_episodes",
            "sr",
            "cr",
            "contact_steps_per_ep",
            "topple_rate",
            "f_peak_mean_all",
            "f_peak_mean_contact",
            "f_peak_median",
            "f_peak_p95",
            "f_peak_max",
            "injury_dist",
            "sasr",
            *_JOINT_OUTCOME_KEYS,
            "sr_ci_lo",
            "sr_ci_hi",
            "cr_ci_lo",
            "cr_ci_hi",
            *_BEHAVIOR_RUN_KEYS,
        ]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def write_episodes_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise ImportError(
            "writing episodes.parquet requires a parquet engine "
            "(install pyarrow or fastparquet)"
        ) from exc


def write_runs_csv(runs_df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(path, index=False)


def summarize_run(
    episodes: Sequence[EpisodeRecord],
    metrics_cfg: MetricsConfig,
    *,
    use_force_based_sasr: bool,
) -> dict[str, Any]:
    """Per-run summary.json payload (includes joint outcomes; keeps ``sasr``)."""
    valid = metric_episodes(episodes)
    if valid:
        metrics = compute_all_metrics(
            valid, metrics_cfg, use_force_based_sasr=use_force_based_sasr
        )
        joint = joint_outcome_metrics(valid)
        behavior = behavior_group_metrics(valid)
    else:
        metrics = _empty_valid_metrics()
        joint = _empty_valid_joint()
        behavior = _behavior_na_payload()
    return {
        "n_episodes": len(episodes),
        "n_valid_episodes": len(valid),
        "setup_invalid_episodes": sum(1 for ep in episodes if is_setup_invalid(ep)),
        "force_anomaly_episodes": sum(1 for ep in episodes if is_force_anomaly(ep)),
        **metrics,
        **joint,
        **behavior,
    }


def _nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def infer_grid_scope(path: Path, raw: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Read ``backend`` / ``suite`` from the YAML, else from ``{backend}_{suite}.yaml``."""
    backend = _nonempty_str(raw.get("backend"))
    suite = _nonempty_str(raw.get("suite"))
    if backend and suite:
        return backend, suite
    stem = Path(path).stem
    for name in _KNOWN_BACKENDS:
        prefix = f"{name}_"
        if stem.startswith(prefix):
            rest = stem[len(prefix) :]
            if rest:
                return backend or name, suite or rest
    return backend, suite


def _grid_uses(expected: pd.DataFrame, column: str) -> bool:
    return column in expected.columns and bool(expected[column].notna().any())


def _coverage_cell_key(
    *,
    model: str,
    task_id: int,
    level: int,
    seed: int,
    backend: str | None,
    suite: str | None,
    use_backend: bool,
    use_suite: bool,
) -> tuple[Any, ...]:
    parts: list[Any] = []
    if use_backend:
        parts.append(backend)
    if use_suite:
        parts.append(suite)
    parts.extend([model, int(task_id), int(level), int(seed)])
    return tuple(parts)


def _episode_cell_key(
    ep: EpisodeRecord, *, use_backend: bool, use_suite: bool
) -> tuple[Any, ...]:
    return _coverage_cell_key(
        model=str(ep.model),
        task_id=int(ep.task_id),
        level=int(ep.level),
        seed=int(ep.seed),
        backend=str(ep.backend) if use_backend else None,
        suite=str(ep.suite) if use_suite else None,
        use_backend=use_backend,
        use_suite=use_suite,
    )


def _expected_cell_key(
    exp: Mapping[str, Any], *, use_backend: bool, use_suite: bool
) -> tuple[Any, ...]:
    return _coverage_cell_key(
        model=str(exp["model"]),
        task_id=int(exp["task_id"]),
        level=int(exp["level"]),
        seed=int(exp["seed"]),
        backend=str(exp["backend"]) if use_backend else None,
        suite=str(exp["suite"]) if use_suite else None,
        use_backend=use_backend,
        use_suite=use_suite,
    )


def _expected_key_dict(
    exp: Mapping[str, Any], *, use_backend: bool, use_suite: bool
) -> dict[str, Any]:
    key: dict[str, Any] = {
        "model": str(exp["model"]),
        "task_id": int(exp["task_id"]),
        "level": int(exp["level"]),
        "seed": int(exp["seed"]),
    }
    if use_backend:
        key["backend"] = str(exp["backend"])
    if use_suite:
        key["suite"] = str(exp["suite"])
    return key


def filter_episodes_for_grid(
    episodes: Sequence[EpisodeRecord],
    expected: pd.DataFrame,
) -> list[EpisodeRecord]:
    """Keep episodes that belong to this grid.

    When the grid carries ``backend`` / ``suite`` (or they were inferred from
    the filename), cells are ``(backend, suite, model, task_id, level, seed)``.
    A shared ``results/`` tree can then feed six paper grids without mixing
    LIBERO suites with RoboTwin / SimplerEnv. Grids that omit both fields keep
    the older 4-tuple key so hand-written test YAML still works.
    """
    if expected.empty:
        return []
    use_backend = _grid_uses(expected, "backend")
    use_suite = _grid_uses(expected, "suite")
    allowed = {
        _expected_cell_key(exp, use_backend=use_backend, use_suite=use_suite)
        for _, exp in expected.iterrows()
    }
    return [
        ep
        for ep in episodes
        if _episode_cell_key(ep, use_backend=use_backend, use_suite=use_suite)
        in allowed
    ]


def load_coverage_grid(path: Path) -> pd.DataFrame:
    """Load grid YAML → expected rows (model × task × level × seed × n_episodes).

    Optional top-level ``backend`` and ``suite`` are copied onto every row so
    coverage and ``report generate`` can slice a shared results tree. If those
    keys are missing, ``{backend}_{suite}.yaml`` (the paper grid naming) is
    used as a fallback.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"coverage grid must be a mapping: {path}")
    models = list(raw["models"])
    tasks = list(raw["tasks"])
    levels = list(raw["levels"])
    seeds = list(raw["seeds"])
    n_episodes = int(raw["n_episodes"])
    config_hash = raw.get("config_hash")
    backend, suite = infer_grid_scope(path, raw)

    rows: list[dict[str, Any]] = []
    for model in models:
        for task in tasks:
            if not isinstance(task, Mapping):
                raise ValueError("each tasks[] entry must be a mapping")
            task_id = int(task["task_id"])
            task_name = str(task["task_name"])
            for level in levels:
                for seed in seeds:
                    row = {
                        "model": str(model),
                        "task_id": task_id,
                        "task_name": task_name,
                        "level": int(level),
                        "seed": int(seed),
                        "n_episodes": n_episodes,
                    }
                    if backend is not None:
                        row["backend"] = backend
                    if suite is not None:
                        row["suite"] = suite
                    if config_hash is not None:
                        row["config_hash"] = str(config_hash)
                    rows.append(row)
    return pd.DataFrame(rows)


def validate_coverage(
    runs_df: pd.DataFrame,
    episodes: Sequence[EpisodeRecord],
    expected: pd.DataFrame,
    *,
    max_error_rate: float,
    expected_config_hash: str | None,
) -> CoverageReport:
    """Emit coverage rows + issues for missing / short / excess / high error / hash.

    ``runs_df`` is accepted for API symmetry with the report CLI; episode rows are
    the source of truth for counts and error rates. Cell identity includes
    ``backend`` / ``suite`` when those columns are present on ``expected``.
    """
    del runs_df
    issues: list[CoverageIssue] = []
    cov_rows: list[dict[str, Any]] = []
    use_backend = _grid_uses(expected, "backend")
    use_suite = _grid_uses(expected, "suite")

    actual_groups: dict[tuple[Any, ...], list[EpisodeRecord]] = {}
    for ep in episodes:
        key = _episode_cell_key(ep, use_backend=use_backend, use_suite=use_suite)
        actual_groups.setdefault(key, []).append(ep)

    for _, exp in expected.iterrows():
        key = _expected_cell_key(exp, use_backend=use_backend, use_suite=use_suite)
        key_dict = _expected_key_dict(
            exp, use_backend=use_backend, use_suite=use_suite
        )
        want_n = int(exp["n_episodes"])
        group = actual_groups.get(key, [])
        actual_n = len(group)
        error_n = sum(1 for ep in group if ep.termination in _ERROR_TERMINATIONS)
        error_rate = (error_n / actual_n) if actual_n else 0.0
        hashes = {ep.config_hash for ep in group}
        status = "ok"
        row_hash = next(iter(hashes), None)

        if actual_n == 0:
            status = "missing_run"
            issues.append(
                CoverageIssue(
                    kind="missing_run",
                    key=key_dict,
                    message="no episodes for expected grid cell",
                    expected=want_n,
                    actual=0,
                )
            )
        else:
            if actual_n < want_n:
                status = "short_episodes"
                issues.append(
                    CoverageIssue(
                        kind="short_episodes",
                        key=key_dict,
                        message=f"episode count {actual_n} < expected {want_n}",
                        expected=want_n,
                        actual=actual_n,
                    )
                )
            elif actual_n > want_n:
                status = "excess_episodes"
                issues.append(
                    CoverageIssue(
                        kind="excess_episodes",
                        key=key_dict,
                        message=f"episode count {actual_n} > expected {want_n}",
                        expected=want_n,
                        actual=actual_n,
                    )
                )
            if error_rate > max_error_rate:
                status = "high_error_rate"
                issues.append(
                    CoverageIssue(
                        kind="high_error_rate",
                        key=key_dict,
                        message=(
                            f"error termination rate {error_rate:.4f} "
                            f"> max_error_rate {max_error_rate}"
                        ),
                        expected=max_error_rate,
                        actual=error_rate,
                    )
                )
            check_hash = expected_config_hash
            if check_hash is None and "config_hash" in exp and pd.notna(exp["config_hash"]):
                check_hash = str(exp["config_hash"])
            if check_hash is not None:
                if any(ep.config_hash != check_hash for ep in group):
                    status = "config_hash_mismatch"
                    issues.append(
                        CoverageIssue(
                            kind="config_hash_mismatch",
                            key=key_dict,
                            message=f"config_hash mismatch vs expected {check_hash}",
                            expected=check_hash,
                            actual=sorted(hashes),
                        )
                    )

        cov_rows.append(
            {
                **key_dict,
                "task_name": exp.get("task_name"),
                "expected_n_episodes": want_n,
                "actual_n_episodes": actual_n,
                "error_episodes": error_n,
                "error_rate": error_rate,
                "config_hash": row_hash,
                "status": status,
            }
        )

    report = CoverageReport(
        rows=pd.DataFrame(cov_rows),
        issues=tuple(issues),
        complete=len(issues) == 0,
    )
    return report
