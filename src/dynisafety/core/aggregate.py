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

# Process-metric columns appended to runs.csv (docs/behavior_diagnostics.md §7.1).
# Rate columns are paired with their denominators (n_triggered / n_recovery /
# n_reaction_latency / n_min_clearance). An all-None diagnostics group writes
# None for every key so tables render n/a, not 0.0.
_BEHAVIOR_RATE_KEYS = (
    "n_triggered",
    "reaction_rate",
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

_ERROR_TERMINATIONS = frozenset({"env_error", "policy_error"})
_FORCE_ANOMALY_KEY = "force_anomaly"


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


def discover_episode_files(results_root: Path) -> list[Path]:
    """Find ``**/episodes.jsonl`` under results_root."""
    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"results root not found: {root}")
    return sorted(root.rglob("episodes.jsonl"))


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

    Sole approved ``extra`` key for aggregate; tables/plots must not call this.
    """
    return bool(ep.extra.get(_FORCE_ANOMALY_KEY, False))


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
    triggered = [d for d in present if d.get("disturbance_triggered") is True]
    n_triggered = len(triggered)
    if n_triggered == 0:
        reaction_rate: float | None = None
        stop_rate: float | None = None
        evade_rate: float | None = None
    else:
        inv_trig = 1.0 / float(n_triggered)
        # reacted / stop_preemptive / evaded are already the gated flags
        # (pre-contact reaction; preemptive brake, not crash-stop).
        reaction_rate = sum(1 for d in triggered if d.get("reacted") is True) * inv_trig
        stop_rate = sum(1 for d in triggered if d.get("stop_preemptive") is True) * inv_trig
        evade_rate = sum(1 for d in triggered if d.get("evaded") is True) * inv_trig

    latencies = [
        float(d["reaction_latency_steps"])
        for d in present
        if d.get("reaction_latency_steps") is not None
    ]
    latency_chunks = [
        float(d["reaction_latency_chunks"])
        for d in present
        if d.get("reaction_latency_chunks") is not None
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


def _use_force_based_sasr(ep: EpisodeRecord) -> bool:
    return int(ep.level) >= 3


def _sasr_per_episode_rules(
    episodes: Sequence[EpisodeRecord],
    sasr_penalty: Mapping[str, Any],
) -> float:
    if not episodes:
        return 0.0
    total = 0.0
    for ep in episodes:
        phi = phi_for_episode(
            ep,
            sasr_penalty,
            use_force_based=_use_force_based_sasr(ep),
        )
        total += float(ep.success) * phi
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
        # Majority level for compute_all_metrics flag; sasr overwritten below.
        use_force = sum(1 for ep in group if _use_force_based_sasr(ep)) >= (
            len(group) / 2.0
        )
        metrics = compute_all_metrics(
            group, metrics_cfg, use_force_based_sasr=use_force
        )
        metrics["sasr"] = _sasr_per_episode_rules(group, metrics_cfg.sasr_penalty)
        joint = joint_outcome_metrics(group)
        injury = metrics["injury_dist"]
        sr_lo, sr_hi = _bootstrap_ci(
            group,
            sr,
            samples=options.bootstrap_samples,
            seed=options.bootstrap_seed,
            confidence=options.confidence,
        )
        cr_lo, cr_hi = _bootstrap_ci(
            group,
            cr,
            samples=options.bootstrap_samples,
            seed=options.bootstrap_seed,
            confidence=options.confidence,
        )
        row: dict[str, Any] = {
            options.group_keys[i]: key[i] for i in range(len(options.group_keys))
        }
        row.update(
            {
                "n_episodes": len(group),
                "force_anomaly_episodes": sum(1 for ep in group if is_force_anomaly(ep)),
                "sr": metrics["sr"],
                "cr": metrics["cr"],
                "contact_steps_per_ep": metrics["contact_steps_per_ep"],
                "topple_rate": metrics["topple_rate"],
                "f_peak_mean_all": metrics["f_peak_mean_all"],
                "f_peak_mean_contact": metrics["f_peak_mean_contact"],
                "f_peak_p95": metrics["f_peak_p95"],
                "f_peak_max": metrics["f_peak_max"],
                "injury_dist": json.dumps(injury, sort_keys=True),
                "sasr": metrics["sasr"],
                "safety_score": metrics["safety_score"],
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
                **behavior_group_metrics(group),
            }
        )
        rows.append(row)

    if not rows:
        cols = list(options.group_keys) + [
            "n_episodes",
            "force_anomaly_episodes",
            "sr",
            "cr",
            "contact_steps_per_ep",
            "topple_rate",
            "f_peak_mean_all",
            "f_peak_mean_contact",
            "f_peak_p95",
            "f_peak_max",
            "injury_dist",
            "sasr",
            "safety_score",
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
    metrics = compute_all_metrics(
        episodes, metrics_cfg, use_force_based_sasr=use_force_based_sasr
    )
    joint = joint_outcome_metrics(episodes)
    return {
        "n_episodes": len(episodes),
        "force_anomaly_episodes": sum(1 for ep in episodes if is_force_anomaly(ep)),
        **metrics,
        **joint,
        **behavior_group_metrics(episodes),
    }


def load_coverage_grid(path: Path) -> pd.DataFrame:
    """Load grid YAML → expected rows (model × task × level × seed × n_episodes)."""
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
    """Emit coverage rows + issues for missing / short / high error / hash mismatch.

    ``runs_df`` is accepted for API symmetry with the report CLI; episode rows are
    the source of truth for counts and error rates.
    """
    del runs_df
    issues: list[CoverageIssue] = []
    cov_rows: list[dict[str, Any]] = []

    # Index actual episodes by (model, task_id, level, seed)
    actual_groups: dict[tuple[Any, ...], list[EpisodeRecord]] = {}
    for ep in episodes:
        key = (ep.model, ep.task_id, ep.level, ep.seed)
        actual_groups.setdefault(key, []).append(ep)

    for _, exp in expected.iterrows():
        key = (
            str(exp["model"]),
            int(exp["task_id"]),
            int(exp["level"]),
            int(exp["seed"]),
        )
        key_dict = {
            "model": key[0],
            "task_id": key[1],
            "level": key[2],
            "seed": key[3],
        }
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
