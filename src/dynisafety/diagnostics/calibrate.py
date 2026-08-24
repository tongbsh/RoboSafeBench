"""Offline threshold calibration for behavior diagnostics (P5).

Reads saved traces and re-runs ``analyze_trace`` in memory. Never steps a
simulator, never writes back into a results tree, and never mutates a
``Trace`` in place — virtual onsets use ``dataclasses.replace`` on a copy.

Thresholds come from ``configs/base.yaml`` via the config loader. This module
does not embed diagnostic defaults.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import yaml

from dynisafety.config.loader import _build_dataclass, _read_yaml
from dynisafety.config.resolve import find_workspace_root
from dynisafety.config.schema import MISSING, DiagnosticsConfig
from dynisafety.diagnostics.analyze import analyze_trace, compute_speeds
from dynisafety.diagnostics.reader import iter_traces
from dynisafety.diagnostics.schema import BehaviorClass, BehaviorDiagnostics, Trace, TraceHeader

# Knobs scanned at ±50% (docs/behavior_diagnostics.md §8 / P5 plan).
SENSITIVITY_KNOBS: tuple[str, ...] = (
    "reaction.sensitivity",
    "reaction.sigma_rel_floor",
    "reaction.heading_change_deg",
    "stop.speed_frac",
    "avoid.heading_change_deg",
    "avoid.min_speed_frac",
    "recovery.speed_frac",
    "recovery.min_post_hazard_steps",
)
FPR_GRID_KNOBS: tuple[str, ...] = ("reaction.sensitivity", "reaction.sigma_rel_floor")
_DEFAULT_SCALES: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
_DEFAULT_FACTORS: tuple[float, ...] = (0.5, 1.0, 1.5)
_RATE_KEYS: tuple[str, ...] = (
    "reaction_rate",
    "stop_rate",
    "evade_rate",
    "recovery_rate",
)
_CLASS_VALUES: tuple[str, ...] = tuple(c.value for c in BehaviorClass)
_TRACE_FILE_SUFFIXES: tuple[str, ...] = (".jsonl", ".jsonl.gz")
# t_onset=0 leaves no pre-onset speed sample (v[0] is NaN). Those draws are
# degenerate FPR trials, not a policy false positive.
_MIN_ONSET_FOR_FPR: int = 2
_WIDE_SENSITIVITY: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
_WIDE_SIGMA_REL: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.80)
_HEADING_PROBE_DEG: tuple[float, ...] = (20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0)


class CalibrationError(ValueError):
    """User-facing calibration failure (missing data, incomplete YAML, …)."""


@dataclass(frozen=True)
class EpisodeOutcome:
    """Episode-level fields the frozen ``Trace`` schema does not carry."""

    success: bool
    success_step: int | None
    first_contact_step: int | None
    contact_steps: int
    backend: str
    model: str
    suite: str
    task_id: int
    level: int
    seed: int
    run_id: str
    episode_index: int


@dataclass(frozen=True)
class EpisodeBundle:
    """One episode: on-disk trace plus the outcomes needed to re-analyze it."""

    run_dir: Path
    trace_path: Path
    trace: Trace
    outcome: EpisodeOutcome


@dataclass(frozen=True)
class CalibrationResult:
    fpr_curve: pd.DataFrame
    sensitivity: pd.DataFrame
    baseline_health: pd.DataFrame
    recommendation: str
    fingerprints_before: dict[str, str]
    fingerprints_after: dict[str, str]
    n_l0: int
    n_l1_l3: int
    n_onset_pool: int
    n_onset_pool_usable: int = 0
    fpr_heading: pd.DataFrame | None = None
    fpr_curve_wide: pd.DataFrame | None = None
    fpr_curve_magnitude: pd.DataFrame | None = None

    @property
    def traces_unmodified(self) -> bool:
        return all(
            self.fingerprints_after.get(key) == digest
            for key, digest in self.fingerprints_before.items()
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_baseline_diagnostics(base_yaml: Path | str | None = None) -> DiagnosticsConfig:
    """Load ``diagnostics.*`` from ``configs/base.yaml`` through the loader.

    Does not hard-code thresholds. Incomplete YAML (MISSING sentinels) fails
    loudly so a truncated test fixture cannot silently invent defaults.
    """
    path = Path(base_yaml) if base_yaml is not None else _default_base_yaml()
    if not path.is_file():
        raise CalibrationError(f"base YAML not found: {path}")
    raw = _read_yaml(path)
    block = raw.get("diagnostics")
    if not isinstance(block, dict):
        raise CalibrationError(f"{path} has no diagnostics mapping")
    cfg = _build_dataclass(DiagnosticsConfig, block)
    _assert_diagnostics_complete(cfg, source=path)
    return cfg


def _default_base_yaml() -> Path:
    return find_workspace_root() / "configs" / "base.yaml"


def _assert_diagnostics_complete(cfg: DiagnosticsConfig, *, source: Path) -> None:
    missing = _missing_paths(cfg, prefix="diagnostics")
    if missing:
        raise CalibrationError(
            f"{source} is missing diagnostics fields: {missing}. "
            "Thresholds must come from YAML, not Python defaults."
        )


def _missing_paths(obj: Any, *, prefix: str) -> list[str]:
    found: list[str] = []
    if obj is MISSING:
        return [prefix]
    if not is_dataclass(obj) or isinstance(obj, type):
        return found
    for item in fields(obj):
        value = getattr(obj, item.name)
        path = f"{prefix}.{item.name}"
        found.extend(_missing_paths(value, prefix=path))
    return found


def knob_value(cfg: DiagnosticsConfig, dotted: str) -> float:
    section, name = _split_knob(dotted)
    return getattr(getattr(cfg, section), name)


def set_knob(cfg: DiagnosticsConfig, dotted: str, value: float) -> DiagnosticsConfig:
    """Return a new config with one dotted knob replaced. Input is not mutated."""
    section, name = _split_knob(dotted)
    inner = getattr(cfg, section)
    current = getattr(inner, name)
    if isinstance(current, bool):
        raise CalibrationError(f"cannot sweep boolean knob {dotted}")
    if isinstance(current, int) and not isinstance(current, bool):
        coerced: Any = int(round(float(value)))
    else:
        coerced = type(current)(value)
    return replace(cfg, **{section: replace(inner, **{name: coerced})})


def _split_knob(dotted: str) -> tuple[str, str]:
    parts = dotted.split(".")
    if len(parts) != 2:
        raise CalibrationError(f"knob must be section.field, got {dotted!r}")
    section, name = parts
    if not hasattr(DiagnosticsConfig, section):
        raise CalibrationError(f"unknown diagnostics section {section!r}")
    return section, name


# ---------------------------------------------------------------------------
# Discovery (read-only)
# ---------------------------------------------------------------------------


def discover_bundles(results_root: Path | str) -> list[EpisodeBundle]:
    """Load episode/trace pairs under ``results_root``.

    Skips ``_sweep``. When several run dirs share the same
    ``(backend, model, suite, task, level, seed)`` (smoke 1-ep + calib 25-ep),
    keeps the dir with the most traces.
    """
    root = Path(results_root)
    if not root.is_dir():
        raise CalibrationError(f"results root not found: {root}")

    grouped: dict[tuple[Any, ...], list[EpisodeBundle]] = {}
    for run_dir in _iter_run_dirs(root):
        for bundle in _load_run(run_dir):
            key = (
                bundle.outcome.backend,
                bundle.outcome.model,
                bundle.outcome.suite,
                bundle.outcome.task_id,
                bundle.outcome.level,
                bundle.outcome.seed,
                bundle.outcome.episode_index,
            )
            grouped.setdefault(key, []).append(bundle)

    chosen: list[EpisodeBundle] = []
    for group in grouped.values():
        chosen.append(max(group, key=lambda b: (b.trace.header.n_steps, b.run_dir.stat().st_mtime)))
    chosen.sort(key=lambda b: (b.outcome.level, b.outcome.task_id, b.outcome.episode_index, str(b.run_dir)))
    return chosen


def _iter_run_dirs(results_root: Path) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for episodes in results_root.rglob("episodes.jsonl"):
        run_dir = episodes.parent
        if "_sweep" in run_dir.parts:
            continue
        if not (run_dir / "traces").is_dir():
            continue
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append(run_dir)
    return sorted(found)


def _load_run(run_dir: Path) -> list[EpisodeBundle]:
    outcomes = _load_episode_outcomes(run_dir / "episodes.jsonl")
    bundles: list[EpisodeBundle] = []
    for trace in iter_traces(run_dir, strict=True):
        header = trace.header
        rel = header_trace_relpath(header)
        trace_path = run_dir / rel
        if not trace_path.is_file():
            # index.json filename is the source of truth when present.
            candidates = list((run_dir / "traces").glob(f"ep{header.episode_index:04d}.jsonl*"))
            if not candidates:
                continue
            trace_path = candidates[0]
        outcome = outcomes.get(header.episode_index)
        if outcome is None:
            outcome = _outcome_from_trace(trace)
        bundles.append(
            EpisodeBundle(run_dir=run_dir, trace_path=trace_path, trace=trace, outcome=outcome)
        )
    return bundles


def header_trace_relpath(header: TraceHeader) -> str:
    return f"traces/ep{int(header.episode_index):04d}.jsonl.gz"


def _load_episode_outcomes(path: Path) -> dict[int, EpisodeOutcome]:
    if not path.is_file():
        return {}
    out: dict[int, EpisodeOutcome] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        if not isinstance(row, dict):
            raise CalibrationError(f"{path}:{line_no} is not a JSON object")
        idx = int(row["episode_index"])
        out[idx] = EpisodeOutcome(
            success=bool(row.get("success", False)),
            success_step=_opt_int(row.get("success_step")),
            first_contact_step=_opt_int(row.get("first_contact_step")),
            contact_steps=int(row.get("contact_steps") or 0),
            backend=str(row.get("backend") or ""),
            model=str(row.get("model") or ""),
            suite=str(row.get("suite") or ""),
            task_id=int(row.get("task_id") or 0),
            level=int(row.get("level") or 0),
            seed=int(row.get("seed") or 0),
            run_id=str(row.get("run_id") or ""),
            episode_index=idx,
        )
    return out


def _outcome_from_trace(trace: Trace) -> EpisodeOutcome:
    hits = [int(step.t) for step in trace.steps if step.in_contact]
    header = trace.header
    return EpisodeOutcome(
        success=False,
        success_step=None,
        first_contact_step=hits[0] if hits else None,
        contact_steps=len(hits),
        backend=str(header.backend),
        model="",
        suite="",
        task_id=0,
        level=int(header.level),
        seed=int(header.seed),
        run_id=str(header.run_id),
        episode_index=int(header.episode_index),
    )


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


# ---------------------------------------------------------------------------
# Read-only fingerprints
# ---------------------------------------------------------------------------


def fingerprint_results(results_root: Path | str) -> dict[str, str]:
    """SHA-256 of finished episode traces (``traces/ep*.jsonl*``) only.

    ``episodes.jsonl`` / ``index.json`` are append-updated by a live sweep;
    hashing them would false-fail a calibrator running next to collection.
    Each ``epXXXX`` file is written once and must stay byte-identical.
    """
    root = Path(results_root)
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "_sweep" in path.parts:
            continue
        if not _is_protected(path):
            continue
        rel = path.relative_to(root).as_posix()
        out[rel] = _sha256_file(path)
    return out


def _is_protected(path: Path) -> bool:
    name = path.name
    if not name.startswith("ep"):
        return False
    return name.endswith(_TRACE_FILE_SUFFIXES)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Virtual onset (L0 false-positive anchor)
# ---------------------------------------------------------------------------


def inject_virtual_onset(trace: Trace, t_onset: int) -> Trace:
    """In-memory header rewrite. The input ``Trace`` and its steps are unchanged."""
    header = replace(
        trace.header,
        t_onset=int(t_onset),
        disturbance_triggered=True,
    )
    return Trace(header=header, steps=trace.steps)


def empirical_onset_pool(
    bundles: Sequence[EpisodeBundle],
    *,
    min_onset: int | None = None,
) -> list[int]:
    """Real L1–L3 onset steps. This is the distribution injected onto L0.

    ``min_onset`` drops degenerate times (default ``_MIN_ONSET_FOR_FPR``):
    ``t_onset < 2`` cannot form a two-sample pre-onset baseline.
    """
    floor = _MIN_ONSET_FOR_FPR if min_onset is None else int(min_onset)
    pool: list[int] = []
    for bundle in bundles:
        header = bundle.trace.header
        if int(header.level) not in {1, 2, 3}:
            continue
        if not header.disturbance_triggered or header.t_onset is None:
            continue
        onset = int(header.t_onset)
        if onset < floor:
            continue
        pool.append(onset)
    return pool


def onset_fits_episode(trace: Trace, t_onset: int) -> bool:
    """True when a virtual onset sits on the episode's task-step clock."""
    if not trace.steps:
        return False
    times = [int(step.t) for step in trace.steps]
    return int(times[0]) <= int(t_onset) < int(times[-1])


# ---------------------------------------------------------------------------
# Analyze helpers
# ---------------------------------------------------------------------------


def analyze_bundle(
    bundle: EpisodeBundle,
    cfg: DiagnosticsConfig,
    *,
    trace: Trace | None = None,
) -> BehaviorDiagnostics:
    used = bundle.trace if trace is None else trace
    outcome = bundle.outcome
    return analyze_trace(
        used,
        cfg,
        success=outcome.success,
        success_step=outcome.success_step,
        first_contact_step=outcome.first_contact_step,
        contact_steps=outcome.contact_steps,
        trace_path=bundle.trace_path.name,
    )


def summarize_diagnostics(diags: Sequence[BehaviorDiagnostics]) -> dict[str, Any]:
    """Four rates (each with its denominator) plus 9-class counts.

    Missing rates stay ``None`` — never 0.0.
    """
    n = len(diags)
    triggered = [d for d in diags if d.disturbance_triggered]
    n_triggered = len(triggered)
    if n_triggered == 0:
        reaction_rate: float | None = None
        stop_rate: float | None = None
        evade_rate: float | None = None
    else:
        inv = 1.0 / float(n_triggered)
        reaction_rate = sum(1 for d in triggered if d.reacted) * inv
        stop_rate = sum(1 for d in triggered if d.stop_preemptive) * inv
        evade_rate = sum(1 for d in triggered if d.evaded) * inv
    applicable = [d for d in diags if d.recovery_applicable]
    n_recovery = len(applicable)
    recovery_rate: float | None
    if n_recovery == 0:
        recovery_rate = None
    else:
        recovery_rate = sum(1 for d in applicable if d.recovered) / float(n_recovery)

    counts = {name: 0 for name in _CLASS_VALUES}
    for diag in diags:
        counts[diag.behavior_class.value] += 1
    row: dict[str, Any] = {
        "n": n,
        "n_triggered": n_triggered,
        "n_recovery": n_recovery,
        "reaction_rate": reaction_rate,
        "stop_rate": stop_rate,
        "evade_rate": evade_rate,
        "recovery_rate": recovery_rate,
    }
    inv_n = (1.0 / float(n)) if n else None
    for name in _CLASS_VALUES:
        row[f"bc_{name}"] = counts[name]
        row[f"bc_{name}_rate"] = None if inv_n is None else counts[name] * inv_n
    return row


# ---------------------------------------------------------------------------
# Baseline health
# ---------------------------------------------------------------------------


def baseline_health_row(
    bundle: EpisodeBundle,
    cfg: DiagnosticsConfig,
    *,
    t_onset: int | None,
    onset_source: str,
) -> dict[str, Any]:
    window = int(cfg.reaction.window_size)
    times, speeds = _kinematics(bundle.trace)
    if t_onset is None:
        return _health_payload(
            bundle,
            t_onset=None,
            onset_source=onset_source,
            window_size=window,
            n_baseline_steps=0,
            window_truncated=None,
            mu_pre=None,
            sigma_raw=None,
            sigma_raw_over_mu=None,
        )
    lo = int(t_onset) - window
    mask = (times >= lo) & (times < int(t_onset)) & np.isfinite(speeds)
    pre = np.asarray(speeds[mask], dtype=np.float64)
    n_pre = int(pre.size)
    truncated = n_pre < window or int(t_onset) < window
    if n_pre < 2:
        return _health_payload(
            bundle,
            t_onset=int(t_onset),
            onset_source=onset_source,
            window_size=window,
            n_baseline_steps=n_pre,
            window_truncated=truncated,
            mu_pre=None,
            sigma_raw=None,
            sigma_raw_over_mu=None,
        )
    mu = float(np.mean(pre))
    sigma_raw = float(np.std(pre, ddof=0))
    cv = None if abs(mu) < 1e-12 else float(sigma_raw / mu)
    return _health_payload(
        bundle,
        t_onset=int(t_onset),
        onset_source=onset_source,
        window_size=window,
        n_baseline_steps=n_pre,
        window_truncated=truncated,
        mu_pre=mu,
        sigma_raw=sigma_raw,
        sigma_raw_over_mu=cv,
    )


def _health_payload(
    bundle: EpisodeBundle,
    *,
    t_onset: int | None,
    onset_source: str,
    window_size: int,
    n_baseline_steps: int,
    window_truncated: bool | None,
    mu_pre: float | None,
    sigma_raw: float | None,
    sigma_raw_over_mu: float | None,
) -> dict[str, Any]:
    header = bundle.trace.header
    return {
        "run_id": bundle.outcome.run_id or header.run_id,
        "episode_index": int(header.episode_index),
        "backend": bundle.outcome.backend or header.backend,
        "level": int(header.level),
        "seed": int(header.seed),
        "task_id": int(bundle.outcome.task_id),
        "n_steps": int(header.n_steps),
        "t_onset": t_onset,
        "onset_source": onset_source,
        "window_size": window_size,
        "n_baseline_steps": n_baseline_steps,
        "window_truncated": window_truncated,
        "mu_pre": mu_pre,
        "sigma_raw": sigma_raw,
        "sigma_raw_over_mu": sigma_raw_over_mu,
    }


def _kinematics(trace: Trace) -> tuple[np.ndarray, np.ndarray]:
    times = np.array([int(step.t) for step in trace.steps], dtype=np.int64)
    if not trace.steps:
        return times, np.zeros(0, dtype=np.float64)
    positions = np.array([step.eef_pos for step in trace.steps], dtype=np.float64)
    return times, compute_speeds(positions)


# ---------------------------------------------------------------------------
# FPR curve + sensitivity
# ---------------------------------------------------------------------------


def compute_fpr_curve(
    l0: Sequence[EpisodeBundle],
    onset_pool: Sequence[int],
    cfg: DiagnosticsConfig,
    *,
    n_virtual_onsets: int,
    seed: int,
    sensitivity_values: Sequence[float],
    sigma_rel_floor_values: Sequence[float],
) -> pd.DataFrame:
    if not onset_pool:
        raise CalibrationError(
            "no L1–L3 onset steps to sample from. L0 false-positive anchoring "
            "needs the empirical onset distribution of triggered episodes."
        )
    rng = np.random.default_rng(int(seed))
    draws_per_ep: list[list[int]] = []
    for bundle in l0:
        raw = rng.choice(np.asarray(onset_pool, dtype=np.int64), size=int(n_virtual_onsets), replace=True)
        draws_per_ep.append([int(v) for v in raw])

    rows: list[dict[str, Any]] = []
    for sensitivity in sensitivity_values:
        for floor in sigma_rel_floor_values:
            trial = set_knob(cfg, "reaction.sensitivity", float(sensitivity))
            trial = set_knob(trial, "reaction.sigma_rel_floor", float(floor))
            n_ok = 0
            n_reacted = 0
            n_skipped = 0
            for bundle, draws in zip(l0, draws_per_ep, strict=True):
                for t_on in draws:
                    if not onset_fits_episode(bundle.trace, t_on):
                        n_skipped += 1
                        continue
                    virtual = inject_virtual_onset(bundle.trace, t_on)
                    # The original header must still be the untriggered L0 one.
                    if bundle.trace.header.t_onset is not None:
                        raise CalibrationError("inject_virtual_onset mutated the original Trace")
                    diag = analyze_bundle(bundle, trial, trace=virtual)
                    n_ok += 1
                    n_reacted += int(bool(diag.reacted))
            fpr = None if n_ok == 0 else n_reacted / float(n_ok)
            rows.append(
                {
                    "sensitivity": float(sensitivity),
                    "sigma_rel_floor": float(floor),
                    "heading_change_deg": float(trial.reaction.heading_change_deg),
                    "n_injections": n_ok,
                    "n_reacted": n_reacted,
                    "n_skipped": n_skipped,
                    "fpr": fpr,
                }
            )
    return pd.DataFrame(rows)


def compute_fpr_heading_curve(
    l0: Sequence[EpisodeBundle],
    onset_pool: Sequence[int],
    cfg: DiagnosticsConfig,
    *,
    n_virtual_onsets: int,
    seed: int,
    heading_values: Sequence[float],
) -> pd.DataFrame:
    """L0 FPR vs ``reaction.heading_change_deg``, other knobs held at YAML."""
    rows: list[dict[str, Any]] = []
    for deg in heading_values:
        trial = set_knob(cfg, "reaction.heading_change_deg", float(deg))
        piece = compute_fpr_curve(
            l0,
            onset_pool,
            trial,
            n_virtual_onsets=n_virtual_onsets,
            seed=seed,
            sensitivity_values=[float(cfg.reaction.sensitivity)],
            sigma_rel_floor_values=[float(cfg.reaction.sigma_rel_floor)],
        )
        for rec in piece.to_dict(orient="records"):
            rec["heading_change_deg"] = float(deg)
            rows.append(rec)
    return pd.DataFrame(rows)


def heading_probe_values(baseline: float) -> list[float]:
    """YAML heading plus a diagnostic ladder up to 180° (direction off)."""
    return sorted({float(baseline), *_HEADING_PROBE_DEG})


def compute_sensitivity_table(
    bundles: Sequence[EpisodeBundle],
    cfg: DiagnosticsConfig,
    *,
    knobs: Sequence[str] = SENSITIVITY_KNOBS,
    factors: Sequence[float] = _DEFAULT_FACTORS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for knob in knobs:
        baseline = float(knob_value(cfg, knob))
        for factor in factors:
            trial = set_knob(cfg, knob, baseline * float(factor))
            diags = [analyze_bundle(bundle, trial) for bundle in bundles]
            row = summarize_diagnostics(diags)
            row.update(
                {
                    "knob": knob,
                    "factor": float(factor),
                    "value": float(knob_value(trial, knob)),
                    "baseline_value": baseline,
                }
            )
            rows.append(row)
    columns = [
        "knob",
        "factor",
        "value",
        "baseline_value",
        "n",
        "n_triggered",
        "n_recovery",
        *_RATE_KEYS,
        *[f"bc_{name}" for name in _CLASS_VALUES],
        *[f"bc_{name}_rate" for name in _CLASS_VALUES],
    ]
    return pd.DataFrame(rows, columns=columns)


def grid_values(baseline: float, scales: Sequence[float] = _DEFAULT_SCALES) -> list[float]:
    return sorted({_scale_knob(baseline, scale) for scale in scales})


def _scale_knob(baseline: float, scale: float) -> float:
    return float(baseline) * float(scale)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def build_recommendation(
    *,
    cfg: DiagnosticsConfig,
    fpr_curve: pd.DataFrame,
    sensitivity: pd.DataFrame,
    baseline_health: pd.DataFrame,
    target_fpr: float,
    n_l0: int,
    n_l1_l3: int,
    n_onset_pool: int,
    seed: int,
    n_virtual_onsets: int,
    n_onset_pool_usable: int | None = None,
    fpr_heading: pd.DataFrame | None = None,
    fpr_curve_wide: pd.DataFrame | None = None,
    fpr_curve_magnitude: pd.DataFrame | None = None,
) -> str:
    picked = pick_fpr_point(fpr_curve, target_fpr=target_fpr)
    baseline_fpr = _fpr_at(fpr_curve, cfg)
    health = _health_summary(baseline_health)
    stability = _stability_notes(sensitivity)
    usable = n_onset_pool if n_onset_pool_usable is None else n_onset_pool_usable
    dropped = max(int(n_onset_pool) - int(usable), 0)
    lines = [
        "# Behavior-diagnostics P5 recommendation",
        "",
        "Thresholds below are **candidates**. Freeze them in `configs/base.yaml`",
        "only after reading `baseline_health.csv` (L1 window truncation is the",
        "known risk), the heading FPR scan, and the ±50% sensitivity table.",
        "",
        "## Setup",
        "",
        f"- L0 episodes (false-positive anchor): {n_l0}",
        f"- L1–L3 episodes (sensitivity / health): {n_l1_l3}",
        f"- Empirical onset pool (raw / usable t≥{_MIN_ONSET_FOR_FPR}): "
        f"{n_onset_pool} / {usable} (dropped {dropped})",
        f"- Virtual onsets per L0 episode: {n_virtual_onsets}",
        f"- RNG seed: {seed}",
        f"- Target FPR: {target_fpr:.3f}",
        "",
        "## Baseline thresholds (from YAML, not Python)",
        "",
        "```yaml",
        _yaml_diagnostics_snippet(cfg).rstrip(),
        "```",
        "",
        "## L0 false-positive anchor",
        "",
    ]
    if baseline_fpr is None:
        lines.append("- FPR at the YAML baseline: n/a (no valid virtual injections).")
    else:
        meet = "meets" if baseline_fpr <= target_fpr else "misses"
        lines.append(
            f"- FPR at the YAML baseline (`sensitivity="
            f"{float(cfg.reaction.sensitivity)}`, `sigma_rel_floor="
            f"{float(cfg.reaction.sigma_rel_floor)}`): "
            f"**{baseline_fpr:.4f}** ({meet} target {target_fpr:.3f})."
        )
    if picked is None:
        lines.append("- Grid pick: n/a (empty FPR curve).")
    else:
        flag = "" if picked["meets_target"] else " **(target not met on this grid)**"
        lines.extend(
            [
                f"- Selected grid point{flag}: `reaction.sensitivity="
                f"{picked['sensitivity']}`, `reaction.sigma_rel_floor="
                f"{picked['sigma_rel_floor']}` → FPR={picked['fpr']:.4f}"
                f" (n_injections={int(picked['n_injections'])}).",
                "- Selection rule: among cells with FPR ≤ target, take the most",
                "  sensitive pair (lowest `sensitivity`, then lowest",
                "  `sigma_rel_floor`). If none meet the target, take the lowest FPR.",
            ]
        )
    lines.extend(_heading_fpr_notes(fpr_heading, cfg, target_fpr=target_fpr))
    lines.extend(_magnitude_fpr_notes(fpr_curve_magnitude, target_fpr=target_fpr))
    if fpr_curve_wide is not None and not fpr_curve_wide.empty:
        wide_min = float(fpr_curve_wide["fpr"].dropna().min()) if not fpr_curve_wide["fpr"].dropna().empty else None
        if wide_min is not None:
            lines.append(
                f"- Wide k×σ grid (YAML heading held): min FPR={wide_min:.4f}."
            )
    lines.extend(["", "## Baseline-window health", ""])
    lines.extend(health)
    lines.extend(["", "## ±50% sensitivity", ""])
    lines.extend(stability)
    lines.extend(
        [
            "",
            "## Do not freeze if",
            "",
            "- L1 `window_truncated` is common (`t_onset` < `reaction.window_size`).",
            "- The FPR grid cannot reach the target without collapsing reaction rate.",
            "- A ±50% nudge of any knob moves a rate by more than 0.10.",
            "",
            "Cross-backend transfer is a separate check (RoboTwin / SimplerEnv).",
        ]
    )
    return "\n".join(lines) + "\n"


def pick_fpr_point(fpr_curve: pd.DataFrame, *, target_fpr: float) -> dict[str, Any] | None:
    if fpr_curve.empty or "fpr" not in fpr_curve.columns:
        return None
    work = fpr_curve.dropna(subset=["fpr"]).copy()
    if work.empty:
        return None
    ok = work[work["fpr"] <= float(target_fpr)]
    if ok.empty:
        work = work.sort_values(["fpr", "sensitivity", "sigma_rel_floor"], kind="mergesort")
        row = work.iloc[0]
        meets = False
    else:
        ok = ok.sort_values(["sensitivity", "sigma_rel_floor"], kind="mergesort")
        row = ok.iloc[0]
        meets = True
    return {
        "sensitivity": float(row["sensitivity"]),
        "sigma_rel_floor": float(row["sigma_rel_floor"]),
        "fpr": float(row["fpr"]),
        "n_injections": int(row["n_injections"]),
        "meets_target": meets,
    }


def _fpr_at(fpr_curve: pd.DataFrame, cfg: DiagnosticsConfig) -> float | None:
    if fpr_curve.empty:
        return None
    k = float(cfg.reaction.sensitivity)
    floor = float(cfg.reaction.sigma_rel_floor)
    hit = fpr_curve[
        np.isclose(fpr_curve["sensitivity"].astype(float), k)
        & np.isclose(fpr_curve["sigma_rel_floor"].astype(float), floor)
    ]
    if hit.empty or pd.isna(hit.iloc[0]["fpr"]):
        return None
    return float(hit.iloc[0]["fpr"])


def _heading_fpr_notes(
    heading: pd.DataFrame | None,
    cfg: DiagnosticsConfig,
    *,
    target_fpr: float,
) -> list[str]:
    if heading is None or heading.empty or "fpr" not in heading.columns:
        return []
    work = heading.dropna(subset=["fpr"]).sort_values("heading_change_deg")
    if work.empty:
        return []
    lines = [
        "",
        "Reaction is magnitude **or** heading. A ±50% k×σ grid can be flat when",
        "task-motion turns already clear `heading_change_deg`. The heading scan",
        "holds k and σ at the YAML baseline.",
        "",
    ]
    for rec in work.to_dict(orient="records"):
        lines.append(
            f"- heading={float(rec['heading_change_deg']):.1f}° → "
            f"FPR={float(rec['fpr']):.4f} (n={int(rec['n_injections'])})"
        )
    meet = work[work["fpr"] <= float(target_fpr)]
    yaml_deg = float(cfg.reaction.heading_change_deg)
    if meet.empty:
        lines.append(
            f"- No heading in this ladder meets FPR ≤ {target_fpr:.3f} at "
            f"YAML k={float(cfg.reaction.sensitivity)} / "
            f"σ_rel={float(cfg.reaction.sigma_rel_floor)}. "
            f"YAML heading is {yaml_deg:.1f}°."
        )
    else:
        first = meet.sort_values("heading_change_deg").iloc[0]
        lines.append(
            f"- Lowest heading that meets the target: "
            f"{float(first['heading_change_deg']):.1f}° "
            f"(FPR={float(first['fpr']):.4f})."
        )
    return lines


def _magnitude_fpr_notes(
    magnitude: pd.DataFrame | None,
    *,
    target_fpr: float,
) -> list[str]:
    if magnitude is None or magnitude.empty or "fpr" not in magnitude.columns:
        return []
    work = magnitude.dropna(subset=["fpr"])
    if work.empty:
        return []
    picked = pick_fpr_point(work, target_fpr=target_fpr)
    lines = [
        "",
        "Magnitude-only FPR (heading_change_deg=180, direction gate off):",
    ]
    if picked is None:
        lines.append("- n/a")
        return lines
    flag = "" if picked["meets_target"] else " **(target not met)**"
    lines.append(
        f"- pick{flag}: k={picked['sensitivity']}, "
        f"σ_rel={picked['sigma_rel_floor']} → FPR={picked['fpr']:.4f}."
    )
    return lines


def _health_summary(health: pd.DataFrame) -> list[str]:
    if health.empty:
        return ["- No baseline-health rows."]
    real = health[health["onset_source"] == "real"]
    lines: list[str] = []
    if real.empty:
        lines.append("- No real (L1–L3) onset rows.")
    else:
        trunc = real["window_truncated"].eq(True)
        n = int(len(real))
        n_trunc = int(trunc.sum())
        l1 = real[real["level"] == 1]
        l1_trunc = int(l1["window_truncated"].eq(True).sum()) if not l1.empty else 0
        lines.append(
            f"- Real-onset rows: {n}; truncated baseline window: {n_trunc}/{n}"
            f" ({n_trunc / n:.1%})."
        )
        if not l1.empty:
            lines.append(
                f"- L1 only: {l1_trunc}/{len(l1)} truncated. This is the known "
                "risk (`t_onset` can be 5–6 while `window_size=10`)."
            )
        mu = real["mu_pre"].dropna()
        cv = real["sigma_raw_over_mu"].dropna()
        if not mu.empty:
            lines.append(
                f"- μ_pre median={float(mu.median()):.6f}; "
                f"σ_raw/μ_pre median="
                f"{'n/a' if cv.empty else f'{float(cv.median()):.4f}'}."
            )
    virt = health[health["onset_source"] == "virtual"]
    if not virt.empty:
        lines.append(f"- Virtual-onset (L0 FPR) health rows: {len(virt)}.")
    return lines


def _stability_notes(sensitivity: pd.DataFrame) -> list[str]:
    if sensitivity.empty:
        return ["- Sensitivity table is empty."]
    lines: list[str] = []
    unstable = 0
    for knob, grp in sensitivity.groupby("knob", sort=False):
        deltas: list[str] = []
        for rate in _RATE_KEYS:
            series = grp[rate].dropna()
            if series.empty:
                deltas.append(f"{rate}=n/a")
                continue
            span = float(series.max() - series.min())
            mark = " UNSTABLE" if span > 0.10 else ""
            if span > 0.10:
                unstable += 1
            deltas.append(f"{rate} Δ={span:.3f}{mark}")
        lines.append(f"- `{knob}`: " + "; ".join(deltas))
    if unstable:
        lines.append(
            f"- {unstable} rate span(s) exceeded 0.10 across ±50%. "
            "Do not treat that knob as frozen without a comment in the paper."
        )
    else:
        lines.append("- No rate moved by more than 0.10 across the ±50% scan.")
    return lines


def _yaml_diagnostics_snippet(cfg: DiagnosticsConfig) -> str:
    payload = {
        "diagnostics": {
            "reaction": {
                "window_size": int(cfg.reaction.window_size),
                "sensitivity": float(cfg.reaction.sensitivity),
                "min_persist_steps": int(cfg.reaction.min_persist_steps),
                "heading_change_deg": float(cfg.reaction.heading_change_deg),
                "sigma_rel_floor": float(cfg.reaction.sigma_rel_floor),
            },
            "stop": {
                "speed_frac": float(cfg.stop.speed_frac),
                "speed_floor_m_per_step": float(cfg.stop.speed_floor_m_per_step),
                "min_steps": int(cfg.stop.min_steps),
            },
            "avoid": {
                "heading_change_deg": float(cfg.avoid.heading_change_deg),
                "window_steps": int(cfg.avoid.window_steps),
                "min_speed_frac": float(cfg.avoid.min_speed_frac),
            },
            "recovery": {
                "speed_frac": float(cfg.recovery.speed_frac),
                "min_steps": int(cfg.recovery.min_steps),
                "min_post_hazard_steps": int(cfg.recovery.min_post_hazard_steps),
            },
        }
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_fpr_curve(fpr_curve: pd.DataFrame, out_path: Path, *, target_fpr: float) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fpr_curve.empty or fpr_curve["fpr"].dropna().empty:
        _unavailable_figure(out_path, "FPR curve (n/a)")
        return
    pivot = fpr_curve.pivot_table(
        index="sigma_rel_floor",
        columns="sensitivity",
        values="fpr",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    mesh = ax.imshow(
        pivot.to_numpy(dtype=float),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(float(target_fpr) * 2.0, float(np.nanmax(pivot.to_numpy()))),
    )
    ax.set_xticks(range(len(pivot.columns)), [f"{float(c):.3g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{float(r):.3g}" for r in pivot.index])
    ax.set_xlabel("reaction.sensitivity")
    ax.set_ylabel("reaction.sigma_rel_floor")
    ax.set_title(f"L0 false-positive rate (target ≤ {target_fpr:.2f})")
    fig.colorbar(mesh, ax=ax, label="FPR")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fpr_heading(heading: pd.DataFrame, out_path: Path, *, target_fpr: float) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if heading.empty or heading["fpr"].dropna().empty:
        _unavailable_figure(out_path, "FPR vs heading (n/a)")
        return
    work = heading.dropna(subset=["fpr"]).sort_values("heading_change_deg")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(work["heading_change_deg"], work["fpr"], marker="o")
    ax.axhline(float(target_fpr), color="0.4", linestyle="--", label=f"target {target_fpr:.2f}")
    ax.set_xlabel("reaction.heading_change_deg")
    ax.set_ylabel("L0 FPR")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("L0 false-positive rate vs heading gate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if sensitivity.empty:
        _unavailable_figure(out_path, "sensitivity (n/a)")
        return
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), sharex=True)
    for ax, rate in zip(axes.ravel(), _RATE_KEYS, strict=True):
        for knob, grp in sensitivity.groupby("knob", sort=False):
            ordered = grp.sort_values("factor")
            y = ordered[rate].astype(float)
            ax.plot(ordered["factor"], y, marker="o", label=str(knob), linewidth=1.2)
        ax.set_title(rate)
        ax.set_ylabel(rate)
        ax.set_ylim(-0.05, 1.05)
        ax.axvline(1.0, color="0.6", linewidth=0.8, linestyle="--")
    axes[1][0].set_xlabel("factor vs YAML baseline")
    axes[1][1].set_xlabel("factor vs YAML baseline")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("±50% threshold sensitivity (L1–L3)")
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _unavailable_figure(out_path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_calibration(
    *,
    results: Path | str,
    out: Path | str,
    target_fpr: float = 0.05,
    n_virtual_onsets: int = 5,
    seed: int = 7,
    base_yaml: Path | str | None = None,
    cfg: DiagnosticsConfig | None = None,
    fpr_scales: Sequence[float] = _DEFAULT_SCALES,
    factors: Sequence[float] = _DEFAULT_FACTORS,
) -> CalibrationResult:
    """Load traces, compute FPR / sensitivity / health, write artifacts.

    The results tree is fingerprinted before and after. The function refuses
    to finish if any protected file changed.
    """
    results_root = Path(results)
    out_dir = Path(out)
    before = fingerprint_results(results_root)
    diagnostics = cfg if cfg is not None else load_baseline_diagnostics(base_yaml)
    bundles = discover_bundles(results_root)
    if not bundles:
        raise CalibrationError(f"no traces with episodes.jsonl under {results_root}")

    l0 = [b for b in bundles if int(b.trace.header.level) == 0]
    l1_l3 = [b for b in bundles if int(b.trace.header.level) in {1, 2, 3}]
    onset_pool_raw = empirical_onset_pool(bundles, min_onset=0)
    onset_pool = empirical_onset_pool(bundles)
    if not l0:
        raise CalibrationError("no L0 traces; the false-positive anchor requires L0")
    if not onset_pool:
        raise CalibrationError(
            "no triggered L1–L3 onsets with t_onset>="
            f"{_MIN_ONSET_FOR_FPR} to sample virtual t_onset from"
        )

    k_grid = grid_values(float(diagnostics.reaction.sensitivity), fpr_scales)
    floor_grid = grid_values(float(diagnostics.reaction.sigma_rel_floor), fpr_scales)
    fpr_curve = compute_fpr_curve(
        l0,
        onset_pool,
        diagnostics,
        n_virtual_onsets=int(n_virtual_onsets),
        seed=int(seed),
        sensitivity_values=k_grid,
        sigma_rel_floor_values=floor_grid,
    )
    fpr_heading = compute_fpr_heading_curve(
        l0,
        onset_pool,
        diagnostics,
        n_virtual_onsets=int(n_virtual_onsets),
        seed=int(seed),
        heading_values=heading_probe_values(float(diagnostics.reaction.heading_change_deg)),
    )
    mag_cfg = set_knob(diagnostics, "reaction.heading_change_deg", 180.0)
    fpr_curve_magnitude = compute_fpr_curve(
        l0,
        onset_pool,
        mag_cfg,
        n_virtual_onsets=int(n_virtual_onsets),
        seed=int(seed),
        sensitivity_values=list(_WIDE_SENSITIVITY),
        sigma_rel_floor_values=list(_WIDE_SIGMA_REL),
    )
    fpr_min = (
        float(fpr_curve["fpr"].dropna().min())
        if not fpr_curve.empty and not fpr_curve["fpr"].dropna().empty
        else None
    )
    fpr_curve_wide: pd.DataFrame | None = None
    if fpr_min is None or fpr_min > float(target_fpr):
        fpr_curve_wide = compute_fpr_curve(
            l0,
            onset_pool,
            diagnostics,
            n_virtual_onsets=int(n_virtual_onsets),
            seed=int(seed),
            sensitivity_values=list(_WIDE_SENSITIVITY),
            sigma_rel_floor_values=list(_WIDE_SIGMA_REL),
        )
    sensitivity = compute_sensitivity_table(l1_l3, diagnostics, factors=factors)

    health_rows: list[dict[str, Any]] = []
    for bundle in l1_l3:
        health_rows.append(
            baseline_health_row(
                bundle,
                diagnostics,
                t_onset=bundle.trace.header.t_onset,
                onset_source="real",
            )
        )
    rng = np.random.default_rng(int(seed))
    for bundle in l0:
        draws = rng.choice(
            np.asarray(onset_pool, dtype=np.int64),
            size=int(n_virtual_onsets),
            replace=True,
        )
        health_rows.append(
            baseline_health_row(
                bundle,
                diagnostics,
                t_onset=None,
                onset_source="l0_raw",
            )
        )
        for t_on in draws:
            if not onset_fits_episode(bundle.trace, int(t_on)):
                continue
            health_rows.append(
                baseline_health_row(
                    bundle,
                    diagnostics,
                    t_onset=int(t_on),
                    onset_source="virtual",
                )
            )
    baseline_health = pd.DataFrame(health_rows)

    recommendation = build_recommendation(
        cfg=diagnostics,
        fpr_curve=fpr_curve,
        sensitivity=sensitivity,
        baseline_health=baseline_health,
        target_fpr=float(target_fpr),
        n_l0=len(l0),
        n_l1_l3=len(l1_l3),
        n_onset_pool=len(onset_pool_raw),
        n_onset_pool_usable=len(onset_pool),
        seed=int(seed),
        n_virtual_onsets=int(n_virtual_onsets),
        fpr_heading=fpr_heading,
        fpr_curve_wide=fpr_curve_wide,
        fpr_curve_magnitude=fpr_curve_magnitude,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "fpr_curve.csv", fpr_curve)
    _write_csv(out_dir / "fpr_heading.csv", fpr_heading)
    _write_csv(out_dir / "fpr_curve_magnitude.csv", fpr_curve_magnitude)
    if fpr_curve_wide is not None:
        _write_csv(out_dir / "fpr_curve_wide.csv", fpr_curve_wide)
    _write_csv(out_dir / "sensitivity.csv", sensitivity)
    _write_csv(out_dir / "baseline_health.csv", baseline_health)
    (out_dir / "recommendation.md").write_text(recommendation, encoding="utf-8")
    plot_fpr_curve(fpr_curve, out_dir / "plots" / "fpr_curve.png", target_fpr=float(target_fpr))
    plot_sensitivity(sensitivity, out_dir / "plots" / "sensitivity.png")
    plot_fpr_heading(fpr_heading, out_dir / "plots" / "fpr_heading.png", target_fpr=float(target_fpr))

    after = fingerprint_results(results_root)
    changed = sorted(key for key, digest in before.items() if after.get(key) != digest)
    if changed:
        raise CalibrationError(
            "calibration changed files in the results tree; this is a bug. "
            f"Changed: {changed[:8]}{'…' if len(changed) > 8 else ''}. "
            "Trace files that existed at start must stay byte-identical. "
            "New files from a concurrent sweep are ignored."
        )
    return CalibrationResult(
        fpr_curve=fpr_curve,
        sensitivity=sensitivity,
        baseline_health=baseline_health,
        recommendation=recommendation,
        fingerprints_before=before,
        fingerprints_after=after,
        n_l0=len(l0),
        n_l1_l3=len(l1_l3),
        n_onset_pool=len(onset_pool_raw),
        n_onset_pool_usable=len(onset_pool),
        fpr_heading=fpr_heading,
        fpr_curve_wide=fpr_curve_wide,
        fpr_curve_magnitude=fpr_curve_magnitude,
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


__all__ = [
    "SENSITIVITY_KNOBS",
    "CalibrationError",
    "CalibrationResult",
    "EpisodeBundle",
    "EpisodeOutcome",
    "analyze_bundle",
    "baseline_health_row",
    "build_recommendation",
    "compute_fpr_curve",
    "compute_fpr_heading_curve",
    "compute_sensitivity_table",
    "discover_bundles",
    "empirical_onset_pool",
    "fingerprint_results",
    "grid_values",
    "inject_virtual_onset",
    "knob_value",
    "load_baseline_diagnostics",
    "onset_fits_episode",
    "pick_fpr_point",
    "run_calibration",
    "set_knob",
    "summarize_diagnostics",
]
