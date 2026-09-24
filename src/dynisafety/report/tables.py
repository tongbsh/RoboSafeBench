"""Paper tables and display-name rendering (sole home for SR / FPeakMean labels)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from dynisafety.core.metrics import wilson_interval
from dynisafety.core.policy_arms import is_safety_layer_model, null_parent_model

# Token for missing / non-reconstructible fields (Prompt 21). Never coerce to 0.
UNAVAILABLE = "unavailable"

# Display names ONLY here (not in core/)
DISPLAY_NAMES: dict[str, str] = {
    "sr": "SR",
    "cr": "CR",
    "contact_steps_per_ep": "ColSteps/Ep",
    "topple_rate": "Topple",
    "f_peak_mean_all": "FPeakMean(N)",
    "f_peak_median": "FPeakMedian(N)",
    "f_peak_mean_contact": "FPeakMeanContact(N)",
    "f_peak_p95": "FPeakP95(N)",
    "f_peak_max": "FPeakMax(N)",
    "sasr": "SASR",
    "ssr": "SSR",
    "ss": "SS",
    "us": "US",
    "cff": "CFF",
    "cif": "CIF",
    "us_rate": "US_rate",
    "cff_rate": "CFF_rate",
    "ssr_ci_lo": "SSR_lo",
    "ssr_ci_hi": "SSR_hi",
    "cif_rate": "CIF_rate",
    "n_episodes": "N",
    "n_valid_episodes": "N_valid",
    "setup_invalid_episodes": "SetupInvalidEps",
    "n_seeds": "N_seeds",
    "force_anomaly_episodes": "ForceAnomalyEps",
    "sr_std": "SR_std",
    "cr_std": "CR_std",
    "topple_rate_std": "Topple_std",
    "sasr_std": "SASR_std",
    "oracle_sr": "Oracle_SR",
    "oracle_ssr": "Oracle_SSR",
    "oracle_cr": "Oracle_CR",
    "base_sr": "Base_SR",
    "base_ssr": "Base_SSR",
    "base_cr": "Base_CR",
    "n_triggered": "N_trig",
    "reaction_rate": "ReactionRate",
    "reaction_rate_ci_lo": "ReactionRate_lo",
    "reaction_rate_ci_hi": "ReactionRate_hi",
    "reaction_rate_kinematic": "ReactionRateKinematic",
    "n_reaction_latency": "N_lat",
    "reaction_latency_p50": "ReactLat_p50",
    "reaction_latency_chunks_p50": "ReactLatChunks_p50",
    "stop_rate": "StopRate",
    "evade_rate": "EvadeRate",
    "n_recovery": "N_recov",
    "recovery_rate": "RecoveryRate",
    "recovery_rate_ci_lo": "RecoveryRate_lo",
    "recovery_rate_ci_hi": "RecoveryRate_hi",
    "n_min_clearance": "N_clear",
    "min_clearance_p50": "MinClear_p50",
    "hazard_persists": "HazardPersists",
    "bc_collide_unreacted": "Collide (no reaction)",
    "bc_collide_reacted": "Collide (late)",
    "bc_evade_success": "Evade",
    "bc_evade_fail": "Evade→fail",
    "bc_stop_recover": "Stop→recover",
    "bc_stop_no_recover": "Stop→no recover",
    "bc_partial_response": "Partial",
    "bc_no_response": "No response",
    "bc_not_triggered": "Not triggered",
    "bc_baseline_unusable": "Baseline unusable",
    "bc_collide_unreacted_rate": "Collide (no reaction) rate",
    "bc_collide_reacted_rate": "Collide (late) rate",
    "bc_evade_success_rate": "Evade rate",
    "bc_evade_fail_rate": "Evade→fail rate",
    "bc_stop_recover_rate": "Stop→recover rate",
    "bc_stop_no_recover_rate": "Stop→no recover rate",
    "bc_partial_response_rate": "Partial rate",
    "bc_no_response_rate": "No response rate",
    "bc_not_triggered_rate": "Not triggered rate",
    "bc_baseline_unusable_rate": "Baseline unusable rate",
}

_JOINT_COUNT_IDS = ("ss", "us", "cff", "cif")

# Known capability facts for footnotes (not hyperparameter defaults)
BACKEND_FORCE_SOURCE: dict[str, str] = {
    "libero": "mujoco_mj_contactForce",
    "simplerenv": "sapien_impulse",
    "robotwin": "sapien_impulse",
    "robocasa": "mujoco_mj_contactForce",
}

# Force in the main table is median + p95, not the mean: peak_force_n is
# heavy-tailed because a penetration-recovery step turns one impulse into a
# kilonewton "force", and a single such episode can carry a whole cell's mean.
# FPeakMean / FPeakMax stay available in table_ablation and runs.csv.
_MAIN_METRICS = (
    "sr",
    "cr",
    "topple_rate",
    "f_peak_median",
    "f_peak_p95",
    "sasr",
)
# Rates that carry seed-to-seed sample std on table_main. Force stays a
# pooled quantile (heavy-tailed peak_force_n); a std of per-seed medians
# would not be the pooled median the footnote describes.
_MAIN_SEED_STD_METRICS = ("sr", "cr", "topple_rate", "sasr")
_MAIN_PM_PAIRS = (
    ("SR", "SR_std"),
    ("CR", "CR_std"),
    ("Topple", "Topple_std"),
    ("SASR", "SASR_std"),
)
_SEED_STD_FOOTNOTE = (
    "SR / CR / Topple / SASR: the point estimate is the episode-pooled rate "
    "(denominator N_valid). ± is the sample standard deviation (ddof=1) across "
    "seeds after collapsing tasks within each seed — not a Wilson interval and "
    "not the episode-level Bernoulli SE. Fewer than two seeds: the point "
    "estimate only. Force columns stay pooled quantiles and have no ±."
)

_PER_TASK_METRICS = (
    "sr",
    "cr",
    "contact_steps_per_ep",
    "topple_rate",
    "f_peak_mean_all",
    "f_peak_mean_contact",
    "f_peak_max",
    "sasr",
    "force_anomaly_episodes",
    "setup_invalid_episodes",
    "n_valid_episodes",
    "n_episodes",
)

_ABLATION_METRICS = (
    "sr",
    "cr",
    "f_peak_median",
    "f_peak_p95",
    "f_peak_mean_all",
    "f_peak_mean_contact",
    "f_peak_max",
    "sasr",
)

_JOINT_OUTCOME_METRICS = (
    "ss",
    "us",
    "cff",
    "cif",
    "ssr",
    "us_rate",
    "cff_rate",
    "cif_rate",
)

# §7.1 paper table. stop_rate is stop_preemptive, not stopped (crash-stops excluded).
_BEHAVIOR_CLASS_IDS = (
    "collide_unreacted",
    "collide_reacted",
    "evade_success",
    "evade_fail",
    "stop_recover",
    "stop_no_recover",
    "partial_response",
    "no_response",
    "not_triggered",
    "baseline_unusable",
)
_BEHAVIOR_COUNT_IDS = tuple(f"bc_{name}" for name in _BEHAVIOR_CLASS_IDS)
_BEHAVIOR_RATE_IDS = tuple(f"bc_{name}_rate" for name in _BEHAVIOR_CLASS_IDS)
_BEHAVIOR_PROCESS_IDS = (
    "n_triggered",
    "reaction_rate",
    "reaction_rate_ci_lo",
    "reaction_rate_ci_hi",
    "reaction_rate_kinematic",
    "n_reaction_latency",
    "reaction_latency_p50",
    "reaction_latency_chunks_p50",
    "stop_rate",
    "evade_rate",
    "recovery_rate",
    "recovery_rate_ci_lo",
    "recovery_rate_ci_hi",
    "n_recovery",
    "n_min_clearance",
    "min_clearance_p50",
    "hazard_persists",
)
_BEHAVIOR_TABLE_IDS = _BEHAVIOR_PROCESS_IDS + _BEHAVIOR_COUNT_IDS + _BEHAVIOR_RATE_IDS
# Exclude from episode-weighted collapse used by table_main / joint tables:
# rates here have their own denominators; hazard_persists is not numeric.
_BEHAVIOR_SKIP_FROM_MEAN = frozenset(_BEHAVIOR_TABLE_IDS)


def format_cell(
    metric_id: str,
    value: Any,
    *,
    available: bool,
    missing_token: str = "n/a",
) -> str:
    """Return missing_token if not available; else formatted number. Never coerce missing → 0.0."""
    del metric_id  # reserved for metric-specific formatting later
    if not available or value is None:
        return missing_token
    if isinstance(value, str) and value.strip().lower() in {"unavailable", "n/a", "na"}:
        return missing_token if value.strip().lower() != "unavailable" else UNAVAILABLE
    if isinstance(value, (bool, str)):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        if pd.isna(value):
            return missing_token
    except (TypeError, ValueError):
        pass
    return f"{float(value):.4g}"


def build_footnotes(
    *,
    backends_present: Sequence[str],
    has_mixed_force_sources: bool,
) -> list[str]:
    """Footnotes: f_peak denominator, force incomparability, n/a meaning."""
    notes = [
        (
            "FPeakMedian(N) / FPeakP95(N) are quantiles of peak_force_n pooled over "
            "the row's valid episodes; episodes without contact contribute 0 N. "
            "Quantiles rather than the mean because peak_force_n is heavy-tailed: a "
            "penetration-recovery step converts one impulse into a "
            "kilonewton-scale force, so a single episode can carry a cell mean. "
            "FPeakMean / FPeakMax are in table_ablation and runs.csv."
        ),
        (
            "FPeakMeanContact(N) is f_peak_mean_contact (contact episodes only); "
            "reported in runs.csv / per-task tables, not the paper main force column."
        ),
        (
            "N is episodes run; N_valid excludes setup-invalid trials (the requested "
            "disturbance never assembled) and is the denominator of SR / CR / SASR / "
            "Topple on the row. SetupInvalidEps = N − N_valid."
        ),
        "n/a means the metric is undefined or the backend lacks the required capability "
        "(never a silent 0.0).",
    ]
    sources = []
    for backend in backends_present:
        src = BACKEND_FORCE_SOURCE.get(backend)
        if src is not None:
            sources.append(f"{backend}={src}")
    if sources:
        notes.append("Force sources: " + "; ".join(sources) + ".")
    if has_mixed_force_sources:
        notes.append(
            "Force magnitudes are not comparable across simulators with different "
            "force_source values; do not rank models across backends by force."
        )
    return notes


def build_joint_footnotes() -> list[str]:
    """Footnotes for joint-outcome tables (SSR ≠ SASR; CFF/CIF naming locks)."""
    return [
        "SSR is the SS quadrant rate (ss/N_valid); SASR uses severity penalty φ — do not conflate or rename.",
        "CFF = contact-free failure (not “Safe Stop”); CIF = contact-involved failure "
        "(not “Collision-induced Failure”).",
        f"{UNAVAILABLE} means the field is missing from the input and was not reconstructed "
        "(never a silent 0).",
        "No subjective unified safety total score is defined or reported here.",
        "N is episodes run; N_valid excludes setup-invalid trials (disturbance never "
        "assembled) and is the denominator of every rate on the row.",
        "SSR_lo / SSR_hi are the 95% Wilson interval on ss/N_valid.",
        "Invariants: ss+us+cff+cif=N_valid; ssr≤sr; us_rate≤cr; sr=(ss+us)/N_valid; "
        "cr=(us+cif)/N_valid.",
        "Safety-layer arms ({base}_oracle_*) are a filter on top of a policy, not a "
        "policy; Base_* columns exclude them and Oracle_* columns pair each arm with "
        "its base by name.",
    ]


def _rename_metrics(df: pd.DataFrame, metric_ids: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    rename = {m: DISPLAY_NAMES.get(m, m) for m in metric_ids if m in out.columns}
    return out.rename(columns=rename)


def _aggregate_metric_mean(runs_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Episode-weighted mean of run-level rates when collapsing seeds/tasks.

    Count columns ``ss/us/cff/cif`` (and ``force_anomaly_episodes``) are **summed**.
    Rate columns are episode-weighted means; when all four quadrant counts are present
    they are recomputed from summed counts for exact consistency.
    """
    if runs_df.empty:
        return runs_df.copy()
    work = runs_df.copy()
    if "n_episodes" not in work.columns:
        work["n_episodes"] = 1
    metric_cols = [
        c
        for c in work.columns
        if c
        not in set(group_cols)
        | {
            "n_episodes",
            "n_valid_episodes",
            "setup_invalid_episodes",
            "force_anomaly_episodes",
            "injury_dist",
            "sr_ci_lo",
            "sr_ci_hi",
            "cr_ci_lo",
            "cr_ci_hi",
            "backend",
            "suite",
            "task_name",
            "checkpoint",
            "config_hash",
            "run_id",
        }
    ]
    # Keep only numeric-ish metric ids we know
    known = set(DISPLAY_NAMES) | {
        "sr",
        "cr",
        "contact_steps_per_ep",
        "topple_rate",
        "f_peak_mean_all",
        "f_peak_mean_contact",
        "f_peak_median",
        "f_peak_p95",
        "f_peak_max",
        "sasr",
        "ss",
        "us",
        "cff",
        "cif",
        "ssr",
        "us_rate",
        "cff_rate",
        "cif_rate",
    }
    metric_cols = [c for c in metric_cols if c in known and c not in _BEHAVIOR_SKIP_FROM_MEAN]
    count_ids = set(_JOINT_COUNT_IDS)

    rows: list[dict[str, Any]] = []
    for key, grp in work.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {group_cols[i]: key[i] for i in range(len(group_cols))}
        written = grp["n_episodes"].astype(float)
        wsum = float(written.sum())
        row["n_episodes"] = int(wsum) if wsum == int(wsum) else wsum
        if "n_valid_episodes" in grp.columns:
            row["n_valid_episodes"] = int(grp["n_valid_episodes"].fillna(0).sum())
        if "setup_invalid_episodes" in grp.columns:
            row["setup_invalid_episodes"] = int(grp["setup_invalid_episodes"].fillna(0).sum())
        if "force_anomaly_episodes" in grp.columns:
            row["force_anomaly_episodes"] = int(grp["force_anomaly_episodes"].fillna(0).sum())
        weights = (
            grp["n_valid_episodes"].astype(float)
            if "n_valid_episodes" in grp.columns
            else written
        )
        for col in metric_cols:
            series = grp[col]
            mask = series.notna()
            if not bool(mask.any()):
                row[col] = None
                continue
            if col in count_ids:
                row[col] = int(series[mask].astype(float).sum())
                continue
            w = weights[mask]
            v = series[mask].astype(float)
            row[col] = float((v * w).sum() / w.sum()) if float(w.sum()) > 0 else None
        # Recompute joint rates from summed counts when all four are present
        denom = float(row["n_valid_episodes"]) if "n_valid_episodes" in row else wsum
        if all(c in row and row[c] is not None for c in _JOINT_COUNT_IDS) and denom > 0:
            ss_i = int(row["ss"])
            us_i = int(row["us"])
            cff_i = int(row["cff"])
            cif_i = int(row["cif"])
            inv = 1.0 / float(denom)
            row["ssr"] = ss_i * inv
            row["us_rate"] = us_i * inv
            row["cff_rate"] = cff_i * inv
            row["cif_rate"] = cif_i * inv
        rows.append(row)
    return pd.DataFrame(rows)


# Columns that name the denominator of every rate on the row. ``n_episodes``
# counts what was run; ``n_valid_episodes`` is what SR / CR / SSR / SASR were
# divided by (setup-invalid trials excluded). A paper reader who sees only N
# takes it for the denominator, which it is not once setup_invalid > 0.
_DENOMINATOR_IDS = ("n_episodes", "n_valid_episodes", "setup_invalid_episodes")

# Per-run quantiles / maxima that a weighted mean cannot collapse: the mean
# of per-run medians is not the pooled median, and averaging per-run maxima
# is not a maximum. When the episode frame is available these are recomputed
# from episode-level peak_force_n over the same (valid) population.
_FORCE_POOLED_IDS = (
    "f_peak_mean_all",
    "f_peak_mean_contact",
    "f_peak_median",
    "f_peak_p95",
    "f_peak_max",
)


def pooled_force_stats(
    episodes_df: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Episode-pooled force statistics per group, on valid episodes only.

    Same definitions as ``core.metrics``: all-episode denominators for mean /
    median / p95 / max (no-contact episodes contribute 0 N), contact-episode
    denominator for ``f_peak_mean_contact`` (``None`` when no contact).
    """
    cols = [c for c in group_cols if c in episodes_df.columns]
    if not cols or "peak_force_n" not in episodes_df.columns or episodes_df.empty:
        return pd.DataFrame(columns=[*group_cols, *_FORCE_POOLED_IDS])
    work = episodes_df
    if "setup_invalid" in work.columns:
        work = work.loc[~work["setup_invalid"].fillna(False).astype(bool)]
    rows: list[dict[str, Any]] = []
    for key, grp in work.groupby(cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        peaks = pd.to_numeric(grp["peak_force_n"], errors="coerce").dropna().to_numpy(dtype=float)
        row: dict[str, Any] = dict(zip(cols, key, strict=True))
        if peaks.size == 0:
            row.update({m: None for m in _FORCE_POOLED_IDS})
        else:
            contact = (
                pd.to_numeric(grp["contact_steps"], errors="coerce").fillna(0).to_numpy() > 0
                if "contact_steps" in grp.columns
                else None
            )
            contact_peaks = (
                pd.to_numeric(grp["peak_force_n"], errors="coerce").to_numpy(dtype=float)[contact]
                if contact is not None
                else None
            )
            row.update(
                {
                    "f_peak_mean_all": float(peaks.mean()),
                    "f_peak_mean_contact": (
                        float(contact_peaks.mean())
                        if contact_peaks is not None and contact_peaks.size
                        else None
                    ),
                    "f_peak_median": float(pd.Series(peaks).median()),
                    "f_peak_p95": float(pd.Series(peaks).quantile(0.95)),
                    "f_peak_max": float(peaks.max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=[*cols, *_FORCE_POOLED_IDS])


def _override_force_from_episodes(
    grouped: pd.DataFrame,
    episodes_df: pd.DataFrame | None,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Replace weighted-mean force columns with episode-pooled statistics."""
    if episodes_df is None or episodes_df.empty or grouped.empty:
        return grouped
    cols = [c for c in group_cols if c in grouped.columns and c in episodes_df.columns]
    if not cols:
        return grouped
    pooled = pooled_force_stats(episodes_df, cols)
    if pooled.empty:
        return grouped
    out = grouped.merge(pooled, on=cols, how="left", suffixes=("", "_pooled"))
    for metric in _FORCE_POOLED_IDS:
        pooled_col = f"{metric}_pooled"
        if pooled_col in out.columns:
            # Both frames had the column: the pooled value replaces the mean.
            out[metric] = out[pooled_col]
            out = out.drop(columns=[pooled_col])
        # Otherwise the merge already brought the pooled column in under its
        # own name (the runs frame lacked it), which is the value we want.
    return out


def _seed_metric_std(
    runs_df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_ids: Sequence[str],
    *,
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Sample std (ddof=1) of per-seed episode-weighted rates.

    Each seed is collapsed over the remaining dimensions first, so this is
    seed-to-seed variation, not task-to-task scatter or episode Bernoulli SE.
    The main-table point estimate stays the pooled mean; this frame only
    supplies ``n_seeds`` and ``{metric}_std``.
    """
    cols = [c for c in group_cols if c in runs_df.columns]
    std_ids = [f"{m}_std" for m in metric_ids]
    empty = pd.DataFrame(columns=[*group_cols, "n_seeds", *std_ids])
    if runs_df.empty or seed_col not in runs_df.columns or not cols:
        return empty
    seed_groups = _aggregate_metric_mean(runs_df, [*cols, seed_col])
    if seed_groups.empty:
        return empty
    rows: list[dict[str, Any]] = []
    for key, grp in seed_groups.groupby(cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, Any] = {cols[i]: key[i] for i in range(len(cols))}
        row["n_seeds"] = int(grp[seed_col].nunique(dropna=True))
        for mid in metric_ids:
            std_id = f"{mid}_std"
            if mid not in grp.columns:
                row[std_id] = None
                continue
            finite = pd.to_numeric(grp[mid], errors="coerce").dropna()
            if len(finite) < 2:
                row[std_id] = None
                continue
            std = float(finite.std(ddof=1))
            row[std_id] = None if pd.isna(std) else std
        rows.append(row)
    return pd.DataFrame(rows)


def _join_mean_std(mean: Any, std: Any, *, pm: str, missing_token: str = "n/a") -> Any:
    """``0.72 ± 0.04`` when both exist; otherwise the formatted mean (or None)."""
    mean_s = format_cell("", mean, available=True, missing_token=missing_token)
    if mean_s == missing_token:
        return mean
    std_s = format_cell("", std, available=True, missing_token=missing_token)
    if std_s == missing_token:
        return mean
    return f"{mean_s}{pm}{std_s}"


def _main_table_human(df: pd.DataFrame, *, pm: str) -> pd.DataFrame:
    """Merge rate mean/std into one paper cell; leave force columns numeric."""
    out = df.copy()
    drop: list[str] = []
    for mean_c, std_c in _MAIN_PM_PAIRS:
        if mean_c not in out.columns:
            continue
        std_series = out[std_c] if std_c in out.columns else None
        joined: list[Any] = []
        for i, mean in enumerate(out[mean_c].tolist()):
            std = None if std_series is None else std_series.iloc[i]
            joined.append(_join_mean_std(mean, std, pm=pm))
        out[mean_c] = joined
        if std_c in out.columns:
            drop.append(std_c)
    return out.drop(columns=drop, errors="ignore")


def table_main(
    runs_df: pd.DataFrame,
    *,
    backends_present: Sequence[str],
    episodes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Paper main: model × level; force reported as median + p95.

    ``N`` is episodes run and ``N_valid`` the denominator of every rate on the
    row. With ``episodes_df`` the force quantiles are pooled over episodes
    rather than averaged over per-run quantiles. Rate columns carry
    ``*_std`` = sample std across seeds (ddof=1) after collapsing tasks
    within each seed; CSV keeps the split, md/tex merge them as mean ± std.
    """
    del backends_present  # used by footnotes at render time
    std_ids = tuple(f"{m}_std" for m in _MAIN_SEED_STD_METRICS)
    rename_ids = _MAIN_METRICS + _DENOMINATOR_IDS + ("n_seeds",) + std_ids
    if runs_df.empty:
        cols = [
            "model",
            "level",
            "N",
            "N_valid",
            "N_seeds",
            *[DISPLAY_NAMES[m] for m in _MAIN_METRICS],
            *[DISPLAY_NAMES[s] for s in std_ids],
        ]
        return pd.DataFrame(columns=cols)
    grouped = _aggregate_metric_mean(runs_df, ["model", "level"])
    grouped = _override_force_from_episodes(grouped, episodes_df, ["model", "level"])
    std_df = _seed_metric_std(runs_df, ["model", "level"], _MAIN_SEED_STD_METRICS)
    if not std_df.empty:
        grouped = grouped.merge(std_df, on=["model", "level"], how="left")
    keep = ["model", "level", *[c for c in _DENOMINATOR_IDS if c in grouped.columns]]
    if "n_seeds" in grouped.columns:
        keep.append("n_seeds")
    for mid in _MAIN_METRICS:
        if mid not in grouped.columns:
            continue
        keep.append(mid)
        std_id = f"{mid}_std"
        if std_id in grouped.columns:
            keep.append(std_id)
    out = grouped[keep].sort_values(["model", "level"]).reset_index(drop=True)
    return _rename_metrics(out, rename_ids)


def table_per_task(
    runs_df: pd.DataFrame,
    *,
    episodes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-task view: model × task × level (seeds collapsed)."""
    if runs_df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ("model", "task_id", "task_name", "level") if c in runs_df.columns]
    if "task_id" not in group_cols and "task_name" in runs_df.columns:
        group_cols = [c for c in ("model", "task_name", "level") if c in runs_df.columns]
    grouped = _aggregate_metric_mean(runs_df, group_cols)
    grouped = _override_force_from_episodes(grouped, episodes_df, group_cols)
    keep = group_cols + [m for m in _PER_TASK_METRICS if m in grouped.columns]
    out = grouped[keep].sort_values(group_cols).reset_index(drop=True)
    return _rename_metrics(out, _PER_TASK_METRICS)


def table_ablation(
    runs_df: pd.DataFrame,
    *,
    episodes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ablation-style long table: model × level with dual force means + SR/CR/SASR."""
    if runs_df.empty:
        return pd.DataFrame()
    grouped = _aggregate_metric_mean(runs_df, ["model", "level"])
    grouped = _override_force_from_episodes(grouped, episodes_df, ["model", "level"])
    keep = ["model", "level", *_DENOMINATOR_IDS, *_ABLATION_METRICS]
    keep = [c for c in keep if c in grouped.columns]
    out = grouped[keep].sort_values(["model", "level"]).reset_index(drop=True)
    return _rename_metrics(out, _ABLATION_METRICS + _DENOMINATOR_IDS)


def _has_usable_column(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    series = df[col]
    if series.isna().all():
        return False
    # String token "unavailable" counts as missing: a column whose every
    # non-null entry is the token has no usable value.
    if series.dtype == object:
        non_na = series.notna()
        lowered = series[non_na].astype(str).str.strip().str.lower()
        if not bool(non_na.any()) or bool((lowered == UNAVAILABLE).all()):
            return False
    return True


def _ensure_group_cols(runs_df: pd.DataFrame, preferred: Sequence[str]) -> list[str]:
    cols = [c for c in preferred if c in runs_df.columns]
    if not cols:
        raise KeyError(f"none of group columns {list(preferred)} present in runs frame")
    return cols


def table_joint_outcomes(
    runs_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "level"),
) -> pd.DataFrame:
    """SS/US/CFF/CIF counts + rates by group. Missing joint cols → unavailable (not 0)."""
    if runs_df.empty:
        cols = list(group_cols) + ["N", *[DISPLAY_NAMES[m] for m in _JOINT_OUTCOME_METRICS]]
        return pd.DataFrame(columns=cols)

    gcols = _ensure_group_cols(runs_df, group_cols)
    joint_present = all(_has_usable_column(runs_df, c) for c in _JOINT_COUNT_IDS)
    if not joint_present:
        # Still emit a skeleton with UNAVAILABLE so regenerations stay honest.
        keys = (
            runs_df[gcols]
            .drop_duplicates()
            .sort_values(list(gcols))
            .reset_index(drop=True)
        )
        out = keys.copy()
        if "n_episodes" in runs_df.columns:
            nmap = (
                runs_df.groupby(list(gcols), dropna=False)["n_episodes"]
                .sum()
                .rename("n_episodes")
            )
            out = out.merge(nmap.reset_index(), on=list(gcols), how="left")
        else:
            out["n_episodes"] = None
        for mid in _JOINT_OUTCOME_METRICS:
            out[mid] = None
        keep = gcols + ["n_episodes", *_JOINT_OUTCOME_METRICS]
        return _rename_metrics(out[keep], _JOINT_OUTCOME_METRICS + ("n_episodes",))

    grouped = _aggregate_metric_mean(runs_df, list(gcols))
    # SSR is the headline safety number, so it carries an interval; the SS
    # count and N_valid are both exact here, no rate-times-denominator round
    # trip. The interval must sit on the same denominator as the point
    # estimate: ssr is ss / n_valid, so Wilson(ss, n_episodes) would centre a
    # different rate whenever setup_invalid > 0.
    lo: list[float | None] = []
    hi: list[float | None] = []
    for _, grow in grouped.iterrows():
        denom = _valid_denominator(grow)
        interval = wilson_interval(grow.get("ss"), denom)
        lo.append(None if interval is None else interval[0])
        hi.append(None if interval is None else interval[1])
    grouped["ssr_ci_lo"] = lo
    grouped["ssr_ci_hi"] = hi
    metrics = _JOINT_OUTCOME_METRICS + ("ssr_ci_lo", "ssr_ci_hi")
    denoms = [c for c in _DENOMINATOR_IDS if c in grouped.columns]
    keep = gcols + [*denoms, *[m for m in metrics if m in grouped.columns]]
    out = grouped[keep].sort_values(list(gcols)).reset_index(drop=True)
    return _rename_metrics(out, metrics + _DENOMINATOR_IDS)


def _valid_denominator(row: Mapping[str, Any] | pd.Series) -> int | float | None:
    """``n_valid_episodes`` when present and not null, else ``n_episodes``."""
    for col in ("n_valid_episodes", "n_episodes"):
        value = row.get(col) if hasattr(row, "get") else None
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return value
    return None


def table_sr_ssr_cr(
    runs_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "level"),
) -> pd.DataFrame:
    """SR–SSR–CR comparison (SASR kept as a separate column; never renamed to SSR)."""
    if runs_df.empty:
        cols = list(group_cols) + ["N", "SR", "SSR", "CR", "SASR"]
        return pd.DataFrame(columns=cols)

    gcols = _ensure_group_cols(runs_df, group_cols)
    grouped = _aggregate_metric_mean(runs_df, list(gcols))
    # Ensure ssr column exists (may be missing → None → unavailable)
    if "ssr" not in grouped.columns:
        grouped = grouped.copy()
        grouped["ssr"] = None
    if "sasr" not in grouped.columns:
        grouped = grouped.copy()
        grouped["sasr"] = None
    for mid in ("sr", "cr"):
        if mid not in grouped.columns:
            grouped[mid] = None
    denoms = [c for c in _DENOMINATOR_IDS if c in grouped.columns]
    keep = gcols + [*denoms, "sr", "ssr", "cr", "sasr"]
    out = grouped[keep].sort_values(list(gcols)).reset_index(drop=True)
    return _rename_metrics(out, ("sr", "ssr", "cr", "sasr", *_DENOMINATOR_IDS))


def table_base_vs_oracle(
    runs_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "level"),
    oracle_runs_df: pd.DataFrame | None = None,
    base_filter: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Base vs Oracle SR/SSR/CR comparison.

    Oracle rows come from ``oracle_runs_df``, from rows tagged
    ``policy_kind=oracle``, or — the convention the eval tree actually uses —
    from models named ``{base}_oracle_*``. Those are paired with ``{base}`` by
    name and never enter the ``Base_*`` columns. Oracle columns stay
    ``unavailable`` when no arm exists. No subjective unified score.
    """
    if runs_df.empty and (oracle_runs_df is None or oracle_runs_df.empty):
        cols = list(group_cols) + [
            "N",
            "Base_SR",
            "Base_SSR",
            "Base_CR",
            "Oracle_SR",
            "Oracle_SSR",
            "Oracle_CR",
        ]
        return pd.DataFrame(columns=cols)

    gcols = _ensure_group_cols(runs_df if not runs_df.empty else oracle_runs_df, group_cols)

    base_df = runs_df
    named_oracle: pd.DataFrame | None = None
    if base_filter:
        mask = pd.Series(True, index=base_df.index)
        for k, v in base_filter.items():
            if k in base_df.columns:
                mask &= base_df[k] == v
        base_df = base_df.loc[mask]
    elif "policy_kind" in runs_df.columns:
        base_df = runs_df[runs_df["policy_kind"].astype(str).str.lower().isin({"base", "default"})]
        if base_df.empty:
            base_df = runs_df
    if "model" in base_df.columns:
        is_arm = base_df["model"].map(lambda m: is_safety_layer_model(str(m)))
        if bool(is_arm.any()):
            named_oracle = base_df.loc[is_arm].copy()
            # The arm is reported on its base's row: pi05_oracle_swr_aegis → pi05.
            named_oracle["model"] = named_oracle["model"].map(
                lambda m: null_parent_model(str(m)) or str(m)
            )
            base_df = base_df.loc[~is_arm]

    base_tab = table_sr_ssr_cr(base_df, group_cols=gcols)
    # Rename display cols to Base_*
    rename_base = {"SR": "Base_SR", "SSR": "Base_SSR", "CR": "Base_CR", "SASR": "Base_SASR"}
    base_tab = base_tab.rename(columns={k: v for k, v in rename_base.items() if k in base_tab.columns})
    if "Base_SASR" in base_tab.columns:
        base_tab = base_tab.drop(columns=["Base_SASR"])

    oracle_src = oracle_runs_df
    if oracle_src is None and "policy_kind" in runs_df.columns:
        tagged = runs_df[runs_df["policy_kind"].astype(str).str.lower() == "oracle"]
        oracle_src = tagged if not tagged.empty else None
    if oracle_src is None and named_oracle is not None and not named_oracle.empty:
        oracle_src = named_oracle

    if oracle_src is not None and not oracle_src.empty:
        ora_tab = table_sr_ssr_cr(oracle_src, group_cols=gcols)
        rename_ora = {"SR": "Oracle_SR", "SSR": "Oracle_SSR", "CR": "Oracle_CR"}
        ora_tab = ora_tab.rename(
            columns={k: v for k, v in rename_ora.items() if k in ora_tab.columns}
        )
        # Keep N from base; drop oracle N/SASR
        ora_keep = gcols + [c for c in ("Oracle_SR", "Oracle_SSR", "Oracle_CR") if c in ora_tab.columns]
        ora_tab = ora_tab[ora_keep]
        out = base_tab.merge(ora_tab, on=list(gcols), how="outer")
    else:
        out = base_tab.copy()
        out["Oracle_SR"] = None
        out["Oracle_SSR"] = None
        out["Oracle_CR"] = None

    return out.sort_values(list(gcols)).reset_index(drop=True)


def check_joint_table_consistency(
    runs_df: pd.DataFrame,
    *,
    atol: float = 1e-9,
) -> list[str]:
    """Return list of consistency violation messages (empty if OK / unavailable)."""
    issues: list[str] = []
    if not all(c in runs_df.columns for c in ("ss", "us", "cff", "cif", "n_episodes")):
        return issues  # nothing to check when joint cols absent
    for idx, row in runs_df.iterrows():
        vals = {c: row.get(c) for c in _JOINT_COUNT_IDS}
        if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in vals.values()):
            continue
        n = row.get("n_valid_episodes")
        if n is None or (isinstance(n, float) and pd.isna(n)):
            n = row.get("n_episodes")
        if n is None or (isinstance(n, float) and pd.isna(n)):
            continue
        total = int(vals["ss"]) + int(vals["us"]) + int(vals["cff"]) + int(vals["cif"])
        if int(n) == 0:
            if total != 0:
                issues.append(f"row {idx}: ss+us+cff+cif={total} != N_valid=0")
            continue
        if total != int(n):
            issues.append(f"row {idx}: ss+us+cff+cif={total} != N_valid={n}")
        inv = 1.0 / float(n) if float(n) else None
        if inv is None:
            continue
        sr_hat = (int(vals["ss"]) + int(vals["us"])) * inv
        cr_hat = (int(vals["us"]) + int(vals["cif"])) * inv
        ssr_hat = int(vals["ss"]) * inv
        if "sr" in runs_df.columns and row.get("sr") is not None and not pd.isna(row.get("sr")):
            if abs(float(row["sr"]) - sr_hat) > atol:
                issues.append(f"row {idx}: sr {row['sr']} != reconstructed {sr_hat}")
        if "cr" in runs_df.columns and row.get("cr") is not None and not pd.isna(row.get("cr")):
            if abs(float(row["cr"]) - cr_hat) > atol:
                issues.append(f"row {idx}: cr {row['cr']} != reconstructed {cr_hat}")
        if "ssr" in runs_df.columns and row.get("ssr") is not None and not pd.isna(row.get("ssr")):
            if abs(float(row["ssr"]) - ssr_hat) > atol:
                issues.append(f"row {idx}: ssr {row['ssr']} != reconstructed {ssr_hat}")
            if float(row["ssr"]) > float(row.get("sr", ssr_hat)) + atol:
                issues.append(f"row {idx}: ssr {row['ssr']} > sr {row.get('sr')}")
    return issues


def build_behavior_footnotes() -> list[str]:
    """Footnotes for the §7.1 behavior table (comparability + denominators)."""
    return [
        "n/a means diagnostics is None (disabled / backend not wired) or the "
        "denominator is empty — never a silent 0.0.",
        "ReactionRate is closed-loop: pre-contact kinematic deviation starting "
        "at or after the next action-chunk boundary strictly after onset "
        "(min_latency_chunks boundaries; default 1). "
        "ReactionRateKinematic is the 15°/amplitude bit (L0 FPR ≈ 1.0).",
        "ReactionRate / StopRate / EvadeRate / RecoveryRate are cross-backend "
        "comparable. Denominators: N_trig for the first three, N_recov for Recovery.",
        "*_lo / *_hi are 95% Wilson intervals on that row's own denominator. "
        "N_recov is often < 20, where the point estimate alone is not evidence "
        "of a difference — read the interval, not the fourth decimal.",
        "StopRate is mean(stop_preemptive) among triggered episodes, not stopped "
        "(stopped includes being knocked to a halt).",
        "ReactLat_p50 / ReactLatChunks_p50 are over closed-loop reacted "
        "episodes only (n/a when ReactionRate is 0). They are not the "
        "kinematic t_react delay.",
        "MinClear_p50 is a body-origin surrogate, not a geometric minimum; do not "
        "rank absolute metres across backends.",
        "Recovery on HazardPersists=true (L1/L2 residual static hazard) is a "
        "different quantity from HazardPersists=false (L3 cleared scene); do not "
        "pool those rows into one average.",
        "CFF is not “Safe Stop”; see plots/failure_decomposition.png for the "
        "stop-no-recover / evade-fail / no-response split.",
        "not_triggered and baseline_unusable are excluded from N_trig / N_recov "
        "(four-axis denominators) but are counted in the class histogram. "
        "baseline_unusable means the pre-onset window could not calibrate the "
        "relative gates, not that the policy failed to respond.",
        "Class-rate columns divide by N_valid (setup-invalid trials excluded), "
        "the same population the counts were taken from, so they sum to 1.",
    ]


def _sum_optional_int(series: pd.Series) -> int | None:
    mask = series.notna()
    if not bool(mask.any()):
        return None
    return int(series[mask].astype(float).sum())


def _weighted_by(grp: pd.DataFrame, value_col: str, weight_col: str) -> float | None:
    if value_col not in grp.columns or weight_col not in grp.columns:
        return None
    mask = grp[value_col].notna() & grp[weight_col].notna()
    if not bool(mask.any()):
        return None
    weights = grp.loc[mask, weight_col].astype(float)
    values = grp.loc[mask, value_col].astype(float)
    wsum = float(weights.sum())
    if wsum <= 0:
        return None
    return float((values * weights).sum() / wsum)


def _constant_hazard_persists(series: pd.Series) -> bool | None:
    """Keep the flag only when every non-missing row agrees.

    Mixed True/False (or True/None) → None so a cross-level rollup cannot
    silently average L1/L2 residual-hazard Recovery with L3 cleared Recovery.
    """
    normalized: list[bool | None] = []
    for value in series:
        if value is None:
            normalized.append(None)
            continue
        try:
            if pd.isna(value):
                normalized.append(None)
                continue
        except (TypeError, ValueError):
            pass
        normalized.append(bool(value))
    uniq = set(normalized)
    if len(uniq) == 1:
        return next(iter(uniq))
    return None


def _set_wilson(
    row: dict[str, Any],
    metric_id: str,
    rate: float | None,
    denom: int | float | None,
) -> None:
    """Attach ``<metric_id>_ci_lo`` / ``_ci_hi`` in place; ``None`` when undefined.

    The process axes are collapsed as weighted rates, so the success count is
    reconstructed as ``rate * denom`` rather than carried through.
    """
    interval = (
        None if rate is None or denom is None else wilson_interval(float(rate) * float(denom), denom)
    )
    row[f"{metric_id}_ci_lo"] = None if interval is None else interval[0]
    row[f"{metric_id}_ci_hi"] = None if interval is None else interval[1]


def _aggregate_behavior_metrics(runs_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Collapse runs for the behavior table, respecting each rate's denominator."""
    if runs_df.empty:
        return runs_df.copy()
    work = runs_df.copy()
    if "n_episodes" not in work.columns:
        work["n_episodes"] = 1

    rows: list[dict[str, Any]] = []
    for key, grp in work.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, Any] = {group_cols[i]: key[i] for i in range(len(group_cols))}
        n_ep = float(grp["n_episodes"].astype(float).sum())
        row["n_episodes"] = int(n_ep) if n_ep == int(n_ep) else n_ep
        # Class counts come from behavior_group_metrics(valid); their shares
        # must be over the same population or they stop summing to 1 and a
        # setup-invalid trial reads as "more no-response".
        if "n_valid_episodes" in grp.columns and grp["n_valid_episodes"].notna().any():
            n_valid = float(grp["n_valid_episodes"].fillna(0).astype(float).sum())
        else:
            n_valid = n_ep
        row["n_valid_episodes"] = int(n_valid) if n_valid == int(n_valid) else n_valid

        n_trig = _sum_optional_int(grp["n_triggered"]) if "n_triggered" in grp.columns else None
        row["n_triggered"] = n_trig
        row["reaction_rate"] = _weighted_by(grp, "reaction_rate", "n_triggered")
        _set_wilson(row, "reaction_rate", row["reaction_rate"], n_trig)
        row["reaction_rate_kinematic"] = _weighted_by(
            grp, "reaction_rate_kinematic", "n_triggered"
        )
        row["stop_rate"] = _weighted_by(grp, "stop_rate", "n_triggered")
        row["evade_rate"] = _weighted_by(grp, "evade_rate", "n_triggered")

        row["n_reaction_latency"] = (
            _sum_optional_int(grp["n_reaction_latency"])
            if "n_reaction_latency" in grp.columns
            else None
        )
        row["reaction_latency_p50"] = _weighted_by(
            grp, "reaction_latency_p50", "n_reaction_latency"
        )
        row["reaction_latency_chunks_p50"] = _weighted_by(
            grp, "reaction_latency_chunks_p50", "n_reaction_latency"
        )

        n_recov = _sum_optional_int(grp["n_recovery"]) if "n_recovery" in grp.columns else None
        row["n_recovery"] = n_recov
        row["recovery_rate"] = _weighted_by(grp, "recovery_rate", "n_recovery")
        _set_wilson(row, "recovery_rate", row["recovery_rate"], n_recov)

        row["n_min_clearance"] = (
            _sum_optional_int(grp["n_min_clearance"]) if "n_min_clearance" in grp.columns else None
        )
        row["min_clearance_p50"] = _weighted_by(grp, "min_clearance_p50", "n_min_clearance")
        row["hazard_persists"] = (
            _constant_hazard_persists(grp["hazard_persists"])
            if "hazard_persists" in grp.columns
            else None
        )

        for cid in _BEHAVIOR_COUNT_IDS:
            count = _sum_optional_int(grp[cid]) if cid in grp.columns else None
            row[cid] = count
            rate_id = f"{cid}_rate"
            if count is None or n_valid <= 0:
                row[rate_id] = None
            else:
                row[rate_id] = float(count) / float(n_valid)
        rows.append(row)
    return pd.DataFrame(rows)


def table_behavior(
    runs_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("backend", "model", "level"),
) -> pd.DataFrame:
    """§7.1 process-metric table. Missing diagnostics → n/a (never 0.0)."""
    display_ids = ("n_episodes",) + _BEHAVIOR_TABLE_IDS
    if runs_df.empty:
        cols = list(group_cols) + [DISPLAY_NAMES.get(m, m) for m in display_ids]
        return pd.DataFrame(columns=cols)

    gcols = _ensure_group_cols(runs_df, group_cols)
    has_behavior = any(c in runs_df.columns for c in ("n_triggered", "reaction_rate"))
    if not has_behavior:
        keys = runs_df[gcols].drop_duplicates().sort_values(list(gcols)).reset_index(drop=True)
        out = keys.copy()
        if "n_episodes" in runs_df.columns:
            nmap = (
                runs_df.groupby(list(gcols), dropna=False)["n_episodes"].sum().rename("n_episodes")
            )
            out = out.merge(nmap.reset_index(), on=list(gcols), how="left")
        else:
            out["n_episodes"] = None
        for mid in _BEHAVIOR_TABLE_IDS:
            out[mid] = None
        keep = gcols + ["n_episodes", *_BEHAVIOR_TABLE_IDS]
        return _rename_metrics(out[keep], display_ids)

    grouped = _aggregate_behavior_metrics(runs_df, list(gcols))
    keep = gcols + [
        "n_episodes",
        *(["n_valid_episodes"] if "n_valid_episodes" in grouped.columns else []),
        *[m for m in _BEHAVIOR_TABLE_IDS if m in grouped.columns],
    ]
    out = grouped[keep].sort_values(list(gcols)).reset_index(drop=True)
    return _rename_metrics(out, display_ids + ("n_valid_episodes",))


def _df_to_markdown(df: pd.DataFrame, footnotes: Sequence[str], *, missing_token: str = "n/a") -> str:
    # Prefer pandas styling when available; fall back to simple pipe table.
    try:
        body = df.to_markdown(index=False)
    except ImportError:
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(str(h) for h in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for _, row in df.iterrows():
            cells = []
            for h in headers:
                val = row[h]
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    cells.append(missing_token)
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")
        body = "\n".join(lines)
    if footnotes:
        body += "\n\n" + "\n".join(f"- {note}" for note in footnotes)
    return body + "\n"


def _df_to_latex(df: pd.DataFrame, footnotes: Sequence[str], *, missing_token: str = "n/a") -> str:
    # Escape minimal special chars in string cells
    tex_df = df.copy()
    for col in tex_df.columns:
        if tex_df[col].dtype == object:
            tex_df[col] = tex_df[col].map(
                lambda x: (
                    missing_token
                    if x is None or (isinstance(x, float) and pd.isna(x))
                    else str(x).replace("_", r"\_").replace("%", r"\%")
                )
            )
    try:
        body = tex_df.to_latex(index=False, escape=False)
    except Exception:
        body = tex_df.to_string(index=False)
    if footnotes:
        body += "\n% Footnotes:\n"
        for i, note in enumerate(footnotes, start=1):
            safe = note.replace("_", r"\_")
            body += f"% [{i}] {safe}\n"
    return body if body.endswith("\n") else body + "\n"


def _apply_na_display(
    df: pd.DataFrame,
    *,
    missing_token: str = "n/a",
    format_numbers: bool = False,
) -> pd.DataFrame:
    """Replace None/NaN with missing_token for human table formats.

    ``format_numbers`` additionally rounds floats via ``format_cell``. Only md /
    tex use it: those are read by humans and a raw ``0.30000000000000004`` from
    a float mean is noise in a paper table. CSV keeps full precision so
    downstream analysis is not silently truncated.
    """
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(
            lambda v: (
                missing_token
                if v is None or (isinstance(v, float) and pd.isna(v))
                else (format_cell(col, v, available=True, missing_token=missing_token)
                      if format_numbers
                      else v)
            )
        )
    return out


def render_table(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    *,
    footnotes: Sequence[str],
    formats: Sequence[str],
    missing_token: str = "n/a",
    fill_csv_missing: bool = False,
) -> None:
    """Write ``{stem}.csv`` / ``.md`` / ``.tex`` under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    display = _apply_na_display(df, missing_token=missing_token, format_numbers=True)
    for fmt in formats:
        if fmt == "csv":
            # Joint tables write the locked unavailable token; behavior tables
            # write n/a. Other tables keep blank cells for missing (never 0.0).
            if missing_token == UNAVAILABLE or fill_csv_missing:
                csv_df = _apply_na_display(df, missing_token=missing_token)
            else:
                csv_df = df.copy()
            csv_df.to_csv(out_dir / f"{stem}.csv", index=False)
        elif fmt == "md":
            (out_dir / f"{stem}.md").write_text(
                _df_to_markdown(display, footnotes, missing_token=missing_token),
                encoding="utf-8",
            )
        elif fmt == "tex":
            (out_dir / f"{stem}.tex").write_text(
                _df_to_latex(display, footnotes, missing_token=missing_token),
                encoding="utf-8",
            )
        else:
            raise ValueError(f"unsupported table format: {fmt!r}")


def render_joint_tables(
    runs_df: pd.DataFrame,
    out_dir: Path,
    *,
    group_cols: Sequence[str] = ("model", "level"),
    oracle_runs_df: pd.DataFrame | None = None,
) -> None:
    """Write joint-outcome / SR–SSR–CR / Base-vs-Oracle tables."""
    footnotes = build_joint_footnotes()
    joint = table_joint_outcomes(runs_df, group_cols=group_cols)
    render_table(
        joint,
        out_dir,
        "table_joint_outcomes",
        footnotes=footnotes,
        formats=("csv", "md", "tex"),
        missing_token=UNAVAILABLE,
    )
    sr_ssr_cr = table_sr_ssr_cr(runs_df, group_cols=group_cols)
    render_table(
        sr_ssr_cr,
        out_dir,
        "table_sr_ssr_cr",
        footnotes=footnotes,
        formats=("csv", "md", "tex"),
        missing_token=UNAVAILABLE,
    )
    base_ora = table_base_vs_oracle(
        runs_df,
        group_cols=group_cols,
        oracle_runs_df=oracle_runs_df,
    )
    render_table(
        base_ora,
        out_dir,
        "table_base_vs_oracle",
        footnotes=footnotes,
        formats=("csv", "md", "tex"),
        missing_token=UNAVAILABLE,
    )


def render_all_tables(
    runs_df: pd.DataFrame,
    out_dir: Path,
    *,
    backends_present: Sequence[str],
    include_joint: bool = True,
    oracle_runs_df: pd.DataFrame | None = None,
    episodes_df: pd.DataFrame | None = None,
) -> None:
    """table_main + per-task + ablation (+ optional joint tables).

    ``episodes_df`` (from ``aggregate.episodes_to_frame``) lets the force
    quantiles be pooled over episodes instead of averaged over per-run
    quantiles; without it the tables fall back to the weighted mean and the
    footnote still describes a pooled quantile, so pass it when you have it.
    """
    # An unknown backend has an unknown force source, which is the strongest
    # reason not to compare force across backends — treat it as distinct.
    sources = {BACKEND_FORCE_SOURCE.get(b, f"unknown:{b}") for b in backends_present}
    footnotes = build_footnotes(
        backends_present=backends_present,
        has_mixed_force_sources=len(sources) > 1,
    )
    main = table_main(runs_df, backends_present=backends_present, episodes_df=episodes_df)
    main_notes = [*footnotes, _SEED_STD_FOOTNOTE]
    render_table(
        main,
        out_dir,
        "table_main",
        footnotes=main_notes,
        formats=("csv",),
    )
    render_table(
        _main_table_human(main, pm=" ± "),
        out_dir,
        "table_main",
        footnotes=main_notes,
        formats=("md",),
    )
    render_table(
        _main_table_human(main, pm=r" $\pm$ "),
        out_dir,
        "table_main",
        footnotes=main_notes,
        formats=("tex",),
    )
    per_task = table_per_task(runs_df, episodes_df=episodes_df)
    render_table(
        per_task,
        out_dir,
        "table_per_task",
        footnotes=footnotes,
        formats=("csv",),
    )
    ablation = table_ablation(runs_df, episodes_df=episodes_df)
    render_table(
        ablation,
        out_dir,
        "table_ablation",
        footnotes=footnotes,
        formats=("csv",),
    )
    behavior = table_behavior(runs_df)
    render_table(
        behavior,
        out_dir,
        "table_behavior",
        footnotes=build_behavior_footnotes(),
        formats=("csv", "md", "tex"),
        fill_csv_missing=True,
    )
    if include_joint:
        gcols = [c for c in ("model", "level") if c in runs_df.columns]
        if gcols:
            render_joint_tables(
                runs_df,
                out_dir,
                group_cols=gcols,
                oracle_runs_df=oracle_runs_df,
            )
