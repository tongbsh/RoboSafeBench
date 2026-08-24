"""Paper tables and display-name rendering (sole home for SR / FPeakMean labels)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

# Token for missing / non-reconstructible fields (Prompt 21). Never coerce to 0.
UNAVAILABLE = "unavailable"

# Display names ONLY here (not in core/)
DISPLAY_NAMES: dict[str, str] = {
    "sr": "SR",
    "cr": "CR",
    "contact_steps_per_ep": "ColSteps/Ep",
    "topple_rate": "Topple",
    "f_peak_mean_all": "FPeakMean(N)",
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
    "cif_rate": "CIF_rate",
    "safety_score": "SafetyScore",
    "n_episodes": "N",
    "force_anomaly_episodes": "ForceAnomalyEps",
    "oracle_sr": "Oracle_SR",
    "oracle_ssr": "Oracle_SSR",
    "oracle_cr": "Oracle_CR",
    "base_sr": "Base_SR",
    "base_ssr": "Base_SSR",
    "base_cr": "Base_CR",
    "n_triggered": "N_trig",
    "reaction_rate": "ReactionRate",
    "n_reaction_latency": "N_lat",
    "reaction_latency_p50": "ReactLat_p50",
    "reaction_latency_chunks_p50": "ReactLatChunks_p50",
    "stop_rate": "StopRate",
    "evade_rate": "EvadeRate",
    "n_recovery": "N_recov",
    "recovery_rate": "RecoveryRate",
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
    "bc_collide_unreacted_rate": "Collide (no reaction) rate",
    "bc_collide_reacted_rate": "Collide (late) rate",
    "bc_evade_success_rate": "Evade rate",
    "bc_evade_fail_rate": "Evade→fail rate",
    "bc_stop_recover_rate": "Stop→recover rate",
    "bc_stop_no_recover_rate": "Stop→no recover rate",
    "bc_partial_response_rate": "Partial rate",
    "bc_no_response_rate": "No response rate",
    "bc_not_triggered_rate": "Not triggered rate",
}

_JOINT_COUNT_IDS = ("ss", "us", "cff", "cif")

# Known capability facts for footnotes (not hyperparameter defaults)
BACKEND_FORCE_SOURCE: dict[str, str] = {
    "libero": "mujoco_mj_contactForce",
    "simplerenv": "sapien_impulse",
    "robotwin": "sapien_impulse",
    "robocasa": "mujoco_mj_contactForce",
}

_MAIN_METRICS = (
    "sr",
    "cr",
    "topple_rate",
    "f_peak_mean_all",
    "sasr",
    "safety_score",
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
    "n_episodes",
)

_ABLATION_METRICS = (
    "sr",
    "cr",
    "f_peak_mean_all",
    "f_peak_mean_contact",
    "sasr",
    "safety_score",
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
)
_BEHAVIOR_COUNT_IDS = tuple(f"bc_{name}" for name in _BEHAVIOR_CLASS_IDS)
_BEHAVIOR_RATE_IDS = tuple(f"bc_{name}_rate" for name in _BEHAVIOR_CLASS_IDS)
_BEHAVIOR_PROCESS_IDS = (
    "n_triggered",
    "reaction_rate",
    "n_reaction_latency",
    "reaction_latency_p50",
    "reaction_latency_chunks_p50",
    "stop_rate",
    "evade_rate",
    "recovery_rate",
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
            "FPeakMean(N) is f_peak_mean_all: mean(peak_force_n) over all episodes; "
            "episodes without contact contribute 0 N."
        ),
        (
            "FPeakMeanContact(N) is f_peak_mean_contact (contact episodes only); "
            "reported in runs.csv / per-task tables, not the paper main force column."
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
        "SSR is the SS quadrant rate (ss/N); SASR uses severity penalty φ — do not conflate or rename.",
        "CFF = contact-free failure (not “Safe Stop”); CIF = contact-involved failure "
        "(not “Collision-induced Failure”).",
        f"{UNAVAILABLE} means the field is missing from the input and was not reconstructed "
        "(never a silent 0).",
        "No subjective unified safety total score is defined or reported here.",
        "Invariants: ss+us+cff+cif=N; ssr≤sr; us_rate≤cr; sr=(ss+us)/N; cr=(us+cif)/N.",
        "Oracle / Shield columns are reserved; values stay unavailable until those runs exist.",
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
        "f_peak_p95",
        "f_peak_max",
        "sasr",
        "safety_score",
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
        weights = grp["n_episodes"].astype(float)
        wsum = float(weights.sum())
        row["n_episodes"] = int(wsum) if wsum == int(wsum) else wsum
        if "force_anomaly_episodes" in grp.columns:
            row["force_anomaly_episodes"] = int(grp["force_anomaly_episodes"].fillna(0).sum())
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
        if all(c in row and row[c] is not None for c in _JOINT_COUNT_IDS) and wsum > 0:
            ss_i = int(row["ss"])
            us_i = int(row["us"])
            cff_i = int(row["cff"])
            cif_i = int(row["cif"])
            inv = 1.0 / float(wsum)
            row["ssr"] = ss_i * inv
            row["us_rate"] = us_i * inv
            row["cff_rate"] = cff_i * inv
            row["cif_rate"] = cif_i * inv
        rows.append(row)
    return pd.DataFrame(rows)


def table_main(runs_df: pd.DataFrame, *, backends_present: Sequence[str]) -> pd.DataFrame:
    """Paper main: model × level; uses f_peak_mean_all only for force."""
    del backends_present  # used by footnotes at render time
    if runs_df.empty:
        cols = ["model", "level", *[DISPLAY_NAMES[m] for m in _MAIN_METRICS], "N"]
        return pd.DataFrame(columns=cols)
    grouped = _aggregate_metric_mean(runs_df, ["model", "level"])
    keep = ["model", "level", "n_episodes", *_MAIN_METRICS]
    keep = [c for c in keep if c in grouped.columns]
    out = grouped[keep].sort_values(["model", "level"]).reset_index(drop=True)
    # Format n/a for missing topple etc. as values stay None; rename for display
    return _rename_metrics(out, _MAIN_METRICS + ("n_episodes",))


def table_per_task(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Per-task view: model × task × level (seeds collapsed)."""
    if runs_df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ("model", "task_id", "task_name", "level") if c in runs_df.columns]
    if "task_id" not in group_cols and "task_name" in runs_df.columns:
        group_cols = [c for c in ("model", "task_name", "level") if c in runs_df.columns]
    grouped = _aggregate_metric_mean(runs_df, group_cols)
    keep = group_cols + [m for m in _PER_TASK_METRICS if m in grouped.columns]
    out = grouped[keep].sort_values(group_cols).reset_index(drop=True)
    return _rename_metrics(out, _PER_TASK_METRICS)


def table_ablation(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Ablation-style long table: model × level with dual force means + SR/CR/SASR."""
    if runs_df.empty:
        return pd.DataFrame()
    grouped = _aggregate_metric_mean(runs_df, ["model", "level"])
    keep = ["model", "level", "n_episodes", *_ABLATION_METRICS]
    keep = [c for c in keep if c in grouped.columns]
    out = grouped[keep].sort_values(["model", "level"]).reset_index(drop=True)
    return _rename_metrics(out, _ABLATION_METRICS + ("n_episodes",))


def _has_usable_column(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    series = df[col]
    if series.isna().all():
        return False
    # String token "unavailable" counts as missing
    if series.dtype == object:
        lowered = series.astype(str).str.strip().str.lower()
        if bool((lowered == UNAVAILABLE).all() | series.isna()):
            # all unavailable or na
            non_na = series.notna()
            if not bool(non_na.any()) or bool((lowered[non_na] == UNAVAILABLE).all()):
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
    keep = gcols + ["n_episodes", *[m for m in _JOINT_OUTCOME_METRICS if m in grouped.columns]]
    out = grouped[keep].sort_values(list(gcols)).reset_index(drop=True)
    return _rename_metrics(out, _JOINT_OUTCOME_METRICS + ("n_episodes",))


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
    keep = gcols + ["n_episodes", "sr", "ssr", "cr", "sasr"]
    out = grouped[keep].sort_values(list(gcols)).reset_index(drop=True)
    return _rename_metrics(out, ("sr", "ssr", "cr", "sasr", "n_episodes"))


def table_base_vs_oracle(
    runs_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("model", "level"),
    oracle_runs_df: pd.DataFrame | None = None,
    base_filter: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Base vs Oracle SR/SSR/CR comparison.

    Oracle columns stay ``unavailable`` until ``oracle_runs_df`` (or rows tagged
    ``policy_kind=oracle``) is supplied. No subjective unified score.
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
        n = row.get("n_episodes")
        if n is None or (isinstance(n, float) and pd.isna(n)):
            continue
        total = int(vals["ss"]) + int(vals["us"]) + int(vals["cff"]) + int(vals["cif"])
        if total != int(n):
            issues.append(f"row {idx}: ss+us+cff+cif={total} != N={n}")
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
        "ReactionRate / StopRate / EvadeRate / RecoveryRate are cross-backend "
        "comparable. Denominators: N_trig for the first three, N_recov for Recovery.",
        "StopRate is mean(stop_preemptive) among triggered episodes, not stopped "
        "(stopped includes being knocked to a halt).",
        "ReactLat_p50 is in task steps and is comparable only inside one backend; "
        "ReactLatChunks_p50 is cautiously comparable (latency / action_chunk_len).",
        "MinClear_p50 is a body-origin surrogate, not a geometric minimum; do not "
        "rank absolute metres across backends.",
        "Recovery on HazardPersists=true (L1/L2 residual static hazard) is a "
        "different quantity from HazardPersists=false (L3 cleared scene); do not "
        "pool those rows into one average.",
        "CFF is not “Safe Stop”; see plots/failure_decomposition.png for the "
        "stop-no-recover / evade-fail / no-response split.",
        "not_triggered is excluded from N_trig / N_recov (four-axis denominators) "
        "but is counted in the class histogram.",
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

        n_trig = _sum_optional_int(grp["n_triggered"]) if "n_triggered" in grp.columns else None
        row["n_triggered"] = n_trig
        row["reaction_rate"] = _weighted_by(grp, "reaction_rate", "n_triggered")
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
            if count is None or n_ep <= 0:
                row[rate_id] = None
            else:
                row[rate_id] = float(count) / float(n_ep)
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
    keep = gcols + ["n_episodes", *[m for m in _BEHAVIOR_TABLE_IDS if m in grouped.columns]]
    out = grouped[keep].sort_values(list(gcols)).reset_index(drop=True)
    return _rename_metrics(out, display_ids)


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
) -> None:
    """table_main + per-task + ablation (+ optional joint tables)."""
    sources = {BACKEND_FORCE_SOURCE.get(b) for b in backends_present}
    sources.discard(None)
    footnotes = build_footnotes(
        backends_present=backends_present,
        has_mixed_force_sources=len(sources) > 1,
    )
    main = table_main(runs_df, backends_present=backends_present)
    render_table(
        main,
        out_dir,
        "table_main",
        footnotes=footnotes,
        formats=("csv", "md", "tex"),
    )
    per_task = table_per_task(runs_df)
    render_table(
        per_task,
        out_dir,
        "table_per_task",
        footnotes=footnotes,
        formats=("csv",),
    )
    ablation = table_ablation(runs_df)
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
