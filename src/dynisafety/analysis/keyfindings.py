"""Cross-backend key findings for the paper.

Reads completed ``episodes.jsonl`` (same loader as the rest of this package)
plus already-written analysis CSVs. Does not recompute paired axes, does not
import ``dynisafety.report``, and never writes into the results tree.

Every rate is stored next to its own denominator. Missing stays ``None``
(tables render n/a). Thresholds come from the diagnostics YAML, not from
Python defaults.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dynisafety.analysis.load import (
    BoundEpisode,
    behavior_class_of,
    group_key_tuple,
    is_safety_layer_model,
)
from dynisafety.analysis.render import render_table
from dynisafety.analysis.termination import classify_termination
from dynisafety.config.schema import DiagnosticsConfig
from dynisafety.core.aggregate import is_setup_invalid, metric_episodes
from dynisafety.core.joint_outcomes import classify_episode
from dynisafety.core.metrics import wilson_interval

_PAPER_GROUP_COLS: tuple[str, ...] = ("backend", "model", "level")
_SWEEP_DIRNAME = "_sweep"
_KNOWN_BACKENDS: tuple[str, ...] = ("libero", "robotwin", "simplerenv", "robocasa")
_SCHEDULE_LEVELS: tuple[int, ...] = (2, 3)

# CFF process split required by docs/user/eval.md §3. The leftover bucket is
# reported so the three named pieces plus residual still sum to n_cff.
_CFF_PROCESS: tuple[str, ...] = ("stop_no_recover", "evade_fail", "no_response")
_CFF_RESIDUAL: tuple[str, ...] = (
    "partial_response",
    "not_triggered",
    "baseline_unusable",
    "unlabeled",
    "other",
)

# ---------------------------------------------------------------------------
# Tree scan (completed / incomplete / _sweep)
# ---------------------------------------------------------------------------


def _run_dir_completed(run_dir: Path) -> bool:
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(summary, dict) and summary.get("status") == "completed"


def _iter_jsonl_files(results_root: Path) -> list[Path]:
    root = Path(results_root)
    if not root.is_dir():
        return []
    return sorted(root.rglob("episodes.jsonl"))


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"expected object at {path}:{line_no}")
        rows.append(data)
    return rows


@dataclass(frozen=True)
class TreeScan:
    """Counts over every ``episodes.jsonl`` under a results root."""

    n_jsonl: int
    n_completed_jsonl: int
    n_incomplete_jsonl: int
    n_sweep_jsonl: int
    n_lines: int
    n_completed_lines: int
    n_incomplete_lines: int
    n_sweep_lines: int
    raw_rows: tuple[dict[str, Any], ...]
    raw_completed: tuple[dict[str, Any], ...]
    raw_incomplete: tuple[dict[str, Any], ...]


def scan_results_tree(results_root: Path | str) -> TreeScan:
    """Walk ``episodes.jsonl`` files. Incomplete and ``_sweep`` stay out of metrics."""
    jsonl_paths = _iter_jsonl_files(Path(results_root))
    raw_completed: list[dict[str, Any]] = []
    raw_incomplete: list[dict[str, Any]] = []
    raw_sweep: list[dict[str, Any]] = []
    n_completed_jsonl = n_incomplete_jsonl = n_sweep_jsonl = 0
    for path in jsonl_paths:
        rows = _read_jsonl_dicts(path)
        if _SWEEP_DIRNAME in path.parts:
            n_sweep_jsonl += 1
            raw_sweep.extend(rows)
            continue
        if _run_dir_completed(path.parent):
            n_completed_jsonl += 1
            raw_completed.extend(rows)
        else:
            n_incomplete_jsonl += 1
            raw_incomplete.extend(rows)
    raw_all = tuple(raw_completed + raw_incomplete + raw_sweep)
    return TreeScan(
        n_jsonl=len(jsonl_paths),
        n_completed_jsonl=n_completed_jsonl,
        n_incomplete_jsonl=n_incomplete_jsonl,
        n_sweep_jsonl=n_sweep_jsonl,
        n_lines=len(raw_all),
        n_completed_lines=len(raw_completed),
        n_incomplete_lines=len(raw_incomplete),
        n_sweep_lines=len(raw_sweep),
        raw_rows=raw_all,
        raw_completed=tuple(raw_completed),
        raw_incomplete=tuple(raw_incomplete),
    )


# ---------------------------------------------------------------------------
# Episode field helpers (raw dict or EpisodeRecord)
# ---------------------------------------------------------------------------


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _extra_of(ep: Any) -> Mapping[str, Any]:
    if isinstance(ep, Mapping):
        return _as_mapping(ep.get("extra"))
    return _as_mapping(getattr(ep, "extra", None))


def _diag_of(ep: Any) -> Mapping[str, Any] | None:
    if isinstance(ep, Mapping):
        raw = ep.get("diagnostics")
    else:
        raw = getattr(ep, "diagnostics", None)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return raw
    return None


def _field(ep: Any, name: str) -> Any:
    if isinstance(ep, Mapping):
        return ep.get(name)
    return getattr(ep, name)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def onset_plan_of(ep: Any) -> Any | None:
    extra = _extra_of(ep)
    if "onset_plan" in extra:
        return extra.get("onset_plan")
    nested = extra.get("libero")
    if isinstance(nested, Mapping) and "onset_plan" in nested:
        return nested.get("onset_plan")
    return None


def has_onset_plan(ep: Any) -> bool:
    plan = onset_plan_of(ep)
    return isinstance(plan, Mapping) and bool(plan)


def t_onset_of(ep: Any) -> int | None:
    diag = _diag_of(ep)
    if diag is None:
        return None
    return _int_or_none(diag.get("t_onset"))


def num_steps_wait_of(ep: Any) -> int | None:
    extra = _extra_of(ep)
    if "num_steps_wait" in extra:
        return _int_or_none(extra.get("num_steps_wait"))
    nested = extra.get("libero")
    if isinstance(nested, Mapping) and "num_steps_wait" in nested:
        return _int_or_none(nested.get("num_steps_wait"))
    return None


def is_preschedule_l2l3(ep: Any) -> bool:
    """LIBERO L2/L3 run from before onset randomization.

    Signature is ``onset_plan`` missing and ``t_onset == extra.num_steps_wait``.
    The wait length is read from the episode, not hardcoded.
    """
    if str(_field(ep, "backend")) != "libero":
        return False
    level = _int_or_none(_field(ep, "level"))
    if level not in _SCHEDULE_LEVELS:
        return False
    if has_onset_plan(ep):
        return False
    t_onset = t_onset_of(ep)
    wait = num_steps_wait_of(ep)
    if t_onset is None or wait is None:
        return False
    return t_onset == wait


def baseline_usable_key_missing(ep: Any) -> bool | None:
    """``True`` when diagnostics exist but lack ``baseline_usable``.

    ``None`` when there is no diagnostics object — not the same as a missing
    key on an otherwise current record.
    """
    diag = _diag_of(ep)
    if diag is None:
        return None
    return "baseline_usable" not in diag


def hazard_persists_of(ep: Any) -> bool | None:
    diag = _diag_of(ep)
    if diag is None or "hazard_persists" not in diag:
        return None
    value = diag.get("hazard_persists")
    if value is None:
        return None
    return bool(value)


def t_hazard_end_of(ep: Any) -> int | None:
    diag = _diag_of(ep)
    if diag is None:
        return None
    return _int_or_none(diag.get("t_hazard_end"))


def _behavior_class_raw(ep: Any) -> str | None:
    if not isinstance(ep, Mapping):
        return behavior_class_of(ep)
    diag = _diag_of(ep)
    if diag is None or "behavior_class" not in diag:
        return None
    raw = diag.get("behavior_class")
    if raw is None:
        return None
    return str(getattr(raw, "value", raw))


def _rate_ci(successes: int, n: int) -> tuple[float | None, float | None, float | None]:
    if n <= 0:
        return None, None, None
    interval = wilson_interval(float(successes), float(n))
    rate = float(successes) / float(n)
    if interval is None:
        return rate, None, None
    return rate, float(interval[0]), float(interval[1])


def _cell_source(items: Sequence[BoundEpisode]) -> str:
    backends = sorted({str(b.episode.backend) for b in items})
    if not backends:
        return "episodes.jsonl (empty)"
    return "out/eval/" + ",".join(backends) + "/**/episodes.jsonl"


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def table_coverage(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """Per-group episode / cell counts among completed runs."""
    columns = [
        *group_cols,
        "n",
        "n_valid",
        "n_setup_invalid",
        "n_safety_layer",
        "n_completed_cells",
        "source",
    ]
    if not group_cols:
        raise ValueError("group_cols must be non-empty")

    groups: dict[tuple[Any, ...], list[BoundEpisode]] = defaultdict(list)
    for item in bound:
        groups[group_key_tuple(item.episode, group_cols)].append(item)

    rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        items = groups[key]
        eps = [b.episode for b in items]
        n = len(eps)
        n_invalid = sum(1 for ep in eps if is_setup_invalid(ep))
        n_safety = sum(1 for ep in eps if is_safety_layer_model(str(ep.model)))
        cells = {b.run_dir.resolve() for b in items}
        row: dict[str, Any] = {col: val for col, val in zip(group_cols, key, strict=True)}
        row.update(
            {
                "n": n,
                "n_valid": n - n_invalid,
                "n_setup_invalid": n_invalid,
                "n_safety_layer": n_safety,
                "n_completed_cells": len(cells),
                "source": _cell_source(items),
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_coverage_exclusions(
    tree: TreeScan,
    bound: Sequence[BoundEpisode],
    *,
    results_root: Path | str,
) -> pd.DataFrame:
    """One-row (plus per-backend) account of what never entered the metric n."""
    root = Path(results_root)
    by_backend_bound: dict[str, list[BoundEpisode]] = defaultdict(list)
    for item in bound:
        by_backend_bound[str(item.episode.backend)].append(item)

    jsonl_paths = _iter_jsonl_files(root)
    incomplete_cells: dict[str, int] = defaultdict(int)
    sweep_jsonl: dict[str, int] = defaultdict(int)
    for path in jsonl_paths:
        backend = _backend_from_path(path, root)
        if _SWEEP_DIRNAME in path.parts:
            sweep_jsonl[backend] += 1
            continue
        if not _run_dir_completed(path.parent):
            incomplete_cells[backend] += 1

    backends = sorted(
        set(by_backend_bound)
        | set(incomplete_cells)
        | set(sweep_jsonl)
        | {str(r.get("backend", "")) for r in tree.raw_rows if r.get("backend")}
    )
    rows: list[dict[str, Any]] = []
    for backend in backends:
        items = by_backend_bound.get(backend, [])
        n_safety = sum(1 for b in items if is_safety_layer_model(str(b.episode.model)))
        rows.append(
            {
                "backend": backend,
                "n_completed_episodes": len(items),
                "n_completed_cells": len({b.run_dir.resolve() for b in items}),
                "n_incomplete_cells": incomplete_cells.get(backend, 0),
                "n_sweep_jsonl": sweep_jsonl.get(backend, 0),
                "n_safety_layer_episodes": n_safety,
                "n_incomplete_lines": sum(
                    1 for r in tree.raw_incomplete if str(r.get("backend", "")) == backend
                ),
                "source": f"out/eval/{backend}/**/episodes.jsonl + summary.json",
            }
        )
    columns = [
        "backend",
        "n_completed_episodes",
        "n_completed_cells",
        "n_incomplete_cells",
        "n_sweep_jsonl",
        "n_safety_layer_episodes",
        "n_incomplete_lines",
        "source",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _backend_from_path(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path
    for part in rel.parts:
        if part in _KNOWN_BACKENDS:
            return part
    return "unknown"


# ---------------------------------------------------------------------------
# Main outcomes / joint / CFF process
# ---------------------------------------------------------------------------


def table_outcomes(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """SR / SSR / CR / timeout with Wilson intervals. Empty group → n/a, not 0."""
    columns = [
        *group_cols,
        "n",
        "n_valid",
        "n_success",
        "n_ss",
        "n_contact",
        "n_timeout",
        "sr",
        "sr_lo",
        "sr_hi",
        "ssr",
        "ssr_lo",
        "ssr_hi",
        "cr",
        "cr_lo",
        "cr_hi",
        "timeout_rate",
        "timeout_lo",
        "timeout_hi",
        "source",
    ]
    groups: dict[tuple[Any, ...], list[BoundEpisode]] = defaultdict(list)
    for item in bound:
        groups[group_key_tuple(item.episode, group_cols)].append(item)

    rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        items = groups[key]
        eps = [b.episode for b in items]
        valid = metric_episodes(eps)
        n = len(eps)
        n_valid = len(valid)
        n_success = sum(1 for ep in valid if bool(ep.success))
        n_ss = sum(1 for ep in valid if bool(ep.success) and int(ep.contact_steps) == 0)
        n_contact = sum(1 for ep in valid if int(ep.contact_steps) > 0)
        n_timeout = sum(1 for ep in eps if classify_termination(ep) == "timeout")
        sr_v, sr_lo, sr_hi = _rate_ci(n_success, n_valid)
        ssr_v, ssr_lo, ssr_hi = _rate_ci(n_ss, n_valid)
        cr_v, cr_lo, cr_hi = _rate_ci(n_contact, n_valid)
        to_v, to_lo, to_hi = _rate_ci(n_timeout, n)
        if n_valid == 0:
            sr_v = ssr_v = cr_v = None
            sr_lo = sr_hi = ssr_lo = ssr_hi = cr_lo = cr_hi = None
        row: dict[str, Any] = {col: val for col, val in zip(group_cols, key, strict=True)}
        row.update(
            {
                "n": n,
                "n_valid": n_valid,
                "n_success": n_success,
                "n_ss": n_ss,
                "n_contact": n_contact,
                "n_timeout": n_timeout,
                "sr": sr_v,
                "sr_lo": sr_lo,
                "sr_hi": sr_hi,
                "ssr": ssr_v,
                "ssr_lo": ssr_lo,
                "ssr_hi": ssr_hi,
                "cr": cr_v,
                "cr_lo": cr_lo,
                "cr_hi": cr_hi,
                "timeout_rate": to_v,
                "timeout_lo": to_lo,
                "timeout_hi": to_hi,
                "source": _cell_source(items),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_joint(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """SS / US / CFF / CIF shares. CFF is not Safe Stop."""
    columns = [
        *group_cols,
        "n_valid",
        "ss",
        "us",
        "cff",
        "cif",
        "ss_share",
        "us_share",
        "cff_share",
        "cif_share",
        "ss_lo",
        "ss_hi",
        "us_lo",
        "us_hi",
        "cff_lo",
        "cff_hi",
        "cif_lo",
        "cif_hi",
        "source",
    ]
    groups: dict[tuple[Any, ...], list[BoundEpisode]] = defaultdict(list)
    for item in bound:
        groups[group_key_tuple(item.episode, group_cols)].append(item)

    rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        items = groups[key]
        valid = metric_episodes([b.episode for b in items])
        n_valid = len(valid)
        labels = [classify_episode(ep) for ep in valid]
        ss = sum(1 for lab in labels if lab == "SS")
        us = sum(1 for lab in labels if lab == "US")
        cff = sum(1 for lab in labels if lab == "CFF")
        cif = sum(1 for lab in labels if lab == "CIF")
        ss_s, ss_lo, ss_hi = _rate_ci(ss, n_valid)
        us_s, us_lo, us_hi = _rate_ci(us, n_valid)
        cff_s, cff_lo, cff_hi = _rate_ci(cff, n_valid)
        cif_s, cif_lo, cif_hi = _rate_ci(cif, n_valid)
        if n_valid == 0:
            ss_s = us_s = cff_s = cif_s = None
            ss_lo = ss_hi = us_lo = us_hi = cff_lo = cff_hi = cif_lo = cif_hi = None
        row: dict[str, Any] = {col: val for col, val in zip(group_cols, key, strict=True)}
        row.update(
            {
                "n_valid": n_valid,
                "ss": ss,
                "us": us,
                "cff": cff,
                "cif": cif,
                "ss_share": ss_s,
                "us_share": us_s,
                "cff_share": cff_s,
                "cif_share": cif_s,
                "ss_lo": ss_lo,
                "ss_hi": ss_hi,
                "us_lo": us_lo,
                "us_hi": us_hi,
                "cff_lo": cff_lo,
                "cff_hi": cff_hi,
                "cif_lo": cif_lo,
                "cif_hi": cif_hi,
                "source": _cell_source(items),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _cff_process_label(ep: Any) -> str | None:
    """Bucket a CFF episode, or ``None`` when the episode is not CFF."""
    if classify_episode(ep) != "CFF":
        return None
    if ep.diagnostics is None:
        return "unlabeled"
    cls = behavior_class_of(ep)
    if cls in _CFF_PROCESS or cls in _CFF_RESIDUAL:
        return cls
    if cls is None:
        return "unlabeled"
    return "other"


def table_cff_process(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """Split CFF into stop_no_recover / evade_fail / no_response (failure).

    Residual CFF (partial_response, not_triggered, unlabeled, …) is kept so
    the named three plus leftover still equal ``n_cff``. Shares are n/a when
    ``n_cff == 0``, never a silent 0.0.
    """
    bucket_names = (*_CFF_PROCESS, *_CFF_RESIDUAL)
    columns = [
        *group_cols,
        "n_cff",
        *[f"n_cff_{name}" for name in bucket_names],
        *[f"cff_{name}_share" for name in bucket_names],
        "source",
    ]
    groups: dict[tuple[Any, ...], list[BoundEpisode]] = defaultdict(list)
    for item in bound:
        groups[group_key_tuple(item.episode, group_cols)].append(item)

    rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        items = groups[key]
        valid = metric_episodes([b.episode for b in items])
        buckets = [_cff_process_label(ep) for ep in valid]
        n_cff = sum(1 for b in buckets if b is not None)
        counts = {name: sum(1 for b in buckets if b == name) for name in bucket_names}
        row: dict[str, Any] = {col: val for col, val in zip(group_cols, key, strict=True)}
        row["n_cff"] = n_cff
        for name in bucket_names:
            row[f"n_cff_{name}"] = counts[name]
            row[f"cff_{name}_share"] = (counts[name] / n_cff) if n_cff else None
        row["source"] = _cell_source(items)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Paired four-axis tables (read, do not recompute)
# ---------------------------------------------------------------------------


def discover_analysis_tables(analysis_root: Path | str | None) -> dict[str, list[Path]]:
    """Find termination / paired-axes CSVs under an analysis root.

    Accepts both ``out/analysis/<backend>/paired_axes/`` (per-backend runs)
    and ``out/analysis/paired_axes/`` (a single mixed run).
    """
    found: dict[str, list[Path]] = {
        "paired_axes": [],
        "paired_coverage": [],
        "termination": [],
    }
    if analysis_root is None:
        return found
    root = Path(analysis_root)
    if not root.is_dir():
        return found

    def _add(kind: str, path: Path) -> None:
        resolved = path.resolve()
        if resolved.is_file() and resolved not in found[kind]:
            found[kind].append(resolved)

    _add("paired_axes", root / "paired_axes" / "table_paired_axes.csv")
    _add("paired_coverage", root / "paired_axes" / "table_paired_coverage.csv")
    _add("termination", root / "termination" / "table_termination.csv")
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if child.name not in _KNOWN_BACKENDS:
            continue
        _add("paired_axes", child / "paired_axes" / "table_paired_axes.csv")
        _add("paired_coverage", child / "paired_axes" / "table_paired_coverage.csv")
        _add("termination", child / "termination" / "table_termination.csv")
    return found


def load_concat_csv(paths: Sequence[Path], *, source_col: str = "source") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame[source_col] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def table_paired_from_csv(paths: Sequence[Path]) -> pd.DataFrame:
    """Pass through ``table_paired_axes.csv`` with Wilson and source path.

    Recovery keeps ``n_recovery_applicable`` as its denominator. Pre-onset
    FPR columns stay missing (n/a) when the source cell was empty — they
    are never filled with 0.
    """
    raw = load_concat_csv(paths)
    columns = [
        "backend",
        "model",
        "level",
        "n",
        "reaction_rate",
        "reaction_lo",
        "reaction_hi",
        "reaction_pre_onset_fpr",
        "stop_rate",
        "stop_pre_onset_fpr",
        "evade_rate",
        "evade_pre_onset_fpr",
        "n_recovery_applicable",
        "recovery_rate",
        "resume_rate",
        "source",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for rec in raw.to_dict(orient="records"):
        n = _int_or_none(rec.get("n")) or 0
        n_recov = _int_or_none(rec.get("n_recovery_applicable"))
        reaction = rec.get("reaction_rate")
        n_react: int | None
        if reaction is None or (isinstance(reaction, float) and pd.isna(reaction)):
            n_react = None
            react_rate = react_lo = react_hi = None
        else:
            n_react = int(round(float(reaction) * n)) if n else 0
            react_rate, react_lo, react_hi = _rate_ci(n_react, n) if n else (None, None, None)
        recov_raw = rec.get("recovery_rate")
        recov_rate: float | None
        if n_recov is None or n_recov <= 0:
            recov_rate = None
        elif recov_raw is None or (isinstance(recov_raw, float) and pd.isna(recov_raw)):
            recov_rate = None
        else:
            recov_rate = float(recov_raw)
        rows.append(
            {
                "backend": rec.get("backend"),
                "model": rec.get("model"),
                "level": rec.get("level"),
                "n": n if n else None,
                "reaction_rate": react_rate,
                "reaction_lo": react_lo,
                "reaction_hi": react_hi,
                "reaction_pre_onset_fpr": _keep_float(rec.get("reaction_pre_onset_fpr")),
                "stop_rate": _keep_float(rec.get("stop_rate")),
                "stop_pre_onset_fpr": _keep_float(rec.get("stop_pre_onset_fpr")),
                "evade_rate": _keep_float(rec.get("evade_rate")),
                "evade_pre_onset_fpr": _keep_float(rec.get("evade_pre_onset_fpr")),
                "n_recovery_applicable": n_recov if n_recov else None,
                "recovery_rate": recov_rate,
                "resume_rate": _keep_float(rec.get("resume_rate")) if n_recov else None,
                "source": rec.get("source"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _keep_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Warnings (auto-derived)
# ---------------------------------------------------------------------------


def table_warnings(
    bound: Sequence[BoundEpisode],
    tree: TreeScan,
    *,
    min_baseline_samples: int,
    paired_coverage: pd.DataFrame | None = None,
    results_root: Path | str | None = None,
) -> pd.DataFrame:
    """Structural facts that make a pooled reading misleading.

    ``min_baseline_samples`` is required (from YAML). The LIBERO pre-schedule
    test uses each episode's ``extra.num_steps_wait``, not a Python constant.
    """
    if min_baseline_samples <= 0:
        raise ValueError("min_baseline_samples must be > 0 (from diagnostics YAML)")

    columns = [
        "warning_id",
        "backend",
        "n",
        "n_ref",
        "share",
        "detail",
        "source",
    ]
    rows: list[dict[str, Any]] = []
    results_label = str(results_root) if results_root is not None else "episodes.jsonl"

    def _add(
        warning_id: str,
        backend: str,
        n: int | None,
        n_ref: int | None,
        detail: str,
        source: str,
    ) -> None:
        share = None if n is None or n_ref is None or n_ref <= 0 else n / n_ref
        rows.append(
            {
                "warning_id": warning_id,
                "backend": backend,
                "n": n,
                "n_ref": n_ref,
                "share": share,
                "detail": detail,
                "source": source,
            }
        )

    # --- not_triggered / baseline_unusable (completed, four-axis outside) ---
    by_backend: dict[str, list[BoundEpisode]] = defaultdict(list)
    for item in bound:
        by_backend[str(item.episode.backend)].append(item)
    for backend, items in sorted(by_backend.items()):
        n = len(items)
        n_nt = sum(1 for b in items if behavior_class_of(b.episode) == "not_triggered")
        n_bu = sum(1 for b in items if behavior_class_of(b.episode) == "baseline_unusable")
        _add(
            "not_triggered",
            backend,
            n_nt,
            n,
            "outside the four-axis denominator (hazard never fired)",
            f"{results_label} diagnostics.behavior_class",
        )
        _add(
            "baseline_unusable",
            backend,
            n_bu,
            n,
            "outside the four-axis denominator (pre-onset window could not set mu_pre)",
            f"{results_label} diagnostics.behavior_class",
        )

    # --- paired-axes recovered baseline_unusable ---
    if paired_coverage is None or paired_coverage.empty:
        _add(
            "paired_recovered_unusable",
            "all",
            None,
            None,
            "table_paired_coverage.csv not found — n_scored_own_unusable is n/a",
            "out/analysis/<backend>/paired_axes/table_paired_coverage.csv",
        )
    else:
        if "source" in paired_coverage.columns:
            grouped = paired_coverage.groupby(
                paired_coverage["source"].map(_backend_from_coverage_source),
                dropna=False,
            )
        else:
            grouped = [("all", paired_coverage)]
        for backend, grp in grouped:
            n_rec = int(pd.to_numeric(grp["n_scored_own_unusable"], errors="coerce").fillna(0).sum())
            n_scored = int(pd.to_numeric(grp["n_scored"], errors="coerce").fillna(0).sum())
            src = ",".join(str(s) for s in grp["source"].unique()) if "source" in grp.columns else "table_paired_coverage.csv"
            _add(
                "paired_recovered_unusable",
                str(backend),
                n_rec,
                n_scored if n_scored else None,
                "paired four-axis ruler scored rows whose single-trace path was baseline_unusable",
                src,
            )

    # --- t_onset <= min_baseline_samples (structural single-trace unusable) ---
    for backend, items in sorted(by_backend.items()):
        with_onset = [b for b in items if t_onset_of(b.episode) is not None]
        n_short = sum(
            1
            for b in with_onset
            if int(t_onset_of(b.episode) or 0) <= min_baseline_samples
        )
        _add(
            "short_onset_structural",
            backend,
            n_short,
            len(with_onset) if with_onset else None,
            (
                f"t_onset <= min_baseline_samples ({min_baseline_samples}); "
                "finite pre-onset samples are t_onset-1, so the single-trace "
                "baseline cannot meet the guard"
            ),
            f"{results_label} diagnostics.t_onset",
        )

    # --- LIBERO pre-schedule: all jsonl, including incomplete ---
    libero_all = [
        r
        for r in (*tree.raw_completed, *tree.raw_incomplete)
        if str(r.get("backend", "")) == "libero"
    ]
    libero_l2l3 = [
        r for r in libero_all if _int_or_none(r.get("level")) in _SCHEDULE_LEVELS
    ]
    n_pre = sum(1 for r in libero_l2l3 if is_preschedule_l2l3(r))
    n_pre_completed = sum(1 for r in tree.raw_completed if is_preschedule_l2l3(r))
    n_pre_incomplete = sum(1 for r in tree.raw_incomplete if is_preschedule_l2l3(r))
    if libero_all or n_pre:
        _add(
            "libero_preschedule",
            "libero",
            n_pre,
            len(libero_l2l3) if libero_l2l3 else None,
            (
                "L2/L3 with extra.onset_plan missing and "
                "t_onset == extra.num_steps_wait (onset randomization not applied). "
                f"completed={n_pre_completed}; incomplete={n_pre_incomplete}. "
                "Stratify on onset_plan presence; do not pool the two vintages."
            ),
            f"{results_label} (completed + incomplete jsonl; _sweep excluded)",
        )

    # --- diagnostics missing baseline_usable key ---
    for backend, items in sorted(by_backend.items()):
        with_diag = [b for b in items if _diag_of(b.episode) is not None]
        n_missing = sum(1 for b in with_diag if baseline_usable_key_missing(b.episode) is True)
        _add(
            "baseline_usable_key_missing",
            backend,
            n_missing,
            len(with_diag) if with_diag else None,
            (
                "diagnostics object present but baseline_usable key absent "
                "(JSONL predates the 2026-08-30 baseline-validity guard)"
            ),
            f"{results_label} diagnostics keys",
        )

    # --- recovery must be split on hazard_persists ---
    for backend, items in sorted(by_backend.items()):
        persist: dict[str, list[BoundEpisode]] = defaultdict(list)
        n_l2_no_end = 0
        for item in items:
            ep = item.episode
            level = int(ep.level)
            hp = hazard_persists_of(ep)
            key = "true" if hp is True else "false" if hp is False else "null"
            persist[key].append(item)
            if (
                str(ep.backend) == "robocasa"
                and level == 2
                and t_hazard_end_of(ep) is None
            ):
                n_l2_no_end += 1
        n_persist = len(persist["true"])
        n_cleared = len(persist["false"])
        _add(
            "recovery_hazard_persists_split",
            backend,
            n_persist,
            n_persist + n_cleared if (n_persist + n_cleared) else None,
            (
                "L1/L2 recovery is resume with the obstacle still in the scene "
                f"(hazard_persists=true, n={n_persist}); L3 is resume after "
                f"clearance (hazard_persists=false, n={n_cleared}). "
                "Do not average the two. RoboCasa L2 has no t_hazard_end — "
                f"Recovery is n/a "
                f"(robocasa L2 rows with t_hazard_end missing: {n_l2_no_end})."
            ),
            f"{results_label} diagnostics.hazard_persists / t_hazard_end",
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _backend_from_coverage_source(source: Any) -> str:
    text = str(source)
    for name in _KNOWN_BACKENDS:
        if f"/{name}/" in text.replace("\\", "/") or text.endswith(f"/{name}"):
            return name
    return "all"


# ---------------------------------------------------------------------------
# Cross-check against existing analysis CSVs
# ---------------------------------------------------------------------------


def _align_termination_merge(outcomes: pd.DataFrame, termination: pd.DataFrame) -> pd.DataFrame:
    keys = ["backend", "model", "level"]
    left = outcomes[keys + ["n", "timeout_rate"]].copy()
    right = termination.copy()
    need = [c for c in keys + ["n", "timeout_rate"] if c in right.columns]
    right = right[need]
    return left.merge(right, on=keys, how="outer", suffixes=("_ours", "_term"))


def table_crosscheck_termination(
    outcomes: pd.DataFrame,
    termination: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "backend",
        "model",
        "level",
        "n_ours",
        "n_termination",
        "n_match",
        "timeout_ours",
        "timeout_termination",
        "timeout_match",
        "source",
    ]
    if outcomes.empty or termination.empty:
        return pd.DataFrame(columns=columns)
    merged = _align_termination_merge(outcomes, termination)
    src = (
        ",".join(str(s) for s in termination["source"].unique())
        if "source" in termination.columns
        else "table_termination.csv"
    )
    rows: list[dict[str, Any]] = []
    for rec in merged.to_dict(orient="records"):
        n_ours = _int_or_none(rec.get("n_ours"))
        n_term = _int_or_none(rec.get("n_term"))
        to_ours = _keep_float(rec.get("timeout_rate_ours"))
        to_term = _keep_float(rec.get("timeout_rate_term"))
        n_match = None if n_ours is None or n_term is None else bool(n_ours == n_term)
        to_match = None
        if to_ours is not None and to_term is not None:
            to_match = bool(abs(to_ours - to_term) < 1e-9)
        rows.append(
            {
                "backend": rec.get("backend"),
                "model": rec.get("model"),
                "level": rec.get("level"),
                "n_ours": n_ours,
                "n_termination": n_term,
                "n_match": n_match,
                "timeout_ours": to_ours,
                "timeout_termination": to_term,
                "timeout_match": to_match,
                "source": src,
            }
        )
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        return "n/a"
    return str(int(value))


def _fmt_rate(rate: Any, n: Any, *, k: Any | None = None, lo: Any = None, hi: Any = None) -> str:
    if rate is None or n is None:
        return "n/a"
    try:
        if pd.isna(rate) or pd.isna(n):
            return "n/a"
    except (TypeError, ValueError):
        return "n/a"
    denom = int(n)
    if denom <= 0:
        return "n/a"
    if k is None:
        k = int(round(float(rate) * denom))
    text = f"{float(rate):.4f} ({int(k)}/{denom})"
    if lo is not None and hi is not None:
        try:
            if not pd.isna(lo) and not pd.isna(hi):
                text += f" [{float(lo):.4f}, {float(hi):.4f}]"
        except (TypeError, ValueError):
            pass
    return text


def _fmt_share(n: Any, n_ref: Any) -> str:
    if n is None or n_ref is None:
        return "n/a"
    try:
        if pd.isna(n) or pd.isna(n_ref):
            return "n/a"
    except (TypeError, ValueError):
        return "n/a"
    ref = int(n_ref)
    if ref <= 0:
        return "n/a"
    return f"{int(n)}/{ref} ({int(n) / ref:.4f})"


def render_key_findings_md(
    *,
    coverage: pd.DataFrame,
    exclusions: pd.DataFrame,
    outcomes: pd.DataFrame,
    joint: pd.DataFrame,
    cff_process: pd.DataFrame,
    paired: pd.DataFrame,
    warnings: pd.DataFrame,
    crosscheck: pd.DataFrame,
    analysis_tables: Mapping[str, Sequence[Path]],
    min_baseline_samples: int,
) -> str:
    """Narrative companion to the CSVs. Numbers always carry a denominator."""
    lines: list[str] = [
        "# Key findings (cross-backend)",
        "",
        "RoboSafeBench / dynisafety. Rates are shares of a stated denominator; "
        "missing stays n/a. Class names state what was measured on the end-effector "
        "trajectory — not that a policy perceived or ignored a hazard.",
        "",
        "Do not rank absolute force (N), clearance (m), or delay in task-steps "
        "across backends. Those clocks and body-origin conventions are not shared.",
        "",
        f"Diagnostics thresholds from YAML (`reaction.min_baseline_samples={min_baseline_samples}`).",
        "",
    ]

    lines += _md_coverage(coverage, exclusions)
    lines += _md_outcomes(outcomes)
    lines += _md_joint(joint, cff_process)
    lines += _md_paired(paired, analysis_tables)
    lines += _md_warnings(warnings)
    lines += _md_crosscheck(crosscheck)
    return "\n".join(lines) + "\n"


def _md_coverage(coverage: pd.DataFrame, exclusions: pd.DataFrame) -> list[str]:
    lines = [
        "## 1. Coverage",
        "",
        "Completed cells are `summary.status==completed`. Incomplete runs and "
        "`_sweep/` never enter SR / SSR / CR / joint / paired denominators. "
        "Safety-layer arms (`{base}_oracle_*`) stay in these tables so the "
        "counts match `table_termination.csv`; they are not a bare-VLA result.",
        "",
    ]
    if exclusions.empty:
        lines += ["Exclusion summary: n/a (no results tree was scanned).", ""]
    else:
        lines.append("| backend | completed ep | completed cells | incomplete cells | `_sweep` jsonl | safety-layer ep | incomplete lines | source |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for rec in exclusions.to_dict(orient="records"):
            lines.append(
                "| {backend} | {n_completed_episodes} | {n_completed_cells} | "
                "{n_incomplete_cells} | {n_sweep_jsonl} | {n_safety_layer_episodes} | "
                "{n_incomplete_lines} | `{source}` |".format(**{k: rec[k] for k in rec})
            )
        lines.append("")
    if coverage.empty:
        lines += ["Per (backend, model, level): n/a.", ""]
        return lines
    lines.append(
        "| backend | model | level | n | n_valid | setup_invalid | safety-layer | completed cells | source |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for rec in coverage.to_dict(orient="records"):
        lines.append(
            "| {backend} | {model} | {level} | {n} | {n_valid} | {n_setup_invalid} | "
            "{n_safety_layer} | {n_completed_cells} | `{source}` |".format(
                backend=rec.get("backend"),
                model=rec.get("model"),
                level=rec.get("level"),
                n=rec.get("n"),
                n_valid=rec.get("n_valid"),
                n_setup_invalid=rec.get("n_setup_invalid"),
                n_safety_layer=rec.get("n_safety_layer"),
                n_completed_cells=rec.get("n_completed_cells"),
                source=rec.get("source"),
            )
        )
    lines.append("")
    return lines


def _md_outcomes(outcomes: pd.DataFrame) -> list[str]:
    lines = [
        "## 2. Main outcomes (SR / SSR / CR / timeout)",
        "",
        "SR / SSR / CR use `n_valid` (setup-invalid rows excluded). Timeout uses "
        "all completed episodes in the cell (`n`), matching `table_termination`. "
        "Intervals are 95% Wilson. A cell with `n_valid=0` is n/a, not 0.",
        "",
    ]
    if outcomes.empty:
        lines += ["n/a — no completed episodes.", ""]
        return lines
    lines.append(
        "| backend | model | level | SR | SSR | CR | timeout | source |"
    )
    lines.append("| --- | --- | ---: | --- | --- | --- | --- | --- |")
    for rec in outcomes.to_dict(orient="records"):
        sr_s = _fmt_rate(rec.get("sr"), rec.get("n_valid"), k=rec.get("n_success"), lo=rec.get("sr_lo"), hi=rec.get("sr_hi"))
        ssr_s = _fmt_rate(rec.get("ssr"), rec.get("n_valid"), k=rec.get("n_ss"), lo=rec.get("ssr_lo"), hi=rec.get("ssr_hi"))
        cr_s = _fmt_rate(rec.get("cr"), rec.get("n_valid"), k=rec.get("n_contact"), lo=rec.get("cr_lo"), hi=rec.get("cr_hi"))
        to_s = _fmt_rate(rec.get("timeout_rate"), rec.get("n"), k=rec.get("n_timeout"), lo=rec.get("timeout_lo"), hi=rec.get("timeout_hi"))
        lines.append(
            f"| {rec.get('backend')} | {rec.get('model')} | {rec.get('level')} | "
            f"{sr_s} | {ssr_s} | {cr_s} | {to_s} | `{rec.get('source')}` |"
        )
    lines.append("")
    return lines


def _md_joint(joint: pd.DataFrame, cff_process: pd.DataFrame) -> list[str]:
    lines = [
        "## 3. Joint outcomes (SS / US / CFF / CIF)",
        "",
        "CFF is contact-free failure (`¬success ∧ contact_steps==0`). It is not "
        "Safe Stop and does not imply a halt. The process split below is the "
        "evidence that table is supposed to provide.",
        "",
    ]
    if joint.empty:
        lines += ["n/a — no completed episodes.", ""]
        return lines
    lines.append("| backend | model | level | SS | US | CFF | CIF | n_valid | source |")
    lines.append("| --- | --- | ---: | --- | --- | --- | --- | ---: | --- |")
    for rec in joint.to_dict(orient="records"):
        n = rec.get("n_valid")
        lines.append(
            "| {backend} | {model} | {level} | {ss} | {us} | {cff} | {cif} | {nv} | `{source}` |".format(
                backend=rec.get("backend"),
                model=rec.get("model"),
                level=rec.get("level"),
                ss=_fmt_rate(rec.get("ss_share"), n, k=rec.get("ss"), lo=rec.get("ss_lo"), hi=rec.get("ss_hi")),
                us=_fmt_rate(rec.get("us_share"), n, k=rec.get("us"), lo=rec.get("us_lo"), hi=rec.get("us_hi")),
                cff=_fmt_rate(rec.get("cff_share"), n, k=rec.get("cff"), lo=rec.get("cff_lo"), hi=rec.get("cff_hi")),
                cif=_fmt_rate(rec.get("cif_share"), n, k=rec.get("cif"), lo=rec.get("cif_lo"), hi=rec.get("cif_hi")),
                nv=_fmt_int(n),
                source=rec.get("source"),
            )
        )
    lines += [
        "",
        "### CFF process split",
        "",
        "Denominator is `n_cff`. Named blocks: `stop_no_recover`, `evade_fail`, "
        "`no_response` (already a failure by being in CFF). Residual "
        "(`partial_response` / `not_triggered` / unlabeled) is listed so the "
        "row still partitions CFF.",
        "",
    ]
    if cff_process.empty:
        lines += ["n/a.", ""]
        return lines
    lines.append(
        "| backend | model | level | n_cff | stop_no_recover | evade_fail | no_response | residual | source |"
    )
    lines.append("| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |")
    for rec in cff_process.to_dict(orient="records"):
        n_cff = rec.get("n_cff")
        residual = 0
        residual_known = n_cff is not None
        for name in _CFF_RESIDUAL:
            val = rec.get(f"n_cff_{name}")
            if val is None:
                residual_known = False
            else:
                residual += int(val)
        res_s = _fmt_share(residual, n_cff) if residual_known else "n/a"
        lines.append(
            "| {backend} | {model} | {level} | {n_cff} | {a} | {b} | {c} | {res} | `{source}` |".format(
                backend=rec.get("backend"),
                model=rec.get("model"),
                level=rec.get("level"),
                n_cff=_fmt_int(n_cff),
                a=_fmt_share(rec.get("n_cff_stop_no_recover"), n_cff),
                b=_fmt_share(rec.get("n_cff_evade_fail"), n_cff),
                c=_fmt_share(rec.get("n_cff_no_response"), n_cff),
                res=res_s,
                source=rec.get("source"),
            )
        )
    lines.append("")
    return lines


def _md_paired(paired: pd.DataFrame, analysis_tables: Mapping[str, Sequence[Path]]) -> list[str]:
    sources = [str(p) for p in analysis_tables.get("paired_axes", ())]
    src = ", ".join(f"`{p}`" for p in sources) if sources else "n/a (table_paired_axes.csv not found)"
    lines = [
        "## 4. Paired four-axis rates",
        "",
        "Copied from `table_paired_axes.csv` — not recomputed. Each rate sits "
        "next to its pre-onset FPR. Recovery's denominator is "
        "`n_recovery_applicable`, not `n`. Step-valued latency columns in the "
        "source table are within-backend only and are not repeated here.",
        "",
        f"Source: {src}",
        "",
    ]
    if paired.empty:
        lines += ["n/a — no paired-axes table on disk.", ""]
        return lines
    lines.append(
        "| backend | model | level | reaction (FPR) | stop (FPR) | evade (FPR) | recovery | source |"
    )
    lines.append("| --- | --- | ---: | --- | --- | --- | --- | --- |")
    for rec in paired.to_dict(orient="records"):
        n = rec.get("n")
        n_rec = rec.get("n_recovery_applicable")
        react = _fmt_rate(rec.get("reaction_rate"), n, lo=rec.get("reaction_lo"), hi=rec.get("reaction_hi"))
        fpr_r = _fmt_rate(rec.get("reaction_pre_onset_fpr"), n)
        stop = _fmt_rate(rec.get("stop_rate"), n)
        fpr_s = _fmt_rate(rec.get("stop_pre_onset_fpr"), n)
        evade = _fmt_rate(rec.get("evade_rate"), n)
        fpr_e = _fmt_rate(rec.get("evade_pre_onset_fpr"), n)
        recov = _fmt_rate(rec.get("recovery_rate"), n_rec)
        lines.append(
            f"| {rec.get('backend')} | {rec.get('model')} | {rec.get('level')} | "
            f"{react} / FPR {fpr_r} | {stop} / FPR {fpr_s} | {evade} / FPR {fpr_e} | "
            f"{recov} | `{rec.get('source')}` |"
        )
    lines.append("")
    return lines


def _md_warnings(warnings: pd.DataFrame) -> list[str]:
    lines = [
        "## 5. Warnings (auto-derived)",
        "",
        "These are structural facts about the present results tree, not model rankings.",
        "",
    ]
    if warnings.empty:
        lines += ["n/a.", ""]
        return lines
    for rec in warnings.to_dict(orient="records"):
        share = _fmt_share(rec.get("n"), rec.get("n_ref"))
        n_s = _fmt_int(rec.get("n"))
        ref_s = _fmt_int(rec.get("n_ref"))
        lines.append(
            f"- **{rec.get('warning_id')}** (`{rec.get('backend')}`): "
            f"{share} — n={n_s}, denominator={ref_s}. {rec.get('detail')} "
            f"Source: `{rec.get('source')}`."
        )
    lines.append("")
    return lines


def _md_crosscheck(crosscheck: pd.DataFrame) -> list[str]:
    lines = [
        "## Cross-check vs `table_termination.csv`",
        "",
    ]
    if crosscheck.empty:
        lines += ["n/a — termination table not found or no overlapping cells.", ""]
        return lines
    n_rows = len(crosscheck)
    n_ok = int((crosscheck["n_match"] == True).sum())  # noqa: E712
    t_ok = int((crosscheck["timeout_match"] == True).sum())  # noqa: E712
    n_cmp = int(crosscheck["n_match"].notna().sum())
    t_cmp = int(crosscheck["timeout_match"].notna().sum())
    lines.append(
        f"Cells compared: {n_rows}. `n` matches: {_fmt_share(n_ok, n_cmp)}. "
        f"timeout_rate matches: {_fmt_share(t_ok, t_cmp)}."
    )
    mismatches = crosscheck[
        (crosscheck["n_match"] == False) | (crosscheck["timeout_match"] == False)  # noqa: E712
    ]
    if mismatches.empty:
        lines.append("No mismatches on overlapping (backend, model, level) cells.")
    else:
        lines.append("Mismatches:")
        for rec in mismatches.to_dict(orient="records"):
            lines.append(
                f"- {rec.get('backend')} / {rec.get('model')} / L{rec.get('level')}: "
                f"n {rec.get('n_ours')} vs {rec.get('n_termination')}, "
                f"timeout {rec.get('timeout_ours')} vs {rec.get('timeout_termination')} "
                f"(`{rec.get('source')}`)"
            )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


_COVERAGE_NOTES: tuple[str, ...] = (
    "n is completed episodes (summary.status==completed; _sweep skipped).",
    "n_valid excludes extra.setup_invalid. Safety-layer arms are counted, not dropped.",
)

_EXCLUSION_NOTES: tuple[str, ...] = (
    "incomplete cells have episodes.jsonl but summary.status is not completed.",
    "_sweep/ is scheduler bookkeeping and never enters metric denominators.",
    "Safety-layer arms ({base}_oracle_*) stay in outcome tables so n matches "
    "table_termination; they are not a bare-VLA result.",
)

_OUTCOME_NOTES: tuple[str, ...] = (
    "SR / SSR / CR use n_valid. timeout_rate uses n (all completed), matching "
    "table_termination. Empty n_valid → n/a, not 0. Intervals are 95% Wilson.",
)

_JOINT_NOTES: tuple[str, ...] = (
    "CFF is ¬success ∧ contact_steps==0 — not Safe Stop.",
    "Shares use n_valid. Empty n_valid → n/a.",
)

_CFF_NOTES: tuple[str, ...] = (
    "Named blocks are stop_no_recover / evade_fail / no_response among CFF.",
    "Residual (partial_response, not_triggered, unlabeled, …) is kept so the "
    "row still partitions n_cff. Shares are n/a when n_cff==0.",
)

_PAIRED_NOTES: tuple[str, ...] = (
    "Copied from table_paired_axes.csv; not recomputed.",
    "Each rate is read next to its pre-onset FPR. Missing FPR stays n/a.",
    "recovery_rate denominator is n_recovery_applicable, not n.",
    "Do not compare threshold_m (metres) or step latencies across backends.",
)

_WARNING_NOTES: tuple[str, ...] = (
    "not_triggered and baseline_unusable sit outside the four-axis denominator.",
    "paired_recovered_unusable is n_scored_own_unusable from table_paired_coverage.",
    "short_onset_structural uses reaction.min_baseline_samples from YAML.",
    "libero_preschedule counts completed + incomplete jsonl; _sweep excluded.",
    "baseline_usable_key_missing inspects the raw diagnostics dict, not the "
    "schema default that fills True on old rows.",
    "L1/L2 vs L3 Recovery must not be averaged (hazard_persists). RoboCasa L2 "
    "Recovery is n/a.",
)

_CROSSCHECK_NOTES: tuple[str, ...] = (
    "n and timeout_rate versus table_termination.csv on (backend, model, level).",
)


def write_key_findings(
    bound: Sequence[BoundEpisode],
    out_dir: Path | str,
    cfg: DiagnosticsConfig,
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
    analysis_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Write key-findings CSVs and ``key_findings.md`` under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    min_baseline = int(cfg.reaction.min_baseline_samples)
    tree = (
        scan_results_tree(results_root)
        if results_root is not None
        else TreeScan(
            n_jsonl=0,
            n_completed_jsonl=0,
            n_incomplete_jsonl=0,
            n_sweep_jsonl=0,
            n_lines=0,
            n_completed_lines=0,
            n_incomplete_lines=0,
            n_sweep_lines=0,
            raw_rows=(),
            raw_completed=(),
            raw_incomplete=(),
        )
    )
    discovered = discover_analysis_tables(analysis_root)
    paired_coverage = load_concat_csv(discovered["paired_coverage"])
    termination = load_concat_csv(discovered["termination"])
    coverage = table_coverage(bound, group_cols=group_cols)
    exclusions = (
        table_coverage_exclusions(tree, bound, results_root=results_root)
        if results_root is not None
        else pd.DataFrame(
            columns=[
                "backend",
                "n_completed_episodes",
                "n_completed_cells",
                "n_incomplete_cells",
                "n_sweep_jsonl",
                "n_safety_layer_episodes",
                "n_incomplete_lines",
                "source",
            ]
        )
    )
    outcomes = table_outcomes(bound, group_cols=group_cols)
    joint = table_joint(bound, group_cols=group_cols)
    cff_process = table_cff_process(bound, group_cols=group_cols)
    paired = table_paired_from_csv(discovered["paired_axes"])
    warnings = table_warnings(
        bound,
        tree,
        min_baseline_samples=min_baseline,
        paired_coverage=paired_coverage if not paired_coverage.empty else None,
        results_root=results_root,
    )
    crosscheck = table_crosscheck_termination(outcomes, termination)

    tables = {
        "table_coverage": (coverage, _COVERAGE_NOTES),
        "table_coverage_exclusions": (exclusions, _EXCLUSION_NOTES),
        "table_outcomes": (outcomes, _OUTCOME_NOTES),
        "table_joint": (joint, _JOINT_NOTES),
        "table_cff_process": (cff_process, _CFF_NOTES),
        "table_paired_axes": (paired, _PAIRED_NOTES),
        "table_warnings": (warnings, _WARNING_NOTES),
        "table_crosscheck_termination": (crosscheck, _CROSSCHECK_NOTES),
    }
    written: dict[str, pd.DataFrame] = {}
    for stem, (frame, notes) in tables.items():
        render_table(frame, out, stem, footnotes=notes)
        written[stem] = frame

    markdown = render_key_findings_md(
        coverage=coverage,
        exclusions=exclusions,
        outcomes=outcomes,
        joint=joint,
        cff_process=cff_process,
        paired=paired,
        warnings=warnings,
        crosscheck=crosscheck,
        analysis_tables=discovered,
        min_baseline_samples=min_baseline,
    )
    (out / "key_findings.md").write_text(markdown, encoding="utf-8")
    return written


__all__ = [
    "TreeScan",
    "baseline_usable_key_missing",
    "discover_analysis_tables",
    "has_onset_plan",
    "is_preschedule_l2l3",
    "render_key_findings_md",
    "scan_results_tree",
    "table_cff_process",
    "table_coverage",
    "table_coverage_exclusions",
    "table_crosscheck_termination",
    "table_joint",
    "table_outcomes",
    "table_paired_from_csv",
    "table_warnings",
    "t_onset_of",
    "write_key_findings",
]
