"""Read-only loader for the stage-03 paired four-axis CSVs (docs/user/eval.md §2.5).

The paper body reads the four axes on the paired ruler, not the single-trace one:
the kinematic reaction detector saturates (LIBERO L0 control FPR 0.92) and
``reaction_rate`` is identically 0 for any policy whose action chunk is longer
than the noise floor, which is a property of the estimator rather than of the
policy (§2.1b / §9 limitation 1). The paired ruler also scores the episodes the
single-trace baseline guard refused (§9 limitation 8), so the two populations
have different denominators and must never be pooled.

Figures consume the CSV ``dynisafety analysis paired-axes`` already wrote rather
than re-scoring: re-scoring LIBERO takes ~96 s and needs the ``diagnostics.*``
thresholds from ``configs/base.yaml``, which this package does not read. A
missing or empty CSV yields an unavailable frame whose message names the file;
callers render that as n/a and never as 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dynisafety.analysis.paired_axes import (
    OWN_BASELINE_STATES,
    table_paired_axes,
    table_paired_classes,
)

_AXIS_GROUP_COLS: tuple[str, ...] = ("backend", "model", "level")
_CLASS_GROUP_COLS: tuple[str, ...] = ("backend", "level")

# Axis bits are non-null by construction in ``paired_axes.episode_table``; a null
# would silently read as True through ``astype(bool)``, so the file is refused.
_REQUIRED_BOOL_COLS: tuple[str, ...] = (
    "reacted",
    "stopped",
    "evaded",
    "resumed",
    "recovered",
    "recovery_applicable",
)
# n/a where the row has no pre-onset segment (``t_onset <= 1``). Dropped from the
# FPR mean by ``_mean_bool_skipna``; never counted as False.
_NULLABLE_BOOL_COLS: tuple[str, ...] = (
    "reaction_pre_onset_fp",
    "stop_pre_onset_fp",
    "evade_pre_onset_fp",
)
_REQUIRED_COLS: tuple[str, ...] = (
    "backend",
    "model",
    "level",
    "own_baseline",
    "paired_class",
    "threshold_m",
    "threshold_source",
    "mu_ref",
    "reference_window",
    "t_onset",
    "reaction_latency_steps",
    "stop_latency_steps",
    "max_heading_deg",
    *_REQUIRED_BOOL_COLS,
    *_NULLABLE_BOOL_COLS,
)
_TRUE_TOKENS = frozenset({"true", "1", "yes", "t"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "f"})


@dataclass(frozen=True)
class PairedAxesFrame:
    """Concatenated per-episode paired rows plus the provenance of each backend."""

    episodes: pd.DataFrame
    sources: tuple[Path, ...]
    missing: tuple[str, ...]
    root: Path | None

    @property
    def available(self) -> bool:
        return not self.episodes.empty

    def unavailable_message(self) -> str:
        head = "paired four-axis CSV unavailable (n/a)"
        if self.root is None:
            return f"{head}\nno analysis root configured"
        if self.missing:
            return "\n".join((head, *self.missing, "run scripts/eval/03_analysis.sh paired-axes"))
        return f"{head}\nno paired rows under {self.root}"

    def note(self) -> str:
        """One-line provenance for a figure footnote; empty when nothing is missing."""
        if not self.missing:
            return ""
        return "Missing paired CSV (rendered as n/a, not 0): " + "; ".join(self.missing)


def load_paired_axes(
    root: Path | None,
    *,
    backends: Sequence[str],
    episodes_relpath: Path,
) -> PairedAxesFrame:
    """Load ``<root>/<backend>/<episodes_relpath>`` for each backend, skipping gaps."""
    if root is None:
        return PairedAxesFrame(pd.DataFrame(), (), (), None)
    root = Path(root)
    frames: list[pd.DataFrame] = []
    sources: list[Path] = []
    missing: list[str] = []
    for backend in backends:
        path = root / str(backend) / episodes_relpath
        frame, reason = _read_one(path)
        if frame is None:
            missing.append(f"{path}: {reason}")
            continue
        frames.append(frame)
        sources.append(path)
    episodes = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    if not episodes.empty:
        episodes = episodes[episodes["backend"].astype(str).isin({str(b) for b in backends})].copy()
    return PairedAxesFrame(episodes, tuple(sources), tuple(missing), root)


def paired_axis_rates(
    episodes: pd.DataFrame,
    *,
    group_cols: Sequence[str] = _AXIS_GROUP_COLS,
) -> pd.DataFrame:
    """Four paired rates per group, each beside its own denominator.

    Delegates to the analysis module so the figure cannot drift from
    ``table_paired_axes.csv``: ``recovery_rate`` keeps the
    ``recovery_applicable`` denominator and every ``*_pre_onset_fpr`` keeps
    dropping its null rows instead of reading them as False.
    """
    if episodes.empty:
        return pd.DataFrame()
    return table_paired_axes(episodes, group_cols=group_cols)


def paired_class_shares(
    episodes: pd.DataFrame,
    *,
    group_cols: Sequence[str] = _CLASS_GROUP_COLS,
) -> pd.DataFrame:
    """8-class histogram split by ``own_baseline`` (§2.5 reading discipline)."""
    if episodes.empty:
        return pd.DataFrame()
    return table_paired_classes(episodes, group_cols=group_cols, split_by_own_baseline=True)


def paired_baseline_states(frame: pd.DataFrame) -> tuple[str, ...]:
    if frame.empty or "own_baseline" not in frame.columns:
        return ()
    present = set(frame["own_baseline"].astype(str))
    known = tuple(s for s in OWN_BASELINE_STATES if s in present)
    return known + tuple(sorted(present - set(known)))


def _read_one(path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.is_file():
        return None, "file not found"
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        return None, f"unreadable ({exc})"
    if frame.empty:
        return None, "no rows"
    absent = [c for c in _REQUIRED_COLS if c not in frame.columns]
    if absent:
        return None, f"missing columns {absent}"
    work = frame.copy()
    for col in (*_REQUIRED_BOOL_COLS, *_NULLABLE_BOOL_COLS):
        work[col] = _coerce_bool(work[col])
    null_axis = [c for c in _REQUIRED_BOOL_COLS if work[c].isna().any()]
    if null_axis:
        return None, f"null axis bits in {null_axis}"
    work["level"] = work["level"].astype(int)
    work["t_onset"] = work["t_onset"].astype(int)
    return work, ""


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Object series of True / False / NaN, whatever the CSV round-trip produced."""

    def one(value: object) -> object:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value is None:
            return np.nan
        if isinstance(value, float) and math.isnan(value):
            return np.nan
        token = str(value).strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        if token in {"", "nan", "none"}:
            return np.nan
        raise ValueError(f"{series.name}: cannot read {value!r} as a boolean")

    return series.map(one).astype(object)
