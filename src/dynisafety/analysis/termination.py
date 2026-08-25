"""Timeout / termination tables. Outcome fields only; no backend imports.

CFF is still ``¬success ∧ contact_steps==0``. This module additionally
splits those contact-free failures into timeout vs early-exit vs
``not_triggered``, which ``dynisafety report`` does not emit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from dynisafety.analysis.load import BoundEpisode, behavior_class_of, group_key_tuple
from dynisafety.analysis.render import render_table
from dynisafety.core.joint_outcomes import classify_episode

TerminationKind = Literal["success", "timeout", "error", "truncated", "other"]
CffBucket = Literal["timeout", "not_triggered", "remainder", "unlabeled"]

TERMINATION_KINDS: tuple[TerminationKind, ...] = (
    "success",
    "timeout",
    "error",
    "truncated",
    "other",
)
CFF_BUCKETS: tuple[CffBucket, ...] = (
    "timeout",
    "not_triggered",
    "remainder",
    "unlabeled",
)

_ERROR_TERMINATIONS = frozenset({"error", "env_error", "policy_error", "sim_unstable"})
_PAPER_GROUP_COLS: tuple[str, ...] = ("backend", "model", "level")


def classify_termination(ep: Any) -> TerminationKind:
    """Map one episode to a termination kind.

    ``success`` wins. Simulator/policy errors win over ``max_steps`` so a
    crash on the last step is not counted as a horizon timeout.
    """
    if bool(ep.success):
        return "success"
    term = str(ep.termination or "")
    if term in _ERROR_TERMINATIONS:
        return "error"
    if term == "truncated":
        return "truncated"
    if term == "max_steps" or int(ep.steps) >= int(ep.max_steps):
        return "timeout"
    return "other"


def is_timeout(ep: Any) -> bool:
    return classify_termination(ep) == "timeout"


def cff_bucket(ep: Any) -> CffBucket | None:
    """CFF sub-bucket, or ``None`` when the episode is not CFF."""
    if classify_episode(ep) != "CFF":
        return None
    cls = behavior_class_of(ep)
    if cls == "not_triggered":
        return "not_triggered"
    if classify_termination(ep) == "timeout":
        return "timeout"
    if ep.diagnostics is None:
        return "unlabeled"
    return "remainder"


def table_termination(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """Per-group timeout rates and CFF split.

    ``n_cff_not_triggered`` is ``None`` (tables render n/a) when the group
    has no diagnostics at all — never a silent 0.0.
    """
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
        kinds = [classify_termination(ep) for ep in eps]
        n_success = sum(1 for k in kinds if k == "success")
        n_timeout = sum(1 for k in kinds if k == "timeout")
        n_error = sum(1 for k in kinds if k == "error")
        n_truncated = sum(1 for k in kinds if k == "truncated")
        n_other = sum(1 for k in kinds if k == "other")
        buckets = [cff_bucket(ep) for ep in eps]
        n_cff = sum(1 for b in buckets if b is not None)
        n_cff_timeout = sum(1 for b in buckets if b == "timeout")
        n_cff_remainder = sum(1 for b in buckets if b == "remainder")
        n_cff_unlabeled = sum(1 for b in buckets if b == "unlabeled")
        has_diag = any(ep.diagnostics is not None for ep in eps)
        n_cff_not_triggered: int | None
        if has_diag:
            n_cff_not_triggered = sum(1 for b in buckets if b == "not_triggered")
        else:
            n_cff_not_triggered = None
        row: dict[str, Any] = {col: val for col, val in zip(group_cols, key, strict=True)}
        row.update(
            {
                "n": n,
                "n_success": n_success,
                "n_timeout": n_timeout,
                "n_error": n_error,
                "n_truncated": n_truncated,
                "n_other": n_other,
                "timeout_rate": (n_timeout / n) if n else None,
                "error_rate": (n_error / n) if n else None,
                "n_cff": n_cff,
                "n_cff_timeout": n_cff_timeout,
                "n_cff_remainder": n_cff_remainder,
                "n_cff_unlabeled": n_cff_unlabeled,
                "n_cff_not_triggered": n_cff_not_triggered,
                "cff_timeout_share": (n_cff_timeout / n_cff) if n_cff else None,
            }
        )
        rows.append(row)

    columns = [
        *group_cols,
        "n",
        "n_success",
        "n_timeout",
        "n_error",
        "n_truncated",
        "n_other",
        "timeout_rate",
        "error_rate",
        "n_cff",
        "n_cff_timeout",
        "n_cff_remainder",
        "n_cff_unlabeled",
        "n_cff_not_triggered",
        "cff_timeout_share",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


_TERMINATION_FOOTNOTES: tuple[str, ...] = (
    "timeout = termination==max_steps (or steps>=max_steps) and not success; "
    "errors win over max_steps.",
    "CFF is still ¬success ∧ contact_steps==0 — not “Safe Stop”.",
    "n_cff_timeout = CFF ∩ horizon timeout. n_cff_remainder = CFF that ended "
    "early with diagnostics (stop / evade-fail / no_response / …).",
    "n_cff_not_triggered is n/a when the group has no diagnostics field.",
)


def write_termination_tables(
    bound: Sequence[BoundEpisode],
    out_dir: Path | str,
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """Write ``table_termination.{csv,md,tex}`` under ``out_dir``."""
    table = table_termination(bound, group_cols=group_cols)
    render_table(
        table,
        Path(out_dir),
        "table_termination",
        footnotes=_TERMINATION_FOOTNOTES,
    )
    return table


def plot_cff_termination(
    bound: Sequence[BoundEpisode],
    out_path: Path | str,
    *,
    level_col: str = "level",
) -> None:
    """Stacked CFF buckets by level. Empty input writes an unavailable figure."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    items = [b for b in bound if cff_bucket(b.episode) is not None]
    if not items:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "CFF termination split (n/a)", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return

    levels = sorted({int(getattr(b.episode, level_col)) for b in items})
    colors = {
        "timeout": "#c44e52",
        "not_triggered": "#937860",
        "remainder": "#4c72b0",
        "unlabeled": "#8c8c8c",
    }
    labels = {
        "timeout": "CFF ∩ timeout",
        "not_triggered": "CFF ∩ not_triggered",
        "remainder": "CFF remainder",
        "unlabeled": "CFF unlabeled",
    }
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(levels) + 2.0), 3.8))
    bottoms = [0.0] * len(levels)
    x = [str(lv) for lv in levels]
    for bucket in CFF_BUCKETS:
        heights = [
            float(
                sum(
                    1
                    for b in items
                    if int(getattr(b.episode, level_col)) == lv
                    and cff_bucket(b.episode) == bucket
                )
            )
            for lv in levels
        ]
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=colors[bucket],
            label=labels[bucket],
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_xlabel("level")
    ax.set_ylabel("CFF episode count")
    ax.set_title('CFF is not "Safe Stop": timeout vs remainder')
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
