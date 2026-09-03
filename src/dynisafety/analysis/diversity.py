"""Read-only L2/L3 diversity analysis from ``extra.spawn_plan`` / ``extra.l3_plan``.

Field convention lives in ``docs/user/eval.md``. This module never imports a
backend and never writes a default mode when the extra key is absent — missing
plan means diversity was off (or the jsonl predates the planner), rendered as
n/a, not 0.0.

``dynisafety report generate`` stays the paper-main pipeline and still does
not read ``extra``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from dynisafety.analysis.load import (
    BoundEpisode,
    behavior_class_of,
    episode_source_row,
    group_key_tuple,
)
from dynisafety.analysis.render import render_table
from dynisafety.core.joint_outcomes import classify_episode, count_joint_outcomes, ssr
from dynisafety.core.metrics import cr, sr
from dynisafety.diagnostics.schema import BehaviorClass

DiversityFamily = Literal["l2_spawn", "l3_reach"]

FAMILY_SPAWN: DiversityFamily = "l2_spawn"
FAMILY_REACH: DiversityFamily = "l3_reach"

PLAN_KEYS: tuple[tuple[str, DiversityFamily], ...] = (
    ("spawn_plan", FAMILY_SPAWN),
    ("l3_plan", FAMILY_REACH),
)

L2_SLOTS: tuple[str, ...] = ("left", "center", "right")
L3_SLOTS: tuple[str, ...] = ("left", "opposite", "right")
L2_STYLES: tuple[str, ...] = ("arc", "linear")
L3_DEPTHS: tuple[str, ...] = ("near", "mid")
L2_MODES: tuple[str, ...] = tuple(f"{slot}_{style}" for slot in L2_SLOTS for style in L2_STYLES)
L3_MODES: tuple[str, ...] = tuple(f"{slot}_{depth}" for slot in L3_SLOTS for depth in L3_DEPTHS)
LEGACY_MODES: frozenset[str] = frozenset({"legacy_jitter", "legacy_opposite"})

_CLASS_VALUES: tuple[str, ...] = tuple(c.value for c in BehaviorClass)
_PAPER_GROUP_COLS: tuple[str, ...] = ("backend", "model", "level")
_NONE_SLOT = "_none"
_NONE_REASON = "_none"


@dataclass(frozen=True)
class DiversityPlan:
    """Normalized view of one extra plan object. Absent extra key → no instance."""

    family: DiversityFamily
    extra_key: str
    enabled: bool | None
    requested_mode: str | None
    accepted_mode: str | None
    slot: str | None
    path_style: str | None
    depth: str | None
    fallback_reason: str | None
    fallback_chain: tuple[str, ...]
    is_legacy: bool
    accepted_as_requested: bool


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _as_chain(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _is_legacy(mode: str | None) -> bool:
    if mode is None:
        return False
    return mode in LEGACY_MODES or mode.startswith("legacy_")


def _rate(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _is_diversity_planner_record(raw: dict[str, Any], extra_key: str) -> bool:
    """True only when the object came from the spawn/reach planner.

    LIBERO L3 always writes a geometric ``l3_plan`` (start_mode / target_mode /
    eef_pos) even when ``reach_diversity.enabled`` is false. That snapshot is
    not a 6-cell diversity plan and must not enter acceptance / slot tables.
    """
    if extra_key == "l3_plan" and raw.get("reach_diversity_enabled") is False:
        return False
    if extra_key == "spawn_plan" and raw.get("enabled") is False:
        return False
    return "accepted_mode" in raw or "requested_mode" in raw


def extract_plan(ep: Any, extra_key: str, family: DiversityFamily) -> DiversityPlan | None:
    """Return a plan only when the extra object is a diversity planner record.

    A missing key, or LIBERO's geometry-only L3 snapshot, is "diversity off"
    — not an implicit default mode.
    """
    extra = ep.extra if isinstance(getattr(ep, "extra", None), dict) else None
    if extra is None or extra_key not in extra:
        return None
    raw = extra[extra_key]
    if not isinstance(raw, dict) or not raw:
        return None
    if not _is_diversity_planner_record(raw, extra_key):
        return None

    if extra_key == "l3_plan":
        enabled = _opt_bool(raw.get("reach_diversity_enabled"))
        if enabled is None:
            enabled = _opt_bool(raw.get("enabled"))
    else:
        enabled = _opt_bool(raw.get("enabled"))

    requested = _opt_str(raw.get("requested_mode"))
    accepted = _opt_str(raw.get("accepted_mode"))
    slot = _opt_str(raw.get("slot")) if "slot" in raw else None
    path_style = _opt_str(raw.get("path_style")) if "path_style" in raw else None
    depth = _opt_str(raw.get("depth")) if "depth" in raw else None
    if depth is None and family == FAMILY_REACH and accepted and not _is_legacy(accepted):
        _slot_part, sep, rest = accepted.partition("_")
        if sep and rest:
            depth = rest
    fallback_reason = _opt_str(raw.get("fallback_reason"))
    return DiversityPlan(
        family=family,
        extra_key=extra_key,
        enabled=enabled,
        requested_mode=requested,
        accepted_mode=accepted,
        slot=slot,
        path_style=path_style,
        depth=depth,
        fallback_reason=fallback_reason,
        fallback_chain=_as_chain(raw.get("fallback_chain")),
        is_legacy=_is_legacy(accepted),
        accepted_as_requested=bool(requested and accepted and requested == accepted),
    )


def iter_plans(bound: Sequence[BoundEpisode]) -> list[tuple[BoundEpisode, DiversityPlan]]:
    rows: list[tuple[BoundEpisode, DiversityPlan]] = []
    for item in bound:
        for extra_key, family in PLAN_KEYS:
            plan = extract_plan(item.episode, extra_key, family)
            if plan is not None:
                rows.append((item, plan))
    return rows


def _group_map(
    bound: Sequence[BoundEpisode], group_cols: Sequence[str]
) -> dict[tuple[Any, ...], list[BoundEpisode]]:
    groups: dict[tuple[Any, ...], list[BoundEpisode]] = defaultdict(list)
    for item in bound:
        groups[group_key_tuple(item.episode, group_cols)].append(item)
    return groups


def _class_count_row(episodes: Sequence[Any]) -> dict[str, int | None]:
    if not any(getattr(ep, "diagnostics", None) is not None for ep in episodes):
        return {f"n_class_{name}": None for name in _CLASS_VALUES}
    counts = dict.fromkeys(_CLASS_VALUES, 0)
    for ep in episodes:
        label = behavior_class_of(ep)
        if label in counts:
            counts[label] += 1
    return {f"n_class_{name}": counts[name] for name in _CLASS_VALUES}


def _outcome_row(episodes: Sequence[Any]) -> dict[str, Any]:
    n = len(episodes)
    if n == 0:
        row: dict[str, Any] = {
            "n": 0,
            "sr": None,
            "cr": None,
            "ssr": None,
            "n_ss": 0,
            "n_us": 0,
            "n_cff": 0,
            "n_cif": 0,
        }
        row.update(_class_count_row(episodes))
        return row
    counts = count_joint_outcomes(episodes)
    row = {
        "n": n,
        "sr": sr(episodes),
        "cr": cr(episodes),
        "ssr": ssr(episodes),
        "n_ss": counts.ss,
        "n_us": counts.us,
        "n_cff": counts.cff,
        "n_cif": counts.cif,
    }
    row.update(_class_count_row(episodes))
    return row


def _unique_or_none(values: Sequence[str | None]) -> str | None:
    present = {v for v in values if v is not None}
    if len(present) == 1:
        return next(iter(present))
    return None


def table_diversity_coverage(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """How many episodes actually carry a plan object.

    ``n_spawn_plan == 0`` is a real count. Acceptance rates among planned
    episodes are n/a in the other tables, not silently 0.
    """
    columns = [
        *group_cols,
        "n",
        "n_spawn_plan",
        "n_l3_plan",
        "spawn_plan_share",
        "l3_plan_share",
    ]
    groups = _group_map(bound, group_cols)
    rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        items = groups[key]
        n = len(items)
        n_spawn = sum(
            1 for b in items if extract_plan(b.episode, "spawn_plan", FAMILY_SPAWN) is not None
        )
        n_reach = sum(
            1 for b in items if extract_plan(b.episode, "l3_plan", FAMILY_REACH) is not None
        )
        row: dict[str, Any] = {col: val for col, val in zip(group_cols, key, strict=True)}
        row.update(
            {
                "n": n,
                "n_spawn_plan": n_spawn,
                "n_l3_plan": n_reach,
                "spawn_plan_share": _rate(n_spawn, n),
                "l3_plan_share": _rate(n_reach, n),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_diversity_acceptance(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """Accept-as-requested / fallback / legacy shares among planned episodes.

    Groups with no plan object are omitted (see coverage). Rates are n/a only
    when the planned denominator is 0, which this table does not emit.
    """
    columns = [
        *group_cols,
        "family",
        "n_planned",
        "n_accepted_as_requested",
        "accept_as_requested_rate",
        "n_fallback",
        "fallback_rate",
        "n_legacy",
        "legacy_share",
    ]
    planned = iter_plans(bound)
    buckets: dict[tuple[Any, ...], list[DiversityPlan]] = defaultdict(list)
    for item, plan in planned:
        key = (*group_key_tuple(item.episode, group_cols), plan.family)
        buckets[key].append(plan)
    rows: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        plans = buckets[key]
        n = len(plans)
        n_ok = sum(1 for p in plans if p.accepted_as_requested)
        n_fb = sum(1 for p in plans if not p.accepted_as_requested)
        n_leg = sum(1 for p in plans if p.is_legacy)
        row: dict[str, Any] = {col: val for col, val in zip(group_cols, key[:-1], strict=True)}
        row.update(
            {
                "family": key[-1],
                "n_planned": n,
                "n_accepted_as_requested": n_ok,
                "accept_as_requested_rate": _rate(n_ok, n),
                "n_fallback": n_fb,
                "fallback_rate": _rate(n_fb, n),
                "n_legacy": n_leg,
                "legacy_share": _rate(n_leg, n),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_diversity_by_mode(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """``accepted_mode`` × ``path_style`` × SR / CR / SSR / quadrants / 9-class counts.

    ``path_style`` is a grouping key, not a collapsed attribute. The mode name
    only encodes ``slot`` and a coarse style, so ``*_linear`` on L2 actually
    realises both ``linear`` and ``trapezoid`` paths, and L3 ``right_*`` mixes
    ``straight`` / ``raised`` / ``via`` while ``left_*`` is pure ``straight``.
    Grouping on the mode alone pooled those and rendered the style column n/a,
    which hid a confound between the slot axis and the path axis.
    """
    class_cols = [f"n_class_{name}" for name in _CLASS_VALUES]
    columns = [
        *group_cols,
        "family",
        "accepted_mode",
        "is_legacy",
        "slot",
        "path_style",
        "depth",
        "n",
        "sr",
        "cr",
        "ssr",
        "n_ss",
        "n_us",
        "n_cff",
        "n_cif",
        *class_cols,
    ]
    planned = iter_plans(bound)
    buckets: dict[tuple[Any, ...], list[tuple[BoundEpisode, DiversityPlan]]] = defaultdict(list)
    for item, plan in planned:
        mode = plan.accepted_mode or "_unknown"
        style = plan.path_style or _NONE_SLOT
        key = (*group_key_tuple(item.episode, group_cols), plan.family, mode, style)
        buckets[key].append((item, plan))
    rows: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        pairs = buckets[key]
        eps = [b.episode for b, _ in pairs]
        plans = [p for _, p in pairs]
        row: dict[str, Any] = {col: val for col, val in zip(group_cols, key[:-3], strict=True)}
        row.update(
            {
                "family": key[-3],
                "accepted_mode": key[-2],
                "is_legacy": _unique_or_none([p.is_legacy for p in plans]),
                "slot": _unique_or_none([p.slot for p in plans]),
                "path_style": key[-1],
                "depth": _unique_or_none([p.depth for p in plans]),
            }
        )
        row.update(_outcome_row(eps))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_diversity_slots(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """Accepted-slot histogram. Legacy / missing slot is ``_none``."""
    columns = [*group_cols, "family", "slot", "n", "share"]
    planned = iter_plans(bound)
    totals: dict[tuple[Any, ...], int] = defaultdict(int)
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for item, plan in planned:
        slot = plan.slot if plan.slot else _NONE_SLOT
        group = (*group_key_tuple(item.episode, group_cols), plan.family)
        totals[group] += 1
        counts[(*group, slot)] += 1
    rows: list[dict[str, Any]] = []
    for key in sorted(counts, key=lambda k: tuple(str(x) for x in k)):
        *head, slot = key
        group = tuple(head)
        n = counts[key]
        row: dict[str, Any] = {col: val for col, val in zip(group_cols, group[:-1], strict=True)}
        row.update(
            {
                "family": group[-1],
                "slot": slot,
                "n": n,
                "share": _rate(n, totals[group]),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_diversity_fallback(
    bound: Sequence[BoundEpisode],
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> pd.DataFrame:
    """Terminal ``fallback_reason`` histogram. ``_none`` = first-form accept."""
    columns = [*group_cols, "family", "fallback_reason", "n", "share"]
    planned = iter_plans(bound)
    totals: dict[tuple[Any, ...], int] = defaultdict(int)
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for item, plan in planned:
        reason = plan.fallback_reason if plan.fallback_reason else _NONE_REASON
        group = (*group_key_tuple(item.episode, group_cols), plan.family)
        totals[group] += 1
        counts[(*group, reason)] += 1
    rows: list[dict[str, Any]] = []
    for key in sorted(counts, key=lambda k: tuple(str(x) for x in k)):
        *head, reason = key
        group = tuple(head)
        n = counts[key]
        row: dict[str, Any] = {col: val for col, val in zip(group_cols, group[:-1], strict=True)}
        row.update(
            {
                "family": group[-1],
                "fallback_reason": reason,
                "n": n,
                "share": _rate(n, totals[group]),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def table_diversity_episodes(bound: Sequence[BoundEpisode]) -> pd.DataFrame:
    """One row per planned episode (identity + plan + outcomes)."""
    columns = [
        "run_dir",
        "run_id",
        "backend",
        "model",
        "suite",
        "task_id",
        "task_name",
        "level",
        "seed",
        "episode_index",
        "family",
        "extra_key",
        "enabled",
        "requested_mode",
        "accepted_mode",
        "slot",
        "path_style",
        "depth",
        "fallback_reason",
        "is_legacy",
        "accepted_as_requested",
        "success",
        "contact_steps",
        "quadrant",
        "behavior_class",
    ]
    rows: list[dict[str, Any]] = []
    for item, plan in iter_plans(bound):
        ep = item.episode
        row = episode_source_row(item)
        row.update(
            {
                "family": plan.family,
                "extra_key": plan.extra_key,
                "enabled": plan.enabled,
                "requested_mode": plan.requested_mode,
                "accepted_mode": plan.accepted_mode,
                "slot": plan.slot,
                "path_style": plan.path_style,
                "depth": plan.depth,
                "fallback_reason": plan.fallback_reason,
                "is_legacy": plan.is_legacy,
                "accepted_as_requested": plan.accepted_as_requested,
                "success": bool(ep.success),
                "contact_steps": int(ep.contact_steps),
                "quadrant": classify_episode(ep),
                "behavior_class": behavior_class_of(ep),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


_COVERAGE_NOTES: tuple[str, ...] = (
    "A missing extra.spawn_plan / extra.l3_plan key means diversity was off "
    "or the jsonl predates the planner — do not impute a default mode.",
    "LIBERO L3 always writes a geometric l3_plan; it counts only when "
    "requested_mode / accepted_mode (or reach_diversity_enabled) is present.",
    "n_spawn_plan / n_l3_plan count planner records, including legacy_* fallbacks.",
    "Acceptance and per-mode rates use only planned episodes; empty tables "
    "are n/a, not a silent 0.0.",
)

_ACCEPT_NOTES: tuple[str, ...] = (
    "accepted_as_requested = requested_mode == accepted_mode (both present).",
    "legacy_jitter is L2; legacy_opposite is L3. legacy_share is among planned episodes.",
    "fallback_rate counts any accepted_mode that is not the requested mode.",
)

_MODE_NOTES: tuple[str, ...] = (
    "Cross of accepted_mode × path_style with SR / CR / SSR. "
    "Do not rank modes across backends.",
    "path_style is a grouping key: the mode name does not determine it "
    "(L2 *_linear realises linear and trapezoid; L3 left_* is straight-only "
    "while right_* / opposite_* also draw raised and via). Comparing slots "
    "without holding path_style fixed confounds the two axes.",
    "is_legacy rows are fallback plans, not requested modes — keep them out of "
    "any per-mode comparison.",
    "n_class_* is n/a when the cell has no diagnostics field.",
    "L2 modes: {left,center,right}_{arc,linear}. L3 modes: {left,opposite,right}_{near,mid}.",
)

_SLOT_NOTES: tuple[str, ...] = (
    "slot is the accepted lateral cell. _none = legacy_* or a null slot field.",
    "L2 slots: left / center / right. L3 slots: left / opposite / right.",
)

_FALLBACK_NOTES: tuple[str, ...] = (
    "_none = requested mode accepted in first-form (fallback_reason is null).",
    "Terminal reason only; inspect fallback_chain on the episode CSV for the trail.",
)


def _empty_figure(out: Path, message: str) -> None:
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_diversity_slots(
    bound: Sequence[BoundEpisode],
    out_path: Path | str,
) -> None:
    """Accepted-slot counts. Empty input writes an unavailable figure."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    planned = iter_plans(bound)
    if not planned:
        _empty_figure(out, "diversity slots (n/a — no spawn_plan / l3_plan)")
        return

    by_family: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, plan in planned:
        slot = plan.slot if plan.slot else _NONE_SLOT
        by_family[plan.family][slot] += 1

    families = [f for f in (FAMILY_SPAWN, FAMILY_REACH) if f in by_family]
    order = {
        FAMILY_SPAWN: (*L2_SLOTS, _NONE_SLOT),
        FAMILY_REACH: (*L3_SLOTS, _NONE_SLOT),
    }
    fig, axes = plt.subplots(
        1,
        len(families),
        figsize=(4.2 * len(families) + 1.0, 3.6),
        squeeze=False,
    )
    colors = {"left": "#4c72b0", "center": "#55a868", "right": "#c44e52",
              "opposite": "#8172b3", _NONE_SLOT: "#8c8c8c"}
    for ax, family in zip(axes[0], families, strict=True):
        counts = by_family[family]
        labels = [s for s in order[family] if s in counts]
        labels.extend(sorted(s for s in counts if s not in labels))
        heights = [float(counts[s]) for s in labels]
        ax.bar(
            labels,
            heights,
            color=[colors.get(s, "#4c72b0") for s in labels],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_title(family)
        ax.set_ylabel("planned episodes")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Accepted diversity slot (legacy → _none)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_diversity_legacy(
    bound: Sequence[BoundEpisode],
    out_path: Path | str,
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> None:
    """Legacy share by group × family. Empty input writes n/a."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    table = table_diversity_acceptance(bound, group_cols=group_cols)
    if table.empty:
        _empty_figure(out, "diversity legacy share (n/a — no spawn_plan / l3_plan)")
        return

    labels = [
        " / ".join([*(str(row[c]) for c in group_cols), str(row["family"])])
        for _, row in table.iterrows()
    ]
    shares = [float(v) for v in table["legacy_share"].tolist()]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.45 * len(labels) + 2.5), 3.8))
    ax.bar(range(len(labels)), shares, color="#c44e52", edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("legacy_share")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("legacy_* share among planned episodes")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_diversity(
    bound: Sequence[BoundEpisode],
    out_dir: Path | str,
    *,
    group_cols: Sequence[str] = _PAPER_GROUP_COLS,
) -> dict[str, pd.DataFrame]:
    """Write coverage / acceptance / mode / slot / fallback tables and plots."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "table_diversity_coverage": (
            table_diversity_coverage(bound, group_cols=group_cols),
            _COVERAGE_NOTES,
        ),
        "table_diversity_acceptance": (
            table_diversity_acceptance(bound, group_cols=group_cols),
            _ACCEPT_NOTES,
        ),
        "table_diversity_by_mode": (
            table_diversity_by_mode(bound, group_cols=group_cols),
            _MODE_NOTES,
        ),
        "table_diversity_slots": (
            table_diversity_slots(bound, group_cols=group_cols),
            _SLOT_NOTES,
        ),
        "table_diversity_fallback": (
            table_diversity_fallback(bound, group_cols=group_cols),
            _FALLBACK_NOTES,
        ),
        "diversity_episodes": (table_diversity_episodes(bound), ()),
    }
    written: dict[str, pd.DataFrame] = {}
    for stem, (frame, notes) in tables.items():
        render_table(frame, out, stem, footnotes=notes)
        written[stem] = frame
    plot_diversity_slots(bound, out / "plots" / "diversity_slots.png")
    plot_diversity_legacy(bound, out / "plots" / "diversity_legacy.png", group_cols=group_cols)
    return written
