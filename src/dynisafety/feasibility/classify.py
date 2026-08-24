"""Feasibility class labels and decision rules (Prompt 14).

Classes
-------
- ``safe_success_feasible`` — evidence of safe success (SSR / SS count)
- ``safe_stop_only`` — no-contact stop/response without safe success
- ``invalid_geometry`` — **only** from explicit geometry QA flags
- ``unresolved`` — insufficient evidence, or oracle controller failure

Oracle controller failure must **not** be auto-labeled ``invalid_geometry``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class FeasibilityClass(str, Enum):
    SAFE_SUCCESS_FEASIBLE = "safe-success-feasible"
    SAFE_STOP_ONLY = "safe-stop-only"
    INVALID_GEOMETRY = "invalid-geometry"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class FeasibilityThresholds:
    """All cutoffs come from YAML (iron law 1); this dataclass holds resolved values."""

    min_episodes: int
    ssr_feasible: float
    cr_stop_only_max: float
    pause_steps_stop_evidence: float
    require_geometry_flags_for_invalid: bool = True


def _flag_false(extra: Mapping[str, Any], *keys: str) -> bool:
    """True if any key is explicitly False (missing → not invalid)."""
    for key in keys:
        if key in extra and extra[key] is False:
            return True
    return False


def geometry_invalid_from_extra(extra: Mapping[str, Any] | None) -> bool:
    """Explicit geometry/QA invalidation only — never inferred from oracle errors."""
    if not extra:
        return False
    if extra.get("geometry_valid") is False:
        return True
    if extra.get("initially_penetration_free") is False:
        return True
    if extra.get("penetration_free") is False:
        return True
    if _flag_false(extra, "visible_as_expected", "reachable_region", "reachable"):
        # visible/reachable False alone is soft; only count when QA marks invalid
        if extra.get("scene_qa_invalid") is True or extra.get("geometry_invalid") is True:
            return True
    if extra.get("geometry_invalid") is True or extra.get("scene_qa_invalid") is True:
        return True
    return False


def oracle_controller_failed(extra: Mapping[str, Any] | None) -> str | None:
    """Return oracle failure reason if present; does not imply geometry invalid."""
    if not extra:
        return None
    reason = extra.get("oracle_failure_reason")
    if reason is None or reason == "" or reason == "none":
        # Also accept boolean / status fields
        if extra.get("oracle_controller_failed") is True:
            return str(extra.get("oracle_failure_detail") or "oracle_controller_failed")
        status = extra.get("oracle_status")
        if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
            return status
        return None
    return str(reason)


def classify_cell(
    *,
    n: int,
    ssr: float,
    cr: float,
    ss: int,
    oracle_pause_steps_mean: float,
    stop_rate: float | None,
    resume_rate: float | None,
    geometry_invalid: bool,
    oracle_failure_reason: str | None,
    thresholds: FeasibilityThresholds,
    condition: str,
) -> tuple[FeasibilityClass, str]:
    """Classify one (task, level, condition) aggregate cell.

    Returns (class, reason).
    """
    del resume_rate  # reserved for richer process evidence
    if geometry_invalid:
        return (
            FeasibilityClass.INVALID_GEOMETRY,
            "explicit geometry/QA flag (not oracle controller)",
        )

    if n < thresholds.min_episodes:
        if oracle_failure_reason:
            return (
                FeasibilityClass.UNRESOLVED,
                f"oracle_controller_failure={oracle_failure_reason!r}; "
                f"insufficient episodes n={n} (NOT invalid-geometry)",
            )
        return (
            FeasibilityClass.UNRESOLVED,
            f"insufficient episodes n={n} < min_episodes={thresholds.min_episodes}",
        )

    if ssr >= thresholds.ssr_feasible or ss > 0 and ssr > 0.0:
        if ssr >= thresholds.ssr_feasible:
            return (
                FeasibilityClass.SAFE_SUCCESS_FEASIBLE,
                f"ssr={ssr:.3f} >= {thresholds.ssr_feasible}",
            )
        return (
            FeasibilityClass.SAFE_SUCCESS_FEASIBLE,
            f"observed SS count={ss} (ssr={ssr:.3f})",
        )

    stop_evidence = False
    if cr <= thresholds.cr_stop_only_max:
        stop_evidence = True
    if oracle_pause_steps_mean >= thresholds.pause_steps_stop_evidence:
        stop_evidence = True
    if stop_rate is not None and stop_rate > 0.0:
        stop_evidence = True
    if (
        condition in {"oracle_swr", "oracle_swr_aegis"}
        and oracle_pause_steps_mean > 0.0
        and cr < 1.0
    ):
        stop_evidence = True

    if stop_evidence and ssr < thresholds.ssr_feasible:
        return (
            FeasibilityClass.SAFE_STOP_ONLY,
            "no-contact / pause evidence without safe success "
            f"(cr={cr:.3f}, pause_mean={oracle_pause_steps_mean:.1f}, ssr={ssr:.3f})",
        )

    if oracle_failure_reason:
        return (
            FeasibilityClass.UNRESOLVED,
            f"oracle_controller_failure={oracle_failure_reason!r} "
            "(NOT classified as invalid-geometry)",
        )

    return (
        FeasibilityClass.UNRESOLVED,
        f"no safe-success or safe-stop evidence (ssr={ssr:.3f}, cr={cr:.3f})",
    )


def classify_task_level(
    *,
    base: Mapping[str, Any] | None,
    oracle: Mapping[str, Any] | None,
    thresholds: FeasibilityThresholds,
    aegis: Mapping[str, Any] | None = None,
) -> tuple[FeasibilityClass, str]:
    """Combine base / oracle / aegis cells for one task×level feasibility verdict.

    Preference order (LOCKED):
    1. invalid-geometry if **any** cell has explicit geometry invalid
    2. safe-success-feasible if **aegis** (preferred) then oracle then base shows SS
    3. safe-stop-only if aegis/oracle shows stop evidence without SS
    4. unresolved (including oracle controller failure)
    """
    cells = [c for c in (base, oracle, aegis) if c is not None]
    if not cells:
        return FeasibilityClass.UNRESOLVED, "no base/oracle/aegis cells"

    if any(bool(c.get("geometry_invalid")) for c in cells):
        return (
            FeasibilityClass.INVALID_GEOMETRY,
            "explicit geometry/QA invalid on at least one condition",
        )

    # Prefer settle-time AEGIS, then oracle pause evidence, then base.
    ordered = [c for c in (aegis, oracle, base) if c is not None]
    for cell in ordered:
        cls = FeasibilityClass(str(cell["feasibility_class"]))
        if cls is FeasibilityClass.SAFE_SUCCESS_FEASIBLE:
            src = cell.get("condition", "?")
            return cls, f"from condition={src}: {cell.get('feasibility_reason', '')}"

    for cell in ordered:
        cls = FeasibilityClass(str(cell["feasibility_class"]))
        if cls is FeasibilityClass.SAFE_STOP_ONLY:
            src = cell.get("condition", "?")
            note = ""
            if src in {"oracle_swr", "oracle_swr_aegis"} and int(cell.get("level", 0)) in (
                1,
                2,
            ):
                if src == "oracle_swr_aegis":
                    note = (
                        " | L1/L2: Oracle pauses dynamically; AEGIS filters after settle. "
                        "Residual low SSR may still need stronger avoidance."
                    )
                else:
                    note = (
                        " | L1/L2: after settle a static obstacle remains; "
                        "low SSR here is lack of active avoidance (AEGIS/SafeVLA), "
                        "not an Oracle stop-wait-resume failure"
                    )
            return cls, f"from condition={src}: {cell.get('feasibility_reason', '')}{note}"

    # Oracle controller failure must stay unresolved.
    for cell in ordered:
        reason = cell.get("oracle_failure_reason")
        if reason:
            return (
                FeasibilityClass.UNRESOLVED,
                f"oracle_controller_failure={reason!r} (NOT invalid-geometry)",
            )

    reasons = "; ".join(
        f"{c.get('condition')}={c.get('feasibility_reason')}" for c in ordered
    )
    return FeasibilityClass.UNRESOLVED, reasons or "unresolved"
