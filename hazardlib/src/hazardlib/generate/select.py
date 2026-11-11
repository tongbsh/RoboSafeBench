"""Selection of an admitted candidate per cell and the generation statistics (pure; no host).

A host runs its own checks (exact overlap, render visibility, physical episodes) and hands them to
:func:`select_admitted` as a callback; the ledgers it leaves behind are summarised per cell
(family x entry-direction class x Sigma target) by :func:`cell_statistics` / :func:`generation_table`.
"""
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from .stats import distribution

#: Rejection categories whose detail (partners / reasons after the first colon) is counted separately.
DETAILED = ("layout_prefilter", "scene_clearance", "scene_audit", "g0", "physical_nominal_hit_or_visibility",
            "visibility")


def select_admitted(shortlisted: Sequence[Tuple[Any, Any, Dict[str, Any]]],
                    check: Callable[[Any, Dict[str, Any]], Tuple[bool, bool]], budget: int):
    """First candidate (in the given order) that ``check(instance, item)`` admits. ``check`` returns
    ``(attempted, admitted)``; only an attempted candidate (a physical episode ran) uses the ``budget``.
    Candidates never reached are marked ``not_evaluated``. Returns ``(chosen entry or None, attempts)``."""
    attempts, chosen = 0, None
    for position, entry in enumerate(shortlisted):
        _, instance, item = entry
        attempted, admitted = check(instance, item)
        attempts += bool(attempted)
        if admitted:
            chosen = entry
            reached = position + 1
            break
        if attempts >= budget:
            reached = position + 1
            break
    else:
        reached = len(shortlisted)
    for _, _, item in shortlisted[reached:]:
        item["not_evaluated"] = "selected_earlier" if chosen is not None else "attempt_budget"
    return chosen, attempts


def rejection_keys(reason: str):
    """``(category, [category:detail, ...])`` of a ledger rejection string; details are the partners or
    reasons after the first colon (separated by commas or semicolons)."""
    category, _, rest = reason.partition(":")
    details = [d.strip() for d in rest.replace(";", ",").split(",") if d.strip()]
    return category, ([category + ":" + d for d in details] if category in DETAILED else [])


def _count(items) -> Dict[Any, int]:
    out = {}  # type: Dict[Any, int]
    for value in items:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def ledger_statistics(ledgers: Sequence[Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Counts over the candidate ledgers of one cell (one list of items per layout): rejection reasons by
    category and by detail, candidates not evaluated, the release kind of every physical nominal-hit episode,
    the audit outcomes and the same for the selected candidates (with the G1 diagnostic)."""
    category, detail, not_evaluated = [], [], []
    nominal_kinds, audits, selected = [], [], []
    for ledger in ledgers:
        for item in ledger:
            if "rejection" in item:
                c, d = rejection_keys(item["rejection"])
                category.append(c)
                detail.extend(d)
            if "not_evaluated" in item:
                not_evaluated.append(item["not_evaluated"])
            if "nominal_hit" in item:
                nominal_kinds.append(str(item["nominal_hit"].get("release_kind")))
            if "scene_audit" in item:
                audits.append(item["scene_audit"])
            if item.get("selected"):
                selected.append(item)

    def audit_counts(records):
        return {"n": len(records), "exited": sum(a["exited"] for a in records),
                "released": sum(a["released"] for a in records),
                "release_kinds": _count([str(a["release_kind"]) for a in records]),
                "hold_incomplete": sum(not a["hold_complete"] for a in records),
                "task_success": sum(bool(a["success"]) for a in records),
                "admitted": sum(a["admitted"] for a in records)}
    return {"rejections_by_category": _count(category), "rejections_by_detail": _count(detail),
            "not_evaluated": _count(not_evaluated),
            "nominal_release_kinds_all_attempts": _count(nominal_kinds),
            "nominal_release_kinds_selected": _count([str(i["nominal_hit"].get("release_kind")) for i in selected]),
            "audit_all_attempts": audit_counts(audits),
            "audit_selected": audit_counts([i["scene_audit"] for i in selected]),
            "g1_diagnostic_selected": {"n": len(selected),
                                       "passed": sum(bool(i["g1"]["passed"]) for i in selected)}}


def top_counts(counts: Mapping[Any, int], n: int) -> Dict[Any, int]:
    return dict(list(counts.items())[:n])


def cell_statistics(entries: Sequence[Tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
                    quantiles: Sequence[float]) -> Dict[str, Any]:
    """One cell over the layouts. ``entries``: ``(result, ledger)`` per layout, ``result`` the
    ``<cell>.sigma<t>_result.json`` content and ``ledger`` its candidate ledger (``[]`` if none was written).
    Counts layouts by outcome (admitted / unreachable by reason / no admitted candidate), the top rejection
    reasons, the G1 diagnostic of the admitted instances and the distributions of tau, achieved Sigma, visible
    lead and hit-point coverage over the admitted instances."""
    top = 5     # display cap of the reason lists, not a threshold
    results = [r for r, _ in entries]
    selected = [r["selected"] for r in results if r["selected"]]
    unreachable = {}  # type: Dict[str, int]
    for r in results:
        if r["unreachable"] is not None:
            unreachable[r["unreachable"]] = unreachable.get(r["unreachable"], 0) + 1
    stats = ledger_statistics([ledger for _, ledger in entries])
    passed = sum(bool(s["g1"]["passed"]) for s in selected)
    coverage = [s["evidence"]["coverage"] for s in selected if (s.get("evidence") or {}).get("coverage") is not None]
    return {"n_layouts": len(results), "admitted": len(selected),
            "wait_solvable": sum(r["status"] == "wait_solvable" for r in results),
            "uncertified": sum(r["status"] == "uncertified" for r in results),
            "unreachable": sum(unreachable.values()), "unreachable_by_reason": dict(sorted(unreachable.items())),
            "no_admitted_candidate": sum(r["status"] == "no_admitted_candidate" for r in results),
            "n_candidates": sum(r["n_candidates"] for r in results),
            "n_prefilter_clear": sum(r.get("n_prefilter_clear", 0) for r in results),
            "n_scene_clear": sum(r["n_scene_clear"] for r in results),
            "n_visible": sum(r["n_visible"] for r in results),
            "dynamic_attempts": sum(r["dynamic_attempts"] for r in results),
            "top_rejection_categories": top_counts(stats["rejections_by_category"], top),
            "top_rejection_details": top_counts(stats["rejections_by_detail"], top),
            "g1_diagnostic": {"n": len(selected), "passed": passed,
                              "pass_rate": None if not selected else passed / len(selected)},
            "tau_s": distribution([s["sigma"]["tau_s"] for s in selected], quantiles),
            "sigma_achieved": distribution([s["sigma"]["sigma_achieved"] for s in selected], quantiles),
            "visible_lead_s": distribution([s["sigma"]["visible_lead_s"] for s in selected], quantiles),
            "coverage": distribution(coverage, quantiles),
            "ledger": stats}


def t_stop_statistics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-layout T_stop records (``validation.json: t_stop``): status counts, min / median / max of the
    medians over the layouts whose protocol stopped every repetition (a censored value is excluded)."""
    status = {}  # type: Dict[str, int]
    for r in records:
        status[r["status"]] = status.get(r["status"], 0) + 1
    values = [r["median_s"] for r in records if r["status"] == "ok"]
    return {"n_layouts": len(records), "status": dict(sorted(status.items())),
            "n_censored_layouts": sum(r["status"] == "censored" for r in records),
            "median_s": None if not values else {"min": float(min(values)), "median": float(np.median(values)),
                                                 "max": float(max(values))}}


def generation_table(families: Mapping[str, Mapping[str, Any]]) -> str:
    """Markdown table of the generation report: one row per (family, entry-direction class, Sigma target) with
    the admitted count, unreachable by reason, no-candidate count, rejection reasons, the coverage of the
    admitted hit points (median [min, max]) and the G1 diagnostic. ``families`` maps a cell id
    (``<family>@<class>``) to ``{"targets": {sigma key: cell_statistics}}``."""
    def cover(c):
        if not c["n"]:
            return "-"
        return "%.2f [%.2f, %.2f]" % (c["quantiles"]["0.5"], c["min"], c["max"])
    lines = ["| family | entry class | Sigma | layouts | admitted | unreachable | no candidate | top rejections | "
             "coverage median [min, max] | G1 diagnostic |", "|---|---|---:|---:|---:|---|---:|---|---|---|"]
    for cid, value in families.items():
        family, _, tag = cid.partition("@")
        for sigma, c in value["targets"].items():
            g1 = c["g1_diagnostic"]
            lines.append("| %s | %s | %s | %d | %d | %s | %d | %s | %s | %s |" % (
                family, tag, sigma, c["n_layouts"], c["admitted"],
                ", ".join("%s:%d" % kv for kv in c["unreachable_by_reason"].items()) or "0",
                c["no_admitted_candidate"],
                ", ".join("%s:%d" % kv for kv in list(c["top_rejection_details"].items())[:3]) or
                ", ".join("%s:%d" % kv for kv in list(c["top_rejection_categories"].items())[:3]) or "-",
                cover(c["coverage"]), "-" if not g1["n"] else "%d/%d" % (g1["passed"], g1["n"])))
    return "\n".join(lines) + "\n"
