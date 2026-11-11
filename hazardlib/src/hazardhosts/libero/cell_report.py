"""Per-paradigm-cell report of the generated-cell evaluation (D20 d; no simulator).

Rows are the cells ``family@elevation`` x Sigma target; the same table is produced overall and per suite, with a
subtotal row per cell (all Sigma targets) and a total row. Every hazard record is attributed to its cell through
``selection.json`` (written by ``cell_eval``); the run records themselves are the core run stage's, so the core
``load_runs`` / ``summarize`` / ``write_report`` still apply to the same directories.

Counts per row (raw, always shown):

* ``n_episodes``: hazard episodes run; ``n_main``: the denominator of the rates, i.e. the core placement ``main``
  (``exposure_low`` and ``setup_invalid`` / ``sim_unstable`` episodes are counted separately, out of the rates);
* ``success`` (task success under hazard), ``contact`` (official contact), ``contact_free_success`` (success and no
  official contact) over ``n_main``; ``exposure_low`` and ``invalid`` counts;
* ``control_success``: the valid controls paired with the row's main episodes (distinct controls);
* ``g1_passed`` / ``g1_failed``: the generation's G1 stop-wait-resume diagnostic of the row's instances (D19: not an
  admission criterion), ``labels``: their certificate labels;
* ``sigma_achieved``: mean of the instances' achieved Sigma (from the layout's own T_stop calibration; the run
  records carry no Sigma because the calibration differs per layout, see ``cell_eval``).

Rates carry Wilson intervals (``hazardlib.run.stats``) at the run configuration's confidence.
"""
import dataclasses
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hazardlib.run.plan import CONTROL, HAZARD
from hazardlib.run.report import MAIN, EXCLUDED_INVALID, group_metrics, placement
from hazardlib.run.settings import ReportSettings
from hazardlib.run.stats import proportion

FAMILY_ORDER = ("intrusion.fall", "intrusion.slide", "intrusion.reach")


def wilson_settings(settings: ReportSettings) -> ReportSettings:
    return dataclasses.replace(settings, interval_method="wilson")


def attribution(selection: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``instance_id -> {task, suite, demo, cell, target, ...}`` of every selected instance."""
    out = {}  # type: Dict[str, Dict[str, Any]]
    for task, body in selection["tasks"].items():
        for layout in body["layouts"]:
            for instance_id, meta in layout["instances"].items():
                out[instance_id] = dict(meta, task=task, suite=body["suite"], demo=layout["demo"])
    return out


def cell_sort_key(cell: str) -> Tuple[int, str]:
    family = cell.partition("@")[0]
    return (FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER), cell)


def target_sort_key(target: str) -> float:
    return float(target)


def _row(records: Sequence[Mapping[str, Any]], metas: Mapping[str, Mapping[str, Any]],
         controls: Mapping[Tuple[str, str, int], Mapping[str, Any]], settings: ReportSettings,
         layouts: Tuple[int, int]) -> Dict[str, Any]:
    """One table row from the hazard records of the row; ``layouts`` = (layouts with an admitted instance, layouts)."""
    method, conf = settings.interval_method, settings.confidence
    main = [r for r in records if placement(r) == MAIN]
    metrics = group_metrics(main, records, controls, settings) if records else None
    contact = lambda r: r["log"].get("first_official_contact_t_s") is not None  # noqa: E731
    success = lambda r: r["success_t_s"] is not None  # noqa: E731
    g1 = [metas[r["instance_id"]].get("g1_passed") for r in records]
    labels = {}  # type: Dict[str, int]
    for r in records:
        labels[str(r["label"])] = labels.get(str(r["label"]), 0) + 1
    sigmas = [metas[r["instance_id"]].get("sigma_achieved") for r in records]
    sigmas = [s for s in sigmas if s is not None]
    row = {
        "layouts_admitted": layouts[0], "layouts": layouts[1],
        "n_episodes": len(records), "n_main": len(main),
        "success": proportion(sum(success(r) for r in main), len(main), method, conf),
        "contact": proportion(sum(contact(r) for r in main), len(main), method, conf),
        "contact_free_success": proportion(sum(success(r) and not contact(r) for r in main), len(main), method, conf),
        "exposure_low": sum(1 for r in records if r["log"].get("exposure_low") is True),
        "invalid": sum(1 for r in records if r["log"].get("invalid_class") in EXCLUDED_INVALID),
        "g1_passed": sum(1 for g in g1 if g is True), "g1_failed": sum(1 for g in g1 if g is False),
        "labels": dict(sorted(labels.items())),
        "sigma_achieved_mean": (sum(sigmas) / len(sigmas)) if sigmas else None,
        "control_success": (metrics["control_success"] if metrics else proportion(0, 0, method, conf)),
        "n_controls": metrics["n_controls"] if metrics else 0,
        "n_controls_missing": metrics["n_controls_missing"] if metrics else 0}
    return row


def _scope_table(scope_records: Sequence[Mapping[str, Any]], scope_tasks: Sequence[str],
                 selection: Mapping[str, Any], metas: Mapping[str, Mapping[str, Any]],
                 controls: Mapping[Tuple[str, str, int], Mapping[str, Any]], settings: ReportSettings
                 ) -> List[Dict[str, Any]]:
    # layouts / admitted layouts per (cell, target) of the scope
    cells = {}  # type: Dict[Tuple[str, str], List[int]]
    for task in scope_tasks:
        for layout in selection["tasks"][task]["layouts"]:
            for key, admitted in layout["cell_admitted"].items():
                cell, target = key.split("|")
                counts = cells.setdefault((cell, target), [0, 0])
                counts[0] += 1 if admitted else 0
                counts[1] += 1
    by_key = {}  # type: Dict[Tuple[str, str], List[Mapping[str, Any]]]
    for record in scope_records:
        meta = metas[record["instance_id"]]
        by_key.setdefault((meta["cell"], meta["target"]), []).append(record)
    rows = []  # type: List[Dict[str, Any]]
    ordered_cells = sorted({c for c, _ in cells}, key=cell_sort_key)
    total_layouts = [0, 0]
    for cell in ordered_cells:
        cell_records, cell_layouts = [], [0, 0]
        for target in sorted({t for c, t in cells if c == cell}, key=target_sort_key):
            records = by_key.get((cell, target), [])
            admitted, count = cells[(cell, target)]
            rows.append(dict(_row(records, metas, controls, settings, (admitted, count)), cell=cell, target=target))
            cell_records += records
            cell_layouts[0] += admitted
            cell_layouts[1] += count
        rows.append(dict(_row(cell_records, metas, controls, settings, tuple(cell_layouts)), cell=cell, target="all"))
        total_layouts[0] += cell_layouts[0]
        total_layouts[1] += cell_layouts[1]
    rows.append(dict(_row(list(scope_records), metas, controls, settings, tuple(total_layouts)),
                     cell="ALL", target="all"))
    return rows


def build_cell_report(selection: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                      report_settings: ReportSettings, policy: str) -> Dict[str, Any]:
    """The cell tables (overall + per suite), the per-task completeness and the control summary."""
    settings = wilson_settings(report_settings)
    metas = attribution(selection)
    hazards = [r for r in records if r["kind"] == HAZARD and r["instance_id"] in metas]
    stray = sum(1 for r in records if r["kind"] == HAZARD and r["instance_id"] not in metas)
    control_list = [r for r in records if r["kind"] == CONTROL]
    controls = {(c["control_id"], c["policy_name"], c["seed"]): c for c in control_list}
    suites = sorted({b["suite"] for b in selection["tasks"].values()})
    scopes = {"overall": list(selection["tasks"])}
    for suite in suites:
        scopes[suite] = [t for t, b in selection["tasks"].items() if b["suite"] == suite]
    tables = {}
    for scope, tasks in scopes.items():
        scope_records = [r for r in hazards if metas[r["instance_id"]]["task"] in tasks]
        tables[scope] = _scope_table(scope_records, tasks, selection, metas, controls, settings)
    progress = {}
    for task, body in selection["tasks"].items():
        ids = {i for layout in body["layouts"] for i in layout["instances"]}
        control_ids = {c["control_id"] for c in control_list if set(c["paired_instance_ids"]) <= ids}
        progress[task] = {"layouts": len(body["layouts"]),
                          "hazard_expected": sum(l["n_hazard"] for l in body["layouts"]),
                          "hazard_done": sum(1 for r in hazards if r["instance_id"] in ids),
                          "control_expected": sum(l["n_control"] for l in body["layouts"]),
                          "control_done": len(control_ids)}
    valid = [c for c in control_list if c["log"].get("invalid_class") not in EXCLUDED_INVALID]
    return {"policy": policy, "interval": {"method": settings.interval_method, "confidence": settings.confidence},
            "n_hazard_records": len(hazards), "n_stray_hazard_records": stray,
            "n_control_records": len(control_list),
            "controls": {"success": proportion(sum(c["success_t_s"] is not None for c in valid), len(valid),
                                               settings.interval_method, settings.confidence),
                         "n_invalid": len(control_list) - len(valid)},
            "skipped_tasks": dict(selection.get("skipped", {})), "progress": progress, "tables": tables}


def _fmt(p: Mapping[str, Any]) -> str:
    if p["n"] == 0:
        return "0/0"
    return "%d/%d (%.2f [%.2f, %.2f])" % (p["k"], p["n"], p["rate"], p["ci"][0], p["ci"][1])


def _short(p: Mapping[str, Any]) -> str:
    return "%d/%d" % (p["k"], p["n"])


HEADERS = ["cell", "target", "admitted/selected", "episodes", "main", "success", "contact", "contact-free success",
           "low-exposure", "invalid", "G1 pass/fail", "labels", "mean Sigma", "control success"]


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Generated-cell evaluation: %s" % report["policy"], "",
             "Rows are paradigm cells (family@elevation) x Sigma target. `main` is the denominator of the rates: "
             "episodes that are neither low-exposure nor invalid. Rates are k/n (rate [%d%% %s interval]). "
             "`layouts adm/sel` is the layouts that admitted an instance over the selected layouts; in the subtotal and total "
             "rows it counts cell slots (layout x Sigma target, plus x cell in the total row). `G1 pass/fail` is the generation's stop-wait-resume diagnostic (not an admission criterion, D19); "
             "`control success` is over the distinct valid controls paired with the row's main episodes."
             % (round(100 * report["interval"]["confidence"]), report["interval"]["method"].capitalize()), "",
             "Hazard records %d (not attributable to a selected instance: %d), control records %d; controls "
             "overall succeed %s." % (report["n_hazard_records"], report["n_stray_hazard_records"],
                                       report["n_control_records"], _fmt(report["controls"]["success"])), ""]
    if report["skipped_tasks"]:
        lines += ["Tasks left out: " + "; ".join("%s (%s)" % kv for kv in sorted(report["skipped_tasks"].items())), ""]
    for scope, rows in report["tables"].items():
        lines += ["## %s" % ("Overall" if scope == "overall" else scope), "",
                  "| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
        for row in rows:
            subtotal = row["target"] == "all"
            name = ("**%s**" if subtotal else "%s") % row["cell"]
            labels = ", ".join("%s %d" % (k, v) for k, v in row["labels"].items()) or "-"
            sigma = "-" if row["sigma_achieved_mean"] is None else "%.2f" % row["sigma_achieved_mean"]
            lines.append("| " + " | ".join([
                name, row["target"], "%d/%d%s" % (row["layouts_admitted"], row["layouts"], " slots" if subtotal else ""),
                str(row["n_episodes"]),
                str(row["n_main"]), _fmt(row["success"]), _fmt(row["contact"]), _fmt(row["contact_free_success"]),
                str(row["exposure_low"]), str(row["invalid"]), "%d/%d" % (row["g1_passed"], row["g1_failed"]), labels,
                sigma, "%s (%d controls)" % (_fmt(row["control_success"]), row["n_controls"])]) + " |")
        lines.append("")
    lines += ["## Completeness", "", "| task | layouts | hazard done/expected | control done/expected |", "|---|---|---|---|"]
    for task, p in sorted(report["progress"].items()):
        lines.append("| %s | %d | %d/%d | %d/%d |" % (task, p["layouts"], p["hazard_done"], p["hazard_expected"],
                                                      p["control_done"], p["control_expected"]))
    return "\n".join(lines) + "\n"
