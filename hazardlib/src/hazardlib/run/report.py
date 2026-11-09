"""``hazardctl report``: run directories -> ``summary.yaml`` + ``report.md`` (R1 item 5).

Inputs are one or more output directories of ``hazardctl run``. They must agree on ``pilot``, ``manifest_hash``,
the benchmark and BD hash, the calibration hash and the episode settings (``run_meta.yaml``); otherwise the
report is refused (pilot and formal results are never mixed, D16 c). The same (episode, policy, seed) in two
directories is refused too (it would be counted twice).

Where every hazard episode goes (exactly one place, nothing is dropped):

1. certificate label ``unsolvable_by_probes`` -> bucket ``unsolvable_by_probes`` (the "unavoidable" bucket, 00 §8);
   ``baseline_unusable`` -> bucket ``baseline_unusable``;
2. ``invalid_class`` ``setup_invalid`` or ``sim_unstable`` -> bucket ``invalid`` (out of the denominator, counted);
3. ``exposure_low`` -> bucket ``exposure_low`` (00 §5.2);
4. everything else -> the **main table**, in a ``certified`` row (``in_main_score``: ``wait_solvable`` /
   ``evade_solvable``) or an ``uncertified`` row (footnoted: solvability not asserted). ``hazard_induced_unstable``
   episodes stay here, in the denominator.

Main-table rows: policy x certificate class x family x tau bin (``run.yaml: report.tau_bin_edges_s``); with a
calibration a second table uses Sigma bins (``report.sigma_bin_edges``; Sigma is flagged as an upper bound when
the T_stop was censored). Per row, over the row's main episodes unless stated:

* ``n_episodes``; ``n_eligible`` = the row's episodes before steps 2-3 (same label class, family and bin);
* ``success``: ``success_t_s`` is not None (``task_success`` true at some control step, as in the gates);
  ``control_success``: the same over the distinct valid controls paired with the row's episodes;
* ``official_contact``: ``first_official_contact_t_s`` is not None;
* ``impulse_analytic_ns`` (J) and ``v_rel_normal_mps`` (v_n): median and quartiles over the episodes with an
  official contact; ``min_separation_m``, ``ttc_min_s``: median (and quartiles) over the episodes that have one;
* ``clamp``: ``constraint_state == clamp``;
* ``toppled`` (episodes where it was monitored), ``control_toppled`` (paired valid controls) and
  ``toppled_net`` = their difference (D10 h; no interval for the difference);
* ``exposure_low``: over the eligible episodes; ``invalid``: ``metric_denominator`` over the eligible episodes.

Every proportion has an interval (``run.yaml: report.interval``, :mod:`hazardlib.run.stats`). Rows are sorted by
policy, certificate class, family and bin, never by a metric. ``engine_peak_force_n`` appears only in the
appendix table (00 §6). Controls get their own table per policy.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from ..gates.decision import MAIN_SCORE_LABELS
from ..semantics.event_log import HazardEventLog
from ..semantics.invalid import metric_denominator
from .plan import CONTROL, HAZARD
from .runner import EPISODES_FILE, RUN_META_FILE, read_episode_records, record_key
from .settings import ReportSettings
from .stats import proportion, quartiles

SUMMARY_FILE = "summary.yaml"
REPORT_FILE = "report.md"
SUMMARY_VERSION = "1"
RUN_IDENTITY = ("pilot", "manifest_hash", "benchmark", "bd_hash", "calibration_hash", "episode_settings")
CERTIFIED = "certified"
UNCERTIFIED = "uncertified"
CERT_CLASSES = (CERTIFIED, UNCERTIFIED)
MAIN = "main"
LABEL_BUCKETS = ("unsolvable_by_probes", "baseline_unusable")
BUCKETS = LABEL_BUCKETS + ("invalid", "exposure_low")
EXCLUDED_INVALID = ("setup_invalid", "sim_unstable")
CLAMP = "clamp"


class ReportInputError(ValueError):
    """The run directories cannot be summarised together (every problem is listed)."""


# ------------------------------------------------------------------------------------------------- loading

def load_runs(run_dirs: Sequence[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(metas, records)`` of the run directories, after the consistency checks of the module docstring."""
    if not run_dirs:
        raise ReportInputError("no run directory given")
    metas, records, problems = [], [], []  # type: List[Dict[str, Any]], List[Dict[str, Any]], List[str]
    for run_dir in run_dirs:
        path = Path(run_dir)
        if not (path / RUN_META_FILE).exists():
            raise ReportInputError("%s has no %s: not a hazardctl run directory" % (path, RUN_META_FILE))
        meta = yaml.safe_load((path / RUN_META_FILE).read_text(encoding="utf-8"))
        meta = dict(meta, run_dir=str(run_dir))
        metas.append(meta)
        for record in read_episode_records(path / EPISODES_FILE, repair=False):
            for name in ("pilot", "manifest_hash"):
                if record.get(name) != meta.get(name):
                    problems.append("%s: record %s has %s=%r, its run_meta.yaml %r"
                                    % (path, record_key(record), name, record.get(name), meta.get(name)))
            records.append(dict(record, run_dir=str(run_dir)))
    for name in RUN_IDENTITY:
        values = {json.dumps(m.get(name), sort_keys=True) for m in metas}
        if len(values) > 1:
            listed = ", ".join("%s: %r" % (m["run_dir"], m.get(name)) for m in metas)
            what = {"pilot": "pilot and formal results cannot be mixed (D16 c)",
                    "manifest_hash": "results of different Manifests cannot be mixed"}.get(name, "they differ")
            problems.append("%s: %s (%s)" % (name, what, listed))
    seen = {}  # type: Dict[Tuple[str, str, int], str]
    for record in records:
        key = record_key(record)
        if key in seen:
            problems.append("episode %s appears in %s and %s" % (key, seen[key], record["run_dir"]))
        seen[key] = record["run_dir"]
    if problems:
        raise ReportInputError("cannot report these runs together: %d problem(s):\n  %s"
                               % (len(problems), "\n  ".join(problems)))
    return metas, records


# ------------------------------------------------------------------------------------------- classification

def bin_labels(edges: Sequence[float]) -> List[str]:
    """Labels in order: ``<e0``, ``[e0,e1)``, ..., ``>=e_last``."""
    edges = list(edges)
    labels = ["<%g" % edges[0]]
    labels += ["[%g,%g)" % (a, b) for a, b in zip(edges, edges[1:])]
    labels.append(">=%g" % edges[-1])
    return labels


def bin_label(value: float, edges: Sequence[float]) -> str:
    labels = bin_labels(edges)
    if value < edges[0]:
        return labels[0]
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return labels[index + 1]
    return labels[-1]


def cert_class(label: Optional[str]) -> Optional[str]:
    if label in MAIN_SCORE_LABELS:
        return CERTIFIED
    if label == UNCERTIFIED:
        return UNCERTIFIED
    return None


def placement(record: Mapping[str, Any]) -> str:
    """``main`` or the bucket of a hazard record (module docstring, steps 1-4)."""
    if cert_class(record["label"]) is None:
        if record["label"] not in LABEL_BUCKETS:
            raise ReportInputError("record %s has an unexpected label %r" % (record_key(record), record["label"]))
        return record["label"]
    log = record["log"]
    if log.get("invalid_class") in EXCLUDED_INVALID:
        return "invalid"
    if log.get("exposure_low") is True:
        return "exposure_low"
    return MAIN


def _valid_control(record: Mapping[str, Any]) -> bool:
    return record["log"].get("invalid_class") not in EXCLUDED_INVALID


# ------------------------------------------------------------------------------------------------ metrics

def _count(records: Iterable[Mapping[str, Any]], test: Callable[[Mapping[str, Any]], bool]) -> int:
    return sum(1 for r in records if test(r))


def group_metrics(main: Sequence[Mapping[str, Any]], eligible: Sequence[Mapping[str, Any]],
                  controls: Mapping[Tuple[str, str, int], Mapping[str, Any]], settings: ReportSettings) -> Dict[str, Any]:
    """Metrics of one row (module docstring); ``controls`` maps (control id, policy, seed) to the control record."""
    method, conf = settings.interval_method, settings.confidence

    def prop(k: int, n: int) -> Dict[str, Any]:
        return proportion(k, n, method, conf)

    paired, missing = {}, 0  # type: Dict[Tuple[str, str, int], Mapping[str, Any]], int
    for record in main:
        key = (record["control_id"], record["policy_name"], record["seed"])
        if key in controls:
            paired[key] = controls[key]
        else:
            missing += 1
    valid_controls = [c for c in paired.values() if _valid_control(c)]
    contact = [r for r in main if r["log"].get("first_official_contact_t_s") is not None]
    topple_h = [r for r in main if r["log"].get("toppled") is not None]
    topple_c = [c for c in valid_controls if c["log"].get("toppled") is not None]
    toppled = prop(_count(topple_h, lambda r: r["log"]["toppled"] is True), len(topple_h))
    control_toppled = prop(_count(topple_c, lambda r: r["log"]["toppled"] is True), len(topple_c))
    net = None
    if toppled["rate"] is not None and control_toppled["rate"] is not None:
        net = toppled["rate"] - control_toppled["rate"]
    denominator = metric_denominator([HazardEventLog.from_dict(r["log"]) for r in eligible])
    return {
        "n_eligible": len(eligible), "n_episodes": len(main),
        "success": prop(_count(main, lambda r: r["success_t_s"] is not None), len(main)),
        "control_success": prop(_count(valid_controls, lambda r: r["success_t_s"] is not None), len(valid_controls)),
        "n_controls": len(paired), "n_controls_invalid": len(paired) - len(valid_controls),
        "n_controls_missing": missing,
        "official_contact": prop(len(contact), len(main)),
        "impulse_analytic_ns": quartiles([r["log"].get("impulse_analytic_ns") for r in contact]),
        "v_rel_normal_mps": quartiles([r["log"].get("v_rel_normal_mps") for r in contact]),
        "min_separation_m": quartiles([r["log"].get("min_separation_m") for r in main]),
        "ttc_min_s": quartiles([r["log"].get("ttc_min_s") for r in main]),
        "clamp": prop(_count(main, lambda r: r["log"].get("constraint_state") == CLAMP), len(main)),
        "toppled": toppled, "control_toppled": control_toppled, "toppled_net": net,
        "exposure_low": prop(_count(eligible, lambda r: r["log"].get("exposure_low") is True), len(eligible)),
        "invalid": {name: getattr(denominator, name) for name in
                    ("n_total", "n_none", "n_setup_invalid", "n_sim_unstable", "n_hazard_induced_unstable",
                     "n_attribution_ambiguous", "n_in_denominator")},
    }


def _sorted_keys(keys: Iterable[Tuple[Any, ...]], orders: Sequence[Optional[Sequence[str]]]) -> List[Tuple[Any, ...]]:
    def rank(key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return tuple(order.index(part) if order is not None else part for part, order in zip(key, orders))
    return sorted(set(keys), key=rank)


def _binned_rows(hazards: Sequence[Mapping[str, Any]], controls: Mapping[Tuple[str, str, int], Mapping[str, Any]],
                 settings: ReportSettings, value_of: Callable[[Mapping[str, Any]], Optional[float]],
                 edges: Sequence[float], bin_name: str) -> List[Dict[str, Any]]:
    groups = {}  # type: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]]
    for record in hazards:
        cls = cert_class(record["label"])
        value = value_of(record)
        if cls is None or value is None:
            continue
        key = (record["policy_name"], cls, record["family"], bin_label(value, edges))
        groups.setdefault(key, []).append(record)
    rows = []
    for key in _sorted_keys(groups, (None, CERT_CLASSES, None, bin_labels(edges))):
        eligible = groups[key]
        main = [r for r in eligible if placement(r) == MAIN]
        row = {"policy": key[0], "cert": key[1], "family": key[2], bin_name: key[3]}
        row.update(group_metrics(main, eligible, controls, settings))
        rows.append(row)
    return rows


def _engine_force_rows(hazards: Sequence[Mapping[str, Any]], edges: Sequence[float]) -> List[Dict[str, Any]]:
    groups = {}  # type: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]]
    for record in hazards:
        cls = cert_class(record["label"])
        if cls is None or placement(record) != MAIN:
            continue
        key = (record["policy_name"], cls, record["family"], bin_label(record["tau_s"], edges))
        groups.setdefault(key, []).append(record)
    rows = []
    for key in _sorted_keys(groups, (None, CERT_CLASSES, None, bin_labels(edges))):
        logs = [r["log"] for r in groups[key]]
        rows.append({"policy": key[0], "cert": key[1], "family": key[2], "tau_bin_s": key[3],
                     "n_episodes": len(logs),
                     "engine_peak_force_n": quartiles([log.get("engine_peak_force_n") for log in logs]),
                     "force_source": sorted({log["force_source"] for log in logs if log.get("force_source")})})
    return rows


def summarize(metas: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
              settings: ReportSettings, report_config_hash: str) -> Dict[str, Any]:
    hazards = [r for r in records if r["kind"] == HAZARD]
    control_list = [r for r in records if r["kind"] == CONTROL]
    controls = {(c["control_id"], c["policy_name"], c["seed"]): c for c in control_list}
    placements = {}  # type: Dict[str, int]
    for record in hazards:
        where = placement(record)
        placements[where] = placements.get(where, 0) + 1
    bucket_groups = {}  # type: Dict[Tuple[str, str, str], List[Mapping[str, Any]]]
    for record in hazards:
        where = placement(record)
        if where != MAIN:
            bucket_groups.setdefault((record["policy_name"], where, record["family"]), []).append(record)
    buckets = []
    for key in _sorted_keys(bucket_groups, (None, BUCKETS, None)):
        row = {"policy": key[0], "bucket": key[1], "family": key[2]}
        row.update(group_metrics(bucket_groups[key], bucket_groups[key], controls, settings))
        buckets.append(row)
    control_rows = []
    for policy in sorted({c["policy_name"] for c in control_list}):
        mine = [c for c in control_list if c["policy_name"] == policy]
        valid = [c for c in mine if _valid_control(c)]
        monitored = [c for c in valid if c["log"].get("toppled") is not None]
        control_rows.append({
            "policy": policy, "n_controls": len(mine),
            "success": proportion(_count(valid, lambda r: r["success_t_s"] is not None), len(valid),
                                  settings.interval_method, settings.confidence),
            "toppled": proportion(_count(monitored, lambda r: r["log"]["toppled"] is True), len(monitored),
                                  settings.interval_method, settings.confidence),
            "invalid": {name: getattr(metric_denominator([HazardEventLog.from_dict(c["log"]) for c in mine]), name)
                        for name in ("n_none", "n_setup_invalid", "n_sim_unstable")}})
    with_sigma = [r for r in hazards if r.get("sigma") is not None]
    first = metas[0]
    return {
        "summary_version": SUMMARY_VERSION,
        "pilot": first.get("pilot"), "manifest_hash": first.get("manifest_hash"),
        "benchmark": first.get("benchmark"), "bd_hash": first.get("bd_hash"),
        "calibration_hash": first.get("calibration_hash"), "episode_settings": first.get("episode_settings"),
        "runs": [{"run_dir": m["run_dir"], "policy_name": m.get("policy_name"),
                  "run_config_hash": m.get("run_config_hash")} for m in metas],
        "report_settings": {"tau_bin_edges_s": list(settings.tau_bin_edges_s),
                            "sigma_bin_edges": list(settings.sigma_bin_edges),
                            "interval_method": settings.interval_method, "confidence": settings.confidence,
                            "run_config_hash": report_config_hash},
        "counts": {"hazard_episodes": len(hazards), "control_episodes": len(control_list),
                   "placement": dict(sorted(placements.items()))},
        "sigma_is_upper_bound": any(r.get("sigma_is_upper_bound") for r in with_sigma) if with_sigma else None,
        "main_tau": _binned_rows(hazards, controls, settings, lambda r: r["tau_s"], settings.tau_bin_edges_s,
                                 "tau_bin_s"),
        "main_sigma": _binned_rows(hazards, controls, settings, lambda r: r.get("sigma"), settings.sigma_bin_edges,
                                   "sigma_bin") if with_sigma else [],
        "buckets": buckets,
        "controls": control_rows,
        "appendix_engine_peak_force": _engine_force_rows(hazards, settings.tau_bin_edges_s),
    }


# ----------------------------------------------------------------------------------------------- markdown

def _num(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.3g" % value


def _prop(p: Mapping[str, Any]) -> str:
    if p["rate"] is None:
        return "n/a (0/0)"
    return "%.2f [%.2f, %.2f] (%d/%d)" % (p["rate"], p["ci"][0], p["ci"][1], p["k"], p["n"])


def _quart(q: Mapping[str, Any]) -> str:
    if q["median"] is None:
        return "n/a"
    return "%s [%s, %s] (n=%d)" % (_num(q["median"]), _num(q["q1"]), _num(q["q3"]), q["n"])


def _invalid(inv: Mapping[str, Any]) -> str:
    return "%d/%d/%d" % (inv["n_setup_invalid"], inv["n_sim_unstable"], inv["n_hazard_induced_unstable"])


def _metric_cells(row: Mapping[str, Any]) -> List[str]:
    net = row["toppled_net"]
    return [str(row["n_episodes"]), _prop(row["success"]), _prop(row["control_success"]),
            _prop(row["official_contact"]), _quart(row["impulse_analytic_ns"]), _quart(row["v_rel_normal_mps"]),
            _num(row["min_separation_m"]["median"]), _num(row["ttc_min_s"]["median"]), _prop(row["clamp"]),
            "%s; net %s" % (_prop(row["toppled"]), "n/a" if net is None else "%+.2f" % net),
            _prop(row["exposure_low"]), _invalid(row["invalid"])]


METRIC_HEADERS = ["n", "success", "control success", "official contact", "J med [IQR] (N·s)",
                  "vₙ med [IQR] (m/s)", "min sep med (m)", "TTC min med (s)", "clamp", "toppled; net vs control",
                  "exposure_low", "invalid setup/sim/hazard"]


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in rows]
    return lines


def render_markdown(summary: Mapping[str, Any]) -> str:
    rs = summary["report_settings"]
    mode = "PILOT (not a formal result)" if summary["pilot"] else "formal, Manifest %s" % summary["manifest_hash"]
    out = ["# Run report", "",
           "- Mode: %s" % mode,
           "- Benchmark: %s (BD %s); calibration %s" % (summary["benchmark"], summary["bd_hash"],
                                                        summary["calibration_hash"]),
           "- Runs: %s" % ", ".join("%s (%s)" % (r["run_dir"], r["policy_name"]) for r in summary["runs"]),
           "- Episodes: %d hazard, %d control; placement %s" % (summary["counts"]["hazard_episodes"],
                                                               summary["counts"]["control_episodes"],
                                                               summary["counts"]["placement"]),
           "- Proportions: rate [%s %g interval] (k/n)" % (rs["interval_method"], rs["confidence"]), ""]

    def cert(row: Mapping[str, Any]) -> str:
        return row["cert"] + ("†" if row["cert"] == UNCERTIFIED else "")

    out += ["## Main table: tau bins", ""]
    out += _table(["policy", "cert", "family", "τ bin (s)"] + METRIC_HEADERS,
                  [[r["policy"], cert(r), r["family"], r["tau_bin_s"]] + _metric_cells(r) for r in summary["main_tau"]])
    if summary["main_sigma"]:
        bound = " (Σ is an upper bound: censored T_stop)" if summary["sigma_is_upper_bound"] else ""
        out += ["", "## Main table: Σ bins%s" % bound, ""]
        out += _table(["policy", "cert", "family", "Σ bin"] + METRIC_HEADERS,
                      [[r["policy"], cert(r), r["family"], r["sigma_bin"]] + _metric_cells(r)
                       for r in summary["main_sigma"]])
    out += ["", "Rows hold the episodes that are certified or uncertified, not exposure_low and not setup_invalid / "
                "sim_unstable; `hazard_induced_unstable` stays in the denominator. `n` counts them; `exposure_low` and "
                "`invalid` are over the row's episodes before those exclusions. J and vₙ are over the episodes with an "
                "official contact. Control success / toppled are over the distinct valid paired controls; the net "
                "topple is hazard rate minus control rate (D10 h). Rows are not ranked.",
            "", "† uncertified: G1 failed and G2 was not evaluated; solvability is not asserted (00 §8).", ""]
    out += ["## Separate buckets", ""]
    out += _table(["policy", "bucket", "family"] + METRIC_HEADERS,
                  [[r["policy"], r["bucket"], r["family"]] + _metric_cells(r) for r in summary["buckets"]])
    out += ["", "## Control episodes", ""]
    out += _table(["policy", "n", "success", "toppled", "invalid none/setup/sim"],
                  [[r["policy"], str(r["n_controls"]), _prop(r["success"]), _prop(r["toppled"]),
                    "%d/%d/%d" % (r["invalid"]["n_none"], r["invalid"]["n_setup_invalid"],
                                  r["invalid"]["n_sim_unstable"])] for r in summary["controls"]])
    out += ["", "## Appendix: engine peak force (auxiliary, not used for ranking, 00 §6)", ""]
    out += _table(["policy", "cert", "family", "τ bin (s)", "n", "engine_peak_force_n med [IQR]", "force_source"],
                  [[r["policy"], cert(r), r["family"], r["tau_bin_s"], str(r["n_episodes"]),
                    _quart(r["engine_peak_force_n"]), ", ".join(r["force_source"]) or "n/a"]
                   for r in summary["appendix_engine_peak_force"]])
    return "\n".join(out) + "\n"


def write_report(out_dir: Any, summary: Mapping[str, Any]) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / SUMMARY_FILE).write_text(yaml.safe_dump(dict(summary), sort_keys=False, allow_unicode=True),
                                    encoding="utf-8")
    (out / REPORT_FILE).write_text(render_markdown(summary), encoding="utf-8")
    return {"summary": str(out / SUMMARY_FILE), "report": str(out / REPORT_FILE)}
