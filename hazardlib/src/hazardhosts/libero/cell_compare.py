"""Compare several policies evaluated on the same generated cells (L6; no simulator).

    python -m hazardhosts.libero.cell_compare --run pi05=<eval dir> --run pi0=<eval dir> --run openvla_oft=<eval dir> \\
        --out _logs/libero40_compare

Each ``--run label=<cell_eval output dir>[:<policy name>]`` is read as ``cell_report`` reads it (its
``selection.json`` and ``runs/<task>/<policy>/episodes.jsonl``; the policy name defaults to the label). The selections
must list the same instances; the first run's selection attributes the instances to cells and suites.

Output ``compare.md`` / ``compare.json``:

* per scope (overall, per suite) and per paradigm cell (``family@elevation``, per Sigma target and as a subtotal),
  for every policy: task success, official contact, contact-free success (Wilson intervals over the ``main``
  episodes, the denominator of ``cell_report``), low-exposure and invalid counts, control success; plus the
  generation's G1 (stop-wait-resume oracle) pass rate of the same instances as a reference column;
* paired comparison by instance id: for every pair of policies, over the instances that are ``main`` under both,
  the contact (and contact-free success) rates, their paired difference with a normal-approximation interval of the
  paired differences, and the exact McNemar test (two-sided exact binomial on the discordant pairs; Holm-adjusted
  p-values within each group of tests). Instances of one layout share a scene and are not independent: the p-values are
  a description of the paired outcomes, not a population claim over layouts.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hazardlib.run.plan import HAZARD
from hazardlib.run.report import MAIN, placement
from hazardlib.run.stats import normal_quantile, proportion

from .cell_report import attribution, build_cell_report, cell_sort_key, wilson_settings


# ------------------------------------------------------------------------------------------- paired statistics

def binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test of ``k`` successes in ``n`` trials at p = 1/2 (the McNemar exact test on the
    discordant pairs): twice the smaller tail, at most 1."""
    if n < 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n, got k=%r n=%r" % (k, n))
    if n == 0:
        return 1.0
    low = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(low + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, in the order given."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def paired_difference(a: Sequence[bool], b: Sequence[bool], confidence: float = 0.95) -> Dict[str, Any]:
    """Paired comparison of two binary outcomes over the same units (``a`` minus ``b``): the 2x2 counts, both rates,
    the difference of the rates with a normal-approximation interval (variance of the paired differences) and the
    exact McNemar p-value."""
    if len(a) != len(b):
        raise ValueError("paired outcomes need equal length")
    n = len(a)
    both = sum(1 for x, y in zip(a, b) if x and y)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    neither = n - both - only_a - only_b
    out = {"n": n, "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
           "rate_a": None, "rate_b": None, "diff": None, "ci": None, "p": binom_two_sided_p(only_a, only_a + only_b)}
    if n:
        diff = (only_a - only_b) / float(n)
        variance = max(0.0, (only_a + only_b) / float(n) - diff * diff) / n
        z = normal_quantile(1.0 - (1.0 - confidence) / 2.0)
        out.update(rate_a=(both + only_a) / float(n), rate_b=(both + only_b) / float(n), diff=diff,
                   ci=[diff - z * math.sqrt(variance), diff + z * math.sqrt(variance)])
    return out


def contact(record: Mapping[str, Any]) -> bool:
    return record["log"].get("first_official_contact_t_s") is not None


def contact_free_success(record: Mapping[str, Any]) -> bool:
    return record["success_t_s"] is not None and not contact(record)


OUTCOMES = {"contact": contact, "contact_free_success": contact_free_success}  # type: Dict[str, Callable[..., bool]]


def main_by_instance(records: Sequence[Mapping[str, Any]], metas: Mapping[str, Mapping[str, Any]]
                     ) -> Dict[str, Mapping[str, Any]]:
    """The hazard records of selected instances that are ``main`` (neither low-exposure nor invalid), by instance id."""
    return {r["instance_id"]: r for r in records
            if r["kind"] == HAZARD and r["instance_id"] in metas and placement(r) == MAIN}


def paired_groups(metas: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """``scope -> group -> instance ids``: overall / per suite / per cell (``family@elevation``) of the overall scope."""
    groups = {"overall": {"ALL": sorted(metas)}}  # type: Dict[str, Dict[str, List[str]]]
    for suite in sorted({m["suite"] for m in metas.values()}):
        groups.setdefault("suite", {})[suite] = sorted(i for i, m in metas.items() if m["suite"] == suite)
    for cell in sorted({m["cell"] for m in metas.values()}, key=cell_sort_key):
        groups.setdefault("cell", {})[cell] = sorted(i for i, m in metas.items() if m["cell"] == cell)
    return groups


def paired_tests(per_policy: Mapping[str, Mapping[str, Mapping[str, Any]]], metas: Mapping[str, Mapping[str, Any]],
                 confidence: float = 0.95) -> Dict[str, Any]:
    """``outcome -> "A vs B" -> scope -> group -> paired_difference`` (+ ``p_holm`` over the groups of a scope)."""
    labels = list(per_policy)
    groups = paired_groups(metas)
    out = {}  # type: Dict[str, Any]
    for outcome, test in OUTCOMES.items():
        by_pair = {}  # type: Dict[str, Any]
        for i, first in enumerate(labels):
            for second in labels[i + 1:]:
                scopes = {}  # type: Dict[str, Any]
                for scope, members in groups.items():
                    rows = {}
                    for group, ids in members.items():
                        shared = [x for x in ids if x in per_policy[first] and x in per_policy[second]]
                        rows[group] = paired_difference([test(per_policy[first][x]) for x in shared],
                                                        [test(per_policy[second][x]) for x in shared], confidence)
                    adjusted = holm([row["p"] for row in rows.values()])
                    for row, p in zip(rows.values(), adjusted):
                        row["p_holm"] = p
                    scopes[scope] = rows
                by_pair["%s vs %s" % (first, second)] = scopes
        out[outcome] = by_pair
    return out


# ------------------------------------------------------------------------------------------------- loading

def parse_run(spec: str) -> Tuple[str, str, str]:
    """``label=dir[:policy]`` -> (label, dir, policy)."""
    label, sep, rest = spec.partition("=")
    if not sep or not label or not rest:
        raise ValueError("--run expects label=dir[:policy], got %r" % spec)
    directory, _, policy = rest.partition(":")
    return label, directory, policy or label


def load_run(directory: str, policy: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Any]:
    """``(selection, records, report settings)`` of one ``cell_eval`` output directory."""
    from hazardlib.run import report as core
    from hazardlib.run.settings import RunSettings
    from .cell_eval import run_dir
    out = Path(directory)
    selection = json.loads((out / "selection.json").read_text())
    settings = RunSettings.load(out / "config")
    dirs = [run_dir(out, t, policy) for t in selection["tasks"] if (run_dir(out, t, policy) / "episodes.jsonl").is_file()]
    _, records = core.load_runs(dirs)
    return selection, records, settings.report


def selection_instances(selection: Mapping[str, Any]) -> Dict[str, List[str]]:
    return {task: sorted(i for layout in body["layouts"] for i in layout["instances"])
            for task, body in selection["tasks"].items()}


# ------------------------------------------------------------------------------------------------- the tables

def build_comparison(runs: Mapping[str, Tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Any]]) -> Dict[str, Any]:
    """The comparison document of ``runs`` (label -> (selection, records, report settings)), in label order."""
    labels = list(runs)
    reference = runs[labels[0]][0]
    expected = selection_instances(reference)
    for label in labels[1:]:
        if selection_instances(runs[label][0]) != expected:
            raise ValueError("selection of %s lists other instances than that of %s" % (label, labels[0]))
    metas = attribution(reference)
    reports = {label: build_cell_report(runs[label][0], runs[label][1], runs[label][2], label) for label in labels}
    settings = wilson_settings(runs[labels[0]][2])
    tables = {}  # type: Dict[str, List[Dict[str, Any]]]
    for scope in reports[labels[0]]["tables"]:
        rows = []
        for index, row in enumerate(reports[labels[0]]["tables"][scope]):
            entry = {"cell": row["cell"], "target": row["target"], "layouts_admitted": row["layouts_admitted"],
                     "layouts": row["layouts"], "n_instances": row["n_episodes"], "g1_passed": row["g1_passed"],
                     "g1_failed": row["g1_failed"], "g1": proportion(row["g1_passed"], row["g1_passed"] + row["g1_failed"],
                                                                      settings.interval_method, settings.confidence),
                     "policies": {}}
            for label in labels:
                other = reports[label]["tables"][scope][index]
                if (other["cell"], other["target"]) != (row["cell"], row["target"]):
                    raise ValueError("table rows of %s do not line up with %s" % (label, labels[0]))
                entry["policies"][label] = {k: other[k] for k in (
                    "n_episodes", "n_main", "success", "contact", "contact_free_success", "exposure_low", "invalid",
                    "control_success", "n_controls")}
            rows.append(entry)
        tables[scope] = rows
    per_policy = {label: main_by_instance(runs[label][1], metas) for label in labels}
    return {"labels": labels, "interval": reports[labels[0]]["interval"],
            "controls": {label: reports[label]["controls"] for label in labels},
            "n_hazard_records": {label: reports[label]["n_hazard_records"] for label in labels},
            "n_instances": len(metas), "n_main_all_policies": len(set.intersection(*[set(v) for v in per_policy.values()])),
            "tables": tables, "paired": paired_tests(per_policy, metas, settings.confidence)}


def _fmt(p: Mapping[str, Any]) -> str:
    if p["n"] == 0:
        return "-"
    return "%d/%d %.2f [%.2f, %.2f]" % (p["k"], p["n"], p["rate"], p["ci"][0], p["ci"][1])


def _row_name(row: Mapping[str, Any]) -> str:
    return ("**%s**" if row["target"] == "all" else "%s") % row["cell"]


def _render_metric(rows: Sequence[Mapping[str, Any]], labels: Sequence[str], key: str, with_g1: bool,
                   only_subtotals: bool) -> List[str]:
    header = ["cell", "target", "instances"] + (["G1 oracle pass"] if with_g1 else []) + list(labels)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        if only_subtotals and row["target"] != "all":
            continue
        cells = [_row_name(row), row["target"], str(row["n_instances"])]
        if with_g1:
            cells.append(_fmt(row["g1"]))
        cells += [_fmt(row["policies"][label][key]) for label in labels]
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def _render_exclusions(rows: Sequence[Mapping[str, Any]], labels: Sequence[str], only_subtotals: bool) -> List[str]:
    header = ["cell", "target", "instances"] + ["%s %s" % (label, what) for label in labels
                                                for what in ("main / low-exposure / invalid", "control success")]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        if only_subtotals and row["target"] != "all":
            continue
        cells = [_row_name(row), row["target"], str(row["n_instances"])]
        for label in labels:
            p = row["policies"][label]
            cells += ["%d / %d / %d" % (p["n_main"], p["exposure_low"], p["invalid"]),
                      "%s (%d)" % (_fmt(p["control_success"]), p["n_controls"])]
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def _render_paired(comparison: Mapping[str, Any], outcome: str) -> List[str]:
    lines = []
    for pair, scopes in comparison["paired"][outcome].items():
        first, second = pair.split(" vs ")
        lines += ["#### %s: %s" % (outcome.replace("_", "-"), pair), "",
                  "| scope | group | n pairs | %s only | %s only | both | neither | %s rate | %s rate | diff [95%% CI] | "
                  "p exact | p Holm |" % (first, second, first, second), "|" + "---|" * 12]
        for scope, rows in scopes.items():
            for group, r in rows.items():
                if r["n"] == 0:
                    continue
                lines.append("| %s | %s | %d | %d | %d | %d | %d | %.2f | %.2f | %+.3f [%+.3f, %+.3f] | %.3g | %.3g |" % (
                    scope, group, r["n"], r["only_a"], r["only_b"], r["both"], r["neither"], r["rate_a"], r["rate_b"],
                    r["diff"], r["ci"][0], r["ci"][1], r["p"], r["p_holm"]))
        lines.append("")
    return lines


def render_markdown(comparison: Mapping[str, Any]) -> str:
    labels = comparison["labels"]
    interval = comparison["interval"]
    lines = ["# Policy comparison on the generated cells: %s" % ", ".join(labels), "",
             "Same %d selected instances for every policy. Rates are k/n rate [%d%% %s interval] over the `main` "
             "episodes of the row (neither low-exposure nor invalid; the denominators differ per policy and are in "
             "the exclusion tables). `G1 oracle pass` is the generation's stop-wait-resume diagnostic of the same "
             "instances (policy independent, over the row's instances; it is the reference for contact-free "
             "success). Control success is over the distinct valid controls paired with the row's main episodes "
             "(number of controls in brackets)." % (comparison["n_instances"], round(100 * interval["confidence"]),
                                                    interval["method"].capitalize()), "",
             "Controls overall: " + "; ".join("%s %s" % (label, _fmt(comparison["controls"][label]["success"]))
                                              for label in labels) + ".",
             "Instances that are `main` under every policy: %d." % comparison["n_main_all_policies"], ""]
    for scope, rows in comparison["tables"].items():
        title = "Overall" if scope == "overall" else scope
        subtotals = scope != "overall"
        lines += ["## %s" % title, ""]
        if subtotals:
            lines += ["(cell subtotals over all Sigma targets; per-Sigma rows are in the JSON)", ""]
        for key, name, g1 in (("contact", "Official contact", False), ("contact_free_success", "Contact-free success", True),
                              ("success", "Task success", False)):
            lines += ["### %s: %s" % (title, name), ""] + _render_metric(rows, labels, key, g1, subtotals)
        lines += ["### %s: exclusions and controls" % title, ""] + _render_exclusions(rows, labels, subtotals)
    lines += ["## Paired comparison by instance id", "",
              "Instances that are `main` under both policies of a pair; `only X` = contact (or contact-free success) "
              "under X but not under the other. diff = rate(first) - rate(second) with a normal-approximation "
              "interval of the paired differences; p = exact McNemar test; Holm adjusts within one table scope. "
              "Instances of one layout are correlated, so read p as a description of the paired outcomes.", ""]
    for outcome in OUTCOMES:
        lines += _render_paired(comparison, outcome)
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="label=dir[:policy]; the first is the reference")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    runs = {}
    for spec in args.run:
        label, directory, policy = parse_run(spec)
        if label in runs:
            raise SystemExit("duplicate label %s" % label)
        runs[label] = load_run(directory, policy)
    comparison = build_comparison(runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(comparison, indent=1, sort_keys=True))
    (out / "compare.md").write_text(render_markdown(comparison))
    print("wrote", out / "compare.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
