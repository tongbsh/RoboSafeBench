"""The JSON report of a conformance run (T05) and its reader.

Layout (``schema`` / ``schema_version`` identify it; a reader must reject other versions)::

    {
      "schema": "hazardlib.conformance.report", "schema_version": "1",
      "hazardlib_version": "...", "host": "<registry name>", "adapter_version": "...",
      "bd_name": "...", "bd_hash": "<sha256 of the BD record>", "tier": "A" | "B" | "C" | "undetermined",
      "config_source": "<config dir>", "config_hashes": {"conformance": "...", ...},
      "runtime_config": {"<field>": {"value": ..., "source": "<file>:<key>"}},
      "verdict": "pass" | "fail" | "unverified",
      "summary": {"pass": n, "fail": n, "error": n, "skip": n, "unverified": n},
      "checks": [{"id": "C1", "title": "...", "claim": "...", "status": "...", "reason": "..." | null,
                  "measured": {...}, "tolerances": {"<section>.<key>": value}, "duration_s": float}]
    }

``verdict`` is ``fail`` if any check failed or errored, else ``unverified`` if any check could not be verified on
this host, else ``pass``. A ``skip`` always carries the reason the rules allow it (01 §5: C6 on Tier B, C12 on a
single-environment host).
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Union

REPORT_SCHEMA = "hazardlib.conformance.report"
REPORT_SCHEMA_VERSION = "1"

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_ERROR = "error"
STATUS_SKIP = "skip"
STATUS_UNVERIFIED = "unverified"
STATUSES = (STATUS_PASS, STATUS_FAIL, STATUS_ERROR, STATUS_SKIP, STATUS_UNVERIFIED)

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_UNVERIFIED = "unverified"

TOP_LEVEL_KEYS = ("schema", "schema_version", "hazardlib_version", "host", "adapter_version", "bd_name", "bd_hash",
                  "tier", "config_source", "config_hashes", "runtime_config", "verdict", "summary", "checks")
CHECK_KEYS = ("id", "title", "claim", "status", "reason", "measured", "tolerances", "duration_s")


class ReportError(ValueError):
    """A report that does not follow schema version 1."""


def verdict_of(statuses: List[str]) -> str:
    if any(s in (STATUS_FAIL, STATUS_ERROR) for s in statuses):
        return VERDICT_FAIL
    if any(s == STATUS_UNVERIFIED for s in statuses):
        return VERDICT_UNVERIFIED
    return VERDICT_PASS


def validate_report(data: Any) -> Dict[str, Any]:
    """Check the layout above (keys, statuses, reasons on non-passing checks, consistent summary and verdict)
    and return ``data``."""
    if not isinstance(data, dict):
        raise ReportError("a conformance report must be a JSON object")
    if data.get("schema") != REPORT_SCHEMA:
        raise ReportError("not a conformance report (schema=%r)" % (data.get("schema"),))
    if data.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ReportError("conformance report schema_version %r is not supported (expected %r)"
                          % (data.get("schema_version"), REPORT_SCHEMA_VERSION))
    missing = [k for k in TOP_LEVEL_KEYS if k not in data]
    if missing:
        raise ReportError("conformance report lacks %s" % missing)
    checks = data["checks"]
    if not isinstance(checks, list):
        raise ReportError("checks must be a list")
    counts = {s: 0 for s in STATUSES}
    seen = set()
    for item in checks:
        if not isinstance(item, dict) or [k for k in CHECK_KEYS if k not in item]:
            raise ReportError("every check needs the keys %s, got %r" % (list(CHECK_KEYS), item))
        if item["status"] not in STATUSES:
            raise ReportError("check %r has status %r, not one of %s" % (item["id"], item["status"], STATUSES))
        if item["id"] in seen:
            raise ReportError("check %r appears twice" % item["id"])
        seen.add(item["id"])
        if item["status"] != STATUS_PASS and not item["reason"]:
            raise ReportError("check %r is %s without a reason" % (item["id"], item["status"]))
        counts[item["status"]] += 1
    if data["summary"] != counts:
        raise ReportError("summary %r does not match the checks %r" % (data["summary"], counts))
    expected = verdict_of([item["status"] for item in checks])
    if data["verdict"] != expected:
        raise ReportError("verdict %r does not match the checks (%r)" % (data["verdict"], expected))
    return data


def read_report(path: Union[str, Path]) -> Dict[str, Any]:
    with open(str(path), "r", encoding="utf-8") as handle:
        return validate_report(json.load(handle))


def write_report(data: Dict[str, Any], path: Union[str, Path]) -> None:
    validate_report(data)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
