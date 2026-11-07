"""``python -m hazardlib.conformance --host <name> [--out report.json]``: run C1-C13 and print one line per check.

Exit status: 0 when every check passed (or was skipped as 01 §5 allows), 1 when any check failed or errored, 2 when
nothing failed but some check could not be verified on this host.
"""
import argparse
import sys
from typing import List, Optional

from .report import VERDICT_FAIL, VERDICT_PASS, VERDICT_UNVERIFIED, write_report
from .suite import SuiteSession, run_suite

EXIT_CODES = {VERDICT_PASS: 0, VERDICT_FAIL: 1, VERDICT_UNVERIFIED: 2}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hazardlib.conformance", description=__doc__.split("\n")[0])
    parser.add_argument("--host", required=True, help="registry name of the host (built in: fake)")
    parser.add_argument("--host-module", action="append", default=[],
                        help="module to import first (it calls register_host); repeatable")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--select", default=None, help="comma-separated check ids, e.g. C1,C5")
    parser.add_argument("--config-dir", default=None, help="configs/hazard directory (default: auto-detected)")
    args = parser.parse_args(argv)
    select = [s.strip() for s in args.select.split(",") if s.strip()] if args.select else None
    session = SuiteSession.open(args.host, args.host_module, args.config_dir)
    report = run_suite(session, select)
    width = max(len(c["title"]) for c in report["checks"]) if report["checks"] else 0
    for check in report["checks"]:
        line = "%-4s %-*s %-10s" % (check["id"], width, check["title"], check["status"].upper())
        if check["reason"]:
            line += " " + check["reason"]
        print(line)
    print("host=%s adapter=%s bd=%s bd_hash=%s verdict=%s %s"
          % (report["host"], report["adapter_version"], report["bd_name"], report["bd_hash"][:12],
             report["verdict"].upper(), report["summary"]))
    if args.out:
        write_report(report, args.out)
        print("report written to %s" % args.out)
    return EXIT_CODES[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
