"""``dynisafety`` console entry (argparse)."""

from __future__ import annotations

import argparse
import sys

from dynisafety.cli.config_cmd import dispatch_config, register_config_parser
from dynisafety.cli.doctor import dispatch_doctor, register_doctor_parser
from dynisafety.cli.eval import dispatch_eval, register_eval_parser
from dynisafety.cli.report import dispatch_report, register_report_parser
from dynisafety.cli.serve import dispatch_serve, register_serve_parser
from dynisafety.cli.sweep import dispatch_sweep, register_sweep_parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynisafety",
        description="DyniSafety benchmark CLI",
    )
    sub = parser.add_subparsers(dest="command")

    register_eval_parser(sub)
    register_sweep_parser(sub)
    register_serve_parser(sub)
    register_doctor_parser(sub)
    for name, help_text in (
        ("tasks", "List available tasks (stub)"),
        ("models", "List registered models (stub)"),
    ):
        sub.add_parser(name, help=help_text)

    register_config_parser(sub)
    register_report_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "eval":
        return dispatch_eval(args)
    if args.command == "sweep":
        return dispatch_sweep(args)
    if args.command == "serve":
        return dispatch_serve(args)
    if args.command == "doctor":
        return dispatch_doctor(args)
    if args.command == "config":
        return dispatch_config(args)
    if args.command == "report":
        return dispatch_report(args)
    print(f"dynisafety {args.command}: not implemented yet (stub)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
