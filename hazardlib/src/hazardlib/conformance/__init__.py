"""Conformance tooling (01 §5). T04 provides the simulator-free ``FakeHost``; T05 the suite C1-C13.

Run it with ``python -m hazardlib.conformance --host <name> --out report.json`` or
``pytest hazardlib/src/hazardlib/conformance/tests --host=<name>``. Adapter packages make themselves known with
``register_host(name, factory)`` at import time (``hazardhosts.<name>`` is imported on demand).
"""
from .fake_target import make_fake_target
from .fakehost import (ACTION_MODES, MUTATIONS, TABLE_NAME, FakeAction, FakeHost, FakeSphere, FakeTask,
                       FakeTaskObject, FakeWorldConfig)
from .report import REPORT_SCHEMA, REPORT_SCHEMA_VERSION, ReportError, read_report, validate_report
from .settings import ConformanceConfig
from .suite import Check, SuiteSession, load_checks, run_suite
from .target import ConformanceTarget, EpisodeRef, Scenario, StrikeTarget, register_host, registered_hosts, \
    resolve_target

__all__ = ["ACTION_MODES", "MUTATIONS", "TABLE_NAME", "FakeAction", "FakeHost", "FakeSphere", "FakeTask",
           "FakeTaskObject", "FakeWorldConfig", "make_fake_target", "REPORT_SCHEMA", "REPORT_SCHEMA_VERSION",
           "ReportError", "read_report", "validate_report", "ConformanceConfig", "Check", "SuiteSession",
           "load_checks", "run_suite", "ConformanceTarget", "EpisodeRef", "Scenario", "StrikeTarget",
           "register_host", "registered_hosts", "resolve_target"]
