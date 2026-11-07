"""Runner of the conformance checks C1-C13 (01 §5, T05), shared by ``python -m hazardlib.conformance`` and the
pytest files in ``conformance/tests`` (one check per file; each file defines ``CHECK`` and a test that executes it).

A :class:`Check` is a function of a :class:`CheckContext`. The context gives the target, its BD, the settings and
fresh :class:`~hazardlib.conformance.harness.Probe` objects, and collects the outcome:

* ``expect(ok, message)``: a soft requirement (the check goes on and reports every violation);
* ``require(ok, message)``: a hard requirement (the check stops here);
* ``unverified(message)``: the property cannot be verified on this host (e.g. C13 without a repeatable replay);
  never used for a missing adapter capability, which is a failure;
* ``record(key, value)``: a measured value for the report; ``number(section, key)``: a tolerance from
  ``conformance.yaml``, recorded with the outcome.

Statuses: ``pass``, ``fail`` (a requirement was violated or a needed setting is unresolved), ``error`` (the
check crashed: the host raised, or a contract method returned something malformed), ``skip`` (only where 01 §5
allows it, with the reason), ``unverified``.
"""
import dataclasses
import importlib
import math
import numbers
import pkgutil
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .. import __version__ as HAZARDLIB_VERSION
from ..config import ConfigError
from ..host import CapabilityError, Tier
from ..spec.benchmark_descriptor import TierUndeterminedError
from . import fake_target  # noqa: F401 - registers the built-in host "fake"
from .harness import Probe
from .report import (REPORT_SCHEMA, REPORT_SCHEMA_VERSION, STATUS_ERROR, STATUS_FAIL, STATUS_PASS, STATUS_SKIP,
                     STATUS_UNVERIFIED, STATUSES, verdict_of, write_report)
from .settings import ConformanceConfig
from .target import ConformanceTarget, resolve_target

CHECKS_PACKAGE = "hazardlib.conformance.tests"


class CheckFailed(AssertionError):
    """A hard requirement of a check was violated."""


class CheckUnverified(Exception):
    """The property of a check cannot be verified on this host."""


@dataclasses.dataclass(frozen=True)
class Check:
    """``skip_when(session)`` returns the reason when 01 §5 allows skipping on this host, else ``None``."""
    check_id: str
    title: str
    claim: str
    run: Callable[["CheckContext"], None]
    skip_when: Optional[Callable[["SuiteSession"], Optional[str]]] = None

    @property
    def number(self) -> int:
        return int(self.check_id[1:])


def jsonable(value: Any) -> Any:
    """Plain JSON types; non-finite floats become the strings ``"nan"`` / ``"inf"`` / ``"-inf"``."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, str):
        return value
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):  # enums
        return value.value
    return repr(value)


@dataclasses.dataclass
class CheckOutcome:
    check_id: str
    title: str
    claim: str
    status: str
    reason: Optional[str]
    measured: Dict[str, Any]
    tolerances: Dict[str, Any]
    duration_s: float
    exc_info: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.check_id, "title": self.title, "claim": self.claim, "status": self.status,
                "reason": self.reason, "measured": jsonable(self.measured), "tolerances": jsonable(self.tolerances),
                "duration_s": self.duration_s}


class CheckContext:
    def __init__(self, session: "SuiteSession", check: Check) -> None:
        self.session = session
        self.check = check
        self.target = session.target  # type: ConformanceTarget
        self.host = session.target.host
        self.scenario = session.target.scenario
        self.bd = session.bd
        self.config = session.config
        self.runtime_cfg = session.runtime_cfg
        self.measured = {}  # type: Dict[str, Any]
        self.tolerances = {}  # type: Dict[str, Any]
        self.failures = []  # type: List[str]
        self.notes = []  # type: List[str]

    # -- settings ------------------------------------------------------------------------------------------

    def number(self, section: str, key: str) -> float:
        value = self.config.number(section, key)
        self.tolerances["%s.%s" % (section, key)] = value
        return value

    def integer(self, section: str, key: str) -> int:
        value = self.config.integer(section, key)
        self.tolerances["%s.%s" % (section, key)] = value
        return value

    def value(self, section: str, key: str) -> Any:
        value = self.config.value(section, key)
        self.tolerances["%s.%s" % (section, key)] = value
        return value

    def number_if_set(self, section: str, key: str) -> Optional[float]:
        """The number at ``section.key``, or None while it is still a placeholder (recorded as such)."""
        if not self.config.is_set(section, key):
            self.tolerances["%s.%s" % (section, key)] = self.config.raw(section).get(key)
            return None
        return self.number(section, key)

    # -- host helpers ------------------------------------------------------------------------------------

    def probe(self) -> Probe:
        return Probe(self.host, self.runtime_cfg)

    @property
    def physics_dt(self) -> float:
        return float(self.host.time_state(0).physics_dt_s)

    def on_grid(self, t: float) -> float:
        """``t`` rounded to the host's physics-step grid (episode times of hook starts are multiples of the
        physics step), so that probe events fall where the check expects them relative to the hooks."""
        dt = self.physics_dt
        return round(t / dt) * dt

    # -- outcome -----------------------------------------------------------------------------------------

    def record(self, key: str, value: Any) -> None:
        self.measured[key] = value

    def note(self, message: str) -> None:
        self.notes.append(message)

    def expect(self, ok: Any, message: str) -> bool:
        if not ok:
            self.failures.append(message)
        return bool(ok)

    def require(self, ok: Any, message: str) -> None:
        if not ok:
            raise CheckFailed(message)

    def unverified(self, message: str) -> None:
        raise CheckUnverified(message)


class SuiteSession:
    """One target, its BD and settings, and the outcomes of the checks run so far."""

    def __init__(self, target: ConformanceTarget, config: Optional[ConformanceConfig] = None) -> None:
        if not isinstance(target, ConformanceTarget):
            raise TypeError("target must be a ConformanceTarget, got %r" % (type(target).__name__,))
        self.target = target
        self.config = config if config is not None else ConformanceConfig.load()
        self.bd = target.host.describe()
        self.runtime_cfg, self.runtime_sources = self.config.runtime_config()
        self.outcomes = {}  # type: Dict[str, CheckOutcome]

    @classmethod
    def open(cls, host: str, modules: Sequence[str] = (), config_dir: Optional[str] = None) -> "SuiteSession":
        config = ConformanceConfig.load(config_dir)
        return cls(resolve_target(host, modules), config)

    @property
    def tier(self) -> Optional[Tier]:
        """The BD's tier, or None while tier-relevant fields are unknown (never guessed)."""
        try:
            return self.bd.tier()
        except TierUndeterminedError:
            return None

    def run(self, check: Check) -> CheckOutcome:
        started = time.perf_counter()
        ctx = CheckContext(self, check)
        exc_info = None
        try:
            reason = check.skip_when(self) if check.skip_when is not None else None
        except Exception as exc:  # noqa: BLE001 - reported like a crashing check
            reason, exc_info = "%s: %s" % (type(exc).__name__, exc), traceback.format_exc()
        if exc_info is not None:
            status = STATUS_ERROR
        elif reason is not None:
            status = STATUS_SKIP
        else:
            try:
                check.run(ctx)
                status, reason = STATUS_PASS, None
                if ctx.failures:
                    status, reason = STATUS_FAIL, "; ".join(ctx.failures)
            except CheckFailed as exc:
                status, reason = STATUS_FAIL, "; ".join(ctx.failures + [str(exc)])
            except CheckUnverified as exc:
                if ctx.failures:
                    status, reason = STATUS_FAIL, "; ".join(ctx.failures)
                else:
                    status, reason = STATUS_UNVERIFIED, str(exc)
            except ConfigError as exc:
                status, reason = STATUS_FAIL, "settings: %s" % exc
            except CapabilityError as exc:
                status, reason = STATUS_FAIL, "; ".join(ctx.failures + ["capability: %s" % exc])
            except Exception as exc:  # noqa: BLE001 - a crashing check is reported, not raised
                status, reason = STATUS_ERROR, "%s: %s" % (type(exc).__name__, exc)
                exc_info = traceback.format_exc()
        if ctx.notes:
            ctx.measured["notes"] = list(ctx.notes)
        outcome = CheckOutcome(check.check_id, check.title, check.claim, status, reason, ctx.measured,
                               ctx.tolerances, time.perf_counter() - started, exc_info)
        self.outcomes[check.check_id] = outcome
        return outcome

    def report(self) -> Dict[str, Any]:
        outcomes = sorted(self.outcomes.values(), key=lambda o: int(o.check_id[1:]))
        summary = {s: 0 for s in STATUSES}
        for outcome in outcomes:
            summary[outcome.status] += 1
        return {
            "schema": REPORT_SCHEMA, "schema_version": REPORT_SCHEMA_VERSION,
            "hazardlib_version": HAZARDLIB_VERSION, "host": self.target.name,
            "adapter_version": self.target.adapter_version, "bd_name": self.bd.name, "bd_hash": self.bd.hash(),
            "tier": self.tier.value if self.tier is not None else "undetermined", "config_source": self.config.source,
            "config_hashes": self.config.config_hashes(), "runtime_config": jsonable(self.runtime_sources),
            "verdict": verdict_of([o.status for o in outcomes]), "summary": summary,
            "checks": [o.to_dict() for o in outcomes],
        }

    def write_report(self, path: str) -> Dict[str, Any]:
        data = self.report()
        write_report(data, path)
        return data


def load_checks(select: Optional[Sequence[str]] = None) -> List[Check]:
    """Every ``CHECK`` defined by a ``test_cNN_*`` module of the checks package, in C-number order;
    ``select`` keeps the given ids (``["C1", "C5"]``)."""
    package = importlib.import_module(CHECKS_PACKAGE)
    checks = []
    for info in pkgutil.iter_modules(package.__path__):
        if not info.name.startswith("test_c"):
            continue
        module = importlib.import_module("%s.%s" % (CHECKS_PACKAGE, info.name))
        check = getattr(module, "CHECK", None)
        if not isinstance(check, Check):
            raise TypeError("%s defines no CHECK" % module.__name__)
        checks.append(check)
    checks.sort(key=lambda c: c.number)
    ids = [c.check_id for c in checks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate check ids %r" % ids)
    if select:
        unknown = sorted(set(select) - set(ids))
        if unknown:
            raise ValueError("unknown check id(s) %s (known: %s)" % (unknown, ids))
        checks = [c for c in checks if c.check_id in set(select)]
    return checks


def run_suite(session: SuiteSession, select: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    for check in load_checks(select):
        session.run(check)
    return session.report()


__all__ = ["Check", "CheckContext", "CheckFailed", "CheckOutcome", "CheckUnverified", "SuiteSession", "jsonable",
           "load_checks", "run_suite", "STATUS_PASS", "STATUS_FAIL", "STATUS_ERROR", "STATUS_SKIP",
           "STATUS_UNVERIFIED"]
