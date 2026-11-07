"""pytest glue of the conformance suite: ``pytest hazardlib/src/hazardlib/conformance/tests --host=<name>``.

Options: ``--host`` (registry name, default ``fake``), ``--host-module`` (modules to import first, repeatable;
they call ``register_host``), ``--conformance-report PATH`` (write the JSON report at the end). Outcomes map to
pytest as: pass -> passed, skip -> skipped (with the reason 01 §5 allows), fail / unverified -> failed (with the
reason; ``unverified`` is not a pass), error -> the original exception.

This is the only module of hazardlib that imports pytest (the hygiene test whitelists it).
"""
import pytest

from hazardlib.conformance.suite import (STATUS_ERROR, STATUS_FAIL, STATUS_PASS, STATUS_SKIP, STATUS_UNVERIFIED,
                                         SuiteSession)

DEFAULT_HOST = "fake"


def pytest_addoption(parser):
    group = parser.getgroup("hazardlib-conformance")
    group.addoption("--host", action="store", default=DEFAULT_HOST,
                    help="conformance host registry name (default: %s)" % DEFAULT_HOST)
    group.addoption("--host-module", action="append", default=[],
                    help="module to import before resolving --host (it calls register_host); repeatable")
    group.addoption("--conformance-report", action="store", default=None,
                    help="write the JSON conformance report to this path")


class _Runner:
    def __init__(self, session):
        self.session = session

    def execute(self, check):
        outcome = self.session.run(check)
        if outcome.status == STATUS_PASS:
            return outcome
        if outcome.status == STATUS_SKIP:
            pytest.skip(outcome.reason)
        if outcome.status in (STATUS_FAIL, STATUS_UNVERIFIED):
            pytest.fail("%s %s: %s" % (check.check_id, outcome.status.upper(), outcome.reason), pytrace=False)
        assert outcome.status == STATUS_ERROR
        pytest.fail("%s ERROR: %s\n%s" % (check.check_id, outcome.reason, outcome.exc_info), pytrace=False)


def _option(config, name, default):
    try:
        return config.getoption(name)
    except ValueError:  # the options are registered only when this conftest is an initial conftest
        return default


@pytest.fixture(scope="session")
def conformance(request):
    config = request.config
    host = _option(config, "--host", DEFAULT_HOST)
    modules = _option(config, "--host-module", [])
    session = SuiteSession.open(host, modules)
    yield _Runner(session)
    path = _option(config, "--conformance-report", None)
    if path:
        session.write_report(path)
