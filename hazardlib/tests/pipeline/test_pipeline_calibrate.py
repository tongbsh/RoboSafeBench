"""T_stop protocol, censoring and the protected Sigma entry point (T06 item 4; acceptance 12)."""
import ast
import dataclasses
from pathlib import Path

import pytest
import yaml

import pipeline_world as W
from hazardlib.calibrate import (CensoredTStopError, TStopProtocolSettings, aggregate_t_stop, margin_sigma,
                                 read_calibration_summary, run_t_stop_protocol, sigma_from_summary,
                                 summary_to_dict, write_calibration)
from hazardlib.config import ConfigError

SRC = Path(__file__).resolve().parents[2] / "src" / "hazardlib"
PROTECTED = {("calibrate/protocol.py", "sigma_from_summary")}


def protocol(host, settings=None, task="low"):
    return run_t_stop_protocol(W.session(host), W.PathFollower(host), W.task_context(task),
                               settings or TStopProtocolSettings.load(W.FIXTURE_CONFIG_DIR), 0, 0)


def test_t_stop_of_a_braking_host(tmp_path):
    result = protocol(W.make_host(decel_mps2=2.5))
    assert result.nominal_speed_mps == pytest.approx(0.5)
    assert result.trigger_speed_mps == pytest.approx(0.8 * 0.5)
    assert result.samples_s == pytest.approx([0.2, 0.2, 0.2], abs=1e-9)      # 0.5 m/s / 2.5 m/s^2, 20 ms samples
    assert result.summary.n_censored == 0 and not result.is_lower_bound
    path = tmp_path / "calibration.yaml"
    write_calibration(path, result)
    data = yaml.safe_load(path.read_text())
    assert data["t_stop_is_lower_bound"] is False and data["notes"] == ""
    assert data["nominal_speed_source"] == "TaskContext.provenance.eef_speed_median_mps"
    summary, lower = read_calibration_summary(path)
    assert summary_to_dict(summary) == summary_to_dict(result.summary) and lower is False   # eq=False (T02)
    assert sigma_from_summary(0.1, summary, False) == (pytest.approx(0.5), False)


def test_instant_stop_resolves_to_one_control_step():
    result = protocol(W.make_host())
    assert result.summary.median_s == pytest.approx(W.CONTROL_DT)


def test_censored_t_stop_is_a_lower_bound_and_sigma_refuses_it(tmp_path):
    result = protocol(W.make_host(decel_mps2=2.5, censor="odd_seeds"))       # seeds 0, 1, 2: seed 1 never stops
    assert result.samples_s[1] is None and result.summary.n_censored == 1
    assert result.is_lower_bound
    path = tmp_path / "calibration.yaml"
    write_calibration(path, result)
    data = yaml.safe_load(path.read_text())
    assert data["t_stop_is_lower_bound"] is True and "lower bound" in data["notes"]
    summary, lower = read_calibration_summary(path)
    assert lower is True
    with pytest.raises(CensoredTStopError, match="lower bound"):
        sigma_from_summary(0.1, summary, False)
    sigma, upper = sigma_from_summary(0.1, summary, True)
    assert upper is True and sigma == pytest.approx(0.1 / summary.median_s)


def test_all_repetitions_censored_has_no_sigma():
    result = protocol(W.make_host(decel_mps2=2.5, censor="always"))
    assert result.summary.median_s is None and result.summary.n_censored == 3
    with pytest.raises(ValueError, match="undefined"):
        sigma_from_summary(0.1, result.summary, True)


def test_hold_that_is_never_triggered_raises():
    settings = dataclasses.replace(TStopProtocolSettings.load(W.FIXTURE_CONFIG_DIR), trigger_speed_fraction=10.0)
    with pytest.raises(RuntimeError, match="never reached"):
        protocol(W.make_host(decel_mps2=2.5), settings)


def test_calibration_flag_must_match_the_summary(tmp_path):
    path = tmp_path / "calibration.yaml"
    write_calibration(path, protocol(W.make_host(decel_mps2=2.5, censor="odd_seeds")))
    W.edit_yaml(path, lambda d: d.update(t_stop_is_lower_bound=False))
    with pytest.raises(ValueError, match="contradicts"):
        read_calibration_summary(path)


def test_sigma_from_summary_contract():
    summary = aggregate_t_stop([0.2, 0.3, 0.4])
    assert sigma_from_summary(0.3, summary, False) == (pytest.approx(1.0), False)
    assert sigma_from_summary(0.3, summary, False)[0] == margin_sigma(0.3, 0.3)
    with pytest.raises(TypeError):
        sigma_from_summary(0.3, 0.3, False)
    with pytest.raises(ValueError):
        sigma_from_summary(0.3, summary, "no")


def test_protocol_settings_refuse_placeholders_of_the_repository_config():
    with pytest.raises(ConfigError, match="trigger_speed_fraction"):
        TStopProtocolSettings.load(W.REPO_CONFIG_DIR)


# ------------------------------------------------------------------------------------------ AST hygiene

class _MarginSigmaUse(ast.NodeVisitor):
    def __init__(self):
        self.scope = ["<module>"]
        self.calls = []
        self.imports = False

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name == "margin_sigma":
            self.calls.append(self.scope[-1])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if any(alias.name == "margin_sigma" for alias in node.names):
            self.imports = True


def margin_sigma_uses(source):
    visitor = _MarginSigmaUse()
    visitor.visit(ast.parse(source))
    return visitor.calls, visitor.imports


def test_only_the_protected_entry_calls_margin_sigma():
    calls, importers = set(), set()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        found, imports = margin_sigma_uses(path.read_text(encoding="utf-8"))
        calls.update((rel, scope) for scope in found)
        if imports:
            importers.add(rel)
    assert calls == PROTECTED
    assert all(rel.startswith("calibrate/") for rel in importers), importers


def test_the_hygiene_scan_catches_direct_and_aliased_calls():
    assert margin_sigma_uses("from hazardlib.calibrate import margin_sigma\ndef f(t):\n    return margin_sigma(t, 1)\n") \
        == (["f"], True)
    assert margin_sigma_uses("import hazardlib.calibrate as c\nx = c.margin_sigma(1, 2)\n") == (["<module>"], False)
