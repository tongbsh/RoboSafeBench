"""The analysis package must stay off the backend / report generate path."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "src" / "dynisafety" / "analysis"
REPORT = REPO / "src" / "dynisafety" / "report"
CLI_MAIN = REPO / "src" / "dynisafety" / "cli" / "__main__.py"


def _import_from_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def test_analysis_package_does_not_import_backends() -> None:
    banned = (
        "dynisafety.backends",
        "dynisafety.report.tables",
        "dynisafety.report.plots",
        "dynisafety.cli.report",
    )
    offenders: list[str] = []
    for path in ANALYSIS.rglob("*.py"):
        for module in _import_from_modules(path):
            if any(module == item or module.startswith(item + ".") for item in banned):
                offenders.append(f"{path.relative_to(REPO)} imports {module}")
    assert offenders == []


def test_report_package_does_not_import_analysis() -> None:
    offenders: list[str] = []
    for path in REPORT.rglob("*.py"):
        for module in _import_from_modules(path):
            if module == "dynisafety.analysis" or module.startswith("dynisafety.analysis."):
                offenders.append(f"{path.relative_to(REPO)} imports {module}")
    assert offenders == []


def test_cli_main_only_adds_analysis_dispatch() -> None:
    text = CLI_MAIN.read_text(encoding="utf-8")
    assert "register_analysis_parser" in text
    assert "dispatch_analysis" in text
    assert "register_report_parser" in text
