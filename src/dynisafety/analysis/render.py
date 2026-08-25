"""Minimal table writers for the analysis package.

Duplicated on purpose: ``dynisafety.report`` stays the paper-main pipeline
and is not imported here, so adding analysis tables cannot change report
generate output.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _cell(value: object, *, missing: str, format_numbers: bool) -> object:
    if _is_missing(value):
        return missing
    if format_numbers and isinstance(value, float):
        return f"{value:.4f}"
    return value


def _apply_display(df: pd.DataFrame, *, missing: str, format_numbers: bool) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(
            lambda v: _cell(v, missing=missing, format_numbers=format_numbers)
        )
    return out


def _to_markdown(df: pd.DataFrame, footnotes: Sequence[str], *, missing: str) -> str:
    shown = _apply_display(df, missing=missing, format_numbers=True)
    try:
        body = shown.to_markdown(index=False)
    except ImportError:
        headers = list(shown.columns)
        lines = [
            "| " + " | ".join(str(h) for h in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for _, row in shown.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        body = "\n".join(lines)
    if footnotes:
        body += "\n\n" + "\n".join(f"- {note}" for note in footnotes)
    return body + "\n"


def _to_latex(df: pd.DataFrame, footnotes: Sequence[str], *, missing: str) -> str:
    shown = _apply_display(df, missing=missing, format_numbers=True)
    for col in shown.columns:
        if shown[col].dtype == object:
            shown[col] = shown[col].map(
                lambda x: str(x).replace("_", r"\_").replace("%", r"\%")
            )
    try:
        body = shown.to_latex(index=False, escape=False)
    except Exception:
        body = shown.to_string(index=False)
    if footnotes:
        body += "\n% Footnotes:\n"
        for i, note in enumerate(footnotes, start=1):
            body += f"% [{i}] {note.replace('_', r'_')}\n"
    return body if body.endswith("\n") else body + "\n"


def render_table(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    *,
    footnotes: Sequence[str] = (),
    missing: str = "n/a",
) -> None:
    """Write ``stem.{csv,md,tex}``. CSV keeps None; md/tex show ``n/a``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(
        _to_markdown(df, footnotes, missing=missing), encoding="utf-8"
    )
    (out_dir / f"{stem}.tex").write_text(
        _to_latex(df, footnotes, missing=missing), encoding="utf-8"
    )
