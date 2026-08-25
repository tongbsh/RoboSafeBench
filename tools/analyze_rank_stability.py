#!/usr/bin/env python3
"""Thin entry for rank stability. Same as: python tools/analyze_paper.py rank ..."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dynisafety.analysis.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["rank", *sys.argv[1:]]))
