#!/usr/bin/env python3
"""Read-only paper analysis: timeout, exemplars, diversity, kinematics, rank.

Does not import backends, does not step a simulator, and refuses to write
inside the results tree.

    python tools/analyze_paper.py termination --results results --out artifacts/paper_analysis/termination
    python tools/analyze_paper.py exemplars --results results --out artifacts/paper_analysis/exemplars
    python tools/analyze_paper.py diversity --results results --out artifacts/paper_analysis/diversity
    python tools/analyze_paper.py kinematics --results results --out artifacts/paper_analysis/kinematics
    python tools/analyze_paper.py rank --results results --out artifacts/paper_analysis/rank
    python tools/analyze_paper.py all --results results --out artifacts/paper_analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dynisafety.analysis.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
