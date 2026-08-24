"""Feasibility evaluation for Base / Oracle-SWR / Oracle-SWR+AEGIS (Prompt 14 / A2).

LOCKED: never map oracle controller failure → invalid-geometry.
Joint metrics reuse Prompt 20 (SR/CR/SSR/SS/US/CFF/CIF).
"""

from dynisafety.feasibility.aggregate import (
    CellMetrics,
    aggregate_episodes_to_cells,
    build_base_vs_oracle_table,
    build_feasibility_matrix,
    load_episode_dicts_from_jsonl,
)
from dynisafety.feasibility.classify import (
    FeasibilityClass,
    FeasibilityThresholds,
    classify_cell,
    classify_task_level,
)
from dynisafety.feasibility.estimate import (
    VALID_CONDITIONS,
    estimate_matrix_resources,
    expand_matrix_jobs,
    load_feasibility_matrix,
)
from dynisafety.feasibility.report import (
    narrative_notes,
    render_feasibility_markdown,
    write_feasibility_artifacts,
)

__all__ = [
    "VALID_CONDITIONS",
    "CellMetrics",
    "FeasibilityClass",
    "FeasibilityThresholds",
    "aggregate_episodes_to_cells",
    "build_base_vs_oracle_table",
    "build_feasibility_matrix",
    "classify_cell",
    "classify_task_level",
    "estimate_matrix_resources",
    "expand_matrix_jobs",
    "load_episode_dicts_from_jsonl",
    "load_feasibility_matrix",
    "narrative_notes",
    "render_feasibility_markdown",
    "write_feasibility_artifacts",
]
