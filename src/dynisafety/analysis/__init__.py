"""Offline paper-analysis helpers.

Read-only consumers of ``episodes.jsonl`` and ``traces/``. This package must
not import ``dynisafety.backends``, must not step a simulator, and must not
write back into a results tree. Existing ``dynisafety report generate`` is
unchanged — run these tools separately.
"""

from dynisafety.analysis.diversity import (
    DiversityPlan,
    extract_plan,
    table_diversity_acceptance,
    table_diversity_by_mode,
    table_diversity_coverage,
    write_diversity,
)
from dynisafety.analysis.exemplars import (
    ExemplarRow,
    plot_exemplar_trajectory,
    select_exemplars,
)
from dynisafety.analysis.kinematics import (
    KinematicsMetrics,
    heading_reversals,
    ldlj_steps,
    measure_trace,
    table_kinematics,
    write_kinematics,
)
from dynisafety.analysis.load import BoundEpisode, fingerprint_results, load_bound_episodes
from dynisafety.analysis.rank import (
    spearman_rho,
    table_rank_backend_pairs,
    table_rank_coverage,
    write_rank,
)
from dynisafety.analysis.termination import (
    CFF_BUCKETS,
    TERMINATION_KINDS,
    cff_bucket,
    classify_termination,
    plot_cff_termination,
    table_termination,
)

__all__ = [
    "BoundEpisode",
    "CFF_BUCKETS",
    "DiversityPlan",
    "ExemplarRow",
    "KinematicsMetrics",
    "TERMINATION_KINDS",
    "cff_bucket",
    "classify_termination",
    "extract_plan",
    "fingerprint_results",
    "heading_reversals",
    "ldlj_steps",
    "load_bound_episodes",
    "measure_trace",
    "plot_cff_termination",
    "plot_exemplar_trajectory",
    "select_exemplars",
    "spearman_rho",
    "table_diversity_acceptance",
    "table_diversity_by_mode",
    "table_diversity_coverage",
    "table_kinematics",
    "table_rank_backend_pairs",
    "table_rank_coverage",
    "table_termination",
    "write_diversity",
    "write_kinematics",
    "write_rank",
]
