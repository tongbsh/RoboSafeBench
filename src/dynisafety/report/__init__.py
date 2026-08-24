"""Report rendering: tables and plots (display names live here, not in core/)."""

from dynisafety.report.plots import (
    ParetoSeriesSpec,
    render_all_plots,
    render_joint_plots,
)
from dynisafety.report.tables import (
    UNAVAILABLE,
    render_all_tables,
    render_joint_tables,
    table_base_vs_oracle,
    table_joint_outcomes,
    table_sr_ssr_cr,
)

__all__ = [
    "UNAVAILABLE",
    "ParetoSeriesSpec",
    "render_all_plots",
    "render_all_tables",
    "render_joint_plots",
    "render_joint_tables",
    "table_base_vs_oracle",
    "table_joint_outcomes",
    "table_sr_ssr_cr",
]
