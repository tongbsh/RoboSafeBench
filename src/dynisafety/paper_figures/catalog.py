"""Paper figure inventory. IDs are stable; panel copy is for captions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigureSpec:
    fig_id: str
    stem: str
    section: str
    title: str
    needs_results: bool
    needs_assets: bool
    caption: str


# Numbering follows the ICLR main-text plan (LIBERO-style teaser + RQ figures).
# Appendix / scaffold IDs stay stable if the main numbering is edited later.
FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        fig_id="fig01",
        stem="fig01_scale_overview",
        section="main",
        title="Benchmark scale and peer comparison",
        needs_results=False,
        needs_assets=False,
        caption=(
            "RoboSafeBench planned matrix (from paper_main.yaml) versus host "
            "benchmarks. Three tabletop backends; dynamic levels are L0–L3."
        ),
    ),
    FigureSpec(
        fig_id="fig02",
        stem="fig02_framework",
        section="main",
        title="Evaluation framework",
        needs_results=False,
        needs_assets=False,
        caption=(
            "Four-stage evaluation: host simulator, L0–L3 disturbance, "
            "VLA / Oracle / shield, and joint outcomes with diagnostics."
        ),
    ),
    FigureSpec(
        fig_id="fig03",
        stem="fig03_main_metrics",
        section="main",
        title="Main results (SR / SSR / CR)",
        needs_results=True,
        needs_assets=False,
        caption="Grouped bars by model and level. Missing cells are omitted, never filled with 0.",
    ),
    FigureSpec(
        fig_id="fig04",
        stem="fig04_joint_stack",
        section="main",
        title="Joint outcomes SS / US / CFF / CIF",
        needs_results=True,
        needs_assets=False,
        caption="CFF is contact-free failure, not Safe Stop.",
    ),
    FigureSpec(
        fig_id="fig05",
        stem="fig05_pareto",
        section="main",
        title="SR–SSR trade-off",
        needs_results=True,
        needs_assets=False,
        caption="Each polyline is one model across L0→L3. Upper-right is better.",
    ),
    FigureSpec(
        fig_id="fig06",
        stem="fig06_behavior",
        section="main",
        title="Ten-class behavior distribution",
        needs_results=True,
        needs_assets=False,
        caption=(
            "Shares among episodes with a diagnostics label (10 single-trace classes). "
            "L0 is almost all not_triggered. Not comparable with the 8-class paired figA5."
        ),
    ),
    FigureSpec(
        fig_id="fig07",
        stem="fig07_oracle",
        section="main",
        title="Base VLA versus Oracle / shield",
        needs_results=True,
        needs_assets=False,
        caption="Paired on backend × suite × task × level × seed. Empty if no oracle arm.",
    ),
    FigureSpec(
        fig_id="fig08",
        stem="fig08_heatmap_ssr",
        section="main",
        title="SSR heatmap (task × level)",
        needs_results=True,
        needs_assets=False,
        caption="Within-backend only. Blank cells are missing, not zero.",
    ),
    FigureSpec(
        fig_id="fig09",
        stem="fig09_process",
        section="main",
        title="Paired reaction / stop / evade / recovery",
        needs_results=True,
        needs_assets=False,
        caption=(
            "Paired four axes (eval.md §2.5): every gate is measured against the matched "
            "L0 twin, so t_onset=0 episodes are inside these denominators. Reaction / stop "
            "/ evade denominator is n scored pairs per backend × model × level; recovery "
            "uses n_recovery_applicable (hazard ended, budget left, robot stopped). Each "
            "bar carries its own denominator. Grey ticks are the same detector on the "
            "pre-onset segment; rows without one are n/a there, never False. Do not read "
            "against figA4."
        ),
    ),
    FigureSpec(
        fig_id="fig10",
        stem="fig10_level_curves",
        section="main",
        title="Difficulty curves L0–L3",
        needs_results=True,
        needs_assets=False,
        caption="Episode-weighted rates. Connecting lines skip missing levels.",
    ),
    FigureSpec(
        fig_id="figA1",
        stem="figA1_force_box",
        section="appendix",
        title="Peak contact index (within-backend)",
        needs_results=True,
        needs_assets=False,
        caption="Do not compare Newtons across force_source / simulators.",
    ),
    FigureSpec(
        fig_id="figA2",
        stem="figA2_failure_decomp",
        section="appendix",
        title="CFF is not Safe Stop",
        needs_results=True,
        needs_assets=False,
        caption="Quadrants split by behavior class. CFF panel is hatched.",
    ),
    FigureSpec(
        fig_id="figA3",
        stem="figA3_coverage",
        section="appendix",
        title="Planned versus completed cells",
        needs_results=True,
        needs_assets=False,
        caption="paper_main cells only. Safety-layer arms are extra and not in the denominator.",
    ),
    FigureSpec(
        fig_id="figA4",
        stem="figA4_process_single_trace",
        section="appendix",
        title="Single-trace four axes (Limitations control)",
        needs_results=True,
        needs_assets=False,
        caption=(
            "Same four axes from EpisodeRecord.diagnostics (eval.md §2.1–§2.4), kept as the "
            "§9 limitation-1 control: gates are relative to each episode's own pre-onset "
            "window, the kinematic reaction detector saturates (L0 control FPR 0.92) and "
            "reaction_rate is 0 by construction at chunk=30. baseline_unusable episodes are "
            "outside every denominator here. Different reference frame from fig09 — the two "
            "must not be read side by side."
        ),
    ),
    FigureSpec(
        fig_id="figA5",
        stem="figA5_behavior_paired",
        section="appendix",
        title="Paired eight-class behavior distribution",
        needs_results=True,
        needs_assets=False,
        caption=(
            "Paired labels (eval.md §2.5) are 8 classes, not 10: not_triggered is skipped "
            "before scoring and baseline_unusable cannot arise when the reference is the "
            "twin. These shares must not be compared with the 10-class shares in fig06. "
            "Columns split own_baseline — whether the single-trace path could also label the "
            "row; the unusable column is the population it left with four blank axes "
            "(coverage: LIBERO 3400, RoboTwin 245; oracle arms are not ranked). "
            "Bar labels are per-level episode counts."
        ),
    ),
    FigureSpec(
        fig_id="figS1",
        stem="figS1_teaser_montage",
        section="scaffold",
        title="Teaser montage (assets optional)",
        needs_results=False,
        needs_assets=True,
        caption="Drop screenshots at assets/teaser/{backend}_l{level}.png.",
    ),
    FigureSpec(
        fig_id="figS2",
        stem="figS2_trajectory",
        section="scaffold",
        title="L1/L2/L3 four-frame trajectory",
        needs_results=False,
        needs_assets=True,
        caption="Drop frames at assets/trajectory/{backend}_l{level}_{phase}.png.",
    ),
)


def spec_by_id(fig_id: str) -> FigureSpec:
    for spec in FIGURES:
        if spec.fig_id == fig_id or spec.stem == fig_id:
            return spec
    known = [s.fig_id for s in FIGURES]
    raise KeyError(f"unknown figure {fig_id!r}; expected one of {known}")
