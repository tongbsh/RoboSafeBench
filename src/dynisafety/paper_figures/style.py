"""ICLR conference matplotlib style. Display only — no eval defaults.

ICLR is a single-column conference template (not a journal, not NeurIPS
two-column). Official ``iclr*_conference.sty`` sets ``\\textwidth`` to 5.5 in
and body type to 10 pt Times. Include figures with
``\\includegraphics[width=\\textwidth]{...}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ICLR single-column text width (inches). Both names are aliases so existing
# plot calls stay valid; there is no second column.
TEXTWIDTH = 5.50
SINGLE = TEXTWIDTH
DOUBLE = TEXTWIDTH

INK = "#222222"
MUTED = "#5A5A5A"
RULE = "#222222"
GRID = "#E8E8E8"
FILL = "#F4F5F7"
ACCENT_FILL = "#EEF2F5"

# Okabe–Ito (colorblind-safe). ICLR review is on-screen PDF, not print.
OKABE = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#7A7A7A",
}

LEVEL_COLORS = {
    0: OKABE["blue"],
    1: OKABE["orange"],
    2: OKABE["sky"],
    3: OKABE["vermillion"],
}
QUAD_COLORS = {
    "ss": OKABE["green"],
    "us": OKABE["orange"],
    "cff": "#8A8A8A",
    "cif": OKABE["vermillion"],
}
BEHAVIOR_COLORS = {
    "collide_unreacted": "#882255",
    "collide_reacted": OKABE["vermillion"],
    "evade_success": OKABE["green"],
    "evade_fail": "#66A182",
    "stop_recover": OKABE["blue"],
    "stop_no_recover": OKABE["sky"],
    "partial_response": OKABE["orange"],
    "no_response": "#4A4A4A",
    "not_triggered": "#D4D4D4",
    "baseline_present": "#B0B0B0",
    "baseline_unusable": "#B0B0B0",
}
BEHAVIOR_LABELS = {
    "collide_unreacted": "Collide (no reaction)",
    "collide_reacted": "Collide (late)",
    "evade_success": "Evade",
    "evade_fail": "Evade→fail",
    "stop_recover": "Stop→recover",
    "stop_no_recover": "Stop→no recover",
    "partial_response": "Partial",
    "no_response": "No response",
    "not_triggered": "Not triggered",
    "baseline_unusable": "Baseline unusable",
}
BEHAVIOR_ORDER = (
    "collide_unreacted",
    "collide_reacted",
    "evade_success",
    "evade_fail",
    "stop_recover",
    "stop_no_recover",
    "partial_response",
    "no_response",
    "not_triggered",
    "baseline_unusable",
)
QUAD_ORDER = ("ss", "us", "cff", "cif")
HEATMAP_CMAP = "cividis"
_MODEL_CYCLE = (
    OKABE["blue"],
    OKABE["orange"],
    OKABE["green"],
    OKABE["vermillion"],
    OKABE["purple"],
    OKABE["sky"],
    OKABE["black"],
    OKABE["yellow"],
)


def apply_iclr_style() -> None:
    """Times, ICLR body-adjacent type, embed TrueType in PDF."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ("Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"),
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "legend.handlelength": 1.0,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.15,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def model_color(model: str, known: Sequence[str]) -> str:
    ordered = list(known)
    if model in ordered:
        return _MODEL_CYCLE[ordered.index(model) % len(_MODEL_CYCLE)]
    return _MODEL_CYCLE[abs(hash(model)) % len(_MODEL_CYCLE)]


def panel_label(ax: Axes, letter: str, *, x: float = 0.0, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="regular",
        va="bottom",
        ha="left",
        color=INK,
    )


def annotate_partial(fig: Figure, text: str) -> None:
    if not text:
        return
    fig.text(0.0, 0.0, text, fontsize=6, color=MUTED, ha="left", va="bottom")


def maybe_stamp(fig: Figure, bundle) -> None:
    """Draft-only coverage line. Off in the ICLR PDF."""
    if not getattr(bundle.settings, "coverage_stamp", False):
        return
    annotate_partial(fig, bundle.stamp())


def light_ygrid(ax: Axes) -> None:
    ax.yaxis.grid(True, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


def save_figure(
    fig: Figure,
    out_dir: Path,
    stem: str,
    *,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
        written.append(path)
    plt.close(fig)
    return written


def draw_booktabs(
    ax: Axes,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    col_align: Sequence[str] | None = None,
    col_widths: Sequence[float] | None = None,
    highlight_rows: Sequence[int] = (),
    title: str = "",
    letter: str | None = None,
    footnote: str = "",
) -> None:
    """Booktabs-style table: top/mid/bottom rules, no cell grid."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if letter:
        panel_label(ax, letter, x=0.0, y=1.03)
    if title:
        ax.text(0.07 if letter else 0.0, 0.99, title, fontsize=8, va="top", ha="left", color=INK)

    n_col = len(headers)
    n_row = len(rows)
    top, bottom = 0.86, 0.12 if footnote else 0.06
    n_lines = n_row + 1
    row_h = (top - bottom) / max(n_lines, 1)
    if col_widths is None:
        xs = [i / n_col for i in range(n_col + 1)]
    else:
        if len(col_widths) != n_col:
            raise ValueError("col_widths must match the number of columns")
        xs = [0.0]
        for width in col_widths:
            xs.append(xs[-1] + float(width))
    align = list(col_align) if col_align is not None else ["center"] * n_col

    def _xpos(c: int) -> float:
        if align[c] == "left":
            return xs[c] + 0.012
        if align[c] == "right":
            return xs[c + 1] - 0.012
        return 0.5 * (xs[c] + xs[c + 1])

    def _ha(c: int) -> str:
        return align[c] if align[c] in {"left", "right", "center"} else "center"

    ax.plot([0, 1], [top, top], color=RULE, lw=0.9, solid_capstyle="butt", clip_on=False)
    header_y = top - row_h
    ax.plot([0, 1], [header_y, header_y], color=RULE, lw=0.45, solid_capstyle="butt", clip_on=False)
    ax.plot([0, 1], [bottom, bottom], color=RULE, lw=0.9, solid_capstyle="butt", clip_on=False)

    for c, head in enumerate(headers):
        ax.text(_xpos(c), top - 0.5 * row_h, head, ha=_ha(c), va="center", fontsize=7, color=INK)

    for r, row in enumerate(rows):
        band_b = header_y - (r + 1) * row_h
        if r in set(highlight_rows):
            ax.add_patch(
                Rectangle(
                    (0.0, band_b),
                    1.0,
                    row_h,
                    facecolor=ACCENT_FILL,
                    edgecolor="none",
                    zorder=0,
                )
            )
        y = header_y - (r + 0.5) * row_h
        weight = "medium" if r in set(highlight_rows) else "regular"
        for c, val in enumerate(row):
            ax.text(
                _xpos(c),
                y,
                str(val),
                ha=_ha(c),
                va="center",
                fontsize=7,
                color=INK,
                fontweight=weight,
                zorder=1,
            )

    if footnote:
        ax.text(0.0, 0.02, footnote, fontsize=6, va="bottom", ha="left", color=MUTED)


def card(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str = FILL,
    edgecolor: str = "#C8C8C8",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.03",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.5,
            mutation_aspect=1.0,
        )
    )


def arrow(ax: Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=7,
            color="#6A6A6A",
            lw=0.7,
            shrinkA=0,
            shrinkB=1,
        )
    )
