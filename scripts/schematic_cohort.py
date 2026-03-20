"""
Generate Figure 2A: Cohort selection schematic diagram.

Produces a clean, Nature-style timeline schematic showing the three phases
of the study design (Baseline Measurements, Index Lab, Monitoring Period)
with test-tube icons representing inpatient/outpatient lab draws and
colored classification bars (Low / Normal / High).

Output: results/manuscript_figures/figure2a_cohort.pdf
"""

import os
import sys
import numpy as np

# plots.py is in the same directory (scripts/)
from plots import plt, PALETTE, save_fig

import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── colours ──────────────────────────────────────────────────────────────
TEAL   = PALETTE['teal']
CORAL  = PALETTE['coral']
GREY   = PALETTE['grey']
GREEN  = PALETTE['green']
GOLD   = PALETTE['gold']

INPATIENT_COLOR  = '#8B1A1A'   # dark red-brown
OUTPATIENT_COLOR = '#F4A7A0'   # light salmon-pink
TIMELINE_COLOR   = '#333333'

# Classification bar colours
LOW_COLOR    = GOLD
NORMAL_COLOR = GREEN
HIGH_COLOR   = CORAL

# ── helpers ──────────────────────────────────────────────────────────────

def draw_test_tube(ax, cx, cy, w=0.3, h=0.9, color='#8B1A1A', alpha=1.0):
    """Draw a simple test tube: rounded rectangle body + flat cap on top."""
    # Body
    body = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h * 0.85,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor='white', linewidth=0.5, alpha=alpha,
        zorder=5,
    )
    ax.add_patch(body)
    # Cap (small rectangle on top)
    cap_h = h * 0.18
    cap_w = w * 1.15
    cap = FancyBboxPatch(
        (cx - cap_w / 2, cy - h / 2 + h * 0.82), cap_w, cap_h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor='white', linewidth=0.5, alpha=alpha,
        zorder=6,
    )
    ax.add_patch(cap)


def draw_tube_cluster(ax, x_center, y_center, n_inpatient=1, n_outpatient=2,
                      tube_w=0.22, tube_h=0.7, spacing=0.28):
    """Draw a small cluster of inpatient + outpatient test tubes."""
    total = n_inpatient + n_outpatient
    x_start = x_center - (total - 1) * spacing / 2
    for i in range(n_inpatient):
        draw_test_tube(ax, x_start + i * spacing, y_center,
                       w=tube_w, h=tube_h, color=INPATIENT_COLOR)
    for i in range(n_outpatient):
        draw_test_tube(ax, x_start + (n_inpatient + i) * spacing, y_center,
                       w=tube_w, h=tube_h, color=OUTPATIENT_COLOR)


def draw_phase_bracket(ax, x0, x1, y, label, color=TIMELINE_COLOR, fontsize=6.5):
    """Draw a horizontal bracket with a centred label above it."""
    mid = (x0 + x1) / 2
    # Horizontal line
    ax.plot([x0, x1], [y, y], color=color, linewidth=1.0, solid_capstyle='butt', zorder=3)
    # Small end ticks
    tick = 0.12
    ax.plot([x0, x0], [y - tick, y + tick], color=color, linewidth=0.8, zorder=3)
    ax.plot([x1, x1], [y - tick, y + tick], color=color, linewidth=0.8, zorder=3)
    # Label
    ax.text(mid, y + 0.22, label, ha='center', va='bottom',
            fontsize=fontsize, fontweight='bold', color=color, zorder=10)


# ── main figure ──────────────────────────────────────────────────────────

def make_figure():
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axis('off')
    ax.set_xlim(-1.5, 27)
    ax.set_ylim(-4.0, 6.0)

    # ── timeline axis ────────────────────────────────────────────────────
    tl_y = -1.8
    ax.annotate('', xy=(26.5, tl_y), xytext=(0, tl_y),
                arrowprops=dict(arrowstyle='->', color=TIMELINE_COLOR,
                                lw=1.2, shrinkA=0, shrinkB=0), zorder=4)

    # Year labels
    year_positions = {2000: 1, 2005: 5, 2010: 9, 2015: 14, 2020: 19, 2025: 24.5}
    for year, xp in year_positions.items():
        ax.text(xp, tl_y - 0.45, str(year), ha='center', va='top',
                fontsize=6, color=TIMELINE_COLOR, zorder=10)
        ax.plot([xp, xp], [tl_y - 0.15, tl_y + 0.15], color=TIMELINE_COLOR,
                linewidth=0.6, zorder=4)

    # ── phase brackets ───────────────────────────────────────────────────
    bracket_y = 4.2
    draw_phase_bracket(ax, 1, 13, bracket_y, 'BASELINE MEASUREMENTS', color=GREY)
    draw_phase_bracket(ax, 19, 24.5, bracket_y, 'MONITORING PERIOD', color=TEAL)

    # Index lab — dashed box
    idx_x, idx_w, idx_h = 14, 3.2, 1.0
    idx_box = FancyBboxPatch(
        (idx_x - idx_w / 2, bracket_y - idx_h / 2), idx_w, idx_h,
        boxstyle="round,pad=0.15",
        facecolor='none', edgecolor=CORAL, linewidth=1.2,
        linestyle='--', zorder=8,
    )
    ax.add_patch(idx_box)
    ax.text(idx_x, bracket_y + 0.22, 'INDEX LAB', ha='center', va='bottom',
            fontsize=6.5, fontweight='bold', color=CORAL, zorder=10)

    # ── classification colour bar ────────────────────────────────────────
    bar_y = 2.5
    bar_h = 0.45

    # Baseline region: show varying Low / Normal / High segments
    segments = [
        (1,   3.5, HIGH_COLOR,   'High'),
        (3.5, 6,   NORMAL_COLOR, 'Normal'),
        (6,   8,   LOW_COLOR,    'Low'),
        (8,  10.5, NORMAL_COLOR, ''),
        (10.5, 13, HIGH_COLOR,   ''),
    ]
    for x0, x1, col, lbl in segments:
        rect = FancyBboxPatch(
            (x0, bar_y - bar_h / 2), x1 - x0, bar_h,
            boxstyle="round,pad=0.03",
            facecolor=col, edgecolor='white', linewidth=0.5, alpha=0.85, zorder=3,
        )
        ax.add_patch(rect)

    # Monitoring region bar (after index)
    mon_segments = [
        (15.6, 18,  NORMAL_COLOR),
        (18,   20.5, LOW_COLOR),
        (20.5, 22.5, NORMAL_COLOR),
        (22.5, 24.5, HIGH_COLOR),
    ]
    for x0, x1, col in mon_segments:
        rect = FancyBboxPatch(
            (x0, bar_y - bar_h / 2), x1 - x0, bar_h,
            boxstyle="round,pad=0.03",
            facecolor=col, edgecolor='white', linewidth=0.5, alpha=0.85, zorder=3,
        )
        ax.add_patch(rect)

    # Colour-bar legend (Low / Normal / High)
    legend_y = bar_y + 0.6
    legend_items = [('Low', LOW_COLOR), ('Normal', NORMAL_COLOR), ('High', HIGH_COLOR)]
    lx_start = 1
    for i, (lbl, col) in enumerate(legend_items):
        lx = lx_start + i * 2.8
        sq = FancyBboxPatch(
            (lx, legend_y - 0.15), 0.6, 0.3,
            boxstyle="round,pad=0.02",
            facecolor=col, edgecolor='white', linewidth=0.4, alpha=0.85, zorder=3,
        )
        ax.add_patch(sq)
        ax.text(lx + 0.8, legend_y, lbl, ha='left', va='center',
                fontsize=5.5, color='#333333', zorder=10)

    # ── test tube clusters ───────────────────────────────────────────────
    tube_y = 0.3
    # Baseline time points
    draw_tube_cluster(ax, 2, tube_y, n_inpatient=1, n_outpatient=2)
    draw_tube_cluster(ax, 5.5, tube_y, n_inpatient=1, n_outpatient=2)
    draw_tube_cluster(ax, 9.5, tube_y, n_inpatient=0, n_outpatient=3)
    draw_tube_cluster(ax, 12, tube_y, n_inpatient=1, n_outpatient=1)

    # Index lab — single prominent tube
    draw_test_tube(ax, 14, tube_y, w=0.35, h=0.9, color=CORAL)

    # Monitoring period tubes
    draw_tube_cluster(ax, 17, tube_y, n_inpatient=0, n_outpatient=2, spacing=0.26)
    draw_tube_cluster(ax, 20, tube_y, n_inpatient=1, n_outpatient=1, spacing=0.26)
    draw_tube_cluster(ax, 23, tube_y, n_inpatient=0, n_outpatient=2, spacing=0.26)

    # ── tube legend (Inpatient / Outpatient) ─────────────────────────────
    leg_y = -0.8
    leg_x = 1
    draw_test_tube(ax, leg_x, leg_y, w=0.20, h=0.55, color=INPATIENT_COLOR)
    ax.text(leg_x + 0.3, leg_y, 'Inpatient', ha='left', va='center',
            fontsize=5.5, color='#333333', zorder=10)
    draw_test_tube(ax, leg_x + 2.5, leg_y, w=0.20, h=0.55, color=OUTPATIENT_COLOR)
    ax.text(leg_x + 2.8, leg_y, 'Outpatient', ha='left', va='center',
            fontsize=5.5, color='#333333', zorder=10)

    # ── inclusion criteria annotation ────────────────────────────────────
    ax.text(14, -2.9, r'Inclusion:  $\geq$ 5 Outpatient Labs, 90 Days Apart',
            ha='center', va='top', fontsize=6, fontstyle='italic',
            color=GREY, zorder=10)

    # ── dotted vertical line at index ────────────────────────────────────
    ax.plot([14, 14], [tl_y + 0.2, bar_y - bar_h / 2 - 0.1],
            color=CORAL, linewidth=0.8, linestyle=':', zorder=2)

    # ── save ─────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'manuscript_figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'figure2a_cohort')
    save_fig(out_path)
    print(f"Saved: {out_path}.pdf")


if __name__ == '__main__':
    make_figure()
