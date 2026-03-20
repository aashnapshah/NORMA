"""
Generate NORMA architecture diagram (Figure 3A) as a PDF schematic.

Produces a vertical flowchart showing:
  Inputs -> Embeddings -> Token Sequence -> Decoder -> Output -> Quantile Head -> Quantiles
"""

import os
import sys

# plots.py is in the same directory (scripts/)
from plots import plt, PALETTE

import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_CTX   = PALETTE['green']      # '#2E7D32'  context
C_HIST  = PALETTE['slate']      # '#5C6BC0'  history
C_QUERY = PALETTE['coral']      # '#E85D4A'  query / output
C_TEAL  = PALETTE['teal']       # '#0097A7'  decoder blocks
C_GREY  = PALETTE['grey']       # '#78909C'  section labels

# Lighter tints for embedding boxes (blend toward white)
def lighten(hex_color, amount=0.35):
    """Return a lighter version of a hex color."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f'#{r:02x}{g:02x}{b:02x}'

C_CTX_L   = lighten(C_CTX, 0.40)
C_HIST_L  = lighten(C_HIST, 0.40)
C_QUERY_L = lighten(C_QUERY, 0.40)
C_TEAL_L  = lighten(C_TEAL, 0.45)
C_QUERY_LL = lighten(C_QUERY, 0.55)

# ---------------------------------------------------------------------------
# Layout constants (all in data coords, figure is ~3.5 x 6 inches)
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 3.5, 6.0
X_CENTER = FIG_W / 2

BOX_W  = 0.7   # width of a single input/embedding box
BOX_H  = 0.32  # height
GAP_X  = 0.12  # horizontal gap between the three boxes
TRIO_W = 3 * BOX_W + 2 * GAP_X  # total width of a trio

# Vertical positions (top of each row, measured from bottom)
Y_QUANTILES   = 0.30
Y_QHEAD       = 0.95
Y_OUTPUT      = 1.60
Y_DECODER_TOP = 2.85  # top of decoder dashed box
Y_TOKEN       = 3.70
Y_EMBED       = 4.45
Y_INPUT       = 5.20

ARROW_STYLE = dict(arrowstyle='->', color='#455A64', lw=1.0,
                    mutation_scale=10, shrinkA=2, shrinkB=2)
LABEL_KW = dict(fontsize=6.5, fontweight='bold', color=C_GREY, ha='center',
                va='bottom', fontstyle='normal')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rounded_box(ax, x_center, y_center, w, h, color, label,
                fontcolor='white', fontsize=6.5, fontweight='semibold',
                edgecolor=None, linestyle='-', linewidth=0.6, alpha=1.0):
    """Draw a rounded-rectangle box with centred text."""
    ec = edgecolor if edgecolor else color
    box = FancyBboxPatch(
        (x_center - w / 2, y_center - h / 2), w, h,
        boxstyle='round,pad=0.04',
        facecolor=color, edgecolor=ec,
        linewidth=linewidth, linestyle=linestyle, alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x_center, y_center, label, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=fontcolor, zorder=4)
    return box


def section_label(ax, y, text):
    """Draw a small-caps-style section label."""
    ax.text(X_CENTER, y, text.upper(), **LABEL_KW)


def arrow_down(ax, x, y_top, y_bot):
    """Draw a downward arrow from y_top to y_bot."""
    ax.annotate('', xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.0,
                                shrinkA=1, shrinkB=1))


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')

# ── 1. INPUTS ──────────────────────────────────────────────────────────────
section_label(ax, Y_INPUT + BOX_H / 2 + 0.12, 'Inputs')

x_left  = X_CENTER - TRIO_W / 2 + BOX_W / 2
x_mid   = X_CENTER
x_right = X_CENTER + TRIO_W / 2 - BOX_W / 2

rounded_box(ax, x_left,  Y_INPUT, BOX_W, BOX_H, C_CTX,   'Context')
rounded_box(ax, x_mid,   Y_INPUT, BOX_W, BOX_H, C_HIST,  'History')
rounded_box(ax, x_right, Y_INPUT, BOX_W, BOX_H, C_QUERY, 'Query')

# Arrow: inputs -> embeddings
for x in [x_left, x_mid, x_right]:
    arrow_down(ax, x, Y_INPUT - BOX_H / 2, Y_EMBED + BOX_H / 2)

# ── 2. EMBEDDINGS ─────────────────────────────────────────────────────────
section_label(ax, Y_EMBED + BOX_H / 2 + 0.12, 'Embeddings')

rounded_box(ax, x_left,  Y_EMBED, BOX_W, BOX_H, C_CTX_L,   'Context',
            fontcolor='#1B5E20')
rounded_box(ax, x_mid,   Y_EMBED, BOX_W, BOX_H, C_HIST_L,  'History',
            fontcolor='#303F9F')
rounded_box(ax, x_right, Y_EMBED, BOX_W, BOX_H, C_QUERY_L, 'Query',
            fontcolor='#C62828')

# Arrow: embeddings -> token sequence (converging to center)
for x in [x_left, x_mid, x_right]:
    arrow_down(ax, x, Y_EMBED - BOX_H / 2, Y_TOKEN + BOX_H / 2)

# ── 3. TOKEN SEQUENCE ─────────────────────────────────────────────────────
section_label(ax, Y_TOKEN + BOX_H / 2 + 0.12, 'Token Sequence')

tok_labels = ['CTX', 'h\u2081', 'h\u2082', 'h\u2083', 'h\u2084', 'Q']
tok_colors = [C_CTX, C_HIST, C_HIST, C_HIST, C_HIST, C_QUERY]
tok_fc     = ['white'] * 6
n_tok = len(tok_labels)
tok_w = 0.34
tok_gap = 0.06
total_tok_w = n_tok * tok_w + (n_tok - 1) * tok_gap
tok_x_start = X_CENTER - total_tok_w / 2 + tok_w / 2

for i, (lab, col, fc) in enumerate(zip(tok_labels, tok_colors, tok_fc)):
    cx = tok_x_start + i * (tok_w + tok_gap)
    rounded_box(ax, cx, Y_TOKEN, tok_w, BOX_H, col, lab,
                fontcolor=fc, fontsize=5.5)

# Arrow: token sequence -> decoder
arrow_down(ax, X_CENTER, Y_TOKEN - BOX_H / 2, Y_DECODER_TOP + 0.02)

# ── 4. DECODER ─────────────────────────────────────────────────────────────
section_label(ax, Y_DECODER_TOP + 0.12, 'Decoder')

DEC_W = 2.2
DEC_H = 1.20
dec_y_bot = Y_DECODER_TOP - DEC_H

# Dashed border
dec_border = FancyBboxPatch(
    (X_CENTER - DEC_W / 2, dec_y_bot), DEC_W, DEC_H,
    boxstyle='round,pad=0.08',
    facecolor='none', edgecolor='#90A4AE',
    linewidth=0.8, linestyle='--', zorder=2,
)
ax.add_patch(dec_border)

# Two internal boxes
inner_w = 1.9
inner_h = 0.38
y_attn = Y_DECODER_TOP - 0.32
y_ffn  = Y_DECODER_TOP - 0.85

rounded_box(ax, X_CENTER, y_attn, inner_w, inner_h, C_TEAL,
            'Masked Self-Attention', fontsize=6.5)
rounded_box(ax, X_CENTER, y_ffn,  inner_w, inner_h, C_TEAL,
            'Feed-Forward Network', fontsize=6.5)

arrow_down(ax, X_CENTER, y_attn - inner_h / 2, y_ffn + inner_h / 2)

# Arrow: decoder -> output
arrow_down(ax, X_CENTER, dec_y_bot - 0.02, Y_OUTPUT + BOX_H / 2)

# ── 5. OUTPUT ──────────────────────────────────────────────────────────────
section_label(ax, Y_OUTPUT + BOX_H / 2 + 0.12, 'Output')

out_w = 1.6
rounded_box(ax, X_CENTER, Y_OUTPUT, out_w, BOX_H, C_QUERY_L,
            'Query Token Output', fontcolor='#C62828', fontsize=6.5)

# Arrow: output -> quantile head
arrow_down(ax, X_CENTER, Y_OUTPUT - BOX_H / 2, Y_QHEAD + BOX_H / 2)

# ── 6. QUANTILE HEAD ──────────────────────────────────────────────────────
rounded_box(ax, X_CENTER, Y_QHEAD, 1.4, BOX_H, C_QUERY_LL,
            'Quantile Head', fontcolor='#B71C1C', fontsize=6.5)

# Arrow: quantile head -> quantiles
arrow_down(ax, X_CENTER, Y_QHEAD - BOX_H / 2, Y_QUANTILES + BOX_H / 2)

# ── 7. OUTPUT QUANTILES ───────────────────────────────────────────────────
section_label(ax, Y_QUANTILES + BOX_H / 2 + 0.12, 'Output Quantiles')

q_labels = ['q\u2085', 'q\u2082\u2085', 'q\u2085\u2080', 'q\u2087\u2085', 'q\u2089\u2085']
n_q = len(q_labels)
q_w = 0.40
q_gap = 0.08
total_q_w = n_q * q_w + (n_q - 1) * q_gap
q_x_start = X_CENTER - total_q_w / 2 + q_w / 2

for i, ql in enumerate(q_labels):
    cx = q_x_start + i * (q_w + q_gap)
    rounded_box(ax, cx, Y_QUANTILES, q_w, BOX_H, lighten(C_QUERY, 0.65),
                ql, fontcolor='#B71C1C', fontsize=5.5, fontweight='bold')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'manuscript_figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure3a_architecture.pdf')
plt.savefig(out_path, bbox_inches='tight', pad_inches=0.15, dpi=300, format='pdf')
plt.close()
print(f"Saved: {out_path}")
