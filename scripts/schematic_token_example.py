"""
Generate an example token sequence visualization (Figure 3 bottom panel).

Shows how a patient's lab history is represented as a token sequence:
  Row 1: Time gaps (days since previous)
  Row 2: Lab values
  Row 3: Binary indicator (inpatient/outpatient)
  With X_t as the prediction target.
"""

import os
import sys

# plots.py is in the same directory (scripts/)
from plots import plt, PALETTE, save_fig

import numpy as np

# Example data (matching the PDF)
time_gaps = [0, 90, 240, 380, 387, 2, 1]
lab_values = [8.9, 9.2, 10.5, 12.6, 13.1, 13.4, '']
indicators = [1, 1, 0, 1, 1, 1, 1]
n = len(time_gaps)

C_HIST = PALETTE['slate']
C_QUERY = PALETTE['coral']
C_TEAL = PALETTE['teal']

fig, ax = plt.subplots(figsize=(5, 1.8))
ax.axis('off')
ax.set_xlim(-0.5, n + 0.5)
ax.set_ylim(-0.5, 3.8)

# Column positions
xs = np.arange(n)

# Row labels on the left
row_labels = ['dt (days)', 'Value', 'Outpatient']
row_ys = [2.6, 1.5, 0.4]
for label, y in zip(row_labels, row_ys):
    ax.text(-0.6, y, label, ha='right', va='center', fontsize=6.5,
            fontstyle='italic', color='#555555')

# Draw values
for i in range(n):
    is_target = (i == n - 1)
    color = C_QUERY if is_target else C_HIST

    # Time gap
    ax.text(xs[i], row_ys[0], str(time_gaps[i]), ha='center', va='center',
            fontsize=7, fontweight='semibold', color=color)

    # Lab value (X_t for target)
    if is_target:
        ax.text(xs[i], row_ys[1], r'X$_t$', ha='center', va='center',
                fontsize=8, fontweight='bold', color=C_QUERY)
    else:
        ax.text(xs[i], row_ys[1], str(lab_values[i]), ha='center', va='center',
                fontsize=7, fontweight='semibold', color=color)

    # Binary indicator
    ax.text(xs[i], row_ys[2], str(indicators[i]), ha='center', va='center',
            fontsize=7, fontweight='semibold', color=color)

# Separator line above target column
ax.axvline(x=n - 1.5, ymin=0.05, ymax=0.95, color='#CCCCCC', lw=0.8,
           ls='--', zorder=0)

# Labels: "History" and "Query"
mid_hist = (n - 2) / 2
ax.text(mid_hist, 3.5, 'History', ha='center', va='center',
        fontsize=7, fontweight='bold', color=C_HIST)
ax.text(n - 1, 3.5, 'Query', ha='center', va='center',
        fontsize=7, fontweight='bold', color=C_QUERY)

# Arrow pointing right across history
ax.annotate('', xy=(n - 1.7, 3.5), xytext=(-0.3, 3.5),
            arrowprops=dict(arrowstyle='-', color=C_HIST, lw=0.6))

out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'prediction', 'figures')
os.makedirs(out_dir, exist_ok=True)
save_fig(os.path.join(out_dir, 'token_example'))
print(f'Saved to {out_dir}/token_example.pdf')
