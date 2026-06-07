"""
Redesigned Figure 6 — H6: Commodity-market evidence
Two-panel layout:
  Panel A  — GSADF rejection rate by commodity group (strong-form test)
  Panel B  — Share of episode-days occurring after Q4 2023 (conditional-form test)
SV-ADF W1 rates shown as small italic annotations under Panel A bars.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

group_labels = [
    'AI-Relevant\nEnergy',
    'AI-Relevant\nMetals',
    'Precious Metals\n[Confound]',
    'Agriculture\n[Placebo]',
    'Livestock\n[Placebo]',
    'Broad\nBenchmark',
]
sample_n = [3, 3, 2, 4, 1, 3]

# Panel A: GSADF rejection rate (%)
gsadf_rates = [100, 67, 100, 100, 100, 100]

# Panel B: post-Q4-2023 episode-day fraction
# Energy:    0/501 = 0%    Metals:  266/424 = 62.7%  Precious: 778/874 = 89.0%
# Agri:     91/1315= 6.9%  Livestock: N/A            Benchmark: 277/1374=20.2%
post_q4_pct = [0.0, 62.7, 89.0, 6.9, None, 20.2]

# SV-ADF W1 episode detected: only precious metals (GLD, SLV)
svadf_w1_detected = [False, False, True, False, False, False]

# ─────────────────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────────────────
BAR_COLORS  = ['#C0392B', '#C0392B', '#D4850A', '#2471A3', '#2471A3', '#808B96']
EDGE_COLORS = ['#922B21', '#922B21', '#A0640A', '#1A5276', '#1A5276', '#5D6D7E']

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 7.0), facecolor='white',
    gridspec_kw={'wspace': 0.32}
)

x  = np.arange(len(group_labels))
bw = 0.55

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#C0C0C0')
    ax.spines['bottom'].set_color('#C0C0C0')
    ax.tick_params(axis='both', length=0)
    ax.yaxis.grid(True, linestyle='--', color='#E0E0E0', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

style_ax(ax1)
style_ax(ax2)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL A — GSADF rejection rate
# ─────────────────────────────────────────────────────────────────────────────
bars_a = ax1.bar(x, gsadf_rates, width=bw,
                 color=BAR_COLORS, edgecolor=EDGE_COLORS,
                 linewidth=0.8, zorder=3)

# Value labels
for bar, val in zip(bars_a, gsadf_rates):
    ax1.text(bar.get_x() + bw / 2, val + 1.5,
             f'{val:.0f}%',
             ha='center', va='bottom', fontsize=10.5,
             fontweight='bold', color='#1C2833')

# Thin reference line at placebo level (100%)
ax1.axhline(100, color='#2471A3', linestyle=':', linewidth=1.0, alpha=0.55, zorder=2)
ax1.text(5.45, 101.5, 'placebo\nlevel', fontsize=7.0, color='#2471A3',
         ha='right', va='bottom', style='italic')

# X-axis labels including N= and SV-ADF indicator
xtick_labels = []
for lbl, n, sv in zip(group_labels, sample_n, svadf_w1_detected):
    sv_str = 'SV-ADF W1: ✓' if sv else 'SV-ADF W1: –'
    xtick_labels.append(f'{lbl}\n(N={n})\n{sv_str}')

ax1.set_xticks(x)
ax1.set_xticklabels(xtick_labels, fontsize=8.0, linespacing=1.4)
ax1.tick_params(axis='x', pad=6)
ax1.set_ylim(0, 122)
ax1.set_yticks([0, 25, 50, 75, 100])
ax1.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
ax1.set_ylabel('Share of instruments rejecting GSADF null', fontsize=9.5, labelpad=8)
ax1.set_title(
    'Panel A — Strong-form test\nGSADF null rejection rate by group',
    fontsize=10, fontweight='bold', pad=12, loc='left'
)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL B — Post-Q4-2023 episode-day fraction
# ─────────────────────────────────────────────────────────────────────────────
for i, val in enumerate(post_q4_pct):
    if val is None:
        # Livestock: no episodes
        ax2.bar(x[i], 3.5, width=bw, color='#E8EAED', edgecolor='#9FA6AD',
                linewidth=0.8, hatch='////', zorder=3)
        ax2.text(x[i], 5.0, 'N/A', ha='center', va='bottom',
                 fontsize=9, color='#7F8C8D', style='italic')
    else:
        lbl = f'{val:.0f}%' + (' *' if i == 1 else '')
        ax2.bar(x[i], val, width=bw,
                color=BAR_COLORS[i], edgecolor=EDGE_COLORS[i],
                linewidth=0.8, zorder=3)
        ax2.text(x[i], val + 1.5, lbl,
                 ha='center', va='bottom', fontsize=10.5,
                 fontweight='bold', color='#1C2833')

# Annotation for confound (precious metals) — SV-ADF W1 confirmed
ax2.annotate(
    'SV-ADF W1:\nGLD (Jan 2025)\nSLV (Sep 2025)',
    xy=(x[2], 89.0),
    xytext=(x[2] + 1.05, 65),
    fontsize=7.5, color='#A0640A',
    arrowprops=dict(arrowstyle='->', color='#A0640A', lw=0.9,
                    connectionstyle='arc3,rad=-0.15'),
    ha='left', va='center'
)

# X-axis labels
ax2.set_xticks(x)
ax2.set_xticklabels(
    [f'{lbl}\n(N={n})' for lbl, n in zip(group_labels, sample_n)],
    fontsize=8.0, linespacing=1.4
)
ax2.tick_params(axis='x', pad=6)
ax2.set_ylim(0, 103)
ax2.set_yticks([0, 25, 50, 75, 100])
ax2.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
ax2.set_ylabel('Episode-days falling after Q4 2023 (%)', fontsize=9.5, labelpad=8)
ax2.set_title(
    'Panel B — Conditional-form test\nShare of episode-days after Q4 2023',
    fontsize=10, fontweight='bold', pad=12, loc='left'
)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED LEGEND
# ─────────────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor='#C0392B', edgecolor='#922B21', label='AI-relevant groups'),
    mpatches.Patch(facecolor='#D4850A', edgecolor='#A0640A', label='Confound (precious metals)'),
    mpatches.Patch(facecolor='#2471A3', edgecolor='#1A5276', label='Placebo groups'),
    mpatches.Patch(facecolor='#808B96', edgecolor='#5D6D7E', label='Broad benchmark'),
]
fig.legend(
    handles=legend_handles, loc='lower center', ncol=4,
    fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.00),
    edgecolor='#CCCCCC', framealpha=0.95
)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTNOTE
# ─────────────────────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.04,
    ('* AI-relevant metals fraction driven entirely by PPLT (platinum, Jun 2025 – Feb 2026, 266 days). '
     'Copper (JJC) and palladium (PALL) contribute zero post-Q4-2023 episode-days.  '
     'Episode-day totals by group: AI Energy 501 d · AI Metals 424 d · Precious 874 d · '
     'Agriculture 1 315 d · Benchmark 1 374 d.\n'
     'Panel A SV-ADF W1 row (italic): episode detected (✓) or not (–) under the Sarkar-Wells (2026) '
     'volatility-robust procedure applied over the post-ChatGPT window (Nov 2021 – May 2026).'),
    ha='center', fontsize=7.3, color='#555555'
)

# ─────────────────────────────────────────────────────────────────────────────
# SUPTITLE
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Figure 6 — H6: Commodity-market evidence for AI-demand propagation\n'
    'GSADF rejection rates and post-Q4 2023 episode timing by commodity group',
    fontsize=11, fontweight='bold', y=1.01
)

plt.tight_layout(rect=[0, 0.10, 1, 0.99])

from config import settings
out = settings.FIGURES_DIR / "thesis" / "fig6_commodity_propagation_v2.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")
