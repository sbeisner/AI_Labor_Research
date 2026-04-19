try:
    from scripts.plot_utils import setup_matplotlib, show_or_save
except ImportError:
    from plot_utils import setup_matplotlib, show_or_save
setup_matplotlib()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Load paired runs (AI scenario only) ──────────────────────────────────────
paired_df = pd.read_parquet('output/paired_runs.parquet')
ai = paired_df[paired_df['scenario'] == 'AI']

def agg(col):
    g = ai.groupby('tick')[col]
    return g.mean(), g.quantile(0.05), g.quantile(0.95)

fired_mean,   fired_p05,   fired_p95   = agg('Total_Fired')
hired_mean,   hired_p05,   hired_p95   = agg('Total_Hired')
newjobs_mean, newjobs_p05, newjobs_p95 = agg('New_Economy_Jobs')

ticks = fired_mean.index.values

# Net change = all hires − all separations
net_mean = hired_mean - fired_mean
net_p05  = hired_p05  - fired_p95
net_p95  = hired_p95  - fired_p05

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8),
                               gridspec_kw={'height_ratios': [3, 1]},
                               sharex=True)
fig.suptitle(
    'Gross Labor Churn Under AI Adoption (N=100 Seeds)',
    fontsize=12, fontweight='bold'
)

# ── Panel 1: Gross flows ──────────────────────────────────────────────────────
ax1.bar(ticks,  hired_mean.values,  color='#2e7d32', alpha=0.75, width=0.8,
        label='Total Hires (All Channels)')
ax1.bar(ticks, -fired_mean.values,  color='#c62828', alpha=0.75, width=0.8,
        label='Job Separations (Turnover + Automation)')

# New-economy vacancies as a hatched overlay — a sub-component of the demand
# that created room for some of the hires above
ax1.bar(ticks, newjobs_mean.values, color='#81c784', alpha=0.9, width=0.8,
        hatch='//', edgecolor='#1b5e20', linewidth=0.5,
        label='of which: New-Economy Vacancies Created')

ax1.axhline(0, color='#333333', linewidth=0.8)
ax1.set_ylabel('Agents per Tick', fontsize=10)
ax1.set_title(
    'Gross Labor Flows — Creation vs. Destruction',
    fontsize=10
)
ax1.legend(fontsize=8.5, loc='lower left')
ax1.spines[['top', 'right']].set_visible(False)

# ── Panel 2: Net change ───────────────────────────────────────────────────────
ax2.fill_between(ticks, net_p05.values, net_p95.values,
                 alpha=0.20, color='#1565C0', label='5th–95th Percentile')
ax2.plot(ticks, net_mean.values,
         color='#1565C0', linewidth=2.0, label='Mean Net Change')
ax2.axhline(0, color='#333333', linewidth=0.9, linestyle='--')

# Shade above/below zero
ax2.fill_between(ticks,
                 np.minimum(net_mean.values, 0),
                 np.zeros(len(ticks)),
                 alpha=0.15, color='#c62828')
ax2.fill_between(ticks,
                 np.zeros(len(ticks)),
                 np.maximum(net_mean.values, 0),
                 alpha=0.15, color='#2e7d32')

ax2.set_xlabel('Simulation Tick (Months Post Burn-In)', fontsize=10)
ax2.set_ylabel('Net Change\n(Agents)', fontsize=10)
ax2.set_title(
    'Net Employment Change per Tick — Masking Gross Churn Above',
    fontsize=10
)
ax2.legend(fontsize=8.5, loc='lower right')
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_xlim(-0.5, 59.5)

plt.tight_layout()
show_or_save(fig, 'abm_churn_plot')
plt.close(fig)
