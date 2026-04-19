try:
    from scripts.plot_utils import setup_matplotlib, show_or_save
except ImportError:
    from plot_utils import setup_matplotlib, show_or_save
setup_matplotlib()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── Load diagnostic snapshots ─────────────────────────────────────────────────
snap = pd.read_parquet('output/validation_snapshots.parquet')

firm_t0  = snap['firm_size_t0'].dropna().values.astype(int)
firm_t60 = snap['firm_size_t60'].dropna().values.astype(int)
wage_t0  = snap['wage_t0'].dropna().values
wage_t60 = snap['wage_t60'].dropna().values

# ── K-S tests ─────────────────────────────────────────────────────────────────
ks_firm = stats.ks_2samp(firm_t0, firm_t60)
ks_wage = stats.ks_2samp(np.log1p(wage_t0), np.log1p(wage_t60))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Kolmogorov–Smirnov Distributional Stability Test\n'
             '(Tick 0 vs. Tick 60 — Zero-Adoption Control)',
             fontsize=11, fontweight='bold')

def plot_ks(ax, x0, x60, ks_result, title, xlabel, log_scale=False):
    if log_scale:
        x0_plot  = np.log1p(x0)
        x60_plot = np.log1p(x60)
        xlabel   = f'log(1 + {xlabel})'
    else:
        x0_plot, x60_plot = x0, x60

    bins = np.linspace(min(x0_plot.min(), x60_plot.min()),
                       max(x0_plot.max(), x60_plot.max()), 50)
    ax.hist(x0_plot,  bins=bins, density=True, alpha=0.55,
            color='#1565C0', label='Tick 0 (Initial)')
    ax.hist(x60_plot, bins=bins, density=True, alpha=0.55,
            color='#C62828', label='Tick 60 (Post-Simulation)')

    verdict = 'Fail to Reject $H_0$' if ks_result.pvalue > 0.05 else 'Reject $H_0$'
    color   = '#2e7d32' if ks_result.pvalue > 0.05 else '#c62828'
    ax.text(0.97, 0.97,
            f'K-S stat = {ks_result.statistic:.4f}\n'
            f'p-value  = {ks_result.pvalue:.4f}\n'
            f'{verdict}',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=color, linewidth=1.5),
            color=color)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

plot_ks(axes[0], firm_t0, firm_t60, ks_firm,
        'Firm Size Distribution\n(Zipf power-law stability)',
        'Firm Size (Workers)', log_scale=False)

plot_ks(axes[1], wage_t0, wage_t60, ks_wage,
        'Wage Distribution\n(Mincer equation stability)',
        'Annual Wage ($)', log_scale=True)

plt.tight_layout()
show_or_save(fig, 'validation_ks')
plt.close(fig)
