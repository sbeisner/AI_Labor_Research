try:
    from scripts.plot_utils import setup_matplotlib, show_or_save
except ImportError:
    from plot_utils import setup_matplotlib, show_or_save
setup_matplotlib()

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.LaborMarketModel import DEFAULT_PARAMS

GSA_PATH = pathlib.Path('output/validation_gsa.parquet')
BASE_VR  = DEFAULT_PARAMS['vacancy_rate']

gsa_df  = pd.read_parquet(GSA_PATH)
summary = gsa_df.groupby('delta')['mean_unemp'].agg(['mean', 'std']).reset_index()
summary.columns = ['delta', 'u_mean', 'u_std']

u_base  = summary.loc[summary['delta'] == 0.0,   'u_mean'].values[0]
u_plus  = summary.loc[summary['delta'] == 0.10,  'u_mean'].values[0]
u_minus = summary.loc[summary['delta'] == -0.10, 'u_mean'].values[0]

elas_up   = ((u_plus  - u_base) / u_base) / 0.10
elas_down = ((u_minus - u_base) / u_base) / (-0.10)
elas_avg  = (elas_up + elas_down) / 2

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    'Global Sensitivity Analysis — vacancy_rate ±10%\n'
    '(Zero-Adoption Control, N=30 Seeds per Condition)',
    fontsize=11, fontweight='bold'
)

COLORS   = ['#C62828', '#1565C0', '#2E7D32']
LABELS   = ['−10% vacancy_rate', 'Baseline (4%)', '+10% vacancy_rate']
PALETTES = ['#EF9A9A', '#90CAF9', '#A5D6A7']
order    = [-0.10, 0.0, 0.10]

# Panel 1: Box + strip
ax = axes[0]
for pos, (delta, color, pal, lbl) in enumerate(zip(order, COLORS, PALETTES, LABELS)):
    vals = gsa_df.loc[gsa_df['delta'] == delta, 'mean_unemp'].values * 100
    ax.boxplot(vals, positions=[pos], widths=0.45,
               patch_artist=True,
               medianprops=dict(color='white', linewidth=2),
               boxprops=dict(facecolor=pal, linewidth=1.2),
               whiskerprops=dict(color=color, linewidth=1.2),
               capprops=dict(color=color, linewidth=1.2),
               flierprops=dict(marker='o', markerfacecolor=color,
                               markersize=4, alpha=0.5))
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(vals))
    ax.scatter(pos + jitter, vals, color=color, alpha=0.55, s=18, zorder=3)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(LABELS, fontsize=8.5)
ax.set_ylabel('Mean Unemployment Rate (%)', fontsize=10)
ax.set_title('Distribution of Steady-State Unemployment\nby vacancy_rate Condition', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)

# Panel 2: Mean ± 1 SD with elasticity annotation
ax2 = axes[1]
vr_vals = [BASE_VR * (1 + d) * 100 for d in order]
u_means = [summary.loc[summary['delta'] == d, 'u_mean'].values[0] * 100 for d in order]
u_stds  = [summary.loc[summary['delta'] == d, 'u_std'].values[0]  * 100 for d in order]

ax2.errorbar(vr_vals, u_means, yerr=u_stds,
             fmt='o-', color='#1565C0', linewidth=2.0,
             capsize=5, markersize=7, label='Mean ± 1 SD')
ax2.axvline(BASE_VR * 100, color='#333333', linewidth=0.9,
            linestyle='--', alpha=0.7, label='Baseline vacancy_rate')
ax2.text(0.05, 0.95,
         f'Arc Elasticity\n'
         f'  +10%: {elas_up:+.3f}\n'
         f'  −10%: {elas_down:+.3f}\n'
         f'  Avg:  {elas_avg:+.3f}',
         transform=ax2.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#1565C0', linewidth=1.3))

ax2.set_xlabel('vacancy_rate (% of occupation size)', fontsize=10)
ax2.set_ylabel('Mean Unemployment Rate (%)', fontsize=10)
ax2.set_title('Unemployment Response to vacancy_rate Perturbation\n(Arc Elasticity)', fontsize=10)
ax2.legend(fontsize=9)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
show_or_save(fig, 'validation_gsa_plot')
plt.close(fig)
