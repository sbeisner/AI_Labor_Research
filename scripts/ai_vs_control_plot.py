try:
    from scripts.plot_utils import setup_matplotlib, show_or_save
except ImportError:
    from plot_utils import setup_matplotlib, show_or_save
setup_matplotlib()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────────────────────────────
paired_df = pd.read_parquet('output/paired_runs.parquet')
sarima_df = pd.read_parquet('output/sarima_forecast.parquet')
sarima_df['ci_lower'] = sarima_df['ci_lower'].clip(lower=0.0)

ai   = paired_df[paired_df['scenario'] == 'AI']
ctrl = paired_df[paired_df['scenario'] == 'Control']
ticks = np.arange(60)
n_seeds = paired_df['seed'].nunique()

# ── Aggregate: mean + 5th/95th percentile per tick ───────────────────────────
def agg(df, col):
    g = df.groupby('tick')[col]
    return g.mean(), g.quantile(0.05), g.quantile(0.95)

ai_u_mean,   ai_u_p05,   ai_u_p95   = agg(ai,   'unemployment_rate')
ctrl_u_mean, ctrl_u_p05, ctrl_u_p95 = agg(ctrl, 'unemployment_rate')

# ── Paired treatment effect: AI_unemp − Control_unemp per seed per tick ──────
pivot = paired_df.pivot_table(
    index=['seed', 'tick'], columns='scenario', values='unemployment_rate'
).reset_index()
pivot['effect'] = pivot['AI'] - pivot['Control']
eff_mean = pivot.groupby('tick')['effect'].mean()
eff_p05  = pivot.groupby('tick')['effect'].quantile(0.05)
eff_p95  = pivot.groupby('tick')['effect'].quantile(0.95)

# ── Figure: two panels ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [3, 2]})
fig.suptitle(
    'AI Adoption vs. Zero-Adoption Control — Aggregate Labor Market Outcomes',
    fontsize=12, fontweight='bold'
)

# ── Panel 1: Unemployment rate trajectories ───────────────────────────────────
ax1 = axes[0]

# SARIMA forecast
ax1.plot(ticks, sarima_df['forecast'] * 100,
         color='#1565C0', linewidth=1.8, linestyle=':', zorder=3,
         label=r'SARIMA$(1,1,0)(0,1,0)_{12}$ Forecast')
ax1.fill_between(ticks,
                 sarima_df['ci_lower'] * 100, sarima_df['ci_upper'] * 100,
                 alpha=0.10, color='#1565C0', zorder=1)

# Control band
ax1.fill_between(ticks, ctrl_u_p05 * 100, ctrl_u_p95 * 100,
                 alpha=0.20, color='#43A047', zorder=2)
ax1.plot(ticks, ctrl_u_mean * 100,
         color='#43A047', linewidth=2.0, linestyle='--', zorder=3,
         label=f'Zero-Adoption Control Mean (N={n_seeds})')

# AI band
ax1.fill_between(ticks, ai_u_p05 * 100, ai_u_p95 * 100,
                 alpha=0.20, color='#E53935', zorder=2)
ax1.plot(ticks, ai_u_mean * 100,
         color='#E53935', linewidth=2.2, zorder=4,
         label=f'AI Adoption Scenario Mean (N={n_seeds})')

ax1.set_ylabel('Unemployment Rate (%)', fontsize=10)
ax1.set_xlim(-0.5, 59.5)
ax1.set_title('Unemployment Rate Trajectories with 5th–95th Percentile Bands', fontsize=10)
ax1.legend(fontsize=8.5, loc='lower right')
ax1.spines[['top', 'right']].set_visible(False)

# ── Panel 2: Paired treatment effect (AI − Control) ───────────────────────────
ax2 = axes[1]

ax2.fill_between(ticks, eff_p05 * 100, eff_p95 * 100,
                 alpha=0.25, color='#E53935', zorder=2,
                 label='5th–95th Percentile')
ax2.plot(ticks, eff_mean * 100,
         color='#B71C1C', linewidth=2.2, zorder=3,
         label='Mean Paired Difference')
ax2.axhline(0.0, color='#555555', linewidth=0.9, linestyle='--')

# Shade positive divergence (AI worse than control)
ax2.fill_between(ticks,
                 np.zeros(60),
                 np.maximum(eff_mean * 100, 0),
                 alpha=0.15, color='#E53935', zorder=1)

ax2.set_xlabel('Simulation Tick (Months Post Burn-In)', fontsize=10)
ax2.set_ylabel('ΔUnemployment Rate\n(AI − Control, pp)', fontsize=10)
ax2.set_xlim(-0.5, 59.5)
ax2.set_title(
    'Paired Treatment Effect: AI-Induced Unemployment Gap',
    fontsize=10
)
ax2.legend(fontsize=8.5, loc='lower right')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
show_or_save(fig, 'ai_vs_control_plot')
plt.close(fig)
