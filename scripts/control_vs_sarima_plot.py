try:
    from scripts.plot_utils import setup_matplotlib, show_or_save
except ImportError:
    from plot_utils import setup_matplotlib, show_or_save
setup_matplotlib()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load cached results ───────────────────────────────────────────────────────
ctrl_df   = pd.read_parquet('output/control_runs.parquet')
sarima_df = pd.read_parquet('output/sarima_forecast.parquet')

# Clamp SARIMA CI lower bound at 0 (unemployment cannot be negative)
sarima_df['ci_lower'] = sarima_df['ci_lower'].clip(lower=0.0)

# ── Aggregate control runs: mean + 5th/95th percentile band ──────────────────
ctrl_agg = (
    ctrl_df.groupby('tick')['unemployment_rate']
    .agg(
        mean=('mean'),
        p05=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
    )
    .reset_index()
)

n_runs = ctrl_df['seed'].nunique()
ticks  = np.arange(60)

# ── Initial unemployment rates for annotation ─────────────────────────────────
abm_initial_u   = float(ctrl_agg.loc[ctrl_agg['tick'] == 0, 'mean'])
sarima_t0_fc    = float(sarima_df.loc[sarima_df['tick'] == 0, 'forecast'])
sarima_anchor   = float(ctrl_df['unemployment_rate'].mean())  # fallback

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))

# SARIMA forecast + CI
ax.plot(ticks, sarima_df['forecast'] * 100,
        color='#1565C0', linewidth=2.2, zorder=3,
        label=r'SARIMA$(1,1,0)(0,1,0)_{12}$ Forecast')
ax.fill_between(ticks,
                sarima_df['ci_lower'] * 100,
                sarima_df['ci_upper'] * 100,
                alpha=0.18, color='#1565C0', zorder=2,
                label='SARIMA 95% Prediction Interval')

# ABM control band + mean
ax.fill_between(ticks,
                ctrl_agg['p05'] * 100,
                ctrl_agg['p95'] * 100,
                alpha=0.22, color='#C62828', zorder=2,
                label=f'ABM Control 5th–95th Percentile (N={n_runs})')
ax.plot(ticks, ctrl_agg['mean'] * 100,
        color='#C62828', linewidth=2.2, linestyle='--', zorder=3,
        label='ABM Control Mean')

# Horizontal reference line at ABM initial level
ax.axhline(abm_initial_u * 100, color='#C62828', linewidth=0.8,
           linestyle=':', alpha=0.5)

# Annotate calibration gap at tick 0
ax.annotate(
    f'Calibration gap\n'
    f'ABM: {abm_initial_u*100:.1f}%\n'
    f'SARIMA: {sarima_t0_fc*100:.1f}%',
    xy=(0, (abm_initial_u + sarima_t0_fc) / 2 * 100),
    xytext=(8, (abm_initial_u + sarima_t0_fc) / 2 * 100 + 0.3),
    fontsize=8, color='#555555',
    arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8),
)

ax.set_xlabel('Simulation Tick (Months)', fontsize=11)
ax.set_ylabel('Unemployment Rate (%)', fontsize=11)
ax.set_xlim(-1, 60)
ax.set_title(
    'Zero-Adoption ABM Control vs. SARIMA Baseline — Convergence Validation',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=9, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
show_or_save(fig, 'control_vs_sarima_plot')
plt.close(fig)
