try:
    from scripts.plot_utils import setup_matplotlib, show_or_save
except ImportError:
    from plot_utils import setup_matplotlib, show_or_save
setup_matplotlib()

import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ── Load BLS UNRATE (same series as stationarity diagnostic) ──────────────────
bls = pd.read_parquet('./data/external/bls_unrate_monthly.parquet')
bls['date'] = pd.to_datetime(bls['date'])
bls = bls.set_index('date').sort_index()
ts = bls.loc['2010-01':'2019-12', 'unrate']

# ── Grid search ───────────────────────────────────────────────────────────────
# d=1, D=1, s=12 fixed from stationarity analysis (prior figure).
# Candidate non-seasonal orders: p,q ∈ {0,1,2,3}
# Candidate seasonal orders:    P,Q ∈ {0,1,2}
D, d, s = 1, 1, 12

records = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for p, q, P, Q in itertools.product(range(4), range(4), range(3), range(3)):
        try:
            fit = SARIMAX(
                ts,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            records.append(dict(p=p, q=q, P=P, Q=Q, AIC=fit.aic, BIC=fit.bic))
        except Exception:
            pass

aic_df = (pd.DataFrame(records)
           .sort_values('AIC')
           .reset_index(drop=True))

best     = aic_df.iloc[0]
best_pq  = (int(best.p), int(best.q))
best_PQ  = (int(best.P), int(best.Q))

# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    r'SARIMA$(p,1,q)(P,1,Q)_{12}$ Model Selection via AIC '
    f'(2010–2019 BLS UNRATE)',
    fontsize=12, fontweight='bold'
)

# ── Left: AIC heatmap over (p, q) at the best seasonal (P, Q) ────────────────
ax = axes[0]

# Pivot: fix P,Q to the best seasonal order; rows=q, cols=p
heat_df = (aic_df[(aic_df.P == best_PQ[0]) & (aic_df.Q == best_PQ[1])]
           .pivot(index='q', columns='p', values='AIC'))
heat_df = heat_df.sort_index(ascending=False)   # q=3 at top

# Normalise colours — emphasise contrast in the low-AIC region
vmin, vmax = heat_df.values.min(), heat_df.values.max()
im = ax.imshow(heat_df.values, cmap='RdYlGn_r', aspect='auto',
               vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax, label='AIC', fraction=0.046, pad=0.04)

ax.set_xticks(range(len(heat_df.columns)))
ax.set_xticklabels(heat_df.columns)
ax.set_yticks(range(len(heat_df.index)))
ax.set_yticklabels(heat_df.index)
ax.set_xlabel('Non-Seasonal AR Order  (p)')
ax.set_ylabel('Non-Seasonal MA Order  (q)')
ax.set_title(
    f'AIC Heatmap — Seasonal orders fixed at\n'
    f'$(P,D,Q)_{{12}}$ = ({best_PQ[0]}, 1, {best_PQ[1]})',
    fontsize=10
)

# Annotate each cell with its AIC value
for i, qi in enumerate(heat_df.index):
    for j, pi in enumerate(heat_df.columns):
        val = heat_df.loc[qi, pi]
        if np.isfinite(val):
            is_best = (pi == best_pq[0]) and (qi == best_pq[1])
            weight  = 'bold' if is_best else 'normal'
            color   = 'white' if is_best else '#333333'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=8, fontweight=weight, color=color)

# Highlight the global best cell
bi = list(heat_df.index).index(best_pq[1])
bj = list(heat_df.columns).index(best_pq[0])
ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                            fill=False, edgecolor='#1565C0', lw=2.5))

# ── Right: Top-15 models ranked by AIC ───────────────────────────────────────
ax2 = axes[1]

top15  = aic_df.head(15).copy()
labels = [
    f'({int(r.p)},1,{int(r.q)})({int(r.P)},1,{int(r.Q)})$_{{12}}$'
    for _, r in top15.iterrows()
]
# Normalise bar colours relative to the displayed range
norm_aic = (top15['AIC'] - top15['AIC'].min()) / (top15['AIC'].max() - top15['AIC'].min() + 1e-9)
bar_colors = plt.cm.RdYlGn_r(0.15 + norm_aic * 0.7)

bars = ax2.barh(range(len(top15)), top15['AIC'], color=bar_colors, edgecolor='white')
bars[0].set_edgecolor('#1565C0')
bars[0].set_linewidth(2.5)

ax2.set_yticks(range(len(top15)))
ax2.set_yticklabels(labels, fontsize=8.5)
ax2.invert_yaxis()
ax2.set_xlabel('AIC')
ax2.set_title('Top 15 Candidate Models (lower AIC = better)', fontsize=10)

# Annotate AIC values on bars
for i, (bar, val) in enumerate(zip(bars, top15['AIC'])):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
             f'{val:.1f}', va='center', fontsize=8,
             fontweight='bold' if i == 0 else 'normal')

ax2.spines[['top', 'right']].set_visible(False)

# Shared caption annotation
best_label = (f'Best model: SARIMA({int(best.p)},1,{int(best.q)})'
              f'({int(best.P)},1,{int(best.Q)})$_{{12}}$  '
              f'AIC = {best.AIC:.2f}')
fig.text(0.5, -0.03, best_label, ha='center', fontsize=10,
         fontweight='bold', color='#1565C0')

plt.tight_layout()

show_or_save(fig, 'sarima_aic_selection')
plt.close(fig)
