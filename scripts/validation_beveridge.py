try:
    from scripts.plot_utils import setup_matplotlib, show_or_save
except ImportError:
    from plot_utils import setup_matplotlib, show_or_save
setup_matplotlib()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Load control scenario from paired runs ────────────────────────────────────
df    = pd.read_parquet('output/paired_runs.parquet')
ctrl  = df[df['scenario'] == 'Control'].copy()
N_WORKERS = 10_000  # fixed population per seed

ctrl['vacancy_rate']     = ctrl['Total_Vacancies']  / N_WORKERS
ctrl['unemployment_rate'] = ctrl['unemployment_rate']  # already computed

# ── Aggregate: mean per tick across seeds ────────────────────────────────────
agg = ctrl.groupby('tick')[['vacancy_rate', 'unemployment_rate']].mean().reset_index()

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

# Color-code ticks from early (light) to late (dark)
colors = cm.Blues(np.linspace(0.35, 0.95, len(agg)))
sc = ax.scatter(agg['unemployment_rate'] * 100,
                agg['vacancy_rate'] * 100,
                c=colors, s=50, zorder=3)

# Connect points with a thin line to show the path
ax.plot(agg['unemployment_rate'] * 100,
        agg['vacancy_rate'] * 100,
        color='#90CAF9', linewidth=0.8, zorder=2)

# Annotate start and end
ax.annotate('Tick 0', xy=(agg['unemployment_rate'].iloc[0]  * 100,
                           agg['vacancy_rate'].iloc[0]       * 100),
            xytext=(6, 3), textcoords='offset points', fontsize=8, color='#1565C0')
ax.annotate('Tick 59', xy=(agg['unemployment_rate'].iloc[-1] * 100,
                            agg['vacancy_rate'].iloc[-1]      * 100),
            xytext=(6, -8), textcoords='offset points', fontsize=8, color='#1565C0')

# Fit and overlay a reference hyperbola (u × v = k) for visual comparison
u_range = np.linspace(agg['unemployment_rate'].min() * 100 * 0.95,
                       agg['unemployment_rate'].max() * 100 * 1.05, 200)
k = np.median((agg['unemployment_rate'] * 100) * (agg['vacancy_rate'] * 100))
ax.plot(u_range, k / u_range, color='#E53935', linewidth=1.2,
        linestyle='--', alpha=0.7, label=r'Reference: $u \times v = k$ (Beveridge)')

# Colorbar for tick progression
sm = plt.cm.ScalarMappable(cmap='Blues',
     norm=plt.Normalize(vmin=0, vmax=59))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label('Simulation Tick', fontsize=9)

ax.set_xlabel('Unemployment Rate (%)', fontsize=11)
ax.set_ylabel('Vacancy Rate (% of labor force)', fontsize=11)
ax.set_title('Emergent Beveridge Curve — Zero-Adoption Control\n'
             '(Downward slope validates Mortensen-Pissarides matching mechanics)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
show_or_save(fig, 'validation_beveridge')
plt.close(fig)
