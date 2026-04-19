try:
    from scripts.plot_utils import setup_matplotlib, show_or_save
except ImportError:
    from plot_utils import setup_matplotlib, show_or_save
setup_matplotlib()

import warnings
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

# ── Load data ─────────────────────────────────────────────────────────────────
bls = pd.read_parquet('./data/external/bls_unrate_monthly.parquet')
bls['date'] = pd.to_datetime(bls['date'])
bls = bls.set_index('date').sort_index()
ts = bls.loc['2010-01':'2019-12', 'unrate']

# ── Fit SARIMA(1,1,0)(0,1,0)_{12} ────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fit = SARIMAX(
        ts,
        order=(1, 1, 0),
        seasonal_order=(0, 1, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

# ── tsdiag-equivalent: 3-panel diagnostic figure ─────────────────────────────
residuals = fit.resid.dropna()
std_resid = residuals / residuals.std()

lb = acorr_ljungbox(residuals, lags=range(1, 25), return_df=True)

fig, axes = plt.subplots(3, 1, figsize=(10, 9))
fig.suptitle(
    r'SARIMA$(1,1,0)(0,1,0)_{12}$ Model Diagnostics',
    fontsize=13, fontweight='bold'
)

# Panel 1 — standardised residuals
ax1 = axes[0]
ax1.plot(std_resid.index, std_resid.values, color='#2c3e50', linewidth=1)
ax1.axhline(0,  color='grey',    linestyle='--', linewidth=0.8)
ax1.axhline( 3, color='#e74c3c', linestyle=':',  linewidth=0.8, label=r'$\pm 3\sigma$')
ax1.axhline(-3, color='#e74c3c', linestyle=':',  linewidth=0.8)
ax1.set_ylabel('Standardised Residuals')
ax1.set_title('Standardised Residuals')
ax1.legend(fontsize=8, loc='upper right')

# Panel 2 — ACF of residuals
plot_acf(residuals, lags=24, alpha=0.05, ax=axes[1],
         title='ACF of Residuals')
axes[1].set_xlabel('Lag (Months)')
axes[1].set_ylabel('Autocorrelation')

# Panel 3 — Ljung-Box p-values
ax3 = axes[2]
ax3.plot(lb.index, lb['lb_pvalue'], 'o', color='#2980b9',
         markersize=5, label='Ljung\u2013Box p-value')
ax3.axhline(0.05, color='#e74c3c', linestyle='--', linewidth=1,
            label='p = 0.05 threshold')
ax3.set_ylim(0, 1)
ax3.set_xlim(0, 25)
ax3.set_xlabel('Lag')
ax3.set_ylabel('p-value')
ax3.set_title(r'Ljung\u2013Box Test $p$-values  (H$_0$: no residual autocorrelation)')
ax3.legend(fontsize=9)

plt.tight_layout()
show_or_save(fig, 'sarima_diagnostics')
plt.close(fig)
