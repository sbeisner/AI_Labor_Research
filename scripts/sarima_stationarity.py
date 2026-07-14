try:
    from scripts.plot_utils import setup_matplotlib, show_or_save
except ImportError:
    from plot_utils import setup_matplotlib, show_or_save
setup_matplotlib()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ── Load BLS UNRATE (seasonally adjusted, monthly, 2010–2019) ─────────────────
# Source: U.S. Bureau of Labor Statistics via FRED (series UNRATE).
# Pre-pandemic window used to establish the SARIMA control baseline before
# any AI-adoption signal contaminates the series.
bls = pd.read_parquet('./data/external/bls_unrate_monthly.parquet')
bls['date'] = pd.to_datetime(bls['date'])
bls = bls.set_index('date').sort_index()
ts = bls.loc['2010-01':'2019-12', 'unrate']

# ── Stationarity test and differencing ───────────────────────────────────────
adf_raw = adfuller(ts, autolag='AIC')
ts_diff = ts.diff(1).diff(12).dropna()   # 1st-order + seasonal (12-month) difference

# ── Build figure ──────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(10, 8))

# 1. Top row (spanning both columns): differenced series
ax_ts = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax_ts.plot(ts_diff.index, ts_diff.values, color='#2c3e50', linewidth=1.5)
ax_ts.set_title(
    "Differenced BLS Unemployment Rate\n"
    "(1st-Order Non-Seasonal + 1st-Order Seasonal, 2010–2019)"
)
ax_ts.set_ylabel("Differenced Rate")
ax_ts.set_xlabel("Year")

# 2. Bottom left: ACF
ax_acf = plt.subplot2grid((2, 2), (1, 0))
plot_acf(ts_diff, lags=24, alpha=0.05, ax=ax_acf,
         title="Autocorrelation Function (ACF)")
ax_acf.set_xlabel("Lags (Months)")
ax_acf.set_ylabel("Autocorrelation")

# 3. Bottom right: PACF
ax_pacf = plt.subplot2grid((2, 2), (1, 1))
plot_pacf(ts_diff, lags=24, alpha=0.05, ax=ax_pacf,
          title="Partial Autocorrelation Function (PACF)")
ax_pacf.set_xlabel("Lags (Months)")
ax_pacf.set_ylabel("Partial Autocorrelation")

plt.tight_layout()

show_or_save(fig, 'sarima_stationarity')
plt.close(fig)
