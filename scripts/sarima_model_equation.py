import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from IPython.display import display, Markdown

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

# ── Extract coefficient and SE ────────────────────────────────────────────────
phi1      = fit.params['ar.L1']
phi1_se   = fit.bse['ar.L1']
sigma2    = fit.params['sigma2']
sigma2_se = fit.bse['sigma2']

# ── Emit model equation in LaTeX with SEs underset ───────────────────────────
# Form: (1 - φ̂₁ B)(1-B)(1-B^12) y_t = ε_t
eq = (
    "$$"
    "\\left(1 - "
    "\\underset{(" + f"{phi1_se:.4f}" + ")}{" + f"{phi1:.4f}" + "}"
    "\\, B\\right)(1-B)\\left(1-B^{12}\\right)\\, y_t = \\varepsilon_t,"
    "\\quad"
    "\\hat{\\sigma}^2 = "
    "\\underset{(" + f"{sigma2_se:.2e}" + ")}{" + f"{sigma2:.2e}" + "}"
    "$$"
)
display(Markdown(eq))
