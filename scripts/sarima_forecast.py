"""Generate 60-step out-of-sample SARIMA forecast and cache to parquet.

Fits SARIMA(1,1,0)(0,1,0)_12 on BLS UNRATE 2010-2019, produces a 60-step
ahead forecast with 95% prediction intervals, and saves to
output/sarima_forecast.parquet.

Set FORCE_RERUN = True to regenerate from scratch.
"""
import warnings
import pathlib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

OUT_PATH    = pathlib.Path('output/sarima_forecast.parquet')
FORCE_RERUN = False

if OUT_PATH.exists() and not FORCE_RERUN:
    print(f"[sarima_forecast] Loaded cached forecast from {OUT_PATH.name}")
    sarima_df = pd.read_parquet(OUT_PATH)
else:
    print("[sarima_forecast] Fitting SARIMA(1,1,0)(0,1,0)_12 on BLS UNRATE 2010–2019 …")
    bls = pd.read_parquet('./data/external/bls_unrate_monthly.parquet')
    bls['date'] = pd.to_datetime(bls['date'])
    bls = bls.set_index('date').sort_index()
    ts  = bls.loc['2010-01':'2019-12', 'unrate']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = SARIMAX(
            ts,
            order=(1, 1, 0),
            seasonal_order=(0, 1, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    fc  = fit.get_forecast(steps=60)
    ci  = fc.conf_int(alpha=0.05)

    sarima_df = pd.DataFrame({
        'tick':     range(60),
        'forecast': fc.predicted_mean.values,
        'ci_lower': ci.iloc[:, 0].values,
        'ci_upper': ci.iloc[:, 1].values,
    })

    # Store the in-sample terminal value (Dec 2019) as the anchor point
    sarima_df.attrs['anchor_unrate'] = float(ts.iloc[-1])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sarima_df.to_parquet(OUT_PATH, index=False)
    print(f"[sarima_forecast] Saved to {OUT_PATH.name}")
    print(f"  In-sample terminal value (Dec 2019): {ts.iloc[-1]*100:.2f}%")
    print(f"  60-month forecast range: "
          f"{sarima_df['forecast'].min()*100:.2f}%–{sarima_df['forecast'].max()*100:.2f}%")
