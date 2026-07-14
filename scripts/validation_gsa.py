"""Global Sensitivity Analysis — vacancy_rate elasticity (data generation only).

Runs 30 seeds × 3 vacancy_rate values (baseline −10%, baseline, baseline +10%)
in the zero-adoption control scenario and measures the steady-state unemployment
rate response, yielding a numerical elasticity estimate.

Cached to output/validation_gsa.parquet.
Set FORCE_RERUN = True to regenerate.
"""
import sys
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path('.').resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS
from scripts.bootstrap_runner import load_shared_data

GSA_PATH    = ROOT / 'output' / 'validation_gsa.parquet'
N_SEEDS     = 30
N_TICKS     = 72
BURN_IN     = 12
BASE_VR     = DEFAULT_PARAMS['vacancy_rate']   # 0.04
DELTAS      = [-0.10, 0.0, 0.10]              # ±10 % perturbation
FORCE_RERUN = True

if GSA_PATH.exists() and not FORCE_RERUN:
    pass  # loaded by plot script
else:
    worker_df, dist_matrix, occ_risk = load_shared_data()

    rows = []
    rng  = np.random.default_rng(42)

    for delta in DELTAS:
        vr     = BASE_VR * (1 + delta)
        params = {**DEFAULT_PARAMS, 'vacancy_rate': vr}
        label  = f'{delta:+.0%}'

        for s in range(N_SEEDS):
            seed = int(rng.integers(0, 2**31))
            sampled_df = worker_df.sample(
                n=len(worker_df), replace=True, random_state=seed
            ).reset_index(drop=True)

            m = LaborMarketModel(
                sampled_df,
                params=params,
                ai_active=False,
                seed=seed,
                skill_distance_matrix=dist_matrix,
                occ_risk_lookup=occ_risk,
                collect_agent_data=False,
            )
            for _ in range(N_TICKS):
                m.step()

            model_df  = m.datacollector.get_model_vars_dataframe()
            post_burn = model_df.iloc[BURN_IN:]
            mean_unemp = (1 - post_burn['Employment_Rate']).mean()

            rows.append({
                'delta':        delta,
                'vacancy_rate': vr,
                'label':        label,
                'seed':         seed,
                'mean_unemp':   mean_unemp,
            })

    gsa_df = pd.DataFrame(rows)
    GSA_PATH.parent.mkdir(parents=True, exist_ok=True)
    gsa_df.to_parquet(GSA_PATH, index=False)
