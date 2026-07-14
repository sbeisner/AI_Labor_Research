"""Run Zero-Adoption (control) Monte Carlo simulations and cache to parquet.

Runs N_RUNS independent LaborMarketModel simulations with ai_active=False
over T=60 ticks, collecting aggregate unemployment rates per tick.
Results are saved to output/control_runs.parquet so the manuscript does not
re-run simulations on every render.

Set FORCE_RERUN = True to regenerate from scratch.
"""
import sys
import pathlib

ROOT = pathlib.Path('.').resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS
from scripts.bootstrap_runner import load_shared_data

N_RUNS    = 100
N_TICKS   = 72   # Fix 3: run 72 ticks; first 12 are burn-in (discarded)
BURN_IN   = 12   # ticks to discard for initialization artifact removal
OUT_PATH  = ROOT / 'output' / 'control_runs.parquet'
FORCE_RERUN = True   # model logic changed — must regenerate

if OUT_PATH.exists() and not FORCE_RERUN:
    print(f"[control_simulation] Loaded cached results from {OUT_PATH.name}")
    ctrl_df = pd.read_parquet(OUT_PATH)
else:
    print(f"[control_simulation] Running {N_RUNS} control simulations × {N_TICKS} ticks "
          f"({BURN_IN}-tick burn-in) …")
    worker_df, dist_matrix, occ_risk = load_shared_data()
    import numpy as np

    rows = []
    for seed in range(N_RUNS):
        # Resample with replacement per seed for proper bootstrap CIs
        rng_seed = np.random.default_rng(seed)
        sampled_df = worker_df.sample(
            n=len(worker_df), replace=True, random_state=int(rng_seed.integers(0, 2**31))
        ).reset_index(drop=True)

        m = LaborMarketModel(
            sampled_df,
            params=DEFAULT_PARAMS,
            ai_active=False,
            seed=seed,
            skill_distance_matrix=dist_matrix,
            occ_risk_lookup=occ_risk,
            collect_agent_data=False,
        )
        for _ in range(N_TICKS):
            m.step()
        df = m.datacollector.get_model_vars_dataframe().copy()
        df.index.name = 'tick'
        df['seed'] = seed
        # Fix 3: discard burn-in ticks; renumber remaining ticks 0–59
        df = df.reset_index()
        df = df[df['tick'] >= BURN_IN].copy()
        df['tick'] = df['tick'] - BURN_IN
        rows.append(df)
        if (seed + 1) % 10 == 0:
            print(f"  … {seed + 1}/{N_RUNS} runs complete")

    ctrl_df = pd.concat(rows, ignore_index=True)
    ctrl_df['unemployment_rate'] = 1.0 - ctrl_df['Employment_Rate']
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ctrl_df.to_parquet(OUT_PATH, index=False)
    print(f"[control_simulation] Saved to {OUT_PATH.name}")
