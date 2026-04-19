"""Paired AI vs. Control bootstrap Monte Carlo runner.

For each seed, runs BOTH the AI scenario and the zero-adoption Control with
the identical seed, so macroeconomic noise cancels when computing the
treatment effect (AI − Control) at each tick.

Run parameters:
  N_RUNS  = 100 paired seeds
  N_TICKS = 144  (24-tick burn-in discarded; 120 analysis ticks retained)
  BURN_IN = 24

Output: output/paired_runs.parquet — tidy DataFrame with columns:
    tick, seed, scenario (AI|Control), Employment_Rate, Unemployed_Count,
    Mean_Wage, Emp_Rate_Entry, Emp_Rate_Senior, Emp_Rate_Q1_Low … Q5_High,
    Retraining_Count, Retrained_Share, New_Economy_Jobs, Total_Vacancies,
    Total_Hired, Total_Fired, unemployment_rate

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

N_RUNS      = 100
N_TICKS     = 144   # run 144; discard first BURN_IN (retains 120 analysis ticks)
BURN_IN     = 24
OUT_PATH    = ROOT / 'output' / 'paired_runs.parquet'
FORCE_RERUN = True   # model logic changed — must regenerate

if OUT_PATH.exists() and not FORCE_RERUN:
    print(f"[paired_bootstrap] Loaded cached results from {OUT_PATH.name}")
    paired_df = pd.read_parquet(OUT_PATH)
else:
    print(f"[paired_bootstrap] Running {N_RUNS} paired seeds × {N_TICKS} ticks "
          f"({BURN_IN}-tick burn-in) …")
    worker_df, dist_matrix, occ_risk = load_shared_data()

    import numpy as np

    rows = []
    for seed in range(N_RUNS):
        # Resample worker_df with replacement using this seed — this is the
        # bootstrap step that gives the CI its proper interpretation as
        # uncertainty over labor force composition, not just within-run noise.
        rng_seed = np.random.default_rng(seed)
        sampled_df = worker_df.sample(
            n=len(worker_df), replace=True, random_state=int(rng_seed.integers(0, 2**31))
        ).reset_index(drop=True)

        for scenario, ai_active in (("AI", True), ("Control", False)):
            m = LaborMarketModel(
                sampled_df,
                params=DEFAULT_PARAMS,
                ai_active=ai_active,
                seed=seed,
                skill_distance_matrix=dist_matrix,
                occ_risk_lookup=occ_risk,
                collect_agent_data=False,
            )
            for _ in range(N_TICKS):
                m.step()
            df = m.datacollector.get_model_vars_dataframe().copy()
            df.index.name = 'tick'
            df['seed']     = seed
            df['scenario'] = scenario
            df = df.reset_index()
            # Discard burn-in; renumber retained ticks 0–59
            df = df[df['tick'] >= BURN_IN].copy()
            df['tick'] = df['tick'] - BURN_IN
            rows.append(df)

        if (seed + 1) % 10 == 0:
            print(f"  … {seed + 1}/{N_RUNS} seeds complete")

    paired_df = pd.concat(rows, ignore_index=True)
    paired_df['unemployment_rate'] = 1.0 - paired_df['Employment_Rate']
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    paired_df.to_parquet(OUT_PATH, index=False)
    print(f"[paired_bootstrap] Saved to {OUT_PATH.name}")
