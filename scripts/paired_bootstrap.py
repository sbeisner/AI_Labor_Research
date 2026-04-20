"""Paired AI vs. Control bootstrap Monte Carlo runner.

For each seed, runs BOTH the AI scenario and the zero-adoption Control with
the identical seed, so macroeconomic noise cancels when computing the
treatment effect (AI − Control) at each tick.

Parallelized via joblib (loky backend): shared data (worker_df,
skill_distance_matrix, occ_risk_lookup) is loaded once per worker process
and cached at module level — no per-seed disk I/O.

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
import time
from datetime import datetime, timedelta

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS

N_RUNS      = 100
N_TICKS     = 144   # run 144; discard first BURN_IN (retains 120 analysis ticks)
BURN_IN     = 24
OUT_PATH    = ROOT / 'output' / 'paired_runs.parquet'
FORCE_RERUN = True   # model logic changed — must regenerate
REPORT_EVERY = 10

# ── Per-process data cache ─────────────────────────────────────────────────────
# joblib (loky backend) spawns persistent worker processes. Each process
# handles multiple seeds. We load shared data once per process so subsequent
# seeds skip disk I/O entirely.
_PROCESS_CACHE: dict = {}


def _get_data():
    """Return (worker_df, dist_matrix, occ_risk) for the current process."""
    if "worker_df" not in _PROCESS_CACHE:
        from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
        (
            _PROCESS_CACHE["worker_df"],
            _PROCESS_CACHE["dist_matrix"],
            _PROCESS_CACHE["occ_risk"],
        ) = load_shared_data()
    return (
        _PROCESS_CACHE["worker_df"],
        _PROCESS_CACHE["dist_matrix"],
        _PROCESS_CACHE["occ_risk"],
    )


def run_seed(seed: int) -> pd.DataFrame:
    """Run one paired (AI + Control) simulation for the given seed.

    Resamples worker_df with replacement (the bootstrap step) so that
    confidence intervals reflect uncertainty over labor-force composition,
    not only within-run noise.

    Args:
        seed: integer seed; controls both the worker resample and the model RNG.

    Returns:
        DataFrame with burn-in discarded and ticks renumbered 0 … N_TICKS-BURN_IN-1.
    """
    worker_df, dist_matrix, occ_risk = _get_data()

    rng = np.random.default_rng(seed)
    sampled_df = worker_df.sample(
        n=len(worker_df), replace=True, random_state=int(rng.integers(0, 2**31))
    ).reset_index(drop=True)

    rows = []
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
        df.index.name = "tick"
        df["seed"]     = seed
        df["scenario"] = scenario
        df = df.reset_index()
        df = df[df["tick"] >= BURN_IN].copy()
        df["tick"] = df["tick"] - BURN_IN
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    if OUT_PATH.exists() and not FORCE_RERUN:
        print(f"[paired_bootstrap] Loaded cached results from {OUT_PATH.name}")
    else:
        print(
            f"[paired_bootstrap] Running {N_RUNS} paired seeds × {N_TICKS} ticks "
            f"({BURN_IN}-tick burn-in) in parallel …"
        )

        start_time = time.monotonic()
        start_wall = datetime.now()
        print(f"  started at : {start_wall:%Y-%m-%d %H:%M:%S}")

        all_frames = []
        for i, result in enumerate(
            Parallel(n_jobs=-1, return_as="generator_unordered")(
                delayed(run_seed)(s) for s in range(N_RUNS)
            ),
            start=1,
        ):
            all_frames.append(result)
            if i % REPORT_EVERY == 0 or i == N_RUNS:
                elapsed   = time.monotonic() - start_time
                rate      = i / elapsed
                remaining = (N_RUNS - i) / rate if rate > 0 else 0
                eta       = datetime.now() + timedelta(seconds=remaining)
                print(
                    f"  [{datetime.now():%H:%M:%S}]  {i:>3}/{N_RUNS} seeds done"
                    f"  |  elapsed: {elapsed/60:.1f}m"
                    f"  |  ETA: {eta:%H:%M:%S}"
                )

        paired_df = pd.concat(all_frames, ignore_index=True)
        paired_df["unemployment_rate"] = 1.0 - paired_df["Employment_Rate"]

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        paired_df.to_parquet(OUT_PATH, index=False)
        print(f"[paired_bootstrap] Saved to {OUT_PATH.name}")
