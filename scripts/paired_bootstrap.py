"""Paired AI vs. Control bootstrap Monte Carlo runner.

For each seed, runs BOTH the AI scenario and the zero-adoption Control with
the identical seed, so macroeconomic noise cancels when computing the
treatment effect (AI − Control) at each tick.

Parallelized via mp.Pool (persistent workers, no loky idle-timeout churn):
shared data (worker_df, skill_distance_matrix, occ_risk_lookup) is loaded
once per worker process via the pool initializer — no per-seed disk I/O.

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
import multiprocessing as mp
from datetime import datetime, timedelta

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS

N_RUNS      = 100
N_TICKS     = 180   # run 180; discard first BURN_IN (retains 120 analysis ticks)
BURN_IN     = 60    # extended from 24 — model needs ~60 ticks to converge from
                    # fully-employed initial state to steady-state equilibrium
OUT_PATH    = ROOT / 'output' / 'paired_runs.parquet'
FORCE_RERUN = True   # model logic changed — must regenerate
REPORT_EVERY = 10

# Cap workers to avoid memory exhaustion.
# Each worker runs two full 144-tick models per seed (~300 MB peak).
# 8 workers × 300 MB ≈ 2.4 GB — safe on a 16 GB machine.
N_WORKERS = min(mp.cpu_count(), 8)

# ── Worker-process state (populated by _worker_init in each pool process) ───────
_worker_df   = None
_dist_matrix = None
_occ_risk    = None


def _worker_init():
    """Load shared data once per worker process (mp.Pool initializer).

    Called exactly once per worker at pool creation.  Subsequent calls to
    run_seed() in the same worker process reuse the globals without disk I/O.
    """
    global _worker_df, _dist_matrix, _occ_risk
    from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()


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
    rng = np.random.default_rng(seed)
    sampled_df = _worker_df.sample(
        n=len(_worker_df), replace=True, random_state=int(rng.integers(0, 2**31))
    ).reset_index(drop=True)

    rows = []
    for scenario, ai_active in (("AI", True), ("Control", False)):
        m = LaborMarketModel(
            sampled_df,
            params=DEFAULT_PARAMS,
            ai_active=ai_active,
            seed=seed,
            skill_distance_matrix=_dist_matrix,
            occ_risk_lookup=_occ_risk,
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


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--n-runs", type=int, default=N_RUNS,
                   help=f"number of paired seeds (default: {N_RUNS})")
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--smoke", action="store_true",
                   help="fast schema check: 2 seeds → paired_runs_smoke.parquet")
    args = p.parse_args()

    if args.smoke:
        n_runs = 2
        out_path = pathlib.Path(args.out).with_name("paired_runs_smoke.parquet")
    else:
        n_runs = args.n_runs
        out_path = pathlib.Path(args.out)

    print(
        f"[paired_bootstrap] Running {n_runs} paired seeds × {N_TICKS} ticks "
        f"({BURN_IN}-tick burn-in) in parallel …\n"
        f"  workers : {N_WORKERS} (cpu_count={mp.cpu_count()}, capped at 8)\n"
        f"  output  : {out_path}",
        flush=True,
    )

    start_time = time.monotonic()
    start_wall = datetime.now()
    print(f"  started at : {start_wall:%Y-%m-%d %H:%M:%S}\n", flush=True)

    all_frames = []

    with mp.Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
        for i, result in enumerate(
            pool.imap_unordered(run_seed, range(n_runs)),
            start=1,
        ):
            all_frames.append(result)
            if i % REPORT_EVERY == 0 or i == n_runs:
                elapsed   = time.monotonic() - start_time
                rate      = i / elapsed
                remaining = (n_runs - i) / rate if rate > 0 else 0
                eta       = datetime.now() + timedelta(seconds=remaining)
                print(
                    f"  [{datetime.now():%H:%M:%S}]  {i:>3}/{n_runs} seeds done"
                    f"  |  elapsed: {elapsed/60:.1f}m"
                    f"  |  ETA: {eta:%H:%M:%S}",
                    flush=True,
                )

    paired_df = pd.concat(all_frames, ignore_index=True)
    paired_df["unemployment_rate"] = 1.0 - paired_df["Employment_Rate"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    paired_df.to_parquet(out_path, index=False)
    print(f"\n[paired_bootstrap] Saved {len(paired_df)} rows to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
