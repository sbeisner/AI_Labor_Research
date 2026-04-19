"""Bootstrap runner for the LaborMarketModel.

Designed to be imported by notebooks/05_bootstrap.ipynb.

run_one() is a top-level function so multiprocessing can pickle it cleanly.
Shared data (worker_df, skill_distance_matrix, occ_risk_lookup) is injected
into each worker process once via a Pool initializer — no per-run disk I/O.

Usage
-----
    from scripts.bootstrap_runner import (
        init_pool, run_one, load_shared_data
    )
    shared = load_shared_data()
    with mp.Pool(initializer=init_pool, initargs=shared) as pool:
        results = pool.map(run_one, range(N_RUNS))
    all_runs = pd.concat(results)
"""

import pathlib
import sys

# Ensure the project root is on the path when imported from notebooks/
_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS

# ── Module-level globals (populated by init_pool in each worker process) ──────
_worker_df   = None
_dist_matrix = None
_occ_risk    = None
_params      = None
_n_ticks     = None


def init_pool(worker_df, dist_matrix, occ_risk, params, n_ticks):
    """Pool initializer: copy shared data into each worker process once."""
    global _worker_df, _dist_matrix, _occ_risk, _params, _n_ticks
    _worker_df   = worker_df
    _dist_matrix = dist_matrix
    _occ_risk    = occ_risk
    _params      = params
    _n_ticks     = n_ticks


def run_one(seed: int) -> pd.DataFrame:
    """Run one paired AI + Control simulation; return tidy model-vars DataFrame."""
    rows = []
    for scenario, ai_active in (("AI", True), ("Control", False)):
        m = LaborMarketModel(
            _worker_df,
            params=_params,
            ai_active=ai_active,
            seed=seed,
            skill_distance_matrix=_dist_matrix,
            occ_risk_lookup=_occ_risk,
            collect_agent_data=False,
        )
        for _ in range(_n_ticks):
            m.step()
        df = m.datacollector.get_model_vars_dataframe().copy()
        df.index.name = "tick"
        df["seed"]     = seed
        df["scenario"] = scenario
        rows.append(df.reset_index())
    return pd.concat(rows, ignore_index=True)


def load_shared_data(data_dir=None):
    """Load worker_df, skill_distance_matrix, and occ_risk_lookup from disk.

    Returns a tuple suitable for passing as initargs to mp.Pool:
        (worker_df, dist_matrix, occ_risk_dict, params, n_ticks)

    Call this once in the main process before spawning workers.
    """
    ddir = pathlib.Path(data_dir) if data_dir else _ROOT / "data" / "processed"

    worker_df = pd.read_parquet(ddir / "worker_sample_with_risk.parquet")
    print(f"  worker_df          : {len(worker_df):,} rows")

    dist_path = ddir / "skill_distance_matrix.parquet"
    if dist_path.exists():
        dist_matrix = pd.read_parquet(dist_path)
        dist_matrix.index   = dist_matrix.index.astype(int)
        dist_matrix.columns = dist_matrix.columns.astype(int)
        print(f"  skill_distance_matrix : {dist_matrix.shape}")
    else:
        dist_matrix = None
        print("  skill_distance_matrix : NOT FOUND — transitions disabled")

    risk_path = ddir / "occ_risk_lookup.parquet"
    if risk_path.exists():
        risk_df = pd.read_parquet(risk_path)
        risk_df.index = risk_df.index.astype(int)
        occ_risk = {
            "r_job": risk_df["r_job"].to_dict(),
            "p_aug": risk_df["p_aug"].to_dict(),
        }
        print(f"  occ_risk_lookup    : {len(occ_risk['r_job'])} occupations")
    else:
        occ_risk = {"r_job": {}, "p_aug": {}}
        print("  occ_risk_lookup    : NOT FOUND — using empty dicts")

    return worker_df, dist_matrix, occ_risk
