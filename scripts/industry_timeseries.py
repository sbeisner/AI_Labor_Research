"""Per-tick industry capacity timeseries for RQ7 tipping-point analysis.

Runs N_SEEDS paired (AI + Control) seeds and records, at every analysis tick,
the aggregate workforce capacity and open vacancies by NAICS 2-digit sector.
Saves to output/industry_timeseries.parquet.

This feeds the RQ7 hypothesis test: is there an AI adoption tipping point
where labor demand for a specific industry collapses non-linearly (>20% drop
in capacity in a single tick)?
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import multiprocessing as mp
import numpy as np
import pandas as pd
from collections import defaultdict

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS

N_SEEDS  = 10
N_TICKS  = 180
BURN_IN  = 60
OUT_PATH = ROOT / "output" / "industry_timeseries.parquet"

NAICS_NAMES = {
    "11": "Agriculture / Forestry",
    "21": "Mining & Oil/Gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",   "45": "Retail Trade",
    "48": "Transportation", "49": "Transportation",
    "51": "Information / Tech",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional Services",
    "55": "Management",
    "56": "Administrative Svcs",
    "61": "Education",
    "62": "Health Care",
    "71": "Arts & Entertainment",
    "72": "Accommodation / Food",
    "81": "Other Services",
    "92": "Public Administration",
}

_worker_df   = None
_dist_matrix = None
_occ_risk    = None


def _worker_init():
    global _worker_df, _dist_matrix, _occ_risk
    from scripts.bootstrap_runner import load_shared_data
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()


def _snapshot_sector(model):
    """Aggregate capacity and vacancies by NAICS 2-digit sector."""
    from agents.Employer import EmployerAgent
    cap_by_sector = defaultdict(int)
    vac_by_sector = defaultdict(int)
    for emp in model.agents_by_type[EmployerAgent]:
        naics = emp.sector[:2]
        cap_by_sector[naics] += emp.capacity
        vac_by_sector[naics] += emp.vacancies
    rows = [
        {"naics": k, "capacity": cap_by_sector[k], "vacancies": vac_by_sector[k]}
        for k in cap_by_sector
    ]
    return pd.DataFrame(rows)


def run_seed(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_df = _worker_df.sample(
        n=len(_worker_df), replace=True,
        random_state=int(rng.integers(0, 2**31)),
    ).reset_index(drop=True)

    frames = []
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
        for t in range(N_TICKS):
            m.step()
            if t >= BURN_IN:
                snap = _snapshot_sector(m)
                snap["tick"]     = t - BURN_IN
                snap["scenario"] = scenario
                snap["seed"]     = seed
                frames.append(snap)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print(f"[industry_timeseries] Running {N_SEEDS} seeds × 2 scenarios × {N_TICKS - BURN_IN} ticks …", flush=True)
    n_workers = min(mp.cpu_count(), 8)
    with mp.Pool(processes=n_workers, initializer=_worker_init) as pool:
        all_frames = []
        for i, df in enumerate(pool.imap_unordered(run_seed, range(N_SEEDS)), start=1):
            all_frames.append(df)
            print(f"  {i}/{N_SEEDS} seeds done", flush=True)

    result = pd.concat(all_frames, ignore_index=True)
    result["industry"] = result["naics"].map(NAICS_NAMES).fillna("Other")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False)
    print(f"[industry_timeseries] Saved {len(result):,} rows to {OUT_PATH}", flush=True)
