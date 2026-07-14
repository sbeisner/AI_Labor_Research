"""HSQ1 (low-exposure quintile) unemployment-cause decomposition.

Decomposes the lowest-hard-skill-quintile (HSQ1_Low) cumulative
unemployment events under the AI scenario into three sources, addressing
the peer-review concern that low-exposure workers paradoxically bear high
displacement pain when their AI substitution risk is low:

  1. Direct AI displacement: HSQ1 workers structurally separated by their
     own employer (Employer.py:_resolve_layoffs).
  2. Credential-blocked from frontier: HSQ1 candidates appeared in
     valid_candidates for a vacancy but failed the credential floor and
     were not hired.
  3. Cascade-bumped: HSQ1 candidates passed the credential floor but
     were out-ranked by higher-scoring (typically retraining-down)
     candidates.

Each event is counted per-worker-per-loss; aggregate event totals across
the analysis window. Output:
  output/q1_decomposition.parquet — long-format with columns
    [scenario, seed, tick, q1_displaced, q1_credential_blocked,
     q1_cascade_bumped]

Runs N_RUNS paired (AI + Control) bootstrap seeds; the manuscript's
`fig-q1-decomposition` chunk uses the AI-minus-Control delta to isolate
AI-induced shares of each mechanism.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import sys
import time
from datetime import datetime, timedelta

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS

N_RUNS       = 30
N_TICKS      = 180
BURN_IN      = 60
OUT_PATH     = ROOT / "output" / "q1_decomposition.parquet"
N_WORKERS    = min(mp.cpu_count(), 8)
REPORT_EVERY = 5

_worker_df   = None
_dist_matrix = None
_occ_risk    = None


def _worker_init():
    global _worker_df, _dist_matrix, _occ_risk
    from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()


KEEP_COLS = [
    "tick", "seed", "scenario",
    "Q1_Displaced",
    "Q1_Credential_Blocked",
    "Q1_Cascade_Bumped",
    "Employment_Rate",
    "Emp_Rate_HSQ1_Low",
]


def run_seed(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_df = _worker_df.sample(
        n=len(_worker_df), replace=True,
        random_state=int(rng.integers(0, 2**31)),
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
        rows.append(df[KEEP_COLS])

    return pd.concat(rows, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--n-runs", type=int, default=N_RUNS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    n_runs = args.n_runs
    out_path = pathlib.Path(args.out)

    print(
        f"[q1_decomposition] paired seeds: {n_runs}\n"
        f"  ticks per run     : {N_TICKS} (burn-in {BURN_IN})\n"
        f"  workers           : {N_WORKERS}\n",
        flush=True,
    )

    start_time = time.monotonic()
    print(f"  started at : {datetime.now():%Y-%m-%d %H:%M:%S}\n", flush=True)

    all_frames = []
    with mp.Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
        for i, result in enumerate(pool.imap_unordered(run_seed, range(n_runs)), start=1):
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

    out_df = pd.concat(all_frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"\n[q1_decomposition] Saved {len(out_df)} rows to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
