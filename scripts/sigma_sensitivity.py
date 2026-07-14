"""σ (reinstatement rate) sensitivity sweep.

Demonstrates that the headline AI-shock results (terminal UR elevation, HSQ
quintile pain) are not artefacts of the calibrated σ = 0.02. Sweeps σ across
{0.005, 0.01, 0.02, 0.04, 0.08} × N_RUNS paired seeds and writes a long-format
parquet that the manuscript's `fig-sigma-sensitivity` chunk reads.

Defends against the peer-review critique that the "New Economy" job-creation
constant could be set arbitrarily low, mathematically forcing high
unemployment.

Output: output/sigma_sensitivity.parquet — same schema as paired_runs.parquet
plus a leading `sigma` column.
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

DEFAULT_GRID  = (0.005, 0.01, 0.02, 0.04, 0.08)
N_RUNS        = 100
N_TICKS       = 180
BURN_IN       = 60
OUT_PATH      = ROOT / "output" / "sigma_sensitivity.parquet"
N_WORKERS     = min(mp.cpu_count(), 8)
REPORT_EVERY  = 10

_worker_df   = None
_dist_matrix = None
_occ_risk    = None


def _worker_init():
    global _worker_df, _dist_matrix, _occ_risk
    from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()


def run_seed(args: tuple[int, float]) -> pd.DataFrame:
    seed, sigma = args
    rng = np.random.default_rng(seed)
    sampled_df = _worker_df.sample(
        n=len(_worker_df), replace=True,
        random_state=int(rng.integers(0, 2**31)),
    ).reset_index(drop=True)

    params = dict(DEFAULT_PARAMS)
    params["sigma"] = float(sigma)

    rows = []
    for scenario, ai_active in (("AI", True), ("Control", False)):
        m = LaborMarketModel(
            sampled_df,
            params=params,
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
        df["sigma"]    = float(sigma)
        df = df.reset_index()
        df = df[df["tick"] >= BURN_IN].copy()
        df["tick"] = df["tick"] - BURN_IN
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def _parse_grid(arg: str | None) -> tuple[float, ...]:
    if not arg:
        return DEFAULT_GRID
    return tuple(float(x.strip()) for x in arg.split(",") if x.strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--grid", default=None,
                   help=f"comma-separated σ values (default: {','.join(map(str, DEFAULT_GRID))})")
    p.add_argument("--n-runs", type=int, default=N_RUNS)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    grid = _parse_grid(args.grid)
    n_runs = args.n_runs
    out_path = pathlib.Path(args.out)

    print(
        f"[sigma_sensitivity] σ grid: {list(grid)}\n"
        f"  paired seeds per σ: {n_runs}\n"
        f"  ticks per run     : {N_TICKS} (burn-in {BURN_IN})\n"
        f"  workers           : {N_WORKERS} (cpu_count={mp.cpu_count()}, capped at 8)\n"
        f"  total simulations : {len(grid) * n_runs * 2}\n",
        flush=True,
    )

    tasks = [(seed, sigma) for sigma in grid for seed in range(n_runs)]
    total = len(tasks)
    start_time = time.monotonic()
    start_wall = datetime.now()
    print(f"  started at : {start_wall:%Y-%m-%d %H:%M:%S}\n", flush=True)

    all_frames = []
    with mp.Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
        for i, result in enumerate(pool.imap_unordered(run_seed, tasks), start=1):
            all_frames.append(result)
            if i % REPORT_EVERY == 0 or i == total:
                elapsed   = time.monotonic() - start_time
                rate      = i / elapsed
                remaining = (total - i) / rate if rate > 0 else 0
                eta       = datetime.now() + timedelta(seconds=remaining)
                print(
                    f"  [{datetime.now():%H:%M:%S}]  {i:>4}/{total} tasks done"
                    f"  |  elapsed: {elapsed/60:.1f}m"
                    f"  |  ETA: {eta:%H:%M:%S}",
                    flush=True,
                )

    out_df = pd.concat(all_frames, ignore_index=True)
    out_df["unemployment_rate"] = 1.0 - out_df["Employment_Rate"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"\n[sigma_sensitivity] Saved {len(out_df)} rows to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
