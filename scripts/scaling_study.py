"""Finite-size scaling study.

The headline results use a 10,000-worker economy. This sweep re-runs the paired
AI-vs-control design at N ∈ {5,000, 20,000} to check that the aggregate effect,
the HSQ cohort ordering, and the frontier fill rate are not artifacts of the
population size. Each seed resamples the worker microdata to the target N with
replacement (the same bootstrap step used in paired_runs).

Sweeps N ∈ {5000, 20000} × N_RUNS paired seeds, current engine, same
180-tick / 60-burn-in window as paired_runs.parquet. Output:
output/scaling_study.parquet — schema matches paired_runs.parquet plus a
leading `N` column.

Deliverable metrics (printed after the run; N=10k reference read from
paired_runs.parquet):
  * terminal aggregate UR delta (pp) per N — scale-free (a rate);
  * Kendall τ of the HSQ displacement ordering vs the N=10k baseline;
  * frontier fill rate = frontier-basket employment Δ / new-economy postings
    (scale-free ratio);
  * per-10k-normalized excess-unemployment headcount.

CLI
---
    python scripts/scaling_study.py             # full: N∈{5000,20000} × 25 seeds, 180 ticks
    python scripts/scaling_study.py --smoke     # N∈{2000,4000} × 2 seeds, 30 ticks
    python scripts/scaling_study.py --analyze output/scaling_study.parquet
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

DEFAULT_GRID = (5000, 20000)
N_RUNS       = 25
N_TICKS      = 180
BURN_IN      = 60
OUT_PATH     = ROOT / "output" / "scaling_study.parquet"
PAIRED_RUNS  = ROOT / "output" / "paired_runs.parquet"
N_WORKERS    = min(mp.cpu_count(), 8)
REPORT_EVERY = 5

_HSQ_COLS = ["Emp_Rate_HSQ1_Low", "Emp_Rate_HSQ2", "Emp_Rate_HSQ3",
             "Emp_Rate_HSQ4", "Emp_Rate_HSQ5_High"]

# ── Worker-process state ────────────────────────────────────────────────────────
_worker_df = _dist_matrix = _occ_risk = None
_n_ticks = N_TICKS
_burn_in = BURN_IN


def _worker_init(n_ticks: int, burn_in: int):
    global _worker_df, _dist_matrix, _occ_risk, _n_ticks, _burn_in
    from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()
    _n_ticks, _burn_in = n_ticks, burn_in


def run_seed(task: tuple[int, int]) -> pd.DataFrame:
    """Run one paired (AI + Control) simulation at the given (seed, N)."""
    seed, n_pop = task
    rng = np.random.default_rng(seed)
    sampled_df = _worker_df.sample(
        n=int(n_pop), replace=True,
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
        for _ in range(_n_ticks):
            m.step()

        df = m.datacollector.get_model_vars_dataframe().copy()
        df.index.name = "tick"
        df["seed"]     = seed
        df["scenario"] = scenario
        df["N"]        = int(n_pop)
        df = df.reset_index()
        df = df[df["tick"] >= _burn_in].copy()
        df["tick"] = df["tick"] - _burn_in
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


# ── Deliverable metrics ─────────────────────────────────────────────────────────
def _hsq_ordering(fa: pd.DataFrame, fc: pd.DataFrame) -> tuple[list, list]:
    """Return (HSQ displacement deltas pp, rank vector) for a terminal-tick slice."""
    deltas = [float(((1 - fa[c]) - (1 - fc[c])).mean() * 100) for c in _HSQ_COLS]
    ranks = list(np.argsort(np.argsort(-np.array(deltas))))
    return deltas, ranks


def analyze(df: pd.DataFrame) -> dict:
    from scipy.stats import kendalltau

    if "unemployment_rate" not in df.columns:
        df = df.copy()
        df["unemployment_rate"] = 1.0 - df["Employment_Rate"]

    # N=10k reference ordering from paired_runs, if available.
    ref_ranks = None
    if PAIRED_RUNS.exists():
        pr = pd.read_parquet(PAIRED_RUNS)
        pf = pr[pr["tick"] == pr["tick"].max()]
        pa = pf[pf["scenario"] == "AI"].set_index("seed").sort_index()
        pc = pf[pf["scenario"] == "Control"].set_index("seed").sort_index()
        _, ref_ranks = _hsq_ordering(pa, pc)

    tmax = df["tick"].max()
    f = df[df["tick"] == tmax]
    out = {"N_values": [int(n) for n in sorted(df["N"].unique())],
           "reference": "paired_runs.parquet (N=10000)" if ref_ranks else "none",
           "per_N": {}}
    for n in sorted(df["N"].unique()):
        fa = f[(f["N"] == n) & (f["scenario"] == "AI")].set_index("seed").sort_index()
        fc = f[(f["N"] == n) & (f["scenario"] == "Control")].set_index("seed").sort_index()
        agg = float(((fa["unemployment_rate"] - fc["unemployment_rate"]) * 100).mean())
        _, ranks = _hsq_ordering(fa, fc)
        tau = float(kendalltau(ranks, ref_ranks)[0]) if ref_ranks else float("nan")
        # frontier fill rate = Δ frontier employment / posted new-economy (scale-free)
        posted = float(fa["New_Economy_Cumulative"].mean())
        filled = float((fa["Frontier_Basket_Employed"] - fc["Frontier_Basket_Employed"]).mean())
        fill_rate = filled / posted if posted > 0 else float("nan")
        # per-10k-normalized excess unemployment headcount
        excess = float((fa["Unemployed_Count"] - fc["Unemployed_Count"]).mean())
        excess_per10k = excess / (n / 10000.0)
        out["per_N"][int(n)] = {
            "agg_ur_delta_pp": round(agg, 3),
            "hsq_ordering_kendall_tau_vs_10k": round(tau, 4),
            "frontier_fill_rate": round(fill_rate, 4),
            "excess_unemp_per_10k": round(excess_per10k, 1),
        }
    return out


def _print_metrics(m: dict) -> None:
    print("\n── Scaling-study deliverable metrics ──────────────────────────", flush=True)
    print(f"  HSQ ordering reference: {m['reference']}", flush=True)
    for n, v in m["per_N"].items():
        print(f"  N={n:>6}: ΔUR={v['agg_ur_delta_pp']:+.2f}pp  "
              f"τ_HSQ={v['hsq_ordering_kendall_tau_vs_10k']}  "
              f"fill={v['frontier_fill_rate']}  "
              f"excess/10k={v['excess_unemp_per_10k']}", flush=True)


def _parse_grid(arg: str | None) -> tuple[int, ...]:
    if not arg:
        return DEFAULT_GRID
    return tuple(int(x.strip()) for x in arg.split(",") if x.strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--grid", default=None,
                   help=f"comma-separated N values (default: {','.join(map(str, DEFAULT_GRID))})")
    p.add_argument("--n-runs", type=int, default=N_RUNS)
    p.add_argument("--n-ticks", type=int, default=N_TICKS)
    p.add_argument("--burn-in", type=int, default=BURN_IN)
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--smoke", action="store_true",
                   help="fast check: N∈{2000,4000} × 2 seeds, 30 ticks (burn-in 10)")
    p.add_argument("--analyze", metavar="PARQUET", default=None,
                   help="skip simulation; compute deliverable metrics on an existing parquet")
    args = p.parse_args()

    if args.analyze:
        _print_metrics(analyze(pd.read_parquet(args.analyze)))
        return 0

    if args.smoke:
        grid = (2000, 4000)
        n_runs, n_ticks, burn_in = 2, 30, 10
        out_path = pathlib.Path(args.out).with_name("scaling_study_smoke.parquet")
    else:
        grid = _parse_grid(args.grid)
        n_runs, n_ticks, burn_in = args.n_runs, args.n_ticks, args.burn_in
        out_path = pathlib.Path(args.out)

    total = len(grid) * n_runs
    print(
        f"[scaling_study] finite-size scaling sweep\n"
        f"  N grid                 : {list(grid)}  (baseline N=10000 from paired_runs)\n"
        f"  paired seeds per N     : {n_runs}\n"
        f"  ticks per run          : {n_ticks} (burn-in {burn_in}, {n_ticks - burn_in} analysis)\n"
        f"  workers                : {N_WORKERS} (cpu_count={mp.cpu_count()}, capped at 8)\n"
        f"  paired tasks           : {total}  ({total * 2} single simulations)\n"
        f"  NOTE: N=20000 runs ~2x slower per tick than N=10000.\n"
        f"  output                 : {out_path}\n",
        flush=True,
    )

    tasks = [(seed, n) for n in grid for seed in range(n_runs)]
    start_time = time.monotonic()
    print(f"  started at : {datetime.now():%Y-%m-%d %H:%M:%S}\n", flush=True)

    all_frames = []
    with mp.Pool(processes=N_WORKERS, initializer=_worker_init,
                 initargs=(n_ticks, burn_in)) as pool:
        for i, result in enumerate(pool.imap_unordered(run_seed, tasks), start=1):
            all_frames.append(result)
            if i % REPORT_EVERY == 0 or i == total:
                elapsed   = time.monotonic() - start_time
                rate      = i / elapsed
                remaining = (total - i) / rate if rate > 0 else 0
                eta_wall  = datetime.now() + timedelta(seconds=remaining)
                print(
                    f"  [{datetime.now():%H:%M:%S}]  {i:>4}/{total} paired tasks"
                    f"  |  elapsed: {elapsed/60:.1f}m  |  ETA: {eta_wall:%H:%M:%S}",
                    flush=True,
                )

    out_df = pd.concat(all_frames, ignore_index=True)
    out_df["unemployment_rate"] = 1.0 - out_df["Employment_Rate"]
    out_df = out_df[["N"] + [c for c in out_df.columns if c != "N"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"\n[scaling_study] Saved {len(out_df)} rows to {out_path}", flush=True)
    print(f"  columns: {list(out_df.columns)}", flush=True)
    print(f"  N values present: {sorted(out_df['N'].unique())}", flush=True)

    try:
        _print_metrics(analyze(out_df))
    except Exception as e:  # noqa: BLE001
        print(f"  (metric computation skipped: {e})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
