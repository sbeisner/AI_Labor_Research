"""Adoption dose-response sweep.

The headline displacement magnitudes are conditional on the AI adoption
trajectory A_{j,t}. This sweep scales the logistic adoption-growth rate by a
global `adoption_velocity_mult` (the "dose"), holding everything else fixed, to
map how the aggregate treatment effect and the cohort ordering respond to
faster or slower adoption.

Sweeps dose ∈ {0.5, 0.75, 1.0, 1.5} (1.0 = the headline paired_runs trajectory)
× N_RUNS paired seeds, current engine, same 180-tick / 60-burn-in window as
paired_runs.parquet. Output: output/dose_response.parquet — schema matches
paired_runs.parquet plus a leading `dose` column.

Deliverable metrics (printed after the run, also via --analyze on an existing
parquet):
  * elasticity of the terminal aggregate UR delta w.r.t. dose (log-log slope);
  * Kendall τ of the HSQ cohort displacement ordering at each dose vs the
    headline (dose = 1.0) ordering.

CLI
---
    python scripts/dose_response.py                 # full: grid × 100 seeds, 180 ticks
    python scripts/dose_response.py --smoke         # 2 seeds, dose∈{0.5,1.5}, 30 ticks
    python scripts/dose_response.py --analyze output/dose_response.parquet
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

DEFAULT_GRID = (0.5, 0.75, 1.0, 1.5)
N_RUNS       = 100
N_TICKS      = 180
BURN_IN      = 60
OUT_PATH     = ROOT / "output" / "dose_response.parquet"
N_WORKERS    = min(mp.cpu_count(), 8)
REPORT_EVERY = 10

_HSQ_COLS = ["Emp_Rate_HSQ1_Low", "Emp_Rate_HSQ2", "Emp_Rate_HSQ3",
             "Emp_Rate_HSQ4", "Emp_Rate_HSQ5_High"]

# ── Worker-process state ────────────────────────────────────────────────────────
_worker_df   = None
_dist_matrix = None
_occ_risk    = None
_n_ticks     = N_TICKS
_burn_in     = BURN_IN


def _worker_init(n_ticks: int, burn_in: int):
    global _worker_df, _dist_matrix, _occ_risk, _n_ticks, _burn_in
    from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()
    _n_ticks = n_ticks
    _burn_in = burn_in


def run_seed(task: tuple[int, float]) -> pd.DataFrame:
    """Run one paired (AI + Control) simulation at the given (seed, dose)."""
    seed, dose = task
    rng = np.random.default_rng(seed)
    sampled_df = _worker_df.sample(
        n=len(_worker_df), replace=True,
        random_state=int(rng.integers(0, 2**31)),
    ).reset_index(drop=True)

    params = dict(DEFAULT_PARAMS)
    params["adoption_velocity_mult"] = float(dose)

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
        for _ in range(_n_ticks):
            m.step()

        df = m.datacollector.get_model_vars_dataframe().copy()
        df.index.name = "tick"
        df["seed"]     = seed
        df["scenario"] = scenario
        df["dose"]     = float(dose)
        df = df.reset_index()
        df = df[df["tick"] >= _burn_in].copy()
        df["tick"] = df["tick"] - _burn_in
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


# ── Deliverable metrics ─────────────────────────────────────────────────────────
def analyze(df: pd.DataFrame) -> dict:
    """Elasticity of terminal aggregate UR delta to dose + Kendall τ of HSQ order."""
    from scipy.stats import kendalltau

    if "unemployment_rate" not in df.columns:
        df = df.copy()
        df["unemployment_rate"] = 1.0 - df["Employment_Rate"]
    tmax = df["tick"].max()
    f = df[df["tick"] == tmax]

    doses = sorted(df["dose"].unique())
    agg_delta, hsq_order = {}, {}
    for d in doses:
        fa = f[(f["dose"] == d) & (f["scenario"] == "AI")].set_index("seed").sort_index()
        fc = f[(f["dose"] == d) & (f["scenario"] == "Control")].set_index("seed").sort_index()
        agg_delta[d] = float(((fa["unemployment_rate"] - fc["unemployment_rate"]) * 100).mean())
        hsq_delta = [float(((1 - fa[c]) - (1 - fc[c])).mean() * 100) for c in _HSQ_COLS]
        # rank HSQ cohorts by displacement (descending); store the ordering
        hsq_order[d] = list(np.argsort(np.argsort(-np.array(hsq_delta))))

    # Elasticity: log-log OLS slope of aggregate UR delta on dose (drop non-positive).
    dd = np.array([d for d in doses if agg_delta[d] > 0])
    yy = np.array([agg_delta[d] for d in doses if agg_delta[d] > 0])
    elasticity = float(np.polyfit(np.log(dd), np.log(yy), 1)[0]) if len(dd) >= 2 else float("nan")

    ref = 1.0 if 1.0 in hsq_order else doses[len(doses) // 2]
    tau = {d: float(kendalltau(hsq_order[d], hsq_order[ref])[0]) for d in doses}

    return {
        "doses":               doses,
        "agg_ur_delta_pp":     {float(d): round(v, 3) for d, v in agg_delta.items()},
        "dose_elasticity":     round(elasticity, 4),
        "hsq_ordering_kendall_tau_vs_ref": {float(d): round(v, 4) for d, v in tau.items()},
        "reference_dose":      ref,
    }


def _print_metrics(metrics: dict) -> None:
    print("\n── Dose-response deliverable metrics ──────────────────────────", flush=True)
    print(f"  terminal aggregate UR delta (pp) by dose : {metrics['agg_ur_delta_pp']}", flush=True)
    print(f"  dose elasticity (d ln ΔUR / d ln dose)   : {metrics['dose_elasticity']}", flush=True)
    print(f"  HSQ ordering Kendall τ vs dose={metrics['reference_dose']}     : "
          f"{metrics['hsq_ordering_kendall_tau_vs_ref']}", flush=True)


def _parse_grid(arg: str | None) -> tuple[float, ...]:
    if not arg:
        return DEFAULT_GRID
    return tuple(float(x.strip()) for x in arg.split(",") if x.strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--grid", default=None,
                   help=f"comma-separated dose values (default: {','.join(map(str, DEFAULT_GRID))})")
    p.add_argument("--n-runs", type=int, default=N_RUNS)
    p.add_argument("--n-ticks", type=int, default=N_TICKS)
    p.add_argument("--burn-in", type=int, default=BURN_IN)
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--smoke", action="store_true",
                   help="fast schema check: 2 seeds, dose∈{0.5,1.5}, 30 ticks (burn-in 10)")
    p.add_argument("--analyze", metavar="PARQUET", default=None,
                   help="skip simulation; compute deliverable metrics on an existing parquet")
    args = p.parse_args()

    if args.analyze:
        _print_metrics(analyze(pd.read_parquet(args.analyze)))
        return 0

    if args.smoke:
        grid = (0.5, 1.5)
        n_runs, n_ticks, burn_in = 2, 30, 10
        out_path = pathlib.Path(args.out).with_name("dose_response_smoke.parquet")
    else:
        grid = _parse_grid(args.grid)
        n_runs, n_ticks, burn_in = args.n_runs, args.n_ticks, args.burn_in
        out_path = pathlib.Path(args.out)

    total = len(grid) * n_runs
    print(
        f"[dose_response] adoption-velocity dose sweep\n"
        f"  dose grid              : {list(grid)}  (1.0 = headline trajectory)\n"
        f"  paired seeds per dose  : {n_runs}\n"
        f"  ticks per run          : {n_ticks} (burn-in {burn_in}, {n_ticks - burn_in} analysis)\n"
        f"  workers                : {N_WORKERS} (cpu_count={mp.cpu_count()}, capped at 8)\n"
        f"  paired tasks           : {total}  ({total * 2} single simulations)\n"
        f"  output                 : {out_path}\n",
        flush=True,
    )

    tasks = [(seed, dose) for dose in grid for seed in range(n_runs)]
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
    out_df = out_df[["dose"] + [c for c in out_df.columns if c != "dose"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"\n[dose_response] Saved {len(out_df)} rows to {out_path}", flush=True)
    print(f"  columns: {list(out_df.columns)}", flush=True)
    print(f"  dose values present: {sorted(out_df['dose'].unique())}", flush=True)

    try:
        _print_metrics(analyze(out_df))
    except Exception as e:  # noqa: BLE001
        print(f"  (metric computation skipped: {e})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
