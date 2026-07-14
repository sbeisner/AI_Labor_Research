"""Adoption-ceiling sweep (terminal-intensity dose-response).

The velocity dose sweep (`scripts/dose_response.py`) scales the logistic
adoption *growth rate*; but terminal mean adoption A_{j,t} saturates near 1.0
in every velocity arm, so it varies the *path* while pinning terminal treatment
intensity at full adoption. This sweep instead varies the logistic *asymptote*
— the carrying-capacity parameter `a_max` (the adoption ceiling) — holding
velocity fixed, so terminal treatment intensity is varied directly.

`a_max` is the existing carrying-capacity knob in DEFAULT_PARAMS (default 1.0);
it scales the logistic asymptote in Employer.a_jt. No engine edit is required:
this sweep simply sets it per arm.

Sweeps ceiling (a_max) ∈ {0.5, 0.75, 1.0} (1.0 = the headline trajectory)
× N_RUNS paired seeds, frozen engine, same 180-tick / 60-burn-in window as
paired_runs.parquet. Output: output/ceiling_sweep.parquet — schema matches
paired_runs.parquet plus a leading `ceiling` column.

Deliverable metrics (printed after the run, also via --analyze on an existing
parquet):
  * terminal aggregate UR delta (pp) per ceiling, with Monte-Carlo 95% CI;
  * terminal mean A_{j,t} per arm (to confirm the ceiling actually binds);
  * Kendall τ of the HSQ cohort displacement ordering at each ceiling vs the
    headline (ceiling = 1.0) ordering.

CLI
---
    python scripts/ceiling_sweep.py                 # full: grid × 100 seeds, 180 ticks
    python scripts/ceiling_sweep.py --smoke         # 2 seeds, ceiling∈{0.5,1.0}, 30 ticks
    python scripts/ceiling_sweep.py --analyze output/ceiling_sweep.parquet
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

DEFAULT_GRID = (0.5, 0.75, 1.0)
N_RUNS       = 100
N_TICKS      = 180
BURN_IN      = 60
OUT_PATH     = ROOT / "output" / "ceiling_sweep.parquet"
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
    """Run one paired (AI + Control) simulation at the given (seed, ceiling)."""
    seed, ceiling = task
    rng = np.random.default_rng(seed)
    sampled_df = _worker_df.sample(
        n=len(_worker_df), replace=True,
        random_state=int(rng.integers(0, 2**31)),
    ).reset_index(drop=True)

    params = dict(DEFAULT_PARAMS)
    params["a_max"] = float(ceiling)

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
        df["ceiling"]  = float(ceiling)
        df = df.reset_index()
        df = df[df["tick"] >= _burn_in].copy()
        df["tick"] = df["tick"] - _burn_in
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


# ── Deliverable metrics ─────────────────────────────────────────────────────────
def analyze(df: pd.DataFrame) -> dict:
    """Terminal aggregate UR delta (+CI), terminal mean A_jt, HSQ Kendall τ per ceiling."""
    from scipy.stats import kendalltau, t as t_dist

    if "unemployment_rate" not in df.columns:
        df = df.copy()
        df["unemployment_rate"] = 1.0 - df["Employment_Rate"]
    tmax = df["tick"].max()
    f = df[df["tick"] == tmax]

    ceilings = sorted(df["ceiling"].unique())
    agg_delta, agg_ci, mean_ajt, hsq_order = {}, {}, {}, {}
    for c in ceilings:
        fa = f[(f["ceiling"] == c) & (f["scenario"] == "AI")].set_index("seed").sort_index()
        fc = f[(f["ceiling"] == c) & (f["scenario"] == "Control")].set_index("seed").sort_index()
        d = (fa["unemployment_rate"] - fc["unemployment_rate"]) * 100
        agg_delta[c] = float(d.mean())
        n = len(d)
        if n > 1:
            half = float(t_dist.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n))
        else:
            half = float("nan")
        agg_ci[c] = (round(agg_delta[c] - half, 3), round(agg_delta[c] + half, 3))
        # terminal mean adoption in the AI arm — should track the ceiling if it binds
        mean_ajt[c] = float(fa["Avg_A_jt"].mean()) if "Avg_A_jt" in fa.columns else float("nan")
        hsq_delta = [float(((1 - fa[col]) - (1 - fc[col])).mean() * 100) for col in _HSQ_COLS]
        hsq_order[c] = list(np.argsort(np.argsort(-np.array(hsq_delta))))

    ref = 1.0 if 1.0 in hsq_order else ceilings[-1]
    tau = {c: float(kendalltau(hsq_order[c], hsq_order[ref])[0]) for c in ceilings}

    return {
        "ceilings":             ceilings,
        "agg_ur_delta_pp":      {float(c): round(v, 3) for c, v in agg_delta.items()},
        "agg_ur_delta_ci95":    {float(c): agg_ci[c] for c in ceilings},
        "terminal_mean_A_jt":   {float(c): round(v, 4) for c, v in mean_ajt.items()},
        "hsq_ordering_kendall_tau_vs_ref": {float(c): round(v, 4) for c, v in tau.items()},
        "reference_ceiling":    ref,
    }


def _print_metrics(metrics: dict) -> None:
    print("\n── Ceiling-sweep deliverable metrics ──────────────────────────", flush=True)
    print(f"  terminal aggregate UR delta (pp) by ceiling : {metrics['agg_ur_delta_pp']}", flush=True)
    print(f"  95% Monte-Carlo CI by ceiling               : {metrics['agg_ur_delta_ci95']}", flush=True)
    print(f"  terminal mean A_jt by ceiling (binds?)      : {metrics['terminal_mean_A_jt']}", flush=True)
    print(f"  HSQ ordering Kendall τ vs ceiling={metrics['reference_ceiling']}    : "
          f"{metrics['hsq_ordering_kendall_tau_vs_ref']}", flush=True)


def _parse_grid(arg: str | None) -> tuple[float, ...]:
    if not arg:
        return DEFAULT_GRID
    return tuple(float(x.strip()) for x in arg.split(",") if x.strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--grid", default=None,
                   help=f"comma-separated ceiling (a_max) values (default: {','.join(map(str, DEFAULT_GRID))})")
    p.add_argument("--n-runs", type=int, default=N_RUNS)
    p.add_argument("--n-ticks", type=int, default=N_TICKS)
    p.add_argument("--burn-in", type=int, default=BURN_IN)
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--smoke", action="store_true",
                   help="fast schema check: 2 seeds, ceiling∈{0.5,1.0}, 30 ticks (burn-in 10)")
    p.add_argument("--analyze", metavar="PARQUET", default=None,
                   help="skip simulation; compute deliverable metrics on an existing parquet")
    args = p.parse_args()

    if args.analyze:
        _print_metrics(analyze(pd.read_parquet(args.analyze)))
        return 0

    if args.smoke:
        grid = (0.5, 1.0)
        n_runs, n_ticks, burn_in = 2, 30, 10
        out_path = pathlib.Path(args.out).with_name("ceiling_sweep_smoke.parquet")
    else:
        grid = _parse_grid(args.grid)
        n_runs, n_ticks, burn_in = args.n_runs, args.n_ticks, args.burn_in
        out_path = pathlib.Path(args.out)

    total = len(grid) * n_runs
    print(
        f"[ceiling_sweep] adoption-ceiling (a_max) sweep — frozen engine\n"
        f"  ceiling grid           : {list(grid)}  (1.0 = headline trajectory)\n"
        f"  paired seeds per arm   : {n_runs}\n"
        f"  ticks per run          : {n_ticks} (burn-in {burn_in}, {n_ticks - burn_in} analysis)\n"
        f"  workers                : {N_WORKERS} (cpu_count={mp.cpu_count()}, capped at 8)\n"
        f"  paired tasks           : {total}  ({total * 2} single simulations)\n"
        f"  output                 : {out_path}\n",
        flush=True,
    )

    tasks = [(seed, c) for c in grid for seed in range(n_runs)]
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
    out_df = out_df[["ceiling"] + [c for c in out_df.columns if c != "ceiling"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"\n[ceiling_sweep] Saved {len(out_df)} rows to {out_path}", flush=True)
    print(f"  columns: {list(out_df.columns)}", flush=True)
    print(f"  ceiling values present: {sorted(out_df['ceiling'].unique())}", flush=True)

    try:
        _print_metrics(analyze(out_df))
    except Exception as e:  # noqa: BLE001
        print(f"  (metric computation skipped: {e})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
