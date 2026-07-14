"""Audit-2 sensitivity: does the _update_effective_vacancies fix change RQ6?

Background
----------
The previous _update_effective_vacancies built the radiation model's V_j as
    eff[occ] = (incumbent count in occ) + (open vacancies in occ)
biasing unemployed-worker pull toward large legacy occupations regardless of
actual labor demand.  The audit alleged this artifact was the source of the
"new economy fails to offset displacement" finding (RQ6).

Design
------
For each seed we run FOUR simulations:
    (legacy, AI), (legacy, Control), (fix, AI), (fix, Control)
where 'legacy' sets eff_vac_legacy_sum=True (the bug) and 'fix' sets it False.
Pairing AI vs Control at the same seed cancels macro noise; pairing legacy
vs fix at the same seed cancels micro noise on the structural change.

Treatment effect under each regime:
    Δ_legacy[t] = legacy_AI[t] - legacy_Control[t]
    Δ_fix[t]    = fix_AI[t]    - fix_Control[t]
    shift[t]    = Δ_fix[t]     - Δ_legacy[t]
A non-trivial shift on RQ6 metrics (Frontier_Basket_Employed,
New_Economy_Cumulative) means the bug materially distorted the conclusion.

CLI
---
    python scripts/eff_vac_sensitivity.py             # full N=30, 180 ticks
    python scripts/eff_vac_sensitivity.py --smoke     # N=4, 90 ticks (fast)
    python scripts/eff_vac_sensitivity.py --seeds 50  # custom N
"""
from __future__ import annotations
import sys
import argparse
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

OUT_PATH = ROOT / "output" / "eff_vac_sensitivity.parquet"
N_WORKERS = min(mp.cpu_count(), 8)

# Worker-process state populated by _worker_init.
_worker_df = None
_dist_matrix = None
_occ_risk = None
_n_ticks = None
_burn_in = None


def _worker_init(n_ticks: int, burn_in: int):
    global _worker_df, _dist_matrix, _occ_risk, _n_ticks, _burn_in
    from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()
    _n_ticks = n_ticks
    _burn_in = burn_in


def _run_one(sampled_df, seed, ai_active, legacy_sum):
    params = dict(DEFAULT_PARAMS)
    params["eff_vac_legacy_sum"] = legacy_sum
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
    df = df.reset_index()
    df = df[df["tick"] >= _burn_in].copy()
    df["tick"] = df["tick"] - _burn_in
    df["seed"] = seed
    df["scenario"] = "AI" if ai_active else "Control"
    df["regime"] = "legacy" if legacy_sum else "fix"
    return df


def run_seed(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_df = _worker_df.sample(
        n=len(_worker_df), replace=True, random_state=int(rng.integers(0, 2**31))
    ).reset_index(drop=True)

    rows = []
    for legacy_sum in (True, False):
        for ai_active in (True, False):
            rows.append(_run_one(sampled_df, seed, ai_active, legacy_sum))
    return pd.concat(rows, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tick paired treatment effect and regime shift on RQ6 metrics."""
    metrics = [
        "Employment_Rate",
        "unemployment_rate",
        "New_Economy_Cumulative",
        "New_Economy_Jobs",
        "Frontier_Basket_Employed",
        "Total_Vacancies",
        "Retrained_Share",
        "Emp_Rate_HSQ1_Low",
    ]
    if "unemployment_rate" not in df.columns:
        df = df.copy()
        df["unemployment_rate"] = 1.0 - df["Employment_Rate"]

    # Wide on (regime, scenario) per (seed, tick), then compute deltas.
    keep = ["seed", "tick", "regime", "scenario"] + [
        m for m in metrics if m in df.columns
    ]
    long = df[keep]
    wide = long.pivot_table(
        index=["seed", "tick"],
        columns=["regime", "scenario"],
        values=[m for m in metrics if m in df.columns],
    )

    summary_rows = []
    for metric in [m for m in metrics if m in df.columns]:
        legacy_ai   = wide[(metric, "legacy", "AI")]
        legacy_ctl  = wide[(metric, "legacy", "Control")]
        fix_ai      = wide[(metric, "fix",    "AI")]
        fix_ctl     = wide[(metric, "fix",    "Control")]
        d_legacy = legacy_ai - legacy_ctl
        d_fix    = fix_ai    - fix_ctl
        shift    = d_fix - d_legacy

        # Average over the analysis window per seed, then summarise across seeds.
        per_seed = pd.DataFrame({
            "d_legacy": d_legacy.groupby(level="seed").mean(),
            "d_fix":    d_fix.groupby(level="seed").mean(),
            "shift":    shift.groupby(level="seed").mean(),
        })

        summary_rows.append({
            "metric": metric,
            "n_seeds": len(per_seed),
            "Δ_legacy_mean": per_seed["d_legacy"].mean(),
            "Δ_fix_mean":    per_seed["d_fix"].mean(),
            "shift_mean":    per_seed["shift"].mean(),
            "shift_se":      per_seed["shift"].std(ddof=1) / np.sqrt(len(per_seed)),
            "shift_t":       (per_seed["shift"].mean()
                              / (per_seed["shift"].std(ddof=1)
                                 / np.sqrt(len(per_seed))))
                             if per_seed["shift"].std(ddof=1) > 0 else np.nan,
            "shift_p2":      _two_sided_p(per_seed["shift"]),
        })
    return pd.DataFrame(summary_rows)


def _two_sided_p(series: pd.Series) -> float:
    """Paired t-test p-value via a Student-t cdf (no scipy dependency)."""
    n = len(series)
    if n < 2:
        return float("nan")
    mean = series.mean()
    sd   = series.std(ddof=1)
    if sd <= 0:
        return float("nan")
    t = mean / (sd / np.sqrt(n))
    # Approximate two-sided p via a normal cdf (n>=30 reasonable; for smoke runs
    # the t-vs-normal gap is small relative to the question we're asking).
    from math import erf, sqrt
    z = abs(t)
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    return float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30, help="paired seeds (default 30)")
    ap.add_argument("--ticks", type=int, default=180, help="ticks per run incl burn-in")
    ap.add_argument("--burn-in", type=int, default=60, help="burn-in ticks discarded")
    ap.add_argument("--smoke", action="store_true",
                    help="N=4 seeds, ticks=90, burn=30 — validates the script fast")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    if args.smoke:
        n_seeds, n_ticks, burn_in = 4, 90, 30
    else:
        n_seeds, n_ticks, burn_in = args.seeds, args.ticks, args.burn_in

    out_path = pathlib.Path(args.out)

    print(
        f"[eff_vac_sensitivity] Running {n_seeds} paired seeds × {n_ticks} ticks × "
        f"4 sims/seed (legacy×{{AI,Control}} + fix×{{AI,Control}})\n"
        f"  workers : {N_WORKERS} (cpu_count={mp.cpu_count()}, capped at 8)",
        flush=True,
    )

    start_time = time.monotonic()
    start_wall = datetime.now()
    print(f"  started at : {start_wall:%Y-%m-%d %H:%M:%S}\n", flush=True)

    all_frames = []
    with mp.Pool(processes=N_WORKERS,
                 initializer=_worker_init,
                 initargs=(n_ticks, burn_in)) as pool:
        for i, result in enumerate(
            pool.imap_unordered(run_seed, range(n_seeds)),
            start=1,
        ):
            all_frames.append(result)
            if i % max(1, n_seeds // 10) == 0 or i == n_seeds:
                elapsed = time.monotonic() - start_time
                rate = i / elapsed
                remaining = (n_seeds - i) / rate if rate > 0 else 0
                eta = datetime.now() + timedelta(seconds=remaining)
                print(
                    f"  [{datetime.now():%H:%M:%S}]  {i:>3}/{n_seeds} seeds done"
                    f"  |  elapsed: {elapsed/60:.1f}m"
                    f"  |  ETA: {eta:%H:%M:%S}",
                    flush=True,
                )

    df = pd.concat(all_frames, ignore_index=True)
    df["unemployment_rate"] = 1.0 - df["Employment_Rate"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[eff_vac_sensitivity] Saved {len(df):,} rows -> {out_path.name}", flush=True)

    print("\n=== Paired AI − Control treatment effect, by regime ===")
    summary = summarise(df)
    with pd.option_context("display.float_format", lambda x: f"{x: .4f}",
                           "display.max_columns", None,
                           "display.width", 160):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
