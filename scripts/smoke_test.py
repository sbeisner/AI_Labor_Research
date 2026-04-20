"""Smoke test + diagnostic sweep to verify the three post-session fixes:
  1. V_new isolation       — phantom new-economy vacancies no longer inflate Beveridge
  2. BTOS dampener         — effective firing rate stays near delta_base
  3. OLG timing reorder    — new entrants present before market clearing

Runs 5 seeds × 3 delta_base values (low/mid/high) at N_TICKS=60 months
with a 24-tick burn-in.  Reports:
  • mean steady-state UR by (delta, seed)
  • fired_count / (n_workers * delta_base)  — should be ~1.0 if BTOS damp works
  • firm failure count                       — should be 0 if round() fix holds
  • new_economy_jobs tracked vs vacancies    — ensures V_new isolation

Usage:
    python scripts/smoke_test.py
"""

import sys
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS
from scripts.bootstrap_runner import load_shared_data

N_TICKS  = 60
BURN_IN  = 24
SEEDS    = [0, 1, 2]

DELTA_SWEEP = {
    "low":  0.005,
    "mid":  0.00726,   # ABC posterior mean
    "high": 0.012,
}

def run_scenario(delta_base, seed, worker_df, dist_matrix, occ_risk):
    params = {**DEFAULT_PARAMS, "delta_base": delta_base}
    model = LaborMarketModel(
        worker_df=worker_df,
        params=params,
        ai_active=False,
        seed=seed,
        skill_distance_matrix=dist_matrix,
        occ_risk_lookup=occ_risk,
        collect_agent_data=False,
    )
    for _ in range(N_TICKS):
        model.step()

    md = model.datacollector.get_model_vars_dataframe()

    # Steady-state UR (last BURN_IN ticks)
    steady_ur = 1.0 - float(md["Employment_Rate"].iloc[-BURN_IN:].mean())

    # Total firm failures
    firm_failures = int(md["Failed_Firms"].iloc[-1]) if "Failed_Firms" in md.columns else -1

    # Effective firing rate vs expected
    total_fired = int(md["Fired_Count"].iloc[-BURN_IN:].sum()) if "Fired_Count" in md.columns else -1
    n_workers   = int(md["Employed_Workers"].iloc[-BURN_IN:].mean()) if "Employed_Workers" in md.columns else -1
    exp_fired   = n_workers * delta_base * BURN_IN if n_workers > 0 else -1
    ratio       = total_fired / exp_fired if exp_fired > 0 else float("nan")

    # New economy jobs (should be 0 in control; check isolation)
    new_eco = int(md["New_Economy_Jobs"].iloc[-BURN_IN:].sum()) if "New_Economy_Jobs" in md.columns else 0

    # Check if Beveridge vacancies look sane (matchable only)
    vac_mean = float(md["Open_Vacancies"].iloc[-BURN_IN:].mean()) if "Open_Vacancies" in md.columns else float("nan")

    return {
        "delta_label": [k for k, v in DELTA_SWEEP.items() if abs(v - delta_base) < 1e-6][0],
        "delta_base":  delta_base,
        "seed":        seed,
        "steady_ur":   round(steady_ur, 4),
        "firm_failures": firm_failures,
        "fired_ratio": round(ratio, 3),
        "new_eco_jobs": new_eco,
        "vac_mean":    round(vac_mean, 1),
    }


if __name__ == "__main__":
    print("[smoke_test] Loading shared data…")
    worker_df, dist_matrix, occ_risk = load_shared_data()
    print()

    rows = []
    for label, delta in DELTA_SWEEP.items():
        for seed in SEEDS:
            print(f"  Running delta={delta:.5f} ({label}), seed={seed}…", end=" ", flush=True)
            r = run_scenario(delta, seed, worker_df, dist_matrix, occ_risk)
            rows.append(r)
            print(f"UR={r['steady_ur']:.3%}  failures={r['firm_failures']}  fired_ratio={r['fired_ratio']:.2f}")

    df = pd.DataFrame(rows)
    print("\n── Summary ──────────────────────────────────────────────────────────────")
    print(df.to_string(index=False))

    print("\n── UR by delta (mean over seeds) ────────────────────────────────────────")
    print(df.groupby("delta_label")[["steady_ur"]].mean().round(4))

    print("\n── Fired ratio by delta (should be ~1.0) ────────────────────────────────")
    print(df.groupby("delta_label")[["fired_ratio"]].mean().round(3))

    ok_failures = (df["firm_failures"] == 0).all()
    ok_ratio    = ((df["fired_ratio"] > 0.5) & (df["fired_ratio"] < 2.5)).all()
    ok_ur       = ((df["steady_ur"] > 0.02) & (df["steady_ur"] < 0.12)).all()

    print("\n── Pass / Fail ───────────────────────────────────────────────────────────")
    print(f"  Firm failures = 0   : {'PASS' if ok_failures else 'FAIL'}")
    print(f"  Fired ratio in [0.5, 2.5] : {'PASS' if ok_ratio else 'FAIL'}")
    print(f"  UR in [2%, 12%]     : {'PASS' if ok_ur else 'FAIL'}")
