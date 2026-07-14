"""Joint calibration of ω (retraining time scalar) and μ (gravity friction) against
empirical retraining-duration and transition-distance anchors.

ω target: median retraining duration ≈ 6 ticks (months). Anchor: BLS Trade
    Adjustment Assistance Annual Reports — median training duration for
    displaced workers is ~6-9 months for non-credential skill switches.

μ target: mean cosine distance of chosen retraining transitions ≈ 0.30.
    Anchor: Macaluso (2017) "Skill Remoteness and Post-Layoff Labor Market
    Outcomes" — mean O*NET skill distance for actual displaced-worker
    transitions is ~0.30 in normalised cosine units. Schubert, Stansbury &
    Taska (2020) corroborate at the SOC-pair transition-rate level.

Approach: 2D grid search over (ω, μ). For each cell, run 3 seeds × 60 ticks
Control scenario; capture median T_retrain across all retraining starts and
mean d(s_i, s_j) for chosen target occupations. Score cells by joint L2
distance to (T_target, d_target) after standardising both metrics. Print the
best cell and a sorted shortlist.

Usage:
    python scripts/calibrate_omega_mu.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import multiprocessing as mp
from itertools import product
import numpy as np
import pandas as pd

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS
from agents.Worker import WorkerAgent

# ── Empirical targets ──────────────────────────────────────────────────────
T_TARGET = 6.0   # BLS TAA: median retraining duration ≈ 6-9 months
D_TARGET = 0.30  # Macaluso (2017): mean transition distance ≈ 0.30

# ── Grid ───────────────────────────────────────────────────────────────────
OMEGA_GRID = [6.0, 9.0, 12.0, 18.0, 24.0]
MU_GRID    = [0.10, 0.25, 0.50, 1.00, 2.00]

N_SEEDS  = 3
N_TICKS  = 60
BURN_IN  = 12   # discard transient warm-up before measuring

# ── Worker init for multiprocessing ────────────────────────────────────────
_worker_df   = None
_dist_matrix = None
_occ_risk    = None


def _worker_init():
    global _worker_df, _dist_matrix, _occ_risk
    from scripts.bootstrap_runner import load_shared_data
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()


def _run_cell(args):
    omega, mu = args
    params = {**DEFAULT_PARAMS, "omega": omega, "mu": mu}

    T_values = []
    d_values = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed * 7919 + int(omega * 100) + int(mu * 1000))
        sampled = _worker_df.sample(
            n=len(_worker_df), replace=True,
            random_state=int(rng.integers(0, 2**31)),
        ).reset_index(drop=True)

        m = LaborMarketModel(
            sampled, params=params, ai_active=False, seed=seed,
            skill_distance_matrix=_dist_matrix, occ_risk_lookup=_occ_risk,
            collect_agent_data=False,
        )
        # Track every retraining episode that begins after burn-in
        seen = set()
        for t in range(N_TICKS):
            m.step()
            if t < BURN_IN:
                continue
            for a in m.agents_by_type[WorkerAgent]:
                key = id(a)
                if a.retraining_total > 0 and key not in seen:
                    seen.add(key)
                    T_values.append(int(a.retraining_total))
                    d_values.append(float(a.retrain_d))
                elif a.retraining_total == 0 and key in seen:
                    seen.discard(key)

    T_med  = float(np.median(T_values)) if T_values else float("nan")
    T_mean = float(np.mean(T_values))   if T_values else float("nan")
    d_mean = float(np.mean(d_values))   if d_values else float("nan")
    n_paths = len(T_values)
    return {
        "omega":   omega,
        "mu":      mu,
        "T_med":   T_med,
        "T_mean":  T_mean,
        "d_mean":  d_mean,
        "n_paths": n_paths,
    }


def _score(row):
    """Joint L2 distance from (T_TARGET, D_TARGET) on standardised axes."""
    if pd.isna(row["T_med"]) or pd.isna(row["d_mean"]):
        return float("inf")
    # Standardise each metric by its target so they're comparable
    t_err = (row["T_med"]  - T_TARGET) / T_TARGET
    d_err = (row["d_mean"] - D_TARGET) / D_TARGET
    return float(np.sqrt(t_err**2 + d_err**2))


if __name__ == "__main__":
    cells = list(product(OMEGA_GRID, MU_GRID))
    print(f"[calibrate_omega_mu] grid: {len(OMEGA_GRID)} ω × {len(MU_GRID)} μ "
          f"= {len(cells)} cells, {N_SEEDS} seeds × {N_TICKS} ticks each",
          flush=True)
    print(f"  targets: T_med={T_TARGET}  d_mean={D_TARGET}", flush=True)

    n_workers = min(mp.cpu_count(), len(cells))
    with mp.Pool(processes=n_workers, initializer=_worker_init) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_run_cell, cells), start=1):
            rows.append(r)
            print(f"  [{i}/{len(cells)}] ω={r['omega']:>5.1f} μ={r['mu']:>4.2f} → "
                  f"T_med={r['T_med']:>4.1f} d_mean={r['d_mean']:.3f} "
                  f"(n={r['n_paths']:,})",
                  flush=True)

    df = pd.DataFrame(rows).sort_values(["omega", "mu"]).reset_index(drop=True)
    df["score"] = df.apply(_score, axis=1)

    print()
    print("─── Full grid (sorted by score) ───────────────────────────────────────")
    print(df.sort_values("score").to_string(index=False), flush=True)

    best = df.loc[df["score"].idxmin()]
    print()
    print("─── Best cell ─────────────────────────────────────────────────────────")
    print(f"  ω* = {best['omega']:.2f}   μ* = {best['mu']:.3f}", flush=True)
    print(f"  simulated median T = {best['T_med']:.1f}   target = {T_TARGET}", flush=True)
    print(f"  simulated mean d   = {best['d_mean']:.3f}  target = {D_TARGET}", flush=True)
    print(f"  L2 score           = {best['score']:.4f}", flush=True)
