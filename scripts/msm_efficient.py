"""MSM inference upgrade: two-step efficient GMM, bootstrap SEs, and J-test.

The incumbent MSM point estimate (output/msm_posterior.csv) was obtained under
an identity (or diagonal) weighting matrix. This script performs the standard
second-step upgrade:

  1. TWO-STEP GMM. At the incumbent point estimate θ̂₁, estimate the full moment
     covariance S from R ≥ 50 independent Control replications of the relative
     moment conditions g(θ) = (m_sim(θ) − m_emp) / m_emp. Re-optimize the GMM
     objective under the efficient weighting W = S⁻¹ to obtain θ̂₂. Both vectors
     are reported.

  2. STANDARD ERRORS. Asymptotic SEs from the GMM sandwich
     Var(θ̂₂) = (1/K)·(G′ W G)⁻¹, where G = ∂g/∂θ is the moment Jacobian
     (central finite differences) and K is the inner-loop seed count that sets
     the sampling variance of the simulated moment estimate. Parametric-
     bootstrap SEs are obtained by drawing B moment vectors g_b ~ N(ĝ, S/K) and
     propagating them through the linearized estimator
     θ_b = θ̂₂ − (G′WG)⁻¹G′W (g_b − ĝ); their SD is the bootstrap SE.

  3. J-TEST. With 4 moments and 3 parameters there is a single overidentifying
     restriction. J = K · ĝ′ W ĝ ~ χ²(1) under correct specification; the p-value
     is reported.

Reuses the moment machinery and empirical targets from scripts/msm_calibration.py.
Output → output/msm_efficient.csv (small, consumed by §2.3).

CLI
---
    python scripts/msm_efficient.py            # full: R=50, K=4, B=2000, 180 ticks
    python scripts/msm_efficient.py --smoke    # R=8, K=2, B=200, 40 ticks (fast check)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS  # noqa: E402
# Pure helpers (no worker-process globals) reused from the base calibration.
from scripts.msm_calibration import (  # noqa: E402
    MOMENT_KEYS, THETA_INIT, THETA_BOUNDS,
    _moments_from_ur, _theta_from_log, load_empirical_targets,
)

OUT_PATH = ROOT / "output" / "msm_efficient.csv"
INCUMBENT_CSV = ROOT / "output" / "msm_posterior.csv"
N_WORKERS = min(mp.cpu_count(), 8)

# ── Worker-process state (own init so ticks are configurable for --smoke) ───────
_worker_df = _dist_matrix = _occ_risk = None
_n_ticks = 180
_burn_in = 60


def _worker_init(n_ticks: int, burn_in: int):
    global _worker_df, _dist_matrix, _occ_risk, _n_ticks, _burn_in
    from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()
    _n_ticks, _burn_in = n_ticks, burn_in


def _simulate_moments_one(args: tuple[dict, int]) -> dict:
    """Run one Control simulation at params/seed; return its moment vector."""
    params, seed = args
    m = LaborMarketModel(
        worker_df=_worker_df, params=params, ai_active=False, seed=seed,
        skill_distance_matrix=_dist_matrix, occ_risk_lookup=_occ_risk,
        collect_agent_data=False,
    )
    for _ in range(_n_ticks):
        m.step()
    md = m.datacollector.get_model_vars_dataframe()
    ur = (1.0 - md["Employment_Rate"]).iloc[_burn_in:].reset_index(drop=True)
    return _moments_from_ur(ur)


def _rel_conditions(sim: dict, targets: dict) -> np.ndarray:
    """Relative moment conditions g = (m_sim − m_emp) / m_emp, ordered by MOMENT_KEYS."""
    return np.array([(sim[k] - targets[k]) / max(abs(targets[k]), 1e-9) for k in MOMENT_KEYS])


def _gbar(theta: dict, pool, targets: dict, k_inner: int, seed0: int = 0) -> np.ndarray:
    """K-seed-averaged relative moment conditions at a parameter vector."""
    params = {**DEFAULT_PARAMS, **theta}
    sims = pool.map(_simulate_moments_one, [(params, seed0 + s) for s in range(k_inner)])
    sim_mean = {k: float(np.mean([m[k] for m in sims])) for k in MOMENT_KEYS}
    return _rel_conditions(sim_mean, targets)


def _load_incumbent() -> dict:
    if INCUMBENT_CSV.exists():
        row = pd.read_csv(INCUMBENT_CSV).iloc[0]
        keys = list(THETA_INIT.keys())
        if all(k in row for k in keys):
            return {k: float(row[k]) for k in keys}
    print(f"  (incumbent CSV missing/incomplete; falling back to THETA_INIT)", flush=True)
    return dict(THETA_INIT)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--reps", type=int, default=50, help="replications for moment covariance S (>=50)")
    p.add_argument("--k-inner", type=int, default=4, help="inner-loop seeds per objective eval")
    p.add_argument("--n-boot", type=int, default=2000, help="parametric-bootstrap draws")
    p.add_argument("--max-iter", type=int, default=60, help="Nelder-Mead max iterations")
    p.add_argument("--n-ticks", type=int, default=180)
    p.add_argument("--burn-in", type=int, default=60)
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--smoke", action="store_true",
                   help="fast check: R=8, K=2, B=200, 40 ticks, 15 iters")
    args = p.parse_args()

    if args.smoke:
        reps, k_inner, n_boot, max_iter, n_ticks, burn_in = 8, 2, 200, 15, 40, 10
        out_path = pathlib.Path(args.out).with_name("msm_efficient_smoke.csv")
    else:
        reps, k_inner, n_boot, max_iter = args.reps, args.k_inner, args.n_boot, args.max_iter
        n_ticks, burn_in = args.n_ticks, args.burn_in
        out_path = pathlib.Path(args.out)

    keys = list(THETA_INIT.keys())
    targets = load_empirical_targets()
    theta1 = _load_incumbent()

    print(
        f"[msm_efficient] two-step efficient GMM + bootstrap SEs + J-test\n"
        f"  incumbent θ̂₁          : {theta1}\n"
        f"  empirical targets     : { {k: round(v,5) for k,v in targets.items()} }\n"
        f"  reps for S            : {reps}\n"
        f"  inner seeds (K)       : {k_inner}\n"
        f"  bootstrap draws       : {n_boot}\n"
        f"  ticks/burn-in         : {n_ticks}/{burn_in}\n"
        f"  workers               : {N_WORKERS}\n",
        flush=True,
    )
    t0 = time.monotonic()

    with mp.Pool(processes=N_WORKERS, initializer=_worker_init,
                 initargs=(n_ticks, burn_in)) as pool:
        # ── Step 1: moment covariance S at the incumbent estimate ──────────────
        print(f"  [{datetime.now():%H:%M:%S}] estimating S from {reps} reps at θ̂₁ ...", flush=True)
        params1 = {**DEFAULT_PARAMS, **theta1}
        rep_sims = pool.map(_simulate_moments_one, [(params1, s) for s in range(reps)])
        G_rel = np.array([_rel_conditions(m, targets) for m in rep_sims])  # reps × 4
        S = np.cov(G_rel, rowvar=False)                                    # 4 × 4
        # Regularize for invertibility on small samples.
        S += np.eye(len(MOMENT_KEYS)) * 1e-10
        W = np.linalg.inv(S)
        print(f"      S diag (Var of rel. conditions): "
              f"{np.round(np.diag(S), 6).tolist()}", flush=True)

        # ── Step 2: re-optimize under W = S⁻¹ ──────────────────────────────────
        def objective(log_theta):
            theta = _theta_from_log(log_theta)
            g = _gbar(theta, pool, targets, k_inner)
            return float(g @ W @ g)

        x0 = np.array([np.log(theta1[k]) for k in keys])
        print(f"  [{datetime.now():%H:%M:%S}] re-optimizing under efficient W ...", flush=True)
        res = minimize(objective, x0=x0, method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-5, "disp": True})
        theta2 = _theta_from_log(res.x)

        # ── Moment Jacobian G = ∂g/∂θ (central differences, raw units) ─────────
        print(f"  [{datetime.now():%H:%M:%S}] computing moment Jacobian ...", flush=True)
        g_hat = _gbar(theta2, pool, targets, k_inner)
        Jg = np.zeros((len(MOMENT_KEYS), len(keys)))
        for j, k in enumerate(keys):
            h = 0.05 * abs(theta2[k])  # 5% relative step
            tp = dict(theta2); tp[k] = theta2[k] + h
            tm = dict(theta2); tm[k] = max(theta2[k] - h, 1e-9)
            gp = _gbar(tp, pool, targets, k_inner, seed0=1000)
            gm = _gbar(tm, pool, targets, k_inner, seed0=1000)
            Jg[:, j] = (gp - gm) / (tp[k] - tm[k])

    # ── Inference (outside the pool) ───────────────────────────────────────────
    GWG = Jg.T @ W @ Jg
    GWG_inv = np.linalg.inv(GWG)
    cov_theta = GWG_inv / k_inner            # Var(θ̂₂): 1/K sets simulated-moment sampling var
    se_asym = np.sqrt(np.clip(np.diag(cov_theta), 0, None))

    # Parametric bootstrap: draw g_b ~ N(ĝ, S/K); propagate through linearized map.
    rng = np.random.default_rng(0)
    M_lin = GWG_inv @ Jg.T @ W            # (3×4): θ_b = θ̂₂ − M_lin (g_b − ĝ)
    draws = rng.multivariate_normal(g_hat, S / k_inner, size=n_boot)
    theta_b = np.array([theta2[k] for k in keys])[None, :] - (draws - g_hat) @ M_lin.T
    se_boot = theta_b.std(axis=0, ddof=1)

    # J-test of the single overidentifying restriction.
    J = float(k_inner * (g_hat @ W @ g_hat))
    dof = len(MOMENT_KEYS) - len(keys)
    p_j = float(chi2.sf(J, dof))

    print("\n=== MSM efficient-weight results ===", flush=True)
    for i, k in enumerate(keys):
        print(f"  {k:15s}  θ̂₁={theta1[k]:.5f}  θ̂₂={theta2[k]:.5f}  "
              f"SE_asym={se_asym[i]:.5f}  SE_boot={se_boot[i]:.5f}", flush=True)
    print(f"  J-test: J={J:.3f}  dof={dof}  p={p_j:.4f}  "
          f"({'reject' if p_j < 0.05 else 'do not reject'} @5%)", flush=True)
    print(f"  elapsed: {(time.monotonic()-t0)/60:.1f}m", flush=True)

    # ── Output CSV (one row per parameter + a J-test summary row) ──────────────
    rows = []
    for i, k in enumerate(keys):
        rows.append({
            "parameter": k,
            "theta_step1_identity": theta1[k],
            "theta_step2_efficient": theta2[k],
            "se_asymptotic": se_asym[i],
            "se_bootstrap": se_boot[i],
            "J_stat": J, "J_dof": dof, "J_pvalue": p_j,
            "reps_for_S": reps, "k_inner": k_inner, "n_boot": n_boot,
            "n_ticks": n_ticks, "burn_in": burn_in,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  saved: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
