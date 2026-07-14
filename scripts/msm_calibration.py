"""Method of Simulated Moments (MSM) calibration.

Replaces the deprecated `abc_calibration.py` (steady-state-mean ABC-rejection),
which assumed a stationary labor-market equilibrium.  After the audit fix
(`Code and Methodology Review.pdf`), the model carries a Keynesian aggregate-
demand feedback loop: the OU drift anchor μ_j now responds to the aggregate
wage bill, so the system exhibits endogenous business-cycle non-stationarity
even in the Control scenario.  A single-mean target is therefore the wrong
estimand — the cyclical second moments (variance, persistence, Beveridge
correlation) are the real macroeconomic features the model should match.

This is the standard calibration approach for ABMs with endogenous demand
and cyclical dynamics (e.g. Dosi et al. 2010; Delli Gatti et al. 2011;
Franke & Westerhoff 2012).

Empirical targets are anchored by moment role across two BLS Civilian-UR
windows (see MOMENT_WINDOWS).  Level and persistence describe the prevailing
regime and use 2015-2019; cyclical volatility describes business-cycle
amplitude — which a cycle-generating model cannot reproduce from a single calm
expansion — and uses the multi-cycle 2000-2019 window (dot-com + GFC, COVID
excluded).  Targets are recomputed from the parquet at runtime; validated
values (ddof=1, proportion units) are:
  m1 — mean UR                    : 0.04415   [2015-2019]
  m2 — std UR                     : 0.018095  [2000-2019]
  m3 — AC(1) UR                   : 0.981889  [2015-2019]
  m4 — std of monthly UR change   : 0.001598  [2000-2019]

Calibrated parameters:
  θ1 — delta_base       (frictional separation rate)
  θ2 — vacancy_rate     (JOLTS-anchored open-position fraction)
  θ3 — btos_macro_std   (common AR(1) macro shock std)

Loss function (relative squared error, GMM-style):
  L(θ) = Σ_i W_i · ((m_sim_i(θ) - m_emp_i) / m_emp_i)^2
  --weighting identity  : W_i = 1 (default; original behaviour)
  --weighting efficient : W_i = 1 / Var(rel_i), the diagonal efficient GMM
      weight, with Var(rel_i) estimated by moving-block bootstrap of the
      empirical moment (preserving serial dependence for AC(1)) and inflated
      by (1 + 1/K_INNER) for K-seed simulation noise.  Weights are normalised
      to mean 1 so the reported loss stays comparable across schemes.

Optimizer:
  Nelder-Mead simplex on log-transformed parameters (positive-only domain).
  Each evaluation averages moments across K_INNER simulation seeds for
  numerical stability against finite-sample noise.

Output → output/msm_posterior.csv  (single-row CSV with calibrated θ + moments)
       → output/msm_calibration_run.log  (per-iteration trace)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import sys
import time
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS  # noqa: E402

# ── Empirical moment targets, anchored by moment role ──────────────────────────
# Level & persistence  → BLS 2015-2019 (prevailing regime the AI shock perturbs).
# Cyclical volatility   → BLS 2000-2019 (representative multi-cycle amplitude;
#   the 2015-2019 leg is a quiescent trough whose std, 0.0062, sits far below
#   historical cyclical amplitude and is unreachable for a cycle-generating model).
BLS_PARQUET = ROOT / "data" / "external" / "bls_unrate_monthly.parquet"

MOMENT_KEYS = ["mean_ur", "std_ur", "ac1_ur", "std_d_ur"]
MOMENT_WINDOWS = {
    "mean_ur":   ("2015", "2019"),
    "std_ur":    ("2000", "2019"),
    "ac1_ur":    ("2015", "2019"),
    "std_d_ur":  ("2000", "2019"),
}


def _series_moment(ur: np.ndarray, key: str) -> float:
    """One moment from a UR array; matches _moments_from_ur (pandas ddof=1)."""
    if key == "mean_ur":
        return float(ur.mean())
    if key == "std_ur":
        return float(ur.std(ddof=1))
    if key == "ac1_ur":
        return float(np.corrcoef(ur[:-1], ur[1:])[0, 1])
    if key == "std_d_ur":
        return float(np.diff(ur).std(ddof=1))
    raise KeyError(key)


def load_empirical_targets() -> dict:
    """Recompute the role-anchored target vector from the BLS parquet."""
    s = pd.read_parquet(BLS_PARQUET).set_index("date")["unrate"]
    return {k: _series_moment(s[a:b].values, k) for k, (a, b) in MOMENT_WINDOWS.items()}


def _moving_block_bootstrap(x: np.ndarray, key: str, n_boot: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of `key` via moving blocks (preserves AC structure)."""
    n = len(x)
    block = max(2, int(round(n ** (1.0 / 3.0))))
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        samp = np.concatenate([x[s:s + block] for s in starts])[:n]
        out[b] = _series_moment(samp, key)
    return out


def efficient_weights(targets: dict, k_inner: int, n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """Diagonal efficient GMM weights on the relative moment conditions.

    W_i = 1 / [(1 + 1/K) · Var_boot(m_emp_i) / m_emp_i²], normalised to mean 1.
    Each moment is bootstrapped from its own source window.
    """
    s = pd.read_parquet(BLS_PARQUET).set_index("date")["unrate"]
    rng = np.random.default_rng(seed)
    raw = {}
    for k, (a, b) in MOMENT_WINDOWS.items():
        boot = _moving_block_bootstrap(s[a:b].values, k, n_boot, rng)
        var_emp = float(np.var(boot, ddof=1))
        var_rel = (1.0 + 1.0 / k_inner) * var_emp / max(targets[k] ** 2, 1e-18)
        raw[k] = 1.0 / max(var_rel, 1e-18)
    mean_w = float(np.mean(list(raw.values())))
    return {k: v / mean_w for k, v in raw.items()}

# ── Configuration ──────────────────────────────────────────────────────────────

N_TICKS   = 180   # simulation length per inner-loop seed
BURN_IN   = 60    # discarded transient before moment computation
K_INNER   = 4     # seeds averaged per parameter evaluation (variance-reduction)
N_WORKERS = min(mp.cpu_count(), 8)

# Optimisation initialisation: prior posterior means from Run 10 ABC, in raw
# (untransformed) units.  Nelder-Mead operates on log-units to enforce
# positivity without explicit constraint handling.
THETA_INIT = {
    "delta_base":     0.013,
    "vacancy_rate":   0.045,
    "btos_macro_std": 0.015,
}
# Search bounds (raw units) — Nelder-Mead doesn't enforce these directly, but
# log-transform + clipping inside _simulate_moments guards against runaway.
THETA_BOUNDS = {
    "delta_base":     (0.005, 0.040),
    "vacancy_rate":   (0.010, 0.080),
    "btos_macro_std": (0.005, 0.030),
}

OUT_PATH = ROOT / "output" / "msm_posterior.csv"
LOG_PATH = ROOT / "output" / "msm_calibration_run.log"

# ── Worker-process state (loaded once per pool worker) ─────────────────────────
_worker_df   = None
_dist_matrix = None
_occ_risk    = None


def _worker_init() -> None:
    global _worker_df, _dist_matrix, _occ_risk
    from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()


def _simulate_one(args: tuple[dict, int]) -> dict:
    """Run a single Control-scenario simulation and return its moments."""
    params, seed = args
    model = LaborMarketModel(
        worker_df=_worker_df,
        params=params,
        ai_active=False,
        seed=seed,
        skill_distance_matrix=_dist_matrix,
        occ_risk_lookup=_occ_risk,
        collect_agent_data=False,
    )
    for _ in range(N_TICKS):
        model.step()
    md = model.datacollector.get_model_vars_dataframe()
    ur = (1.0 - md["Employment_Rate"]).iloc[BURN_IN:].reset_index(drop=True)
    return _moments_from_ur(ur)


def _moments_from_ur(ur: pd.Series) -> dict:
    """Compute the moment vector from a UR series (monthly)."""
    d_ur = ur.diff().dropna()
    return {
        "mean_ur":   float(ur.mean()),
        "std_ur":    float(ur.std()),
        "ac1_ur":    float(ur.autocorr(1)),
        "std_d_ur":  float(d_ur.std()),
    }


def _theta_from_log(log_theta: Sequence[float]) -> dict:
    """Map log-units → raw param dict, clipping to bounds for numerical safety."""
    keys = list(THETA_INIT.keys())
    out = {}
    for k, lt in zip(keys, log_theta):
        lo, hi = THETA_BOUNDS[k]
        out[k] = float(np.clip(np.exp(lt), lo, hi))
    return out


def _objective(log_theta: np.ndarray, pool: mp.pool.Pool, targets: dict,
               weights: dict, trace: list, t_start: float) -> float:
    """GMM objective: weighted relative squared deviation from role-anchored targets."""
    theta = _theta_from_log(log_theta)
    params = {**DEFAULT_PARAMS, **theta}

    # Inner loop: K_INNER seeds for variance-reduction
    seed_args = [(params, s) for s in range(K_INNER)]
    seed_moments = pool.map(_simulate_one, seed_args)

    sim_moments = {k: float(np.mean([m[k] for m in seed_moments])) for k in targets}

    loss = 0.0
    deviations = {}
    for k, m_emp in targets.items():
        m_sim = sim_moments[k]
        rel = (m_sim - m_emp) / max(abs(m_emp), 1e-9)
        deviations[k] = rel
        loss += weights[k] * (rel ** 2)

    elapsed = time.monotonic() - t_start
    line = (f"[{datetime.now():%H:%M:%S}]  iter={len(trace)+1:>3}  "
            f"loss={loss:.5f}  Δt={elapsed/60:.1f}m  "
            f"θ=(δ={theta['delta_base']:.4f}, vr={theta['vacancy_rate']:.4f}, "
            f"σ_macro={theta['btos_macro_std']:.4f})  "
            f"sim=(μ={sim_moments['mean_ur']:.4f}, "
            f"σ={sim_moments['std_ur']:.4f}, "
            f"ρ1={sim_moments['ac1_ur']:.4f}, "
            f"σ_Δ={sim_moments['std_d_ur']:.5f})")
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

    trace.append({
        "iter":       len(trace) + 1,
        "loss":       loss,
        **theta,
        **{f"sim_{k}": v for k, v in sim_moments.items()},
        **{f"dev_{k}": v for k, v in deviations.items()},
    })
    return loss


def main(argv: list[str] | None = None) -> int:
    global K_INNER  # noqa: PLW0603 — replaces module-level default with CLI value

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--max-iter", type=int, default=80,
                   help="Nelder-Mead max iterations (default: 80)")
    p.add_argument("--xatol", type=float, default=1e-3,
                   help="parameter-tolerance (log-units) for convergence")
    p.add_argument("--fatol", type=float, default=1e-4,
                   help="loss-tolerance for convergence")
    p.add_argument("--k-inner", type=int, default=K_INNER,
                   help="seeds per parameter evaluation (default: 4)")
    p.add_argument("--weighting", choices=["identity", "efficient"],
                   default="identity",
                   help="GMM weighting matrix (default: identity)")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="bootstrap reps for efficient weights (default: 2000)")
    args = p.parse_args(argv)
    K_INNER = args.k_inner

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(f"# MSM calibration started {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # Role-anchored targets recomputed from the BLS parquet; weights per scheme.
    targets = load_empirical_targets()
    if args.weighting == "efficient":
        weights = efficient_weights(targets, K_INNER, n_boot=args.n_boot)
    else:
        weights = {k: 1.0 for k in MOMENT_KEYS}

    print(
        f"[msm_calibration] Method of Simulated Moments\n"
        f"  empirical targets : {targets}\n"
        f"  moment windows    : {MOMENT_WINDOWS}\n"
        f"  weighting         : {args.weighting}  weights={ {k: round(v, 3) for k, v in weights.items()} }\n"
        f"  N_TICKS={N_TICKS}, BURN_IN={BURN_IN}, K_INNER={K_INNER}\n"
        f"  workers           : {N_WORKERS}\n"
        f"  initial θ         : {THETA_INIT}\n",
        flush=True,
    )

    log_theta_init = np.array([np.log(THETA_INIT[k]) for k in THETA_INIT])
    trace: list[dict] = []
    t_start = time.monotonic()

    with mp.Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
        result = minimize(
            _objective,
            x0=log_theta_init,
            args=(pool, targets, weights, trace, t_start),
            method="Nelder-Mead",
            options={
                "maxiter": args.max_iter,
                "xatol":   args.xatol,
                "fatol":   args.fatol,
                "disp":    True,
            },
        )

    theta_hat   = _theta_from_log(result.x)
    sim_final   = {k: trace[-1][f"sim_{k}"] for k in MOMENT_KEYS}

    print("\n=== MSM converged ===", flush=True)
    print(f"  iterations : {len(trace)}", flush=True)
    print(f"  final loss : {result.fun:.6f}", flush=True)
    print(f"  θ_hat      : {theta_hat}", flush=True)
    print(f"  m_sim      : {sim_final}", flush=True)
    print(f"  m_emp      : {targets}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    win_str = "; ".join(f"{k}:{a}-{b}" for k, (a, b) in MOMENT_WINDOWS.items())
    pd.DataFrame([{
        "loss":       result.fun,
        **theta_hat,
        **{f"sim_{k}": v for k, v in sim_final.items()},
        **{f"emp_{k}": v for k, v in targets.items()},
        "weighting":      args.weighting,
        **{f"w_{k}": v for k, v in weights.items()},
        "moment_windows": win_str,
    }]).to_csv(OUT_PATH, index=False)

    trace_path = OUT_PATH.with_name("msm_trace.csv")
    pd.DataFrame(trace).to_csv(trace_path, index=False)
    print(f"  saved      : {OUT_PATH} (posterior point estimate)", flush=True)
    print(f"  trace      : {trace_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
