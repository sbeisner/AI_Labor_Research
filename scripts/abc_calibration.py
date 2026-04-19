"""Approximate Bayesian Computation (ABC-rejection) calibration.

Calibrates two unobservable baseline parameters against the BLS 2015-2019
steady-state unemployment rate (4.5 %):

    delta_base   — monthly baseline turnover rate
    vacancy_rate — open positions as a fraction of occupation size

The ABC-rejection algorithm:
  1. Draw (delta_base, vacancy_rate) uniformly from prior bounds.
  2. Run the CONTROL scenario (ai_active=False) for N_TICKS months.
  3. Compute steady-state unemployment from the final BURN_IN ticks.
  4. Accept the particle if |u_sim − u_target| ≤ TOLERANCE (ε).

Only the control scenario is used, so AI parameters (beta, lambda_, beta_run)
have no effect on results.  The posterior gives the parameter region that
simultaneously reproduces:
  • BLS quit-rate-derived turnover   (via delta_base)
  • JOLTS-anchored vacancy rate      (via vacancy_rate)

Output is cached to output/abc_posterior.csv.  Set FORCE_RERUN = True to
regenerate.  For a smoke test use N_SAMPLES = 10; for full calibration use
N_SAMPLES ≥ 1000.
"""

import sys
import pathlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────

TARGET_UNRATE = 0.045   # 2015-2019 BLS average unemployment rate (4.5 %)
TOLERANCE     = 0.005   # ε: accept particles within ±0.5 pp of target
N_SAMPLES     = 2000      # ← 10 for smoke test; use 1000+ for full calibration
N_TICKS       = 120     # simulation length in months (extended for equilibrium)
BURN_IN       = 24      # final ticks averaged for steady-state UR estimate
OUTPUT_PATH   = ROOT / "output" / "abc_posterior.csv"
FORCE_RERUN   = True   # True → ignore cache and re-run even if output exists

# ── Prior bounds (uniform) ─────────────────────────────────────────────────────

PRIOR = {
    # Monthly turnover: wider upper bound to accommodate recall-model dynamics.
    # With employer-recall, UR equilibrium requires higher separation rate
    # (~1-6%/month) to maintain 4.5% frictional unemployment.
    "delta_base":   (0.005, 0.060),
    # Open-position fraction: JOLTS 3-5% baseline; allow 1-8% range.
    "vacancy_rate": (0.010, 0.080),
}

# ── Per-process data cache ─────────────────────────────────────────────────────
# joblib (loky backend) spawns a pool of persistent worker processes. Each
# process handles multiple particles. We load shared data once per process and
# store it in this module-level dict so subsequent particles skip disk I/O.
_PROCESS_CACHE: dict = {}


def _get_data():
    """Return (worker_df, dist_matrix, occ_risk) for the current process."""
    if "worker_df" not in _PROCESS_CACHE:
        from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
        (
            _PROCESS_CACHE["worker_df"],
            _PROCESS_CACHE["dist_matrix"],
            _PROCESS_CACHE["occ_risk"],
        ) = load_shared_data()
    return (
        _PROCESS_CACHE["worker_df"],
        _PROCESS_CACHE["dist_matrix"],
        _PROCESS_CACHE["occ_risk"],
    )


# ── Particle evaluation ────────────────────────────────────────────────────────

def evaluate_particle(seed: int):
    """Draw one parameter particle and return it if accepted by ABC criterion.

    The prior draw is seeded from the particle index so the full sweep is
    exactly reproducible: re-running with the same N_SAMPLES and seed range
    will produce the same accepted set regardless of worker scheduling order.

    Args:
        seed: particle index (0 … N_SAMPLES-1); doubles as the simulation seed.

    Returns:
        dict with sampled params and fit statistics if |u_sim − u_target| ≤ ε,
        else None.
    """
    worker_df, dist_matrix, occ_risk = _get_data()

    # Prior draw — seeded from particle index for reproducibility
    rng           = np.random.default_rng(seed)
    sampled_delta = float(rng.uniform(*PRIOR["delta_base"]))
    sampled_vac   = float(rng.uniform(*PRIOR["vacancy_rate"]))

    # Build full parameter dict: override only the two calibrated params
    params = {
        **DEFAULT_PARAMS,
        "delta_base":   sampled_delta,
        "vacancy_rate": sampled_vac,
    }

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

    md           = model.datacollector.get_model_vars_dataframe()
    steady_emp   = float(md["Employment_Rate"].iloc[-BURN_IN:].mean())
    simulated_ur = 1.0 - steady_emp
    distance     = abs(simulated_ur - TARGET_UNRATE)

    if distance <= TOLERANCE:
        return {
            "seed":             seed,
            "delta_base":       sampled_delta,
            "vacancy_rate":     sampled_vac,
            "simulated_unrate": simulated_ur,
            "distance":         distance,
        }
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if OUTPUT_PATH.exists() and not FORCE_RERUN:
        print(f"[abc_calibration] Cached posterior found — loading {OUTPUT_PATH}")
        posterior_df = pd.read_csv(OUTPUT_PATH)
    else:
        print(
            f"[abc_calibration] Starting ABC-rejection sweep:\n"
            f"  N_SAMPLES = {N_SAMPLES},  N_TICKS = {N_TICKS},  BURN_IN = {BURN_IN}\n"
            f"  target u  = {TARGET_UNRATE:.1%},  ε = {TOLERANCE:.1%}\n"
            f"  priors    : delta_base ∈ {PRIOR['delta_base']}, "
            f"vacancy_rate ∈ {PRIOR['vacancy_rate']}"
        )

        particles = Parallel(n_jobs=-1, verbose=5)(
            delayed(evaluate_particle)(i) for i in range(N_SAMPLES)
        )

        accepted     = [p for p in particles if p is not None]
        posterior_df = pd.DataFrame(accepted)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        posterior_df.to_csv(OUTPUT_PATH, index=False)

        n_acc = len(posterior_df)
        pct   = 100 * n_acc / N_SAMPLES
        print(
            f"\n[abc_calibration] Accepted {n_acc}/{N_SAMPLES} particles "
            f"({pct:.1f} %)\n"
            f"  Saved → {OUTPUT_PATH}"
        )

    if posterior_df.empty:
        print(
            "\n[abc_calibration] No particles accepted.\n"
            "  → Widen TOLERANCE, expand PRIOR bounds, or increase N_TICKS."
        )
    else:
        print("\nPosterior summary:")
        print(
            posterior_df[
                ["delta_base", "vacancy_rate", "simulated_unrate", "distance"]
            ].describe().round(5)
        )
        print("\nPosterior means (candidates for DEFAULT_PARAMS):")
        print(f"  delta_base:   {posterior_df['delta_base'].mean():.5f}")
        print(f"  vacancy_rate: {posterior_df['vacancy_rate'].mean():.5f}")
