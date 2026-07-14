"""DEPRECATED — Approximate Bayesian Computation (ABC-rejection) calibration.

⚠ Do not use this script for current calibration runs. ⚠

The audit fix in `Code and Methodology Review.pdf` (May 2026) introduced a
Keynesian aggregate-demand feedback loop between the labor and goods markets.
With endogenous demand feedback active, the simulation no longer converges to
a stationary steady-state UR — it exhibits endogenous business-cycle
fluctuations that ABC-rejection's single-mean target cannot capture.  Calibrating
to a single steady-state mean would either (a) require artificially damping
the feedback loop (defeating the audit fix) or (b) produce a degenerate
posterior whose simulated UR oscillates around the target rather than
matching it on average.

The current calibrator is `scripts/msm_calibration.py`, which uses the
Method of Simulated Moments to match the *dynamic* moments of the BLS data
(mean, variance, autocorrelation, std of first differences).  This is the
standard approach for ABMs with endogenous demand and cyclical dynamics
(Dosi et al. 2010; Delli Gatti et al. 2011; Franke & Westerhoff 2012).

This file is retained for historical reference and reproducibility of the
pre-audit Run-10 posterior reported in earlier draft manuscripts.

────────────────────────────────────────────────────────────────────────────
ORIGINAL DOCSTRING:

Calibrates two unobservable baseline parameters against the BLS 2015-2019
steady-state unemployment rate (4.5 %):

    delta_base   — monthly baseline turnover rate
    vacancy_rate — open positions as a fraction of occupation size

The ABC-rejection algorithm:
  1. Draw (delta_base, vacancy_rate) uniformly from prior bounds.
  2. Run the CONTROL scenario (ai_active=False) for N_TICKS months.
  3. Compute steady-state unemployment from the final BURN_IN ticks.
  4. Accept the particle if |u_sim − u_target| ≤ TOLERANCE (ε).
"""
import sys

print(
    "[abc_calibration] DEPRECATED — this calibration approach is incompatible\n"
    "  with the post-audit Keynesian feedback loop.\n"
    "  Use `python scripts/msm_calibration.py` instead.\n"
    "  Set ABC_FORCE_RUN=1 to override and run anyway (not recommended).",
    file=sys.stderr,
)
import os as _os  # noqa: E402
if _os.environ.get("ABC_FORCE_RUN") != "1":
    sys.exit(2)


import sys
import pathlib
import time
import multiprocessing as mp
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────

TARGET_UNRATE = 0.045   # 2015-2019 BLS average unemployment rate (4.5 %)
TOLERANCE     = 0.005   # ε: accept particles within ±0.5 pp of target
N_SAMPLES     = 2000    # ← 10 for smoke test; use 1000+ for full calibration
# Audit-fix recalibration: the new code (CES reinstatement, radiation matching,
# Keynesian feedback) requires a longer equilibration horizon and a wider
# averaging window than the legacy 120/24 configuration.  The Keynesian loop
# in particular has half-life ~12 ticks, so a 60-tick burn-in average is
# needed to wash out single-cycle noise.
N_TICKS       = 180     # simulation length in months
BURN_IN       = 60      # final ticks averaged for steady-state UR estimate
OUTPUT_PATH   = ROOT / "output" / "abc_posterior.csv"
FORCE_RERUN   = True    # True → ignore cache and re-run even if output exists

# Cap workers to avoid memory exhaustion.
# Each worker holds the shared data (~15 MB) plus a full Mesa model in flight
# (~150 MB peak).  On a 16 GB machine 8 workers ≈ safe ceiling.
N_WORKERS = min(mp.cpu_count(), 8)

# ── Prior bounds (uniform) ─────────────────────────────────────────────────────

PRIOR = {
    # Monthly turnover — re-widened for the audit-fix recalibration.  Pre-sweep
    # probes (3 seeds × 180 ticks, Control scenario, post-audit code) at
    # vr=0.05:
    #   delta=0.013 → UR≈4.3%  (ACCEPT-region candidate)
    #   delta=0.020 → UR≈6.0%  (reject high)
    #   delta=0.025 → UR≈4.6%  (ACCEPT-region candidate; volatile)
    #   delta=0.030 → UR≈6.6%  (reject high)
    # The volatility around 0.025 reflects nonlinear interaction between the
    # Keynesian feedback and the AR(1) macro shock; wider prior allows the
    # posterior to characterise the non-identifiability rather than masking it.
    "delta_base":   (0.008, 0.030),
    # Open-position fraction: JOLTS 3-5% baseline; partially identified by UR
    # target alone, so kept wide.
    "vacancy_rate": (0.010, 0.080),
}

# ── Worker-process state (populated by _worker_init in each pool process) ───────
# mp.Pool uses "spawn" on macOS (Python 3.8+), so each worker starts a fresh
# interpreter.  _worker_init() loads shared data once and stores it here;
# evaluate_particle() reads it directly — no per-particle disk I/O.
_worker_df   = None
_dist_matrix = None
_occ_risk    = None


def _worker_init():
    """Load shared data once per worker process (mp.Pool initializer).

    Called exactly once per worker at pool creation.  Subsequent calls to
    evaluate_particle() in the same worker process reuse the cached globals
    without any disk I/O.
    """
    global _worker_df, _dist_matrix, _occ_risk
    from scripts.bootstrap_runner import load_shared_data  # noqa: PLC0415
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()


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
        print(f"[abc_calibration] Cached posterior found — loading {OUTPUT_PATH}", flush=True)
        posterior_df = pd.read_csv(OUTPUT_PATH)
    else:
        print(
            f"[abc_calibration] Starting ABC-rejection sweep:\n"
            f"  N_SAMPLES = {N_SAMPLES},  N_TICKS = {N_TICKS},  BURN_IN = {BURN_IN}\n"
            f"  target u  = {TARGET_UNRATE:.1%},  ε = {TOLERANCE:.1%}\n"
            f"  priors    : delta_base ∈ {PRIOR['delta_base']}, "
            f"vacancy_rate ∈ {PRIOR['vacancy_rate']}\n"
            f"  workers   : {N_WORKERS} (cpu_count={mp.cpu_count()}, capped at 8)",
            flush=True,
        )

        start_time = time.monotonic()
        start_wall = datetime.now()
        print(f"  started at  : {start_wall:%Y-%m-%d %H:%M:%S}\n", flush=True)

        particles    = []
        REPORT_EVERY = 200

        with mp.Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
            for i, result in enumerate(
                pool.imap_unordered(evaluate_particle, range(N_SAMPLES)),
                start=1,
            ):
                particles.append(result)
                if i % REPORT_EVERY == 0 or i == N_SAMPLES:
                    elapsed      = time.monotonic() - start_time
                    rate         = i / elapsed
                    remaining    = (N_SAMPLES - i) / rate if rate > 0 else 0
                    eta          = datetime.now() + timedelta(seconds=remaining)
                    n_acc_so_far = sum(1 for p in particles if p is not None)
                    print(
                        f"  [{datetime.now():%H:%M:%S}]  {i:>4}/{N_SAMPLES} done"
                        f"  |  accepted so far: {n_acc_so_far}"
                        f"  |  elapsed: {elapsed/60:.1f}m"
                        f"  |  ETA: {eta:%H:%M:%S}",
                        flush=True,
                    )

        accepted     = [p for p in particles if p is not None]
        posterior_df = pd.DataFrame(accepted)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        posterior_df.to_csv(OUTPUT_PATH, index=False)

        elapsed_total = time.monotonic() - start_time
        n_acc = len(posterior_df)
        pct   = 100 * n_acc / N_SAMPLES
        print(
            f"\n[abc_calibration] Accepted {n_acc}/{N_SAMPLES} particles "
            f"({pct:.1f} %)  |  total time: {elapsed_total/60:.1f}m\n"
            f"  Saved → {OUTPUT_PATH}",
            flush=True,
        )

    if posterior_df.empty:
        print(
            "\n[abc_calibration] No particles accepted.\n"
            "  → Widen TOLERANCE, expand PRIOR bounds, or increase N_TICKS.",
            flush=True,
        )
    else:
        print("\nPosterior summary:", flush=True)
        print(
            posterior_df[
                ["delta_base", "vacancy_rate", "simulated_unrate", "distance"]
            ].describe().round(5)
        )
        print("\nPosterior means (candidates for DEFAULT_PARAMS):", flush=True)
        print(f"  delta_base:   {posterior_df['delta_base'].mean():.5f}", flush=True)
        print(f"  vacancy_rate: {posterior_df['vacancy_rate'].mean():.5f}", flush=True)
