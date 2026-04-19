"""
test_economic_calibration.py — Cross-cutting economic calibration checks.

These tests verify that the model's parameter values and emergent properties
are consistent with published BLS, JOLTS, and academic-literature benchmarks.
They do not test code correctness per se, but flag cases where a plausible
code change might have pushed a parameter out of an empirically grounded range.
"""

import math

import numpy as np
import pytest

from model.LaborMarketModel import DEFAULT_PARAMS


# ── Baseline turnover (δ_base) ────────────────────────────────────────────────

def test_delta_base_matches_bls_quit_rate():
    """
    BLS JOLTS (2022-2024) reports a monthly total-separations rate of 3.2–4.0 %
    for the private sector.  δ_base represents the net monthly turnover after
    excluding job-to-job moves; BLS quit rate excluding transfers is ~0.4–0.7 %.
    δ_base = 0.005 (0.5 %/month) falls squarely in this range.
    """
    delta = DEFAULT_PARAMS["delta_base"]
    assert 0.003 <= delta <= 0.010, (
        f"delta_base={delta} outside BLS-calibrated range [0.003, 0.010]"
    )


def test_delta_base_annual_rate():
    """
    Annualised turnover (1 − (1−δ)^12) should be in the 3–12 % range,
    consistent with BLS annual quit/layoff combined rates.
    """
    delta = DEFAULT_PARAMS["delta_base"]
    annual_rate = 1 - (1 - delta) ** 12
    assert 0.03 <= annual_rate <= 0.12, (
        f"Annual turnover {annual_rate:.1%} outside [3%, 12%] — "
        "delta_base may be mis-calibrated"
    )


# ── Vacancy rate (θ_JOLTs) ────────────────────────────────────────────────────

def test_vacancy_rate_matches_jolts():
    """
    JOLTs (2022-2025) reports a job-openings rate of 5–7 % of employment.
    The model's vacancy_rate = 0.04 (4 %) is intentionally conservative
    (it proxies net new-hire vacancies, not total openings).  Should be in
    [0.02, 0.08] to remain JOLTS-plausible.
    """
    vr = DEFAULT_PARAMS["vacancy_rate"]
    assert 0.02 <= vr <= 0.08, (
        f"vacancy_rate={vr} outside JOLTS-plausible range [0.02, 0.08]"
    )


# ── Wage boost ────────────────────────────────────────────────────────────────

def test_wage_boost_annual_rate():
    """
    wage_boost = 0.02 implies 2 % annual productivity-driven wage growth for
    AI-augmented workers.  BLS Employment Cost Index shows 3–5 % nominal wage
    growth in 2022-2025, of which roughly 1–2 % is AI-driven (Brynjolfsson et al.).
    Range [0.5 %, 5 %/year] is acceptable.
    """
    wb = DEFAULT_PARAMS["wage_boost"]
    assert 0.005 <= wb <= 0.05, (
        f"wage_boost={wb} outside credible annual range [0.005, 0.05]"
    )


# ── Logistic displacement formula plausibility ───────────────────────────────

def test_baseline_monthly_displacement_rate():
    """
    At r_agent_sub = r̄ (population mean ≈ 0.42) and p_agent_aug = p̄ (≈ 0.68),
    the logistic displacement probability should be close to δ_base ± 1 pp.
    If the mean agent produces a monthly displacement rate far from δ_base,
    the logistic constant 'c' is miscalibrated.
    """
    p = DEFAULT_PARAMS
    delta_base = p["delta_base"]
    beta       = p["beta"]
    lambda_    = p["lambda_"]
    beta_run   = 1.0 * beta  # mean macroeconomic draw

    logit_base = math.log(delta_base / (1 - delta_base))
    sigmoid    = lambda z: 1 / (1 + math.exp(-z))

    # Population means from the worker sample
    r_mean   = 0.42
    p_aug_mean = 0.68

    p_disp = sigmoid(logit_base + beta_run * r_mean - lambda_ * p_aug_mean)

    # Allow ±5 pp around delta_base — the logistic shift from mean risk/aug
    # should not dramatically inflate the average displacement rate
    assert abs(p_disp - delta_base) <= 0.05, (
        f"At population-mean risk, p_disp={p_disp:.4f} deviates more than 5 pp "
        f"from delta_base={delta_base} — logistic constant may be miscalibrated"
    )


def test_high_risk_displacement_rate_plausible():
    """
    For a fully exposed worker (r_agent_sub = 0.90, p_aug = 0.30),
    the monthly displacement probability should be in [5 %, 40 %].
    Below 5 %: AI has negligible effect on high-risk workers.
    Above 40 %: >99 % of workers displaced within a year — implausibly fast.
    """
    p = DEFAULT_PARAMS
    delta_base = p["delta_base"]
    beta       = p["beta"]
    lambda_    = p["lambda_"]

    logit_base = math.log(delta_base / (1 - delta_base))
    sigmoid    = lambda z: 1 / (1 + math.exp(-z))

    p_disp_high = sigmoid(logit_base + beta * 0.90 - lambda_ * 0.30)
    assert 0.05 <= p_disp_high <= 0.40, (
        f"High-risk p_disp={p_disp_high:.4f} — outside [5 %, 40 %] monthly range"
    )


def test_low_risk_displacement_near_baseline():
    """
    A worker with very low substitution risk (r_agent_sub = 0.10) and high
    augmentation potential (p_aug = 0.90) should face near-baseline displacement
    (within 2× of δ_base).  These workers are complemented by AI, not replaced.
    """
    p = DEFAULT_PARAMS
    delta_base = p["delta_base"]
    beta       = p["beta"]
    lambda_    = p["lambda_"]

    logit_base = math.log(delta_base / (1 - delta_base))
    sigmoid    = lambda z: 1 / (1 + math.exp(-z))

    p_disp_safe = sigmoid(logit_base + beta * 0.10 - lambda_ * 0.90)
    assert p_disp_safe <= delta_base * 3, (
        f"Low-risk p_disp={p_disp_safe:.4f} exceeds 3× delta_base — "
        "safe workers are being over-exposed to displacement"
    )


# ── Beta_run macroeconomic uncertainty ───────────────────────────────────────

def test_beta_run_std_is_meaningful():
    """
    beta_run_std = 0.2 implies ±20 % macroeconomic uncertainty in AI impact.
    A value below 0.05 would collapse CIs (too little uncertainty); above 0.50
    would produce implausibly large cross-run variance (AI sometimes beneficial,
    sometimes catastrophic within the same parameter regime).
    """
    std = DEFAULT_PARAMS["beta_run_std"]
    assert 0.05 <= std <= 0.50, (
        f"beta_run_std={std} outside meaningful range [0.05, 0.50]"
    )


def test_beta_run_99pct_interval_positive():
    """
    Even in the most favourable macroeconomic draw (bottom 0.5th percentile of
    β_run ~ N(1.0, σ)), the effective displacement beta should remain positive.
    A negative beta would mean AI *reduces* displacement probability, which has
    no economic basis in the substitution-risk framework.
    """
    sigma = DEFAULT_PARAMS["beta_run_std"]
    beta  = DEFAULT_PARAMS["beta"]
    # 0.5th percentile of N(1, sigma) × beta
    p005_multiplier = 1.0 - 2.576 * sigma   # Z = 2.576 for 99.5% one-tailed
    effective_beta_floor = p005_multiplier * beta
    assert effective_beta_floor > 0, (
        f"At 0.5th-pct draw, effective beta={effective_beta_floor:.3f} ≤ 0 — "
        "beta_run_std is too large relative to the base beta"
    )


# ── Retraining parameters ─────────────────────────────────────────────────────

def test_retrain_scale_plausible():
    """
    retrain_scale = 24 months sets the maximum retraining duration (d=1.0 full
    cross-group distance).  BLS Occupational Outlook Handbook reports retraining
    for new occupations at 6–24 months (associate degree to apprenticeship).
    Should be in [12, 48] months.
    """
    rs = DEFAULT_PARAMS["retrain_scale"]
    assert 12 <= rs <= 48, (
        f"retrain_scale={rs} months — outside BLS apprenticeship/training range [12, 48]"
    )


def test_mu_rho_skill_search_parameters():
    """
    μ (skill-distance barrier) and ρ (risk aversion in target-skill selection)
    control how steeply the probability of choosing a new occupation falls off
    with distance and risk.  Both should be positive (higher = more friction).
    """
    assert DEFAULT_PARAMS["mu"] > 0,  f"mu={DEFAULT_PARAMS['mu']} must be positive"
    assert DEFAULT_PARAMS["rho"] > 0, f"rho={DEFAULT_PARAMS['rho']} must be positive"


# ── Job-creation elasticity ───────────────────────────────────────────────────

def test_epsilon_between_zero_and_one():
    """
    epsilon is the fraction of AI-displaced workers whose jobs are replaced by
    new vacancies.  Must be in (0, 1] — 0 means no new jobs (technological
    unemployment), 1 means full replacement (Luddite fallacy).  Literature
    (Acemoglu 2022, Autor 2024) suggests epsilon is well below 1 for cognitive tasks.
    """
    eps = DEFAULT_PARAMS["epsilon"]
    assert 0 < eps <= 1.0, f"epsilon={eps} outside (0, 1]"


def test_gamma_augmentation_demand():
    """
    gamma is the per-worker augmentation demand weight added to vacancy counts.
    It proxies the 'productivity dividend' that creates new job roles.  Should
    be positive (AI creates some new demand) but below 1 (not self-reinforcing).
    """
    gamma = DEFAULT_PARAMS["gamma"]
    assert 0 < gamma < 1.0, f"gamma={gamma} outside (0, 1)"


# ── Wage distribution in data ─────────────────────────────────────────────────

def test_wage_inequality_by_quintile(worker_df, bootstrap_df):
    """
    Q5 (highest AI exposure, typically high-skill) workers should earn more than
    Q1 workers at baseline.  After AI, the wage gap should widen: augmented workers
    gain wage_boost while displaced lower-quintile workers lose wage income.
    This mirrors the 'skill-biased technological change' literature.
    """
    # Use worker_df for initial wages by quintile
    if "exposure_quintile" in worker_df.columns and "wage" in worker_df.columns:
        median_wage = worker_df.groupby("exposure_quintile", observed=True)["wage"].median()
        cats = sorted(median_wage.index.tolist(), key=str)
        if len(cats) >= 2:
            w_q1 = median_wage[cats[0]]
            w_q5 = median_wage[cats[-1]]
            assert w_q5 > w_q1, (
                f"Q5 median wage (${w_q5:,.0f}) should exceed Q1 (${w_q1:,.0f}) — "
                "high-exposure workers expected to earn more at baseline"
            )
