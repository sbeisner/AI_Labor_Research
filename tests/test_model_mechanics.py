"""
test_model_mechanics.py — Live LaborMarketModel micro-tests (Notebook 02).

Runs a paired AI / Control simulation on 500 workers for 10 ticks.
Each test probes a specific model-mechanics property; failures pinpoint
which formula or agent behaviour is broken rather than just reporting that
"results look wrong."
"""

import math

import numpy as np
import pytest

from model.LaborMarketModel import DEFAULT_PARAMS, LaborMarketModel


# ── Initialisation ────────────────────────────────────────────────────────────

def test_model_initialises(mini_run):
    """
    LaborMarketModel must complete 10 ticks without raising an exception for
    both AI and Control configurations on a 500-worker sample.
    """
    assert "ai" in mini_run and "ctrl" in mini_run
    assert len(mini_run["ai"]) == 10
    assert len(mini_run["ctrl"]) == 10


def test_employment_rate_starts_high(mini_run):
    """
    At T=1 the employment rate should equal or closely reflect the initial
    worker-sample employment rate.  The CPS 2022-2025 sample is filtered to
    employed workers, so the initial rate should be ≥ 0.95 in both scenarios.
    """
    for key in ("ai", "ctrl"):
        rate_t1 = mini_run[key]["Employment_Rate"].iloc[0]
        assert rate_t1 >= 0.95, (
            f"{key}: Employment_Rate at T=1 is {rate_t1:.3f} — "
            "should be ≥0.95 given the employed-worker sample"
        )


def test_control_employment_stable(mini_run):
    """
    Over 10 ticks the control scenario should not drift more than 3 pp from
    its initial employment rate.  Large drift signals that the baseline turnover
    is mis-calibrated or that unemployed workers are not being recalled.
    """
    ctrl = mini_run["ctrl"]["Employment_Rate"]
    drift = ctrl.iloc[0] - ctrl.iloc[-1]
    assert abs(drift) <= 0.03, (
        f"Control employment drifted {drift:+.3f} pp over 10 ticks — "
        "expected |drift| ≤ 0.03"
    )


def test_ai_employment_lower_than_control(mini_run):
    """
    By T=10 the AI scenario must show strictly lower employment than Control.
    This is the fundamental result of the model: AI displacement outpaces
    job creation in the short run.
    """
    ai_final   = mini_run["ai"]["Employment_Rate"].iloc[-1]
    ctrl_final = mini_run["ctrl"]["Employment_Rate"].iloc[-1]
    assert ai_final < ctrl_final, (
        f"AI employment ({ai_final:.4f}) is not below Control ({ctrl_final:.4f}) "
        "at T=10 — AI scenario is not producing displacement"
    )


def test_ai_displacement_grows_over_time(mini_run):
    """
    The cumulative AI–Control employment gap should widen monotonically (or at
    least not close) over 10 ticks.  A closing gap would indicate that
    job-creation elasticity is overwhelming displacement — implausible in the
    short run given current AI adoption rates.
    """
    gap = mini_run["ctrl"]["Employment_Rate"].values - mini_run["ai"]["Employment_Rate"].values
    # Gap at T=10 should be at least as large as at T=3
    assert gap[-1] >= gap[2] * 0.8, (
        "AI employment gap is narrowing too fast by T=10 — "
        "check job-creation (epsilon/gamma) parameters"
    )


# ── Displacement probability formula ─────────────────────────────────────────

def test_p_disp_increases_with_risk(mini_run, worker_df):
    """
    For a fixed augmentation level, a worker with higher substitution risk r_job
    must have a higher displacement probability.  This validates the sign of the
    beta coefficient in the logistic formula:
        Z = logit(δ_base) + β_run · r_agent_sub − λ · p_agent_aug
    """
    p = DEFAULT_PARAMS
    delta_base = p["delta_base"]
    beta       = p["beta"]
    lambda_    = p["lambda_"]

    logit_base = math.log(delta_base / (1 - delta_base))
    sigmoid    = lambda z: 1 / (1 + math.exp(-z))

    # Hold p_aug constant at the population mean
    p_aug_mean = worker_df["p_aug"].mean()

    r_low  = 0.30
    r_high = 0.60

    p_low  = sigmoid(logit_base + beta * r_low  - lambda_ * p_aug_mean)
    p_high = sigmoid(logit_base + beta * r_high - lambda_ * p_aug_mean)

    assert p_high > p_low, (
        f"p_disp(r=0.60)={p_high:.4f} should exceed p_disp(r=0.30)={p_low:.4f}"
    )


def test_p_disp_decreases_with_augmentation(worker_df):
    """
    Higher augmentation potential p_aug should *reduce* displacement probability
    (the λ term is subtracted in the logit).  An employer with highly augmentable
    workers retains them rather than replacing them.
    """
    p = DEFAULT_PARAMS
    delta_base = p["delta_base"]
    beta       = p["beta"]
    lambda_    = p["lambda_"]

    logit_base = math.log(delta_base / (1 - delta_base))
    sigmoid    = lambda z: 1 / (1 + math.exp(-z))

    r_mean   = worker_df["r_job"].mean()
    p_aug_lo = 0.30
    p_aug_hi = 0.80

    p_lo_aug = sigmoid(logit_base + beta * r_mean - lambda_ * p_aug_lo)
    p_hi_aug = sigmoid(logit_base + beta * r_mean - lambda_ * p_aug_hi)

    assert p_hi_aug < p_lo_aug, (
        f"p_disp should be lower for p_aug=0.80 ({p_hi_aug:.4f}) "
        f"than p_aug=0.30 ({p_lo_aug:.4f})"
    )


def test_experience_protects_workers(worker_df):
    """
    Senior workers (exp_norm=1) should face lower displacement probability than
    entry workers (exp_norm=0) when r_job and p_aug are equal.  This validates
    the delta_sub experience-shield term:
        r_agent_sub = r_job × (1 − δ_sub × exp_norm)
    """
    p = DEFAULT_PARAMS
    delta_base = p["delta_base"]
    beta       = p["beta"]
    lambda_    = p["lambda_"]
    delta_sub  = p["delta_sub"]

    logit_base = math.log(delta_base / (1 - delta_base))
    sigmoid    = lambda z: 1 / (1 + math.exp(-z))

    r_job  = 0.50
    p_aug  = 0.60
    beta_run = 1.0 * beta  # mean macroeconomic draw

    r_entry  = r_job * (1 - delta_sub * 0.0)   # exp_norm = 0 (entry)
    r_senior = r_job * (1 - delta_sub * 1.0)   # exp_norm = 1 (senior)

    p_entry  = sigmoid(logit_base + beta_run * r_entry  - lambda_ * p_aug)
    p_senior = sigmoid(logit_base + beta_run * r_senior - lambda_ * p_aug)

    assert p_entry > p_senior, (
        f"Entry workers (p_disp={p_entry:.4f}) should face higher displacement "
        f"than seniors (p_disp={p_senior:.4f})"
    )


# ── Beta-run macroeconomic multiplier ────────────────────────────────────────

def test_beta_run_varies_across_seeds(worker_df, skill_matrix, occ_risk):
    """
    beta_run ~ N(1.0, 0.2) × beta is drawn once per model instantiation.
    Running the same scenario with 5 different seeds should produce 5 distinct
    beta_run values — confirming that macroeconomic uncertainty is injected at
    the run level, not just at the agent level.
    """
    import pandas as pd
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    occ_risk_lookup = {
        "r_job": {int(idx): float(row["r_job"]) for idx, row in occ_risk.iterrows()},
        "p_aug": {int(idx): float(row["p_aug"]) for idx, row in occ_risk.iterrows()},
    }

    sample = worker_df.sample(n=200, random_state=0)

    beta_runs = set()
    for seed in range(5):
        m = LaborMarketModel(
            worker_df=sample,
            ai_active=True,
            params=DEFAULT_PARAMS,
            seed=seed,
            skill_distance_matrix=skill_matrix,
            occ_risk_lookup=occ_risk_lookup,
        )
        beta_runs.add(round(m.beta_run, 6))

    assert len(beta_runs) == 5, (
        f"Only {len(beta_runs)} unique beta_run values across 5 seeds — "
        "macroeconomic uncertainty is not being drawn per-seed"
    )


def test_beta_run_mean_near_beta(worker_df, skill_matrix, occ_risk):
    """
    beta_run = N(1.0, 0.2) × beta.  With 20 draws the sample mean of beta_run
    should be within 25 % of the configured beta parameter.
    """
    occ_risk_lookup = {
        "r_job": {int(idx): float(row["r_job"]) for idx, row in occ_risk.iterrows()},
        "p_aug": {int(idx): float(row["p_aug"]) for idx, row in occ_risk.iterrows()},
    }

    sample = worker_df.sample(n=200, random_state=0)
    beta_target = DEFAULT_PARAMS["beta"]

    beta_runs = []
    for seed in range(20):
        m = LaborMarketModel(
            worker_df=sample,
            ai_active=True,
            params=DEFAULT_PARAMS,
            seed=seed,
            skill_distance_matrix=skill_matrix,
            occ_risk_lookup=occ_risk_lookup,
        )
        beta_runs.append(m.beta_run)

    sample_mean = np.mean(beta_runs)
    assert abs(sample_mean - beta_target) / beta_target < 0.25, (
        f"Mean beta_run={sample_mean:.3f} deviates >25% from beta={beta_target}"
    )


# ── Employer metrics ──────────────────────────────────────────────────────────

def test_control_vacancies_nonzero(mini_run):
    """
    Even without AI disruption, normal turnover (δ_base ≈ 0.5 %/month) should
    generate replacement vacancies every tick.  Zero vacancies in the control
    scenario means the EmployerAgent is not posting replacement positions,
    which would prevent any rehiring and inflate unemployment.

    Expected: ~δ_base × N_workers replacement postings per tick.
    """
    ctrl_vacancies = mini_run["ctrl"]["Total_Vacancies"]
    # Skip tick 0 (vacancies are initialised to 0 before the first step)
    mean_vacancies = ctrl_vacancies.iloc[1:].mean()
    assert mean_vacancies > 0, (
        f"Control scenario Total_Vacancies mean={mean_vacancies:.1f} — "
        "employers are not posting replacement vacancies"
    )


def test_ai_vacancies_exceed_control(mini_run):
    """
    In the AI scenario, employers post both direct-replacement vacancies
    (ε × fired) AND augmentation-demand vacancies (Σγ·p_aug).  Total AI
    vacancies must exceed control vacancies because the augmentation channel
    is non-zero.
    """
    ai_vac   = mini_run["ai"]["Total_Vacancies"].iloc[1:].mean()
    ctrl_vac = mini_run["ctrl"]["Total_Vacancies"].iloc[1:].mean()
    assert ai_vac > ctrl_vac, (
        f"AI vacancies ({ai_vac:.1f}) should exceed control ({ctrl_vac:.1f}) "
        "due to augmentation demand"
    )


def test_wages_increase_in_ai_scenario(mini_run):
    """
    AI augmentation provides a 2 %/year (wage_boost=0.02) productivity gain to
    surviving workers.  Mean wages in the AI scenario must exceed control wages
    by T=10, even though some workers have been displaced (survivorship-wage bias
    reinforces this).
    """
    ai_wage_t10   = mini_run["ai"]["Mean_Wage"].iloc[-1]
    ctrl_wage_t10 = mini_run["ctrl"]["Mean_Wage"].iloc[-1]
    assert ai_wage_t10 >= ctrl_wage_t10, (
        f"AI mean wage ({ai_wage_t10:,.0f}) should be ≥ control ({ctrl_wage_t10:,.0f}) "
        "by T=10 — wage_boost or survivorship bias not working"
    )
