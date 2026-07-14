"""
test_bootstrap_sanity.py — Notebook 05 (Bootstrap) output validation.

Validates the 50-seed × 60-tick bootstrap results against:
  - Internal consistency (AI vs Control direction, distributional gradients)
  - BLS / academic benchmarks for AI-era labour-market disruption
  - Statistical validity (CI width, seed-to-seed variance)

All checks are performed on the cached bootstrap_runs.parquet so the suite
runs in seconds without re-running the simulation.
"""

import numpy as np
import pandas as pd
import pytest


# ── Shape & schema ────────────────────────────────────────────────────────────

def test_bootstrap_row_count(bootstrap_df):
    """
    50 seeds × 2 scenarios × 60 ticks = 6 000 rows exactly.
    Any deviation signals that a run was silently dropped or double-counted.
    """
    assert len(bootstrap_df) == 6_000, f"Expected 6 000 rows, got {len(bootstrap_df)}"


def test_bootstrap_tick_range(bootstrap_df):
    """Ticks must run from 0 to 59 inclusive — no missing or extra ticks."""
    ticks = sorted(bootstrap_df["tick"].unique())
    assert ticks == list(range(60)), f"Unexpected tick values: {ticks[:5]}…"


def test_bootstrap_seed_count(bootstrap_df):
    """Exactly 50 seeds must be present for each scenario."""
    for scen in ("AI", "Control"):
        n = bootstrap_df[bootstrap_df["scenario"] == scen]["seed"].nunique()
        assert n == 50, f"{scen}: expected 50 seeds, found {n}"


EXPECTED_COLS = [
    "tick", "Employment_Rate", "Mean_Wage",
    "Emp_Rate_Q1_Low", "Emp_Rate_Q2", "Emp_Rate_Q3",
    "Emp_Rate_Q4", "Emp_Rate_Q5_High",
    "Emp_Rate_Entry", "Emp_Rate_Senior",
    "Retraining_Count", "Retrained_Share",
    "Total_Vacancies", "Total_Hired", "Total_Fired",
    "Avg_BTOS", "seed", "scenario",
]

def test_bootstrap_required_columns(bootstrap_df):
    missing = [c for c in EXPECTED_COLS if c not in bootstrap_df.columns]
    assert not missing, f"Missing bootstrap columns: {missing}"


# ── Employment gap direction ──────────────────────────────────────────────────

def _final(df, scenario):
    return df[(df["scenario"] == scenario) & (df["tick"] == 59)]


def test_employment_gap_is_negative(bootstrap_df):
    """
    The mean employment rate must be lower in the AI scenario than Control at T=59.
    This is the core result: AI displacement outpaces short-run job creation,
    consistent with Acemoglu & Restrepo (2022) task-displacement findings.
    """
    ai_mean   = _final(bootstrap_df, "AI")["Employment_Rate"].mean()
    ctrl_mean = _final(bootstrap_df, "Control")["Employment_Rate"].mean()
    assert ai_mean < ctrl_mean, (
        f"AI employment ({ai_mean:.4f}) must be below Control ({ctrl_mean:.4f})"
    )


def test_employment_gap_substantial(bootstrap_df):
    """
    A well-calibrated AI disruption model should show a gap of at least 5 pp
    over 5 years.  Smaller gaps suggest that displacement parameters or the
    logistic beta are too conservative.
    """
    ai_mean   = _final(bootstrap_df, "AI")["Employment_Rate"].mean()
    ctrl_mean = _final(bootstrap_df, "Control")["Employment_Rate"].mean()
    gap_pp    = (ai_mean - ctrl_mean) * 100
    assert gap_pp <= -5.0, (
        f"Employment gap={gap_pp:.1f} pp — expected ≤ −5 pp for 5-year horizon"
    )


def test_employment_gap_ci_excludes_zero(bootstrap_df):
    """
    The 95 % confidence interval for the employment gap must not include 0.
    A CI that crosses zero means we cannot rule out zero net displacement —
    which would make the model's primary result statistically uninformative.
    """
    ai   = _final(bootstrap_df, "AI")["Employment_Rate"].values
    ctrl = _final(bootstrap_df, "Control")["Employment_Rate"].values
    # Pair each AI run with the same-seed Control run
    df_ai   = _final(bootstrap_df, "AI")[["seed", "Employment_Rate"]].set_index("seed")
    df_ctrl = _final(bootstrap_df, "Control")[["seed", "Employment_Rate"]].set_index("seed")
    gaps = (df_ai["Employment_Rate"] - df_ctrl["Employment_Rate"]).values
    ci_lo = np.percentile(gaps, 2.5)
    ci_hi = np.percentile(gaps, 97.5)
    assert ci_hi < 0, (
        f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}] does not strictly exclude 0 — "
        "AI displacement result is not statistically significant"
    )


def test_employment_gap_widens_over_time(bootstrap_df):
    """
    The AI–Control employment gap should widen from T=1 to T=59.  If the gap
    narrows after the initial shock it implies job-creation outpaces displacement
    in the medium run — inconsistent with 2023-2025 AI-adoption evidence.
    """
    for t_early, t_late in [(5, 30), (30, 59)]:
        ai_e   = bootstrap_df[(bootstrap_df["scenario"] == "AI")   & (bootstrap_df["tick"] == t_early)]["Employment_Rate"].mean()
        ai_l   = bootstrap_df[(bootstrap_df["scenario"] == "AI")   & (bootstrap_df["tick"] == t_late)]["Employment_Rate"].mean()
        ct_e   = bootstrap_df[(bootstrap_df["scenario"] == "Control") & (bootstrap_df["tick"] == t_early)]["Employment_Rate"].mean()
        ct_l   = bootstrap_df[(bootstrap_df["scenario"] == "Control") & (bootstrap_df["tick"] == t_late)]["Employment_Rate"].mean()
        gap_early = ai_e - ct_e
        gap_late  = ai_l - ct_l
        assert gap_late <= gap_early + 0.02, (
            f"Gap narrowed from T={t_early} ({gap_early:.4f}) to T={t_late} ({gap_late:.4f}) "
            "by more than 2 pp — check job-creation elasticity"
        )


# ── Wage dynamics ─────────────────────────────────────────────────────────────

def test_ai_wages_higher_than_control(bootstrap_df):
    """
    AI augmentation boosts productivity wages for surviving workers (wage_boost=0.02/yr).
    Combined with survivorship bias (low-wage high-risk workers displaced first),
    mean wages in AI must exceed Control at T=59.

    Reference: Brynjolfsson, Li & Raymond (2023) find AI raises wages of remaining
    workers by 14% — directional sign must be positive.
    """
    ai_wage   = _final(bootstrap_df, "AI")["Mean_Wage"].mean()
    ctrl_wage = _final(bootstrap_df, "Control")["Mean_Wage"].mean()
    assert ai_wage > ctrl_wage, (
        f"AI mean wage (${ai_wage:,.0f}) should exceed Control (${ctrl_wage:,.0f})"
    )


def test_wage_gap_positive_ci(bootstrap_df):
    """
    The 95% CI for the wage gap (AI − Control) must be entirely above zero,
    confirming that wage gains under AI are robust across macroeconomic scenarios.
    """
    df_ai   = _final(bootstrap_df, "AI")[["seed", "Mean_Wage"]].set_index("seed")
    df_ctrl = _final(bootstrap_df, "Control")[["seed", "Mean_Wage"]].set_index("seed")
    gaps = (df_ai["Mean_Wage"] - df_ctrl["Mean_Wage"]).values
    ci_lo = np.percentile(gaps, 2.5)
    assert ci_lo > 0, (
        f"Wage-gap CI lower bound ${ci_lo:,.0f} ≤ 0 — "
        "wage benefit of AI is not statistically robust"
    )


# ── Distributional inequality ─────────────────────────────────────────────────

def test_senior_workers_employed_more_than_entry_in_ai(bootstrap_df):
    """
    Senior workers (high exp_norm) benefit from both a lower displacement
    probability (experience shield: r_sub = r_job × (1 − δ_sub × exp_norm))
    and preferential rehiring in market clearing (score ∝ 1 + ν·exp_norm).
    In the AI scenario, senior workers should therefore maintain higher
    employment than entry workers at T=59.

    Note: the AI–Control *gap* can be larger for seniors than entries because
    the control scenario's preferential rehiring drives seniors to near-full
    employment (~99.8%), amplifying the relative impact of AI disruption on
    their baseline.  The correct invariant is the within-AI absolute ordering.

    Reference: Acemoglu & Restrepo (2022) — experience-protected workers
    maintain employment advantages even under automation pressure.
    """
    ai_entry  = _final(bootstrap_df, "AI")["Emp_Rate_Entry"].mean()
    ai_senior = _final(bootstrap_df, "AI")["Emp_Rate_Senior"].mean()

    assert ai_senior > ai_entry, (
        f"Senior AI employment ({ai_senior:.4f}) should exceed entry ({ai_entry:.4f}) "
        "— experience shield + rehiring priority should protect senior workers"
    )


def test_quintile_gradient_present(bootstrap_df):
    """
    Higher AI-exposure quintiles (Q5) should face larger employment gaps than
    lower quintiles (Q1).  This validates that the risk-score-based displacement
    formula produces the expected distributional gradient.

    Reference: Felten et al. (2023) — AI exposure is concentrated in specific
    occupation groups, not uniformly distributed.
    """
    df_ai   = _final(bootstrap_df, "AI")[["seed", "Emp_Rate_Q1_Low", "Emp_Rate_Q5_High"]].set_index("seed")
    df_ctrl = _final(bootstrap_df, "Control")[["seed", "Emp_Rate_Q1_Low", "Emp_Rate_Q5_High"]].set_index("seed")

    q1_gap = (df_ai["Emp_Rate_Q1_Low"]  - df_ctrl["Emp_Rate_Q1_Low"]).mean()
    q5_gap = (df_ai["Emp_Rate_Q5_High"] - df_ctrl["Emp_Rate_Q5_High"]).mean()

    # Q5 (highest exposure) should have the more negative gap
    assert q5_gap <= q1_gap, (
        f"Q5 gap ({q5_gap:.4f}) should be ≤ Q1 gap ({q1_gap:.4f}) — "
        "high-exposure quintile not disproportionately displaced"
    )


# ── Control scenario baseline realism ────────────────────────────────────────

def test_control_employment_stable(bootstrap_df):
    """
    Without AI disruption, the economy's employment rate should be near-stable.
    Drift > 5 pp over 60 months indicates that baseline turnover (δ_base) is
    mis-calibrated or the recall/rehire mechanism is broken.
    """
    ctrl = bootstrap_df[bootstrap_df["scenario"] == "Control"]
    t0_mean  = ctrl[ctrl["tick"] == 0]["Employment_Rate"].mean()
    t59_mean = ctrl[ctrl["tick"] == 59]["Employment_Rate"].mean()
    drift_pp = abs(t0_mean - t59_mean) * 100
    assert drift_pp <= 5.0, (
        f"Control employment drifted {drift_pp:.1f} pp over 60 ticks — "
        "expected ≤ 5 pp with calibrated baseline turnover"
    )


def test_control_vacancies_nonzero(bootstrap_df):
    """
    Normal monthly turnover (δ_base ≈ 0.5 %/month) creates replacement vacancies
    every tick in the control scenario.  Mean vacancies > 0 confirms the employer
    vacancy-posting logic is active.  Zero mean indicates the ε=1.0 replacement
    fix for control has not been applied.
    """
    ctrl = bootstrap_df[bootstrap_df["scenario"] == "Control"]
    # Exclude tick 0 (vacancies initialise to 0 before first step)
    mean_vac = ctrl[ctrl["tick"] > 0]["Total_Vacancies"].mean()
    assert mean_vac > 0, (
        f"Control Total_Vacancies mean={mean_vac:.1f} — employers not posting vacancies"
    )


def test_control_hired_matches_fired(bootstrap_df):
    """
    In the control scenario (no structural disruption), the hiring rate should
    closely track the firing rate.  A large persistent gap (> 50 % of fired)
    implies unemployed workers are not being recalled, accumulating hidden
    unemployment that would depress the employment rate.
    """
    ctrl = bootstrap_df[(bootstrap_df["scenario"] == "Control") & (bootstrap_df["tick"] > 0)]
    mean_fired  = ctrl["Total_Fired"].mean()
    mean_hired  = ctrl["Total_Hired"].mean()
    if mean_fired > 0:
        ratio = mean_hired / mean_fired
        assert ratio >= 0.50, (
            f"Control hire/fire ratio={ratio:.2f} — "
            "fewer than 50% of fired workers are being rehired"
        )


# ── Hiring mechanics ──────────────────────────────────────────────────────────

def test_ai_total_hired_nonzero(bootstrap_df):
    """
    The AI scenario posts thousands of vacancies (augmentation demand + direct
    replacement).  After the _market_clearing fix, some of those vacancies must
    be filled — Total_Hired > 0 at steady state.
    """
    ai = bootstrap_df[(bootstrap_df["scenario"] == "AI") & (bootstrap_df["tick"] > 5)]
    mean_hired = ai["Total_Hired"].mean()
    assert mean_hired > 0, (
        f"AI Total_Hired mean={mean_hired:.1f} over ticks 6-59 — "
        "market clearing is not matching workers to AI vacancies"
    )


# ── Retraining ────────────────────────────────────────────────────────────────

def test_retrained_share_plausible(bootstrap_df):
    """
    After 5 years of AI disruption, the share of workers who have retrained
    should be between 10 % and 70 %.  The lower bound reflects the scale of
    observed automation displacement; the upper bound ensures the model has not
    retrained the entire workforce (implausible given retraining cost barriers).
    """
    ai_final = _final(bootstrap_df, "AI")
    mean_share = ai_final["Retrained_Share"].mean()
    assert 0.10 <= mean_share <= 0.70, (
        f"Retrained_Share={mean_share:.2%} — outside plausible 10–70 % range"
    )


def test_retraining_count_higher_in_ai(bootstrap_df):
    """
    AI scenario must trigger more retraining events than Control (which has zero
    retraining by design — _choose_target_skill is only called when ai_active).
    """
    ai_retrain   = bootstrap_df[bootstrap_df["scenario"] == "AI"]["Retraining_Count"].mean()
    ctrl_retrain = bootstrap_df[bootstrap_df["scenario"] == "Control"]["Retraining_Count"].mean()
    assert ai_retrain > ctrl_retrain, (
        f"AI retraining ({ai_retrain:.1f}) not above Control ({ctrl_retrain:.1f})"
    )


# ── BTOS signal ───────────────────────────────────────────────────────────────

def test_btos_drifts_negative_under_ai(bootstrap_df):
    """
    The BTOS (Business Tendency Of Substitution) signal should drift downward
    in the AI scenario as employers respond to labour-market tightening and
    declining hiring intent.  A positive or flat BTOS under heavy AI disruption
    is inconsistent with the sector-drift calibration.
    """
    ai = bootstrap_df[bootstrap_df["scenario"] == "AI"]
    btos_t5  = ai[ai["tick"] == 5]["Avg_BTOS"].mean()
    btos_t59 = ai[ai["tick"] == 59]["Avg_BTOS"].mean()
    assert btos_t59 < btos_t5 + 0.005, (
        f"BTOS did not drift downward: T=5 {btos_t5:.5f} → T=59 {btos_t59:.5f}"
    )


# ── Cross-seed variance (CI width) ───────────────────────────────────────────

def test_employment_gap_has_meaningful_variance(bootstrap_df):
    """
    The β_run ~ N(1.0, 0.2) multiplier is meant to inject macroeconomic
    uncertainty across seeds.  The seed-to-seed standard deviation of the
    employment gap at T=59 should be at least 1 pp — if it is near-zero the
    CI collapses (probability saturation problem persists).
    """
    df_ai   = _final(bootstrap_df, "AI")[["seed", "Employment_Rate"]].set_index("seed")
    df_ctrl = _final(bootstrap_df, "Control")[["seed", "Employment_Rate"]].set_index("seed")
    gaps_pp = (df_ai["Employment_Rate"] - df_ctrl["Employment_Rate"]).values * 100
    std_pp  = gaps_pp.std()
    assert std_pp >= 1.0, (
        f"Employment-gap std={std_pp:.2f} pp — near-zero variance suggests "
        "probability saturation (CI collapse); check beta_run distribution"
    )
