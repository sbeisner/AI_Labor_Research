"""
test_data_quality.py — Notebook 01 (Data Preparation) output validation.

Checks that the worker sample has the expected shape, column completeness,
and distributional properties anchored to BLS / CPS benchmarks.
"""

import numpy as np
import pandas as pd
import pytest

REQUIRED_COLS = [
    "CPSIDP", "AGE", "SEX", "RACE", "EDUC", "EMPSTAT",
    "OCC2010", "IND1990", "NAICS_sector",
    "AIOE", "LM_AIOE", "Industry_AIIE", "Industry_LM_AIIE",
    "exposure_quintile", "age_group",
    "r_job", "p_aug", "h_job", "exp_norm", "wage", "is_employed",
]


# ── Schema ────────────────────────────────────────────────────────────────────

def test_worker_df_row_count(worker_df):
    """
    The worker sample should contain exactly 10 000 observations.
    The base CPS-derived sample is 10 000 workers; the bootstrap replicates
    this via seeded re-sampling rather than storing a pre-expanded 50 K file.
    """
    assert len(worker_df) == 10_000, f"Expected 10 000 rows, got {len(worker_df)}"


def test_worker_df_required_columns(worker_df):
    """
    Every downstream notebook depends on this column set.  Missing columns here
    will silently propagate wrong results through the entire pipeline.
    """
    missing = [c for c in REQUIRED_COLS if c not in worker_df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_duplicate_cpsidp_within_expected_bounds(worker_df):
    """
    CPS is a rotational panel: the same person appears in up to 8 consecutive
    monthly interviews.  Some CPSIDP duplication is therefore expected and
    intentional (cross-month observations).  However, if >20% of records share
    a CPSIDP it suggests over-sampling from a single rotation group, which
    would bias the age/wage distribution of the agent population.
    """
    dup_rate = worker_df["CPSIDP"].duplicated().mean()
    assert dup_rate <= 0.20, (
        f"{dup_rate:.1%} of CPSIDP values are duplicates — "
        "exceeds the 20% threshold for CPS rotation-panel re-appearance"
    )


# ── Risk-score ranges ─────────────────────────────────────────────────────────

def test_r_job_range(worker_df):
    """
    Substitution risk r_job is a calibrated logistic output bounded to [0, 1].
    Empirically the range observed from O*NET + Felten data is roughly 0.25–0.60.
    Values outside [0.20, 0.70] indicate a calibration error.
    """
    lo, hi = worker_df["r_job"].min(), worker_df["r_job"].max()
    assert lo >= 0.20, f"r_job min too low: {lo:.4f}"
    assert hi <= 0.70, f"r_job max too high: {hi:.4f}"


def test_p_aug_range(worker_df):
    """
    Augmentation potential p_aug must stay within the logistic [0, 1] range.
    The O*NET task-scoring pipeline should never produce values outside [0.25, 1.0].
    """
    lo, hi = worker_df["p_aug"].min(), worker_df["p_aug"].max()
    assert lo >= 0.25, f"p_aug min too low: {lo:.4f}"
    assert hi <= 1.00, f"p_aug max too high: {hi:.4f}"


def test_r_job_p_aug_no_nulls(worker_df):
    """
    Null risk scores would propagate into NaN displacement probabilities,
    silently zeroing out displacement for affected agents.
    """
    assert worker_df["r_job"].isna().sum() == 0, "NaN values in r_job"
    assert worker_df["p_aug"].isna().sum() == 0, "NaN values in p_aug"


def test_r_job_mean_plausible(worker_df):
    """
    Mean substitution risk should be in the moderate range (0.35–0.50).
    Felten et al. find the cross-occupation average AIOE is near the midpoint
    of the distribution.  A mean outside this range suggests an indexing error.
    """
    mean_r = worker_df["r_job"].mean()
    assert 0.35 <= mean_r <= 0.50, f"Mean r_job={mean_r:.4f} outside expected [0.35, 0.50]"


def test_p_aug_mean_plausible(worker_df):
    """
    Mean augmentation potential should be above 0.5 — the majority of occupations
    have at least some tasks that AI can assist with rather than fully replace.
    """
    mean_p = worker_df["p_aug"].mean()
    assert 0.50 <= mean_p <= 0.90, f"Mean p_aug={mean_p:.4f} outside expected [0.50, 0.90]"


# ── Experience normalisation ──────────────────────────────────────────────────

def test_exp_norm_non_negative(worker_df):
    """
    exp_norm encodes relative labour-market experience (0 = entry-level,
    higher = more senior).  Negative values are economically meaningless and
    indicate a data-processing error in the normalisation step.
    """
    bad = (worker_df["exp_norm"].dropna() < 0).sum()
    assert bad == 0, f"{bad} workers have negative exp_norm"


def test_exp_norm_coverage(worker_df):
    """
    At least 95 % of workers must have a non-null exp_norm.  A large null share
    would mean the experience protection mechanism is disabled for most agents.
    """
    coverage = worker_df["exp_norm"].notna().mean()
    assert coverage >= 0.95, f"exp_norm coverage only {coverage:.1%}"


# ── Wage distribution ─────────────────────────────────────────────────────────

def test_wage_positive(worker_df):
    """
    All wages must be strictly positive.  The model's wage-update mechanism
    uses multiplicative boosts that break for zero or negative base wages.
    """
    assert (worker_df["wage"] > 0).all(), "Non-positive wages found"


def test_wage_median_plausible(worker_df):
    """
    Median annual wage should be in the $30 000–$60 000 range, consistent with
    the BLS Occupational Employment and Wage Statistics (OEWS) median for 2022
    of ~$45 000.
    """
    med = worker_df["wage"].median()
    assert 30_000 <= med <= 60_000, f"Median wage ${med:,.0f} outside BLS-plausible range"


def test_wage_has_dispersion(worker_df):
    """
    Wage inequality is a key model input.  The coefficient of variation (std/mean)
    should exceed 0.5 — U.S. wage distributions are highly right-skewed (Gini ~0.47).
    A flat distribution would suppress quintile heterogeneity in the bootstrap.
    """
    cv = worker_df["wage"].std() / worker_df["wage"].mean()
    assert cv >= 0.50, f"Wage CoV={cv:.2f} — insufficient dispersion"


# ── AI exposure coverage ──────────────────────────────────────────────────────

def test_aioe_coverage(worker_df):
    """
    Occupation-level AIOE should cover ≥ 95 % of workers.  Large gaps mean
    workers default to population-mean risk scores, blurring distributional results.
    """
    coverage = worker_df["AIOE"].notna().mean()
    assert coverage >= 0.95, f"AIOE coverage {coverage:.1%} below 95 %"


def test_five_exposure_quintiles_present(worker_df):
    """
    The quintile split must produce exactly five non-empty groups so that
    per-quintile employment-rate metrics in notebooks 02 and 05 are well-defined.
    """
    n_quintiles = worker_df["exposure_quintile"].nunique()
    assert n_quintiles == 5, f"Expected 5 exposure quintiles, found {n_quintiles}"
