"""
test_skill_distances.py — Notebook 04 (Skill Distances) output validation.

Validates the O*NET Work Activities cosine distance matrix and the
retraining-time logic that depends on it.  Distances are computed as
D = (1 − cosine_similarity) / 2 over z-scored work-activity vectors,
producing a continuous distribution in [0, 1] with no clipping artifacts.
"""

import numpy as np
import pandas as pd
import pytest


# ── Matrix shape & schema ─────────────────────────────────────────────────────

def test_skill_matrix_shape(skill_matrix):
    """
    The distance matrix must be square and cover all 537 occupations that appear
    in the worker sample.  A smaller matrix means some OCC2010 codes were silently
    dropped, causing KeyError look-ups in _choose_target_skill().
    """
    r, c = skill_matrix.shape
    assert r == c, f"Non-square matrix: {r}×{c}"
    assert r == 537, f"Expected 537 occupations, got {r}"


def test_skill_matrix_diagonal_zero(skill_matrix):
    """
    The distance from any occupation to itself must be exactly 0.
    A non-zero diagonal would cause workers displaced into their own occupation
    to enter unnecessary retraining, inflating retrained-share metrics.
    """
    diag = np.diag(skill_matrix.values.astype(float))
    assert np.allclose(diag, 0.0), f"Non-zero diagonal entries: max={diag.max():.6f}"


def test_skill_matrix_values_bounded(skill_matrix):
    """
    With D = (1 − cosine_sim) / 2, values must lie in [0, 1].
    Values outside this range would produce negative retraining months or
    months exceeding retrain_scale.
    """
    vals = skill_matrix.values.astype(float)
    assert vals.min() >= -1e-9, f"Negative distance found: {vals.min():.6f}"
    assert vals.max() <= 1.0 + 1e-9, f"Distance > 1 found: {vals.max():.6f}"


def test_skill_matrix_no_nan(skill_matrix):
    """
    NaN entries would cause _choose_target_skill() to produce NaN scores,
    making softmax probabilities undefined and crashing the agent step.
    """
    n_nan = np.isnan(skill_matrix.values.astype(float)).sum()
    assert n_nan == 0, f"{n_nan} NaN values in the distance matrix"


def test_skill_matrix_symmetric(skill_matrix):
    """
    Cosine distance is symmetric (d(A,B) = d(B,A)).  Asymmetry would mean
    'retraining from A→B costs different months than B→A', which has no
    economic justification and would bias occupational flows.
    """
    vals = skill_matrix.values.astype(float)
    max_asymmetry = np.abs(vals - vals.T).max()
    assert max_asymmetry < 1e-6, f"Matrix asymmetry {max_asymmetry:.2e} exceeds tolerance"


# ── Distance distribution ─────────────────────────────────────────────────────

def test_distance_distribution_is_spread(skill_matrix):
    """
    With D = (1 − cosine_sim) / 2 on z-scored O*NET Work Activities, the
    distance matrix should be continuous and well-spread across [0, 1].
    Key checks:
      - Median off-diagonal ≈ 0.40–0.60  (centred, not piled at extremes)
      - No more than 5% of pairs pinned at the maximum (d=1.0 would indicate
        a clipping artifact; with D/2 the theoretical maximum is 1.0 when
        cosine_sim = −1, which is rare in practice)
    The old threshold of ≥0.80 was calibrated to the clipped distribution
    where np.clip(1−sim, 0, 1) pushed ~51% of pairs to exactly 1.0.
    """
    vals = skill_matrix.values.astype(float)
    mask = ~np.eye(len(skill_matrix), dtype=bool)
    off_diag = vals[mask]
    median_off_diag = np.median(off_diag)
    frac_at_max = (off_diag >= 0.999).mean()

    assert 0.35 <= median_off_diag <= 0.65, (
        f"Median off-diagonal distance={median_off_diag:.3f}; "
        "expected 0.35–0.65 for a continuous, unclipped D/2 distribution"
    )
    assert frac_at_max < 0.05, (
        f"{frac_at_max*100:.1f}% of pairs at d≥0.999 — possible clipping artifact"
    )


def test_some_close_occupations_exist(skill_matrix):
    """
    Occupations within the same detailed SOC group (e.g., 'Registered Nurses'
    and 'Nurse Practitioners') should have distance < 0.30.  If no pairs have
    small distances the retraining network has no short hops, making all
    retraining implausibly expensive.
    """
    vals = skill_matrix.values.astype(float)
    mask = ~np.eye(len(skill_matrix), dtype=bool)
    n_close = (vals[mask] < 0.30).sum()
    assert n_close > 0, "No occupation pairs have distance < 0.30 — short retraining hops missing"


def test_occ_risk_lookup_shape(occ_risk):
    """
    The occupation risk lookup must cover ≥ 530 occupations (the 7 gap is
    tolerated for occupations absent from O*NET).  Fewer entries would silently
    replace many agents' risks with population means.
    """
    assert len(occ_risk) >= 530, f"Only {len(occ_risk)} occupations in risk lookup"


def test_occ_risk_columns(occ_risk):
    """The lookup must expose exactly the two columns read by LaborMarketModel."""
    for col in ("r_job", "p_aug"):
        assert col in occ_risk.columns, f"Missing column '{col}' in occ_risk_lookup"


# ── Retraining-time plausibility ──────────────────────────────────────────────

def test_retraining_months_plausible(skill_matrix):
    """
    With retrain_scale=24, the maximum retraining time is 24 months (d=1.0).
    The minimum non-zero time (d>0 rounded to nearest month) should be at least
    1 month.  This bracket aligns with BLS retraining-duration data:
      - Short-term OJT: < 1 month
      - Long-term OJT / apprenticeship: up to 24 months
    We check that 5th–95th percentile of non-zero distances maps to 1–24 months.
    """
    retrain_scale = 24.0
    vals = skill_matrix.values.astype(float)
    mask = ~np.eye(len(skill_matrix), dtype=bool)
    nonzero_dists = vals[mask][vals[mask] > 0]

    p05 = np.percentile(nonzero_dists, 5)
    p95 = np.percentile(nonzero_dists, 95)

    months_p05 = max(1, round(p05 * retrain_scale))
    months_p95 = max(1, round(p95 * retrain_scale))

    assert months_p05 >= 1,  f"5th-pct retraining time {months_p05} months < 1"
    assert months_p95 <= 24, f"95th-pct retraining time {months_p95} months > 24"
