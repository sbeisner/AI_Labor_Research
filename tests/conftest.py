"""
Shared pytest fixtures for the ai_labor_research test suite.

All fixtures resolve paths relative to the project root so tests can be
invoked from either `tests/` or the repository root.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR   = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"


# ── Parquet / CSV fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def worker_df():
    """50 k-row worker sample with risk scores attached (notebook 01 output)."""
    path = DATA_DIR / "worker_sample_with_risk.parquet"
    assert path.exists(), f"Missing: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def skill_matrix():
    """537×537 Wu-Palmer distance matrix (notebook 04 output)."""
    path = DATA_DIR / "skill_distance_matrix.parquet"
    assert path.exists(), f"Missing: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def occ_risk():
    """539-row occupation-level risk/augmentation lookup (notebook 01/05 output)."""
    path = DATA_DIR / "occ_risk_lookup.parquet"
    assert path.exists(), f"Missing: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def bootstrap_df():
    """6 000-row bootstrap run output (notebook 05 output)."""
    path = OUTPUT_DIR / "bootstrap_runs.parquet"
    assert path.exists(), f"Missing: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def bootstrap_ai(bootstrap_df):
    return bootstrap_df[bootstrap_df["scenario"] == "AI"].copy()


@pytest.fixture(scope="session")
def bootstrap_ctrl(bootstrap_df):
    return bootstrap_df[bootstrap_df["scenario"] == "Control"].copy()


# ── Mini live-model fixture ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mini_run():
    """
    Runs a paired (AI + Control) LaborMarketModel with 500 workers for 10 ticks.

    Returns dict with keys 'ai', 'ctrl', each a list of model-vars DataFrames
    collected after each tick.
    """
    import numpy as np
    from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS

    worker_path = DATA_DIR / "worker_sample_with_risk.parquet"
    dist_path   = DATA_DIR / "skill_distance_matrix.parquet"
    risk_path   = DATA_DIR / "occ_risk_lookup.parquet"

    full_df     = pd.read_parquet(worker_path)
    dist_matrix = pd.read_parquet(dist_path)
    occ_risk_df = pd.read_parquet(risk_path)

    rng = np.random.default_rng(42)
    sample = full_df.sample(n=500, random_state=42).copy()

    # Worker._choose_target_skill reads occ_risk["r_job"][occ_code], so the
    # lookup must be keyed by field name first, then by occupation code.
    occ_risk_lookup = {
        "r_job": {int(idx): float(row["r_job"]) for idx, row in occ_risk_df.iterrows()},
        "p_aug": {int(idx): float(row["p_aug"]) for idx, row in occ_risk_df.iterrows()},
    }

    results = {}
    for scenario, ai_flag in [("ai", True), ("ctrl", False)]:
        model = LaborMarketModel(
            worker_df=sample,
            ai_active=ai_flag,
            params=DEFAULT_PARAMS,
            seed=42,
            skill_distance_matrix=dist_matrix,
            occ_risk_lookup=occ_risk_lookup,
        )
        frames = []
        for _ in range(10):
            model.step()
            frames.append(model.datacollector.get_model_vars_dataframe().iloc[[-1]])
        results[scenario] = pd.concat(frames).reset_index(drop=True)

    return results
