"""Build per-OCC2010 'routine hard skill' proximity score from the existing
O*NET Work Activities semantic graph.

Each occupation's score = 1 - mean cosine distance to a curated anchor set of
codified hard-skill cognitive occupations explicitly named (or named as
nearest analogues) in the H4 hypothesis: programming, technical writing,
accounting, paralegal, financial analysis, statistics, drafting.

Score range ~ [0, 1]; high = work-activity profile resembles canonical
hard-skill professions; low = manual/physical work distant from them.

Outputs:
    data/processed/hard_skill_scores.parquet  (OCC2010 -> hard_skill_score)
    data/processed/worker_sample_with_risk.parquet  (in-place: adds
        hard_skill_score, hard_skill_quintile columns)
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

PROCESSED = ROOT / "data" / "processed"

# ── Anchor set: occupations explicitly named in H4 + nearest codified-cognitive analogues
# Validated against the existing skill_distance_matrix: top-25 by score includes
# only computer/math/financial/scientific roles; bottom-15 is manual trades.
ANCHORS = {
    1010: "Computer programmers",
    2840: "Technical writers",
    1020: "Software developers",
    1030: "Web developers",
    1006: "Computer systems analysts",
    1060: "Database administrators",
    1050: "Computer support specialists",
    1007: "Information security analysts",
    800:  "Accountants and auditors",
    840:  "Financial analysts",
    820:  "Budget analysts",
    1230: "Statisticians",
    1200: "Actuaries",
    1540: "Drafters",
    2145: "Paralegals and legal assistants",
}


def build_hard_skill_scores(dist_matrix: pd.DataFrame) -> pd.Series:
    anchors = [a for a in ANCHORS if a in dist_matrix.index]
    missing = set(ANCHORS) - set(anchors)
    if missing:
        print(f"  WARN: {len(missing)} anchors missing from dist_matrix: {sorted(missing)}")
    print(f"  Anchors used: {len(anchors)}/{len(ANCHORS)}")

    scores = {}
    for occ in dist_matrix.index:
        a_set = [a for a in anchors if a != occ]  # exclude self if anchor
        scores[occ] = 1.0 - dist_matrix.loc[occ, a_set].mean()

    return pd.Series(scores, name="hard_skill_score")


def main():
    print("Loading skill_distance_matrix.parquet...")
    dist_matrix = pd.read_parquet(PROCESSED / "skill_distance_matrix.parquet")
    dist_matrix.index   = dist_matrix.index.astype(int)
    dist_matrix.columns = dist_matrix.columns.astype(int)
    print(f"  shape: {dist_matrix.shape}")

    print("\nComputing hard_skill_score for each OCC2010...")
    scores = build_hard_skill_scores(dist_matrix)
    print(f"  score distribution: min={scores.min():.3f}  median={scores.median():.3f}  max={scores.max():.3f}")

    # Save the lookup
    score_df = scores.reset_index().rename(columns={"index": "OCC2010"})
    score_df.to_parquet(PROCESSED / "hard_skill_scores.parquet", index=False)
    print(f"  saved -> {PROCESSED / 'hard_skill_scores.parquet'}")

    # Augment worker sample with score + quintile
    print("\nAugmenting worker_sample_with_risk.parquet...")
    ws_path = PROCESSED / "worker_sample_with_risk.parquet"
    ws = pd.read_parquet(ws_path)
    print(f"  worker sample shape (before): {ws.shape}")

    ws = ws.merge(score_df, on="OCC2010", how="left")
    n_unscored = ws["hard_skill_score"].isna().sum()
    if n_unscored:
        # Workers in occupations not in dist_matrix: impute with median
        # (preserves quintile balance; ~0.4% of sample expected)
        print(f"  {n_unscored} workers ({n_unscored/len(ws)*100:.1f}%) have no score; imputing with median")
        ws["hard_skill_score"] = ws["hard_skill_score"].fillna(scores.median())

    # Quintile-cut on the worker-sample-weighted distribution (not occupation-weighted)
    # so the quintiles are population-balanced.
    ws["hard_skill_quintile"] = pd.qcut(
        ws["hard_skill_score"], 5,
        labels=["HSQ1_Low", "HSQ2", "HSQ3", "HSQ4", "HSQ5_High"],
    ).astype(str)

    print(f"  worker sample shape (after):  {ws.shape}")
    print(f"\n  hard_skill_quintile counts:")
    print(ws["hard_skill_quintile"].value_counts().sort_index().to_string())

    # Cross-tab against existing exposure_quintile to show they're related but distinct
    ct = pd.crosstab(ws["hard_skill_quintile"], ws["exposure_quintile"])
    print(f"\n  hard_skill_quintile × exposure_quintile cross-tab:")
    print(ct.to_string())

    ws.to_parquet(ws_path, index=False)
    print(f"\n  saved -> {ws_path}")


if __name__ == "__main__":
    main()
