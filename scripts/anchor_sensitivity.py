"""Leave-one-out sensitivity analysis for the hard-skill anchor set.

For each anchor, recompute the per-OCC2010 score with that anchor removed,
re-quintile the worker sample, and report (a) the rank correlation of
per-occupation scores against the full-anchor baseline, and (b) the share
of workers whose hard_skill_quintile shifts. Stable findings = score and
quintile assignment do not depend on any single anchor.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.build_hard_skill_scores import ANCHORS, build_hard_skill_scores

PROCESSED = ROOT / "data" / "processed"


def main():
    dm = pd.read_parquet(PROCESSED / "skill_distance_matrix.parquet")
    dm.index   = dm.index.astype(int)
    dm.columns = dm.columns.astype(int)

    ws = pd.read_parquet(PROCESSED / "worker_sample_with_risk.parquet")

    # Baseline: full anchor set
    baseline = build_hard_skill_scores(dm)
    base_q = pd.qcut(
        ws["OCC2010"].map(baseline).fillna(baseline.median()), 5,
        labels=["HSQ1_Low", "HSQ2", "HSQ3", "HSQ4", "HSQ5_High"],
    ).astype(str)

    print("\n" + "="*80)
    print("ANCHOR LEAVE-ONE-OUT SENSITIVITY")
    print("="*80)
    print(f"{'Removed Anchor':<40} {'Spearman ρ':>12} {'%Q-shift':>10} {'%top/bot-shift':>16}")
    print("-"*80)

    rows = []
    for omit_occ, omit_name in ANCHORS.items():
        kept = {k: v for k, v in ANCHORS.items() if k != omit_occ}
        # Recompute with reduced anchors
        anchors = [a for a in kept if a in dm.index]
        scores = pd.Series(
            {o: 1.0 - dm.loc[o, [a for a in anchors if a != o]].mean()
             for o in dm.index},
            name="hard_skill_score",
        )
        # Rank correlation vs baseline
        rho, _ = spearmanr(baseline.values, scores.reindex(baseline.index).values)
        # Quintile-shift rate at worker level
        ws_score = ws["OCC2010"].map(scores).fillna(scores.median())
        new_q = pd.qcut(
            ws_score, 5,
            labels=["HSQ1_Low", "HSQ2", "HSQ3", "HSQ4", "HSQ5_High"],
        ).astype(str)
        any_shift = (new_q != base_q).mean() * 100
        # Top-or-bottom shift rate (the cohorts that drive the RQ4 finding)
        in_top_or_bot = base_q.isin(["HSQ1_Low", "HSQ4", "HSQ5_High"])
        topbot_shift = ((new_q != base_q) & in_top_or_bot).sum() / in_top_or_bot.sum() * 100

        print(f"{f'{omit_name} (OCC {omit_occ})':<40} {rho:>12.4f} {any_shift:>9.1f}% {topbot_shift:>15.1f}%")
        rows.append({
            "removed": omit_name, "occ": omit_occ,
            "spearman": rho, "pct_any_shift": any_shift,
            "pct_topbot_shift": topbot_shift,
        })

    sens = pd.DataFrame(rows)
    print("\nSummary:")
    print(f"  Min Spearman ρ vs baseline:  {sens['spearman'].min():.4f}")
    print(f"  Max worker-quintile shift:   {sens['pct_any_shift'].max():.1f}%")
    print(f"  Max top/bot-quintile shift:  {sens['pct_topbot_shift'].max():.1f}%")

    sens.to_csv(ROOT / "output" / "anchor_sensitivity.csv", index=False)
    print(f"\nSaved -> output/anchor_sensitivity.csv")


if __name__ == "__main__":
    main()
