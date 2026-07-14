"""Generate the RQ4 comparison figure: AI-exposure quintile vs hard-skill
proximity quintile, side by side.

Reads:
    output/wage_heterogeneity.parquet
        - group_type == "quintile"            -> existing LM_AIOE quintile
        - group_type == "hard_skill_quintile" -> new anchor-based quintile
    output/hard_skill_quintile_analysis.parquet
        - per-seed UR by hard_skill_quintile (50 seeds, final tick)
    output/industry_analysis.parquet (NOT used; UR-by-exposure is in paired_runs)
    output/paired_runs.parquet
        - Emp_Rate_Q* columns at final tick (100 seeds, existing axis)

Saves figure to output/figures/rq4_metric_comparison.png and prints the
key numerics: per-quintile UR delta and wage delta on each axis.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_FIG = ROOT / "output" / "figures" / "rq4_metric_comparison.png"


def _ur_delta_old_axis():
    """UR delta by LM_AIOE quintile from paired_runs.parquet (100 seeds, final tick)."""
    pr = pd.read_parquet(ROOT / "output" / "paired_runs.parquet")
    final = pr[pr["tick"] == pr["tick"].max()]
    ai  = final[final["scenario"] == "AI"]
    ctl = final[final["scenario"] == "Control"]
    cols = ["Emp_Rate_Q1_Low", "Emp_Rate_Q2", "Emp_Rate_Q3", "Emp_Rate_Q4", "Emp_Rate_Q5_High"]
    ai_mg  = ai[["seed"]  + cols].rename(columns={c: f"{c}_ai"  for c in cols})
    ctl_mg = ctl[["seed"] + cols].rename(columns={c: f"{c}_ctl" for c in cols})
    wide = ai_mg.merge(ctl_mg, on="seed")
    deltas = {}
    for c in cols:
        deltas[c] = ((1 - wide[f"{c}_ai"]) - (1 - wide[f"{c}_ctl"])) * 100
    return pd.DataFrame(deltas)


def _ur_delta_new_axis():
    """UR delta by hard_skill_quintile.

    Prefer the 100-seed paired_runs.parquet (HSQ columns are present once
    paired_bootstrap.py has been re-run with the new reporters); fall back
    to the 50-seed industry-analysis snapshot otherwise.
    """
    pr = pd.read_parquet(ROOT / "output" / "paired_runs.parquet")
    HSQ_COLS = ["Emp_Rate_HSQ1_Low", "Emp_Rate_HSQ2", "Emp_Rate_HSQ3",
                "Emp_Rate_HSQ4", "Emp_Rate_HSQ5_High"]
    if all(c in pr.columns for c in HSQ_COLS):
        final = pr[pr["tick"] == pr["tick"].max()]
        ai  = final[final["scenario"] == "AI"][["seed"]  + HSQ_COLS] \
                .rename(columns={c: f"{c}_ai"  for c in HSQ_COLS})
        ctl = final[final["scenario"] == "Control"][["seed"] + HSQ_COLS] \
                .rename(columns={c: f"{c}_ctl" for c in HSQ_COLS})
        wide = ai.merge(ctl, on="seed")
        rows = []
        for col in HSQ_COLS:
            label = col.replace("Emp_Rate_", "")
            for _, r in wide.iterrows():
                rows.append({
                    "seed": r["seed"],
                    "hard_skill_quintile": label,
                    "delta_ur": ((1 - r[f"{col}_ai"]) - (1 - r[f"{col}_ctl"])) * 100,
                })
        out = pd.DataFrame(rows)
        out.attrs["n_seeds"] = wide["seed"].nunique()
        out.attrs["source"]  = "paired_runs.parquet"
        return out

    # Fallback: 50-seed industry analysis snapshot
    h = pd.read_parquet(ROOT / "output" / "hard_skill_quintile_analysis.parquet")
    ai  = h[h["scenario"] == "AI"][["seed", "hard_skill_quintile", "ur"]] \
            .rename(columns={"ur": "ur_ai"})
    ctl = h[h["scenario"] == "Control"][["seed", "hard_skill_quintile", "ur"]] \
            .rename(columns={"ur": "ur_ctl"})
    wide = ai.merge(ctl, on=["seed", "hard_skill_quintile"])
    wide["delta_ur"] = (wide["ur_ai"] - wide["ur_ctl"]) * 100
    wide.attrs["n_seeds"] = wide["seed"].nunique()
    wide.attrs["source"]  = "hard_skill_quintile_analysis.parquet"
    return wide


def _wage_delta(group_type):
    wh = pd.read_parquet(ROOT / "output" / "wage_heterogeneity.parquet")
    sub = wh[wh["group_type"] == group_type]
    ai  = sub[sub["scenario"] == "AI"].groupby("group")["mean_wage"].mean()
    ctl = sub[sub["scenario"] == "Control"].groupby("group")["mean_wage"].mean()
    return (ai - ctl)


def main():
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    # Old axis (LM_AIOE)
    ur_old = _ur_delta_old_axis()
    wage_old = _wage_delta("quintile").reindex(["Q1_Low", "Q2", "Q3", "Q4", "Q5_High"])

    # New axis (hard-skill proximity)
    ur_new = _ur_delta_new_axis()
    HSQ_ORDER = ["HSQ1_Low", "HSQ2", "HSQ3", "HSQ4", "HSQ5_High"]
    ur_new_means = ur_new.groupby("hard_skill_quintile")["delta_ur"].mean().reindex(HSQ_ORDER)
    ur_new_p05   = ur_new.groupby("hard_skill_quintile")["delta_ur"].quantile(0.05).reindex(HSQ_ORDER)
    ur_new_p95   = ur_new.groupby("hard_skill_quintile")["delta_ur"].quantile(0.95).reindex(HSQ_ORDER)
    wage_new = _wage_delta("hard_skill_quintile").reindex(HSQ_ORDER)

    # ── Print numerics ────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("RQ4 — METRIC COMPARISON")
    print("="*72)

    print("\nLM_AIOE EXPOSURE QUINTILES (old axis):")
    print(f"  {'Quintile':<12} {'ΔUR (pp)':>10} {'ΔWage ($/p)':>14}")
    for q, lbl in zip(ur_old.columns, ["Q1_Low","Q2","Q3","Q4","Q5_High"]):
        ur_m = ur_old[q].mean()
        w    = wage_old.loc[lbl]
        print(f"  {lbl:<12} {ur_m:>+10.2f} {w:>+14.2f}")

    print("\nHARD-SKILL PROXIMITY QUINTILES (new axis):")
    print(f"  {'Quintile':<12} {'ΔUR (pp)':>10} {'ΔWage ($/p)':>14}")
    for q in HSQ_ORDER:
        print(f"  {q:<12} {ur_new_means[q]:>+10.2f} {wage_new[q]:>+14.2f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)

    OLD_LABELS = ["Q1\nLow", "Q2", "Q3", "Q4", "Q5\nHigh"]
    NEW_LABELS = ["HSQ1\nLow", "HSQ2", "HSQ3", "HSQ4", "HSQ5\nHigh"]
    OLD_COLORS = ["#1565C0","#1976D2","#9E9E9E","#E53935","#B71C1C"]
    NEW_COLORS = ["#1565C0","#1976D2","#9E9E9E","#E53935","#B71C1C"]

    # Top-left: UR delta — LM_AIOE
    ax = axes[0, 0]
    for i, c in enumerate(ur_old.columns):
        bp = ax.boxplot(ur_old[c].values, positions=[i+1], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=OLD_COLORS[i], alpha=0.5),
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(color=OLD_COLORS[i]),
                        capprops=dict(color=OLD_COLORS[i]),
                        flierprops=dict(marker="", alpha=0), whis=[5, 95])
    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(OLD_LABELS, fontsize=9)
    ax.set_ylabel("Δ Unemployment Rate (pp)", fontsize=9)
    ax.set_title("LM-AIOE Exposure Quintile\nΔ UR (100 seeds)", fontsize=10, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    # Top-right: UR delta — hard-skill proximity
    ax = axes[0, 1]
    for i, q in enumerate(HSQ_ORDER):
        vals = ur_new[ur_new["hard_skill_quintile"] == q]["delta_ur"].values
        bp = ax.boxplot(vals, positions=[i+1], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=NEW_COLORS[i], alpha=0.5),
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(color=NEW_COLORS[i]),
                        capprops=dict(color=NEW_COLORS[i]),
                        flierprops=dict(marker="", alpha=0), whis=[5, 95])
    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(NEW_LABELS, fontsize=9)
    ax.set_ylabel("Δ Unemployment Rate (pp)", fontsize=9)
    ax.set_title(f"Hard-Skill Proximity Quintile\nΔ UR ({ur_new.attrs.get('n_seeds', '?')} seeds)",
                 fontsize=10, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    # Bottom-left: Wage delta — LM_AIOE
    ax = axes[1, 0]
    bars = ax.bar(OLD_LABELS, wage_old.values, color=OLD_COLORS, alpha=0.75,
                   edgecolor="white", lw=0.5)
    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.5)
    ax.set_ylabel("Δ Mean Wage ($/period)", fontsize=9)
    ax.set_title("LM-AIOE Exposure Quintile\nΔ Wage (50 seeds)", fontsize=10, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    for bar, v in zip(bars, wage_old.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + (0.3 if v >= 0 else -0.5),
                f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    # Bottom-right: Wage delta — hard-skill proximity
    ax = axes[1, 1]
    bars = ax.bar(NEW_LABELS, wage_new.values, color=NEW_COLORS, alpha=0.75,
                   edgecolor="white", lw=0.5)
    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.5)
    ax.set_ylabel("Δ Mean Wage ($/period)", fontsize=9)
    ax.set_title("Hard-Skill Proximity Quintile\nΔ Wage (50 seeds)", fontsize=10, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    for bar, v in zip(bars, wage_new.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + (0.3 if v >= 0 else -0.5),
                f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    fig.suptitle("RQ4 — Comparison of Stratification Metrics:\n"
                 "LM-AIOE Exposure (left) vs. Hard-Skill Proximity (right)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {OUT_FIG}")


if __name__ == "__main__":
    main()
