"""Industry-level impact analysis: unemployment rate and wage changes under AI shock.

Runs N_SEEDS paired (AI + Control) seeds and, at the final analysis tick,
queries all active workers to compute per-NAICS-sector unemployment rate and
mean wage. Saves aggregated results to output/industry_analysis.parquet.

This feeds the RQ1 hypothesis-conclusions figure showing the biggest industry
winners and losers in terms of unemployment and wages.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import multiprocessing as mp
import numpy as np
import pandas as pd

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS

N_SEEDS   = 50
N_TICKS   = 180
BURN_IN   = 60
OUT_PATH      = ROOT / "output" / "industry_analysis.parquet"
OUT_PATH_JZ   = ROOT / "output" / "job_zone_analysis.parquet"
OUT_PATH_WAGE = ROOT / "output" / "wage_heterogeneity.parquet"
OUT_PATH_HSQ  = ROOT / "output" / "hard_skill_quintile_analysis.parquet"
FORCE_RERUN   = True

NAICS_NAMES = {
    "11": "Agriculture / Forestry",
    "21": "Mining & Oil/Gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation",
    "49": "Transportation",
    "51": "Information / Tech",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional Services",
    "55": "Management",
    "56": "Administrative Svcs",
    "61": "Education",
    "62": "Health Care",
    "71": "Arts & Entertainment",
    "72": "Accommodation / Food",
    "81": "Other Services",
    "92": "Public Administration",
}

# Globals populated by pool initializer
_worker_df   = None
_dist_matrix = None
_occ_risk    = None


def _worker_init():
    global _worker_df, _dist_matrix, _occ_risk
    from scripts.bootstrap_runner import load_shared_data
    _worker_df, _dist_matrix, _occ_risk = load_shared_data()


def _snapshot_industry(model) -> pd.DataFrame:
    """Query all active (non-retired) workers and return per-worker records."""
    from agents.Worker import WorkerAgent
    rows = []
    for w in model.agents_by_type[WorkerAgent]:
        if w.is_retired:
            continue
        rows.append({
            "naics": str(getattr(w, "naics_sector", "99"))[:2],
            "is_employed": int(w.is_employed and not w.is_olf),
            "is_olf":      int(w.is_olf),
            "wage":        float(w.wage) if w.is_employed else np.nan,
        })
    return pd.DataFrame(rows)


def _snapshot_job_zone(model) -> pd.DataFrame:
    """Query all active (non-retired) workers; return per-worker job-zone records."""
    from agents.Worker import WorkerAgent
    rows = []
    for w in model.agents_by_type[WorkerAgent]:
        if w.is_retired:
            continue
        rows.append({
            "job_zone":    int(getattr(w, "job_zone", 0)),
            "is_employed": int(w.is_employed and not w.is_olf),
            "is_olf":      int(w.is_olf),
        })
    return pd.DataFrame(rows)


def _snapshot_wages(model) -> pd.DataFrame:
    """Per-worker snapshot capturing job_zone, exposure_quintile, hard_skill_quintile,
    exp_norm, and wage."""
    from agents.Worker import WorkerAgent
    rows = []
    for w in model.agents_by_type[WorkerAgent]:
        if w.is_retired:
            continue
        rows.append({
            "job_zone":             int(getattr(w, "job_zone", 0)),
            "exposure_quintile":    str(getattr(w, "exposure_quintile", "Unknown")),
            "hard_skill_quintile":  str(getattr(w, "hard_skill_quintile", "Unknown")),
            "exp_norm":             float(getattr(w, "exp_norm", 0.0)),
            "naics":                str(getattr(w, "naics_sector", "99"))[:2],
            "occ":                  int(getattr(w, "current_occ", 0)),
            "is_employed":          int(w.is_employed and not w.is_olf),
            "is_olf":               int(w.is_olf),
            "wage":                 float(w.wage) if (w.is_employed and not w.is_olf) else np.nan,
        })
    return pd.DataFrame(rows)


def run_seed(seed: int) -> tuple:
    """Returns (industry_df, job_zone_df, wage_df) for this seed."""
    rng = np.random.default_rng(seed)
    sampled_df = _worker_df.sample(
        n=len(_worker_df), replace=True,
        random_state=int(rng.integers(0, 2**31)),
    ).reset_index(drop=True)

    ind_results  = []
    jz_results   = []
    wage_results = []
    for scenario, ai_active in (("AI", True), ("Control", False)):
        m = LaborMarketModel(
            sampled_df,
            params=DEFAULT_PARAMS,
            ai_active=ai_active,
            seed=seed,
            skill_distance_matrix=_dist_matrix,
            occ_risk_lookup=_occ_risk,
            collect_agent_data=False,
        )
        for t in range(N_TICKS):
            m.step()

        snap_ind = _snapshot_industry(m)
        snap_ind["scenario"] = scenario
        snap_ind["seed"]     = seed
        ind_results.append(snap_ind)

        snap_jz = _snapshot_job_zone(m)
        snap_jz["scenario"] = scenario
        snap_jz["seed"]     = seed
        jz_results.append(snap_jz)

        snap_wage = _snapshot_wages(m)
        snap_wage["scenario"] = scenario
        snap_wage["seed"]     = seed
        wage_results.append(snap_wage)

    return (
        pd.concat(ind_results,  ignore_index=True),
        pd.concat(jz_results,   ignore_index=True),
        pd.concat(wage_results, ignore_index=True),
    )


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--n-seeds", type=int, default=N_SEEDS,
                   help=f"number of paired seeds (default: {N_SEEDS})")
    p.add_argument("--smoke", action="store_true",
                   help="fast schema check: 2 seeds → *_smoke.parquet outputs")
    args = p.parse_args()

    if args.smoke:
        n_seeds = 2
        out_path    = OUT_PATH.with_name("industry_analysis_smoke.parquet")
        out_path_jz = OUT_PATH_JZ.with_name("job_zone_analysis_smoke.parquet")
        out_path_wage = OUT_PATH_WAGE.with_name("wage_heterogeneity_smoke.parquet")
        out_path_hsq  = OUT_PATH_HSQ.with_name("hard_skill_quintile_analysis_smoke.parquet")
    else:
        n_seeds = args.n_seeds
        out_path, out_path_jz = OUT_PATH, OUT_PATH_JZ
        out_path_wage, out_path_hsq = OUT_PATH_WAGE, OUT_PATH_HSQ

    print(f"[industry_analysis] Running {n_seeds} seeds × 2 scenarios …", flush=True)
    n_workers = min(mp.cpu_count(), 8)
    with mp.Pool(processes=n_workers, initializer=_worker_init) as pool:
        ind_frames  = []
        jz_frames   = []
        wage_frames = []
        for i, (ind_df, jz_df, wage_df) in enumerate(
            pool.imap_unordered(run_seed, range(n_seeds)), start=1
        ):
            ind_frames.append(ind_df)
            jz_frames.append(jz_df)
            wage_frames.append(wage_df)
            if i % 5 == 0 or i == n_seeds:
                print(f"  {i}/{n_seeds} seeds done", flush=True)

    if True:
        # ── Industry summary ──────────────────────────────────────────────────
        all_ind = pd.concat(ind_frames, ignore_index=True)
        all_ind["industry"] = all_ind["naics"].map(NAICS_NAMES).fillna("Other")

        def _agg(g):
            active     = (~g["is_olf"].astype(bool))
            employed   = g["is_employed"].astype(bool) & active
            unemployed = (~g["is_employed"].astype(bool)) & active
            n_active   = active.sum()
            ur         = unemployed.sum() / n_active if n_active > 0 else np.nan
            mean_wage  = g.loc[employed, "wage"].mean()
            return pd.Series({"ur": ur, "mean_wage": mean_wage, "n_workers": int(n_active)})

        ind_summary = (
            all_ind.groupby(["scenario", "industry", "seed"])
            .apply(_agg, include_groups=False)
            .reset_index()
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ind_summary.to_parquet(out_path, index=False)
        print(f"[industry_analysis] Industry data saved to {out_path}", flush=True)

        # ── Job zone summary ──────────────────────────────────────────────────
        all_jz = pd.concat(jz_frames, ignore_index=True)
        # Drop job_zone == 0 (unmapped workers)
        all_jz = all_jz[all_jz["job_zone"] > 0].copy()

        def _agg_jz(g):
            active     = (~g["is_olf"].astype(bool))
            employed   = g["is_employed"].astype(bool) & active
            unemployed = (~g["is_employed"].astype(bool)) & active
            n_active   = active.sum()
            ur         = unemployed.sum() / n_active if n_active > 0 else np.nan
            return pd.Series({"ur": ur, "n_workers": int(n_active)})

        jz_summary = (
            all_jz.groupby(["scenario", "job_zone", "seed"])
            .apply(_agg_jz, include_groups=False)
            .reset_index()
        )
        jz_summary.to_parquet(out_path_jz, index=False)
        print(f"[industry_analysis] Job zone data saved to {out_path_jz}", flush=True)
        print(jz_summary.groupby(["scenario", "job_zone"])["ur"].mean().round(4))

        # ── Hard-skill quintile UR summary (RQ4 alternative metric) ──────────
        # Uses the same per-worker wage snapshot, so we reuse wage_frames before
        # filtering to job_zone>0. Active = not OLF; UR = unemployed/active.
        all_hsq = pd.concat(wage_frames, ignore_index=True)

        def _agg_hsq(g):
            active     = (~g["is_olf"].astype(bool))
            employed   = g["is_employed"].astype(bool) & active
            unemployed = (~g["is_employed"].astype(bool)) & active
            n_active   = active.sum()
            ur         = unemployed.sum() / n_active if n_active > 0 else np.nan
            return pd.Series({"ur": ur, "n_workers": int(n_active)})

        hsq_summary = (
            all_hsq.groupby(["scenario", "hard_skill_quintile", "seed"])
            .apply(_agg_hsq, include_groups=False)
            .reset_index()
        )
        hsq_summary.to_parquet(out_path_hsq, index=False)
        print(f"[industry_analysis] Hard-skill quintile data saved to {out_path_hsq}", flush=True)
        print(hsq_summary.groupby(["scenario", "hard_skill_quintile"])["ur"].mean().round(4))

        # ── Wage heterogeneity (per-worker wages by zone/quintile/seniority) ──
        all_wage = pd.concat(wage_frames, ignore_index=True)
        all_wage = all_wage[all_wage["job_zone"] > 0].copy()

        # Seniority bucket: entry (exp_norm ≤ 0.2), senior (exp_norm ≥ 0.8)
        def _seniority(v):
            if v <= 0.2:
                return "Entry-Level"
            elif v >= 0.8:
                return "Senior"
            return "Mid"
        all_wage["seniority"] = all_wage["exp_norm"].map(_seniority)

        # Mean wage of employed workers per group, per seed, per scenario
        wage_by_jz = (
            all_wage[all_wage["is_employed"] == 1]
            .groupby(["scenario", "seed", "job_zone"])["wage"]
            .mean().reset_index().rename(columns={"wage": "mean_wage"})
        )
        wage_by_q = (
            all_wage[all_wage["is_employed"] == 1]
            .groupby(["scenario", "seed", "exposure_quintile"])["wage"]
            .mean().reset_index().rename(columns={"wage": "mean_wage"})
        )
        wage_by_hsq = (
            all_wage[all_wage["is_employed"] == 1]
            .groupby(["scenario", "seed", "hard_skill_quintile"])["wage"]
            .mean().reset_index().rename(columns={"wage": "mean_wage"})
        )
        wage_by_sen = (
            all_wage[(all_wage["is_employed"] == 1) & (all_wage["seniority"] != "Mid")]
            .groupby(["scenario", "seed", "seniority"])["wage"]
            .mean().reset_index().rename(columns={"wage": "mean_wage"})
        )
        wage_by_jz["group_type"]  = "job_zone"
        wage_by_q["group_type"]   = "quintile"
        wage_by_hsq["group_type"] = "hard_skill_quintile"
        wage_by_sen["group_type"] = "seniority"
        wage_by_jz  = wage_by_jz.rename(columns={"job_zone":            "group"})
        wage_by_q   = wage_by_q.rename( columns={"exposure_quintile":   "group"})
        wage_by_hsq = wage_by_hsq.rename(columns={"hard_skill_quintile": "group"})
        wage_by_sen = wage_by_sen.rename(columns={"seniority":           "group"})

        # ── Survivor-bias-corrected: mean INCOME by HSQ and Seniority ────────
        # Includes $0 for unemployed and OLF workers in the cohort, giving
        # the true cohort average rather than the mean of survivors only.
        # The gap between mean_wage (above) and mean_income (here) directly
        # quantifies how much of any apparent wage "gain" is composition shift.
        all_wage_income = all_wage.copy()
        all_wage_income["income"] = all_wage_income["wage"].fillna(0.0)
        income_by_hsq = (
            all_wage_income
            .groupby(["scenario", "seed", "hard_skill_quintile"])["income"]
            .mean().reset_index().rename(columns={"income": "mean_wage"})
        )
        income_by_hsq["group_type"] = "hard_skill_quintile_income"
        income_by_hsq = income_by_hsq.rename(columns={"hard_skill_quintile": "group"})

        income_by_sen = (
            all_wage_income[all_wage_income["seniority"] != "Mid"]
            .groupby(["scenario", "seed", "seniority"])["income"]
            .mean().reset_index().rename(columns={"income": "mean_wage"})
        )
        income_by_sen["group_type"] = "seniority_income"
        income_by_sen = income_by_sen.rename(columns={"seniority": "group"})

        # Within-occupation entry/senior wage normalization (RQ3 reviewer fix).
        # For each (scenario, seed, occ): mean wage of employed entry vs. senior workers.
        # Then for each (scenario, seed, seniority): macro-mean across occupations that
        # contain BOTH entry and senior employed workers in that seed×scenario cell —
        # so each occupation contributes equally regardless of headcount, removing the
        # occupation-mix confound that makes pooled means show entry > senior.
        _emp_sen = all_wage[
            (all_wage["is_employed"] == 1) &
            (all_wage["seniority"] != "Mid") &
            (all_wage["occ"] > 0)
        ]
        _occ_means = (
            _emp_sen.groupby(["scenario", "seed", "occ", "seniority"])["wage"]
            .mean().unstack("seniority")
        )
        _occ_means = _occ_means.dropna(subset=["Entry-Level", "Senior"]).reset_index()
        _occ_means_long = _occ_means.melt(
            id_vars=["scenario", "seed", "occ"],
            value_vars=["Entry-Level", "Senior"],
            var_name="seniority", value_name="occ_mean_wage",
        )
        wage_by_occ_sen = (
            _occ_means_long.groupby(["scenario", "seed", "seniority"])["occ_mean_wage"]
            .mean().reset_index().rename(columns={"occ_mean_wage": "mean_wage"})
        )
        wage_by_occ_sen["group_type"] = "occ_seniority"
        wage_by_occ_sen = wage_by_occ_sen.rename(columns={"seniority": "group"})

        # Industry × seniority: mean wage per (industry, seniority, scenario, seed)
        all_wage["industry"] = all_wage["naics"].map(NAICS_NAMES).fillna("Other")
        wage_by_ind_sen = (
            all_wage[
                (all_wage["is_employed"] == 1) &
                (all_wage["seniority"] != "Mid")
            ]
            .groupby(["scenario", "seed", "industry", "seniority"])["wage"]
            .mean().reset_index().rename(columns={"wage": "mean_wage"})
        )
        wage_by_ind_sen["group_type"] = "industry_seniority"
        wage_by_ind_sen["group"]      = wage_by_ind_sen["industry"] + "|" + wage_by_ind_sen["seniority"]
        wage_by_ind_sen = wage_by_ind_sen.drop(columns=["industry", "seniority"])

        wage_summary = pd.concat(
            [wage_by_jz, wage_by_q, wage_by_hsq, wage_by_sen,
             wage_by_ind_sen, income_by_hsq, income_by_sen, wage_by_occ_sen],
            ignore_index=True,
        )
        wage_summary["group"] = wage_summary["group"].astype(str)
        wage_summary.to_parquet(out_path_wage, index=False)
        print(f"[industry_analysis] Wage heterogeneity data saved to {out_path_wage}", flush=True)
        print(wage_summary.groupby(["group_type","scenario","group"])["mean_wage"].mean().round(2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
