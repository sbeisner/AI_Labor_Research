"""Compute every statistic quoted in tas_manuscript.qmd prose into output/stats.json.

Each entry records the value computed from the current output files, the value
currently quoted in the manuscript prose, and a tolerance. `make check-manuscript`
loads this file and fails loudly if any computed value has drifted from the
quoted value beyond tolerance — the prose–data lockstep that prevents a stale
number from surviving a data regeneration.

Sources:
  output/paired_runs.parquet        — aggregate, cohort, Beveridge, neo-gap
  output/wage_heterogeneity.parquet — survivor wage & cohort income deltas
  output/eta_sensitivity_v2.parquet — η robustness envelope
  output/dose_response.parquet      — adoption dose-response
  output/scaling_study.parquet      — finite-size scaling
  output/msm_posterior.csv          — MSM point estimate & moment fit
  output/msm_efficient.csv          — two-step efficient GMM, SEs, J-test

Run:  python scripts/extract_stats.py         # writes output/stats.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

HSQ = ["Emp_Rate_HSQ1_Low", "Emp_Rate_HSQ2", "Emp_Rate_HSQ3",
       "Emp_Rate_HSQ4", "Emp_Rate_HSQ5_High"]


def _entry(value, quoted, tol, where):
    """One tracked statistic: computed `value` vs prose-`quoted`, within `tol`."""
    return {"value": round(float(value), 4), "quoted": quoted, "tol": tol, "where": where}


def _terminal(pr):
    tmax = pr["tick"].max()
    f = pr[pr["tick"] == tmax]
    ai = f[f["scenario"] == "AI"].set_index("seed").sort_index()
    ct = f[f["scenario"] == "Control"].set_index("seed").sort_index()
    return ai, ct


def _ur_delta(ai, ct, col):
    return ((1 - ai[col]) - (1 - ct[col])) * 100


def _per_seed_beveridge(df):
    emp = df["Employment_Rate"] * 10_000
    df = df.assign(vacancy_rate=df["Total_Vacancies"] / (df["Total_Vacancies"] + emp))
    sl = []
    for _s, g in df.groupby("seed"):
        u = g["unemployment_rate"].values * 100
        v = g["vacancy_rate"].values * 100
        m = (u > 0) & (v > 0)
        if m.sum() >= 3:
            sl.append(np.polyfit(np.log(u[m]), np.log(v[m]), 1)[0])
    sl = np.asarray(sl)
    return sl.mean(), sl.std(ddof=1), int((sl < 0).sum()), int((sl >= 0).sum()), len(sl)


def _wh_delta(wh, gt, key):
    g = wh[wh["group_type"] == gt]
    a = g[g["scenario"] == "AI"].groupby("group")["mean_wage"].mean()
    c = g[g["scenario"] == "Control"].groupby("group")["mean_wage"].mean()
    return float((a - c).get(key, float("nan")))


def _hsq_order(ai, ct):
    d = [_ur_delta(ai, ct, c).mean() for c in HSQ]
    return list(np.argsort(np.argsort(-np.array(d))))


def main() -> int:
    pr = pd.read_parquet(OUT / "paired_runs.parquet")
    wh = pd.read_parquet(OUT / "wage_heterogeneity.parquet")
    ai, ct = _terminal(pr)
    s: dict = {}

    # ── Aggregate (§4.1, abstract) ─────────────────────────────────────────────
    dT = (ai["unemployment_rate"] - ct["unemployment_rate"]) * 100
    lo, hi = sstats.t.interval(0.95, len(dT) - 1, loc=dT.mean(), scale=dT.sem())
    s["agg_ur_delta_pp"] = _entry(dT.mean(), 10.90, 0.3, "§4.1")
    s["agg_ci_lo"] = _entry(lo, 9.9, 0.3, "§4.1/abstract")
    s["agg_ci_hi"] = _entry(hi, 11.9, 0.3, "§4.1/abstract")
    s["agg_seed_lo"] = _entry(np.percentile(dT, 2.5), 2.1, 0.6, "§4.1")
    s["agg_seed_hi"] = _entry(np.percentile(dT, 97.5), 19.7, 1.2, "§4.1")
    s["ai_terminal_ur_pct"] = _entry(ai["unemployment_rate"].mean() * 100, 15.0, 0.4, "§4.1")
    s["ctrl_terminal_ur_pct"] = _entry(ct["unemployment_rate"].mean() * 100, 4.1, 0.4, "§4.1")
    # tick-0 head start
    f0 = pr[pr["tick"] == 0]
    a0 = f0[f0.scenario == "AI"].set_index("seed").sort_index()["unemployment_rate"]
    c0 = f0[f0.scenario == "Control"].set_index("seed").sort_index()["unemployment_rate"]
    s["tick0_headstart_pp"] = _entry((a0 - c0).mean() * 100, 5.1, 0.4, "§4 preamble")

    # ── Beveridge per-seed slopes (§3.1, §4.1) ─────────────────────────────────
    cm, csd, cneg, _, cn = _per_seed_beveridge(pr[pr.scenario == "Control"])
    am, asd, _, anon, an = _per_seed_beveridge(pr[pr.scenario == "AI"])
    s["bev_ctrl_slope_mean"] = _entry(cm, -0.63, 0.03, "§3.1/§4.1")
    s["bev_ctrl_slope_sd"] = _entry(csd, 0.14, 0.03, "§3.1/§4.1")
    s["bev_ctrl_negative"] = _entry(cneg, 100, 0, "§3.1")
    s["bev_ai_slope_mean"] = _entry(am, -0.14, 0.03, "§4.1")
    s["bev_ai_slope_sd"] = _entry(asd, 0.13, 0.03, "§4.1")
    s["bev_ai_nonneg"] = _entry(anon, 14, 3, "§4.1")

    # ── Cohort deltas (§4.2, §4.3.1) ───────────────────────────────────────────
    quintile = {"Emp_Rate_Q1_Low": ("q1", 13.6), "Emp_Rate_Q2": ("q2", 13.0),
                "Emp_Rate_Q3": ("q3", 7.0), "Emp_Rate_Q4": ("q4", 11.6),
                "Emp_Rate_Q5_High": ("q5", 9.0)}
    for col, (k, q) in quintile.items():
        s[f"quintile_{k}_pp"] = _entry(_ur_delta(ai, ct, col).mean(), q, 0.6, "§4.2.1")
    s["entry_pp"] = _entry(_ur_delta(ai, ct, "Emp_Rate_Entry").mean(), 26.9, 0.8, "§4.2.2")
    s["senior_pp"] = _entry(_ur_delta(ai, ct, "Emp_Rate_Senior").mean(), 4.9, 0.5, "§4.2.2")
    gap = _ur_delta(ai, ct, "Emp_Rate_Entry") - _ur_delta(ai, ct, "Emp_Rate_Senior")
    s["entry_senior_gap_pp"] = _entry(gap.mean(), 22.0, 0.8, "§4.2.2")
    hsq = {"Emp_Rate_HSQ1_Low": ("hsq1", 11.6), "Emp_Rate_HSQ2": ("hsq2", 16.7),
           "Emp_Rate_HSQ3": ("hsq3", 5.2), "Emp_Rate_HSQ4": ("hsq4", 7.8),
           "Emp_Rate_HSQ5_High": ("hsq5", 13.3)}
    for col, (k, q) in hsq.items():
        s[f"{k}_pp"] = _entry(_ur_delta(ai, ct, col).mean(), q, 0.8, "§4.3.1")

    # ── Wage / income deltas ($K/yr) ───────────────────────────────────────────
    s["entry_income_delta"] = _entry(_wh_delta(wh, "seniority_income", "Entry-Level"),
                                     -28.8, 3.0, "§4.2.2")
    s["senior_income_delta"] = _entry(_wh_delta(wh, "seniority_income", "Senior"),
                                      -8.1, 2.0, "§4.2.2")
    s["hsq5_wage_delta"] = _entry(_wh_delta(wh, "hard_skill_quintile", "HSQ5_High"),
                                  -3.5, 1.5, "§4.3.1")
    s["hsq5_income_delta"] = _entry(_wh_delta(wh, "hard_skill_quintile_income", "HSQ5_High"),
                                    -25.3, 3.0, "§4.3.1")

    # ── Neo-gap (§4.3.2) ────────────────────────────────────────────────────────
    pr2 = pr.copy()
    pr2["NE_Cum"] = pr2.groupby(["scenario", "seed"])["New_Economy_Jobs"].cumsum()
    aiN = pr2[pr2.scenario == "AI"]
    posted = aiN.pivot(index="tick", columns="seed", values="NE_Cum").mean(axis=1).iloc[-1]
    fb_ai = pr2[pr2.scenario == "AI"].pivot(index="tick", columns="seed", values="Frontier_Basket_Employed")
    fb_ct = pr2[pr2.scenario == "Control"].pivot(index="tick", columns="seed", values="Frontier_Basket_Employed")
    filled = (fb_ai - fb_ct).mean(axis=1).iloc[-1]
    s["neo_posted"] = _entry(posted, 6700, 700, "§4.3.2")
    s["neo_fill_rate_pct"] = _entry(filled / posted * 100, 1.5, 0.6, "§4.3.2")

    # ── η robustness envelope (§4.3.3) ──────────────────────────────────────────
    eta_path = OUT / "eta_sensitivity_v2.parquet"
    if eta_path.exists():
        e = pd.read_parquet(eta_path)
        tmax = e["tick"].max()
        deltas, taus = [], []
        ref_order = None
        for ev in sorted(e["eta"].unique()):
            g = e[(e.eta == ev) & (e.tick == tmax)]
            ea = g[g.scenario == "AI"].set_index("seed").sort_index()
            ec = g[g.scenario == "Control"].set_index("seed").sort_index()
            deltas.append(((ea.unemployment_rate - ec.unemployment_rate) * 100).mean())
            order = _hsq_order(ea, ec)
            if abs(ev - 0.02) < 1e-9:
                ref_order = order
            taus.append(order)
        ref_order = ref_order or taus[0]
        tau_min = min(kendalltau(o, ref_order).correlation for o in taus)
        s["eta_delta_min_pp"] = _entry(min(deltas), 10.0, 0.6, "§4.3.3")
        s["eta_delta_max_pp"] = _entry(max(deltas), 12.1, 0.6, "§4.3.3")
        s["eta_hsq_tau_min"] = _entry(tau_min, 1.0, 0.05, "§4.3.3")

    # ── dose-response (§4.3.3) — computed inline (no model import) ──────────────
    dose_path = OUT / "dose_response.parquet"
    if dose_path.exists():
        d = pd.read_parquet(dose_path)
        tmax = d["tick"].max()
        doses = sorted(d["dose"].unique())
        deltas, orders = {}, {}
        for dv in doses:
            g = d[(d.dose == dv) & (d.tick == tmax)]
            da = g[g.scenario == "AI"].set_index("seed").sort_index()
            dc = g[g.scenario == "Control"].set_index("seed").sort_index()
            deltas[dv] = ((da.unemployment_rate - dc.unemployment_rate) * 100).mean()
            orders[dv] = _hsq_order(da, dc)
        pos = [(dv, deltas[dv]) for dv in doses if deltas[dv] > 0]
        elasticity = float(np.polyfit(np.log([p[0] for p in pos]),
                                      np.log([p[1] for p in pos]), 1)[0]) if len(pos) >= 2 else float("nan")
        ref = 1.0 if 1.0 in orders else doses[len(doses) // 2]
        tau_min = min(kendalltau(orders[dv], orders[ref]).correlation for dv in doses)
        s["dose_delta_min_pp"] = _entry(min(deltas.values()), 8.1, 0.6, "§4.3.3")
        s["dose_delta_max_pp"] = _entry(max(deltas.values()), 10.7, 0.6, "§4.3.3")
        s["dose_elasticity"] = _entry(elasticity, 0.22, 0.08, "§4.3.3")
        s["dose_hsq_tau_min"] = _entry(tau_min, 0.8, 0.15, "§4.3.3")
        # terminal mean adoption per velocity arm — evidence the velocity sweep saturates
        da_ajt = d[(d.tick == tmax) & (d.scenario == "AI")].groupby("dose")["Avg_A_jt"].mean()
        s["dose_ajt_min"] = _entry(da_ajt.min(), 0.92, 0.05, "§4.3.3")
        s["dose_ajt_max"] = _entry(da_ajt.max(), 1.00, 0.03, "§4.3.3")

    # ── scaling (§4.3.3) — computed inline (no model import) ────────────────────
    scale_path = OUT / "scaling_study.parquet"
    if scale_path.exists():
        sc = pd.read_parquet(scale_path)
        ref_order = _hsq_order(ai, ct)  # paired_runs N=10k reference
        tmax = sc["tick"].max()
        aggs, fills, taus = [], [], []
        for n in sorted(sc["N"].unique()):
            g = sc[(sc.N == n) & (sc.tick == tmax)]
            na = g[g.scenario == "AI"].set_index("seed").sort_index()
            nc = g[g.scenario == "Control"].set_index("seed").sort_index()
            aggs.append(((na.unemployment_rate - nc.unemployment_rate) * 100).mean())
            posted = na["New_Economy_Cumulative"].mean()
            filled = (na["Frontier_Basket_Employed"] - nc["Frontier_Basket_Employed"]).mean()
            fills.append(filled / posted * 100 if posted > 0 else float("nan"))
            taus.append(kendalltau(_hsq_order(na, nc), ref_order).correlation)
        s["scaling_delta_min_pp"] = _entry(min(aggs), 9.3, 0.6, "§4.3.3")
        s["scaling_delta_max_pp"] = _entry(max(aggs), 10.4, 0.6, "§4.3.3")
        s["scaling_fill_min_pct"] = _entry(min(fills), 1.3, 0.5, "§4.3.3")
        s["scaling_fill_max_pct"] = _entry(max(fills), 1.5, 0.5, "§4.3.3")
        s["scaling_hsq_tau_min"] = _entry(min(taus), 1.0, 0.05, "§4.3.3")

    # ── adoption-ceiling sweep (§4.3.3) ─────────────────────────────────────────
    ceil_path = OUT / "ceiling_sweep.parquet"
    if ceil_path.exists():
        cs = pd.read_parquet(ceil_path)
        tmax = cs["tick"].max()
        ceilings = sorted(cs["ceiling"].unique())
        cdelta, cajt, corder = {}, {}, {}
        for cv in ceilings:
            g = cs[(cs.ceiling == cv) & (cs.tick == tmax)]
            ca = g[g.scenario == "AI"].set_index("seed").sort_index()
            cc = g[g.scenario == "Control"].set_index("seed").sort_index()
            cdelta[cv] = ((ca.unemployment_rate - cc.unemployment_rate) * 100).mean()
            cajt[cv] = ca["Avg_A_jt"].mean()
            corder[cv] = _hsq_order(ca, cc)
        cref = 1.0 if 1.0 in corder else ceilings[-1]
        s["ceil_delta_c05_pp"] = _entry(cdelta[0.5], 4.2, 0.6, "§4.3.3")
        s["ceil_delta_c075_pp"] = _entry(cdelta[0.75], 5.6, 0.6, "§4.3.3")
        s["ceil_delta_c10_pp"] = _entry(cdelta[1.0], 11.5, 0.7, "§4.3.3")
        s["ceil_ajt_c05"] = _entry(cajt[0.5], 0.50, 0.05, "§4.3.3")
        s["ceil_ajt_c10"] = _entry(cajt[1.0], 1.00, 0.05, "§4.3.3")
        s["ceil_hsq_tau_c05"] = _entry(kendalltau(corder[0.5], corder[cref])[0], 0.8, 0.15, "§4.3.3")
        s["ceil_hsq_tau_c075"] = _entry(kendalltau(corder[0.75], corder[cref])[0], 1.0, 0.05, "§4.3.3")

    # ── MSM point estimate & fit (§2.3) ─────────────────────────────────────────
    msm = pd.read_csv(OUT / "msm_posterior.csv").iloc[0]
    s["msm_delta_pct"] = _entry(msm["delta_base"] * 100, 1.35, 0.05, "§2.3")
    s["msm_vac_pct"] = _entry(msm["vacancy_rate"] * 100, 4.46, 0.1, "§2.3")
    s["msm_sigma_pct"] = _entry(msm["btos_macro_std"] * 100, 1.55, 0.1, "§2.3")
    s["msm_sim_mean_pct"] = _entry(msm["sim_mean_ur"] * 100, 4.47, 0.15, "§2.3")
    s["msm_sim_std"] = _entry(msm["sim_std_ur"], 0.0153, 0.001, "§2.3")
    s["msm_sim_ac1"] = _entry(msm["sim_ac1_ur"], 0.982, 0.003, "§2.3")
    s["msm_sim_std_d"] = _entry(msm["sim_std_d_ur"], 0.00293, 0.0004, "§2.3")
    targets = {"sim_mean_ur": 0.0442, "sim_std_ur": 0.0181,
               "sim_ac1_ur": 0.9819, "sim_std_d_ur": 0.00160}
    loss = sum(((msm[k] - e) / e) ** 2 for k, e in targets.items())
    s["msm_unweighted_loss"] = _entry(loss, 0.71, 0.1, "§2.3")

    # ── MSM efficient two-step (§2.3) ────────────────────────────────────────────
    eff_path = OUT / "msm_efficient.csv"
    if eff_path.exists():
        eff = pd.read_csv(eff_path).set_index("parameter")
        s["msm2_delta"] = _entry(eff.loc["delta_base", "theta_step2_efficient"], 0.0134, 0.0008, "§2.3")
        s["msm2_vac"] = _entry(eff.loc["vacancy_rate", "theta_step2_efficient"], 0.0446, 0.001, "§2.3")
        s["msm2_sigma"] = _entry(eff.loc["btos_macro_std", "theta_step2_efficient"], 0.0126, 0.0015, "§2.3")
        s["msm2_se_delta"] = _entry(eff.loc["delta_base", "se_bootstrap"], 0.0030, 0.0008, "§2.3")
        s["msm2_se_vac"] = _entry(eff.loc["vacancy_rate", "se_bootstrap"], 0.0073, 0.0015, "§2.3")
        s["msm2_se_sigma"] = _entry(eff.loc["btos_macro_std", "se_bootstrap"], 0.0021, 0.0008, "§2.3")
        s["msm_j_stat"] = _entry(eff.iloc[0]["J_stat"], 87.1, 8.0, "§2.3")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(s, indent=2))
    print(f"[extract_stats] wrote {OUT / 'stats.json'} with {len(s)} tracked statistics")
    # Report any current drift (informational; the gate lives in check_manuscript.py)
    drift = [(k, v) for k, v in s.items() if abs(v["value"] - v["quoted"]) > v["tol"]]
    if drift:
        print(f"[extract_stats] WARNING: {len(drift)} value(s) drifted from quoted prose:")
        for k, v in drift:
            print(f"    {k}: computed {v['value']} vs quoted {v['quoted']} (tol {v['tol']}) [{v['where']}]")
    else:
        print("[extract_stats] all computed values match quoted prose within tolerance.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.exit(main())
