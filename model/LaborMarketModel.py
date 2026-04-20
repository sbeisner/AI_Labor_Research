"""Labor Market Agent-Based Model.

Orchestrates WorkerAgent instances sampled from IPUMS CPS microdata and
steps through time (ticks = months) while tracking employment, wages, and
displacement dynamics under control vs. AI adoption scenarios.

Key mechanisms:
  - Employer-driven displacement: logistic hazard P(D) = sigmoid(logit(δ) + β1*A*R - β2*A*P - β3*E)
  - C* vacancy generation per occupation: floor(C0*(1+g)*(1 - A*R + γ*A*P))
  - Poisson matching: P(H) = 1 - exp(-ρ·θ),  ρ derived analytically
  - OLG demography: stochastic retirement hazard + monthly workforce entry
  - Firm entry/exit: spin-off probability + Healthy/Distressed/Failed states
  - Mincer wages: ln(W) = ln(W_base) + r*Z + β1*E - β2*E²
"""

import math
import numpy as np
import pandas as pd
import mesa
from mesa.datacollection import DataCollector

from agents.Worker import WorkerAgent
from agents.Employer import EmployerAgent
from agents.PublicSectorEmployer import PublicSectorEmployerAgent


# ── Default simulation parameters ─────────────────────────────────────────────

DEFAULT_PARAMS = {
    # ── Displacement hazard (new formula) ─────────────────────────────────────
    # P(D) = sigmoid(logit(δ_base) + β1*(A_jt*R_job) - β2*(A_jt*P_aug) - β3*E_i)
    "delta_base":    0.01201, # baseline monthly turnover intercept (ABC posterior mean, N=1341/2000)
    "beta":          3.5,     # backward-compat alias for beta1
    "beta1":         3.5,     # automation displacement coefficient
    "beta2":         0.5,     # augmentation protection coefficient
    "beta3_exp":     0.3,     # experience protection coefficient
    "lambda_":       0.5,     # backward-compat alias for beta2
    "beta_run_std":  0.2,     # legacy — kept for bootstrap_runner compatibility

    # ── Experience modifiers (market-clearing match score only) ───────────────
    "delta_sub":  0.30,
    "delta_aug":  0.40,

    # ── AI adoption logistic curve ─────────────────────────────────────────────
    "k_adoption":    0.05,    # logistic growth rate per tick
    "a_max":         1.0,     # ceiling adoption maturity

    # ── Retraining ─────────────────────────────────────────────────────────────
    "eta_base":      0.02,    # employed proactive upskilling rate
    "eta_unemp":     0.05,    # unemployed reactive retraining rate
    "kappa":         0.06,    # automation-fear multiplier
    "xi":            0.03,    # unemployment-duration multiplier
    "omega":         0.5,     # retrain time scaling (fraction of semantic dist)
    "mu":            5.0,     # gravity-model cognitive friction penalty
    "retrain_blend": 0.7,     # weight toward target occ's risk profile
    # Credential gap: ticks per zone gap (index = gap size 0-4)
    "zone_ticks":    [0, 6, 12, 24, 36],

    # ── Mincer wage equation ───────────────────────────────────────────────────
    "r_edu":         0.09,    # wage return per O*NET Job Zone level
    "mincer_b1":     0.04,    # Mincer experience coefficient
    "mincer_b2":     0.002,   # Mincer experience-squared coefficient
    "pub_wage_damp": 0.6,     # public-sector premium dampener (× r_edu & mincer_b1)
    "wage_boost":    0.02,    # annual augmentation productivity boost

    # ── Retirement hazard ──────────────────────────────────────────────────────
    "alpha_retire":  -6.5,    # retirement logit intercept (calibrated: P(R,55)≈0.23%/mo, P(R,65)≈1%/mo)
    "beta_age":       0.15,   # age acceleration coefficient
    "beta_wealth":    0.1,    # wealth-wage coefficient
    "tau_retire":    55,      # early-retirement age threshold

    # ── Firm entry / exit ──────────────────────────────────────────────────────
    "lambda_spinoff": 0.001,  # baseline spin-off probability per employed worker
    "psi":            3.0,    # seniority exponential multiplier for spin-offs
    "tau_exit":       15,     # distress ticks before firm failure
    "sigma":          0.02,   # new-economy vacancy reinstatement rate

    # ── Labor force OLG entry ──────────────────────────────────────────────────
    "v_entry_rate":  0.0025,  # monthly workforce entry rate (fraction of pool)

    # ── Matching function ──────────────────────────────────────────────────────
    # ρ is derived analytically at model init: ρ = -ln(1-f_target)/θ_base
    "f_target":       0.28,   # target monthly job-finding rate (Shimer 2005)
    "theta_base":     0.5,    # baseline tightness for ρ calibration
    "vacancy_rate":   0.04485, # open positions as fraction of employment (ABC posterior mean)
    "nu":             2.0,    # legacy experience premium (kept for compat)

    # ── Employer vacancy generation ────────────────────────────────────────────
    "gamma":          0.3,    # augmentation demand elasticity
    "epsilon":        0.5,    # legacy direct-replacement fraction (kept for compat)
    "btos_shock_std":  0.02,   # BTOS monthly shock std dev (idiosyncratic, firm-level)
    "btos_macro_std":  0.015,  # common macro shock std dev (same draw for all firms → Beveridge cyclicality)
    "theta_ou":        0.1,    # OU mean-reversion speed (half-life ≈ 7 months)
    "btos_disp_damp":  0.5,    # BTOS pass-through dampener: eff_base = delta*(1 - damp*btos); 0→no BTOS effect, 1→full pass-through

    # ── Firm-size distribution ─────────────────────────────────────────────────
    "zipf_alpha":     2.0,    # Zipf exponent
    "employer_ratio": 22,     # Census 22:1 worker-to-employer ratio

    # ── Calibrated from ABC Run 9 (output/abc_posterior.csv, 2026-04-19) ────────
    # Model fixes applied before Run 9:
    #   (1) search_occ hard-redirect gated to unemployed workers only
    #   (2) floor()→round() in C* vacancy formula (prevents small-firm distress cascade)
    #   (3) BTOS dampener: eff_base = delta*(1 - btos_disp_damp*btos_signal)
    #   (4) OLG timing: retirements + entries BEFORE employer clearing
    #   (5) Matching fallback: workers seek in {search_occ, current_occ} — eliminates
    #       permanent occupational mismatch lock-up that caused UR to drift indefinitely
    # 1341/2000 particles accepted (ε=0.005), target UR=4.5%, mean simulated UR=4.54%
    # delta_base posterior: mean=0.01201
    # vacancy_rate posterior: mean=0.04485 (partially identified)
}


# ── Credential system ─────────────────────────────────────────────────────────
#
# Workers hold a credential level derived from their IPUMS CPS EDUC code.
# Occupations require a minimum credential derived from their O*NET Job Zone.
# Hiring is gated: workers below the required credential are excluded from
# valid_candidates in _market_clearing().  Retraining time includes the time
# to traverse the credential DAG from the worker's current level to the
# minimum required by the target occupation.
#
# Workers can always abandon a credential path mid-way (they remain at their
# current credential level; partial progress is not saved — a simplification
# that reflects the all-or-nothing value of formal credentials).

CREDENTIAL_LEVELS = ["high_school", "vocational", "associates",
                     "bachelors", "masters", "doctoral"]
CREDENTIAL_IDX    = {c: i for i, c in enumerate(CREDENTIAL_LEVELS)}

# Directed graph: source → [(target, months)]
# Encodes the diagram: HS is the root; doctoral is the ceiling.
# Vocational and Associates are lateral entry-points for trades.
CREDENTIAL_GRAPH = {
    "high_school": [("vocational", 12), ("associates", 24), ("bachelors", 48)],
    "vocational":  [("associates", 12)],
    "associates":  [("bachelors",  24)],
    "bachelors":   [("masters",    24), ("doctoral",  48)],
    "masters":     [("doctoral",   24)],
    "doctoral":    [],
}

# Minimum credential required by O*NET Job Zone
ZONE_MIN_CREDENTIAL = {
    1: "high_school",
    2: "vocational",
    3: "bachelors",
    4: "masters",
    5: "doctoral",
}

# IPUMS CPS EDUC code → credential level
def educ_to_credential(educ: int) -> str:
    if educ <= 50:                   # ≤ HS diploma / GED
        return "high_school"
    if educ < 80:                    # some college, associate's (EDUC 60–73)
        return "associates"
    if educ == 80:                   # bachelor's
        return "bachelors"
    if educ == 90:                   # master's
        return "masters"
    return "doctoral"                # professional (100) or doctoral (110)


def credential_months_to(src: str, tgt: str) -> int:
    """Shortest path (months) from src credential to tgt in the DAG.

    Returns 0 if src already meets or exceeds tgt.
    Returns a large sentinel (999) if the path is unreachable (should never
    happen with a properly connected graph).
    """
    if CREDENTIAL_IDX.get(src, 0) >= CREDENTIAL_IDX.get(tgt, 0):
        return 0
    from collections import deque
    q: deque = deque([(src, 0)])
    seen = {src}
    while q:
        node, cost = q.popleft()
        for nxt, months in CREDENTIAL_GRAPH.get(node, []):
            total = cost + months
            if nxt == tgt:
                return total
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, total))
    return 999   # unreachable


def _default_data_dir():
    import pathlib
    return pathlib.Path(__file__).parent.parent / "data" / "processed"


class LaborMarketModel(mesa.Model):
    """Agent-Based Model of the US labor market under AI adoption.

    Args:
        worker_df  : DataFrame from data/processed/worker_sample_with_risk.parquet
        params     : dict of simulation parameters (defaults to DEFAULT_PARAMS)
        ai_active  : if True, run the AI displacement scenario
        seed       : random seed for reproducibility
        data_dir   : path to processed data directory (optional override)
        skill_distance_matrix : pre-loaded distance DataFrame (avoids disk I/O)
        occ_risk_lookup       : pre-built dict {"r_job": {...}, "p_aug": {...}}
        collect_agent_data    : if True, collect per-agent reporters each tick
    """

    def __init__(self, worker_df, params=None, ai_active=True, seed=42,
                 data_dir=None,
                 skill_distance_matrix=None, occ_risk_lookup=None,
                 collect_agent_data=True):
        super().__init__(seed=seed)
        self.ai_active           = ai_active
        self.params              = params or DEFAULT_PARAMS
        self.tick                = 0
        self._collect_agent_data = collect_agent_data

        # Legacy: draw beta_run for backward compat with bootstrap_runner
        beta_run_std  = self.params.get("beta_run_std", 0.2)
        self.beta_run = self.random.gauss(1.0, beta_run_std) * self.params.get("beta", 3.5)

        ddir = data_dir if data_dir else _default_data_dir()
        import pathlib
        ddir = pathlib.Path(ddir)

        # ── Skill distance matrix ──────────────────────────────────────────────
        if skill_distance_matrix is not None:
            self.skill_distance_matrix = skill_distance_matrix
        else:
            dist_path = ddir / "skill_distance_matrix.parquet"
            if dist_path.exists():
                self.skill_distance_matrix = pd.read_parquet(dist_path)
                self.skill_distance_matrix.index   = self.skill_distance_matrix.index.astype(int)
                self.skill_distance_matrix.columns = self.skill_distance_matrix.columns.astype(int)
            else:
                self.skill_distance_matrix = None

        # ── Occupation risk lookup ─────────────────────────────────────────────
        if occ_risk_lookup is not None:
            self.occ_risk_lookup = occ_risk_lookup
        else:
            risk_path = ddir / "occ_risk_lookup.parquet"
            if risk_path.exists():
                risk_df = pd.read_parquet(risk_path)
                risk_df.index = risk_df.index.astype(int)
                self.occ_risk_lookup = {
                    "r_job": risk_df["r_job"].to_dict(),
                    "p_aug": risk_df["p_aug"].to_dict(),
                }
            else:
                self.occ_risk_lookup = {"r_job": {}, "p_aug": {}}

        # ── Job Zone lookup (OCC2010 → job_zone 1-5) ──────────────────────────
        jz_path = ddir / "job_zone_lookup.parquet"
        if jz_path.exists():
            jz_df = pd.read_parquet(jz_path)
            self.job_zone_lookup = dict(zip(jz_df["OCC2010"].astype(int),
                                            jz_df["job_zone"].astype(int)))
        else:
            self.job_zone_lookup = {}

        # ── Occupation minimum-credential lookup (OCC2010 → credential string) ──
        # Derived from job_zone_lookup using ZONE_MIN_CREDENTIAL mapping.
        self.occ_min_credential = {
            occ: ZONE_MIN_CREDENTIAL.get(zone, "high_school")
            for occ, zone in self.job_zone_lookup.items()
        }

        # ── Occupation wage lookup (OCC2010 → median annual wage $K) ──────────
        wg_path = ddir / "occ_wage_lookup.parquet"
        if wg_path.exists():
            wg_df = pd.read_parquet(wg_path)
            self.occ_wage_lookup = dict(zip(wg_df["OCC2010"].astype(int),
                                            wg_df["median_wage"].astype(float)))
        else:
            self.occ_wage_lookup = {}

        # ── BTOS sector signals (naics_sector → a_init, g_init) ───────────────
        btos_path = ddir / "btos_sector_signals.parquet"
        if btos_path.exists():
            btos_df = pd.read_parquet(btos_path)
            self._btos_sector = {
                str(r["naics_sector"]): {
                    "a_init": float(r["a_init"]),
                    "g_init": float(r["g_init"]),
                }
                for _, r in btos_df.iterrows()
            }
        else:
            self._btos_sector = {}

        # ── BDS sector dynamics (sector → entry_rate, exit_rate) ──────────────
        bds_path = ddir / "bds_sector_dynamics.parquet"
        if bds_path.exists():
            bds_df = pd.read_parquet(bds_path)
            self._bds_sector = {
                str(r["sector"]): {
                    "entry_rate": float(r["entry_rate"]),
                    "exit_rate":  float(r["exit_rate"]),
                }
                for _, r in bds_df.iterrows()
            }
        else:
            self._bds_sector = {}

        # ── Enrich worker_df with job_zone and w_base columns ─────────────────
        worker_df = worker_df.copy()
        if "job_zone" not in worker_df.columns and self.job_zone_lookup:
            worker_df["job_zone"] = (worker_df["OCC2010"]
                                     .astype(int)
                                     .map(self.job_zone_lookup)
                                     .fillna(3)
                                     .astype(int))
        if "w_base" not in worker_df.columns and self.occ_wage_lookup:
            worker_df["w_base"] = (worker_df["OCC2010"]
                                   .astype(int)
                                   .map(self.occ_wage_lookup)
                                   .fillna(worker_df["wage"]))

        # Macro pool for OLG entry: full workforce distribution, not youth-only.
        # Sampling from the full CPS distribution produces entrants whose
        # occupation mix mirrors the economy-wide demand captured in C0,
        # preventing the generational skills-gap that caused UR drift when
        # sampling strictly from 18-24 year olds (retail/food-heavy).
        # Age and experience are overwritten to youth values post-creation.
        self._macro_worker_pool = worker_df.copy()

        # ── Vacancy and job-creation state ─────────────────────────────────────
        self.vacancy_counts           = {}   # employed-worker counts per OCC2010
        self.effective_vacancy_counts = {}   # per-occ vacancy signal for gravity model
        self._displacement_this_tick     = 0
        self._new_economy_jobs_this_tick = 0
        self._open_market_hired_this_tick = 0
        self._spinoffs_this_tick          = 0
        self._retirements_this_tick       = 0
        self._entries_this_tick           = 0
        self._macro_shock_this_tick       = 0.0
        # Per-occupation matching inputs (θ for Poisson matching)
        self._tightness = {}

        # ── Instantiate WorkerAgents ───────────────────────────────────────────
        for _, row in worker_df.iterrows():
            WorkerAgent(self, row, self.params)

        # ── Instantiate EmployerAgents ─────────────────────────────────────────
        # Public sector (NAICS "92"): one immortal PublicSectorEmployerAgent.
        # Private sector: Zipf-distributed firms per industry.
        self._employers: dict = {}
        _pub_employer: "PublicSectorEmployerAgent | None" = None
        _private_workers_by_ind: dict = {}

        for worker in list(self.agents_by_type[WorkerAgent]):
            if str(getattr(worker, "naics_sector", "")) == "92":
                if _pub_employer is None:
                    pub_btos = self._btos_sector.get("92", {}).get("g_init", 0.0)
                    _pub_employer = PublicSectorEmployerAgent(self, initial_btos=pub_btos)
                    self._employers["public_sector"] = _pub_employer
                _pub_employer.assign_worker(worker)
            else:
                ind = str(getattr(worker, "ind1990", worker.naics_sector))
                _private_workers_by_ind.setdefault(ind, []).append(worker)

        _np_rng      = np.random.default_rng(self.random.randint(0, 2**32 - 1))
        zipf_alpha   = self.params.get("zipf_alpha",     2.0)
        employer_ratio = self.params.get("employer_ratio", 22)
        firm_counter = 0

        for ind, workers in _private_workers_by_ind.items():
            n_workers = len(workers)
            n_firms   = max(1, n_workers // employer_ratio)
            naics_sec = str(getattr(workers[0], "naics_sector", ind))

            # Sector-level BTOS signals for this industry
            btos_sig  = self._btos_sector.get(naics_sec[:2], {})
            g_init    = btos_sig.get("g_init", 0.0)
            a_init    = btos_sig.get("a_init", 0.05)

            c_max      = max(2, int(n_workers * 0.20))
            zipf_draws = np.clip(_np_rng.zipf(a=zipf_alpha, size=n_firms), 1, c_max)
            draw_sum   = int(zipf_draws.sum())
            firm_capacities = [max(1, int((d / draw_sum) * n_workers)) for d in zipf_draws]
            remainder = n_workers - sum(firm_capacities)
            if remainder != 0:
                firm_capacities[int(np.argmax(firm_capacities))] += remainder

            workers_shuffled = list(workers)
            self.random.shuffle(workers_shuffled)

            worker_idx = 0
            for capacity in firm_capacities:
                firm_id  = f"{ind}_{firm_counter}"
                new_firm = EmployerAgent(
                    self, naics_sec,
                    initial_btos=g_init,
                    ind_key=firm_id,
                    initial_capacity=capacity,
                    a_adoption=a_init,
                )
                self._employers[firm_id] = new_firm
                for w in workers_shuffled[worker_idx : worker_idx + capacity]:
                    new_firm.assign_worker(w)
                worker_idx   += capacity
                firm_counter += 1

        # ── Initialize vacancy snapshots ───────────────────────────────────────
        self._update_vacancy_counts()
        self._update_effective_vacancies()

        # ── Data collection ────────────────────────────────────────────────────
        _agent_reporters = {
            "is_employed":           lambda a: getattr(a, "is_employed",           None),
            "months_unemployed":     lambda a: getattr(a, "months_unemployed",     None),
            "r_job":                 lambda a: getattr(a, "r_job",                 None),
            "p_aug":                 lambda a: getattr(a, "p_aug",                 None),
            "exp_norm":              lambda a: getattr(a, "exp_norm",              None),
            "wage":                  lambda a: getattr(a, "wage",                  None),
            "exposure_quintile":     lambda a: getattr(a, "exposure_quintile",     None),
            "r_agent_sub":           lambda a: getattr(a, "r_agent_sub",           None),
            "p_agent_aug":           lambda a: getattr(a, "p_agent_aug",           None),
            "p_disp":                lambda a: getattr(a, "p_disp",                None),
            "current_occ":           lambda a: getattr(a, "current_occ",           None),
            "has_retrained":         lambda a: getattr(a, "has_retrained",         None),
            "retraining_ticks_left": lambda a: getattr(a, "retraining_ticks_left", None),
            "job_zone":              lambda a: getattr(a, "job_zone",              None),
            "age":                   lambda a: getattr(a, "age",                   None),
            "is_retired":            lambda a: getattr(a, "is_retired",            False),
        } if self._collect_agent_data else {}

        self.datacollector = DataCollector(
            model_reporters={
                "Employment_Rate":      lambda m: _emp_rate(m),
                "Unemployed_Count":     lambda m: _worker_sum(m, lambda a: not a.is_employed and not a.is_retired),
                "Mean_Wage":            lambda m: _mean_wage(m),
                "Emp_Rate_Q1_Low":      lambda m: _emp_rate_q(m, "Q1_Low"),
                "Emp_Rate_Q2":          lambda m: _emp_rate_q(m, "Q2"),
                "Emp_Rate_Q3":          lambda m: _emp_rate_q(m, "Q3"),
                "Emp_Rate_Q4":          lambda m: _emp_rate_q(m, "Q4"),
                "Emp_Rate_Q5_High":     lambda m: _emp_rate_q(m, "Q5_High"),
                "Emp_Rate_Entry":       lambda m: _emp_rate_exp(m, 0.0, 0.2),
                "Emp_Rate_Senior":      lambda m: _emp_rate_exp(m, 0.8, 1.0),
                "Retraining_Count":     lambda m: _worker_sum(
                    m, lambda a: a.retraining_ticks_left > 0 and not a.is_retired),
                "Retrained_Share":      lambda m: _retrained_share(m),
                "New_Economy_Jobs":     lambda m: m._new_economy_jobs_this_tick,
                "Total_Vacancies":      lambda m: sum(
                    e.vacancies for e in m._employers.values()),
                "Total_Hired":          lambda m: sum(
                    e._hired_this_tick for e in m._employers.values())
                    + m._open_market_hired_this_tick,
                "Total_Fired":          lambda m: sum(
                    e._fired_this_tick for e in m._employers.values()),
                "Avg_BTOS":             lambda m: float(np.mean(
                    [e.btos_signal for e in m._employers.values()])),
                "Avg_A_jt":             lambda m: float(np.mean(
                    [e.a_jt for e in m._employers.values()])) if m.ai_active else 0.0,
                "Firms_Healthy":        lambda m: sum(
                    1 for e in m._employers.values() if e.state == "Healthy"),
                "Firms_Distressed":     lambda m: sum(
                    1 for e in m._employers.values() if e.state == "Distressed"),
                "Firms_Failed":         lambda m: sum(
                    1 for e in m._employers.values() if e.state == "Failed"),
                "Spinoffs_This_Tick":   lambda m: m._spinoffs_this_tick,
                "Retirements_This_Tick": lambda m: m._retirements_this_tick,
                "Entries_This_Tick":    lambda m: m._entries_this_tick,
            },
            agent_reporters=_agent_reporters,
        )

    # ── Vacancy and market helpers ─────────────────────────────────────────────

    def _update_vacancy_counts(self):
        """Recount employed workers per OCC2010 (occupation-size proxy)."""
        counts = {}
        for a in self.agents_by_type[WorkerAgent]:
            if a.is_employed and not a.is_retired:
                counts[a.current_occ] = counts.get(a.current_occ, 0) + 1
        self.vacancy_counts = counts

    def _update_effective_vacancies(self):
        """Aggregate per-occupation vacancies from all employers for the
        gravity model in Worker._choose_target_skill().

        Uses employer._vacancies_by_occ (populated each tick by _generate_vacancies).
        Falls back to employment counts if vacancies aren't yet populated.
        """
        eff: dict = dict(self.vacancy_counts)
        has_emp_vacancies = any(
            getattr(e, "_vacancies_by_occ", {}) for e in self._employers.values()
        )
        if has_emp_vacancies:
            for emp in self._employers.values():
                for occ, v in getattr(emp, "_vacancies_by_occ", {}).items():
                    eff[occ] = eff.get(occ, 0) + v
        self.effective_vacancy_counts = eff

    def _update_job_market(self):
        """Compute per-occupation labor market tightness θ for Poisson matching.

        Uses actual employer-posted vacancies (from _vacancies_by_occ) when
        available.  Falls back to a JOLTs-anchored estimate for occupations
        where no employer is explicitly posting.

        θ(occ) = V(occ, t-1) / max(1, seekers(occ, t))
        """
        vacancy_rate = self.params.get("vacancy_rate", 0.04)

        # Aggregate actual vacancies from previous tick's employer steps
        actual_vac: dict = {}
        for emp in self._employers.values():
            if emp.state != "Failed":
                for occ, v in getattr(emp, "_vacancies_by_occ", {}).items():
                    actual_vac[occ] = actual_vac.get(occ, 0) + v

        # Count active job seekers per occupation
        seeker_count: dict = {}
        for a in self.agents_by_type[WorkerAgent]:
            if not a.is_employed and not a.is_retired and a.retraining_ticks_left == 0:
                occ = a.search_occ if a.search_occ is not None else a.current_occ
                seeker_count[occ] = seeker_count.get(occ, 0) + 1

        self._tightness = {}
        for occ, n in seeker_count.items():
            if n == 0:
                continue
            # Actual posted vacancies; fall back to JOLTs proxy if none posted
            v = actual_vac.get(occ,
                    max(1, int(vacancy_rate * self.vacancy_counts.get(occ, 10))))
            self._tightness[occ] = v / n

    # ── OLG: retirement and workforce entry ────────────────────────────────────

    def _process_retirements(self):
        """Evaluate stochastic retirement for all workers at or above τ_retire."""
        tau = self.params.get("tau_retire", 55)
        self._retirements_this_tick = 0
        for worker in list(self.agents_by_type[WorkerAgent]):
            if worker.is_retired or worker.age < tau:
                continue
            if worker.evaluate_retirement():
                worker.is_retired  = True
                worker.is_employed = False
                self._retirements_this_tick += 1
                if worker.employer is not None:
                    worker.employer._roster.discard(worker)
                    worker.employer = None

    def _process_workforce_entry(self):
        """Replace each retirement with a new entrant drawn from the full CPS
        occupation distribution, reset to youth demographics.

        Sampling from the macro pool (not youth-only) ensures the entrant's
        occupation matches the economy-wide distribution of C0 demand.
        Overwriting age/experience to 18/0.0 simulates a stable educational
        pipeline that continuously replenishes each occupation with fresh workers.
        """
        if self._macro_worker_pool.empty or self._retirements_this_tick <= 0:
            self._entries_this_tick = 0
            return

        n_new = self._retirements_this_tick
        sample = self._macro_worker_pool.sample(
            n=n_new,
            replace=True,
            random_state=self.random.randint(0, 2**31 - 1),
        )
        for _, row in sample.iterrows():
            w = WorkerAgent(self, row, self.params)
            w.age                  = 18
            w.exp_norm             = 0.0
            w.is_employed          = False
            w.months_unemployed    = 0
            w.just_fired           = False
            w.retraining_ticks_left = 0
        self._entries_this_tick = n_new

    # ── Spin-off: triggered by Worker._maybe_spinoff() ────────────────────────

    def _trigger_spinoff(self, founder: "WorkerAgent"):
        """Instantiate a new one-person EmployerAgent as a spin-off.

        The founding worker is immediately employed at the new firm. The new firm
        inherits the founder's industry sector and BTOS signals.
        """
        sector    = str(getattr(founder, "naics_sector", "51"))
        btos_sig  = self._btos_sector.get(sector[:2], {})
        g_init    = btos_sig.get("g_init", 0.0)
        a_init    = btos_sig.get("a_init", 0.05)
        firm_id   = f"spinoff_{self.tick}_{founder.unique_id}"

        new_firm = EmployerAgent(
            self, sector,
            initial_btos=g_init,
            ind_key=firm_id,
            initial_capacity=1,
            a_adoption=a_init,
        )
        self._employers[firm_id] = new_firm

        # Detach founder from old employer
        if founder.employer is not None:
            founder.employer._roster.discard(founder)
        new_firm.assign_worker(founder)
        founder.is_employed       = True
        founder.months_unemployed = 0

        self._spinoffs_this_tick += 1

    # ── Main step ──────────────────────────────────────────────────────────────

    def step(self):
        self.datacollector.collect(self)
        self._update_vacancy_counts()
        self._update_effective_vacancies()
        self._update_job_market()

        self._displacement_this_tick      = 0
        self._new_economy_jobs_this_tick  = 0
        self._open_market_hired_this_tick = 0
        self._spinoffs_this_tick          = 0

        # Draw common macro shock once per tick — all employers add this same draw
        # to their individual OU shock, creating aggregate BTOS cyclicality needed
        # for the Beveridge curve to emerge.
        macro_std = self.params.get("btos_macro_std", 0.015)
        self._macro_shock_this_tick = self.random.gauss(0.0, macro_std)

        # OLG runs BEFORE employer clearing so that:
        #   (a) retiring workers are removed from rosters before _layoff_phase
        #       counts them, and emp_by_occ gaps are visible to _generate_vacancies;
        #   (b) new entrants are present in agents_by_type[WorkerAgent] when
        #       _market_clearing builds global_seekers, eliminating the guaranteed
        #       one-tick hiring delay that caused the unemployment pool to grow.
        self._process_retirements()
        self._process_workforce_entry()

        # Employers: layoff + vacancy generation + firm state + market clearing
        _employer_list = list(self._employers.values())
        self.random.shuffle(_employer_list)
        for _e in _employer_list:
            _e.step()

        # Workers: retraining / job search / proactive upskilling / spin-offs
        self.agents_by_type[WorkerAgent].shuffle_do("step")

        # Global reset of temporal friction flag — ensures workers fired while
        # retraining (who detached from the roster before the flag could be
        # reset by _layoff_phase) are never permanently locked out of clearing.
        for w in self.agents_by_type[WorkerAgent]:
            w.just_fired = False

        self.tick += 1


# ── Reporter helpers (module-level for pickling) ───────────────────────────────

def _workers(m):
    return [a for a in m.agents_by_type[WorkerAgent] if not a.is_retired]


def _worker_sum(m, fn):
    return sum(fn(a) for a in _workers(m))


def _retrained_share(m):
    ws = _workers(m)
    return sum(a.has_retrained for a in ws) / len(ws) if ws else 0.0


def _emp_rate(m):
    ws = _workers(m)
    return sum(a.is_employed for a in ws) / len(ws) if ws else 0.0


def _mean_wage(m):
    wages = [a.wage for a in _workers(m) if a.is_employed and a.wage > 0]
    return float(np.mean(wages)) if wages else 0.0


def _emp_rate_q(m, quintile):
    grp = [a for a in _workers(m) if a.exposure_quintile == quintile]
    return sum(a.is_employed for a in grp) / len(grp) if grp else float("nan")


def _emp_rate_exp(m, lo, hi):
    grp = [a for a in _workers(m) if lo <= a.exp_norm <= hi]
    return sum(a.is_employed for a in grp) / len(grp) if grp else float("nan")
