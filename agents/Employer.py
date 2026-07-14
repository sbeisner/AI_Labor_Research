"""Employer agent for the AI Labor Market ABM.

Implements BTOS-modulated hiring/firing with a 5-phase step:
  1. BTOS signal update    — Ornstein-Uhlenbeck mean-reversion toward BTOS g_init anchor
  2. Layoff phase          — bifurcated frictional + structural separation
  3. Vacancy generation    — V = max(0, C* - E) per occupation + V_new new-economy vacancies
  4. Firm state update     — Healthy / Distressed / Failed transitions
  5. Market clearing       — hire ranked unemployed workers by match score

Layoff phase (Phase 2) is split into three steps:
  Step A — Frictional turnover: stochastic separation at eff_base (AI-independent).
  Step B — Compute C*: firm's target capacity for the tick (industry-specific γ).
  Step C — Structural AI layoffs: if E_post_attrition > C*, deterministically sever
           the top max(0, E_post - C*) workers ranked by vulnerability score
                Z = β1*(A_jt*R_job) - β2*(A_jt*P_aug) - β3*E_i
           High Z → high substitution risk, low augmentation potential, low seniority.
           P_aug therefore acts as a *selection filter* (high P_aug retained), not as
           a hazard-rate dampener — separating aggregate labor demand (γ in C*) from
           individual worker value (P_aug in the ranking).

Firm states
-----------
  Healthy    : total C* > 0 (default)
  Distressed : total C* <= 0 for at least 1 tick
  Failed     : distress_counter >= tau_exit; firm discharges all workers and
               stops participating in the simulation
"""

import math

import numpy as np
import mesa

from agents.Worker import WorkerAgent


# Monthly drift by NAICS 2-digit sector prefix
_SECTOR_DRIFT = {
    "62": +0.05 / 12,   # Health Care and Social Assistance
    "61": +0.04 / 12,   # Educational Services
    "51": +0.03 / 12,   # Information
    "52": +0.02 / 12,   # Finance and Insurance
    "54": +0.02 / 12,   # Professional, Scientific, Technical Services
    "56": +0.01 / 12,   # Administrative and Support Services
    "72": +0.01 / 12,   # Accommodation and Food Services
    "44": +0.00 / 12,   # Retail Trade
    "45": +0.00 / 12,   # Retail Trade (cont.)
    "48": -0.01 / 12,   # Transportation and Warehousing
    "49": -0.01 / 12,
    "42": -0.01 / 12,   # Wholesale Trade
    "23": -0.01 / 12,   # Construction
    "31": -0.02 / 12,   # Manufacturing
    "32": -0.02 / 12,
    "33": -0.02 / 12,
    "21": -0.02 / 12,   # Mining, Quarrying, Oil and Gas
    "11": -0.01 / 12,   # Agriculture, Forestry, Fishing
}
_DEFAULT_DRIFT = 0.0


class EmployerAgent(mesa.Agent):
    """Represents a single firm in the labor market.

    Multiple EmployerAgents are created per IND1990 industry, distributed
    according to a Zipf (power-law) firm-size distribution so that roster
    sizes reflect the empirical Pareto scaling of US firm sizes.
    Workers are registered via assign_worker() at model initialisation.

    The employer drives displacement and hiring for its roster; WorkerAgent.step()
    skips _check_displacement() and _search_for_job() when employer is set.

    Attributes:
        capacity        : int — Zipf-drawn initial roster target (1 when not specified).
        ai_modifier     : float ∈ [0.1, 2.0] — firm-specific AI adoption velocity,
                          combining log-scaled size inertia with a Gaussian draw for
                          unobservable firm culture (σ² term).
        a_adoption      : float ∈ [0.0, 1.0] — sector initial AI adoption level from
                          BTOS Q7; seeds the logistic growth trajectory.
        state           : str — Healthy | Distressed | Failed
        distress_counter: int — consecutive ticks with total C* <= 0
    """

    def __init__(self, model, sector, initial_btos=0.0, ind_key=None,
                 initial_capacity=None, a_adoption=0.0):
        super().__init__(model)
        self.sector      = str(sector)    # NAICS 2-digit prefix — used for BTOS drift
        self.ind_key     = str(ind_key) if ind_key is not None else self.sector
        self.btos_signal = float(np.clip(initial_btos, -0.15, 0.15))
        self.capacity    = int(initial_capacity) if initial_capacity is not None else 1
        self._roster: set = set()  # WorkerAgent instances — set for O(1) add/remove
        self.vacancies          = 0
        self._fired_this_tick   = 0
        self._hired_this_tick   = 0

        # ── Firm-specific AI adoption velocity ───────────────────────────────
        # size_inertia: log₁₀-scaled bureaucratic drag — a 1,000-person firm is
        #   penalised ~0.15 relative to a 5-person startup (~0.03).
        # cultural_variance: Gaussian draw representing unobservable firm culture
        #   (CEO AI enthusiasm, change-management capacity, etc.)
        # ai_modifier: clipped to [0.1, 2.0] so no firm fully opts out or
        #   hyper-adopts beyond twice the macro shock.
        size_inertia      = math.log10(self.capacity + 1) * 0.05
        cultural_variance = self.random.gauss(0.0, 0.1)
        self.ai_modifier  = float(np.clip(
            1.0 - size_inertia + cultural_variance, 0.1, 2.0
        ))

        # ── AI adoption maturity seeding ─────────────────────────────────────
        self.a_adoption        = float(np.clip(a_adoption, 0.0, 1.0))  # sector initial AI adoption (BTOS Q7)
        self._cap_by_occ: dict = {}  # C_{j,o,0}: FIXED baseline capacity per OCC2010 (set once at tick 0)
        self._cstar_this_tick: dict = {}    # C* computed this tick (refreshed each _generate_vacancies call)
        self._vacancies_by_occ: dict = {}   # per-occupation open vacancies this tick
        self.vacancy_age_by_occ: dict = {}  # ticks a vacancy for occ has been continuously open
        self.state            = "Healthy"   # Healthy | Distressed | Failed
        self.distress_counter = 0           # ticks with total C* <= 0

        # ── Firm-specific Mincer coefficient draws ───────────────────────────
        # Per Lemieux (2006) and Heckman-Lochner-Todd (2006), the experience-
        # premium coefficients in the Mincer earnings function vary across
        # employers — high-tacit-knowledge firms and efficiency-wage payers
        # (Card, Heining & Kline 2013; Abowd-Kramarz-Margolis 1999) have
        # measurably steeper experience-wage profiles. When mincer_firm_std > 0,
        # each firm draws its own scalar adjusters (1 + ε) for the four
        # quartic coefficients.  Workers read these via Worker.compute_mincer_wage
        # so that retention incentives for senior workers emerge endogenously
        # rather than being globally hard-coded.
        p_init = self.model.params if hasattr(self.model, "params") else {}
        firm_std = float(p_init.get("mincer_firm_std", 0.0))
        if firm_std > 0.0:
            self.mincer_b1_adj = max(0.1, 1.0 + self.random.gauss(0.0, firm_std))
            self.mincer_b2_adj = max(0.1, 1.0 + self.random.gauss(0.0, firm_std))
            self.mincer_b3_adj = max(0.1, 1.0 + self.random.gauss(0.0, firm_std))
            self.mincer_b4_adj = max(0.1, 1.0 + self.random.gauss(0.0, firm_std))
        else:
            self.mincer_b1_adj = 1.0
            self.mincer_b2_adj = 1.0
            self.mincer_b3_adj = 1.0
            self.mincer_b4_adj = 1.0

    # ── AI adoption maturity property ────────────────────────────────────────

    @property
    def a_jt(self) -> float:
        """Current AI adoption maturity for this firm.

        In the control scenario (ai_active=False): always 0.
        In the AI scenario: logistic growth from (a_adoption * ai_modifier),
        capped at a_max.

        Formula: a(t) = a_max / (1 + ((a_max - a0) / a0) * exp(-k * t))
        where a0 = a_adoption * ai_modifier (firm-specific initial level).

        The logistic growth rate k is scaled by the global dose parameter
        `adoption_velocity_mult` (default 1.0), which slows or accelerates the
        adoption trajectory uniformly across firms for dose-response sweeps.
        """
        if not self.model.ai_active:
            return 0.0
        p     = self.model.params
        k     = p.get("k_adoption", 0.05) * p.get("adoption_velocity_mult", 1.0)
        a_max = p.get("a_max", 1.0)
        a0    = float(np.clip(self.a_adoption * self.ai_modifier, 1e-6, a_max - 1e-6))
        t     = self.model.tick
        denom = 1.0 + ((a_max - a0) / a0) * math.exp(-k * t)
        return float(np.clip(a_max / denom, 0.0, 1.0))

    def assign_worker(self, worker):
        """Register a worker to this employer's roster."""
        self._roster.add(worker)
        worker.employer = self

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self):
        if self.state == "Failed":
            return  # Failed firms do nothing
        self._update_btos()
        self._layoff_phase()
        self._generate_vacancies()
        self._update_firm_state()
        self._market_clearing()

    # ── Phase 1: BTOS signal ─────────────────────────────────────────────────

    def _update_btos(self):
        p         = self.model.params
        shock_std = p.get("btos_shock_std", 0.02)
        theta     = p.get("theta_ou", 0.1)  # mean-reversion speed

        # Anchor (μ_j): sector g_init from BTOS, fall back to legacy _SECTOR_DRIFT,
        # plus the model-wide aggregate-demand shift from the Keynesian feedback
        # loop.  When mass layoffs depress the aggregate wage bill, the
        # consumption shortfall pulls every firm's long-term drift downward —
        # closing the loop between labor and goods markets.
        btos_data = self.model._btos_sector.get(self.sector[:2], {})
        mu        = btos_data.get("g_init", _SECTOR_DRIFT.get(self.sector[:2], _DEFAULT_DRIFT))
        mu       += getattr(self.model, "_consumption_anchor_shift", 0.0)

        shock = self.random.gauss(0.0, shock_std)

        # Common macro shock: same draw for all firms this tick (from model).
        # Adds aggregate demand cyclicality so the Beveridge curve can emerge.
        macro_shock = getattr(self.model, "_macro_shock_this_tick", 0.0)

        # Ornstein-Uhlenbeck Euler-Maruyama step:
        # g_{j,t} = g_{j,t-1} + θ(μ_j - g_{j,t-1}) + σ_idio*ε_{j,t} + σ_macro*η_t
        reversion_pull = theta * (mu - self.btos_signal)
        self.btos_signal = float(
            np.clip(self.btos_signal + reversion_pull + shock + macro_shock, -0.15, 0.15)
        )

    # ── Phase 2: Layoff (bifurcated frictional + structural) ─────────────────

    def _layoff_phase(self):
        """Three-step separation: frictional turnover → compute C* → structural cut.

        Step A: AI-independent frictional turnover at eff_base (BTOS-modulated).
        Step B: Compute target capacity C* using industry-specific γ (cached for
                Phase 3 vacancy generation).
        Step C: If E_post_attrition > C*, deterministically lay off exactly the
                gap, ranked by vulnerability Z = β1*A*R_job - β2*A*P_aug - β3*E_i.
                P_aug functions as a survivor-selection filter, not a hazard cap.
        """
        p = self.model.params
        # BTOS modulates frictional turnover with a dampener so negative macro
        # cycles don't overwhelm baseline separation. btos_disp_damp=0 → BTOS has
        # no effect on separations; 1.0 → full pass-through.
        btos_damp = p.get("btos_disp_damp", 0.5)
        eff_base = float(np.clip(
            p["delta_base"] * (1.0 - btos_damp * self.btos_signal), 1e-9, 1.0 - 1e-9
        ))
        self._fired_this_tick = 0
        a_jt = self.a_jt  # compute once per tick

        # Reset temporal friction flag before evaluating layoffs this tick.
        for worker in self._roster:
            worker.just_fired = False

        # ── Step A: Frictional turnover ─────────────────────────────────────
        # Quits, performance fires, voluntary exits — entirely independent of AI.
        for worker in [w for w in self._roster if w.is_employed]:
            if self.random.random() < eff_base:
                worker.is_employed       = False
                worker.months_unemployed = 0
                worker.just_fired        = True
                self._fired_this_tick   += 1

        # ── Step B: Compute target capacity C* ──────────────────────────────
        # Industry-specific γ is consumed inside the helper. _cstar_this_tick is
        # cached for Phase 3 (_generate_vacancies) so we only compute once.
        cstar_total = self._compute_cstar()

        # ── Step C: Structural AI layoffs ───────────────────────────────────
        # No structural cut without AI. Wage boost on remaining employed workers
        # is also AI-conditional and applies post-Phase-2.
        if self.model.ai_active and a_jt > 0:
            post_attrition = [w for w in self._roster if w.is_employed]
            layoffs_needed = max(0, len(post_attrition) - cstar_total)

            # Organisational-friction cap: cannot cut more than max_layoff_rate
            # of post-attrition roster in a single tick. Realistic firms face
            # severance liabilities, WARN Act notice periods, and operational
            # continuity constraints that prevent instantaneous capacity
            # adjustment. The employment line E therefore lags the target C*
            # on the descent rather than tracking it discontinuously.
            max_rate    = p.get("max_layoff_rate", 0.05)
            layoff_cap  = math.ceil(max_rate * len(post_attrition)) if post_attrition else 0
            allowed_cuts = min(layoffs_needed, layoff_cap)

            if allowed_cuts > 0:
                beta1 = p.get("beta1",    p.get("beta", 3.5))
                beta2 = p.get("beta2",    p.get("lambda_", 0.5))
                beta3 = p.get("beta3_exp", 0.3)

                # Managerial-information noise (peer-review robustness): fresh
                # per-tick observation of each worker's r_job/p_aug at the firm.
                # Underlying agent attributes are unchanged. With both noise
                # sigmas at 0.0 (default) this collapses to the perfect-info
                # baseline.
                r_sd = p.get("r_job_noise", 0.0)
                p_sd = p.get("p_aug_noise", 0.0)
                if r_sd > 0.0 or p_sd > 0.0:
                    _rng = self.model.random
                    obs = {
                        w: (
                            max(0.0, min(1.0, w.r_job + _rng.gauss(0.0, r_sd))),
                            max(0.0, min(1.0, w.p_aug + _rng.gauss(0.0, p_sd))),
                        )
                        for w in post_attrition
                    }
                else:
                    obs = {w: (w.r_job, w.p_aug) for w in post_attrition}

                # Vulnerability score (no sigmoid — only relative rank matters).
                # Higher Z → higher chopping-block priority. P_aug protects, but
                # only conditional on the firm needing to cut someone.
                def _z(w, _b1=beta1, _b2=beta2, _b3=beta3, _a=a_jt, _o=obs):
                    r_o, p_o = _o[w]
                    return (_b1 * (_a * r_o)
                            - _b2 * (_a * p_o)
                            - _b3 * w.exp_norm)

                ranked = sorted(post_attrition, key=_z, reverse=True)
                for worker in ranked[:allowed_cuts]:
                    worker.is_employed       = False
                    worker.months_unemployed = 0
                    worker.just_fired        = True
                    self._fired_this_tick   += 1
                    self.model._displacement_this_tick += 1
                    if worker.hard_skill_quintile == "HSQ1_Low":
                        self.model._q1_displaced_this_tick += 1
                    worker._choose_target_skill()

            # Wage boost: augmentation lifts productivity for everyone still
            # employed after structural cuts (the high-P_aug survivors).
            wage_boost = p.get("wage_boost", 0.02)
            for worker in self._roster:
                if worker.is_employed:
                    worker.wage *= 1.0 + (wage_boost * worker.p_aug) / 12.0

    # ── Phase 2b helper: target capacity ────────────────────────────────────

    def _compute_cstar(self) -> int:
        """Compute C* per occupation using industry-specific γ; cache and return total.

        C*_{j,o,t} = round(C_{j,o,0} * (1 + g_jt) * (1 - A_jt*R_job(o) + γ_naics*A_jt*P_aug(o)))

        γ_naics replaces the global γ — high-elasticity sectors (Information,
        Finance, Professional Services) expand capacity under augmentation;
        inelastic sectors (Admin Support, Retail, Food) contract.

        Side effect: writes self._cstar_this_tick (per-occ dict) for Phase 3.
        Lazy-initialises self._cap_by_occ on first call (workers attached after
        __init__).
        """
        p     = self.model.params
        a_jt  = self.a_jt
        g_jt  = self.btos_signal

        # NAICS-2 specific γ; fall back to legacy global gamma for missing sectors.
        gamma_table = p.get("gamma_by_naics", {}) or {}
        gamma = float(gamma_table.get(self.sector[:2], p.get("gamma", 0.3)))

        occ_risk = self.model.occ_risk_lookup
        r_by_occ = occ_risk.get("r_job", {})
        p_by_occ = occ_risk.get("p_aug", {})

        # Lazy-init C_{j,o,0} baseline on first call this tick if empty.
        if not self._cap_by_occ:
            for worker in self._roster:
                occ = worker.current_occ
                self._cap_by_occ[occ] = self._cap_by_occ.get(occ, 0) + 1

        cstar_by_occ: dict = {}
        total = 0
        for occ, c0 in self._cap_by_occ.items():
            r_occ = r_by_occ.get(occ, 0.5)
            p_occ = p_by_occ.get(occ, 0.3)
            # round() not floor() so small firms (c0=1-2) aren't driven to zero
            # by tiny negative BTOS fluctuations.
            cstar = round(c0 * (1.0 + g_jt) * (1.0 - a_jt * r_occ + gamma * a_jt * p_occ))
            cstar_by_occ[occ] = max(0, cstar)
            total += cstar_by_occ[occ]

        self._cstar_this_tick = cstar_by_occ
        return total

    # ── Phase 3: Vacancy generation ───────────────────────────────────────────

    def _generate_vacancies(self):
        """Derive vacancies from the cached C* (computed in Phase 2).

        V_{j,o,t} = max(0, C*_{j,o,t} - E_{j,o,t})

        New-economy vacancies V_new follow the Acemoglu-Restrepo (2018) CES
        task production framework rather than the prior linear σ-multiplier:

            Y_j = (∫_{N-1}^{N} y_j(i)^((σ-1)/σ) di)^(σ/(σ-1))

        Tasks below the automation margin are produced by capital, tasks above
        by labor.  The reinstatement boundary N evolves endogenously: as the
        cost of algorithmic capital falls and automation displaces labor on the
        intensive margin, the relative wage of labor on the (now cheaper)
        upper margin generates an economic incentive to pioneer new tasks.

        With σ ∈ [0.8, 1.5] the elasticity of substitution between tasks
        (Acemoglu & Restrepo 2018, 2020; Bessen 2019; per recent empirical
        calibrations σ ≈ 1.2 for cognitive-task economies):
            • σ > 1: gross substitutes — capital cost-savings disproportionately
              fund new task creation; reinstatement effect is amplified.
            • σ = 1: Cobb-Douglas baseline.
            • σ < 1: gross complements — automation suppresses reinstatement.

        Concretely, given the firm-level automated-mass aggregate
        Σ_o(A_jt · R_job(o) · C_{j,o,0}), the CES-derived reinstatement intensity
        is φ(σ, A_jt) = σ · (1 - A_jt)^(σ-1).  The new-task generation rate is
        then capped by a reinstatement-efficiency parameter η that absorbs
        institutional friction (training, R&D lag) — ABC-calibrated, not
        hard-coded to a literal 2% rate.

        C* is no longer recomputed here — _layoff_phase calls _compute_cstar()
        first, so this phase consumes self._cstar_this_tick directly.
        """
        p     = self.model.params
        sigma_elast = float(p.get("sigma_elast", p.get("sigma", 1.2)))
        # Backstop: if calibration grids pass the legacy σ ∈ [0, 0.1] range
        # (literal 2% rate), treat it as the reinstatement-efficiency η and
        # default σ_elast to 1.2.  This keeps the σ-sensitivity script
        # interpretable as a sweep of reinstatement efficiency rather than
        # a misappropriated elasticity.
        if sigma_elast < 0.5:
            reinstatement_efficiency = sigma_elast
            sigma_elast = float(p.get("sigma_elast_default", 1.2))
        else:
            reinstatement_efficiency = float(p.get("reinstatement_efficiency", 0.05))
        a_jt  = self.a_jt

        occ_risk = self.model.occ_risk_lookup
        r_by_occ = occ_risk.get("r_job", {})

        # Current employment per occupation (post-layoff state).
        emp_by_occ: dict = {}
        for worker in self._roster:
            if worker.is_employed:
                occ = worker.current_occ
                emp_by_occ[occ] = emp_by_occ.get(occ, 0) + 1

        total_vacancies = 0
        vac_by_occ: dict = {}
        for occ, cstar in self._cstar_this_tick.items():
            e_occ  = emp_by_occ.get(occ, 0)
            v_occ  = max(0, cstar - e_occ)
            if v_occ > 0:
                vac_by_occ[occ] = v_occ
                self.vacancy_age_by_occ[occ] = self.vacancy_age_by_occ.get(occ, 0) + 1
            else:
                self.vacancy_age_by_occ[occ] = 0
            total_vacancies += v_occ

        # V_new: new-economy vacancies from AI-automated roles, posted into
        # the empirically grounded "Frontier Basket" of high-end existing
        # OCC2010 codes (Autor, Salomons & Seegmiller 2021; Babina et al. 2024)
        # so unemployed workers can discover and retrain into them via the
        # radiation-model mobility kernel. Equal-split distribution with
        # deterministic rounding remainder allocation.
        if self.model.ai_active and a_jt > 0:
            auto_sum = sum(a_jt * r_by_occ.get(o, 0.5) * c0
                           for o, c0 in self._cap_by_occ.items())
            # CES reinstatement intensity: φ(σ, A_jt) = σ · (1 - A_jt)^(σ-1)
            #   σ=1.0  → φ = 1                     (Cobb-Douglas baseline)
            #   σ=1.5  → φ = 1.5·(1-A_jt)^0.5      (amplified at low A_jt)
            #   σ=0.8  → φ = 0.8·(1-A_jt)^(-0.2)
            labor_share = max(1e-3, 1.0 - a_jt)
            ces_phi     = sigma_elast * (labor_share ** (sigma_elast - 1.0))

            # Labor-abundance response ψ(σ, U_t/U_baseline): when aggregate
            # unemployment is elevated relative to the empirical baseline, the
            # relative price of labor on the new-task margin has fallen, and the
            # CES kernel says firms should pioneer more new tasks to exploit
            # cheap labor. Without this term, V_new is purely a function of
            # automation maturity and is therefore perfectly inelastic to the
            # state of the labor market — pinning the vacancy rate at a static
            # ceiling and flattening the AI-scenario Beveridge curve into a
            # horizontal line.  Using the previous tick's aggregate UR breaks
            # the same-tick simultaneity (firms can't see vacancies they're
            # about to post).
            ur_lagged = float(getattr(self.model, "_ur_lagged",
                                      p.get("ur_baseline", 0.045)))
            ur_baseline = float(p.get("ur_baseline", 0.045))
            labor_abundance = max(0.5, min(5.0, ur_lagged / max(ur_baseline, 1e-3)))
            # σ > 1 (gross substitutes): cheap labor amplifies new-task creation
            # σ = 1 (Cobb-Douglas):       no response
            # σ < 1 (gross complements):  cheap labor dampens new-task creation
            ces_psi = labor_abundance ** (sigma_elast - 1.0)

            v_new = round(reinstatement_efficiency
                          * ces_phi * ces_psi
                          * a_jt * auto_sum)
            if v_new > 0:
                self.model._new_economy_jobs_this_tick += v_new
                self.model._new_economy_jobs_cumulative += v_new
                basket = p.get("frontier_basket", (1006, 1010, 1020, 1240))
                if basket:
                    n = len(basket)
                    per_occ   = v_new // n
                    remainder = v_new - per_occ * n
                    for i, occ in enumerate(basket):
                        add = per_occ + (1 if i < remainder else 0)
                        if add > 0:
                            vac_by_occ[occ] = vac_by_occ.get(occ, 0) + add
                            total_vacancies += add
                            self.vacancy_age_by_occ[occ] = self.vacancy_age_by_occ.get(occ, 0)

        self._vacancies_by_occ = vac_by_occ
        self.vacancies = max(0, total_vacancies)

    # ── Phase 4: Firm state update ────────────────────────────────────────────

    def _update_firm_state(self):
        """Update Healthy/Distressed/Failed state based on aggregate C*.

        An employer is Distressed if its total target capacity across all
        occupations is <= 0. After tau_exit consecutive ticks of distress,
        it transitions to Failed and discharges all workers.
        """
        tau = self.model.params.get("tau_exit", 15)
        total_cstar = sum(self._cstar_this_tick.values()) if self._cstar_this_tick else self.capacity

        if total_cstar <= 0:
            self.distress_counter += 1
            self.state = "Distressed"
            if self.distress_counter >= tau:
                self._fail()
        else:
            self.distress_counter = 0
            self.state = "Healthy"

    def _fail(self):
        """Transition to Failed state: discharge all workers."""
        self.state = "Failed"
        self.vacancies = 0
        for worker in list(self._roster):
            if worker.is_employed:
                worker.is_employed       = False
                worker.months_unemployed = 0
                worker.just_fired        = True
                worker.employer          = None
                self._fired_this_tick   += 1   # count firm-failure discharges
            self._roster.discard(worker)

    # ── Phase 5: Market clearing ──────────────────────────────────────────────

    def _market_clearing(self):
        self._hired_this_tick = 0
        if self.vacancies <= 0:
            return

        # Retraining is a parallel state, not a lock-out.  Workers actively
        # retraining remain visible to employers and can be hired for vacancies
        # that match their current occupation.  If hired, is_employed flips to
        # True and retraining_ticks_left continues to decrement in the background
        # (see Worker._retrain(): months_unemployed and search_occ assignment are
        # already conditioned on not is_employed, so no further changes needed).
        global_seekers = [
            w for w in self.model.agents_by_type[WorkerAgent]
            if not w.is_employed
            and not w.is_retired
            and not w.just_fired
        ]

        if not global_seekers:
            return

        for occ, n_open in self._vacancies_by_occ.items():
            if n_open <= 0:
                continue

            # Workers are eligible for a vacancy in occ if they hold skills for it
            # via either their retrained occupation (search_occ) or their prior
            # occupation (current_occ).  Retraining adds skills; it doesn't void
            # prior experience.  This prevents the mismatch lock-up where retrained
            # workers are permanently excluded from their original occupation's
            # vacancies even when no search_occ positions are available.
            valid_candidates = [
                w for w in global_seekers
                if occ in ({w.search_occ, w.current_occ} - {None})
            ]

            if not valid_candidates:
                continue

            # Rank by match score × dynamic credential multiplier.
            # For credentialed candidates the multiplier is always 1.0.
            # For under-credentialed candidates the multiplier starts at 0.3 and
            # ramps linearly to 1.0 over 6 ticks of vacancy age, reflecting the
            # real-world pattern where employers progressively relax credential
            # requirements for hard-to-fill roles (SHRM/ManpowerGroup survey data:
            # ~60-75% of employers lower credential bars after 60-90 days unfilled).
            occ_min_cred_idx = self.model.occ_min_cred_idx.get(occ, 0)
            vac_age = self.vacancy_age_by_occ.get(occ, 0)
            cred_mult_under = min(1.0, 0.3 + (vac_age / 6.0) * 0.7)

            # Managerial-information noise (peer-review robustness): same
            # mechanism as in the layoff phase, applied to the firm's read of
            # each candidate's r_job/p_aug. The derived match-score components
            # (p_agent_aug, r_agent_sub) are recomputed locally from the noisy
            # observation, leaving the underlying worker attributes untouched.
            _params = self.model.params
            r_sd = _params.get("r_job_noise", 0.0)
            p_sd = _params.get("p_aug_noise", 0.0)
            d_sub = _params.get("delta_sub", 0.30)
            d_aug = _params.get("delta_aug", 0.40)
            if r_sd > 0.0 or p_sd > 0.0:
                _rng = self.model.random
                obs_match = {
                    w: (
                        max(0.0, min(1.0, w.r_job + _rng.gauss(0.0, r_sd))),
                        max(0.0, min(1.0, w.p_aug + _rng.gauss(0.0, p_sd))),
                    )
                    for w in valid_candidates
                }
            else:
                obs_match = {w: (w.r_job, w.p_aug) for w in valid_candidates}

            def _match_score(w, _min=occ_min_cred_idx, _mult=cred_mult_under,
                             _o=obs_match, _ds=d_sub, _da=d_aug):
                r_o, p_o = _o[w]
                p_agent_aug_obs = p_o * (1.0 + _da * w.exp_norm)
                r_agent_sub_obs = r_o * (1.0 - _ds * w.exp_norm)
                cred = 1.0 if w.credential_idx >= _min else _mult
                return p_agent_aug_obs * (1.0 - r_agent_sub_obs) * cred

            # Probabilistic matching (Mortensen-Pissarides 1994; review §
            # "Matching Gating and the Cascade Bump Artifact"): replaces the
            # deterministic perfect-information rank-and-cut with a noisy
            # ranking that captures interview performance, cultural fit, and
            # administrative friction.  Each candidate's deterministic match
            # score is multiplied by exp(ε), ε ~ N(0, σ_match), so the firm
            # observes a noisy estimate of fit at decision time.  When
            # match_noise_std = 0 this collapses to the original deterministic
            # baseline (preserved for ABC reproducibility and noise-sweep
            # robustness checks).
            match_noise_std = float(_params.get("match_noise_std", 0.15))
            if match_noise_std > 0.0:
                _rng = self.model.random
                def _noisy_score(w, _f=_match_score, _s=match_noise_std,
                                 _r=_rng):
                    base = _f(w)
                    return base * math.exp(_r.gauss(0.0, _s))
                ranked = sorted(valid_candidates, key=_noisy_score, reverse=True)
            else:
                ranked = sorted(valid_candidates, key=_match_score, reverse=True)

            # Q1 loss-decomposition (peer-review): for HSQ1 candidates who
            # competed for this vacancy but were not in the top n_open, classify
            # the loss as either credential-blocked (under-credentialed AND lost)
            # or cascade-bumped (qualified AND lost to a higher-ranked worker).
            _losers = ranked[n_open:]
            for _w in _losers:
                if _w.hard_skill_quintile != "HSQ1_Low":
                    continue
                if _w.credential_idx < occ_min_cred_idx:
                    self.model._q1_credential_blocked_this_tick += 1
                else:
                    self.model._q1_cascade_bumped_this_tick += 1

            hires_this_occ = 0
            for worker in ranked[:n_open]:
                if worker.employer is not None and worker.employer is not self:
                    worker.employer._roster.discard(worker)

                self.assign_worker(worker)
                worker.is_employed       = True
                worker.months_unemployed = 0
                worker.is_olf            = False  # job offer pulls student back into labor force

                if worker.search_occ is not None:
                    if occ == worker.search_occ:
                        # Worker filled a vacancy in their retrained occupation:
                        # complete the career pivot and update wage/zone tables.
                        worker.current_occ = worker.search_occ
                        worker.job_zone = self.model.job_zone_lookup.get(
                            worker.current_occ, worker.job_zone)
                        worker.w_base = self.model.occ_wage_lookup.get(
                            worker.current_occ, worker.w_base)
                    # Whether hired into search_occ or current_occ, the worker
                    # is employed again — clear the pending occupational redirect.
                    worker.search_occ = None

                self._hired_this_tick += 1
                hires_this_occ += 1
                global_seekers.remove(worker)

            # A successful hire for this occupation resets the desperation clock.
            # The vacancy_age_by_occ counter will also be zeroed in the next tick's
            # _generate_vacancies() if v_occ drops to 0, but resetting here ensures
            # the multiplier is correct if the same occupation posts again next tick.
            if hires_this_occ > 0:
                self.vacancy_age_by_occ[occ] = 0
