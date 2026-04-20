"""Employer agent for the AI Labor Market ABM.

Implements BTOS-modulated hiring/firing with a 5-phase step:
  1. BTOS signal update    — Ornstein-Uhlenbeck mean-reversion toward BTOS g_init anchor
  2. Layoff phase          — logit displacement modulated by A_{j,t} * R_job / P_aug
  3. Vacancy generation    — C* per occupation + V_new new-economy vacancies
  4. Firm state update     — Healthy / Distressed / Failed transitions
  5. Market clearing       — hire ranked unemployed workers by match score

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
        self.state            = "Healthy"   # Healthy | Distressed | Failed
        self.distress_counter = 0           # ticks with total C* <= 0

    # ── AI adoption maturity property ────────────────────────────────────────

    @property
    def a_jt(self) -> float:
        """Current AI adoption maturity for this firm.

        In the control scenario (ai_active=False): always 0.
        In the AI scenario: logistic growth from (a_adoption * ai_modifier),
        capped at a_max.

        Formula: a(t) = a_max / (1 + ((a_max - a0) / a0) * exp(-k * t))
        where a0 = a_adoption * ai_modifier (firm-specific initial level).
        """
        if not self.model.ai_active:
            return 0.0
        p     = self.model.params
        k     = p.get("k_adoption", 0.05)
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

        # Anchor (μ_j): sector g_init from BTOS, fall back to legacy _SECTOR_DRIFT
        btos_data = self.model._btos_sector.get(self.sector[:2], {})
        mu        = btos_data.get("g_init", _SECTOR_DRIFT.get(self.sector[:2], _DEFAULT_DRIFT))

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

    # ── Phase 2: Layoff ──────────────────────────────────────────────────────

    def _layoff_phase(self):
        p = self.model.params
        # BTOS modulates layoffs, but with a dampener so negative macro cycles
        # don't overwhelm baseline turnover.  btos_disp_damp=0 → BTOS has no
        # effect on separations; 1.0 → full pass-through (old behaviour).
        btos_damp = p.get("btos_disp_damp", 0.5)
        eff_base = float(np.clip(
            p["delta_base"] * (1.0 - btos_damp * self.btos_signal), 1e-9, 1.0 - 1e-9
        ))
        c = math.log(eff_base / (1.0 - eff_base))   # logit(eff_base)
        self._fired_this_tick = 0
        a_jt = self.a_jt  # compute once per tick

        # Reset temporal friction flag before evaluating layoffs this tick.
        for worker in self._roster:
            worker.just_fired = False

        for worker in [w for w in self._roster if w.is_employed]:
            if self.model.ai_active:
                # P(D) = sigmoid(logit(δ_base) + β1*(A_jt*R_job) - β2*(A_jt*P_aug) - β3*E_i)
                beta1 = p.get("beta1",    p.get("beta", 3.5))
                beta2 = p.get("beta2",    p.get("lambda_", 0.5))
                beta3 = p.get("beta3_exp", 0.3)
                Z     = (c
                         + beta1 * (a_jt * worker.r_job)
                         - beta2 * (a_jt * worker.p_aug)
                         - beta3 * worker.exp_norm)
                prob  = 1.0 / (1.0 + math.exp(-Z))
            else:
                prob = eff_base

            if self.random.random() < prob:
                worker.is_employed       = False
                worker.months_unemployed = 0
                worker.just_fired        = True
                self._fired_this_tick   += 1
                if self.model.ai_active:
                    self.model._displacement_this_tick += 1
                    worker._choose_target_skill()
            elif self.model.ai_active:
                monthly_boost = (p.get("wage_boost", 0.02) * worker.p_aug) / 12.0
                worker.wage  *= 1.0 + monthly_boost

    # ── Phase 3: Vacancy generation ───────────────────────────────────────────

    def _generate_vacancies(self):
        """Generate vacancies using C* formula per occupation.

        C*_{j,o,t} = floor(C_{j,o,0} * (1 + g_jt) * (1 - A_jt*R_job(o) + γ*A_jt*P_aug(o)))
        V_{j,o,t}  = max(0, C*_{j,o,t} - E_{j,o,t})
        V_new      = floor(σ * A_jt * Σ_o(A_jt * R_job(o) * C_{j,o,0}))
        """
        p     = self.model.params
        gamma = p.get("gamma", 0.3)
        sigma = p.get("sigma", 0.02)
        a_jt  = self.a_jt
        g_jt  = self.btos_signal   # BTOS signal = g_{j,t}

        occ_risk = self.model.occ_risk_lookup
        r_by_occ = occ_risk.get("r_job", {})
        p_by_occ = occ_risk.get("p_aug", {})

        # Initialize _cap_by_occ on first tick (tick 0) — workers are assigned
        # after __init__, so we capture C_{j,o,0} lazily here.
        if not self._cap_by_occ:
            for worker in self._roster:
                occ = worker.current_occ
                self._cap_by_occ[occ] = self._cap_by_occ.get(occ, 0) + 1

        # Current employment per occupation
        emp_by_occ: dict = {}
        for worker in self._roster:
            if worker.is_employed:
                occ = worker.current_occ
                emp_by_occ[occ] = emp_by_occ.get(occ, 0) + 1

        # C* per occupation and vacancy generation.
        # _cap_by_occ is the FIXED C_{j,o,0} baseline — never overwritten here.
        # _cstar_this_tick is refreshed each tick and consumed by _update_firm_state.
        total_vacancies = 0
        cstar_by_occ: dict = {}
        vac_by_occ: dict = {}
        for occ, c0 in self._cap_by_occ.items():
            r_occ = r_by_occ.get(occ, 0.5)
            p_occ = p_by_occ.get(occ, 0.3)
            # Use round() not floor() so that small firms (c0=1-2 per occupation)
            # are not driven to cstar=0 by tiny negative BTOS fluctuations.
            # With BTOS clipped to [-0.15, 0.15]: round(1 × 0.85) = 1 (never 0).
            # Large firms still contract proportionally: round(10 × 0.85) = 9.
            cstar = round(c0 * (1.0 + g_jt) * (1.0 - a_jt * r_occ + gamma * a_jt * p_occ))
            cstar_by_occ[occ] = max(0, cstar)
            e_occ  = emp_by_occ.get(occ, 0)
            v_occ  = max(0, cstar_by_occ[occ] - e_occ)
            if v_occ > 0:
                vac_by_occ[occ] = v_occ
            total_vacancies += v_occ

        # Cache this-tick C* (for firm-state check) and per-occ vacancies.
        # Do NOT touch _cap_by_occ — it holds the fixed C_{j,o,0} baseline.
        self._cstar_this_tick  = cstar_by_occ
        self._vacancies_by_occ = vac_by_occ

        # V_new: new economy vacancies from AI-automated roles.
        # Tracked for the data collector but NOT added to self.vacancies or
        # _vacancies_by_occ — there is no occupation-matched pool for these
        # slots, so counting them as posted vacancies would inflate the
        # Beveridge curve with phantom demand that can never be filled.
        if self.model.ai_active and a_jt > 0:
            auto_sum = sum(a_jt * r_by_occ.get(o, 0.5) * c0
                           for o, c0 in self._cap_by_occ.items())
            v_new = round(sigma * a_jt * auto_sum)
            if v_new > 0:
                self.model._new_economy_jobs_this_tick += v_new

        self.vacancies = max(0, total_vacancies)  # matchable vacancies only

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

            # Rank by match score × credential discount.
            # Workers below the occupation's minimum credential are not excluded
            # (employers do hire under-credentialed workers, especially in tight
            # markets) but they face a 70% score penalty, making credentialed
            # workers strongly preferred. This reflects real-world employer
            # credential screening without creating structural unemployment traps.
            occ_min_cred_idx = self.model.occ_min_cred_idx.get(occ, 0)
            ranked = sorted(
                valid_candidates,
                key=lambda w: (
                    w.p_agent_aug * (1.0 - w.r_agent_sub)
                    * (1.0 if w.credential_idx >= occ_min_cred_idx else 0.3)
                ),
                reverse=True,
            )

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
                global_seekers.remove(worker)
