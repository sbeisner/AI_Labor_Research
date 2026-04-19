"""Worker agent for the AI Labor Market ABM.

Implements the displacement, augmentation, retraining, and wage equations
from the manuscript.

Displacement probability (sigmoid form):
    P(D) = sigmoid(logit(δ_base) + β1*(A_jt*R_job) - β2*(A_jt*P_aug) - β3*E_i)

Augmented Mincer wage equation:
    ln(W) = ln(W_base(o)) + r*Z_i + β1*E_i - β2*E_i^2

Retraining gravity model (with credential gap):
    P(s_j|S_i) ∝ V(s_j) · (1-R_job(s_j)) · exp(−μ · T_retrain(S_i, s_j))
    T_retrain = C(Z_i, Z_j) + ceil(ω · min_{s∈S_i} d(s, s_j))

Spin-off trigger:
    P(B) = λ · (1 − exp(−ψ·E_i)) · max(0, g_jt)

Poisson matching function:
    P(H) = 1 − exp(−ρ · θ(occ, t))

Properties are initialized from IPUMS CPS microdata via
worker_sample_with_risk.parquet.
"""

import math

import numpy as np
import mesa


class WorkerAgent(mesa.Agent):
    """Represents one worker in the US labor force.

    Demographic attributes are drawn from IPUMS CPS microdata (static).
    r_job / p_aug come from the O*NET-based risk scoring pipeline.
    Employment status, current occupation, and wage evolve each tick.
    """

    def __init__(self, model, row, params):
        super().__init__(model)

        # ── Static demographic attributes (from CPS) ──
        self.age               = int(row["AGE"])
        self.sex               = int(row["SEX"])
        self.race              = int(row["RACE"])
        self.educ              = int(row["EDUC"])
        self.naics_sector      = row["NAICS_sector"]
        self.ind1990           = str(row.get("IND1990", row["NAICS_sector"]))
        self.exposure_quintile = row["exposure_quintile"]

        # ── Occupation ──
        # current_occ: the occupation this worker is currently in / most recently held.
        # search_occ:  after retraining, the occupation whose skills they just acquired
        #              and are now targeting in the job market. None until retraining
        #              completes. current_occ only updates to search_occ once the worker
        #              is actually hired — retraining adds skills, it doesn't teleport
        #              the worker into a new occupation.
        self.current_occ = int(row["OCC2010"])
        self.search_occ  = None   # set after retraining; cleared on hire

        # ── Risk / augmentation scores (blend toward target after retraining) ──
        self.r_job    = float(row["r_job"])
        self.p_aug    = float(row["p_aug"])
        self.h_job    = float(row["h_job"])
        self.exp_norm = float(row["exp_norm"])

        # ── Dynamic state — initialized from CPS EMPSTAT ──
        # EMPSTAT 10 (at work) / 12 (has job, not at work) → employed
        # EMPSTAT 21 (unemployed, experienced) / 22 (new entrant) → unemployed
        self.is_employed       = bool(row.get("is_employed",
                                    int(row.get("EMPSTAT", 10)) in (10, 12)))
        self.months_unemployed = 0 if self.is_employed else 1
        self.wage              = float(row["wage"])

        # ── Retraining state ──
        self.target_occ            = self.current_occ
        self.retraining_ticks_left = 0
        self.has_retrained         = False

        # Temporal friction flag: True for the tick in which this worker
        # was just fired. Prevents same-tick rehire — the worker cannot enter
        # market clearing until the following tick.
        self.just_fired = False

        self.params   = params
        self.employer = None   # set by EmployerAgent.assign_worker()

        # ── New attributes ──
        self.job_zone     = int(row.get("job_zone", 3))          # O*NET Job Zone (1-5)
        self.w_base       = float(row.get("w_base", row["wage"])) # OEWS baseline annual wage ($K)
        self.is_retired   = False   # terminal state
        self.is_retraining = False  # can be employed AND retraining simultaneously

        # Initialise wage from Mincer equation for unit consistency.
        # (row["wage"] may be in raw dollars; w_base is in $K from OEWS.)
        self.compute_mincer_wage()

    # ── Derived risk quantities ──────────────────────────────────────────────

    @property
    def r_agent_sub(self):
        """Individualized substitution risk — experience shields from replacement."""
        return self.r_job * (1.0 - self.params.get("delta_sub", 0.30) * self.exp_norm)

    @property
    def p_agent_aug(self):
        """Individualized augmentation potential — experience amplifies gains."""
        return self.p_aug * (1.0 + self.params.get("delta_aug", 0.40) * self.exp_norm)

    @property
    def p_disp(self):
        """Probability of displacement this tick.

        P(D) = sigmoid(logit(δ_base) + β1*(A_jt*R_job) - β2*(A_jt*P_aug) - β3*E_i)

        A_jt comes from the employer. For workers not attached to an employer
        (open-market), A_jt = 0, reducing to the baseline rate.
        """
        p   = self.params
        db  = p["delta_base"]
        c   = math.log(db / (1.0 - db))  # logit(δ_base)
        if not self.model.ai_active:
            return db
        a_jt = getattr(self.employer, "a_jt", 0.0) if self.employer else 0.0
        beta1     = p.get("beta1",     p.get("beta", 3.5))
        beta2     = p.get("beta2",     p.get("lambda_", 0.5))
        beta3     = p.get("beta3_exp", 0.3)
        Z = c + beta1 * (a_jt * self.r_job) - beta2 * (a_jt * self.p_aug) - beta3 * self.exp_norm
        return float(1.0 / (1.0 + math.exp(-Z)))

    # ── Wage equation ────────────────────────────────────────────────────────

    def compute_mincer_wage(self):
        """Recalculate wage from augmented Mincer equation.

        ln(W) = ln(W_base(o)) + r*Z_i + β1*E_i - β2*E_i^2
        W_base is in annual $K; result stored in self.wage as annual $K.
        Public sector workers use dampened r and β1 coefficients.
        """
        p       = self.params
        r_edu   = p.get("r_edu",     0.09)
        mb1     = p.get("mincer_b1", 0.04)
        mb2     = p.get("mincer_b2", 0.002)
        # Public sector: compress premium schedule (narrower distribution)
        if str(getattr(self, "naics_sector", "")) == "92":
            damp  = p.get("pub_wage_damp", 0.6)
            r_edu *= damp
            mb1   *= damp
        ln_w = (math.log(max(self.w_base, 1.0))
                + r_edu * self.job_zone
                + mb1 * self.exp_norm
                - mb2 * self.exp_norm ** 2)
        self.wage = math.exp(ln_w)

    # ── Step logic ───────────────────────────────────────────────────────────

    def step(self):
        if self.is_retired:
            return

        if self.employer is not None:
            # Displacement/hiring handled by EmployerAgent
            if self.retraining_ticks_left > 0:
                self._retrain()
            if self.is_employed:
                self._accumulate_experience()
                self._maybe_proactive_upskill()
                self._maybe_spinoff()       # spin-off trigger
                self.compute_mincer_wage()  # update wage each tick
            else:
                self.months_unemployed += 1
                if self.retraining_ticks_left == 0:
                    self._maybe_retrain_unemployed()
            return

        # Open-market workers (no employer attachment)
        if self.is_employed:
            self._check_displacement()
            self._accumulate_experience()
            self._maybe_proactive_upskill()
            self.compute_mincer_wage()
        elif self.retraining_ticks_left > 0:
            self._retrain()
        else:
            # Workers wait to be claimed by Employer._market_clearing()
            self.months_unemployed += 1
            self._maybe_retrain_unemployed()

    # ── Displacement ─────────────────────────────────────────────────────────

    def _check_displacement(self):
        p    = self.params
        prob = self.p_disp if self.model.ai_active else p["delta_base"]

        if self.random.random() < prob:
            self.is_employed = False
            self.months_unemployed = 0
            if self.model.ai_active:
                # Signal to model for job-creation elasticity accounting
                self.model._displacement_this_tick += 1
                self._choose_target_skill()
        elif self.model.ai_active:
            monthly_boost = (p["wage_boost"] * self.p_agent_aug) / 12.0
            self.wage *= 1.0 + monthly_boost

    # ── Retraining ───────────────────────────────────────────────────────────

    def _maybe_retrain_unemployed(self):
        """Evaluate probability of starting retraining while unemployed.

        P(UR) = (η_unemp + κ·R_job + ξ·ln(1+D_i)) · (1 − E_i)
        """
        if self.retraining_ticks_left > 0:
            return
        p  = self.params
        d  = self.months_unemployed
        pr = ((p.get("eta_unemp", 0.05)
               + p.get("kappa", 0.06) * self.r_job
               + p.get("xi", 0.03) * math.log(1.0 + d))
              * (1.0 - self.exp_norm))
        pr = max(0.0, min(1.0, pr))
        if self.random.random() < pr:
            self.is_retraining = True
            self._choose_target_skill()

    def _maybe_proactive_upskill(self):
        """Evaluate probability of proactive upskilling while employed.

        P(U) = (η_base + κ·R_job) · (1 − E_i)
        """
        if self.retraining_ticks_left > 0:
            return
        p  = self.params
        pu = ((p.get("eta_base", 0.02)
               + p.get("kappa", 0.06) * self.r_job)
              * (1.0 - self.exp_norm))
        pu = max(0.0, min(1.0, pu))
        if self.random.random() < pu:
            self.is_retraining = True
            self._choose_target_skill()

    def _choose_target_skill(self):
        """Select target occupation via gravity model:

        P(s_j|S_i) ∝ V(s_j) · (1-R_job(s_j)) · exp(−μ · T_retrain(S_i, s_j))
        T_retrain = C(Z_i, Z_j) + ceil(ω · min_{s∈S_i} d(s, s_j))
        """
        model       = self.model
        dist_matrix = model.skill_distance_matrix
        occ_risk    = model.occ_risk_lookup
        vacancies   = model.effective_vacancy_counts
        p           = self.params
        mu          = p.get("mu", 5.0)
        omega       = p.get("omega", 0.5)
        # Credential gap table: index = zone gap (0-4), value = ticks penalty
        zone_ticks  = p.get("zone_ticks", [0, 6, 12, 24, 36])

        if dist_matrix is None or self.current_occ not in dist_matrix.index:
            return

        candidates = dist_matrix.columns.tolist()
        d_row      = dist_matrix.loc[self.current_occ, candidates].values.astype(float)
        v          = np.array([max(1, vacancies.get(c, 1)) for c in candidates], dtype=float)
        r          = np.array([occ_risk["r_job"].get(c, 0.5) for c in candidates], dtype=float)

        # Job zones for candidate occupations
        jz_lookup  = getattr(model, "job_zone_lookup", {})
        z_i        = self.job_zone
        z_j_arr    = np.array([jz_lookup.get(c, 3) for c in candidates], dtype=int)
        # C(Z_i, Z_j): ticks for credential gap (only penalize upward transitions)
        gap_arr    = np.clip(z_j_arr - z_i, 0, len(zone_ticks) - 1)
        cred_ticks = np.array([zone_ticks[g] for g in gap_arr], dtype=float)
        # Total retraining time: credential gap + skill distance component
        t_retrain  = cred_ticks + np.ceil(omega * d_row)

        scores = v * (1.0 - r) * np.exp(-mu * t_retrain)
        total  = scores.sum()
        if total <= 0:
            return

        probs  = scores / total
        chosen = self.random.choices(candidates, weights=probs.tolist(), k=1)[0]
        self.target_occ = chosen

        if chosen != self.current_occ:
            dist           = float(dist_matrix.loc[self.current_occ, chosen])
            z_j            = jz_lookup.get(chosen, 3)
            gap            = max(0, z_j - self.job_zone)
            cred_pen       = zone_ticks[min(gap, len(zone_ticks) - 1)]
            skill_pen      = max(1, math.ceil(omega * dist))
            self.retraining_ticks_left = max(1, cred_pen + skill_pen)

    def _retrain(self):
        """Count down retraining period; update skill profile when complete.

        Intra-firm human capital accumulation: employed workers who upskill
        stay at their current firm — retraining does not force a resignation.
        Only unemployed workers increment months_unemployed during retraining.

        At completion:
          - Risk scores blend toward target occupation (r_job drops, p_aug rises).
          - search_occ is set so the worker can target the new role if displaced.
          - Employed workers remain on their employer's roster with updated skills.
          - is_retraining flag is cleared.
        """
        self.retraining_ticks_left -= 1

        # Only unemployed workers accumulate unemployment duration during retraining
        if not self.is_employed:
            self.months_unemployed += 1

        if self.retraining_ticks_left == 0:
            model   = self.model
            new_occ = self.target_occ
            alpha   = self.params.get("retrain_blend", 0.7)

            # Blend risk scores toward target: r_job drops, p_aug rises
            if new_occ in model.occ_risk_lookup["r_job"]:
                target_r = model.occ_risk_lookup["r_job"][new_occ]
                target_p = model.occ_risk_lookup["p_aug"][new_occ]
                self.r_job = alpha * target_r + (1.0 - alpha) * self.r_job
                self.p_aug = alpha * target_p + (1.0 - alpha) * self.p_aug

            # Only unemployed workers get a hard occupational redirect.
            # Employed workers keep their incumbent identity but retain the
            # blended r_job / p_aug benefit — career pivot comes from
            # disruption, not from background upskilling.
            if not self.is_employed:
                self.search_occ = new_occ
            self.has_retrained = True
            self.is_retraining = False
            # Employed workers stay on their roster — no detachment

    # ── Experience & aging ───────────────────────────────────────────────────

    def _accumulate_experience(self):
        """Gain one tick of experience; advance age by one year every 12 ticks."""
        self.exp_norm = min(1.0, self.exp_norm + 1.0 / 480.0)
        # Age advances ~1 year every 12 ticks
        if self.model.tick % 12 == 0:
            self.age += 1

    # ── Spin-off trigger ─────────────────────────────────────────────────────

    def _maybe_spinoff(self):
        """Evaluate spin-off probability for employed private-sector workers.

        P(B) = λ · (1 − exp(−ψ·E_i)) · max(0, g_jt)
        Only fires for private sector (not NAICS 92) with g_jt > 0.
        """
        if str(getattr(self, "naics_sector", "")) == "92":
            return
        employer = self.employer
        if employer is None:
            return
        g_jt = getattr(employer, "btos_signal", 0.0)
        if g_jt <= 0:
            return
        p   = self.params
        lam = p.get("lambda_spinoff", 0.001)
        psi = p.get("psi", 3.0)
        pb  = lam * (1.0 - math.exp(-psi * self.exp_norm)) * g_jt
        if self.random.random() < pb:
            self.model._trigger_spinoff(self)

    # ── Retirement ───────────────────────────────────────────────────────────

    def evaluate_retirement(self):
        """Evaluate whether this worker retires (called by model at end of tick).

        Returns True if worker retires (caller handles roster release).
        """
        p   = self.params
        tau = p.get("tau_retire", 55)
        if self.age < tau:
            return False
        a_r = p.get("alpha_retire", -3.0)
        b_a = p.get("beta_age",      0.15)
        b_w = p.get("beta_wealth",   0.1)
        Z   = a_r + b_a * (self.age - tau) + b_w * math.log(max(self.wage, 1.0))
        p_r = 1.0 / (1.0 + math.exp(-Z))
        return self.random.random() < p_r
