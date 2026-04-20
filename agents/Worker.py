"""Worker agent for the AI Labor Market ABM.

Credential system
-----------------
Workers hold a ``credential`` attribute (one of the strings in
``CREDENTIAL_LEVELS``) derived from their IPUMS CPS EDUC code at
initialisation.  When retraining, the time cost now includes the DAG
path from the worker's current credential to the minimum credential
required by the target occupation (via ``credential_months_to``).
On retraining completion the credential is upgraded to ``target_credential``.
Young workers (age ≤ 22, credential == "high_school") receive an education-
pipeline multiplier on their retraining entry probability.

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

from model.credentials import (
    educ_to_credential, credential_months_to,
    ZONE_MIN_CREDENTIAL, CREDENTIAL_IDX, CRED_DIST_MATRIX,
)


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

        # ── Credential system ──
        # Initial credential derived from IPUMS CPS EDUC code.
        # target_credential is set when a retraining path begins and cleared
        # (by upgrading self.credential) when retraining completes.
        self.credential        = educ_to_credential(self.educ)
        self.credential_idx    = CREDENTIAL_IDX.get(self.credential, 0)  # cached int for fast comparisons
        self.target_credential: str | None = None

        # ── Out-of-Labor-Force (OLF) flag ──
        # Workers pursuing a formal credential upgrade (e.g., going back to
        # school for an associate's or bachelor's degree) are classified as OLF,
        # not unemployed, mirroring BLS methodology: full-time students who are
        # not actively seeking work are excluded from both the numerator and
        # denominator of the unemployment rate.  The flag is set in
        # _choose_target_skill() when an unemployed worker begins a credential
        # path, and cleared on hiring or on retraining completion.
        self.is_olf = False

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
        # Young workers (≤22) with only a HS credential are disproportionately
        # likely to enter the education pipeline — reflecting real-world patterns
        # where recent high-school graduates pursue post-secondary credentials
        # at much higher rates than mid-career workers.
        # Multiplier: 2.0 at age 22, rising to ~4.0 at age 18.
        if self.age <= 22 and self.credential == "high_school":
            age_edu_mult = 2.0 + max(0.0, 22 - self.age) * 0.5
            pr = min(1.0, pr * age_edu_mult)
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
        T_retrain = C(credential_src → credential_tgt) + ceil(ω · d(occ_i, occ_j))

        The credential gap component is now computed via BFS over the credential
        DAG (HS → Vocational → Associates → Bachelors → Masters → Doctoral),
        replacing the old zone_ticks linear table.  This means a HS-diploma
        worker targeting a bachelor's-zone occupation pays 48 months of
        credential time, while an associate's-degree worker pays only 24 months
        — making the gravity model realistically non-linear.
        """
        model = self.model
        p     = self.params
        mu       = p.get("mu", 5.0)
        omega    = p.get("omega", 0.5)

        # Use precomputed arrays from model init — avoids per-call pandas .loc
        # and 537-item Python list comprehensions on every retraining event.
        if model._dist_array is None or self.current_occ not in model._cand_occ_to_row:
            return

        candidates = model._cand_occs                           # list[int], len=537
        row_idx    = model._cand_occ_to_row[self.current_occ]
        d_row      = model._dist_array[row_idx].astype(float)  # numpy row slice — no pandas
        r          = model._cand_r_arr.astype(float)            # precomputed static
        v          = model._cand_vacancy_arr.astype(float)     # rebuilt each tick in _update_effective_vacancies

        # Credential gap — single numpy fancy-index into 6×6 distance matrix
        cred_months = CRED_DIST_MATRIX[
            self.credential_idx, model._cand_min_cred_idx_arr
        ].astype(float)
        occ_min_cred_idx = model.occ_min_cred_idx  # still needed for chosen-occ lookup below

        # Total retraining time: credential path + semantic skill distance
        t_retrain = cred_months + np.ceil(omega * d_row)

        scores = v * (1.0 - r) * np.exp(-mu * t_retrain)
        total  = scores.sum()
        if total <= 0:
            return

        probs  = scores / total
        # Use cumsum + searchsorted: avoids converting numpy array to Python list
        # (self.random.choices requires a list, which is expensive at 537 elements).
        # self.random.random() keeps us on the Mesa-seeded RNG for reproducibility.
        chosen = candidates[int(np.searchsorted(np.cumsum(probs), self.random.random())
                                .clip(0, len(candidates) - 1))]
        self.target_occ = chosen

        if chosen != self.current_occ:
            col_idx      = model._cand_occ_to_col.get(chosen, -1)
            dist         = float(model._dist_array[row_idx, col_idx]) if col_idx >= 0 else 1.0
            tgt_cred_idx = int(occ_min_cred_idx.get(chosen, 0))
            cred_pen     = int(CRED_DIST_MATRIX[self.credential_idx, tgt_cred_idx])
            skill_pen    = max(1, math.ceil(omega * dist))
            self.retraining_ticks_left = max(1, cred_pen + skill_pen)
            # Store the credential level this path is working toward so
            # _retrain() can upgrade self.credential on completion.
            self.target_credential = (CREDENTIAL_LEVELS[tgt_cred_idx]
                                      if cred_pen > 0 else None)

            # Unemployed workers pursuing a formal credential upgrade become OLF
            # (full-time students), consistent with BLS methodology.  They remain
            # visible to employers (a job offer can pull them back) but are not
            # counted in the unemployment rate while enrolled.
            if not self.is_employed and self.target_credential is not None:
                self.is_olf = True

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

            # Upgrade credential if this retraining path required one.
            if self.target_credential is not None:
                tgt_idx = CREDENTIAL_IDX.get(self.target_credential, 0)
                if tgt_idx > self.credential_idx:
                    self.credential     = self.target_credential
                    self.credential_idx = tgt_idx
            self.target_credential = None
            self.is_olf = False  # re-enter labor force on program completion

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
