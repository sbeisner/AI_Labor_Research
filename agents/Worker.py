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

Augmented Mincer wage equation (quartic in raw chronological years X_i):
    ln(W) = ln(W_base(o)) + r·Z_i + β1·X_i + β2·X_i² + β3·X_i³ + β4·X_i⁴

Following Lemieux (2006) and Heckman-Lochner-Todd (2006), the quartic
captures the concave but non-monotonic life-cycle earnings profile that a
simple quadratic (in fractional [0,1] experience) cannot represent.  When
firms post heterogeneous coefficient draws (mincer_firm_std > 0), the
β's vary stochastically across employers — high-tacit-knowledge firms
draw larger linear premia, organically incentivising senior retention.

Retraining (radiation model for occupation choice):
    P(i → j) = (V_i · V_j) / ((V_i + S_ij) · (V_i + S_ij + V_j))
    where S_ij = Σ_{k: d(i,k) < d(i,j)} V_k  (intervening opportunities)
    Retraining time T_retrain = C(Z_i, Z_j) + ceil(ω · d(i,j))
    is computed separately and only affects retraining_ticks_left.

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
    ZONE_MIN_CREDENTIAL, CREDENTIAL_IDX, CREDENTIAL_LEVELS, CRED_DIST_MATRIX,
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
        # Hard-skill proximity quintile (anchor-based score over O*NET Work
        # Activities semantic graph; see scripts/build_hard_skill_scores.py).
        # Captures "routine hard skill" cohort for RQ4 — programming, technical
        # writing, accounting, paralegal-class codified cognitive work.
        self.hard_skill_quintile = str(row.get("hard_skill_quintile", "Unknown"))

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
        # Raw chronological years of labor-force experience (uncapped). Used by
        # the quartic Mincer wage equation per Lemieux (2006). exp_norm is
        # retained as the [0,1]-bounded shield/risk modulator (it appears in
        # r_agent_sub, p_agent_aug, displacement hazard, retirement, etc.) but
        # is no longer the wage-equation input. Initialised from the CPS
        # exp_norm * 40 (saturation horizon) so existing seeded distributions
        # remain consistent at tick 0.
        max_years_anchor = float(params.get("experience_years_max", 40.0))
        self.experience_years = float(row.get("experience_years",
                                              self.exp_norm * max_years_anchor))

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
        self.retraining_total      = 0     # set when a path begins; for dropout-hazard duration term
        self.retrain_d             = 0.0   # semantic distance d(s_i, s_j) of current retraining path
        self.has_retrained         = False
        self.has_dropped_out       = False  # ever experienced a retraining dropout

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
        """Recalculate wage from quartic augmented Mincer equation.

        ln(W) = ln(W_base(o)) + r·Z_i + β1·X_i + β2·X_i² + β3·X_i³ + β4·X_i⁴

        X_i is RAW chronological years of experience (uncapped), per Lemieux
        (2006).  The quartic generates the empirically observed concave-then-
        plateau life-cycle earnings profile (≈ 40-60 % peak premium) that the
        prior fractional-quadratic specification structurally suppressed.

        When the worker's employer draws stochastic coefficient adjusters
        (mincer_b1_adj … mincer_b4_adj, populated when mincer_firm_std > 0),
        firm-specific tacit-knowledge premia replace the deterministic
        coefficient table — letting heterogeneous senior-retention incentives
        emerge organically rather than being globally hard-coded.

        W_base is in annual $K; result stored in self.wage as annual $K.
        Public sector workers use dampened r and β1 coefficients.
        """
        p     = self.params
        r_edu = p.get("r_edu", 0.09)

        # Quartic Mincer coefficients (Lemieux 2006 calibration; defaults
        # produce ~50-65 % peak log-wage premium around X=30 then decline).
        b1 = p.get("mincer_beta1",  0.060)
        b2 = p.get("mincer_beta2", -0.0020)
        b3 = p.get("mincer_beta3",  0.00003)
        b4 = p.get("mincer_beta4", -0.00000020)

        # Firm-specific stochastic coefficient adjustments (heterogeneous
        # experience premia).  Drawn once per employer at firm init when
        # mincer_firm_std > 0; otherwise default 1.0 multiplier.
        emp = self.employer
        if emp is not None:
            b1 *= getattr(emp, "mincer_b1_adj", 1.0)
            b2 *= getattr(emp, "mincer_b2_adj", 1.0)
            b3 *= getattr(emp, "mincer_b3_adj", 1.0)
            b4 *= getattr(emp, "mincer_b4_adj", 1.0)

        # Public sector: compress premium schedule (narrower distribution)
        if str(getattr(self, "naics_sector", "")) == "92":
            damp = p.get("pub_wage_damp", 0.6)
            r_edu *= damp
            b1   *= damp

        x  = float(self.experience_years)
        x2 = x * x
        x3 = x2 * x
        x4 = x3 * x
        ln_w = (math.log(max(self.w_base, 1.0))
                + r_edu * self.job_zone
                + b1 * x + b2 * x2 + b3 * x3 + b4 * x4)
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

    def _check_retraining_dropout(self) -> bool:
        """Evaluate the per-tick retraining dropout hazard. Returns True if dropped out.

        On dropout the worker re-enters the active labor pool: retraining state
        is wiped, target_credential is cleared, is_olf is False, but
        months_unemployed is preserved so the worker faces full desperation
        pressure in the next market-clearing pass.
        """
        if self.retraining_ticks_left <= 0:
            return False
        p = self.params
        a0 = p.get("dropout_alpha0",      -3.5)
        a1 = p.get("dropout_alpha_dur",    0.05)
        a2 = p.get("dropout_alpha_unemp",  0.02)
        a3 = p.get("dropout_alpha_dist",   1.5)

        ticks_in = max(0, self.retraining_total - self.retraining_ticks_left)
        z = (a0
             + a1 * ticks_in
             + a2 * self.months_unemployed
             + a3 * self.retrain_d)
        # Clip z to avoid math.exp overflow for very long retraining paths
        z = max(-30.0, min(30.0, z))
        p_drop = 1.0 / (1.0 + math.exp(-z))

        if self.random.random() >= p_drop:
            return False

        # Drop out: abandon the credential path, return to active labor pool.
        self.retraining_ticks_left = 0
        self.retraining_total      = 0
        self.retrain_d             = 0.0
        self.target_credential     = None
        self.search_occ            = None
        self.target_occ            = self.current_occ
        self.is_olf                = False
        self.is_retraining         = False
        self.has_dropped_out       = True
        # months_unemployed is intentionally preserved — the worker has
        # been out of the active labor force, but their financial pressure
        # accumulates regardless. Their next market-clearing pass uses the
        # accumulated duration to trigger the desperation-multiplier ladder.
        if hasattr(self.model, "_dropouts_this_tick"):
            self.model._dropouts_this_tick += 1
        return True

    def _choose_target_skill(self):
        """Select target occupation via radiation model (Simini et al. 2012).

        P(i → j) = (V_i · V_j) / ((V_i + S_ij) · (V_i + S_ij + V_j))

        where V_i, V_j are the available vacancies at the worker's current
        occupation (origin) and the candidate occupation (destination), and
        S_ij is the total vacancies in all occupations whose semantic distance
        from i is strictly less than d(i, j) — the "intervening opportunities"
        the worker would consider before committing to j.

        Unlike the gravity model it replaces, the radiation model is
        parameter-free (no μ or ω weighting decay): transition probabilities
        derive purely from the spatial / semantic distribution of opportunity.
        This dismantles the deterministic cascade-bump waterfall, in which
        higher-credentialed workers reliably out-bid lower-skill incumbents in
        every localised contest — that was a computational artefact of the
        gravity model's perfect-information rank-and-cut, not an emergent
        labor-market truth.

        Retraining time T_retrain = C(credential_src → credential_tgt) +
        ceil(ω · d(occ_i, occ_j)) is computed separately and only governs the
        worker's retraining_ticks_left countdown — it no longer biases the
        target-occupation choice itself.  Workers are still penalised by the
        time they will spend out of the labor pool, but via the dropout hazard
        and OLF flag rather than through a friction-weighted choice probability.
        """
        model = self.model
        p     = self.params
        omega = p.get("omega", 0.5)  # Retained for retraining-time computation only.

        # Use precomputed arrays from model init — avoids per-call pandas .loc
        # and 537-item Python list comprehensions on every retraining event.
        if model._dist_array is None or self.current_occ not in model._cand_occ_to_row:
            return

        candidates = model._cand_occs                          # list[int], len=537
        row_idx    = model._cand_occ_to_row[self.current_occ]
        d_row      = model._dist_array[row_idx].astype(float)  # numpy row slice
        v          = model._cand_vacancy_arr.astype(float)     # rebuilt each tick

        # Origin vacancies V_i — the worker's home-occupation opportunity stock.
        col_i = model._cand_occ_to_col.get(self.current_occ, -1)
        V_i = float(v[col_i]) if col_i >= 0 else 1.0
        V_i = max(V_i, 1.0)  # numerical floor — prevents division-by-zero
        V_j = np.maximum(v, 1.0)

        # ── Intervening-opportunities S_ij ──────────────────────────────────
        # For each candidate j, S_ij = sum of vacancies V_k for all k with
        # d(i, k) < d(i, j).  Compute via sort + cumulative sum.
        sort_order = np.argsort(d_row, kind="stable")
        sorted_d   = d_row[sort_order]
        sorted_V   = V_j[sort_order]
        # Cumulative vacancy sum at each position in distance-sorted order.
        cum_V_sorted = np.cumsum(sorted_V)
        # Strict-less-than: for ties at distance d_row[j], all are credited
        # the same S_ij = cumulative-up-to-but-not-including-this-tie-block.
        # Use unique distances to compute the "before-this-tie" cumulative.
        # For practical performance we accept the strict-inequality form via:
        #   S_ij(sorted_pos = k) = cum_V_sorted[k] - sorted_V[k]
        S_ij_sorted = cum_V_sorted - sorted_V
        # Map back to original candidate ordering.
        S_ij = np.empty_like(S_ij_sorted)
        S_ij[sort_order] = S_ij_sorted

        # Radiation model probability (unnormalised; the constant origin-mass
        # m_i drops out under proportional sampling — only relative pull matters).
        # P_j ∝ V_i · V_j / ((V_i + S_ij) · (V_i + S_ij + V_j))
        denom_left  = V_i + S_ij
        denom_right = denom_left + V_j
        # Both denominators are strictly positive (V_i ≥ 1, V_j ≥ 1), so safe.
        scores = (V_i * V_j) / (denom_left * denom_right)

        # Credential gating: zero-out destinations the worker cannot reach via
        # any credential path (entry == 999 sentinel in the credential DAG).
        # All reachable paths remain valid candidates — the radiation model
        # imposes no soft penalty for credential-path length itself; instead the
        # length is paid as retraining_ticks_left and the dropout hazard.
        cred_months = CRED_DIST_MATRIX[
            self.credential_idx, model._cand_min_cred_idx_arr
        ].astype(float)
        scores = np.where(cred_months < 999, scores, 0.0)

        total = scores.sum()
        if total <= 0:
            return

        probs  = scores / total
        # cumsum + searchsorted on numpy arrays (avoids the Python-list
        # conversion that random.choices would require for 537 candidates).
        chosen = candidates[int(np.searchsorted(np.cumsum(probs), self.random.random())
                                .clip(0, len(candidates) - 1))]
        self.target_occ = chosen
        occ_min_cred_idx = model.occ_min_cred_idx  # for chosen-occ credential lookup

        if chosen != self.current_occ:
            col_idx      = model._cand_occ_to_col.get(chosen, -1)
            dist         = float(model._dist_array[row_idx, col_idx]) if col_idx >= 0 else 1.0
            tgt_cred_idx = int(occ_min_cred_idx.get(chosen, 0))
            cred_pen     = int(CRED_DIST_MATRIX[self.credential_idx, tgt_cred_idx])
            skill_pen    = max(1, math.ceil(omega * dist))
            self.retraining_ticks_left = max(1, cred_pen + skill_pen)
            self.retraining_total      = self.retraining_ticks_left
            self.retrain_d             = float(dist)  # for dropout-hazard distance term
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

        A per-tick stochastic dropout hazard fires before the countdown:

            P_dropout = sigmoid(α0 + α1*ticks_in_retraining
                                   + α2*months_unemployed
                                   + α3*d(s_i, s_j))

        Three pressures (Jacobson, LaLonde & Sullivan 2005; NCES adult-learner
        persistence data): duration fatigue, financial desperation as
        unemployment lengthens, and academic difficulty proxied by the semantic
        distance between origin and target occupations. On dropout the worker
        abandons the credential path, returns to the active labor pool with
        their accumulated unemployment duration intact, and re-enters market
        clearing with peaked desperation — forcing acceptance of downward
        wage transitions to escape unemployment.

        Intra-firm human capital accumulation: employed workers who upskill
        stay at their current firm — retraining does not force a resignation.
        Only unemployed workers increment months_unemployed during retraining.

        At completion:
          - Risk scores blend toward target occupation (r_job drops, p_aug rises).
          - search_occ is set so the worker can target the new role if displaced.
          - Employed workers remain on their employer's roster with updated skills.
          - is_retraining flag is cleared.
        """
        # ── Dropout hazard ──────────────────────────────────────────────────
        if self._check_retraining_dropout():
            return  # state has been reset; bail before the countdown

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
            self.has_retrained     = True
            self.is_retraining     = False
            # Reset path-length bookkeeping so a future retraining episode
            # gets a fresh (retraining_total, retrain_d) pair.
            self.retraining_total = 0
            self.retrain_d        = 0.0
            # Employed workers stay on their roster — no detachment

    # ── Experience & aging ───────────────────────────────────────────────────

    def _accumulate_experience(self):
        """Gain one tick (= one month) of experience.

        ``experience_years`` tracks the worker's raw, chronological labor-
        market tenure (uncapped — per Lemieux's quartic specification, late-
        career declines are captured by the higher-order coefficients rather
        than by truncating the input).  ``exp_norm`` continues to be a
        normalised [0,1] tenure proxy used by hazard, augmentation, and
        retraining heuristics — it caps at 1.0 (≈ 40 years).
        """
        self.experience_years += 1.0 / 12.0
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
