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
    "delta_base":    0.0128,  # baseline monthly turnover intercept; MSM point estimate
                              # (Nelder-Mead, 31 iters, K=4 inner seeds; validated at K=20).
                              # Replaces the legacy ABC posterior (0.01334), which was tuned
                              # to a single steady-state mean and is incompatible with the
                              # audit-fix Keynesian feedback loop's endogenous cycles.
                              # See output/msm_posterior.csv and scripts/msm_calibration.py.
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
    "adoption_velocity_mult": 1.0,  # global dose multiplier on k_adoption; 1.0 = headline
                              # trajectory. Swept in scripts/dose_response.py to test the
                              # conditionality of results on the adoption speed A_{j,t}.

    # ── Retraining ─────────────────────────────────────────────────────────────
    "eta_base":      0.02,    # employed proactive upskilling rate
    "eta_unemp":     0.05,    # unemployed reactive retraining rate
    "kappa":         0.06,    # automation-fear multiplier
    "xi":            0.03,    # unemployment-duration multiplier
    "omega":        18.0,     # Retrain time scalar — translates semantic distance d_ij ∈ [0,1]
                              # into retraining months. Calibrated jointly with μ via grid
                              # search (scripts/calibrate_omega_mu.py) against two empirical
                              # anchors: simulated median retraining duration matches BLS
                              # Trade Adjustment Assistance training data (~6 months for
                              # non-credential skill switches), and simulated mean cosine
                              # transition distance matches Macaluso (2017) "Skill Remoteness"
                              # estimates (~0.30). Best-cell L2 score = 0.066.
    "mu":            0.25,    # Gravity-model cognitive friction penalty — paired with ω=18.
                              # exp(-μ·T) = 0.78 at T=1, 0.22 at T=6, 0.05 at T=12, 0.002 at T=24.
                              # Calibrated jointly with ω; see comment above and
                              # scripts/calibrate_omega_mu.py.
    "retrain_blend": 0.7,     # weight toward target occ's risk profile
    # Credential gap: ticks per zone gap (index = gap size 0-4)
    "zone_ticks":    [0, 6, 12, 24, 36],

    # ── Mincer wage equation (quartic in raw chronological years X) ───────────
    # ln(W) = ln(W_base) + r·Z + β1·X + β2·X² + β3·X³ + β4·X⁴
    # Coefficients calibrated to produce a Lemieux-2006-shaped life-cycle
    # earnings profile: ~50-65% peak log-wage premium near X≈30 then slight
    # decline through retirement.  Replaces the prior fractional-quadratic in
    # exp_norm ∈ [0,1], which structurally capped the senior premium at 3.8%
    # — the catastrophic dimensionality bug flagged in the audit.
    "r_edu":               0.09,           # wage return per O*NET Job Zone level
    "mincer_beta1":        0.060,          # linear experience coefficient (per year)
    "mincer_beta2":       -0.0020,         # quadratic experience coefficient
    "mincer_beta3":        0.00003,        # cubic experience coefficient
    "mincer_beta4":       -0.00000020,     # quartic experience coefficient
    "experience_years_max": 40.0,          # mapping anchor for exp_norm → years at init
    # Firm-level coefficient heterogeneity (Card, Heining & Kline 2013):
    # std-dev of multiplicative noise applied to the four β coefficients at
    # firm init.  When > 0, employers pay non-trivially different experience
    # premia, organically incentivising senior retention without hard-coded
    # array sorting.  Defaults to 0 so the deterministic baseline is reproducible.
    "mincer_firm_std":     0.10,
    # Legacy aliases kept for any older calibration grids that haven't been
    # migrated to the quartic specification.  The runtime no longer reads them.
    "mincer_b1":     0.04,    # legacy fractional-experience coefficient (DEPRECATED)
    "mincer_b2":     0.002,   # legacy fractional-experience² coefficient (DEPRECATED)
    "pub_wage_damp": 0.6,     # public-sector premium dampener (× r_edu & mincer_beta1)
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
    # ── CES task production (Acemoglu-Restrepo 2018) ─────────────────────────
    # σ_elast is the elasticity of substitution between tasks in
    #   Y = (∫_{N-1}^{N} y(i)^((σ-1)/σ) di)^(σ/(σ-1))
    # Empirically calibrated to [0.8, 1.5] in cognitive-task economies
    # (Acemoglu-Restrepo 2018; Bessen 2019; default 1.2).  REPLACES the prior
    # σ-as-2%-rate misappropriation flagged by the audit — σ is no longer a
    # linear job-creation cap.
    "sigma_elast":             1.2,
    # Reinstatement efficiency η: institutional friction (R&D lag, training
    # capacity) capping the rate at which CES-implied new-task gradients
    # convert into actual posted vacancies.  ABC-tunable, not theory-fixed.
    "reinstatement_efficiency": 0.05,
    # Legacy alias.  If a calibration grid passes σ < 0.5 (old 2%-rate scale),
    # the runtime in Employer._generate_vacancies treats it as η and uses
    # sigma_elast_default for the actual elasticity term.
    "sigma":                   0.02,
    "sigma_elast_default":     1.2,

    # ── Managerial-information noise (peer-review robustness sweep) ───────────
    # Standard deviation of a Gaussian observation error applied to the firm's
    # read of each worker's r_job (substitution risk) and p_aug (augmentation
    # potential) at decision time. Defaults to 0.0 (perfect information, the
    # original assumption). Setting these > 0 means firms see a noisy estimate
    # at each layoff and matching decision (fresh draw per decision; underlying
    # worker attributes are not mutated). Sweep via scripts/info_noise_sensitivity.py.
    "p_aug_noise":    0.0,
    "r_job_noise":    0.0,

    # ── Probabilistic matching idiosyncratic noise ────────────────────────────
    # Audit fix (review §"Matching Gating and the Cascade Bump Artifact"):
    # replaces deterministic perfect-information rank-and-cut with stochastic
    # matching.  std-dev of multiplicative log-normal noise on each candidate's
    # match score at hiring time, capturing interview performance, cultural fit,
    # and administrative friction.  Set to 0 to recover the deterministic
    # baseline used during ABC calibration.
    "match_noise_std": 0.15,

    # ── Keynesian aggregate-demand feedback ───────────────────────────────────
    # Audit fix (review §"Closing the Macroeconomic Loop"): closes the gap
    # between the labor and goods markets.  Aggregate wage-bill loss feeds
    # back into the OU drift anchor μ_j so firms cannot keep expanding under
    # a static historical baseline while mass layoffs cascade.
    #
    # The feedback is intentionally aggressive (strength=0.30, half-life=6).
    # In a properly specified macro-financial model this WILL produce
    # endogenous business-cycle non-stationarity in the Control scenario as
    # well — the system is no longer linearised around a fixed-point steady
    # state, and that is the point.  Calibration therefore uses Method of
    # Simulated Moments (scripts/msm_calibration.py) targeting the empirical
    # cyclical moments (variance, autocorrelation, Beveridge correlation),
    # not a single steady-state mean as ABC-rejection did.
    "keynesian_feedback":  True,    # master switch
    "mpc":                 0.7,     # marginal propensity to consume from labor income
                                    # (Carroll & Slacalek 2017; Berger et al. 2018)
    "feedback_strength":   0.30,    # OU-anchor pass-through coefficient on consumption gap
    "feedback_half_life":  6.0,     # ticks for the anchor to converge to a new target

    # ── CES reinstatement labor-abundance response anchor ─────────────────────
    # Empirical baseline UR used to compute the labor-abundance multiplier
    # ψ(σ, U_t/U_baseline) in Employer._generate_vacancies.  Aligned with the
    # MSM moment-fit target (mean UR of the BLS 2015 to 2019 monthly series).
    "ur_baseline":         0.045,

    # ── Frontier basket (RQ6: cleared-market new-economy absorption) ──────────
    # OCC2010 codes for the empirically grounded "new work" basket: existing,
    # high-end occupations into which AI-driven new-economy demand actually
    # concentrates per Autor, Salomons & Seegmiller (2021) "New Frontiers" and
    # Babina et al. (2024) on firm-level AI investment hiring patterns:
    #   1006 - Computer & Information Research Scientists (Data Scientist proxy)
    #   1010 - Computer Systems Analysts
    #   1020 - Software Developers
    #   1240 - Operations Research Analysts
    # When a firm generates v_new vacancies, they are posted into this pool
    # (equal split with deterministic rounding) so unemployed agents can
    # discover, retrain into, and fill them via the existing gravity model.
    # This converts the new-economy metric from a hardcoded counter (echoing σ)
    # into an emergent fill-rate measurement bounded by skill distance, the
    # 24-month retraining penalty, and dropout hazard.
    "frontier_basket": (1006, 1010, 1020, 1240),

    # ── Effective-vacancy aggregation (Audit-2 fix: "Incumbent Mass Fallacy") ──
    # When True, _update_effective_vacancies starts the eff dict from
    # vacancy_counts (per-occupation employed-incumbent counts) and ADDS open
    # vacancies on top — the legacy behavior, which biases the radiation
    # model's V_j toward large legacy occupations regardless of actual demand.
    # When False (default, post-fix), eff is built purely from open vacancies,
    # with vacancy_counts used only as a tick-0 fallback before any vacancies
    # have been generated. Used by scripts/eff_vac_sensitivity.py to A/B the
    # RQ6 reinstatement-shortfall conclusion against the bug.
    "eff_vac_legacy_sum": False,

    # ── Labor force OLG entry ──────────────────────────────────────────────────
    "v_entry_rate":  0.0025,  # monthly workforce entry rate (fraction of pool)

    # ── Matching function ──────────────────────────────────────────────────────
    # ρ is derived analytically at model init: ρ = -ln(1-f_target)/θ_base
    "f_target":       0.28,   # target monthly job-finding rate (Shimer 2005)
    "theta_base":     0.5,    # baseline tightness for ρ calibration
    "vacancy_rate":   0.0444,  # open positions as fraction of employment (MSM point estimate;
                               # see output/msm_posterior.csv).
    "nu":             2.0,    # legacy experience premium (kept for compat)

    # ── Employer vacancy generation ────────────────────────────────────────────
    "gamma":          0.3,    # fallback γ for any NAICS-2 not in gamma_by_naics
    # Industry-specific demand-elasticity γ in C* = round(C0*(1+g)*(1 - A*R + γ*A*P))
    # High γ (≈0.5): elastic, scalable cognitive output — Information, Finance,
    #   Professional Services. Augmentation expands aggregate capacity (Bessen 2019;
    #   Autor & Salomons 2018: reinstatement effect dominates).
    # Moderate γ (≈0.2): bounded by physical/temporal constraints — Healthcare,
    #   Education, Manufacturing. Demand is latent but capacity grows slowly.
    # Inelastic γ (≈-0.05 to 0.0): internal-overhead or saturated-demand sectors —
    #   Admin Support, Retail, Food. Augmentation contracts headcount; the firm
    #   keeps a smaller team to manage the AI rather than scaling output.
    # ── Industry-specific labor-demand elasticity γ ───────────────────────────
    # Audit fix (review §"The Illusion of Emergence"): the prior 0.0–0.50 hard
    # cap structurally suppressed augmentation-driven net expansion in highly
    # elastic sectors.  Recent empirical work (Wang et al. 2026 ArXiv 2604.01066;
    # Bessen 2019) places γ > 1.0 for Information, Finance, and Professional
    # Services where LLM augmentation drives disproportionate demand expansion.
    # The calibration here lifts the cap; the C* formula
    #   C* = round(C0 · (1+g) · (1 - A·R + γ·A·P))
    # is monotonic in γ but bounded by the round() floor and post-attrition
    # roster size, so super-unitary γ does not produce runaway hiring.
    "gamma_by_naics": {
        # Highly elastic — reinstatement and demand expansion dominate
        "51": 1.20,   # Information / Tech (LLM-augmented productivity)
        "52": 1.10,   # Finance & Insurance
        "54": 0.90,   # Professional, Scientific & Technical
        "55": 0.70,   # Management of Companies
        # Moderate elasticity — bounded latent demand
        "62": 0.30,   # Health Care
        "61": 0.30,   # Education
        "31": 0.25, "32": 0.25, "33": 0.25,  # Manufacturing
        "21": 0.15,   # Mining / Oil & Gas
        "22": 0.15,   # Utilities
        "23": 0.15,   # Construction
        "11": 0.15,   # Agriculture
        "53": 0.25,   # Real Estate
        "71": 0.20,   # Arts & Entertainment
        "92": 0.10,   # Public Administration (budget-constrained)
        # Inelastic — augmentation produces no aggregate demand expansion, but
        # γ=0 (not negative) avoids double-counting capacity destruction with
        # the (1 - A*R_job) substitution term that already shrinks headcount.
        # Inelastic demand means productivity gains DO NOT GENERATE NEW DEMAND,
        # not that they actively destroy demand.
        "56": 0.00,   # Administrative & Support Services
        "44": 0.00, "45": 0.00,  # Retail Trade
        "72": 0.00,   # Accommodation & Food
        "48": 0.05, "49": 0.05,  # Transportation & Warehousing
        "42": 0.05,   # Wholesale Trade
        "81": 0.05,   # Other Services
    },
    "epsilon":        0.5,    # legacy direct-replacement fraction (kept for compat)
    "btos_shock_std":  0.02,    # BTOS monthly shock std dev (idiosyncratic, firm-level)
    "btos_macro_std":  0.0133,  # common macro shock std dev (MSM point estimate; same draw for all
                                # firms → Beveridge cyclicality)
    "theta_ou":        0.1,    # OU mean-reversion speed (half-life ≈ 7 months)
    "btos_disp_damp":  0.5,    # BTOS pass-through dampener: eff_base = delta*(1 - damp*btos); 0→no BTOS effect, 1→full pass-through
    "max_layoff_rate": 0.05,   # Organisational-friction cap: structural layoffs in Step C
                               # cannot exceed ceil(max_layoff_rate * post_attrition_roster)
                               # per tick. Real firms can't legally or operationally fire 50%
                               # of a department in one month (severance, WARN Act notice,
                               # operational continuity). The actual employment line E lags
                               # the target capacity C* on the descent, smoothing the AI shock.

    # ── Retraining dropout hazard ──────────────────────────────────────────────
    # P_dropout = sigmoid(α0 + α1*ticks_in_retraining + α2*months_unemployed + α3*d_ij)
    # Calibrated for ~50% completion of a 24-month average retraining path
    # (Jacobson, LaLonde & Sullivan 2005; NCES adult learner persistence data).
    # α0=-3.5 ⇒ baseline ~3% per-tick hazard at the start; duration fatigue and
    # financial desperation accelerate dropout; large semantic distance d_ij
    # raises the academic-failure component.
    "dropout_alpha0":      -3.5,   # baseline logit (≈ 2.9% hazard at t=0, m=0, d=0)
    "dropout_alpha_dur":    0.05,  # per-tick duration-fatigue coefficient
    "dropout_alpha_unemp":  0.02,  # per-month unemployment financial-desperation coef
    "dropout_alpha_dist":   1.5,   # semantic-distance d_ij ∈ [0,1] academic-difficulty coef

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
    #   (6) Dynamic desperation threshold: credential soft-penalty ramps 0.3→1.0 linearly
    #       over 6 ticks of vacancy age, resolving phantom vacancies from OLG retirements
    # Run 10 ABC: 1069/2000 particles accepted (53.5%, ε=0.005), simulated UR=4.544%
    # delta_base posterior: mean=0.01222 (std=0.00351)
    # vacancy_rate posterior: mean=0.04548 (std=0.020, partially identified by UR target alone)
}


# ── Credential system (see model/credentials.py) ──────────────────────────────
# All credential constants and helpers live in model/credentials.py to avoid
# circular imports (LaborMarketModel imports Worker/Employer, which need these).
# Re-export here so existing call sites (occ_min_credential build, etc.) work.
from model.credentials import (                          # noqa: E402
    CREDENTIAL_LEVELS, CREDENTIAL_IDX, CREDENTIAL_GRAPH,
    ZONE_MIN_CREDENTIAL, educ_to_credential, credential_months_to,
)

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
        # Integer-index version for O(1) vectorized use in _choose_target_skill()
        # and market clearing.  Avoids repeated CREDENTIAL_IDX dict lookups.
        self.occ_min_cred_idx = {
            occ: CREDENTIAL_IDX.get(cred, 0)
            for occ, cred in self.occ_min_credential.items()
        }

        # ── Precomputed gravity-model arrays (aligned to dist_matrix columns) ──
        # These are computed once at init and shared across all workers.
        # Eliminates per-retraining-call pandas .loc lookups and 537-item Python
        # list comprehensions that were the main per-tick overhead bottleneck.
        if self.skill_distance_matrix is not None:
            self._cand_occs: list = self.skill_distance_matrix.columns.tolist()
            # OCC2010 → position index in _cand_occs
            self._cand_occ_to_col: dict = {occ: i for i, occ in enumerate(self._cand_occs)}
            # Row index for dist_matrix (index may differ from columns order)
            self._cand_occ_to_row: dict = {
                occ: i for i, occ in enumerate(self.skill_distance_matrix.index.tolist())
            }
            # Raw numpy distance array — avoids pandas label-lookup overhead
            self._dist_array: np.ndarray = self.skill_distance_matrix.values.astype(np.float32)
            # Static per-occupation arrays (don't change between ticks)
            _r_job = self.occ_risk_lookup.get("r_job", {})
            self._cand_r_arr: np.ndarray = np.array(
                [_r_job.get(c, 0.5) for c in self._cand_occs], dtype=np.float32
            )
            from model.credentials import CRED_DIST_MATRIX  # noqa: PLC0415
            self._cand_min_cred_idx_arr: np.ndarray = np.array(
                [self.occ_min_cred_idx.get(c, 0) for c in self._cand_occs], dtype=np.int32
            )
            self._cand_vacancy_arr: np.ndarray = np.ones(len(self._cand_occs), dtype=np.float32)
        else:
            self._cand_occs = []
            self._cand_occ_to_col = {}
            self._cand_occ_to_row = {}
            self._dist_array = None
            self._cand_r_arr = np.array([], dtype=np.float32)
            self._cand_min_cred_idx_arr = np.array([], dtype=np.int32)
            self._cand_vacancy_arr = np.array([], dtype=np.float32)

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
        self._new_economy_jobs_cumulative = 0
        self._open_market_hired_this_tick = 0
        self._spinoffs_this_tick          = 0
        # ── Q1 (HSQ1_Low) loss-decomposition counters (peer-review robustness) ─
        # Decomposes lowest-hard-skill-quintile unemployment into three sources:
        #   1. Structural displacement: HSQ1 worker is structurally separated.
        #   2. Credential-blocked:      HSQ1 candidate is in valid_candidates
        #                               for a vacancy but fails the credential
        #                               floor and is not hired.
        #   3. Cascade-bumped:          HSQ1 candidate passes the credential
        #                               floor but is out-ranked by a higher-HSQ
        #                               worker (typically retraining downward)
        #                               and is not hired.
        self._q1_displaced_this_tick           = 0
        self._q1_credential_blocked_this_tick  = 0
        self._q1_cascade_bumped_this_tick      = 0
        self._retirements_this_tick       = 0
        self._entries_this_tick           = 0
        self._dropouts_this_tick          = 0
        self._macro_shock_this_tick       = 0.0
        # Persistent macro shock level for AR(1) business-cycle driver.
        # AR(1): level_t = phi * level_{t-1} + N(0, macro_std)
        # phi ~ 0.85 gives half-life of ~4 ticks — realistic business cycle duration.
        # Without AR(1), i.i.d. macro shocks die within one OU mean-reversion step
        # and produce near-zero Beveridge curve correlation.
        self._macro_shock_level           = 0.0

        # ── Keynesian feedback state ───────────────────────────────────────────
        # The aggregate-demand loop (review §"Closing the Macroeconomic Loop")
        # closes the gap between labor income and goods-market demand.  Each
        # tick we compute the aggregate wage bill, apply an MPC multiplier to
        # derive consumption, and feed the consumption shortfall back into the
        # employer OU drift anchor as an aggregate-demand shock.  Without this,
        # firms revert to historical pre-shock g_init even after mass layoffs —
        # producing the "Ghost GDP" externality the audit flagged.
        self._baseline_wage_bill_initialized = False
        self._baseline_wage_bill   = 0.0
        self._consumption_anchor_shift = 0.0  # current OU-anchor shift (∈ ~±0.05)

        # ── Lagged aggregate unemployment rate ─────────────────────────────────
        # Read by Employer._generate_vacancies to scale the CES reinstatement
        # intensity by labor-market tightness.  Initialized to the empirical
        # 2015-2019 BLS baseline so the abundance response equals 1.0 at tick
        # zero before any displacement has occurred.  Updated at the end of
        # each step.
        self._ur_lagged = float((params or DEFAULT_PARAMS).get("ur_baseline", 0.045))
        # Per-occupation matching inputs (θ for Poisson matching)
        self._tightness = {}
        # Rolling vacancy history for vacancy-proportional OLG entrant assignment
        self._vacancy_history: list = []  # list of per-tick {occ: count} dicts

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
                "Unemployed_Count":     lambda m: _worker_sum(m, lambda a: not a.is_employed and not a.is_retired and not a.is_olf),
                "OLF_Count":           lambda m: _worker_sum(m, lambda a: a.is_olf and not a.is_retired),
                "Mean_Wage":            lambda m: _mean_wage(m),
                "Emp_Rate_Q1_Low":      lambda m: _emp_rate_q(m, "Q1_Low"),
                "Emp_Rate_Q2":          lambda m: _emp_rate_q(m, "Q2"),
                "Emp_Rate_Q3":          lambda m: _emp_rate_q(m, "Q3"),
                "Emp_Rate_Q4":          lambda m: _emp_rate_q(m, "Q4"),
                "Emp_Rate_Q5_High":     lambda m: _emp_rate_q(m, "Q5_High"),
                "Emp_Rate_HSQ1_Low":    lambda m: _emp_rate_hsq(m, "HSQ1_Low"),
                "Emp_Rate_HSQ2":        lambda m: _emp_rate_hsq(m, "HSQ2"),
                "Emp_Rate_HSQ3":        lambda m: _emp_rate_hsq(m, "HSQ3"),
                "Emp_Rate_HSQ4":        lambda m: _emp_rate_hsq(m, "HSQ4"),
                "Emp_Rate_HSQ5_High":   lambda m: _emp_rate_hsq(m, "HSQ5_High"),
                "Emp_Rate_Entry":       lambda m: _emp_rate_exp(m, 0.0, 0.2),
                "Emp_Rate_Senior":      lambda m: _emp_rate_exp(m, 0.8, 1.0),
                "Retraining_Count":     lambda m: _worker_sum(
                    m, lambda a: a.retraining_ticks_left > 0 and not a.is_retired),
                "Retrained_Share":      lambda m: _retrained_share(m),
                "Dropouts_This_Tick":   lambda m: m._dropouts_this_tick,
                "Dropouts_Cumulative":  lambda m: _worker_sum(
                    m, lambda a: a.has_dropped_out and not a.is_retired),
                "New_Economy_Jobs":     lambda m: m._new_economy_jobs_this_tick,
                "New_Economy_Cumulative": lambda m: m._new_economy_jobs_cumulative,
                "Frontier_Basket_Employed": lambda m: _worker_sum(
                    m, lambda a: (a.is_employed and not a.is_retired
                                  and a.current_occ in m.params.get(
                                      "frontier_basket", (1006, 1010, 1020, 1240)))),
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
                "Q1_Displaced":        lambda m: m._q1_displaced_this_tick,
                "Q1_Credential_Blocked": lambda m: m._q1_credential_blocked_this_tick,
                "Q1_Cascade_Bumped":   lambda m: m._q1_cascade_bumped_this_tick,
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
        """Aggregate per-occupation OPEN vacancies from all employers for the
        radiation model in Worker._choose_target_skill().

        Uses employer._vacancies_by_occ (populated each tick by
        _generate_vacancies).  Falls back to vacancy_counts (incumbent
        occupation-size proxy) only when no employer has yet posted any
        vacancy — i.e., at tick 0 before _generate_vacancies has run — so
        the radiation kernel still has a defined opportunity field.

        Audit-2 fix: the prior implementation initialised eff from
        vacancy_counts and then ADDED per-employer vacancies on top, so the
        radiation V_j was (incumbents + open vacancies).  That biased
        unemployed-worker pull toward large legacy occupations regardless of
        actual labor demand and structurally suppressed reinstatement into
        Frontier-Basket new-economy postings.  When eff_vac_legacy_sum=True
        the buggy behavior is preserved for paired A/B comparison against
        the fix (see scripts/eff_vac_sensitivity.py).
        """
        legacy_sum = bool(self.params.get("eff_vac_legacy_sum", False))

        if legacy_sum:
            eff: dict = dict(self.vacancy_counts)
            for emp in self._employers.values():
                for occ, v in getattr(emp, "_vacancies_by_occ", {}).items():
                    eff[occ] = eff.get(occ, 0) + v
        else:
            eff = {}
            for emp in self._employers.values():
                for occ, v in getattr(emp, "_vacancies_by_occ", {}).items():
                    if v > 0:
                        eff[occ] = eff.get(occ, 0) + v
            if not eff:
                eff = dict(self.vacancy_counts)

        self.effective_vacancy_counts = eff
        # Rebuild per-candidate vacancy array aligned to _cand_occs so
        # Worker._choose_target_skill() can skip the 537-item dict comprehension.
        if self._cand_occs:
            self._cand_vacancy_arr = np.maximum(
                1.0,
                np.array([eff.get(c, 1) for c in self._cand_occs], dtype=np.float32)
            )

    def _update_vacancy_history(self):
        """Append a snapshot of current employer-posted vacancies to the rolling
        12-tick window used by _process_workforce_entry() for vacancy-proportional
        OLG entrant occupation assignment."""
        snapshot: dict = {}
        for emp in self._employers.values():
            if emp.state != "Failed":
                for occ, v in getattr(emp, "_vacancies_by_occ", {}).items():
                    if v > 0:
                        snapshot[occ] = snapshot.get(occ, 0) + v
        self._vacancy_history.append(snapshot)
        if len(self._vacancy_history) > 12:
            self._vacancy_history.pop(0)

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

    # ── Keynesian feedback loop ────────────────────────────────────────────────

    def _update_keynesian_feedback(self):
        """Close the aggregate-demand loop between labor income and firm health.

        At each tick:
          1. Compute the aggregate monthly wage bill across all employed workers.
          2. Apply the marginal propensity to consume (MPC) to derive aggregate
             consumption demand — workers vulnerable to displacement have an
             empirically high MPC out of labor income (Carroll & Slacalek 2017;
             Berger et al. 2018).
          3. Compare current consumption to the baseline (tick-0) consumption
             to obtain a relative consumption gap.
          4. Smoothly adjust the OU-drift anchor shift toward feedback_strength
             × consumption_gap, with a damped pass-through (half-life ≈ 6 ticks)
             so the feedback is persistent rather than instantaneous.

        The resulting ``_consumption_anchor_shift`` is read by every employer's
        OU step (Employer._update_btos / PublicSectorEmployer._update_btos),
        shifting the long-term mean μ_j by a common macro term.  When AI
        aggressively automates labor without sufficient reinstatement, wages
        fall, consumption drops, μ_j shifts negative, and firms further
        contract their target capacity — producing the displacement spiral
        that a static-anchor model cannot capture.
        """
        if not self.params.get("keynesian_feedback", True):
            return

        # Aggregate monthly wage bill over employed (non-retired) workers.
        wage_bill = 0.0
        for w in self.agents_by_type[WorkerAgent]:
            if w.is_employed and not w.is_retired:
                # self.wage stores annual $K; divide by 12 to convert to monthly.
                wage_bill += w.wage / 12.0

        # Tick-0 baseline (locked once, used as the denominator for the relative
        # consumption gap).  We delay the lock until after the first step's
        # employer clearing has reached its initial steady state — guarantees a
        # non-zero baseline even if some workers initialize with wage=0.
        if not self._baseline_wage_bill_initialized and wage_bill > 0:
            self._baseline_wage_bill = wage_bill
            self._baseline_wage_bill_initialized = True
            return

        if not self._baseline_wage_bill_initialized or self._baseline_wage_bill <= 0:
            return

        # Relative consumption gap: (C_t - C_0) / C_0.  MPC is multiplicatively
        # constant and cancels out of the relative form, but keeping it explicit
        # documents the literature link.
        mpc = float(self.params.get("mpc", 0.7))  # noqa: F841 (documented constant)
        cons_gap = (wage_bill - self._baseline_wage_bill) / self._baseline_wage_bill

        # Damped pass-through to OU anchor: new shift ← old + (target - old)/half_life.
        feedback_strength = float(self.params.get("feedback_strength", 0.30))
        half_life         = float(self.params.get("feedback_half_life", 6.0))
        target_shift = feedback_strength * cons_gap
        # Convergence: a fraction (1/half_life) of the gap closes each tick.
        self._consumption_anchor_shift += (target_shift - self._consumption_anchor_shift) / max(1.0, half_life)
        # Clip so a runaway feedback can't dominate the bounded BTOS state space.
        self._consumption_anchor_shift = float(np.clip(
            self._consumption_anchor_shift, -0.10, 0.10
        ))

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
        """Replace each retirement with a new entrant, reset to youth demographics.

        Occupation assignment uses a 60/40 blend:
          • 60% vacancy-proportional: occupations with more open vacancies (rolling
            12-tick mean) receive proportionally more new entrants, channeling fresh
            workers toward actual demand and producing the Beveridge curve's negative
            UR–VR correlation.
          • 40% CPS-organic: uniform draw from the macro worker pool preserves the
            realistic demographic and skill distribution from the CPS data.

        After assignment, a worker's credential is bumped to the occupation zone
        minimum if their EDUC-derived credential falls short, ensuring the entrant
        is immediately eligible for market clearing.
        """
        if self._macro_worker_pool.empty or self._retirements_this_tick <= 0:
            self._entries_this_tick = 0
            return

        n_new  = self._retirements_this_tick
        pool   = self._macro_worker_pool
        n_pool = len(pool)

        # ── Build blended sampling weights ────────────────────────────────────
        # Aggregate the rolling vacancy window into per-occupation totals.
        if self._vacancy_history:
            agg_vac: dict = {}
            for snap in self._vacancy_history:
                for occ, v in snap.items():
                    agg_vac[occ] = agg_vac.get(occ, 0) + v
            pool_occs = pool["OCC2010"].values
            vac_w = np.array(
                [float(agg_vac.get(int(o), 0)) for o in pool_occs], dtype=np.float64
            )
            vac_total = vac_w.sum()
            if vac_total > 0:
                # Normalize so the vacancy component sums to n_pool (same scale as uniform)
                vac_w = vac_w / vac_total * n_pool
                blended_w = 0.6 * vac_w + 0.4 * np.ones(n_pool, dtype=np.float64)
            else:
                blended_w = np.ones(n_pool, dtype=np.float64)
        else:
            blended_w = np.ones(n_pool, dtype=np.float64)

        blended_w = blended_w / blended_w.sum()

        sample = pool.sample(
            n=n_new,
            replace=True,
            weights=blended_w,
            random_state=self.random.randint(0, 2**31 - 1),
        )
        for _, row in sample.iterrows():
            w = WorkerAgent(self, row, self.params)
            w.age                   = 18
            w.exp_norm              = 0.0
            w.experience_years      = 0.0   # raw chronological years (Mincer X_i)
            w.is_employed           = False
            w.months_unemployed     = 0
            w.just_fired            = False
            w.retraining_ticks_left = 0
            # Bump credential to the occupation zone minimum if the entrant is
            # under-credentialed — prevents immediate credential-barrier rejection
            # in market clearing and allows the vacancy-fill cycle to close.
            occ_min = self.occ_min_credential.get(int(row["OCC2010"]), "high_school")
            if CREDENTIAL_IDX.get(w.credential, 0) < CREDENTIAL_IDX.get(occ_min, 0):
                w.credential     = occ_min
                w.credential_idx = CREDENTIAL_IDX[occ_min]
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
        self._update_vacancy_history()
        self._update_job_market()

        self._displacement_this_tick      = 0
        self._new_economy_jobs_this_tick  = 0
        self._open_market_hired_this_tick = 0
        self._spinoffs_this_tick          = 0
        self._dropouts_this_tick          = 0
        self._q1_displaced_this_tick           = 0
        self._q1_credential_blocked_this_tick  = 0
        self._q1_cascade_bumped_this_tick      = 0

        # AR(1) persistent macro shock — all employers add this same value to their
        # individual OU step, creating aggregate BTOS cyclicality for the Beveridge curve.
        # AR(1): level_t = phi * level_{t-1} + ε_t, ε_t ~ N(0, macro_std)
        # phi=0.85 ≈ 4-tick half-life, mimicking multi-month business cycle persistence.
        # Without persistence, i.i.d. shocks die within one OU reversion step and the
        # Beveridge correlation collapses to ~0.
        macro_std = self.params.get("btos_macro_std", 0.015)
        macro_phi = self.params.get("btos_macro_ar1", 0.85)
        self._macro_shock_level = (
            macro_phi * self._macro_shock_level
            + self.random.gauss(0.0, macro_std)
        )
        self._macro_shock_this_tick = self._macro_shock_level

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

        # Aggregate-demand feedback: compute consumption gap from this tick's
        # post-clearing wage bill so next tick's employer OU step incorporates
        # the demand shortfall (or expansion).
        self._update_keynesian_feedback()

        # Update lagged aggregate unemployment rate so the next tick's
        # Employer._generate_vacancies can scale CES reinstatement by labor
        # abundance.  Lagging by one tick avoids same-tick simultaneity (firms
        # cannot see vacancies they are about to post).
        _emp = 0
        _lf  = 0
        for w in self.agents_by_type[WorkerAgent]:
            if w.is_retired or w.is_olf:
                continue
            _lf += 1
            if w.is_employed:
                _emp += 1
        if _lf > 0:
            self._ur_lagged = 1.0 - _emp / _lf

        self.tick += 1


# ── Reporter helpers (module-level for pickling) ───────────────────────────────

def _workers(m):
    """All non-retired workers (employed + unemployed + OLF)."""
    return [a for a in m.agents_by_type[WorkerAgent] if not a.is_retired]


def _labor_force(m):
    """Active labor force: employed + unemployed, excluding OLF students.

    Mirrors BLS methodology: full-time students pursuing a credential upgrade
    are Out of Labor Force (OLF) and appear in neither the numerator nor the
    denominator of the unemployment rate.
    """
    return [a for a in _workers(m) if not a.is_olf]


def _worker_sum(m, fn):
    return sum(fn(a) for a in _workers(m))


def _retrained_share(m):
    ws = _workers(m)
    return sum(a.has_retrained for a in ws) / len(ws) if ws else 0.0


def _emp_rate(m):
    """Employment rate over the active labor force (excludes OLF students)."""
    lf = _labor_force(m)
    return sum(a.is_employed for a in lf) / len(lf) if lf else 0.0


def _mean_wage(m):
    wages = [a.wage for a in _workers(m) if a.is_employed and a.wage > 0]
    return float(np.mean(wages)) if wages else 0.0


def _emp_rate_q(m, quintile):
    grp = [a for a in _workers(m) if a.exposure_quintile == quintile and not a.is_olf]
    return sum(a.is_employed for a in grp) / len(grp) if grp else float("nan")


def _emp_rate_hsq(m, hsq):
    grp = [a for a in _workers(m) if a.hard_skill_quintile == hsq and not a.is_olf]
    return sum(a.is_employed for a in grp) / len(grp) if grp else float("nan")


def _emp_rate_exp(m, lo, hi):
    grp = [a for a in _workers(m) if lo <= a.exp_norm <= hi and not a.is_olf]
    return sum(a.is_employed for a in grp) / len(grp) if grp else float("nan")
