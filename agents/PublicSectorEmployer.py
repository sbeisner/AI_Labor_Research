"""Public Sector Employer Agent for the AI Labor Market ABM.

The public sector is represented as a single monolithic employer that:
  - Follows the same BTOS signal evolution as private firms but with a heavy
    friction multiplier (0.3), reflecting slower institutional technology
    adoption in government.
  - AI adoption maturity (A_{j,t}) is permanently locked to 0.  The C* formula
    therefore collapses to floor(C0 * (1 + g_jt)) — pure BTOS-modulated
    replacement with no automation-induced capacity shrinkage or new-economy
    vacancy creation.
  - Displacement uses only the BTOS-modulated baseline turnover rate δ_base,
    bypassing the logistic AI hazard entirely (civil service protections).
    Augmentation wage boosts still apply to surviving employees.
  - The Public Sector Agent is immortal: it never enters the Distressed or
    Failed states regardless of its C* value.
"""

import math

import numpy as np

from agents.Employer import EmployerAgent, _SECTOR_DRIFT, _DEFAULT_DRIFT


# Friction penalty: public sector adopts AI at 30 % of the private-sector rate.
PUBLIC_SECTOR_FRICTION = 0.3

# NAICS 2-digit sector code for Public Administration.
PUBLIC_SECTOR_NAICS = "92"


class PublicSectorEmployerAgent(EmployerAgent):
    """Single employer representing the entire US public sector.

    Inherits the 4-phase step from EmployerAgent and overrides four phases:

    Phase 1 — BTOS update:
        Both the sector drift and the monthly shock are scaled by
        PUBLIC_SECTOR_FRICTION (0.3) so the government's health trajectory
        evolves more slowly than any private-sector counterpart.

    Phase 2 — Layoff:
        Uses only the BTOS-modulated baseline turnover rate (prob = eff_base).
        The logistic AI hazard (β1, β2, β3 terms) is bypassed entirely.
        Augmentation wage boosts still apply to surviving employees.
        Layoffs do NOT increment _displacement_this_tick.

    Phase 3 — Vacancy generation:
        a_jt is permanently 0, so C* = floor(C0 * (1+g_jt)) — BTOS-modulated
        replacement only. No new-economy vacancies are generated.

    Phase 4 — Firm state:
        No-op. The public sector never enters Distressed or Failed states.
    """

    is_public_sector: bool = True

    def __init__(self, model, initial_btos: float = 0.0):
        super().__init__(
            model,
            sector=PUBLIC_SECTOR_NAICS,
            initial_btos=initial_btos,
            ind_key="public_sector",
            # Capacity 1 initially; real capacity comes from assigned workers.
            initial_capacity=1,
            # AI adoption locked to 0: a_jt property always returns 0.
            a_adoption=0.0,
        )

    # ── Override: a_jt always 0 for public sector ────────────────────────────

    @property
    def a_jt(self) -> float:
        """Public sector AI adoption maturity is permanently locked to 0."""
        return 0.0

    # ── Phase 1: BTOS signal (friction-dampened) ─────────────────────────────

    def _update_btos(self):
        p         = self.model.params
        shock_std = p.get("btos_shock_std", 0.02)
        theta     = p.get("theta_ou", 0.1) * PUBLIC_SECTOR_FRICTION  # dampened reversion

        # Anchor (μ_j): sector g_init from BTOS, fall back to legacy _SECTOR_DRIFT.
        # The Keynesian aggregate-demand shift is also applied here (dampened by
        # PUBLIC_SECTOR_FRICTION), since federal/state budgets are partially
        # buffered against private-sector demand swings but not fully insulated.
        btos_data = self.model._btos_sector.get(self.sector[:2], {})
        mu        = btos_data.get("g_init", _SECTOR_DRIFT.get(self.sector[:2], _DEFAULT_DRIFT))
        mu       += getattr(self.model, "_consumption_anchor_shift", 0.0) * PUBLIC_SECTOR_FRICTION

        shock = self.random.gauss(0.0, shock_std * PUBLIC_SECTOR_FRICTION)

        # OU step with institutional friction applied to both θ and σ
        reversion_pull = theta * (mu - self.btos_signal)
        self.btos_signal = float(np.clip(
            self.btos_signal + reversion_pull + shock,
            -0.15, 0.15,
        ))

    # ── Phase 2: Layoff (frictional turnover only — no structural AI cuts) ───

    def _layoff_phase(self):
        """Public-sector layoff phase: frictional turnover only.

        Public-sector workers are exempt from structural AI layoffs by statute /
        political-economy assumption (civil-service protections, budget cycle
        rigidity). C* is still computed via the shared helper so that Phase 3
        vacancy generation has the baseline data it needs. AI augmentation wage
        boost still applies to all surviving employed workers.
        """
        p = self.model.params
        eff_base = float(np.clip(
            p["delta_base"] * (1.0 - self.btos_signal), 1e-9, 1.0 - 1e-9
        ))
        self._fired_this_tick = 0

        for worker in self._roster:
            worker.just_fired = False

        # Frictional turnover (natural separations, not AI displacement).
        for worker in [w for w in self._roster if w.is_employed]:
            if self.random.random() < eff_base:
                worker.is_employed       = False
                worker.months_unemployed = 0
                worker.just_fired        = True
                self._fired_this_tick   += 1

        # Compute C* for Phase 3 — γ_naics for sector "92" (~0.10 in defaults)
        # produces a small augmentation-driven capacity expansion bounded by
        # public-sector budget rigidity.
        self._compute_cstar()

        # Wage boost: augmentation productivity still raises wages even though
        # public workers are not subject to structural displacement.
        if self.model.ai_active:
            wage_boost = p.get("wage_boost", 0.02)
            for worker in self._roster:
                if worker.is_employed:
                    worker.wage *= 1.0 + (wage_boost * worker.p_aug) / 12.0

    # ── Phase 4: Firm state (immortal — always Healthy) ──────────────────────

    def _update_firm_state(self):
        """Public sector never becomes Distressed or Failed."""
        self.state            = "Healthy"
        self.distress_counter = 0
