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

        # Anchor (μ_j): sector g_init from BTOS, fall back to legacy _SECTOR_DRIFT
        btos_data = self.model._btos_sector.get(self.sector[:2], {})
        mu        = btos_data.get("g_init", _SECTOR_DRIFT.get(self.sector[:2], _DEFAULT_DRIFT))

        shock = self.random.gauss(0.0, shock_std * PUBLIC_SECTOR_FRICTION)

        # OU step with institutional friction applied to both θ and σ
        reversion_pull = theta * (mu - self.btos_signal)
        self.btos_signal = float(np.clip(
            self.btos_signal + reversion_pull + shock,
            -0.15, 0.15,
        ))

    # ── Phase 2: Layoff (baseline turnover only — AI hazard bypassed) ────────

    def _layoff_phase(self):
        p = self.model.params
        # BTOS-modulated baseline turnover — no AI terms.
        eff_base = float(np.clip(
            p["delta_base"] * (1.0 - self.btos_signal), 1e-9, 1.0 - 1e-9
        ))
        self._fired_this_tick = 0

        for worker in self._roster:
            worker.just_fired = False

        for worker in [w for w in self._roster if w.is_employed]:
            if self.random.random() < eff_base:
                worker.is_employed       = False
                worker.months_unemployed = 0
                worker.just_fired        = True
                self._fired_this_tick   += 1
                # Public-sector layoffs are natural turnover, NOT AI displacement.
            elif self.model.ai_active:
                # Augmentation wage boost still applies even though AI doesn't
                # displace public workers — productivity gains are real.
                monthly_boost = (p.get("wage_boost", 0.02) * worker.p_aug) / 12.0
                worker.wage  *= 1.0 + monthly_boost

    # ── Phase 4: Firm state (immortal — always Healthy) ──────────────────────

    def _update_firm_state(self):
        """Public sector never becomes Distressed or Failed."""
        self.state            = "Healthy"
        self.distress_counter = 0
