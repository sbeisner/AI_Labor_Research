# Methodology Changes: AI Labor Market ABM

## Overview

This document describes six methodological revisions made to the Agent-Based Model (ABM) of AI-driven labor market disruption. Each change addresses a specific limitation in the prior implementation; the rationale, mechanism, and impact on results are described for each.

---

## 1. Logistic Displacement Probability Formula

**Prior method:** Worker displacement probability was computed as a linear function of AI occupation risk scores, which produced saturation artifacts — once risk scores exceeded a threshold, virtually all workers in a sector were displaced within a few ticks, collapsing confidence intervals to near-zero width.

**Revised method:** Displacement probability is computed via a logistic (sigmoid) transformation:

```
Z = logit(δ_base) + β_run · r_i^sub − λ · p_i^aug
P_disp = sigmoid(Z) = 1 / (1 + e^{−Z})
```

Where:
- `δ_base` = baseline monthly displacement rate (calibrated to BLS JOLTS separation rates, ≈ 0.005)
- `β_run` = macroeconomic AI impact multiplier (see §2)
- `r_i^sub` = individualized substitution risk (see §5)
- `λ` = augmentation suppression parameter
- `p_i^aug` = individualized augmentation potential

The logistic transformation ensures displacement probabilities remain bounded in (0, 1) regardless of extreme risk scores, preventing saturation and preserving meaningful cross-seed variance.

**Impact:** 95% CI width for the employment gap increased from 0.65 pp (effectively collapsed) to 10.6 pp, indicating genuine uncertainty across macroeconomic scenarios.

---

## 2. Macroeconomic Uncertainty Multiplier (β_run)

**Prior method:** The AI impact multiplier β was a fixed scalar applied identically to all bootstrap seeds, producing artificially tight confidence intervals that did not reflect real-world uncertainty about the pace of AI adoption.

**Revised method:** At the start of each bootstrap seed, a seed-specific multiplier is drawn:

```
β_run ~ N(β, σ_β)    where σ_β = 0.20
```

This value is held constant for all 60 ticks within that seed, representing a single "macroeconomic scenario" (e.g., rapid vs. gradual AI adoption). The multiplier scales the contribution of occupation-level AI substitution risk to displacement probability.

**Rationale:** The distribution N(1.0, 0.2) reflects genuine uncertainty in the speed of AI capability improvement and adoption rates across the 2024–2029 horizon. Slower adoption (β_run < 1) corresponds to regulatory constraints or delayed enterprise integration; faster adoption (β_run > 1) corresponds to accelerated diffusion following major capability releases.

**Impact:** Bootstrap confidence intervals now correctly reflect cross-scenario uncertainty (CI width ≈ 10.6 pp for the employment gap) while the mean estimate remains anchored to the calibrated β.

---

## 3. O\*NET Work Activities Cosine Distance (replacing Wu-Palmer)

**Prior method:** Skill distance between occupations was computed using Wu-Palmer semantic similarity on O\*NET occupation titles (via an occupational taxonomy tree). This produced a degenerate distribution: 94% of occupation pairs received a distance of exactly 1.0 (maximum), because Wu-Palmer is sensitive only to shared taxonomy nodes and most occupation pairs belong to different major SOC groups.

**Revised method:** Skill distance is computed from O\*NET Work Activities survey data:

1. Load the 41 Work Activities for all SOC codes (Importance scale).
2. Z-score each activity column across occupations: `(x − μ) / σ`. Z-scoring is critical — raw activity scores are highly correlated (all occupations require basic activities like communicating), compressing cosine distances toward zero.
3. Build an (N_occ × 41) matrix and L2-normalize each row.
4. Compute the cosine distance matrix: `D = 1 − X_norm @ X_norm.T`, clipped to [0, 1].
5. Hierarchical fallback for unmatched SOC codes: exact → 7-char prefix → 5-char prefix → 4-char prefix → zero vector.

**Resulting distribution:** p10 = 0.40 (10-month retraining), p50 = 1.00 (24-month retraining), mean = 0.82. The distribution spans the full [0, 1] range and produces semantically correct results: software developers ↔ programmers: d = 0.000; software developers ↔ carpenters: d = 1.000.

**Impact:** Retraining durations now range from near-zero (for closely related occupations) to 24 months (for maximally distant occupations), producing realistic cross-occupation mobility patterns rather than a degenerate mass at maximum distance.

---

## 4. BLS OES Multi-Industry Occupation Distribution

**Prior method:** Each worker's industry sector (NAICS 2-digit) was assigned deterministically from their CPS-reported IND1990 code. Since the CPS records the specific employer's industry for each respondent, occupations that work across many industries (e.g., janitors, security guards, food service workers) were pinned to whichever industry happened to employ that CPS respondent. This created spurious sector effects: in the CPS sample, janitors appeared overwhelmingly in healthcare (NAICS 62) simply due to sampling variation, while maids appeared in Other Services (NAICS 81).

**Revised method:** For each worker, NAICS sector is drawn stochastically from P(sector | occupation) distributions derived from BLS Occupational Employment Statistics (OES) national employment data:

1. Load `oes_4dig_naics.dta` (national, detailed occupation level).
2. Collapse employment counts to (SOC code × 2-digit NAICS sector).
3. Normalize to probabilities: `P(sector | SOC) = emp(sector, SOC) / Σ_sector emp(sector, SOC)`.
4. For Census aggregate SOC codes containing 'X' (e.g., `37-201X` for janitors), use prefix matching restricted to codes not separately represented in the crosswalk — this prevents cross-contamination between closely related occupations coded separately (e.g., maids `37-2012` are excluded from janitor prefix expansions).
5. Workers whose occupation has no OES coverage fall back to IND1990 deterministic assignment.

**Resulting distributions (examples):**
- Janitors: 57.6% in NAICS 56 (Building Services), 15.8% in 61 (Education), 6.3% in 62 (Healthcare) — vs. OES target: 45.9%, 19.1%, 7.2%
- Maids: 45.4% in NAICS 72 (Accommodation), 27.8% in 62 (Healthcare), 18.5% in 56 (Building Services) — vs. OES target: 52.1%, 26.1%, 14.9%

Coverage: 66.8% of OCC2010 codes have OES distributions; 33.2% fall back to IND1990.

**Impact:** Resolves the maids/janitors sector artifact. Previously, janitors (lower AI risk: r_job = 0.42) appeared to gain employment while maids (higher risk: r_job = 0.48) appeared to lose employment — the opposite of what risk scores predict — because janitors were placed in high-BTOS-drift healthcare while maids were placed in negative-BTOS-drift Other Services. Under OES-based assignment, both occupy economically appropriate sectors and outcomes align with their risk score ordering.

---

## 5. Individualized Substitution Risk with Experience Shield

**Method (unchanged but clarified):** Each worker's effective substitution risk and augmentation potential are individualized based on `exp_norm` (normalized tenure/experience, [0, 1]):

```
r_i^sub = r_job × (1 − δ_sub × exp_norm)
p_i^aug = p_aug × (1 + δ_aug × exp_norm)
```

Where `δ_sub = 0.30` and `δ_aug = 0.40`. Senior workers (exp_norm ≈ 1.0) face 30% lower substitution risk and 40% higher augmentation potential. Combined with preferential rehiring in market clearing (hiring score ∝ 1 + ν·exp_norm), senior workers maintain higher employment in the AI scenario.

**Empirical result:** At T=59, senior workers maintain 89.9% employment in the AI scenario vs. 85.4% for entry workers — consistent with the theoretical expectation that experience shields workers from displacement.

---

## 6. Full Replacement Vacancies in Control Scenario (ε = 1.0)

**Prior method:** The vacancy creation parameter ε (fraction of AI-displaced workers who create replacement vacancies) was applied uniformly to both AI and control scenarios. This caused near-zero vacancy creation in the control model (since AI displacement = 0, ε × 0 ≈ 0), creating a spurious asymmetry where control employers never posted jobs.

**Revised method:** The control scenario uses ε = 1.0 (full replacement): every worker who leaves their position creates a replacement vacancy. The AI scenario uses the calibrated ε < 1.0 (partial replacement, reflecting that AI automation reduces labor demand for some roles). This correctly represents the counterfactual: in a world without AI disruption, employers replace all departing workers through normal turnover.

**Impact:** Control-scenario vacancies restored to economically realistic levels (~155/tick), enabling proper comparison of AI-driven vs. normal-turnover labor market dynamics.

---

## Summary Table

| # | Change | Prior Limitation | Resolution |
|---|--------|-----------------|------------|
| 1 | Logistic displacement formula | Linear formula saturated; CI collapsed to 0.65 pp | Sigmoid bounds P_disp in (0,1); CI width = 10.6 pp |
| 2 | β_run macroeconomic multiplier | Fixed β gave identical macroeconomic conditions across seeds | Per-seed draw from N(1.0, 0.2) introduces realistic macro uncertainty |
| 3 | O\*NET cosine skill distance | Wu-Palmer: 94% of pairs at d=1.0 (degenerate) | Z-scored cosine over 41 Work Activities; full-range distribution |
| 4 | BLS OES industry assignment | CPS single-industry pinning distorted sector BTOS effects | Stochastic draw from P(sector\|occupation) anchored to national OES employment |
| 5 | Experience-based risk individualization | (Clarification) | δ_sub=0.30 shield; δ_aug=0.40 boost; preferential rehiring |
| 6 | ε=1.0 for control replacement | ε applied to control created near-zero control vacancies | Control uses full replacement; AI uses calibrated partial ε |
