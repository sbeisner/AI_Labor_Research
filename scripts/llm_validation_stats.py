"""LLM risk-score validation statistics (no simulation).

Computes the validation statistics for the LLM-generated occupational risk
scores (R_job) from artifacts already in the repository, and records honestly
which requested statistics cannot be computed because their source artifact is
not present.

Computed from available data:
  * Spearman ρ of R_job against the Felten et al. AI Occupational Exposure
    index (AIOE) and its Language-Modeling variant (LM-AIOE), at the
    occupation level (primary) and worker level. Both are pre-joined into
    data/processed/worker_sample_with_risk.parquet.
  * Anchor-perturbation robustness of the hard-skill quintile assignment from
    output/anchor_sensitivity.csv (leave-one-out over the 15 curated anchors):
    Spearman ρ of the perturbed vs. baseline hard-skill score, and the share
    of workers changing quintile — the perturbation actually available in the
    repository.

Requested but NOT computable from repository artifacts (recorded as null with
an explicit note so the archived files can be supplied later):
  * Inter-rater agreement (κ / ICC) on the 5% human-validated sample — no
    human-scored file is present in the repo.
  * Spearman ρ of R_job against the Eloundou et al. GPT-exposure index — no
    Eloundou index file is present in the repo.

Output → output/llm_validation_stats.json
       → supplement_llm_validation.md  (draft subsection for supplement.qmd, S-LLM)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).parent.parent.resolve()
WORKER_DF = ROOT / "data" / "processed" / "worker_sample_with_risk.parquet"
ANCHOR_CSV = ROOT / "output" / "anchor_sensitivity.csv"
RISK_SCORES = ROOT / "output" / "risk_scores_complete.csv"
OUT_JSON = ROOT / "output" / "llm_validation_stats.json"
OUT_MD = ROOT / "supplement_llm_validation.md"

# Artifacts that, if added to the repo, would unlock the remaining statistics.
HUMAN_SAMPLE_CANDIDATES = ["human_scores.csv", "human_validation_sample.csv",
                           "risk_scores_human.csv"]
ELOUNDOU_CANDIDATES = ["eloundou_gpt_exposure.csv", "eloundou_exposure.csv",
                       "gpts_are_gpts_exposure.csv"]


def _first_present(names: list[str]) -> str | None:
    for n in names:
        for base in (ROOT / "data" / "external", ROOT / "data" / "processed",
                     ROOT / "output", ROOT / "data"):
            if (base / n).exists():
                return str((base / n).relative_to(ROOT))
    return None


def felten_correlations() -> dict:
    w = pd.read_parquet(WORKER_DF)
    out = {"n_workers": int(len(w)), "n_occupations": int(w["OCC2010"].nunique())}

    # Worker level
    m = w[["r_job", "AIOE", "LM_AIOE"]].dropna()
    out["worker_level"] = {
        "spearman_rjob_AIOE": round(float(spearmanr(m["r_job"], m["AIOE"]).correlation), 4),
        "spearman_rjob_LM_AIOE": round(float(spearmanr(m["r_job"], m["LM_AIOE"]).correlation), 4),
        "n": int(len(m)),
    }
    # Occupation level (one row per OCC2010; the natural unit for R_job vs an index)
    occ = (w.dropna(subset=["r_job", "AIOE", "LM_AIOE"])
             .groupby("OCC2010")
             .agg(r_job=("r_job", "first"), AIOE=("AIOE", "first"),
                  LM_AIOE=("LM_AIOE", "first")))
    out["occupation_level"] = {
        "spearman_rjob_AIOE": round(float(spearmanr(occ["r_job"], occ["AIOE"]).correlation), 4),
        "spearman_rjob_LM_AIOE": round(float(spearmanr(occ["r_job"], occ["LM_AIOE"]).correlation), 4),
        "n": int(len(occ)),
    }
    return out


def anchor_robustness() -> dict:
    a = pd.read_csv(ANCHOR_CSV)
    return {
        "n_anchors": int(len(a)),
        "spearman_perturbed_vs_baseline": {
            "mean": round(float(a["spearman"].mean()), 4),
            "min": round(float(a["spearman"].min()), 4),
        },
        "pct_workers_changing_quintile": {
            "max_any_shift": round(float(a["pct_any_shift"].max()), 2),
            "mean_any_shift": round(float(a["pct_any_shift"].mean()), 2),
            "max_top_bottom_shift": round(float(a["pct_topbot_shift"].max()), 2),
        },
        "note": ("Perturbation = leave-one-out removal of each of the 15 curated "
                 "hard-skill anchors. This is the perturbation available in the "
                 "repository; the human-LLM disagreement-SD perturbation requested "
                 "in the task requires the human-scored sample (see unavailable)."),
    }


def main() -> int:
    stats: dict = {
        "description": "LLM risk-score validation statistics",
        "felten_index_correlation": felten_correlations(),
        "anchor_perturbation_robustness": anchor_robustness(),
        "unavailable": {},
    }

    human = _first_present(HUMAN_SAMPLE_CANDIDATES)
    stats["unavailable"]["inter_rater_agreement_human_5pct_sample"] = {
        "kappa": None, "icc": None,
        "status": "computed" if human else "artifact_not_in_repo",
        "required_artifact": human or f"one of {HUMAN_SAMPLE_CANDIDATES}",
        "note": ("No human-scored validation file is present in the repository. "
                 "Supply the archived 5% human-scored sample (occupation-level "
                 "human R_job) to compute Cohen's κ / ICC against the LLM scores."),
    }

    eloundou = _first_present(ELOUNDOU_CANDIDATES)
    stats["unavailable"]["spearman_rjob_eloundou"] = {
        "value": None,
        "status": "computed" if eloundou else "artifact_not_in_repo",
        "required_artifact": eloundou or f"one of {ELOUNDOU_CANDIDATES}",
        "note": ("No Eloundou et al. GPT-exposure index file is present in the "
                 "repository. Supply it (SOC-coded exposure) to compute the "
                 "Spearman correlation with R_job; the Felten AIOE correlation "
                 "is reported above as the available cross-reference."),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(f"[llm_validation_stats] wrote {OUT_JSON}")
    print(json.dumps(stats, indent=2))

    _write_supplement_draft(stats)
    print(f"[llm_validation_stats] wrote {OUT_MD}")
    return 0


def _write_supplement_draft(s: dict) -> None:
    fc = s["felten_index_correlation"]
    ar = s["anchor_perturbation_robustness"]
    occ = fc["occupation_level"]
    md = f"""## S-LLM. Validation of the LLM-generated risk scores {{#sec-supp-llm}}

The occupational substitution-risk and augmentation scores ($R_{{job}}$,
$P_{{aug}}$) are generated by a large language model over O\\*NET task
statements. This section reports the validation statistics for those scores.

**Convergent validity against an external exposure index.** We correlate the
LLM $R_{{job}}$ score with the Felten, Raj, and Seamans AI Occupational
Exposure index [@Felten2021]. At the occupation level ($n = {occ['n']}$
occupations), Spearman $\\rho = {occ['spearman_rjob_AIOE']:.2f}$ against the
general AIOE and $\\rho = {occ['spearman_rjob_LM_AIOE']:.2f}$ against its
language-modeling variant (LM-AIOE); at the worker level the corresponding
correlations are ${fc['worker_level']['spearman_rjob_AIOE']:.2f}$ and
${fc['worker_level']['spearman_rjob_LM_AIOE']:.2f}$. The moderate positive
association is consistent with $R_{{job}}$ and the AIOE measuring related but
distinct constructs: $R_{{job}}$ targets generative-AI *substitution* risk over
task content, whereas the AIOE aggregates ability-level exposure.

**Robustness of the hard-skill anchoring.** The hard-skill proximity quintile
(HSQ) used in @sec-results-decomp is defined by cosine similarity to a curated
set of {ar['n_anchors']} codified-cognitive anchor occupations. Under
leave-one-out removal of each anchor, the perturbed hard-skill score correlates
with the baseline at Spearman
$\\rho \\geq {ar['spearman_perturbed_vs_baseline']['min']:.3f}$
(mean ${ar['spearman_perturbed_vs_baseline']['mean']:.3f}$), and at most
{ar['pct_workers_changing_quintile']['max_any_shift']:.1f}\\% of workers change
quintile when any single anchor is dropped. The quintile assignment is
therefore not an artifact of any individual anchor choice.

**Statistics pending archived artifacts.** Two validation statistics referenced
in the main text could not be recomputed from the files in the replication
package as assembled and are flagged for inclusion once the source artifacts are
restored: (i) inter-rater agreement (Cohen's $\\kappa$ / ICC) between the LLM
scores and the 5\\% human-scored validation sample, and (ii) the Spearman
correlation of $R_{{job}}$ with the Eloundou et al. GPT-exposure index
[@eloundou2023]. Both require external files (the human-scored sample and the
Eloundou SOC-coded index, respectively) not currently in the repository.

<!-- Generated by scripts/llm_validation_stats.py; numbers read from
     output/llm_validation_stats.json. Re-run to refresh. -->
"""
    OUT_MD.write_text(md)


if __name__ == "__main__":
    sys.exit(main())
