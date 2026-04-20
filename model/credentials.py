"""Credential system constants and helpers for the AI Labor Market ABM.

Isolated here to avoid circular imports: LaborMarketModel imports Worker/Employer,
and those agents need the credential utilities, so they must not import from
LaborMarketModel directly.
"""

from collections import deque
import numpy as np


# ── Credential hierarchy ──────────────────────────────────────────────────────
CREDENTIAL_LEVELS = ["high_school", "vocational", "associates",
                     "bachelors", "masters", "doctoral"]
CREDENTIAL_IDX    = {c: i for i, c in enumerate(CREDENTIAL_LEVELS)}

# Directed graph: source → [(target, months)]
# HS is the root; doctoral is the ceiling.
# Vocational and Associates are lateral entry-points for trades.
CREDENTIAL_GRAPH = {
    "high_school": [("vocational", 12), ("associates", 24), ("bachelors", 48)],
    "vocational":  [("associates", 12)],
    "associates":  [("bachelors",  24)],
    "bachelors":   [("masters",    24), ("doctoral",  48)],
    "masters":     [("doctoral",   24)],
    "doctoral":    [],
}

# Minimum credential required by O*NET Job Zone.
# Aligned to the O*NET Job Zone Reference descriptions in our data:
#   Zone 2 (combined 1-2): "Usually requires a high school diploma or GED"
#   Zone 3: "vocational schools, related on-the-job experience, or an associate's degree"
#   Zone 4: "Most require a four-year bachelor's degree"
#   Zone 5: "Most require graduate school"
# Note: O*NET merged zones 1-2 into a single zone 2 category in modern releases,
# so zone 2 is the practical floor with a high school minimum.
ZONE_MIN_CREDENTIAL = {
    1: "high_school",   # floor (no zone 1 in current O*NET; kept for completeness)
    2: "high_school",   # zone 2 = "little to some prep" — HS diploma sufficient
    3: "associates",    # zone 3 = vocational/trade school or associate's degree
    4: "bachelors",     # zone 4 = bachelor's degree expected
    5: "masters",       # zone 5 = graduate school expected
}


def educ_to_credential(educ: int) -> str:
    """Map IPUMS CPS EDUC code to credential level string.

    IPUMS CPS EDUC detailed codes:
      002–073 : no schooling through HS diploma/GED      → high_school
      081     : some college, no degree                  → vocational
      091–092 : associate's degree (occupational/academic) → associates
      111     : bachelor's degree                        → bachelors
      123     : master's degree                          → masters
      124–125 : professional school / doctoral degree    → doctoral
    """
    if educ <= 73:    # no schooling through HS diploma/GED
        return "high_school"
    if educ <= 81:    # some college, no degree (closest to vocational training)
        return "vocational"
    if educ <= 92:    # associate's degree (occupational 091, academic 092)
        return "associates"
    if educ <= 111:   # bachelor's degree
        return "bachelors"
    if educ <= 123:   # master's degree
        return "masters"
    return "doctoral" # professional school (124) or doctoral (125)


def _build_distance_cache() -> dict:
    """Precompute all pairwise shortest-path distances between credential levels.

    Only 6×6 = 36 pairs exist.  Storing them as a dict turns every subsequent
    credential_months_to() call into an O(1) lookup instead of a BFS traversal,
    which matters when _choose_target_skill() iterates over all 537 candidate
    occupations for each retraining worker.
    """
    cache: dict = {}
    for src in CREDENTIAL_LEVELS:
        for tgt in CREDENTIAL_LEVELS:
            if CREDENTIAL_IDX[src] >= CREDENTIAL_IDX[tgt]:
                cache[(src, tgt)] = 0
                continue
            q: deque = deque([(src, 0)])
            seen = {src}
            result = 999
            while q:
                node, cost = q.popleft()
                for nxt, months in CREDENTIAL_GRAPH.get(node, []):
                    total = cost + months
                    if nxt == tgt:
                        result = total
                        q.clear()
                        break
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, total))
            cache[(src, tgt)] = result
    return cache


# Module-level cache — built once at import time, shared across all workers
# in the joblib process pool (each process imports once).
_CRED_DIST: dict = _build_distance_cache()


def credential_months_to(src: str, tgt: str) -> int:
    """Shortest path (months) from src credential to tgt in the DAG.

    O(1) lookup via precomputed cache.  Returns 0 if src already meets or
    exceeds tgt; returns 999 if the path is unreachable (should never occur
    with a properly connected graph).
    """
    return _CRED_DIST.get((src, tgt), 0 if CREDENTIAL_IDX.get(src, 0) >= CREDENTIAL_IDX.get(tgt, 0) else 999)


# ── Vectorized distance matrix ────────────────────────────────────────────────
# Shape (6, 6): CRED_DIST_MATRIX[src_idx, tgt_idx] → months.
# Used in Worker._choose_target_skill() to replace the 537-iteration Python
# list comprehension with a single numpy fancy-index operation.
CRED_DIST_MATRIX: np.ndarray = np.array(
    [[_CRED_DIST[(src, tgt)] for tgt in CREDENTIAL_LEVELS]
     for src in CREDENTIAL_LEVELS],
    dtype=np.int32,
)
