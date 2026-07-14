"""Snapshot extraction for the live ABM visualizer.

The full economy is too large to render in a browser graph (~100k workers,
several thousand employers). We pick a stratified worker subsample once at
session start and follow those same agents across every tick — that way the
visual story is stable: the same worker dot stays on screen as it moves
through employed → displaced → retraining → re-employed, etc.

Employers shown are exactly those touching any sampled worker (current
attachment, or recent attachment if a worker just separated). Their stats
(roster size, vacancies, btos, a_jt, state) reflect the *full* roster, not
the sampled one — sizes and colors stay meaningful.
"""

from __future__ import annotations

from typing import Any

from agents.Worker import WorkerAgent
from agents.Employer import EmployerAgent


def stratified_worker_sample(model, n_per_quintile: int = 100, seed: int = 42):
    """Return a list of WorkerAgent instances stratified by exposure quintile.

    Sampled once at session start; the same UIDs are tracked every tick.
    """
    import random

    rng = random.Random(seed)
    by_q: dict[str, list] = {}
    for w in model.agents_by_type[WorkerAgent]:
        by_q.setdefault(str(w.exposure_quintile), []).append(w)

    sampled: list = []
    for q, group in by_q.items():
        rng.shuffle(group)
        sampled.extend(group[:n_per_quintile])
    return sampled


def _worker_state(w: WorkerAgent) -> str:
    if w.is_retired:
        return "retired"
    if getattr(w, "is_olf", False):
        return "olf"
    if w.retraining_ticks_left > 0:
        return "retraining"
    if w.is_employed:
        return "employed"
    return "unemployed"


def build_snapshot(model, sampled_workers: list[WorkerAgent]) -> dict[str, Any]:
    """Build one tick's worth of state for the frontend.

    Returns a dict with:
      tick:      int
      ai_active: bool
      workers:   list of {id, state, occ, exp, credential, retrained, employer}
      employers: list of {id, sector, size, vacancies, state, btos, a_jt}
      aggregates: {employment_rate, unemployed, retraining, mean_wage, ...}
    """
    workers_payload = []
    employers_seen: dict[str, EmployerAgent] = {}

    for w in sampled_workers:
        emp = w.employer
        emp_id = None
        if emp is not None and emp.state != "Failed":
            emp_id = str(emp.ind_key)
            employers_seen[emp_id] = emp

        workers_payload.append({
            "id": f"w{w.unique_id}",
            "state": _worker_state(w),
            "occ": int(w.current_occ),
            "search_occ": int(w.search_occ) if w.search_occ is not None else None,
            "exp": round(float(w.exp_norm), 3),
            "credential": w.credential,
            "retrained": bool(w.has_retrained),
            "exposure_q": str(w.exposure_quintile),
            "wage": round(float(w.wage), 2),
            "age": int(w.age),
            "employer": emp_id,
        })

    # Always include any non-failed employer that has at least one tracked
    # worker. Could expand to "employers in same industry" if you want
    # spectator firms; left out to keep the graph readable.
    employers_payload = []
    for emp_id, emp in employers_seen.items():
        employers_payload.append({
            "id": emp_id,
            "sector": str(emp.sector),
            "size": len(emp._roster),
            "vacancies": int(emp.vacancies),
            "state": emp.state,
            "btos": round(float(emp.btos_signal), 4),
            "a_jt": round(float(emp.a_jt), 4),
            "hired": int(getattr(emp, "_hired_this_tick", 0)),
            "fired": int(getattr(emp, "_fired_this_tick", 0)),
        })

    # Aggregates over the FULL economy (not the sample) — for the sidebar.
    all_workers = list(model.agents_by_type[WorkerAgent])
    active = [w for w in all_workers
              if not w.is_retired and not getattr(w, "is_olf", False)]
    n_active = len(active) or 1
    employed = sum(1 for w in active if w.is_employed)
    retraining = sum(1 for w in all_workers
                     if w.retraining_ticks_left > 0 and not w.is_retired)
    wages = [w.wage for w in all_workers
             if w.is_employed and not w.is_retired and w.wage > 0]
    mean_wage = float(sum(wages) / len(wages)) if wages else 0.0

    employer_states = {"Healthy": 0, "Distressed": 0, "Failed": 0}
    total_vacancies = 0
    for emp in model._employers.values():
        employer_states[emp.state] = employer_states.get(emp.state, 0) + 1
        if emp.state != "Failed":
            total_vacancies += int(emp.vacancies)

    aggregates = {
        "tick": int(model.tick),
        "employment_rate": round(employed / n_active, 4),
        "unemployment_rate": round(1.0 - employed / n_active, 4),
        "retraining": retraining,
        "mean_wage": round(mean_wage, 2),
        "total_vacancies": total_vacancies,
        "employer_states": employer_states,
        "spinoffs_this_tick": int(getattr(model, "_spinoffs_this_tick", 0)),
        "retirements_this_tick": int(getattr(model, "_retirements_this_tick", 0)),
        "entries_this_tick": int(getattr(model, "_entries_this_tick", 0)),
    }

    return {
        "tick": int(model.tick),
        "ai_active": bool(model.ai_active),
        "workers": workers_payload,
        "employers": employers_payload,
        "aggregates": aggregates,
    }
