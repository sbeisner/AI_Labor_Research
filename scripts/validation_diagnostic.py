"""Single diagnostic simulation with full agent-level data collection.

Shared prerequisite for the K-S and Kaplan-Meier validation tests.
Runs ONE control simulation (ai_active=False) with agent reporters enabled,
captures firm-size and wage snapshots at tick 0 and tick 60 (post burn-in).

Cached to:
  output/validation_agent_df.parquet   — per-tick per-agent state
  output/validation_snapshots.parquet  — firm sizes and wages at t0 / t60
Set FORCE_RERUN = True to regenerate.
"""
import sys
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path('.').resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS
from agents.Worker import WorkerAgent
from scripts.bootstrap_runner import load_shared_data

AGENT_PATH    = ROOT / 'output' / 'validation_agent_df.parquet'
SNAPSHOT_PATH = ROOT / 'output' / 'validation_snapshots.parquet'
N_TICKS       = 72
BURN_IN       = 12
FORCE_RERUN   = True   # set False after first run

if AGENT_PATH.exists() and SNAPSHOT_PATH.exists() and not FORCE_RERUN:
    print('[validation_diagnostic] Loaded from cache.')
else:
    print('[validation_diagnostic] Running single diagnostic simulation …')
    worker_df, dist_matrix, occ_risk = load_shared_data()

    # Use seed=0 with no resampling for a reproducible diagnostic
    m = LaborMarketModel(
        worker_df,
        params=DEFAULT_PARAMS,
        ai_active=False,
        seed=0,
        skill_distance_matrix=dist_matrix,
        occ_risk_lookup=occ_risk,
        collect_agent_data=True,
    )

    # ── Snapshot firm sizes and wages at tick 0 (post-init, pre-step) ─────────
    firm_sizes_t0 = [len(e._roster) for e in m._employers.values()]
    wages_t0      = [w.wage        for w in m.agents_by_type[WorkerAgent]]

    # ── Run full simulation ───────────────────────────────────────────────────
    for _ in range(N_TICKS):
        m.step()

    # ── Snapshot at tick 60 (post burn-in) ────────────────────────────────────
    firm_sizes_t60 = [len(e._roster) for e in m._employers.values()]
    wages_t60      = [w.wage        for w in m.agents_by_type[WorkerAgent] if w.is_employed]

    # ── Save agent dataframe (large — one row per agent per tick) ─────────────
    agent_df = m.datacollector.get_agent_vars_dataframe().reset_index()
    agent_df.columns = agent_df.columns.str.lower().str.replace(' ', '_')
    # Discard burn-in ticks; renumber 0–59
    agent_df = agent_df[agent_df['step'] >= BURN_IN].copy()
    agent_df['tick'] = agent_df['step'] - BURN_IN
    agent_df = agent_df.drop(columns=['step'])
    AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    agent_df.to_parquet(AGENT_PATH, index=False)

    # ── Save snapshot arrays ─────────────────────────────────────────────────
    max_len = max(len(firm_sizes_t0), len(firm_sizes_t60),
                  len(wages_t0),      len(wages_t60))
    snap = pd.DataFrame({
        'firm_size_t0':  pd.array(firm_sizes_t0 + [np.nan] * (max_len - len(firm_sizes_t0))),
        'firm_size_t60': pd.array(firm_sizes_t60 + [np.nan] * (max_len - len(firm_sizes_t60))),
        'wage_t0':       pd.array(wages_t0  + [np.nan] * (max_len - len(wages_t0))),
        'wage_t60':      pd.array(wages_t60 + [np.nan] * (max_len - len(wages_t60))),
    })
    snap.to_parquet(SNAPSHOT_PATH, index=False)
    print(f'[validation_diagnostic] Done. '
          f'Firms t0={len(firm_sizes_t0)}, t60={len(firm_sizes_t60)} | '
          f'Wages t0={len(wages_t0)}, t60={len(wages_t60)}')
