try:
    from scripts.plot_utils import setup_matplotlib, show_or_save
except ImportError:
    from plot_utils import setup_matplotlib, show_or_save
setup_matplotlib()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Load agent-level data ─────────────────────────────────────────────────────
agent_df = pd.read_parquet('output/validation_agent_df.parquet')

# Pivot to wide format: rows = (agent_id, tick), need is_employed trajectory
# agent_df columns: agentid, tick, is_employed, months_unemployed, ...
agent_df = agent_df[['agentid', 'tick', 'is_employed']].dropna()
agent_df['is_employed'] = agent_df['is_employed'].astype(bool)

# ── Build cohort: agents employed at tick 9 and unemployed at tick 10 ─────────
ENTRY_TICK = 10
employed_before = set(
    agent_df.loc[(agent_df['tick'] == ENTRY_TICK - 1) & agent_df['is_employed'], 'agentid']
)
unemployed_at_entry = set(
    agent_df.loc[(agent_df['tick'] == ENTRY_TICK) & ~agent_df['is_employed'], 'agentid']
)
cohort = employed_before & unemployed_at_entry

# ── For each cohort agent, find spell duration ────────────────────────────────
# Spell ends when is_employed flips back to True; otherwise censored at tick 59
MAX_TICK      = agent_df['tick'].max()
FOLLOW_TICKS  = MAX_TICK - ENTRY_TICK

records = []
cohort_df = agent_df[agent_df['agentid'].isin(cohort)].sort_values(['agentid', 'tick'])

for agent_id, traj in cohort_df.groupby('agentid'):
    post = traj[traj['tick'] >= ENTRY_TICK].reset_index(drop=True)
    rehire = post[post['is_employed']]
    if len(rehire) > 0:
        duration = int(rehire.iloc[0]['tick']) - ENTRY_TICK
        censored = False
    else:
        duration = FOLLOW_TICKS
        censored = True
    records.append({'duration': duration, 'censored': censored})

spell_df = pd.DataFrame(records)
n_cohort  = len(spell_df)
n_events  = (~spell_df['censored']).sum()

# ── Kaplan-Meier estimator ────────────────────────────────────────────────────
times    = sorted(spell_df.loc[~spell_df['censored'], 'duration'].unique())
surv     = 1.0
km_times = [0]
km_surv  = [1.0]
km_ci_lo = [1.0]
km_ci_hi = [1.0]

at_risk = n_cohort
event_idx = 0
sorted_durations = spell_df['duration'].sort_values().values

for t in times:
    n_events_t = ((spell_df['duration'] == t) & ~spell_df['censored']).sum()
    # Greenwood's formula for 95% CI
    if at_risk > 0 and n_events_t > 0:
        surv *= (1 - n_events_t / at_risk)
    se = surv * np.sqrt(
        sum(d['n_events_t'] / (d['at_risk'] * (d['at_risk'] - d['n_events_t']))
            for d in [{'n_events_t': n_events_t, 'at_risk': at_risk}]
            if d['at_risk'] > d['n_events_t'] > 0)
    ) if at_risk > n_events_t > 0 else 0.0

    km_times.append(t)
    km_surv.append(surv)
    km_ci_lo.append(max(0, surv - 1.96 * se))
    km_ci_hi.append(min(1, surv + 1.96 * se))

    # Advance at-risk count past censored observations before next event time
    next_t = times[times.index(t) + 1] if t != times[-1] else MAX_TICK
    censored_between = ((spell_df['duration'] >  t) &
                        (spell_df['duration'] <= next_t) &
                         spell_df['censored']).sum()
    at_risk -= (n_events_t + censored_between)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.step(km_times, km_surv, where='post',
        color='#1565C0', linewidth=2.2, label='K-M Survival Estimate')
ax.fill_between(km_times, km_ci_lo, km_ci_hi,
                step='post', alpha=0.18, color='#1565C0',
                label='95% Confidence Band')

# Median survival time
median_idx = next((i for i, s in enumerate(km_surv) if s <= 0.5), None)
if median_idx is not None:
    med_t = km_times[median_idx]
    ax.axvline(med_t, color='#E53935', linewidth=1.2,
               linestyle='--', alpha=0.8, label=f'Median spell = {med_t} ticks')
    ax.axhline(0.5,   color='#E53935', linewidth=0.8, linestyle=':', alpha=0.6)

ax.set_xlabel('Months Unemployed (Ticks Since Separation)', fontsize=11)
ax.set_ylabel('Probability of Remaining Unemployed S(t)', fontsize=11)
ax.set_title(
    f'Kaplan–Meier Survival Analysis — Unemployment Spell Duration\n'
    f'Cohort: agents displaced at tick {ENTRY_TICK} '
    f'(N={n_cohort}, {n_events} re-employed, '
    f'{n_cohort - n_events} censored)',
    fontsize=10, fontweight='bold'
)
ax.set_xlim(0, FOLLOW_TICKS)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
show_or_save(fig, 'validation_km')
plt.close(fig)
