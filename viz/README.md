# Live ABM Visualizer

A small browser-based viewer that streams the `LaborMarketModel` simulation
tick-by-tick and renders workers + employers as a force-directed graph.

This is intentionally rudimentary — one Python service, one HTML file, no
Node toolchain, no build step. Good enough for a thesis demo or a sanity
check while iterating on the model. Graduate to a real Vite/Next project
later if you want production polish.

## What you see

- **Worker nodes** (small dots) colored by employment state:
  - green = employed, gray = unemployed, blue = retraining,
    purple = OLF (in school), dim = retired.
- **Employer nodes** (larger dots) colored by firm state:
  - amber = Healthy, dark amber = Distressed, gray = Failed.
- **Edges** = current `worker → employer` membership. Workers with no edge
  are unemployed / OLF / retired and float free in the unemployment pool.
- **Sidebar** — controls + economy-wide aggregates over the *full*
  simulation (not the visible sample): UR, retraining count, mean wage,
  vacancies, firm-state counts, monthly spinoffs / retirements / entries.

## Sampling note

The full CPS-sampled economy has ~10k workers and ~500 employers. We render
a stratified sample (default 80 workers per exposure quintile = 400 worker
nodes) plus the employers those workers are attached to. The same UIDs are
followed across every tick, so a single dot's lifecycle (employed →
displaced → retraining → re-employed) is visually traceable. Aggregate
stats in the sidebar are computed over the *full* model, not the sample.

## Setup

```bash
# from the repo root, in the existing venv
pip install -r requirements.txt   # picks up fastapi/uvicorn/websockets
```

## Run

```bash
# from repo root
venv/bin/uvicorn viz.server:app --reload --port 8765
```

Then open http://localhost:8765/ in a browser.

The shared CPS data (`worker_sample_with_risk.parquet`, distance matrix,
risk lookup) is loaded once at server startup — first request takes a
moment, subsequent re-inits are fast.

### In the browser

1. Adjust **AI scenario active**, **seed**, **workers/quintile** if desired.
2. Click **Init / Reset** — model is built; tick 0 snapshot appears.
3. Click **Step** to advance one month at a time, or **Play** to auto-tick
   at the chosen interval. **Pause** stops the loop.

## File layout

```
viz/
  server.py        # FastAPI app + WebSocket /stream
  snapshot.py      # stratified worker sampling + per-tick snapshot builder
  static/
    index.html     # React + react-force-graph-2d (loaded from esm.sh)
  README.md
```

## Architecture

```
  ┌──────────────┐   WebSocket /stream    ┌─────────────────────┐
  │   Browser    │ ◄────────────────────► │  FastAPI server     │
  │  React +     │  ← {type:"snapshot"}   │   ↳ Session         │
  │  force-graph │  → {cmd:"step"...}     │     ↳ LaborMarket-  │
  └──────────────┘                        │       Model         │
                                          └─────────────────────┘
```

One WebSocket per browser tab; the model lives on the server side and
survives across messages. CPU-bound `model.step()` runs in
`asyncio.to_thread()` so the WebSocket stays responsive to pause commands
mid-loop.

### Wire format

Server → client:

```json
{
  "type": "snapshot",
  "tick": 12,
  "ai_active": true,
  "workers":   [{"id":"w42","state":"employed","occ":1010,"exp":0.31,"credential":"bachelors","retrained":false,"exposure_q":"Q3","wage":68.4,"age":34,"employer":"5141_3"}, ...],
  "employers": [{"id":"5141_3","sector":"51","size":18,"vacancies":2,"state":"Healthy","btos":0.014,"a_jt":0.27,"hired":1,"fired":0}, ...],
  "aggregates": {"tick":12,"employment_rate":0.954,"unemployment_rate":0.046,"retraining":612, ...}
}
```

Client → server:

```json
{"cmd":"init",  "ai_active":true, "seed":42, "n_per_quintile":80}
{"cmd":"step"}
{"cmd":"play",  "interval":0.6}
{"cmd":"pause"}
{"cmd":"reset"}
```

## Known limitations / next steps

- **Single user.** Sessions are scoped per WebSocket — fine for one person
  at the laptop, not multi-tenant.
- **No persistence.** Reload the page = fresh run. If you want to scrub
  back to tick 7, you'd need to add server-side history or replay from
  serialized snapshots.
- **No model parameter UI.** Only `ai_active`, `seed`, sample size are
  exposed. Adding sliders for `delta_base`, `k_adoption`, etc. is a small
  edit to `Sidebar` + `init_model()`.
- **No event highlighting.** A worker firing or a firm failing happens in
  one tick and the dot just changes color. If you want flashes / pulses,
  diff successive snapshots in the React layer and animate.
- **Babel-standalone-free build.** The current page uses native ESM via an
  importmap; no JSX (the code uses `React.createElement` directly). If you
  want JSX ergonomics later, swap to a Vite project (needs Node ≥ 18).
