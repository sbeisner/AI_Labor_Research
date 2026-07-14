"""Live ABM visualizer — FastAPI server.

Run from the repo root:
    venv/bin/uvicorn viz.server:app --reload --port 8765

Then open  http://localhost:8765/  in a browser.

Architecture
------------
One WebSocket at /stream, bidirectional:
  • Client → server: JSON commands  {"cmd": "init|step|play|pause|reset", ...}
  • Server → client: JSON snapshots {"type": "snapshot", ...} or
                                   {"type": "status",   "msg":  "..."}

The model, sampled-worker list, and play/pause state are session-scoped
(per WebSocket connection). Single-user assumption; fine for a thesis demo.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from typing import Any

# Ensure the project root is on sys.path when uvicorn imports this module
_ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from model.LaborMarketModel import LaborMarketModel, DEFAULT_PARAMS
from scripts.bootstrap_runner import load_shared_data
from viz.snapshot import build_snapshot, stratified_worker_sample


app = FastAPI(title="ai_labor_research viz")

_STATIC_DIR = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def index():
    # No-cache so a freshly edited index.html shows up on the next reload
    # without needing to evict browser cache manually.
    return FileResponse(
        str(_STATIC_DIR / "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ── Shared CPS / O*NET data, loaded once at startup ──────────────────────────
# Loading worker_sample_with_risk.parquet costs a few seconds; we do it once
# at process start and reuse across sessions.

_shared: dict[str, Any] = {}


@app.on_event("startup")
async def _preload_data():
    print("[viz] preloading shared simulation data…", flush=True)
    worker_df, dist_matrix, occ_risk = load_shared_data()
    _shared["worker_df"]    = worker_df
    _shared["dist_matrix"]  = dist_matrix
    _shared["occ_risk"]     = occ_risk
    print("[viz] ready.", flush=True)


# ── WebSocket session ────────────────────────────────────────────────────────


class Session:
    """Per-connection simulation state."""

    def __init__(self):
        self.model: LaborMarketModel | None = None
        self.sampled: list = []
        self.playing: bool = False
        self.tick_interval: float = 0.6   # seconds between auto-ticks
        self.n_per_quintile: int = 80     # ~400 worker nodes total
        self.ai_active: bool = True
        self.seed: int = 42

    def init_model(self, *, ai_active: bool, seed: int, n_per_quintile: int):
        self.ai_active = ai_active
        self.seed = seed
        self.n_per_quintile = n_per_quintile
        self.model = LaborMarketModel(
            worker_df=_shared["worker_df"],
            params=DEFAULT_PARAMS,
            ai_active=ai_active,
            seed=seed,
            skill_distance_matrix=_shared["dist_matrix"],
            occ_risk_lookup=_shared["occ_risk"],
            collect_agent_data=False,
        )
        self.sampled = stratified_worker_sample(
            self.model, n_per_quintile=n_per_quintile, seed=seed
        )
        self.playing = False

    def snapshot(self) -> dict:
        assert self.model is not None
        return build_snapshot(self.model, self.sampled)

    def step(self):
        assert self.model is not None
        self.model.step()


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    sess = Session()

    async def send_status(msg: str):
        await ws.send_json({"type": "status", "msg": msg})

    async def send_snapshot():
        await ws.send_json({"type": "snapshot", **sess.snapshot()})

    async def auto_tick_loop():
        """While playing, advance the model and stream snapshots."""
        while sess.playing and sess.model is not None:
            try:
                # Run the (CPU-bound) step in a thread so the websocket
                # event loop stays responsive to pause commands.
                await asyncio.to_thread(sess.step)
                await send_snapshot()
                await asyncio.sleep(sess.tick_interval)
            except Exception as exc:  # noqa: BLE001
                sess.playing = False
                await send_status(f"error during step: {exc!r}")
                return

    auto_task: asyncio.Task | None = None

    try:
        await send_status("connected — send {\"cmd\":\"init\"} to start")

        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd")

            if cmd == "init":
                if auto_task and not auto_task.done():
                    sess.playing = False
                    auto_task.cancel()
                await send_status("initializing model…")
                await asyncio.to_thread(
                    sess.init_model,
                    ai_active=bool(msg.get("ai_active", True)),
                    seed=int(msg.get("seed", 42)),
                    n_per_quintile=int(msg.get("n_per_quintile", 80)),
                )
                await send_snapshot()
                await send_status(
                    f"model ready (ai_active={sess.ai_active}, "
                    f"seed={sess.seed}, sampled={len(sess.sampled)})"
                )

            elif cmd == "step":
                if sess.model is None:
                    await send_status("init the model first")
                    continue
                await asyncio.to_thread(sess.step)
                await send_snapshot()

            elif cmd == "play":
                if sess.model is None:
                    await send_status("init the model first")
                    continue
                if sess.playing:
                    continue
                sess.playing = True
                if "interval" in msg:
                    sess.tick_interval = max(0.05, float(msg["interval"]))
                auto_task = asyncio.create_task(auto_tick_loop())
                await send_status(f"playing (interval={sess.tick_interval}s)")

            elif cmd == "pause":
                sess.playing = False
                if auto_task and not auto_task.done():
                    auto_task.cancel()
                await send_status("paused")

            elif cmd == "reset":
                sess.playing = False
                if auto_task and not auto_task.done():
                    auto_task.cancel()
                sess.model = None
                sess.sampled = []
                await send_status("reset — send init to start a new run")

            else:
                await send_status(f"unknown cmd: {cmd!r}")

    except WebSocketDisconnect:
        sess.playing = False
        if auto_task and not auto_task.done():
            auto_task.cancel()
