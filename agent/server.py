"""FastAPI web interface: one page, one websocket, three models behind one shape.

Run it, open http://127.0.0.1:8770, pick a policy, press Start. The decision log shows
every typed answer with its probability and latency, which is the point: watching the
probabilities rather than the score is how you tell a policy that is playing from one
that is guessing.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .envs import make_env
from .loop import ControlLoop, LoopConfig
from .policies import build

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="arcade-decision-agent")

EVENTS: deque = deque(maxlen=400)
SUBS: set[asyncio.Queue] = set()
LOOP: asyncio.AbstractEventLoop | None = None


class Runner:
    """Owns the worker thread. Playwright's sync API must stay on its creating thread,
    so env construction happens inside the thread, not in the request handler."""

    def __init__(self) -> None:
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[ControlLoop] = None
        self.env = None
        self.config: dict = {}
        self.summary: dict = {}
        self.error: str = ""

    # ------------------------------------------------------------------ control ---
    def start(self, cfg: dict) -> dict:
        if self.running():
            return {"ok": False, "error": "already running"}
        self.config, self.summary, self.error = cfg, {}, ""
        self.thread = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        self.thread.start()
        return {"ok": True, "config": cfg}

    def _run(self, cfg: dict) -> None:
        try:
            env_kw = {}
            if cfg["env"] == "page":
                env_kw = {"url": cfg.get("url"), "headed": cfg.get("headed", False),
                          "hold_ms": cfg.get("hold_ms", 60)}
            self.env = make_env(cfg["env"], seed=cfg.get("seed", 0), **env_kw)
            policy_kw = {"threshold": cfg.get("threshold", 0.5),
                         "state_format": cfg.get("state_format", "prose")}
            if cfg.get("policy", "").startswith("luna") or (cfg.get("supervisor") or "").startswith("luna"):
                policy_kw["send_image"] = cfg.get("send_image", True)
            reflex = build(cfg["policy"], **policy_kw)
            supervisor = build(cfg["supervisor"], **policy_kw) if cfg.get("supervisor") else None
            loop_cfg = LoopConfig(max_steps=cfg.get("steps", 400),
                                  decimate=cfg.get("decimate", 1),
                                  supervisor_every=cfg.get("supervisor_every", 20),
                                  frame_delay_ms=cfg.get("hold_ms", 60) if cfg["env"] == "page" else 0)
            self.loop = ControlLoop(self.env, reflex, supervisor, loop_cfg, on_event=self._emit)
            self._publish({"type": "started", "config": cfg,
                           "reflex_health": reflex.health(),
                           "supervisor_health": supervisor.health() if supervisor else None})
            self.loop.run()
            self.summary = self.loop.summary()
        except Exception as exc:  # surface the failure in the UI instead of a dead thread
            self.error = f"{type(exc).__name__}: {exc}"
            self._publish({"type": "error", "error": self.error})
        finally:
            try:
                if self.env:
                    self.env.close()
            except Exception:
                pass
            self.env = None
            self._publish({"type": "finished", "summary": self.summary, "error": self.error})

    def stop(self) -> dict:
        if self.loop:
            self.loop.stop()
        return {"ok": True}

    def manual(self, action: str) -> dict:
        if self.loop:
            self.loop.push_manual(action)
            return {"ok": True, "queued": action}
        return {"ok": False, "error": "not running"}

    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    # ------------------------------------------------------------------ publish ---
    def _emit(self, event: dict) -> None:
        if event["type"] == "obs" and self.env is not None:
            try:
                if hasattr(self.env, "ascii_frame"):
                    event["ascii"] = self.env.ascii_frame()
                frame = self.env.frame() if self.config.get("env") == "page" else None
                if frame:
                    event["frame"] = base64.b64encode(frame).decode()
            except Exception:
                pass
        self._publish(event)

    def _publish(self, event: dict) -> None:
        event["t"] = event.get("t", time.time())
        EVENTS.append(event)
        if LOOP:
            for q in list(SUBS):
                try:
                    LOOP.call_soon_threadsafe(q.put_nowait, event)
                except Exception:
                    pass


RUNNER = Runner()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({
        "runner": {"running": RUNNER.running(), "config": RUNNER.config,
                   "summary": RUNNER.summary, "error": RUNNER.error},
        "events": len(EVENTS),
        "policies": {
            spec: build(spec).health() for spec in ("scripted", "laya", "jev", "luna")
        },
        "envs": ["stub", "page"],
    })


@app.get("/api/state")
def state(limit: int = 40) -> JSONResponse:
    evts = [e for e in list(EVENTS)[-limit:]]
    return JSONResponse({"running": RUNNER.running(), "config": RUNNER.config,
                         "summary": RUNNER.summary, "error": RUNNER.error, "events": evts})


@app.post("/api/start")
async def start(cfg: dict) -> JSONResponse:
    cfg.setdefault("env", "stub")
    cfg.setdefault("policy", "scripted")
    return JSONResponse(RUNNER.start(cfg))


@app.post("/api/stop")
def stop() -> JSONResponse:
    return JSONResponse(RUNNER.stop())


@app.post("/api/manual")
async def manual(body: dict) -> JSONResponse:
    return JSONResponse(RUNNER.manual(body.get("action", "hold")))


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    SUBS.add(q)
    try:
        for e in list(EVENTS)[-25:]:
            await sock.send_text(json.dumps(e, default=str))
        while True:
            try:
                e = await asyncio.wait_for(q.get(), timeout=20)
                await sock.send_text(json.dumps(e, default=str))
            except asyncio.TimeoutError:
                await sock.send_text(json.dumps({"type": "ping", "t": time.time()}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        SUBS.discard(q)


@app.on_event("startup")
async def _startup() -> None:
    global LOOP
    LOOP = asyncio.get_running_loop()


def main(argv: Optional[list] = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(prog="agent.server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args(argv)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())