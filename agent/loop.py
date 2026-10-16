"""The control loop: reflex tier every frame, supervisor tier every N frames.

Why two tiers, in one line of arithmetic: local Laya answers in ~10-30 ms measured on
this machine, while a network decision API costs a round trip (Jev's own docs quote
requests that are ~10x faster than an LLM chat call, which still lands in the 100 ms+
range from here). Galaga wants a steering decision maybe 10 times a second. So the
model that can answer in 20 ms drives the stick, and the model that can see or reason
better is asked less often - once every N frames - to pick the *mode* the reflex runs in.

Neither tier is allowed to stop the loop: a timeout or an HTTP error degrades to
"hold" and is recorded as a decision with an error field.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .obs import Action, GameEnv, Obs
from .policies import Policy


@dataclass
class LoopConfig:
    max_steps: int = 600
    decimate: int = 1              # ask the reflex every k-th step (hold in between)
    supervisor_every: int = 20     # ask the supervisor for a mode every k steps
    frame_delay_ms: int = 0        # real envs: how long a key stays held
    idle_sleep_ms: int = 5
    manual_priority: bool = True


@dataclass
class LoopState:
    steps: int = 0
    score: int = 0
    decisions: int = 0
    latencies: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    actions: dict = field(default_factory=dict)
    errors: int = 0
    mode: str = "engage"
    last_obs: Optional[Obs] = None
    last_action: Optional[Action] = None
    running: bool = False
    stop_reason: str = ""


class ControlLoop:
    def __init__(self, env: GameEnv, reflex: Policy, supervisor: Optional[Policy] = None,
                 cfg: Optional[LoopConfig] = None,
                 on_event: Optional[Callable[[dict], None]] = None) -> None:
        self.env = env
        self.reflex = reflex
        self.supervisor = supervisor or reflex
        self.cfg = cfg or LoopConfig()
        self.on_event = on_event or (lambda e: None)
        self.state = LoopState()
        self._manual: list[str] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------- manual input ---
    def push_manual(self, action: str) -> None:
        with self._lock:
            self._manual.append(action)

    def _take_manual(self) -> Optional[str]:
        with self._lock:
            if self.cfg.manual_priority and self._manual:
                return self._manual.pop(0)
        return None

    def stop(self) -> None:
        self.state.running = False

    # --------------------------------------------------------------------- run ---
    def run(self) -> LoopState:
        st = self.state
        st.running = True
        obs = self.env.reset()
        self._emit({"type": "obs", "step": 0, "prose": obs.prose,
                    "features": obs.features, "score": obs.score})
        while st.running and st.steps < self.cfg.max_steps and not obs.done:
            st.mode = self._mode(obs)
            obs.info["mode"] = st.mode

            manual = self._take_manual()
            if manual:
                action = Action(manual, source="manual")
            elif st.steps % max(1, self.cfg.decimate) == 0:
                action = self.reflex.decide(obs, self.env.action_space)
            else:
                action = Action("hold", source="decimated")

            if action.source in ("laya", "jev", "luna"):
                st.decisions += 1
                if lat := action.meta.get("latency_ms"):
                    st.latencies.append(float(lat))
                if cost := action.meta.get("cost_usd"):
                    st.costs.append(float(cost))
                if action.meta.get("error"):
                    st.errors += 1
            st.actions[action.name] = st.actions.get(action.name, 0) + 1

            obs = self.env.step(action)
            st.steps += 1
            st.score = obs.score
            st.last_obs, st.last_action = obs, action
            self._emit({"type": "decision", "step": st.steps, "action": action.name,
                        "source": action.source, "mode": st.mode, "score": obs.score,
                        "prose": obs.prose, "meta": action.meta})
            if self.cfg.frame_delay_ms:
                time.sleep(self.cfg.frame_delay_ms / 1000.0)
            elif self.cfg.idle_sleep_ms:
                time.sleep(self.cfg.idle_sleep_ms / 1000.0)

        st.running = False
        st.stop_reason = "done" if obs.done else "max_steps" if st.steps >= self.cfg.max_steps else "stopped"
        summary = self.summary()
        self._emit({"type": "summary", **summary})
        return st

    def _mode(self, obs: Obs) -> str:
        if self.cfg.supervisor_every <= 0 or self.state.steps % self.cfg.supervisor_every != 0:
            return self.state.mode
        try:
            mode, meta = self.supervisor.decide_mode(obs)
        except Exception as exc:
            self._emit({"type": "supervisor_error", "error": f"{type(exc).__name__}: {exc}"})
            return self.state.mode
        if meta.get("error"):
            self._emit({"type": "supervisor_error", "error": meta["error"]})
        self._emit({"type": "mode", "mode": mode, "step": self.state.steps,
                    "policy": self.supervisor.name, "meta": meta})
        return mode

    def _emit(self, event: dict) -> None:
        event.setdefault("t", time.time())
        try:
            self.on_event(event)
        except Exception:
            pass

    def summary(self) -> dict:
        st = self.state
        lat = sorted(st.latencies)
        return {
            "env": self.env.name,
            "reflex": self.reflex.name,
            "supervisor": self.supervisor.name,
            "steps": st.steps,
            "score": st.score,
            "decisions": st.decisions,
            "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
            "p95_latency_ms": round(lat[int(len(lat) * 0.95) - 1], 1) if lat else None,
            "cost_usd": round(sum(st.costs), 6),
            "errors": st.errors,
            "actions": st.actions,
            "stop_reason": st.stop_reason,
        }


def run_episode(env: GameEnv, reflex: Policy, supervisor: Optional[Policy] = None,
                cfg: Optional[LoopConfig] = None,
                on_event: Optional[Callable[[dict], None]] = None) -> LoopState:
    return ControlLoop(env, reflex, supervisor, cfg, on_event).run()