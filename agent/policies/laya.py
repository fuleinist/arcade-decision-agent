"""Local Laya (ggmlc) as the reflex tier: one typed question per frame, ~10-30 ms.

Measured on this machine, 2026-10-16, `layabin/laya.exe` + `laya_multilingual_q8_0.gguf`:
one-sentence prose state separates the two directions by +0.526 (0.833 vs 0.307),
the same facts as a JSON dict by -0.199 and as bare numbers by -0.088. The JSON path is
kept only as a switch, so the ablation in eval.py can re-measure that claim rather than
trusting this comment.

The daemon is one long-lived child process speaking newline-delimited JSON on stdio.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Optional

from ..obs import Action, Obs
from ..perceive import threat_prose
from .base import Policy, combine

DEFAULT_EXE = r"G:/dev/AI/laya/layabin/laya.exe"
DEFAULT_MODEL = r"G:/dev/AI/laya/laya_multilingual_q8_0.gguf"


class LayaPolicy(Policy):
    name = "laya"

    def __init__(self, exe: Optional[str] = None, model: Optional[str] = None,
                 threshold: float = 0.5, state_format: str = "prose",
                 timeout_ms: int = 2000, device: str = "auto",
                 questions_design: str = "choice3", fire_mode: str = "aligned") -> None:
        self.exe = exe or os.environ.get("LAYA_EXE", DEFAULT_EXE)
        self.model = model or os.environ.get("LAYA_MODEL", DEFAULT_MODEL)
        self.threshold = threshold
        self.state_format = state_format
        self.timeout_ms = timeout_ms
        self.device = device
        #: "choice3" = one 3-option steering question (harness fires on alignment),
        #: "choice" = one 4-option question including fire, "noul" = three yes/no questions.
        #: Measured against a scripted baseline on 120 real stub states, agreement was
        #: 0.758 (choice/4) vs 0.392 (noul/3) vs 0.283 (choice/6); in-game the 4-option
        #: form never chose "fire", so steering is asked alone and firing is deterministic.
        self.questions_design = questions_design
        #: "aligned" = fire when the extractor says the ship is lined up (mechanical act),
        #: "model" = let the engine answer the fire question (the pure-model ablation).
        self.fire_mode = fire_mode
        self._last_steer = "hold"
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._rid = 0
        self.stats = {"calls": 0, "total_ms": 0.0, "errors": 0, "last_ms": 0.0}
        if not os.path.exists(self.exe):
            self.available = False
            self.detail = f"laya.exe not found at {self.exe}"
        elif not os.path.exists(self.model):
            self.available = False
            self.detail = f"GGUF not found at {self.model}"

    # ------------------------------------------------------------------ daemon ---
    def _ensure(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [self.exe, "daemon", self.model, "--device", self.device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, cwd=os.path.dirname(self.model) or ".",
        )
        # first line is the ready banner; the GGUF load takes seconds on a cold cache
        deadline = time.time() + 120
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("laya daemon exited during load")
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("status") == "ready":
                return
        raise RuntimeError("laya daemon did not report ready in 120 s")

    def _ask(self, state: str, questions: dict) -> dict:
        with self._lock:
            self._ensure()
            self._rid += 1
            rid = self._rid
            t0 = time.time()
            self._proc.stdin.write(json.dumps({"id": rid, "state": state, "questions": questions}) + "\n")
            self._proc.stdin.flush()
            while True:
                if (time.time() - t0) * 1000 > self.timeout_ms:
                    self.stats["errors"] += 1
                    raise TimeoutError(f"laya daemon exceeded {self.timeout_ms} ms")
                line = self._proc.stdout.readline()
                if not line:
                    self.stats["errors"] += 1
                    raise RuntimeError("laya daemon closed stdout")
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("id") not in (None, rid):
                    continue
                if "answers" in obj:
                    ms = (time.time() - t0) * 1000
                    self.stats["calls"] += 1
                    self.stats["total_ms"] += ms
                    self.stats["last_ms"] = ms
                    obj["_ms"] = ms
                    return obj

    # ----------------------------------------------------------------- decision ---
    def decide_mode(self, obs: Obs) -> tuple[str, dict]:
        """Ask the engine whether the ship is in trouble, instead of reading a flag."""
        state = threat_prose(obs.features) + " " + obs.prose
        try:
            out = self._ask(state, {"dodge": {"type": "noul", "instructions":
                   "The player ship is in immediate danger from a diving enemy and must dodge now."}})
        except Exception as exc:
            return "engage", {"error": f"{type(exc).__name__}: {exc}"}
        p = float(out.get("answers", {}).get("dodge", {}).get("noul", 0.0))
        return ("dodge" if p >= self.threshold else "engage",
                {"source": self.name, "p": p, "latency_ms": round(out.get("_ms", 0.0), 1)})

    def _questions(self, mode: str) -> dict:
        if self.questions_design in ("choice", "choice3"):
            if self.questions_design == "choice3":
                if mode == "dodge":
                    return {"action": {"type": "choice",
                                       "instructions": "Which way should the player ship move next to avoid the diving enemy",
                                       "criteria": {"left": "move left", "right": "move right",
                                                    "hold": "stay where it is"}}}
                return {"action": {"type": "choice",
                                   "instructions": "Which way should the player ship move next to line up with the enemy",
                                   "criteria": {"left": "move left, toward the enemy",
                                                "right": "move right, toward the enemy",
                                                "hold": "stay where it is"}}}
            if mode == "dodge":
                return {"action": {"type": "choice",
                                   "instructions": "Which way should the player ship move next to avoid the diving enemy",
                                   "criteria": {"left": "move left", "right": "move right",
                                                "hold": "stay where it is"}}}
            return {"action": {"type": "choice",
                               "instructions": "Which move should the player ship make next",
                               "criteria": {"left": "move left, toward the enemy",
                                            "right": "move right, toward the enemy",
                                            "hold": "stay where it is",
                                            "fire": "shoot, the ship is lined up"}}}
        if mode == "dodge":
            return {
                "move_left": {"type": "noul", "instructions":
                              "The player ship must move left now to get out of the way."},
                "move_right": {"type": "noul", "instructions":
                               "The player ship must move right now to get out of the way."},
                "fire": {"type": "noul", "instructions":
                         "The player ship is lined up with an enemy and should fire now."},
            }
        return {
            "move_left": {"type": "noul", "instructions":
                          "The player ship is right of the enemy and must move left now."},
            "move_right": {"type": "noul", "instructions":
                           "The player ship is left of the enemy and must move right now."},
            "fire": {"type": "noul", "instructions":
                     "The player ship is lined up with an enemy and should fire now."},
        }

    def _state(self, obs: Obs, mode: str) -> str:
        if self.state_format == "json":
            return json.dumps(obs.features)
        if mode == "dodge":
            return threat_prose(obs.features)
        return obs.prose

    def _fire(self, obs: Obs, model_fire: bool) -> bool:
        """Firing is a mechanical act with a deterministic read; the model is judged on
        steering. `fire_mode="model"` keeps the engine's own answer, for the ablation."""
        if self.fire_mode == "model":
            return model_fire
        return bool(obs.features.get("aligned"))

    def decide(self, obs: Obs, actions: list[str]) -> Action:
        if not self.available:
            return Action("hold", source=self.name, meta={"error": self.detail, "unavailable": True})
        mode = obs.info.get("mode", "engage")
        state = self._state(obs, mode)
        try:
            out = self._ask(state, self._questions(mode))
        except Exception as exc:  # a stalled engine must not stop the game loop
            return Action("hold", source=self.name,
                          meta={"error": f"{type(exc).__name__}: {exc}"})
        answers = out.get("answers", {})
        t = self.threshold
        if self.questions_design in ("choice", "choice3"):
            a = answers.get("action", {}) or {}
            dist = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
            choice = a.get("choice") or (max(dist, key=dist.get) if dist else "hold")
            conf = dist.get(choice, float(a.get("confidence") or 0.0))
            if conf < t:
                choice = "hold"
            model_fire = choice == "fire"
            if model_fire:
                steer = self._last_steer
            else:
                steer = choice
                self._last_steer = choice if choice in ("left", "right") else "hold"
            fire = self._fire(obs, model_fire)
            return Action(combine(steer, fire), source=self.name,
                          meta={"mode": mode, "choice": choice, "probs": dist, "threshold": t,
                                "fire": fire, "steer": steer,
                                "latency_ms": round(out.get("_ms", 0.0), 1), "state": state})

        p = {k: float(answers.get(k, {}).get("noul", 0.0)) for k in ("move_left", "move_right", "fire")}
        left, right, fire = p["move_left"], p["move_right"], p["fire"]

        if max(p.values()) < t:
            steer = "hold"
        elif fuse := ((left >= t) and (right >= t)):
            steer = "left" if left > right else "right"
        else:
            steer = max((("left", left), ("right", right)), key=lambda kv: kv[1])[0] if max(left, right) >= t else "hold"
        action = combine(steer, fire >= t)
        return Action(action, source=self.name,
                      meta={"mode": mode, "probs": p, "threshold": t,
                            "latency_ms": round(out.get("_ms", 0.0), 1),
                            "fused": bool(fuse if max(p.values()) >= t else False),
                            "state": state})

    def health(self) -> dict:
        h = super().health()
        h.update({"exe": self.exe, "model": self.model, "threshold": self.threshold,
                  "state_format": self.state_format, "stats": self.stats})
        return h

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
            except Exception:
                pass