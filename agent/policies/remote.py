"""Remote decision models behind one adapter: TypeSafe Jev and OpenAI Decisions.

Both are typed-question APIs (choice / noul / score) and neither is a chat model, so
neither can be pointed at /v1/chat/completions. Three transports, one class family:

  jev    POST {base}/v1/systemone      state + questions map           (TypeSafe, LiteLLM)
  luna   POST {base}/decisions         input messages + questions array (OpenAI)
  router decisions gateway             jev shape, model-specific        (OpenRouter alpha)

Pricing to keep in mind while reading the eval output (verified 2026-10-16):
Jev $0.042/M input tokens, OpenAI gpt-6-luna $0.10/M, output tokens free on both.
A per-frame policy at 30 Hz is ~30 requests/second, so the budget is a design input,
not an afterthought: run these as supervisors, not as reflexes.
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Optional

from ..obs import Action, Obs
from ..perceive import threat_prose
from .base import Policy, combine

try:  # httpx is a soft dependency: the loop must import without it
    import httpx
except Exception:  # pragma: no cover
    httpx = None


def _lookup(d: dict, *names):
    """Tolerant field lookup: beta decision APIs have renamed fields between revisions."""
    for n in names:
        if isinstance(d, dict) and n in d and d[n] is not None:
            return d[n]
    return None


class _RemotePolicy(Policy):
    transport = "jev"
    vision = False
    env_key = "TYPESAFE_API_KEY"
    default_base = "https://api.typesafe.ai/v1"
    default_model = "jev-latest"
    price_per_mtok = 0.042

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, threshold: float = 0.5,
                 timeout_s: float = 12.0, max_questions: int = 4) -> None:
        self.api_key = api_key or os.environ.get(self.env_key) or os.environ.get("DECISIONS_API_KEY")
        self.base_url = (base_url or os.environ.get("DECISIONS_BASE_URL") or self.default_base).rstrip("/")
        self.model = model or os.environ.get("DECISIONS_MODEL") or self.default_model
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.max_questions = max_questions
        self.stats = {"calls": 0, "total_ms": 0.0, "errors": 0, "input_tokens": 0, "cost_usd": 0.0}
        if httpx is None:
            self.available, self.detail = False, "httpx not installed"
        elif not self.api_key:
            self.available = False
            self.detail = f"no API key (set {self.env_key} or DECISIONS_API_KEY)"
        else:
            self.detail = f"{self.transport}:{self.model} via {self.base_url}"

    # ------------------------------------------------------------------ payload ---
    def _questions_systemone(self, mode: str) -> dict:
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
            "steer": {"type": "choice", "instructions":
                      "Which way should the player ship move next",
                      "criteria": {"left": "move left, toward the enemy",
                                   "right": "move right, toward the enemy",
                                   "hold": "stay where it is"}},
            "fire": {"type": "noul", "instructions":
                     "The player ship is lined up with an enemy and should fire now."},
        }

    def _questions_openai(self, mode: str) -> list:
        if mode == "dodge":
            return [
                {"type": "choice", "name": "steer",
                 "instructions": "Which way should the player ship move next to avoid the diving enemy",
                 "choices": [{"value": "left", "description": "move left"},
                             {"value": "right", "description": "move right"},
                             {"value": "hold", "description": "stay in place"}]},
                {"type": "predicate", "name": "fire",
                 "instructions": "The player ship is lined up with an enemy and should fire now."},
            ]
        return [
            {"type": "choice", "name": "steer",
             "instructions": "Which way should the player ship move next to line up with the nearest enemy",
             "choices": [{"value": "left", "description": "move left"},
                         {"value": "right", "description": "move right"},
                         {"value": "hold", "description": "already lined up, stay"}]},
            {"type": "predicate", "name": "fire",
             "instructions": "The player ship is lined up with an enemy and should fire now."},
        ]

    def _state(self, obs: Obs, mode: str) -> str:
        if mode == "dodge":
            return threat_prose(obs.features) + " " + obs.prose
        return obs.prose

    def _post(self, url: str, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, headers=headers, json=body)
            r.raise_for_status()
            return r.json()

    # ----------------------------------------------------------------- decision ---
    def decide_mode(self, obs: Obs) -> tuple[str, dict]:
        state = self._state(obs, "dodge")
        t0 = time.time()
        try:
            if self.transport == "openai":
                body = {"model": self.model,
                        "input": [{"role": "user", "content": [{"type": "input_text", "text": state}]}],
                        "questions": [{"type": "predicate", "name": "dodge",
                                       "instructions": "The player ship is in immediate danger from a diving enemy and must dodge now."}]}
                raw = self._post(f"{self.base_url}/decisions", body)
                ans = raw.get("answers") or raw.get("result", {}).get("answers") or []
                p = 0.0
                if isinstance(ans, list):
                    for item in ans:
                        p = float(_lookup(item, "probability", "probability_true", "noul", "score") or 0.0)
                elif isinstance(ans, dict):
                    p = float(_lookup(ans.get("dodge", {}), "noul", "probability") or 0.0)
            else:
                raw = self._post(f"{self.base_url}/systemone",
                                 {"model": self.model, "state": state,
                                  "questions": {"dodge": {"type": "noul", "instructions":
                                                "The player ship is in immediate danger from a diving enemy and must dodge now."}}})
                ans = raw.get("answers", raw.get("result", {}).get("answers", {})) or {}
                p = float(_lookup(ans.get("dodge", {}), "noul", "probability") or 0.0)
        except Exception as exc:
            self.stats["errors"] += 1
            return "engage", {"error": f"{type(exc).__name__}: {exc}"}
        ms = (time.time() - t0) * 1000
        self.stats["calls"] += 1
        self.stats["total_ms"] += ms
        return ("dodge" if p >= self.threshold else "engage",
                {"source": self.name, "p": p, "latency_ms": round(ms, 1)})

    def decide(self, obs: Obs, actions: list[str]) -> Action:
        if not self.available:
            return Action("hold", source=self.name,
                          meta={"error": self.detail, "unavailable": True})
        mode = obs.info.get("mode", "engage")
        state = self._state(obs, mode)
        t0 = time.time()
        try:
            raw = self._call(state, obs, mode)
        except Exception as exc:
            self.stats["errors"] += 1
            return Action("hold", source=self.name,
                          meta={"error": f"{type(exc).__name__}: {exc}"})
        ms = (time.time() - t0) * 1000
        self.stats["calls"] += 1
        self.stats["total_ms"] += ms
        self.stats["last_ms"] = ms
        steer, fire, probs, usage = self._parse(raw)
        self.stats["input_tokens"] += usage.get("input_tokens", 0)
        self.stats["cost_usd"] += usage.get("input_tokens", 0) / 1e6 * self.price_per_mtok
        if max(probs.values(), default=0.0) < self.threshold and steer not in ("left", "right"):
            steer = "hold"
        action = combine(steer if steer in ("left", "right") else "hold", fire)
        return Action(action, source=self.name,
                      meta={"mode": mode, "probs": probs, "threshold": self.threshold,
                            "latency_ms": round(ms, 1), "state": state, "usage": usage})

    def _call(self, state: str, obs: Obs, mode: str) -> dict:  # pragma: no cover - subclass
        raise NotImplementedError

    def _parse(self, raw: dict) -> tuple[str, bool, dict, dict]:
        raise NotImplementedError

    def health(self) -> dict:
        h = super().health()
        h.update({"transport": self.transport, "base_url": self.base_url, "model": self.model,
                  "threshold": self.threshold, "stats": self.stats})
        return h


class JevPolicy(_RemotePolicy):
    """TypeSafe Jev / System One (text only - it cannot see the screen)."""

    name = "jev"
    transport = "jev"
    vision = False
    env_key = "TYPESAFE_API_KEY"
    default_base = "https://api.typesafe.ai/v1"
    default_model = "jev-latest"
    price_per_mtok = 0.042

    def _call(self, state: str, obs: Obs, mode: str) -> dict:
        body = {"model": self.model, "state": state, "questions": self._questions_systemone(mode)}
        return self._post(f"{self.base_url}/systemone", body)

    def _parse(self, raw: dict) -> tuple[str, bool, dict, dict]:
        ans = raw.get("answers", raw.get("result", {}).get("answers", {})) or {}
        probs: dict[str, float] = {}
        steer = "hold"
        fire = False
        if "steer" in ans:
            a = ans["steer"]
            steer = str(_lookup(a, "choice", "value", "selected") or "hold")
            dist = _lookup(a, "probabilities") or {}
            probs["steer"] = float(max(dist.values())) if dist else float(_lookup(a, "confidence") or 0.0)
        if "fire" in ans:
            fire = float(_lookup(ans["fire"], "noul", "probability", "probability_true") or 0.0) >= self.threshold
        if not any(k in ans for k in ("steer", "fire")):
            # dodge mode: two noul directions
            left = float(_lookup(ans.get("move_left", {}), "noul", "probability") or 0.0)
            right = float(_lookup(ans.get("move_right", {}), "noul", "probability") or 0.0)
            fire_v = float(_lookup(ans.get("fire", {}), "noul", "probability") or 0.0)
            probs.update({"move_left": left, "move_right": right, "fire": fire_v})
            if max(left, right) >= self.threshold:
                steer = "left" if left > right else "right"
            fire = fire_v >= self.threshold
        usage = _lookup(raw, "usage") or {}
        return steer, fire, probs, usage


class LunaPolicy(_RemotePolicy):
    """OpenAI Decisions (gpt-6-luna): the only tier here that can SEE the frame."""

    name = "luna"
    transport = "openai"
    vision = True
    env_key = "OPENAI_API_KEY"
    default_base = "https://api.openai.com/v1"
    default_model = "gpt-6-luna"
    price_per_mtok = 0.10

    def __init__(self, send_image: bool = True, image_detail: str = "original", **kw) -> None:
        super().__init__(**kw)
        self.send_image = send_image
        self.image_detail = image_detail

    def _call(self, state: str, obs: Obs, mode: str) -> dict:
        content: list[dict] = [{"type": "input_text", "text": state}]
        if self.send_image and obs.frame:
            b64 = base64.b64encode(obs.frame).decode()
            content.append({"type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}",
                            "detail": self.image_detail})
        body = {"model": self.model,
                "input": [{"role": "user", "content": content}],
                "questions": self._questions_openai(mode)}
        url = f"{self.base_url}/decisions"
        return self._post(url, body)

    def _parse(self, raw: dict) -> tuple[str, bool, dict, dict]:
        ans = raw.get("answers") or raw.get("result", {}).get("answers") or []
        probs: dict[str, float] = {}
        steer, fire = "hold", False
        if isinstance(ans, dict):  # gateway may return the systemone shape
            for k, v in ans.items():
                if "steer" in k or "action" in k:
                    steer = str(_lookup(v, "choice", "value", "selected") or steer)
                if "fire" in k:
                    fire = float(_lookup(v, "noul", "probability", "probability_true") or 0.0) >= self.threshold
        else:
            for item in ans:
                name = str(item.get("name", "")).lower()
                if "steer" in name or "action" in name:
                    v = _lookup(item, "choice", "value", "selected")
                    if isinstance(v, dict):
                        v = _lookup(v, "value", "label")
                    steer = str(v if v is not None else steer)
                elif "fire" in name:
                    p = float(_lookup(item, "probability", "probability_true", "noul", "score") or 0.0)
                    probs["fire"] = p
                    fire = p >= self.threshold
        usage = _lookup(raw, "usage") or {}
        return steer, fire, probs, usage


TRANSPORTS = {"jev": JevPolicy, "luna": LunaPolicy}


def make_policy(spec: str, **kw) -> Policy:
    """`--policy jev`, `--policy luna`, `--policy luna:openrouter` ..."""
    name, _, variant = spec.partition(":")
    if name in ("none",):
        from .base import Policy as _P

        class _Null(_P):
            name = "none"
            available = False
            detail = "no policy"

            def decide(self, obs, actions):
                from ..obs import Action
                return Action("hold", source="none")

        return _Null()
    if name == "router":
        name, variant = "jev", variant or "router"
    if variant == "openrouter":
        kw.setdefault("base_url", "https://openrouter.ai/api/alpha/decisions")
        kw.setdefault("api_key", os.environ.get("OPENROUTER_API_KEY"))
        kw.setdefault("model", "typesafe/jev-1.13" if name == "jev" else "openai/gpt-6-luna-decisions")
    elif variant == "litellm":
        kw.setdefault("base_url", os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1"))
        kw.setdefault("api_key", os.environ.get("LITELLM_API_KEY", "sk-local"))
    return TRANSPORTS[name](**kw)


def json_dump_meta(meta: dict) -> str:
    return json.dumps(meta, default=str)