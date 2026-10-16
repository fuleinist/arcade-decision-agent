"""Policy interface + the deterministic baseline.

A policy sees an Obs and returns an Action. That is the whole contract, which is what
lets a 4 ms local encoder, a $0.042/M decision API and a scripted heuristic sit behind
the same loop and be compared on the same episodes.
"""
from __future__ import annotations

from ..obs import Action, Obs


class Policy:
    name = "policy"
    #: True when the policy can consume the frame as well as the prose state
    vision = False
    #: False when credentials/binary/model are missing - the loop skips it, loudly
    available = True
    detail = ""

    def decide(self, obs: Obs, actions: list[str]) -> Action:  # pragma: no cover - interface
        raise NotImplementedError

    def decide_mode(self, obs: Obs) -> tuple[str, dict]:
        """Supervisor tier: which mode should the reflex run in?

        The free default is a one-line read of the features; model-backed policies ask
        the engine instead, so the tiering is measurable rather than decorative.
        """
        mode = "dodge" if obs.features.get("diving_near") else "engage"
        return mode, {"source": "features", "mode": mode}

    def health(self) -> dict:
        return {"policy": self.name, "available": self.available, "vision": self.vision,
                "detail": self.detail}

    def close(self) -> None:
        pass


def combine(steer: str, fire: bool) -> str:
    """Map (steer, shoot) onto the env's action names."""
    if steer == "hold":
        return "fire" if fire else "hold"
    if fire:
        return f"{steer}_fire"
    return steer