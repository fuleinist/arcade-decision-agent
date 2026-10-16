"""A scripted baseline: no model, no network, no cost.

Its job is to be the thing the models have to beat, and to keep the harness honest:
if a decision model cannot out-score a 20-line heuristic, the harness is fine and
the model is not being used well.
"""
from __future__ import annotations

from ..obs import Action, Obs
from .base import Policy, combine


class ScriptedPolicy(Policy):
    name = "scripted"

    def __init__(self, dead_zone: float = 0.03) -> None:
        self.dead_zone = dead_zone

    def decide(self, obs: Obs, actions: list[str]) -> Action:
        f = obs.features
        mode = obs.info.get("mode", "engage")
        dx = f.get("threat_dx_frac") if mode == "dodge" else f.get("dx_frac")
        if dx is None:
            dx = f.get("dx_frac")

        if dx is None:
            steer = "hold"
        elif dx > self.dead_zone:
            steer = "right"
        elif dx < -self.dead_zone:
            steer = "left"
        else:
            steer = "hold"

        fire = bool(f.get("aligned")) or (dx is not None and abs(dx) < 0.05)
        action = combine(steer, fire)
        return Action(action, source=self.name,
                      meta={"mode": mode, "dx_frac": dx, "steer": steer, "fire": fire})