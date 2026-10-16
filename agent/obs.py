"""Core types: what an observation is, what an action is, what an environment must do."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Action:
    """A single game action. ``name`` is from the env's action space."""

    name: str
    source: str = "scripted"          # scripted | laya | jev | luna | manual | none
    meta: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return self.name


@dataclass
class Obs:
    """One observed frame. ``prose`` is the ONLY thing decision models should see."""

    t: float = field(default_factory=time.time)
    frame: Optional[bytes] = None      # PNG bytes of the game canvas, if any
    features: dict = field(default_factory=dict)
    prose: str = ""
    score: int = 0
    done: bool = False
    info: dict = field(default_factory=dict)


class GameEnv:
    """The seam every game adapter implements. One adapter per game source."""

    name = "env"
    #: keys the policy may choose from, in a stable order
    action_space: list[str] = ["hold"]

    def reset(self) -> Obs:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, action: Action) -> Obs:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass

    #: optional: current screen as PNG bytes, for the vision tier and the dashboard
    def frame(self) -> Optional[bytes]:
        return None

    #: optional: MAME work-RAM reader, when the adapter can reach the emulator heap
    def read_ram(self, addr: int, length: int = 1) -> Optional[bytes]:
        return None