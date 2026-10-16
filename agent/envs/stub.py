"""The dependency-free stub game: Galaga-shaped enough to exercise the whole loop.

It has the two things the real game has that matter to a decision policy: a player
ship on one axis, and enemies that are either lined up with you (shoot) or diving at
you (dodge). Deterministic under a seed, so eval runs are reproducible.
"""
from __future__ import annotations

import random

from ..obs import Action, GameEnv, Obs
from ..perceive import prose_from_features, NEUTRAL

ACTIONS = ["hold", "left", "right", "fire", "left_fire", "right_fire"]


class StubGalaga(GameEnv):
    name = "stub"
    action_space = ACTIONS

    def __init__(self, seed: int = 0, width: int = 660, max_steps: int = 600) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.width = width
        self.max_steps = max_steps
        self.ship_x = width // 2
        self.enemies: list[dict] = []
        self.bullets: list[dict] = []
        self.steps = 0
        self.ship_speed = 14
        self.score = 0
        self.lives = 3
        self._spawn_wave()

    def _spawn_wave(self, n: int = 5) -> None:
        for _ in range(n):
            self.enemies.append({
                "x": self.rng.randrange(40, self.width - 40),
                "y": self.rng.randrange(120, 300),
                "vx": self.rng.choice([-1, 0, 1]) * self.rng.uniform(0.2, 1.2),
                "diving": self.rng.random() < 0.35,
                "vy": self.rng.uniform(0.2, 1.0),
            })

    def reset(self) -> Obs:
        self.rng = random.Random(self.seed)
        self.ship_x = self.width // 2
        self.enemies, self.bullets = [], []
        self.steps, self.score, self.lives = 0, 0, 3
        self._spawn_wave()
        return self._obs()

    def step(self, action: Action) -> Obs:
        self.steps += 1
        name = action.name
        if name.startswith("left"):
            self.ship_x = max(24, self.ship_x - self.ship_speed)
        elif name.startswith("right"):
            self.ship_x = min(self.width - 24, self.ship_x + self.ship_speed)

        for e in self.enemies:
            e["x"] += e["vx"]
            if e["x"] < 24 or e["x"] > self.width - 24:
                e["vx"] *= -1
            if e["diving"]:
                e["y"] += e["vy"]
        # diving enemies bounce back up when they reach the ship line, or kill you
        for e in self.enemies:
            if e["diving"] and e["y"] > 420:
                if abs(e["x"] - self.ship_x) < 26:
                    self.lives -= 1
                    e["y"], e["diving"] = 200, False
                    if self.lives <= 0:
                        self.steps = self.max_steps
                elif e["y"] > 460:
                    e["y"], e["diving"] = 180, self.rng.random() < 0.3

        if name.endswith("fire") or name == "fire":
            self.bullets.append({"x": self.ship_x, "y": 400})
        for b in self.bullets:
            b["y"] -= 26
        live_bullets = []
        for b in self.bullets:
            hit = None
            for i, e in enumerate(self.enemies):
                if abs(e["x"] - b["x"]) < 20 and abs(e["y"] - b["y"]) < 26:
                    hit = i
                    break
            if hit is not None:
                self.enemies.pop(hit)
                self.score += 100
            elif b["y"] > 0:
                live_bullets.append(b)
        self.bullets = live_bullets
        if not self.enemies:
            self._spawn_wave(n=min(6, 3 + self.score // 300))

        return self._obs()

    def _obs(self) -> Obs:
        f = self.features()
        return Obs(frame=None, features=f, prose=prose_from_features(f),
                   score=self.score, done=self.steps >= self.max_steps,
                   info={"lives": self.lives, "enemies": len(self.enemies)})

    def features(self) -> dict:
        nearest, nd = None, 1e9
        for e in self.enemies:
            d = abs(e["x"] - self.ship_x)
            if d < nd:
                nearest, nd = e, d
        diving = [e for e in self.enemies if e["diving"] and e["y"] > 300]
        threat = min((abs(e["x"] - self.ship_x) for e in diving), default=None)
        nearest_dive = min(diving, key=lambda e: abs(e["x"] - self.ship_x), default=None)
        return {
            "ship_x": self.ship_x,
            "ship_frac": self.ship_x / self.width,
            "enemy_x": nearest["x"] if nearest else None,
            "dx": (nearest["x"] - self.ship_x) if nearest else None,
            "dx_frac": ((nearest["x"] - self.ship_x) / self.width) if nearest else None,
            "aligned": bool(nearest and abs(nearest["x"] - self.ship_x) < 24),
            "diving_near": threat is not None and threat < 90,
            "diving_dist": threat,
            "threat_dx_frac": ((nearest_dive["x"] - self.ship_x) / self.width) if nearest_dive else None,
            "enemies": len(self.enemies),
            "incoming": len([b for b in self.bullets if b["y"] < 200]),
            "lives": self.lives,
        }

    # a cheap text screen so the dashboard has something to draw without a real emulator
    def ascii_frame(self, width: int = 60, height: int = 22) -> str:
        grid = [[" "] * width for _ in range(height)]
        sx = int(self.ship_x / self.width * (width - 1))
        grid[height - 1][sx] = "A"
        for e in self.enemies:
            ex = int(e["x"] / self.width * (width - 1))
            ey = int(e["y"] / 480 * (height - 1))
            if 0 <= ey < height - 1:
                grid[ey][ex] = "W"
        for b in self.bullets:
            bx = int(b["x"] / self.width * (width - 1))
            by = int(b["y"] / 480 * (height - 1))
            if 0 <= by < height - 1:
                grid[by][bx] = "|"
        return "\n".join("".join(r) for r in grid)