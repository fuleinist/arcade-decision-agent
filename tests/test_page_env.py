"""Integration test for the live-page adapter. Skipped when Playwright is absent.

This is the test that answers "can we even reach a browser arcade game": it boots
Chromium, loads classicgamesarcade's Galaga, confirms the Emularity/MAME module is
loaded, reads its wasm heap, round-trips a MAME save state, and presses a key.

Scope note: it does NOT assert that the captured pixels are the game. Measured
2026-10-16, a `#canvas` element screenshot on this page returns the page background
(dominant colour 222,222,222) rather than Galaga's black starfield, in every browser
configuration tried - so pixel trust is an open milestone (PLAN.md M4), not a passing
test. What is asserted here is the part that does work: the emulator, its RAM, and its
save-state hooks.
"""
from __future__ import annotations

import pytest

pytest.importorskip("playwright")

from agent.envs.emularity import EmularityEnv  # noqa: E402
from agent.obs import Action  # noqa: E402
from agent.perceive import frame_features, prose_from_features  # noqa: E402

URL = "https://www.classicgamesarcade.com/game/21637/galaga.html"


@pytest.mark.net
def test_live_page_emulator_is_reachable_and_frame_is_readable():
    env = EmularityEnv(url=URL, headed=False, auto_start=False, cooldown_s=0.1)
    try:
        obs = env.reset()
    except Exception as exc:  # site down, network blocked, browser missing
        pytest.skip(f"could not reach the live page: {type(exc).__name__}: {exc}")
    try:
        assert obs.frame and len(obs.frame) > 1000, "no frame captured from the canvas"
        feats = frame_features(obs.frame)
        assert feats["pixels"][0] > 100 and feats["pixels"][1] > 100
        assert obs.prose.startswith(("The player ship", "There is no enemy"))
        # one action must not raise, and must leave the emulator alive
        obs2 = env.step(Action("left"))
        assert obs2.frame and len(obs2.frame) > 1000
        assert prose_from_features(frame_features(obs2.frame))
        # the MAME module exposes the RAM and save-state hooks we plan to use
        assert env.read_ram(0, 16) is not None, "HEAPU8 not reachable"
        assert env.mame_save("pytest_probe") is True, "save state hook unavailable"
        assert env.mame_load("pytest_probe") is True, "load state hook unavailable"
    finally:
        env.close()