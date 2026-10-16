"""Tests that would fail if the loop, the prose rule, or a policy adapter broke."""
from __future__ import annotations

import os
import json

import pytest

from agent.envs import make_env
from agent.envs.stub import StubGalaga
from agent.loop import ControlLoop, LoopConfig
from agent.obs import Action
from agent.perceive import bucket_dx, prose_from_features, threat_prose
from agent.policies import build
from agent.policies.laya import LayaPolicy
from agent.policies.remote import JevPolicy, LunaPolicy


# ------------------------------------------------------------------ prose rule ---
def test_buckets_are_words_not_numbers():
    assert bucket_dx(-0.4) == "far to the left of"
    assert bucket_dx(0.0) == "lined up with"
    assert bucket_dx(0.4) == "far to the right of"
    assert bucket_dx(None) == "neutral"


def test_prose_never_serialises_as_json():
    f = {"enemy_x": 600, "dx": 560, "dx_frac": 0.85, "aligned": False}
    s = prose_from_features(f)
    assert s.startswith("The player ship is")
    assert "{" not in s and ":" not in s and "0.85" not in s


def test_prose_reports_alignment_as_its_own_sentence():
    s = prose_from_features({"enemy_x": 300, "dx": 4, "dx_frac": 0.006, "aligned": True})
    assert s == "The player ship is lined up with the nearest enemy."


def test_threat_prose_is_separate_and_answerable():
    assert "diving straight at" in threat_prose({"diving_near": True, "diving_dist": 10})
    assert "no enemy is diving" in threat_prose({"diving_near": False, "diving_dist": None}).lower()


# ------------------------------------------------------------------- stub env ---
def test_stub_is_deterministic_for_a_seed():
    a, b = StubGalaga(seed=7), StubGalaga(seed=7)
    for _ in range(40):
        a.step(Action("hold"))
        b.step(Action("hold"))
    assert a.features() == b.features()


def test_scripted_policy_scores_and_beats_doing_nothing():
    env = make_env("stub", seed=1)
    loop = ControlLoop(env, build("scripted"), cfg=LoopConfig(max_steps=400))
    loop.run()
    played = loop.state.score
    idle_env = make_env("stub", seed=1)
    idle = ControlLoop(idle_env, build("none"), cfg=LoopConfig(max_steps=400))
    idle.run()
    assert played > idle.state.score


def test_loop_survives_a_policy_that_always_errors():
    class Broken(build("scripted").__class__):
        name = "broken"

        def decide(self, obs, actions):
            return Action("hold", source="laya", meta={"error": "boom"})

    loop = ControlLoop(make_env("stub", seed=2), Broken(), cfg=LoopConfig(max_steps=60))
    loop.run()
    assert loop.state.errors == loop.state.decisions > 0
    assert loop.state.stop_reason == "max_steps"


def test_manual_action_overrides_the_policy():
    events = []
    loop = ControlLoop(make_env("stub", seed=3), build("scripted"), cfg=LoopConfig(max_steps=10),
                       on_event=events.append)
    loop.push_manual("left_fire")
    loop.run()
    first = next(e for e in events if e["type"] == "decision")
    assert first["action"] == "left_fire" and first["source"] == "manual"
    assert sum(1 for e in events if e["type"] == "decision" and e["source"] == "manual") == 1


def test_supervisor_switches_mode_on_threat():
    env = make_env("stub", seed=5)
    loop = ControlLoop(env, build("scripted"), build("scripted"),
                       cfg=LoopConfig(max_steps=200, supervisor_every=5))
    loop.run()
    assert loop.state.mode in ("engage", "dodge")


# --------------------------------------------------------------- adapters -------
def test_laya_policy_reports_unavailable_without_binary_and_model():
    p = LayaPolicy(exe="no-such-dir/laya.exe", model="no-such-dir/model.gguf")
    assert not p.available
    assert "no Laya model" in p.detail
    a = p.decide(_dummy_obs(), ["hold"])
    assert a.name == "hold" and a.meta.get("unavailable")


def test_laya_policy_takes_its_paths_from_the_environment(monkeypatch, tmp_path):
    fake_exe = tmp_path / "laya.exe"
    fake_model = tmp_path / "model.gguf"
    fake_exe.write_bytes(b"stub")
    fake_model.write_bytes(b"stub")
    monkeypatch.setenv("LAYA_EXE", str(fake_exe))
    monkeypatch.setenv("LAYA_MODEL", str(fake_model))
    p = LayaPolicy()
    assert str(fake_exe) == p.exe and str(fake_model) == p.model
    assert p.available, p.detail
    # the reported health carries file names, not the operator's directories
    h = p.health()
    assert h["exe"] == "laya.exe" and h["model"] == "model.gguf"
    assert str(tmp_path) not in json.dumps(h)


def test_laya_policy_without_configuration_says_so(monkeypatch):
    monkeypatch.delenv("LAYA_EXE", raising=False)
    monkeypatch.delenv("LAYA_MODEL", raising=False)
    p = LayaPolicy()
    assert not p.available
    assert "LAYA_MODEL" in p.detail


def test_remote_policies_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("DECISIONS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for cls, spec in ((JevPolicy, "jev"), (LunaPolicy, "luna")):
        p = cls()
        assert not p.available, spec
        assert p.decide(_dummy_obs(), ["hold"]).meta.get("unavailable")
    assert LunaPolicy.vision is True and JevPolicy.vision is False


def test_luna_payload_carries_the_frame(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    p = LunaPolicy()
    assert p.available
    obs = _dummy_obs()
    obs.frame = b"\x89PNG\r\n\x1a\n0000"
    sent = {}

    def fake_post(url, body):
        sent["url"], sent["body"] = url, body
        return {"answers": [{"type": "choice", "name": "steer", "choice": "left"},
                            {"type": "predicate", "name": "fire", "probability": 0.9}],
                "usage": {"input_tokens": 1200}}

    p._post = fake_post
    a = p.decide(obs, ["hold"])
    assert sent["url"].endswith("/decisions")
    kinds = [c["type"] for c in sent["body"]["input"][0]["content"]]
    assert kinds == ["input_text", "input_image"]
    assert a.name == "left_fire"
    assert p.stats["input_tokens"] == 1200


def test_jev_parses_the_systemone_shape(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    p = JevPolicy()
    p._post = lambda url, body: {"answers": {"steer": {"type": "choice", "choice": "right",
                                                       "probabilities": {"right": 0.7, "left": 0.3}},
                                             "fire": {"type": "noul", "noul": 0.05}},
                                 "usage": {"input_tokens": 400}}
    a = p.decide(_dummy_obs(), ["hold"])
    assert a.name == "right"
    assert a.meta["usage"]["input_tokens"] == 400
    assert abs(p.stats["cost_usd"] - 400 / 1e6 * 0.042) < 1e-9


# ------------------------------------------------------------------ laya design ---
def _laya(**kw):
    p = LayaPolicy(**kw)
    p.available = True
    p.detail = "fake"
    return p


def _answer(mapping):
    def _ask(state, questions):
        return {"answers": {"action": {"type": "choice", "choice": mapping["choice"],
                                       "probabilities": mapping["probs"],
                                       "confidence": mapping.get("conf", 0.9)}},
                "_ms": 7.0}
    return _ask


def test_laya_choice3_steers_and_the_harness_fires():
    p = _laya(questions_design="choice3", fire_mode="aligned", threshold=0.3)
    p._ask = _answer({"choice": "right", "probs": {"left": 0.1, "right": 0.8, "hold": 0.1}})
    obs = _dummy_obs()
    obs.features["aligned"] = True
    a = p.decide(obs, ["hold"])
    assert a.name == "right_fire"
    assert a.meta["steer"] == "right" and a.meta["fire"] is True
    assert a.meta["latency_ms"] == 7.0


def test_laya_low_confidence_holds_but_still_fires_when_aligned():
    p = _laya(questions_design="choice3", fire_mode="aligned", threshold=0.5)
    p._ask = _answer({"choice": "right", "probs": {"left": 0.35, "right": 0.4, "hold": 0.25}})
    obs = _dummy_obs()
    obs.features["aligned"] = True
    a = p.decide(obs, ["hold"])
    assert a.meta["choice"] == "hold"          # below threshold -> no steering
    assert a.name == "fire"                    # but the aligned read still fires


def test_laya_model_fire_mode_ignores_the_feature():
    p = _laya(questions_design="choice3", fire_mode="model", threshold=0.3)
    p._ask = _answer({"choice": "left", "probs": {"left": 0.7, "right": 0.2, "hold": 0.1}})
    obs = _dummy_obs()
    obs.features["aligned"] = True
    assert p.decide(obs, ["hold"]).name == "left"   # feature says aligned, model did not fire


def test_laya_fire_option_keeps_the_last_steer():
    p = _laya(questions_design="choice", fire_mode="model", threshold=0.3)
    p._ask = _answer({"choice": "right", "probs": {"left": 0.1, "right": 0.6, "hold": 0.1, "fire": 0.2}})
    obs = _dummy_obs()
    assert p.decide(obs, ["hold"]).name == "right"
    p._ask = _answer({"choice": "fire", "probs": {"left": 0.1, "right": 0.2, "hold": 0.1, "fire": 0.6}})
    assert p.decide(obs, ["hold"]).name == "right_fire"


def _dummy_obs():
    from agent.obs import Obs
    return Obs(prose="The player ship is far to the left of the nearest enemy.",
               features={"ship_x": 40, "enemy_x": 620, "dx": 580, "dx_frac": 0.85,
                         "aligned": False, "diving_near": False, "diving_dist": None,
                         "threat_dx_frac": None})