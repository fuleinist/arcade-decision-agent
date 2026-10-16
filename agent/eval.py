"""Episode runner: the actual test bench.

Design rules taken from what is measurable rather than what is convenient:
  * run >= 3 episodes and report the SPREAD, never a single score (score variance on a
    reflex game is huge - the Flappy reference run in the skill library scored 14, 20, 58);
  * every model is measured on identical episodes (same seed set, same env config);
  * the state-serialisation ablation is a first-class flag, because prose vs JSON is the
    difference between a policy that plays and one that loses in both directions;
  * latency and cost are reported per decision, since that is what decides which tier a
    model can live in.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Optional

from .envs import make_env
from .loop import ControlLoop, LoopConfig
from .policies import Policy, build


def run_suite(env_kind: str, reflex_spec: str, episodes: int = 3, seed: int = 0,
              supervisor_spec: Optional[str] = None, cfg: Optional[LoopConfig] = None,
              env_kw: Optional[dict] = None, policy_kw: Optional[dict] = None,
              on_event=None) -> dict:
    env_kw = env_kw or {}
    policy_kw = policy_kw or {}
    reflex = build(reflex_spec, **policy_kw)
    supervisor = build(supervisor_spec, **policy_kw) if supervisor_spec else None

    rows = []
    for ep in range(episodes):
        env = make_env(env_kind, seed=seed + ep, **env_kw)
        loop = ControlLoop(env, reflex, supervisor, cfg, on_event)
        try:
            loop.run()
        finally:
            if env_kind not in ("stub", "fake"):
                env.close()
        s = loop.summary()
        s["episode"] = ep
        s["seed"] = seed + ep
        rows.append(s)

    scores = [r["score"] for r in rows]
    lat = [r["mean_latency_ms"] for r in rows if r["mean_latency_ms"]]
    out = {
        "env": env_kind,
        "reflex": reflex_spec,
        "supervisor": supervisor_spec,
        "policy_health": reflex.health(),
        "state_format": policy_kw.get("state_format"),
        "threshold": policy_kw.get("threshold"),
        "episodes": rows,
        "score_mean": round(statistics.mean(scores), 1),
        "score_spread": scores,
        "score_stdev": round(statistics.pstdev(scores), 1) if len(scores) > 1 else 0.0,
        "mean_latency_ms": round(statistics.mean(lat), 1) if lat else None,
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "errors": sum(r["errors"] for r in rows),
    }
    reflex.close()
    if supervisor and supervisor is not reflex:
        supervisor.close()
    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="agent.eval", description="run arcade episodes and report")
    ap.add_argument("--env", default="stub", help="stub | page")
    ap.add_argument("--policy", default="scripted", help="scripted | laya | jev | luna | jev:openrouter | none")
    ap.add_argument("--supervisor", default=None, help="optional second-tier policy spec")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--decimate", type=int, default=1)
    ap.add_argument("--supervisor-every", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--state-format", default="prose", choices=["prose", "json"])
    ap.add_argument("--questions", default="choice3", choices=["choice3", "choice", "noul"],
                    help="laya only: one 3-option steering question vs 4-option vs three yes/no")
    ap.add_argument("--fire", default="aligned", choices=["aligned", "model"],
                    help="aligned = harness fires when the extractor says lined up; model = engine answers")
    ap.add_argument("--send-image", action="store_true", help="luna only: attach the frame")
    ap.add_argument("--judge", action="store_true", help="run Jev against the state as a second reader")
    ap.add_argument("--url", default=None, help="env=page: game URL")
    ap.add_argument("--headed", action="store_true", help="env=page: show the browser window")
    ap.add_argument("--hold-ms", type=int, default=60, help="env=page: key hold time")
    ap.add_argument("--json", default=None, help="write the full result JSON here")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    policy_kw = {"threshold": a.threshold, "state_format": a.state_format,
                 "questions_design": a.questions, "fire_mode": a.fire}
    if a.policy == "luna" or (a.supervisor or "").startswith("luna"):
        policy_kw["send_image"] = a.send_image
    cfg = LoopConfig(max_steps=a.steps, decimate=a.decimate, supervisor_every=a.supervisor_every,
                     frame_delay_ms=a.hold_ms if a.env == "page" else 0)
    env_kw = {}
    if a.env == "page":
        env_kw = {"url": a.url, "headed": a.headed, "hold_ms": a.hold_ms}

    def log(e: dict) -> None:
        if a.quiet:
            return
        if e["type"] == "decision":
            meta = e.get("meta", {})
            extra = f" p={meta.get('probs')}" if meta.get("probs") else ""
            err = f" ERROR {meta['error']}" if meta.get("error") else ""
            print(f"  step {e['step']:4d} mode={e['mode']:6s} {e['action']:10s}"
                  f" ({e['source']}) score={e['score']}{extra}{err}")
        elif e["type"] == "mode":
            print(f"  [mode] {e['mode']} via {e['policy']} at step {e['step']}")

    res = run_suite(a.env, a.policy, episodes=a.episodes, seed=a.seed,
                    supervisor_spec=a.supervisor, cfg=cfg, env_kw=env_kw,
                    policy_kw=policy_kw, on_event=log)

    print(f"\nenv={res['env']}  reflex={res['reflex']}  supervisor={res['supervisor']}"
          f"  state={res['state_format']}  threshold={res['threshold']}")
    print(f"scores by episode: {res['score_spread']}   mean={res['score_mean']}  sd={res['score_stdev']}")
    print(f"mean decision latency: {res['mean_latency_ms']} ms   cost: ${res['cost_usd']}   errors: {res['errors']}")
    if not res["policy_health"].get("available", True):
        print(f"NOTE: policy unavailable - {res['policy_health'].get('detail')}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=str)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())