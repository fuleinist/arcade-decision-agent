# arcade-decision-agent

Play browser arcade games with typed-decision models. A local Laya encoder makes a decision for
each frame. Jev (TypeSafe System One) and the OpenAI Decisions API (`gpt-6-luna`) supervise. A
local web page shows every answer with its probability.

Read [PLAN.md](PLAN.md) for the research, the measured numbers and the milestone plan. This file
shows how to run the code.

## Quickstart

```bash
git clone https://github.com/fuleinist/arcade-decision-agent.git
cd arcade-decision-agent
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

.venv/Scripts/python.exe -m pytest -q                       # unit tests
.venv/Scripts/python.exe -m agent.server --port 8770        # then open http://127.0.0.1:8770
```

In the dashboard, pick `env=stub` and `policy=laya`. Press **start**. The left panel shows the
playfield and the KPIs. The right panel shows the exact sentence that the model received, the
option probabilities and the decision log.

## Run episodes from the CLI

```bash
P=.venv/Scripts/python.exe
$P -m agent.eval --env stub --policy scripted --episodes 3            # the thing to beat
$P -m agent.eval --env stub --policy laya --questions choice3 --threshold 0.5
$P -m agent.eval --env stub --policy laya --state-format json         # the wording ablation
$P -m agent.eval --env stub --policy laya --supervisor jev            # tiered reflex and judge
$P -m agent.eval --env page --policy scripted --headed --steps 20     # the live page
```

## Keys and paths for the local and remote tiers

| policy | setting | notes |
|---|---|---|
| `jev` | `TYPESAFE_API_KEY` | $0.042 for each 1M input tokens. `jev:openrouter` uses `OPENROUTER_API_KEY`. `jev:litellm` uses a local LiteLLM |
| `luna` | `OPENAI_API_KEY` | $0.10 for each 1M input tokens. This tier alone accepts an image, through `--send-image` |
| `laya` | `LAYA_EXE`, `LAYA_MODEL` | Optional. The code defaults to `G:/dev/AI/laya/layabin/laya.exe` and `G:/dev/AI/laya/laya_multilingual_q8_0.gguf`, which are the paths on the author's machine. Set both to run Laya from anywhere else. Get the binary and the GGUF from [monatis/ggmlc](https://github.com/monatis/ggmlc) and [mys/laya-multilingual-GGUF](https://huggingface.co/mys/laya-multilingual-GGUF) |

Without keys, both remote tiers stay visible in `/api/health` and report the reason. The loop
continues to run on `scripted` or `laya`.

## Layout

```
agent/obs.py             Obs, Action and GameEnv contracts
agent/perceive.py        pixels to features to PROSE, never JSON
agent/envs/stub.py       a small Galaga-shaped game for eval and tests
agent/envs/emularity.py  the live page adapter: keys, RAM, save states
agent/policies/          scripted, laya (local daemon) and remote (jev, luna)
agent/loop.py            reflex tier for each frame, supervisor tier every N frames
agent/eval.py            episode runner: spreads, ablations, latency, cost
agent/server.py          FastAPI app with a websocket
web/index.html           the dashboard, with no build step
results/*.json           the recorded runs
```

## Scope of this draft

This repository holds the M0 and M1 draft of PLAN.md. It works today: the stub game, the three
policy adapters, the two-tier loop, the eval harness, the dashboard, and a live-page adapter that
reaches the emulator and its save states.

It does not do these things yet: RAM perception, deterministic episodes on the live page,
trustworthy pixel capture, and any run with real Jev or OpenAI keys. Section 8 of PLAN.md lists
the milestones for each gap.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Status

I completed and verified M0 and M1 of PLAN.md. M2, the RAM perception step, is the next real
step. The current scoreboard on the stub game: the scripted baseline scores 4,167, and the best
Laya configuration scores 333. The harness holds. The model in the loop is not good enough yet.
Section 3 and section 7 of PLAN.md name the measurements that come next.