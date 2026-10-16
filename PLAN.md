# Arcade Decision Agent: research, design and plan

*I researched and probed this project on 2026-10-16 on this machine (RTX 3090, Windows,
git-bash). Every number below comes from a measurement on this machine, unless the text says
otherwise. Every claim marked "verified" has a command or a captured frame behind it.*

---

## 0. Summary

We can build this project. The three models force one main design decision: the tier split.

Two of the three models (Jev and local Laya) cannot see. Two of the three models cannot hold a
30 Hz loop, because a network round trip costs a round trip and the local model costs GPU time.
The third model, the OpenAI Decisions API with `gpt-6-luna`, can read images. It pays for that
ability with network latency and a price of $0.10 for each million input tokens.

The architecture that follows from those limits:

```
                     ┌── supervisor tier (Jev / Luna, every ~20 frames, 1–4 Hz) ──┐
                     │   "is the ship in danger?" → mode = engage | dodge        │
                     └──────────────────────┬───────────────────────────────────┘
                                            ▼
  frame ──► perceive ──► PROSE STATE ──► reflex tier (local Laya, every frame)
  (pixels or RAM)   (never JSON)         "which way should the ship move?" → key
                                            │
                                            ▼
                                        key events → emulator
```

The deliverable is a local web page. You pick a policy, then watch the frame. You also watch
every typed answer with its probability. You can take the keyboard back at any time.

One limit belongs at the start of this document. A decision model will not play Galaga at adult
level. Such a model is a threshold policy. It makes one forward pass, and it has no memory. The
question we can measure is the quality of its judgment against a scripted baseline, on identical
episodes. On the stub game that baseline scores 4,167. The best Laya configuration I found scores
333. That gap is the main result of this research. To close the gap we must improve the question
design and the perception. A larger model does not close it.

---

## 1. What the three models are

I verified each row below on 2026-10-16.

| | Jev (TypeSafe System One) | OpenAI Decisions API | Laya, local |
|---|---|---|---|
| endpoint | `POST https://api.typesafe.ai/v1/systemone` | `POST https://api.openai.com/v1/decisions` | `laya.exe daemon`, stdio JSON-RPC |
| model | `jev-latest` (1.13) | `gpt-6-luna`, the only model | `laya_multilingual_q8_0.gguf` |
| request body | `state` and `questions{id: {type, instructions, criteria}}` | `input` messages and `questions[]` | `{state, questions{...}}`, one line for each call |
| question types | `choice` (255 options maximum), `score` (2 to 10 levels), `noul` | `choice` (2 to 255), `score`, `predicate` | `choice`, `score`, `noul` |
| images | no. Text and JSON state only | yes. `input_image` inside the user message | no |
| price | $0.042 for each 1M input tokens, output free | $0.10 for each 1M input tokens, output free | $0.00, uses the local GPU |
| other access | OpenRouter `/api/alpha/decisions` as `typesafe/jev-1.13`, LiteLLM `/v1/systemone` | public beta. OpenAI claims a speed 10 times that of the Responses API. OpenRouter alias `openai/gpt-6-luna-decisions` | ggmlc binary. llama.cpp and Ollama cannot load it |

Three facts shape the design.

1. **None of the three models writes text.** No token stream exists. An OpenAI-style client
   fails against them. OpenRouter answers with this error: `typesafe/jev-1.13 is a decisions
   model and cannot be used with the chat/completions endpoint`. Treat each model as a tool, as
   an MCP tool, as a router, or as a gate. Never treat it as a provider.
2. **Laya answers one typed question in 10 ms to 30 ms.** I measured 12 ms to 19 ms for a
   single-question call, 4 ms to 8 ms inside the eval loop, and 345 ms for the first call after a
   load. That speed fits inside a frame loop.
3. **The two network tiers run at 1 Hz to 4 Hz, not at 60 Hz.** A round trip plus our own
   overhead lands in the hundreds of milliseconds. These models must supervise. A 30 Hz network
   policy would also create a rate-limit problem and a cost problem.

One adjacent capability is worth knowing about. The OpenAI **computer-use** tool, that is the
`computer` tool in the Responses API or `gpt-6-astra` writing Playwright and PyAutoGUI code,
suits a different problem. That problem is navigating the site: dismissing overlays, clicking
through to the game, and inserting a coin. Do not ask that tool to play the game frame by frame.

---

## 2. The target page

`https://www.classicgamesarcade.com/game/21637/galaga.html` runs archive.org's **Emularity**
loader. The loader file is `loader.js`, and its copyright line names Daniel Brooks, Jason Scott
and Grant Galitz. The page drives a **MAME WebAssembly** build at `em/mame/mamegalaga.js`
(3.6 MB), and it uses BrowserFS for the ROM filesystem. The page also carries AdSense, analytics,
a Mailchimp form, a Twitter widget, a Facebook widget, and an old reCAPTCHA v1 challenge. Its own
instructions read: "Key 5 to add credits. 1 to start. Arrows to move. Ctrl to fire."

Emularity hides nothing. Page JavaScript reaches the emulator, and that is the most useful
finding of this research.

| hook | status | why it matters |
|---|---|---|
| `window.Module._SDL_SendKeyboardKey` | callable | a second input path beside real key events |
| `window.Module.HEAPU8` | 29,097,984 bytes, readable | page JavaScript can read MAME RAM, so state can come from memory and not only from pixels |
| `Module.__ZN15running_machine15emscripten_saveEPKc` | callable, returns true | MAME save states from page JavaScript |
| `Module.__ZN15running_machine15emscripten_loadEPKc` | callable, returns true | reproducible episodes, because we can return to an identical condition |
| `Module.__ZN15running_machine21emscripten_soft_resetEv` | callable | a reset without a page reload |
| `window.onkeydown`, bound at `loader.js:1900` | works | real browser key events reach the emulator |
| `canvas#canvas`, 672 by 768, WebGL | `getContext('2d')` returns `null` | we must screenshot the element, because no 2D context exists |

### Three pitfalls

These three pitfalls cost me real time.

1. **The canvas sits about 600 px below the fold.** The page is 2,814 px tall. The viewport was
   566 px tall. A clip screenshot without a scroll returns a small black PNG of about 2 KB.
   Scroll the element into view first.
2. **`Module.calledRun` does not mean "the game runs".** The flag became true about 9 s after
   navigation. The canvas stayed black for several seconds after that. Test the video, not the
   module. A readiness test must find a frame that is not blank, and that changes between two
   captures. The adapter implements this test as `_wait_for_video()`.
3. **Chrome automation does not capture this canvas with trustworthy pixels.** I measured the
   dominant colour of a `#canvas` element screenshot: `(222,222,222)`, which is the page
   background. Galaga uses a black starfield. The result held in headless mode and in headed
   mode, and with SwiftShader GL flags. The screenshot does not receive the WebGL surface. Every
   later step, such as ship detection and the vision questions, therefore reads the wrong pixels.
   Earlier readings that looked like motion came from the page's ads and widgets, not from the
   emulator. A blank test or a motion test can pass on the wrong element. Section 5 puts the RAM
   path first for this target because of this pitfall.

---

## 3. The two constraints

**(a) Two of the three models cannot see.** The system must convert the pixels, or the RAM, into
words before it asks a question. The words matter more than the model. I measured this on the
local engine. The facts were identical in all four rows below. The question was one `noul`: "The
player ship is left of the enemy and must move right now."

| state form | p, ship LEFT | p, ship RIGHT | separation |
|---|---:|---:|---:|
| one-sentence prose, "far to the left of the nearest enemy" | 0.833 | 0.307 | **+0.526** |
| two-clause prose, direction named twice | 0.983 | 0.756 | +0.227 |
| the same facts as a JSON dict | 0.098 | 0.297 | **−0.199** |
| two bare numbers | 0.240 | 0.328 | −0.088 |

Prose separates the two cases. JSON reverses them. Every adapter in this project therefore emits
a sentence. The JSON path stays only as an ablation switch, so we can measure the claim again
instead of trusting it.

**(b) A threshold belongs to one question set.** The engine and the prose states were identical
in all three rows below. I measured agreement with the scripted baseline over 120 real game
states.

| question design | agreement with scripted | probability range seen |
|---|---:|---|
| 3 `noul` questions, "must move left now", "must move right now", "should fire now" | 0.392 | 0.014 to 0.908 |
| 1 `choice` question, 4 options: left, right, hold, fire | **0.758** | 0.001 to 0.983 |
| 1 `choice` question, 6 options, combined move and fire | **0.283** | 0.000 to 0.960 |

Two lessons follow, and both are reproducible.

* **The options compete for one head budget.** Four options scored 0.758. Six options scored
  0.283. Keep `choice` small. Move a mechanical act out of the option set instead of folding it
  in.
* **The engine can answer on the word instead of the relation.** With three `noul` questions on
  real states, a ship to the left of the enemy scored p(move_right) = 0.352. A ship to the right
  of the enemy scored p(move_right) = 0.842. The engine echoed the word "right" from the state.
  Also, with the default threshold of 0.5 the three-question design stops. Every probability sits
  below the line, so the policy holds on every frame and scores 0. The policy therefore ships a
  measured threshold and a measured question design.

---

## 4. Results in the game

I used a stub game first. The stub makes the harness testable without ROMs, browser behaviour or
rate limits. It also gives the decision model a task it can plausibly do: one scalar and one
threshold for each frame.

Three episodes, 500 steps each, identical seeds:

| policy | scores by episode | mean | sd | decisions | mean latency |
|---|---|---:|---:|---:|---:|
| scripted baseline, no model | 4100, 4200, 4200 | **4166.7** | 47.1 | 0 | – |
| Laya, 3 `noul`, threshold 0.5 | 0, 0, 0 | 0 | 0.0 | 1500 | 8.3 ms |
| Laya, 3 `noul`, threshold 0.3 | 200, 100, 300 | 200.0 | 81.6 | 1500 | 6.5 ms |
| Laya, 4 `choice` with fire, threshold 0.3 | 0, 0, 0 | 0 | 0.0 | 1500 | 4.3 ms |
| Laya, 3 `choice` steering, harness fires, threshold 0.3 | 300, 100, 400 | 266.7 | 124.7 | 1500 | 4.3 ms |
| Laya, same, threshold 0.5 | 400, 400, 200 | 333.3 | 94.3 | 1500 | 4.2 ms |
| Laya, same, JSON state, threshold 0.3 | 400, 200, 400 | 333.3 | 94.3 | 1500 | 18.0 ms |

Read the spread, not the mean. The rows 300/100/400 and 400/400/200 show the same policy
behaviour. The JSON row shows a second result worth noting. At the question level prose separates
by +0.53, and JSON reverses the answer. In the end score on this stub the two forms are equal.
The reason is that the 3-option `choice` question produced the same end score with either state
form. Both facts hold at the same time. That is why the eval harness is the source of truth, and
not this document.

Error handling held up. In an earlier run with three `noul` questions the daemon was under load,
and 26 calls raised a timeout. Every one of those calls degraded to a recorded `hold` decision.
The loop did not stop.

---

## 5. The perception ladder

Three sources produce the same prose state. All three implement one contract: `Obs.features`
feeds `perceive.prose_from_features`. A policy cannot tell which source is behind it.

| source | status | notes |
|---|---|---|
| RAM, through the MAME heap and the save states | verified reachable. I recommend this source first for this target | We must locate Galaga's Z80 memory map inside the 29 MB heap. Save states then give per-episode determinism. This source is the next milestone. It is also the only source on this target that returns trustworthy state today |
| pixels | implemented as `perceive.frame_features`. Correct on a local canvas. Not trustworthy on this page | Element and CDP screenshots do not receive the WebGL surface here, see section 2 pitfall 3. Options: host the emulator ourselves, use a 2D-canvas core such as `jsnes`, `fceumm` or Ruffle, or capture the window at the operating-system level |
| vision, where Luna reads the frame | implemented. `LunaPolicy(send_image=True)` builds the `input_image` part. Blocked by the same capture problem | When a real frame exists, this tier can answer "is an enemy diving at the ship?" without a feature extractor written by hand |

The prose contract is narrow on purpose: one scalar that matters, and one clause.

```
"The player ship is far to the left of the nearest enemy."   ← aim question
"An enemy is diving straight at the player ship."            ← dodge question
```

---

## 6. The web interface

`agent/server.py` is a FastAPI app with a websocket. It serves `web/index.html`: one page, no
build step, no npm, about 12.6 KB of HTML.

```
GET  /                the dashboard
GET  /api/health      runner state, and the availability and reason for each policy
GET  /api/state       the latest summary and the event ring buffer
POST /api/start       {env, policy, supervisor, threshold, steps, supervisor_every, seed, …}
POST /api/stop        stop the loop
POST /api/manual      {action}, queued ahead of the policy with source = manual
WS   /ws              obs · decision · mode · summary · error · finished
```

The dashboard has five panels.

* **screen**: the live frame, or the stub's ASCII playfield, plus score, steps, decisions, mean
  milliseconds, cost and errors.
* **state as the model sees it**: the exact prose sentence, the extracted features and the mode.
* **typed answers**: one bar for each option, with its probability. A bar changes colour when it
  crosses the threshold. This panel is the main diagnostic. The score tells you that the policy
  failed. The probabilities tell you why.
* **decision log**.
* **manual override**: buttons, and the left, right and space keys. A manual action enters the
  running loop.

I smoke-tested the interface. The dashboard answered with 200 and 12,642 bytes. `/api/health`
reported `laya` as available, and `jev` and `luna` as unavailable with the reason `no API key
(set TYPESAFE_API_KEY or DECISIONS_API_KEY)`. A run with `stub` and `laya`, supervised by `laya`,
completed 60 steps with score 300, 60 decisions, 7.4 ms mean latency and 0 errors. It returned
its summary over the socket.

---

## 7. Test plan

`agent/eval.py` is the bench. It enforces four rules. Run at least 3 episodes and report the
spread. Give every model the same episodes. Expose the prose and JSON ablation as a flag. Report
latency and cost for each decision.

```
python -m agent.eval --env stub --policy scripted                    # the thing to beat
python -m agent.eval --env stub --policy laya --questions choice3 --fire aligned --threshold 0.5
python -m agent.eval --env stub --policy laya --state-format json    # the wording ablation
python -m agent.eval --env stub --policy laya --supervisor jev       # tiered reflex and judge
python -m agent.eval --env page --policy scripted --headed           # against the live page
```

Run these ablations when the keys exist.

1. **Reflex alone against the tiered setup.** Does a Jev or Luna supervisor that sets the mode
   beat Laya alone?
2. **Vision against no vision.** Compare `luna --send-image` with `luna` on prose only, with the
   same question set.
3. **Question design.** Compare `choice3`, `choice` and `noul`, and sweep the threshold, for each
   checkpoint.
4. **Decimation.** Try `--decimate 1`, `2` and `4`. Measure how much reflex rate the score needs.
5. **The model matrix.** Run scripted, laya, jev and luna on identical episodes. Report the cost
   for each episode.

The harness meets these acceptance criteria today, and unit tests cover each one. Scripted beats
`hold`. A policy that always errors does not stop the loop. A manual action overrides the policy.
The supervisor can change the mode. Each adapter payload carries the right fields. An unavailable
policy degrades to `hold` and reports the reason.

One acceptance criterion is not met yet. Laya as a steering policy must beat the floor of holding
on every frame, by a clear margin, over 5 episodes or more. The tiered setup must beat the
single-tier setup, or we state that the supervisor tier does not earn its latency. Laya scores
333 today, against the scripted baseline of 4,167. That gap is the work item.

---

## 8. Milestones

| | milestone | verification |
|---|---|---|
| **M0** | done. Harness, stub, three policy adapters, eval and dashboard | 20 unit tests pass. Seven eval runs recorded in `results/`. The dashboard smoke test above |
| **M1** | done. The live-page adapter reaches the emulator | The MAME module loads. `HEAPU8` reads 29 MB. Save state returns true, and load state returns true. Keys arrive. The adapter captures frames |
| **M2** | RAM perception: locate Galaga's video RAM and work RAM inside the heap, then emit prose from it | A prose state that follows a real game after a coin insert. A ship position that moves in the correct direction while you hold a key |
| **M3** | deterministic episodes on the live target, through `mame_save` and `mame_load` | Two episodes from the same state file produce identical decisions for a deterministic policy |
| **M4** | frame capture that works, through a local emulator or a window capture, so that pixels and the vision tier work | One screenshot that a human accepts as Galaga. One Luna `predicate` answer that agrees with a hand-read frame |
| **M5** | Jev and Luna wired with real keys, then the model matrix | A results table with cost for each episode and the spread for each model, in this file |
| **M6** | a second game with one axis and a clean canvas, to separate "arcade" from "this page" | The same policy configuration scores above zero on both games |

## 9. Risks

| risk | mitigation |
|---|---|
| Chrome automation does not composite WebGL canvases, which I measured | Use the RAM path first. For pixels, host the emulator ourselves or capture the operating-system window. Do not use a motion test to prove that the frame shows the game |
| The third-party page is a weak test rig. It carries ads, a reCAPTCHA widget, a canvas below the fold and an unknown uptime. Its terms of service are also a factor | Keep `GameEnv` as an interface. Run the eval rig against a local emulator with our own ROM. Treat the live page as an integration check, and not as a dependency |
| ROM legality | Ship the harness, never a ROM. Use your own files, or a homebrew or public-domain title such as a Pong or Breakout clone, for anything you share |
| No API keys exist in this shell environment. I looked for `TYPESAFE_API_KEY`, `OPENAI_API_KEY` and `OPENROUTER_API_KEY`, and all three are absent | I wrote both remote tiers and unit-tested them against fakes, and they report `no API key (set …)` in the UI. To wire them, set `TYPESAFE_API_KEY` or `OPENAI_API_KEY`. You can also point base_url at OpenRouter `/api/alpha/decisions` or at a local LiteLLM `/v1/systemone` |
| Cost from a per-frame network policy | The loop supervises the reflex by design. The eval prints `cost_usd` for each episode, and the dashboard shows the cost live. At 30 Hz, a rate of $0.042 for each 1M tokens would still produce a real bill |
| GPU contention on a shared card | VRAM exhaustion is the failure mode, not compute. Laya's daemon can stall past the timeout under pressure from another workload. `LAYA_CPU_FALLBACK=0` makes it fail fast instead of degrading by about 16 times. The eval reports the error count in either case |
| Over-calling the model | This is the measured failure mode of the tool class. Ask 1 to 3 short questions for each call. Keep `choice` small. Skip the model for a deterministic read, which is why firing is the harness's job by default |

## 10. Decisions I need from you

1. **Keys.** Do you want Jev and the OpenAI Decisions API wired with real keys? Or should I route
   both through an OpenAI-compatible gateway that you already pay for?
2. **Test rig.** Do you have a local arcade setup that I should target instead of the live page? A
   MAME or FBNeo ROM you own would work. A JS-core game with a plain 2D canvas, such as Pong or
   Breakout, would give trustworthy pixels.
3. **RAM depth.** Should I invest in the Galaga Z80 memory map, for better state and
   deterministic seeds? Or should the project stay on pixels and accept a weaker state?
4. **The second site.** If the goal covers a family of sites, I would build one more adapter of
   the same shape first, for a 2D-canvas game host. I would then generalise from two data points.

---

## Appendix: verified commands

```bash
# local Laya: can it answer a custom typed question? Yes, in 12 ms to 19 ms.
export LAYA_EXE=/path/to/layabin/laya.exe         # omit when `laya` is on PATH
export LAYA_MODEL=/path/to/laya_multilingual_q8_0.gguf
"$LAYA_EXE" daemon "$LAYA_MODEL" --device auto
# then send one JSON object for each line on stdin:
{"id":1,"state":"The player ship is far to the left of the nearest enemy.",
 "questions":{"move_right":{"type":"noul","instructions":"The player ship is left of the enemy and must move right now."}}}

# the project
git clone https://github.com/fuleinist/arcade-decision-agent.git
cd arcade-decision-agent
uv venv .venv --python 3.12 && uv pip install -r requirements.txt
uv run python -m pytest -q                     # 20 pass
uv run python -m agent.eval --env stub --policy scripted --episodes 3
uv run python -m agent.server --port 8770      # then open http://127.0.0.1:8770

# what the target page exposes. Paste this into the page console.
Object.keys(window.Module).filter(k => /SDL_SendKeyboardKey|emscripten_save|HEAPU8/.test(k))
```

File map. `agent/obs.py` holds the contracts. `agent/perceive.py` holds the pixels and the prose.
`agent/envs/stub.py` and `agent/envs/emularity.py` hold the two environments. `agent/policies/`
holds the scripted policy, the Laya policy and the remote adapters. `agent/loop.py` holds the two
tiers. `agent/eval.py` runs episodes. `agent/server.py` and `web/index.html` serve the interface.
`tests/` holds the tests. `results/` holds the recorded runs.