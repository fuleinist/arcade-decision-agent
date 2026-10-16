# Recorded runs

Each file is the full JSON output of one `agent.eval` run. The harness prints the same table to
stdout, and writes this file with `--json`.

| file | command | result |
|---|---|---|
| `scripted.json` | `--env stub --policy scripted --episodes 3 --steps 500` | 4100, 4200, 4200. Mean 4166.7 |
| `laya_prose.json` | `--env stub --policy laya --episodes 3 --steps 500` | 0, 0, 0. The three-`noul` design at the default threshold of 0.5 stops |
| `laya_noul_03.json` | same, `--questions noul --threshold 0.3` | 200, 100, 300. Mean 200.0 |
| `laya_json.json` | same, `--state-format json --threshold 0.5` | 300, 200, 400. 26 timeout errors, all degraded to `hold` |
| `laya_choice_0.0.json` | `--questions choice --threshold 0.0` | 0, 0, 0. The 4-option design never picks `fire` |
| `laya_choice_0.3.json` | `--questions choice --threshold 0.3` | 0, 0, 0 |
| `laya_choice_0.5.json` | `--questions choice --threshold 0.5` | 0, 0, 0 |
| `laya_choice_json.json` | `--questions choice --state-format json --threshold 0.3` | 400, 0, 0 |
| `laya_steer_0.3.json` | `--questions choice3 --fire aligned --threshold 0.3` | 300, 100, 400. Mean 266.7 |
| `laya_steer_0.5.json` | `--questions choice3 --fire aligned --threshold 0.5` | 400, 400, 200. Mean 333.3, 4.2 ms for each decision |
| `laya_steer_json.json` | same as `laya_steer_0.3` with `--state-format json` | 400, 200, 400 |

Read the spread, not the mean. `laya_steer_0.3` and `laya_steer_0.5` show the same policy
behaviour on different seeds.

These runs use the stub game. The stub needs no ROM, no browser and no API key, so you can
reproduce every row with the command in the table.