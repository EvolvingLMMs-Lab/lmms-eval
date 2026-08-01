# Agentic Game Loop (`generate_until_game`)

`generate_until_game` evaluates a model as a *policy in an interactive environment* instead of a single-shot generator. Each dataset row becomes one episode: the environment produces observations, the model picks actions, the environment steps, and the task scores the finished episode.

## Architecture

```
task YAML                              CLI (model side)
  game_env ────────────┐                --agentic_model_server(+_args)
  model_specific_      │                --agentic_max_parallel_rollouts
    parsers ───────────┤
                       ▼                          ▼
              ┌─────────────────── run_episode ───────────────────┐
              │                                                    │
   env.reset(doc) → EnvState                                       │
        │                                                          │
        ▼            per step                                      │
   observation pipeline ▶ AgentInput ─▶ ModelServer.generate       │
        ▲                                     │                    │
        │                              action pipeline             │
        │                                     │                    │
   env.step(action) ◀────────────── ParsedAction                   │
        │                                                          │
        ▼                                                          │
   EpisodeResult ─▶ JSON response ─▶ process_results               │
                └─▶ artifacts (summary.md / actions.jsonl / mp4)   │
              └────────────────────────────────────────────────────┘
```

The framework lives in `lmms_eval/agentic/`, with extensible component
families split into packages:

| Module | Contents |
|---|---|
| `types.py` | Dataclass vocabulary: `ContentBlock`, `AgentInput/Output`, `EnvState`, `GameAction`, `ParsedAction`, `StepResult`, `EpisodeStep/Result`, `ActionDef`, `ActionSpec` |
| `env.py` | `EnvManager` ABC (`reset` / `step` / `get_state` / `action_spec` / `close`) |
| `pipelines.py` | Select model-specific task pipelines and compose `Any -> Any` functions |
| `servers/` | `ModelServer` ABC plus one module per backend (`openai`, `debug`) |
| `episode.py` | `run_episode()` — the rollout loop, a plain function |
| `components.py` | Spec resolution: registry names, import paths, callables, dict specs |
| `runner.py` | `run_generate_until_game()` — Instances → thread pool → JSON responses |
| `trace.py` / `artifacts.py` | Episode → JSON payload / summary.md, actions.jsonl, rollout.mp4 |

Concrete environments live with their task under `lmms_eval/tasks/<task>/env.py`,
even when they also have a registry short name. This keeps simulator-specific
code out of the framework package. Simulator imports happen inside `reset`,
never at module import time.

Every parser implementation lives with its task (for example,
`lmms_eval/tasks/vizdoom_agentic/parsers.py`), not in the framework. A parser
is an ordinary `(value, ParserContext) -> value` function. Pipelines may
therefore carry text, images, tensors, latents, structured actions, or any
other representation; the framework only validates the final `AgentInput`
and `ParsedAction` values at the model-server and environment boundaries.

## Task contract (YAML)

The task owns its environment and parser behavior. `model_specific_parsers`
has a `default` entry and may override either pipeline with an exact model name
or a case-insensitive glob. A pipeline is one function or a list of functions:

```yaml
output_type: generate_until_game
game_env: !function utils.vizdoom_env_manager            # -> EnvManager
model_specific_parsers:
  default:
    observation: !function parsers.vizdoom_observation_parser
    action: !function parsers.vizdoom_action_parser
  "*Qwen*":
    action:
      - !function parsers.vizdoom_qwen_output_parser
      - !function parsers.vizdoom_action_parser
generation_kwargs:
  max_new_tokens: 64
  temperature: 0
  max_game_steps: 64        # loop keys ride in generation_kwargs
  # game_seed: 1234
  # multiturn: true         # send conversation history to the server
  # history_turns: 6
process_results: !function utils.vizdoom_process_results
```

The model identity comes from the selected `ModelServer`; for the OpenAI
server it is the `model=...` value in `--agentic_model_server_args`. Exact
keys take precedence over glob keys, and a selected entry is merged over
`default`, so the Qwen entry above only needs to replace the action pipeline.
`ParserContext` exposes the current state, request, raw output, history,
document metadata, and selected model name without restricting pipeline value
types.

### Task-local reference: MiniGrid

MiniGrid shows the same contract with a short registry name for its task-local
environment. Its parser functions still live beside the task:

```yaml
output_type: generate_until_game
dataset_path: json
dataset_kwargs:
  data_files:
    test: lmms_eval/tasks/minigrid_agentic/data/minigrid.jsonl
game_env: minigrid                # registry name; dict form takes kwargs:
                                  # game_env: {name: minigrid, max_episode_steps: 100}
model_specific_parsers:
  default:
    observation: !function parsers.minigrid_observation_parser
    action: !function parsers.minigrid_action_parser
  "*Qwen*":
    action:
      - !function parsers.minigrid_qwen_output_parser
      - !function parsers.minigrid_action_parser
generation_kwargs:
  max_game_steps: 40
process_results: !function utils.minigrid_process_results
```

Dataset rows select the scenario (doc-as-scenario): `env_id`, `seed`, and
`max_episode_steps` override manager defaults per episode. Small scenario sets
may use a tracked JSONL; established benchmarks should use their canonical
Hugging Face dataset.

`EnvManager.action_spec()` remains the single declaration of the action space:
`ActionSpec(kind=..., actions=[ActionDef(name, description, schema, aliases)],
submit_actions=[...])`. The task-local observation parser uses it to render
action help, and the task-local action parser uses the same value to validate
model output. The framework does not choose or implement an action parser.

`process_results` receives one JSON string per episode:

```json
{"success": true,
 "metrics": {"vizdoom_success": 1.0, "vizdoom_steps": 12.0},
 "final_state": {"env_id": "...", "step_idx": 12, "observation": {...}, "terminal": true, "metadata": {...}},
 "steps": [{"step_idx": 0, "raw_model_output": "...", "model_output": "...", "action": {...},
            "parse_error": null, "reward": 1.0, "done": false, "info": {...}}],
 "metadata": {"max_steps": 64, "artifacts": {"summary": "...", "video": "..."}}}
```

Arrays, tensors, and images in observations are serialized as compact `{type, shape, dtype}` descriptors, never raw pixels.

## Model side (CLI)

The CLI owns serving. One model server is built per run and shared by all rollouts:

```bash
python -m lmms_eval --tasks vizdoom --model dummy \
  --agentic_model_server openai \
  --agentic_model_server_args model=Qwen/Qwen3.5-9B,base_url=http://127.0.0.1:8000/v1 \
  --agentic_max_parallel_rollouts 8 \
  --output_path ./logs/
```

- `--agentic_model_server`: `openai`, `debug`, or an import path (`my_pkg.servers:MyServer`).
- `--agentic_model_server_args`: comma `k=v` pairs; JSON values allowed (`chat_template_kwargs={"enable_thinking":false}`). `openai` accepts `model`, `base_url`, `api_key`, `timeout`, `default_max_tokens`, `max_concurrent_requests`, `enable_thinking`.
- `--agentic_max_parallel_rollouts`: the single concurrency knob. Each rollout thread has at most one in-flight request, so endpoint concurrency equals this value; OpenAI-compatible servers batch concurrent requests server-side. Set `max_concurrent_requests` on the server only to cap a weak endpoint below the rollout count.
- `--agentic_episode_retries`: extra attempts per episode for flaky simulators (default 0).

### Episode failures never kill the run

Component wiring bugs (unknown registry name or missing parser pipeline) fail the run immediately. Anything the environment or backend raises *during* a rollout is retried `--agentic_episode_retries` times and then recorded as a failed episode: `success=false`, metric `env_error=1.0`, and the exception in `metadata.error`. Aggregate `env_error` in your `metric_list` to make crash rates visible instead of silent.

### No-backend smoke test

```bash
python -m lmms_eval --tasks vizdoom --model dummy \
  --agentic_model_server debug --agentic_model_server_args action=ATTACK \
  --limit 1 --output_path ./logs/
```

`debug` always answers with a fixed action, exercising env, parsers, loop, tracing, and artifacts with zero GPUs.

## Artifacts

With `--output_path`, each episode writes `<output_path>/agentic_artifacts/<task>_doc<id>_<timestamp>/`:

- `summary.md` — per-step table (requested vs executed action, reward, errors)
- `actions.jsonl` — one row per step for programmatic analysis
- `rollout.mp4` — stitched screen frames (requires `av`; scale/fps via `LMMS_AGENTIC_ARTIFACT_SCALE`, `LMMS_AGENTIC_ARTIFACT_FPS`)
- `segments/step_NNN.mp4` — per-action clips when the env emits `action_frames`

## Extending

- **New environment**: add `env.py` next to the task, subclass `EnvManager`, keep simulator imports inside `reset`, and expose it through a task factory or an optional registry short name.
- **New parser**: add an `Any -> Any` function beside the task and reference it from `model_specific_parsers`. Use a list when normalization and action decoding are separate steps. The observation pipeline must end at `AgentInput`; the action pipeline must end at `ParsedAction`. Non-text payloads can travel through arbitrary intermediate values or typed `ContentBlock`s such as `type="tensor"`.
- **New model server**: add a module under `lmms_eval/agentic/servers/`, subclass the thread-safe `ModelServer`, re-export it when it is public, and pass its import path or registry name to `--agentic_model_server`.

Known limits, by design: single-agent loops only, no response-cache integration for rollouts, no Ray/RL serving hooks, and in-process environments only — a remote env server (reset/step/close over HTTP for browser farms or emulator hosts) is deliberately deferred until the first heavy environment lands; the `EnvManager` boundary already fits a thin HTTP client.
