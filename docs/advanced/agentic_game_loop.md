# Agentic Game Loop (`generate_until_game`)

`generate_until_game` evaluates a model as a *policy in an interactive environment* instead of a single-shot generator. Each dataset row becomes one episode: the environment produces observations, the model picks actions, the environment steps, and the task scores the finished episode.

## Architecture

```
task YAML (environment side)          CLI (model side)
  game_env ────────────┐                --agentic_model_server(+_args)
  observation_parser ──┤                --agentic_output_parser(+_args)
  action_parser ───────┤                --agentic_max_parallel_rollouts
                       ▼                          ▼
              ┌─────────────────── run_episode ───────────────────┐
              │                                                    │
   env.reset(doc) → EnvState                                       │
        │                                                          │
        ▼            per step                                      │
   ObservationParser ──▶ AgentInput ──▶ ModelServer.generate       │
        ▲                                     │                    │
        │                              ModelOutputParser           │
        │                                     │                    │
   env.step(action) ◀── ParsedAction ◀── ActionParser              │
        │                                                          │
        ▼                                                          │
   EpisodeResult ─▶ JSON response ─▶ process_results               │
                └─▶ artifacts (summary.md / actions.jsonl / mp4)   │
              └────────────────────────────────────────────────────┘
```

Everything lives in `lmms_eval/agentic/` (one module per concept):

| Module | Contents |
|---|---|
| `types.py` | Dataclass vocabulary: `ContentBlock`, `AgentInput/Output`, `EnvState`, `GameAction`, `ParsedAction`, `StepResult`, `EpisodeStep/Result` |
| `env.py` | `EnvManager` ABC (`reset` / `step` / `get_state` / `close`) |
| `parsers.py` | Typed parser ABCs + generic parsers (`identity`, `qwen`, `action_name`) |
| `servers.py` | `ModelServer` ABC + `openai` (HTTP) + `debug` (fixed action) |
| `episode.py` | `run_episode()` — the rollout loop, a plain function |
| `components.py` | Spec resolution: registry names, import paths, callables, dict specs |
| `runner.py` | `run_generate_until_game()` — Instances → thread pool → JSON responses |
| `trace.py` / `artifacts.py` | Episode → JSON payload / summary.md, actions.jsonl, rollout.mp4 |

Task-specific components (the ViZDoom environment and its parsers) live with their task under `lmms_eval/tasks/vizdoom_agentic/`, not in the framework.

## Task contract (YAML)

The task owns the environment side. Component fields accept a callable (`!function`), an import path string, or a `{name: ..., kwargs...}` dict:

```yaml
output_type: generate_until_game
game_env: !function utils.vizdoom_env_manager            # -> EnvManager
observation_parser: !function utils.vizdoom_observation_parser  # EnvState -> AgentInput
action_parser: !function utils.vizdoom_action_parser     # AgentOutput -> ParsedAction
generation_kwargs:
  max_new_tokens: 64
  temperature: 0
  max_game_steps: 64        # loop keys ride in generation_kwargs
  # game_seed: 1234
  # multiturn: true         # send conversation history to the server
  # history_turns: 6
process_results: !function utils.vizdoom_process_results
```

Factories are called with `doc` and `lmms_eval_specific_kwargs` (signature-filtered), so a factory can specialize per document.

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
  --agentic_output_parser qwen \
  --agentic_max_parallel_rollouts 8 \
  --output_path ./logs/
```

- `--agentic_model_server`: `openai`, `debug`, or an import path (`my_pkg.servers:MyServer`).
- `--agentic_model_server_args`: comma `k=v` pairs; JSON values allowed (`chat_template_kwargs={"enable_thinking":false}`). `openai` accepts `model`, `base_url`, `api_key`, `timeout`, `default_max_tokens`, `max_concurrent_requests`, `enable_thinking`.
- `--agentic_output_parser`: model-side output normalization (`identity` default, `qwen` strips `<think>` and extracts tool calls).
- `--agentic_max_parallel_rollouts`: the single concurrency knob. Each rollout thread has at most one in-flight request, so endpoint concurrency equals this value; OpenAI-compatible servers batch concurrent requests server-side. Set `max_concurrent_requests` on the server only to cap a weak endpoint below the rollout count.

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

- **New environment**: subclass `EnvManager` next to your task, expose a factory in `utils.py`, point `game_env` at it.
- **New parsers**: subclass the typed ABCs (`ObservationParser` must return `AgentInput`; non-text payloads go inside `ContentBlock`s, e.g. `type="tensor"`).
- **New model server**: subclass `ModelServer` (thread-safe `generate`), pass its import path to `--agentic_model_server`.

Known limits, by design of this first iteration: single-agent loops only, no response-cache integration for rollouts, and no Ray/RL serving hooks (a follow-up can add a `ray` server behind the same `ModelServer` boundary).
