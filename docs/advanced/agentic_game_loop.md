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
| `types.py` | Dataclass vocabulary: `ContentBlock`, `AgentInput/Output`, `EnvState`, `GameAction`, `ParsedAction`, `StepResult`, `EpisodeStep/Result`, `ActionDef`, `ActionSpec` |
| `env.py` | `EnvManager` ABC (`reset` / `step` / `get_state` / `action_spec` / `close`) |
| `envs/` | Reusable environments, one module each, referenced by registry name (`minigrid`) |
| `parsers.py` | Typed parser ABCs + generic parsers (`identity`, `qwen`, `action_name`, `schema`, `free_text`, `template`) |
| `servers.py` | `ModelServer` ABC + `openai` (HTTP) + `debug` (fixed action) |
| `episode.py` | `run_episode()` — the rollout loop, a plain function |
| `components.py` | Spec resolution: registry names, import paths, callables, dict specs |
| `runner.py` | `run_generate_until_game()` — Instances → thread pool → JSON responses |
| `trace.py` / `artifacts.py` | Episode → JSON payload / summary.md, actions.jsonl, rollout.mp4 |

Environments come in two flavors: reusable ones live in `lmms_eval/agentic/envs/` behind a registry name, while bespoke single-task ones (the ViZDoom environment and its parsers) live with their task under `lmms_eval/tasks/vizdoom_agentic/`. Either way the simulator import happens inside `reset`, never at module import time.

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

### Minimal contract: registry env, no parser code

Both task-side parsers are optional. When the YAML omits them the loop falls back to the generic `TemplateObservationParser` and to an action parser built from the environment's `action_spec()`, so a task over a registered environment is just the env name plus metrics:

```yaml
output_type: generate_until_game
dataset_path: !function utils.minigrid_dataset   # docs-from-code, see below
game_env: minigrid                # registry name; dict form takes kwargs:
                                  # game_env: {name: minigrid, max_episode_steps: 100}
generation_kwargs:
  max_game_steps: 40
process_results: !function utils.minigrid_process_results
```

`lmms_eval/tasks/minigrid_agentic/` is the reference: zero component code in the task directory. Dataset rows select the scenario (doc-as-scenario): `env_id`, `seed`, and `max_episode_steps` in a doc override the manager defaults per episode.

For env-loop tasks the "dataset" is usually a handful of scenario configs, not real data, so `dataset_path` also accepts `!function`: a factory in `utils.py` returning a `datasets.DatasetDict` (e.g. `datasets.DatasetDict({"test": datasets.Dataset.from_list(SCENARIOS)})`). That keeps scenario lists in code — no tracked data file, no hub dependency.

Two conventions make the defaults work:

- **Reserved observation keys.** A dict `EnvState.observation` may carry `text` (str), `images` (list of frames), `video` (list of frames for one clip), `variables` (dict), and `actions` (pre-rendered action help overriding `action_spec()`). `TemplateObservationParser` renders them into the prompt plus media blocks; a `template` string with `{instruction}` `{text}` `{variables}` `{actions}` `{step_line}` placeholders customizes the text without code (`observation_parser: {name: template, template: "..."}`).
- **`EnvManager.action_spec()`.** The environment declares its action space once — `ActionSpec(kind=..., actions=[ActionDef(name, description, schema, aliases)], submit_actions=[...])` — instead of the action list living in the env config, the prompt text, and the parser three times over. `kind` picks the default parser: `discrete` → `ActionNameParser`, `parameterized` → `SchemaActionParser` (tool calls or `{"action": ..., ...args}` JSON, checked against each action's schema), `free_text` → `FreeTextActionParser` (the reply is the command; the env validates it).

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
- `--agentic_episode_retries`: extra attempts per episode for flaky simulators (default 0).

### Episode failures never kill the run

Component wiring bugs (unknown registry name, missing `action_parser` with no `action_spec()`) fail the run immediately. Anything the environment or backend raises *during* a rollout is retried `--agentic_episode_retries` times and then recorded as a failed episode: `success=false`, metric `env_error=1.0`, and the exception in `metadata.error`. Aggregate `env_error` in your `metric_list` to make crash rates visible instead of silent.

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

- **New shared environment**: add one module under `lmms_eval/agentic/envs/` — subclass `EnvManager`, declare `action_spec()`, emit reserved observation keys, keep simulator imports inside `reset` — and register its import path in `components.REGISTRY["env_manager"]`. The bar to hit (see `envs/minigrid.py`): a new task should be the YAML, a dataset, and `process_results`, with zero parser code.
- **Bespoke environment**: subclass `EnvManager` next to your task, expose a factory in `utils.py`, point `game_env: !function utils.<factory>` at it (the ViZDoom pattern, for envs with task-specific prompting or action semantics).
- **New parsers**: subclass the typed ABCs (`ObservationParser` must return `AgentInput`; non-text payloads go inside `ContentBlock`s, e.g. `type="tensor"`).
- **New model server**: subclass `ModelServer` (thread-safe `generate`), pass its import path to `--agentic_model_server`.

Known limits, by design: single-agent loops only, no response-cache integration for rollouts, no Ray/RL serving hooks, and in-process environments only — a remote env server (reset/step/close over HTTP for browser farms or emulator hosts) is deliberately deferred until the first heavy environment lands; the `EnvManager` boundary already fits a thin HTTP client.
