# Agentic Game Loop Design

This document describes the lightweight `generate_until_game` design used for
interactive simulator tasks such as VizDoom-style environments.

The core design constraint is composability. If there are `N` model output
styles and `M` task action schemas, the framework should not require `N * M`
custom task parsers. Model-specific cleanup lives in a model output parser, and
task-specific action parsing lives in an action parser.

## Goals

- Support interactive game or simulator loops where each model response affects
  the next environment state.
- Keep model backend concerns separate from task action schemas.
- Make model-family quirks reusable across tasks. Qwen thinking/tool-call output
  is a representative example, but the layer is not Qwen-specific.
- Keep task definitions YAML-first through named registries.
- Preserve raw model outputs for debugging while scoring normalized actions.
- Keep the protocol flexible enough for text, multimodal, policy, or latent
  model interfaces instead of assuming every model is text-in/text-out.

## Non-goals

- `generate_until_game` does not replace the existing `generate_until_agentic`
  tool-call loop. It is for environment-step loops with explicit `EnvManager`
  state transitions.
- It does not require task code to know how inference is served. The shipped
  inference `ModelServer` targets OpenAI-compatible HTTP endpoints (plus a
  backend-free `debug` server for smoke tests); future implementations may wrap
  other serving or in-process runtimes behind the same base protocol.
- Task code should not hard-code model-family parsing rules.

## Pipeline

Each game request is executed by a `LoopWorker`:

```text
EnvState
  -> ObservationParser
  -> model request object
  -> ModelServer
  -> raw model output object
  -> ModelOutputParser
  -> normalized output object
  -> ActionParser
  -> GameAction
  -> EnvManager.step(...)
  -> EnvState
```

The loop repeats until the environment is terminal or `max_game_steps` is
reached.

## Components

| Component | Direction | Responsibility |
|-----------|-----------|----------------|
| `EnvManager` | task state | Owns environment lifecycle, reset/step logic, latest `EnvState`, terminal state, rewards, and task metrics. |
| `ObservationParser` | `EnvState -> Any` | Runtime-selected adapter that turns environment state into the model-facing request object. Chat/VLM tasks commonly return `AgentInput`; policy or VLA tasks can return tensors, dataclasses, dicts, or runtime-native objects. |
| `ModelServer` | `Any -> Any` | Runs inference. It hides backend details such as HTTP APIs, request batching, generation parameters, async-friendly calls, and native tensor/runtime adapters. |
| `ModelOutputParser` | raw `Any -> Any` | Normalizes model-family or runtime output without knowing the task action schema. Examples: strip `<think>...</think>` from an `AgentOutput`, extract Qwen XML tool calls into metadata, or unwrap a policy output dataclass. |
| `ActionParser` | normalized `Any -> ParsedAction` | Runtime-selected adapter that maps normalized model output into an environment action. It may know the task action schema, but the concrete parser choice belongs with the model/runtime command. |
| `LoopWorker` | orchestration | Wires components together and records per-step traces. |
| `LoopManager` | scheduling | Owns rollout concurrency, thread management, and loop-level batching across rollout sessions. |

All parser roles implement the same base protocol:

```python
class Parser:
    def parse(self, value: Any, ctx: ParserContext) -> Any:
        ...
```

`ParserContext` carries rollout side channels such as `state`, `agent_id`,
`step_idx`, the pending `request`, raw model output, conversation history, and
metadata. Parser payloads themselves intentionally stay unconstrained; use
`AgentInput` and `AgentOutput` only when a chat-style envelope is useful.

## Parser Layering

Parsers are expected to grow quickly across both model/runtime families and
tasks, so they are organized by parser role:

```text
lmms_eval/agentic/parsers/
  model_output/
    identity.py
    qwen.py
  observation/
    vizdoom.py
  action/
    action_name.py
    vizdoom.py
```

Each parser role has its own factory map, so the same short name can be used
for different roles. For example, `vizdoom` can mean a VizDoom
`ObservationParser` and a VizDoom `ActionParser`, then be selected at runtime:

```bash
python -m lmms_eval \
  --tasks vizdoom \
  --agentic_observation_parser vizdoom \
  --agentic_action_parser vizdoom
```

When a parser is truly reusable, keep it generic:

```text
QwenModelOutputParser
LlavaModelOutputParser
...

ActionNameParser(actions=[...])
MinecraftActionParser(...)
...
```

When the model/runtime input-output convention and the task action schema are
tightly coupled, use the task/domain name such as `vizdoom`. The implementation
can still reuse shared helper classes, but the factory name should make the
actual contract obvious.

The runtime command chooses all model-facing parser contracts:

```bash
python -m lmms_eval \
  --model huggingface \
  --model_args pretrained=Qwen/Qwen4-VL-Example,trust_remote_code=True \
  --tasks vizdoom \
  --agentic_model_output_parser qwen \
  --agentic_observation_parser vizdoom \
  --agentic_action_parser vizdoom \
  --agentic_action_parser_args 'submit_actions=["SUBMIT"],noop_actions=["NOOP"]'
```

The Qwen parser is reusable across any task that needs Qwen-style output
normalization. The `vizdoom` observation/action parsers are
runtime-facing adapters between a particular model convention and the native
VizDoom state/action schema.

## YAML Contract

A `generate_until_game` task config provides these components:

```yaml
output_type: generate_until_game
game_env: !function utils.vizdoom_env_manager
generation_kwargs:
  max_game_steps: 64
  max_new_tokens: 64
  temperature: 0
```

The `game_env` field is kept as the task YAML hook for compatibility, but its
value should be a callable factory that returns an `EnvManager`.

Model-side runtime choices stay in the command line:

```bash
python -m lmms_eval \
  --model dummy \
  --tasks vizdoom \
  --agentic_model_server openai \
  --agentic_model_server_args 'model=Qwen/Qwen4-VL-Example,base_url=http://127.0.0.1:8000/v1,max_concurrent_requests=4' \
  --agentic_max_parallel_rollouts 4 \
  --agentic_model_output_parser qwen \
  --agentic_observation_parser vizdoom \
  --agentic_observation_parser_args 'video=true,video_buffer=screen_history,image_buffers=["screen"],include_structured_state=true,include_raw_buffers=true' \
  --agentic_action_parser vizdoom \
  --agentic_action_parser_args 'submit_actions=["SUBMIT"],noop_actions=["NOOP"]' \
  --agentic_loop_worker simple \
  --gen_kwargs max_new_tokens=512,temperature=0,max_game_steps=24
```

`agentic_model_server` currently defaults to `openai`, an OpenAI-compatible
Chat Completions endpoint. `agentic_model_output_parser` defaults to
`identity`. `agentic_loop_worker` defaults to `simple`.

This keeps environment semantics in the task config and model-family input /
output formatting in runtime configuration.

## Single-Turn vs Multi-Turn Rollouts

By default, `simple` runs each environment step as an independent model call:
the observation parser builds the current model request, the model returns an
action, and the next step starts from a fresh request. This is useful for
stateless policy models and keeps context size bounded.

For chat VLMs, a game rollout is often better represented as a multi-turn
conversation:

```text
user: current prompt + video segment + current frame
assistant:
  <tool_call>
  <function=press_buttons>
  <parameter=buttons>ATTACK</parameter>
  <parameter=tics>1</parameter>
  </function>
  </tool_call>
user: next prompt + new video segment + current frame
assistant: ...
```

Enable that at runtime through the loop worker, not in task YAML:

```bash
python -m lmms_eval \
  --model dummy \
  --tasks vizdoom \
  --agentic_model_server openai \
  --agentic_model_server_args 'model=Qwen/Qwen4-VL-Example,base_url=http://127.0.0.1:8000/v1' \
  --agentic_model_output_parser qwen \
  --agentic_observation_parser vizdoom \
  --agentic_observation_parser_args 'video=true,video_buffer=screen_history,image_buffers=["screen"],include_structured_state=true,include_raw_buffers=true' \
  --agentic_action_parser vizdoom \
  --agentic_action_parser_args 'submit_actions=["SUBMIT"],noop_actions=["NOOP"]' \
  --agentic_loop_worker simple \
  --agentic_loop_worker_args 'multiturn=True,history_turns=6' \
  --agentic_trace_mode full \
  --log_samples
```

`history_turns` is a window over previous user/assistant pairs. Use a small
value for video models because each user turn can contain visual tokens. For
VizDoom, 6 turns is a practical debugging default because each model decision
now covers a short action segment instead of a single simulator tic.

Do not confuse this with environment frame history. VizDoom's `tics_per_action`
controls how long a selected action is held before the model is called again.
VizDoom's `frame_history` controls the frames inside the current
`screen_history` video segment. Multi-turn history controls how many previous
model turns are included around the current segment. The default VizDoom task
defaults to 12 simulator frames per model decision and returns those recent
frames as the next video input.

## Request Shape

`ConfigurableTask` and `ConfigurableMessagesTask` build a
`generate_until_game` request with this argument order:

```text
(
  ctx,
  generation_kwargs,
  doc_to_visual,
  game_env,
  observation_parser,
  action_parser,
  lmms_eval_specific_kwargs,
  doc_id,
  task_name,
  split,
)
```

`run_generate_until_game()` reads model-side runtime configuration from CLI
arguments. It still accepts older 12- and 13-element shapes for compatibility,
and the 10-element shape still contains legacy parser slots, but task YAML
should leave observation/action parsers unset so CLI runtime configuration owns
the model-facing parser contract.

## Model Server Scheduling and Multi-GPU

The loop boundary is intentionally split in two:

- `LoopWorker` and `LoopSession` own environment control flow: reset, observe,
  parse model output, parse task action, step the environment, and record trace.
- `EnvManager` owns environment lifecycle and state lookup.
- `ModelServer` owns inference calls: single-request generation, batched
  generation, backend request pools, and request-level GPU/runtime adapters.
- `LoopManager` owns rollout scheduling: simultaneous environments,
  thread-level concurrency, and loop-level batching across sessions.

`run_generate_until_game()` builds one top-level `ModelServer` per model-server
runtime spec and passes rollout jobs to `LoopManager.run_jobs()`. The loop
manager starts a bounded set of rollout sessions, collects all ready model
request objects at each decision point, calls `ModelServer.generate_batch()`,
and returns the outputs to the sessions. This lets multiple independent
rollouts share one inference backend without putting rollout scheduling in task
YAML, task-specific loop workers, or concrete model-server implementations.

Two concrete model servers ship here:

- `OpenAIModelServer` (registered name `openai`, the default) sends
  OpenAI-compatible Chat Completions requests and uses thread-level concurrency
  for request batches. Use `--agentic_max_parallel_rollouts` to bound
  simultaneous environments and `max_concurrent_requests` to bound concurrent
  HTTP calls.
- `FixedActionModelServer` (registered name `debug`) ignores the observation and
  always returns a single fixed action. It needs no backend, so it is for
  smoke-testing the whole loop — see "Smoke Testing Without a Backend" below.

Other real inference backends are intentionally not carried in this iteration.
If a new backend needs GPU replicas, model ensembles, or another scheduler, keep
that behind a future `ModelServer` implementation rather than adding backend
details to task YAML.

### Smoke Testing Without a Backend

`FixedActionModelServer` lets you exercise the entire pipeline — env reset,
observation parser, action parser, `EnvManager.step`, tracing, and artifact
writing — without serving a model. It always emits the same action text
(default `ATTACK`), which the action parser maps to a real environment action:

```bash
python -m lmms_eval \
  --model dummy \
  --tasks vizdoom \
  --agentic_model_server debug \
  --agentic_model_server_args 'action=ATTACK' \
  --agentic_observation_parser vizdoom \
  --agentic_action_parser vizdoom \
  --gen_kwargs max_game_steps=12 \
  --limit 1 --log_samples --output_path outputs/vizdoom_debug
```

Set `action=` to any valid button (for example `action=MOVE_LEFT`). Because the
server is instant and synchronous, this is also the cheapest way to stress
rollout scheduling: combine it with `--repeats N` and
`--agentic_max_parallel_rollouts K` to run many concurrent environments. Note
that the resulting wall-clock reflects environment stepping and artifact
encoding under the GIL, not the latency a real `ModelServer` would add, so it is
not a proxy for backend throughput.

## Tracing

Each episode is serialized as JSON in the normal lmms-eval sample output when
`--log_samples` is enabled. The default trace mode is compact:

- `raw_model_output`: the exact first text block returned by the model server.
- `model_output`: the normalized first text block after `ModelOutputParser`.
- `action`: the parsed `GameAction`.
- `parse_error`: parser failure text, if any.
- `reward`, `done`, and environment `info`.

This makes it possible to debug whether a failure belongs to model generation,
model-output normalization, task action parsing, or environment dynamics.

For rollout-level debugging, run with `--agentic_trace_mode full`:

```bash
python -m lmms_eval \
  --model huggingface \
  --model_args pretrained=Qwen/Qwen4-VL-Example,trust_remote_code=True \
  --tasks vizdoom \
  --agentic_model_output_parser qwen \
  --agentic_observation_parser vizdoom \
  --agentic_observation_parser_args 'video=true,video_buffer=screen_history,image_buffers=["screen"],include_structured_state=true,include_raw_buffers=true' \
  --agentic_action_parser vizdoom \
  --agentic_action_parser_args 'submit_actions=["SUBMIT"],noop_actions=["NOOP"]' \
  --agentic_trace_mode full \
  --log_samples \
  --output_path outputs/vizdoom_full_trace
```

For non-text payloads, compact trace fields record a safe JSON summary instead
of assuming `first_text()` exists. In `full` mode, each step also records:

- `state_before`: environment id, step index, observation, terminal flag,
  active agents, and metadata before model inference.
- `request`: the request object sent to the model server. `AgentInput` requests
  include content blocks, generation kwargs, metadata, and first text block;
  other objects are logged as safe summaries.
- `raw_output`: the full raw output returned by the model server.
- `output`: the normalized output after `ModelOutputParser`.
- `parsed_action`: parsed action, submit flag, parser error, and parser
  metadata.
- `result`: reward, done flag, environment info, and `state_after`.

Text blocks are preserved. Large or non-JSON media/tensor payloads are logged as
safe summaries such as type, byte length, shape, dtype, image size, or repr, so
the trace shows what happened without dumping full frame buffers into JSON.

When `--output_path` is set, `generate_until_game` also writes human-readable
rollout artifacts under:

```text
<output_path>/agentic_artifacts/<task>_doc<id>_<timestamp>/
```

The artifact directory contains:

- `summary.md`: success, metrics, requested/executed action counts, and a
  per-step table.
- `actions.jsonl`: one normalized action/reward row per step.
- `rollout.mp4`: a gameplay video when the environment exposes screen frames
  such as VizDoom `screen_buffer`. By default it contains **one frame per model
  decision** (the screen at each decision point), so a short episode renders as a
  brief decision-by-decision clip rather than real-time gameplay. The video is
  written at 12 FPS and upscaled 4x for easier viewing at low simulator
  resolutions; set `LMMS_AGENTIC_ARTIFACT_FPS` or `LMMS_AGENTIC_ARTIFACT_SCALE`
  to override.
- `segments/step_NNN.mp4`: one short clip per action, holding the intra-action
  frames captured while that single decision was held for `tics_per_action`
  simulator tics. Only written when action frames are emitted (see below).

By default the intra-action frames captured during `tics_per_action` are used
only as the model's next `screen_history` input and are not retained for the
artifacts, which is why `rollout.mp4` is one-frame-per-decision. Enable
**`emit_action_frames`** on the env (toggle the default VizDoom task with the
`VIZDOOM_EMIT_ACTION_FRAMES=1` env var) to keep them: `rollout.mp4` then becomes
a full-motion video of every captured frame, and per-action `segments/` clips
are written. The raw frame arrays are attached to `StepResult.info` for the
artifact writers only and are stripped from the JSON trace, so `actions.jsonl`
and the sample log stay compact. This roughly multiplies per-episode frame
memory and video-encoding work by `tics_per_action`, so leave it off for large
parallel runs unless you need the playback.

## VizDoom Interfaces

Use `vizdoom` for VizDoom rollouts. The env wraps
`vizdoom.DoomGame` and exposes the main VizDoom interfaces through
`EnvState.observation`. The concrete env implementation lives with the task
under `lmms_eval/tasks/vizdoom_agentic/`; `lmms_eval/agentic` only owns the
generic `EnvManager` interface, loop, factory, and reusable parser layers.

- visual/audio buffers: `screen_buffer`, `depth_buffer`, `labels_buffer`,
  `automap_buffer`, `audio_buffer`, and `notifications_buffer`;
- frame history: `screen_history`, controlled by `frame_history`;
- action cadence: `tics_per_action` controls simulator frames per model
  decision; `capture_action_frames` records the intermediate frames during that
  action for the next `screen_history` segment; `emit_action_frames` (off by
  default, toggled by `VIZDOOM_EMIT_ACTION_FRAMES`) additionally retains those
  frames so the rollout artifacts render full-motion video and per-action
  `segments/` clips instead of one frame per decision;
- game variables from `set_available_game_variables`, plus optional
  `tracked_game_variables` read through `get_game_variable`;
- world metadata: `labels`, `objects`, `sectors`, and multiplayer
  `server_state`;
- action metadata: `available_buttons`, `last_action`, reward, episode time,
  timeout, and player-death state.

The `vizdoom` observation parser can emit both a video block and a
current screen image when selected at runtime:

```bash
--agentic_observation_parser vizdoom \
--agentic_observation_parser_args 'video=true,video_buffer=screen_history,image_buffers=["screen"],include_structured_state=true,include_raw_buffers=true' \
--agentic_action_parser vizdoom
```

With `video: true`, each model request includes a
`ContentBlock(type="video", data=[...frames...])` built from `screen_history`.
It can also include `ContentBlock(type="image")` for the latest screen and
structured blocks for non-visual state. The OpenAI-compatible model server
preserves media types when converting `AgentInput` into model messages, so video
blocks are not silently downgraded to images.

By default, `vizdoom` asks the model to call a small set of
VizDoom action skills instead of emitting raw JSON. The preferred Qwen-style
call is:

```text
<tool_call>
<function=press_buttons>
<parameter=buttons>MOVE_LEFT, ATTACK</parameter>
<parameter=tics>1</parameter>
</function>
</tool_call>
```

The action parser also accepts `press_buttons(ATTACK)`, `noop()`, and
`submit()` for runtimes that do not emit structured tool calls. JSON action
payloads are still parsed as a compatibility fallback, but new VizDoom prompts
should use skill calls.

The built-in `vizdoom` task is a minimal example around
VizDoom's `basic.cfg`. It should be treated as a starting point for real
scenarios, not as the only supported scenario. Other bundled VizDoom configs
such as `deadly_corridor.cfg`, `defend_the_center.cfg`, or
`health_gathering.cfg` can be selected by changing task-side `config_path` and
the available buttons/variables.

## Common Extension Scenarios

This section gives the intended extension path for common changes. The main
rule is: add the smallest component that matches the new behavior. Do not put
model-family behavior into task code, and do not put task action semantics into
model backend code.

### 1. Adding a New VLM Endpoint

Example: a future `Qwen4-VL` checkpoint appears on HuggingFace.

Prefer this path:

1. Serve the checkpoint through an OpenAI-compatible Chat Completions endpoint.
2. Keep the game task unchanged.
3. Choose the endpoint, generation settings, and parser contracts at runtime.

```bash
python -m lmms_eval \
  --model dummy \
  --tasks vizdoom \
  --agentic_model_server openai \
  --agentic_model_server_args 'model=Qwen/Qwen4-VL-Example,base_url=http://127.0.0.1:8000/v1' \
  --agentic_model_output_parser qwen \
  --agentic_observation_parser vizdoom \
  --agentic_observation_parser_args 'video=true,video_buffer=screen_history,image_buffers=["screen"],include_structured_state=true,include_raw_buffers=true' \
  --agentic_action_parser vizdoom \
  --agentic_action_parser_args 'submit_actions=["SUBMIT"],noop_actions=["NOOP"]' \
  --gen_kwargs max_new_tokens=512,temperature=0,max_game_steps=24 \
  --limit 1 \
  --log_samples \
  --output_path outputs/qwen4_openai_smoke
```

Use this path when:

- The model can be exposed through an OpenAI-compatible Chat Completions API.
- The environment state/action schema is unchanged.
- The model output format is already handled by an existing
  `ModelOutputParser`, such as `qwen` or `identity`.

Add a new `ModelOutputParser` only if the new model family introduces a reusable
output wrapper that existing parsers do not handle. For example, if `Qwen4`
keeps the same thinking/tool-call format as current Qwen models, reuse
`--agentic_model_output_parser qwen`. If it introduces a different wrapper,
add `Qwen4ModelOutputParser` and register it under a new name.

Add a new concrete `ModelServer` only if an OpenAI-compatible endpoint cannot
represent the model's native interface cleanly.

Inspect `raw_model_output`, `model_output`, `action`, and `parse_error` in the
sample log before trusting aggregate metrics.

### 2. Adding a New Non-VLM Architecture

Example: a TML interaction model consumes structured state, latent tokens, or a
policy interface instead of chat-style image/text messages.

Do not force it into the VLM path. Implement parsers and a `ModelServer` around
the model's native request/output objects:

```python
class TmlModelServer(ModelServer):
    def generate(self, request: TmlRequest) -> TmlPolicyOutput:
        # Run the native TML runtime directly.
        ...
```

Then pass the backend by import path:

```bash
--agentic_model_server my_project.agentic:TmlModelServer
```

Then choose parser layers based on the output:

- If the model emits valid task actions directly, use
  `--agentic_model_output_parser identity`.
- If the model emits a reusable TML wrapper, add a `TmlModelOutputParser`.
- If the model emits non-text policy outputs, add an `ActionParser` that reads
  those native objects instead of relying on `first_text()`.

The runtime observation parser may emit a native object directly:

```python
class VlaObservationParser(ObservationParser):
    def parse(self, state, ctx):
        return {
            "image": image_tensor,
            "state": proprioception_tensor,
            "instruction": state.observation.get("instruction"),
        }
```

This keeps the loop protocol stable while allowing non-text models to bypass
chat formatting entirely.

### 3. Adding a New Game

Example: adding a CSGO-style task.

Start with a model-agnostic base task. The base task should define the game,
observations, actions, and metrics, but not a specific model family.

Implement:

1. An `EnvManager` wrapper around the simulator.
2. Optional observation/action parser implementations for common model
   conventions.
3. `process_results` and metric aggregation functions.
4. A YAML task that wires the EnvManager and task data/metrics together.

The task YAML stays model-agnostic:

```yaml
task: csgo_agentic
output_type: generate_until_game
game_env: !function utils.csgo_env_manager
```

For a simple discrete action game, select `ActionNameParser` at runtime:

```bash
python -m lmms_eval \
  --tasks csgo_agentic \
  --agentic_observation_parser csgo_video_text \
  --agentic_action_parser action_name \
  --agentic_action_parser_args 'actions=["MOVE_FORWARD","STRAFE_LEFT","STRAFE_RIGHT","TURN_LEFT","TURN_RIGHT","FIRE","RELOAD","NOOP"],submit_actions=["SUBMIT"]'
```

For richer control, add a custom `ActionParser`. For example, a CSGO action may
need buttons plus view deltas:

```python
GameAction(
    type="csgo_control",
    data={
        "buttons": ["MOVE_FORWARD", "FIRE"],
        "view_delta": [12.0, -3.0],
    },
)
```

In that case, the parser implementation can know CSGO control semantics, but
the selected parser still belongs in CLI/runtime configuration. Qwen, LLaVA, or
TML output cleanup should stay in `ModelOutputParser`.

If a game needs task-side variants, those variants should include the base game
YAML and override only task-side fields. Model-side choices stay in CLI/runtime
args:

```bash
python -m lmms_eval \
  --model huggingface \
  --model_args pretrained=Qwen/Qwen3.5-9B,trust_remote_code=True \
  --tasks csgo_agentic \
  --agentic_model_output_parser qwen \
  --agentic_observation_parser csgo_video_text \
  --agentic_action_parser csgo_qwen_action
```

### 4. Adding a New Agentic Pipeline

Example: multiple models collaborate to play one game, such as a vision model,
planner, and critic fused into one action policy.

Choose the integration point based on what changes.

Use a composite `ModelServer` when the pipeline still maps one request object
to one final model output object:

```text
model request object
  -> vision model
  -> planner model
  -> critic/reranker
  -> final model output object
```

The composite server can own multiple submodels internally and expose one
registered backend:

```python
class FusionModelServer(ModelServer):
    def __init__(self, planner, critic, vision=None, **kwargs):
        ...

    def generate(self, request):
        ...
```

The task YAML can keep the standard loop:

```bash
python -m lmms_eval \
  --model dummy \
  --tasks vizdoom \
  --agentic_model_server my_project.agentic:FusionModelServer \
  --agentic_model_server_args planner=Qwen/Qwen3.5-9B,critic=another/model \
  --agentic_model_output_parser qwen \
  --agentic_observation_parser vizdoom \
  --agentic_action_parser vizdoom \
  --agentic_loop_worker simple
```

Use a custom `LoopWorker` when the pipeline changes the environment interaction
pattern. Examples:

- Multiple agents act in the same environment step.
- One game step requires model A to inspect state, model B to propose actions,
  and model C to request extra simulator probes before acting.
- The pipeline needs memory, planning state, debate, voting, or retries across
  steps.

Implement the worker:

```python
class FusionLoopWorker(LoopWorker):
    def run(self, doc, seed=None, agent_id="agent") -> EpisodeResult:
        ...
```

Then select it at runtime:

```bash
python -m lmms_eval \
  --model dummy \
  --tasks vizdoom \
  --agentic_loop_worker my_project.agentic:FusionLoopWorker \
  --agentic_model_server my_project.agentic:FusionModelServer \
  --agentic_model_output_parser identity \
  --agentic_observation_parser vizdoom \
  --agentic_action_parser vizdoom
```

With the current request contract, the runner builds one top-level
`model_server` and passes it to the loop worker. If a pipeline needs multiple
independently configured model servers from CLI/runtime args, prefer
implementing a composite top-level `ModelServer` first. Extend the request
shape only when the pipeline cannot reasonably be represented as either a
composite server or a custom worker.

## Extension Guide

### Adding a Model Backend

Implement `ModelServer.generate()` or `generate_batch()`:

```python
class MyModelServer(ModelServer):
    def generate(self, request):
        ...
```

Select it with an import path:

```bash
--agentic_model_server my_project.agentic:MyModelServer
```

For code-driven use, create a modified factory instead of mutating global state:

```python
factory = DEFAULT_AGENTIC_FACTORY.with_components(
    model_servers={"my_backend": MyModelServer},
)
```

The backend should convert `AgentInput` typed content into the backend's native
request format and return `AgentOutput` without task-specific parsing when it
is a chat/VLM backend. Native policy or VLA backends may instead accept and
return tensors, dataclasses, dicts, or other Python objects. Keep task-specific
action parsing outside the backend either way.
If the backend owns resources such as GPU replicas, worker queues, remote
endpoints, or model ensembles, keep that request-level scheduling inside
`generate()` or `generate_batch()`. Rollout scheduling remains in `LoopManager`;
use a custom `LoopWorker` when the environment interaction pattern itself
changes.

### Adding a Model Output Parser

Implement `ModelOutputParser` when a model family has reusable output wrappers:

```python
class MyModelOutputParser(ModelOutputParser):
    def parse(self, output, ctx):
        ...
```

Select it with an import path or add it to a local factory:

```bash
--agentic_model_output_parser my_project.agentic:MyModelOutputParser
```

Place it under `lmms_eval/agentic/parsers/model_output/`.
Keep this parser model-facing only. It may add structured metadata, but it
should not decide whether `MOVE_FORWARD` or `ATTACK` is a valid task action.

### Adding a Task Action Parser

Use `ActionNameParser` for simple finite action sets. Add a custom
`ActionParser` when the task action schema is structured, multi-agent, spatial,
or otherwise richer than a single action name. Place action parsers under
`lmms_eval/agentic/parsers/action/`. For model/runtime-specific task parsers,
use explicit names such as `vizdoom.py`.

### Adding a Game Task

Implement:

- `EnvManager` for state transitions and metrics.
- Optional `ObservationParser` / `ActionParser` implementations for common
  runtime conventions.
- `process_results` and metric aggregation functions.
- YAML wiring for `game_env`, task data, and metrics. Parser selection happens
  through CLI/runtime args.

## Notes on Non-text Models

`AgentInput` and `AgentOutput` are optional chat-style envelopes with typed
`ContentBlock` lists. They are not the parser protocol itself. Text is one block
type; image, video, audio, tensor, embedding, latent, logits, or model-specific
policy outputs can be represented by blocks when an envelope is convenient.
They can also be passed as native objects directly.

Parser implementations should therefore avoid assuming that all useful data is
in `first_text()`. Text parsers can use it as a convenience, but non-text
models should be able to communicate through tensors, blocks, metadata, or
custom runtime objects without changing the loop contract.
