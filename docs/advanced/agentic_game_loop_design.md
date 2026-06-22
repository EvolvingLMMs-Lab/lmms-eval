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
  tool-call loop. It is for environment-step loops with explicit `GameEnv`
  state transitions.
- It does not require task code to know how inference is served. The current
  concrete `ModelServer` targets OpenAI-compatible HTTP endpoints; future
  implementations may wrap other serving or in-process runtimes behind the same
  base protocol.
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
  -> GameEnv.step(...)
  -> EnvState
```

The loop repeats until the environment is terminal or `max_game_steps` is
reached.

## Components

| Component | Direction | Responsibility |
|-----------|-----------|----------------|
| `GameEnv` | task state | Owns reset/step logic, terminal state, rewards, and task metrics. |
| `ObservationParser` | `EnvState -> Any` | Turns environment state into the model-facing request object. Chat/VLM tasks commonly return `AgentInput`; policy or VLA tasks can return tensors, dataclasses, dicts, or runtime-native objects. |
| `ModelServer` | `Any -> Any` | Runs inference. It hides backend details such as HTTP APIs, request batching, generation parameters, async-friendly calls, and native tensor/runtime adapters. |
| `ModelOutputParser` | raw `Any -> Any` | Normalizes model-family or runtime output without knowing the task action schema. Examples: strip `<think>...</think>` from an `AgentOutput`, extract Qwen XML tool calls into metadata, or unwrap a policy output dataclass. |
| `ActionParser` | normalized `Any -> ParsedAction` | Parses one task action from normalized model output. It knows valid task actions but not the model backend. |
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
    vizdoom_vllm_parser.py
  action/
    action_name.py
    vizdoom_vllm_parser.py
```

Each registry is separate, so the same parser name can be used for different
roles. For example, `vizdoom_vllm_parser` can be registered as both an
`ObservationParser` and an `ActionParser`:

```yaml
observation_parser:
  name: vizdoom_vllm_parser

action_parser:
  name: vizdoom_vllm_parser
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
tightly coupled, use an explicit combined name such as
`vizdoom_vllm_parser`. The implementation can still reuse shared helper
classes, but the registry name should make the actual contract obvious.

The task YAML defines task-side observation/action parsers:

```yaml
action_parser:
  name: vizdoom_vllm_parser
  submit_actions:
    - SUBMIT
  noop_actions:
    - NOOP
```

The runtime command chooses the model-side parser:

```bash
python -m lmms_eval \
  --model huggingface \
  --model_args pretrained=Qwen/Qwen4-VL-Example,trust_remote_code=True \
  --tasks vizdoom_native_agentic \
  --agentic_model_output_parser qwen
```

The Qwen parser is reusable across any task that needs Qwen-style output
normalization. The `vizdoom_vllm_parser` action parser is task-side and maps
skill calls or common action text into native VizDoom buttons.

## YAML Contract

A `generate_until_game` task config provides these components:

```yaml
output_type: generate_until_game
game_env:
  name: vizdoom_native
  config_path: basic.cfg
  screen_resolution: RES_320X240
  frame_history: 12
  tics_per_action: 12
observation_parser:
  name: vizdoom_vllm_parser
  video: true
  video_buffer: screen_history
action_parser:
  name: vizdoom_vllm_parser
  submit_actions: [SUBMIT]
  noop_actions: [NOOP]
generation_kwargs:
  max_game_steps: 64
  max_new_tokens: 64
  temperature: 0
```

Model-side runtime choices stay in the command line:

```bash
python -m lmms_eval \
  --model dummy \
  --tasks vizdoom_native_agentic \
  --agentic_model_server openai \
  --agentic_model_server_args 'model=Qwen/Qwen4-VL-Example,base_url=http://127.0.0.1:8000/v1,max_concurrent_requests=4' \
  --agentic_max_parallel_rollouts 4 \
  --agentic_model_output_parser qwen \
  --agentic_loop_worker simple \
  --gen_kwargs max_new_tokens=512,temperature=0,max_game_steps=24
```

`agentic_model_server` currently defaults to `openai`, an OpenAI-compatible
Chat Completions endpoint. `agentic_model_output_parser` defaults to
`identity`. `agentic_loop_worker` defaults to `simple`.

This keeps task action semantics in the task config and model-family formatting
in runtime/model configuration.

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
  --tasks vizdoom_native_agentic \
  --agentic_model_server openai \
  --agentic_model_server_args 'model=Qwen/Qwen4-VL-Example,base_url=http://127.0.0.1:8000/v1' \
  --agentic_model_output_parser qwen \
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
model turns are included around the current segment. The native VizDoom task
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
but task construction should not emit model-side fields.

## Model Server Scheduling and Multi-GPU

The loop boundary is intentionally split in two:

- `LoopWorker` and `LoopSession` own environment control flow: reset, observe,
  parse model output, parse task action, step the environment, and record trace.
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

The only concrete model server currently shipped here is `OpenAIModelServer`.
It sends OpenAI-compatible Chat Completions requests and uses thread-level
concurrency for request batches. Use `--agentic_max_parallel_rollouts` to bound
simultaneous environments and `max_concurrent_requests` to bound concurrent HTTP
calls.

Other concrete backends are intentionally not carried in this iteration. If a
new backend needs GPU replicas, model ensembles, or another scheduler, keep that
behind a future `ModelServer` implementation rather than adding backend details
to task YAML.

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
  --tasks vizdoom_native_agentic \
  --agentic_model_output_parser qwen \
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
  such as VizDoom `screen_buffer`. The default artifact video is written at 12
  FPS, matching the native VizDoom task's default decision segment length. It is
  also upscaled 4x for easier viewing when the task uses low simulator
  resolutions. Set
  `LMMS_AGENTIC_ARTIFACT_FPS` or `LMMS_AGENTIC_ARTIFACT_SCALE` to override these
  defaults.

## Native VizDoom Interfaces

Use `vizdoom_native_agentic` for VizDoom rollouts. The native env wraps
`vizdoom.DoomGame` and exposes the main VizDoom interfaces through
`EnvState.observation`. The concrete env implementation lives with the task
under `lmms_eval/tasks/vizdoom_agentic/`; `lmms_eval/agentic` only owns the
generic `GameEnv` interface, loop, registry, and reusable parser layers.

- visual/audio buffers: `screen_buffer`, `depth_buffer`, `labels_buffer`,
  `automap_buffer`, `audio_buffer`, and `notifications_buffer`;
- frame history: `screen_history`, controlled by `frame_history`;
- action cadence: `tics_per_action` controls simulator frames per model
  decision, and `capture_action_frames` records intermediate frames during that
  action for the next `screen_history` segment;
- game variables from `set_available_game_variables`, plus optional
  `tracked_game_variables` read through `get_game_variable`;
- world metadata: `labels`, `objects`, `sectors`, and multiplayer
  `server_state`;
- action metadata: `available_buttons`, `last_action`, reward, episode time,
  timeout, and player-death state.

The `vizdoom_vllm_parser` observation parser can emit both a video block and a
current screen image:

```yaml
game_env:
  name: vizdoom_native
  config_path: basic.cfg
  screen_resolution: RES_320X240
  screen_format: RGB24
  available_buttons: [MOVE_LEFT, MOVE_RIGHT, ATTACK]
  available_game_variables: [AMMO2, HEALTH, KILLCOUNT]
  depth_buffer: true
  labels_buffer: true
  automap_buffer: true
  objects_info: true
  sectors_info: true
  notifications_buffer: true
  frame_history: 4
  window_visible: false

observation_parser:
  name: vizdoom_vllm_parser
  video: true
  video_buffer: screen_history
  image_buffers: [screen]
  include_structured_state: true
  include_raw_buffers: true

action_parser:
  name: vizdoom_vllm_parser
```

With `video: true`, each model request includes a
`ContentBlock(type="video", data=[...frames...])` built from `screen_history`.
It can also include `ContentBlock(type="image")` for the latest screen and
structured blocks for non-visual state. The OpenAI-compatible model server
preserves media types when converting `AgentInput` into model messages, so video
blocks are not silently downgraded to images.

By default, `vizdoom_vllm_parser` asks the model to call a small set of
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

The built-in `vizdoom_native_agentic` task is a minimal native example around
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
3. Choose the endpoint, generation settings, and model-output parser at runtime.

```bash
python -m lmms_eval \
  --model dummy \
  --tasks vizdoom_native_agentic \
  --agentic_model_server openai \
  --agentic_model_server_args 'model=Qwen/Qwen4-VL-Example,base_url=http://127.0.0.1:8000/v1' \
  --agentic_model_output_parser qwen \
  --gen_kwargs max_new_tokens=512,temperature=0,max_game_steps=24 \
  --limit 1 \
  --log_samples \
  --output_path outputs/qwen4_openai_smoke
```

Use this path when:

- The model can be exposed through an OpenAI-compatible Chat Completions API.
- The task observation and action schema are unchanged.
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

register_model_server("tml", TmlModelServer)
```

Then choose parser layers based on the output:

- If the model emits valid task actions directly, use
  `--agentic_model_output_parser identity`.
- If the model emits a reusable TML wrapper, add a `TmlModelOutputParser`.
- If the model emits non-text policy outputs, add an `ActionParser` that reads
  those native objects instead of relying on `first_text()`.

The task observation parser may emit a native object directly:

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

1. A `GameEnv` wrapper around the simulator.
2. An `ObservationParser` that converts simulator state into a model request
   object, commonly `AgentInput` for chat/VLM tasks.
3. An `ActionParser` or `ActionNameParser` config for the task action schema.
4. `process_results` and metric aggregation functions.
5. A YAML task that wires these components together.

For a simple discrete action game, use `ActionNameParser`:

```yaml
task: csgo_agentic
output_type: generate_until_game
game_env: csgo_env
observation_parser: csgo_video_text
action_parser:
  name: action_name
  actions:
    - MOVE_FORWARD
    - STRAFE_LEFT
    - STRAFE_RIGHT
    - TURN_LEFT
    - TURN_RIGHT
    - FIRE
    - RELOAD
    - NOOP
  submit_actions:
    - SUBMIT
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

In that case, the parser belongs to the task action layer because it knows CSGO
control semantics. Qwen, LLaVA, or TML output cleanup should still stay in
`ModelOutputParser`.

If a game needs task-side variants, those variants should include the base game
YAML and override only task-side fields. Model-side choices stay in CLI/runtime
args:

```bash
python -m lmms_eval \
  --model huggingface \
  --model_args pretrained=Qwen/Qwen3.5-9B,trust_remote_code=True \
  --tasks csgo_agentic \
  --agentic_model_output_parser qwen
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

register_model_server("fusion", FusionModelServer)
```

The task YAML can keep the standard loop:

```bash
python -m lmms_eval \
  --model dummy \
  --tasks vizdoom_native_agentic \
  --agentic_model_server fusion \
  --agentic_model_server_args planner=Qwen/Qwen3.5-9B,critic=another/model \
  --agentic_model_output_parser qwen \
  --agentic_loop_worker simple
```

Use a custom `LoopWorker` when the pipeline changes the environment interaction
pattern. Examples:

- Multiple agents act in the same environment step.
- One game step requires model A to inspect state, model B to propose actions,
  and model C to request extra simulator probes before acting.
- The pipeline needs memory, planning state, debate, voting, or retries across
  steps.

Register the worker:

```python
class FusionLoopWorker(LoopWorker):
    def run(self, doc, seed=None, agent_id="agent") -> EpisodeResult:
        ...

register_loop_worker("fusion", FusionLoopWorker)
```

Then select it at runtime:

```bash
python -m lmms_eval \
  --model dummy \
  --tasks vizdoom_native_agentic \
  --agentic_loop_worker fusion \
  --agentic_model_server fusion \
  --agentic_model_output_parser identity
```

With the current request contract, the runner builds one top-level
`model_server` and passes it to the loop worker. If a pipeline needs multiple
independently configured model servers from CLI/runtime args, prefer
implementing a composite top-level `ModelServer` first. Extend the request
shape only when the pipeline cannot reasonably be represented as either a
composite server or a custom worker.

## Extension Guide

### Adding a Model Backend

Implement `ModelServer.generate()` or `generate_batch()` and register it:

```python
register_model_server("my_backend", MyModelServer)
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

register_model_output_parser("my_model_family", MyModelOutputParser)
```

Place it under `lmms_eval/agentic/parsers/model_output/`.
Keep this parser model-facing only. It may add structured metadata, but it
should not decide whether `MOVE_FORWARD` or `ATTACK` is a valid task action.

### Adding a Task Action Parser

Use `ActionNameParser` for simple finite action sets. Add a custom
`ActionParser` when the task action schema is structured, multi-agent, spatial,
or otherwise richer than a single action name. Place action parsers under
`lmms_eval/agentic/parsers/action/`. For model/runtime-specific task parsers,
use explicit names such as `vizdoom_vllm_parser.py`.

### Adding a Game Task

Implement:

- `GameEnv` for state transitions and metrics.
- `ObservationParser` for task-specific observations.
- `process_results` and metric aggregation functions.
- YAML wiring for `game_env`, `observation_parser`, and `action_parser`.

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
