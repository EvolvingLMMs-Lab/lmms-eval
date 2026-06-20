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
- It does not require a long-running HTTP server. A `ModelServer` may wrap an
  in-process model, the existing lmms-eval model object, direct vLLM Python API,
  or an external endpoint.
- Task code should not hard-code model-family parsing rules.

## Pipeline

Each game request is executed by a `LoopWorker`:

```text
EnvState
  -> ObservationParser
  -> AgentInput
  -> ModelServer
  -> raw AgentOutput
  -> ModelOutputParser
  -> normalized AgentOutput
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
| `ObservationParser` | `EnvState -> AgentInput` | Turns environment state into model-facing inputs. This may include text, image, video, tensors, embeddings, or other `ContentBlock` types. |
| `ModelServer` | `AgentInput -> AgentOutput` | Runs inference. It hides backend details such as lmms-eval model calls, vLLM, HTTP APIs, batching, and generation parameters. |
| `ModelOutputParser` | raw `AgentOutput -> AgentOutput` | Normalizes model-family output without knowing the task action schema. Examples: strip `<think>...</think>`, extract Qwen XML tool calls into metadata, normalize assistant wrappers. |
| `ActionParser` | normalized `AgentOutput -> ParsedAction` | Parses one task action from normalized model output. It knows valid task actions but not the model backend. |
| `LoopWorker` | orchestration | Wires components together and records per-step traces. |

## Parser Layering

The parser split prevents combinatorial growth.

Bad:

```text
Qwen + VizDoom parser
Qwen + Minecraft parser
LLaVA + VizDoom parser
LLaVA + Minecraft parser
...
```

Good:

```text
QwenModelOutputParser
LlavaModelOutputParser
...

ActionNameParser(actions=[...])
MinecraftActionParser(...)
...
```

Then YAML combines the pieces:

```yaml
model_output_parser: qwen
action_parser:
  name: action_name
  actions:
    - MOVE_FORWARD
    - TURN_LEFT
    - TURN_RIGHT
    - ATTACK
    - USE
    - NOOP
  submit_actions:
    - SUBMIT
```

The Qwen parser is reusable across any task that needs Qwen-style output
normalization. The `action_name` parser is reusable across any task that exposes
a finite set of action names.

## YAML Contract

A `generate_until_game` task config provides these components:

```yaml
output_type: generate_until_game
model_server: lmms
loop_worker: simple
game_env: vizdoom_grid
observation_parser: vizdoom_text
model_output_parser: identity
action_parser:
  name: action_name
  actions: [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, ATTACK, USE, NOOP]
  submit_actions: [SUBMIT]
generation_kwargs:
  max_game_steps: 6
  max_new_tokens: 64
  temperature: 0
```

Model-specific variants override only the model-related pieces:

```yaml
include: vizdoom_agentic.yaml
task: vizdoom_agentic_qwen35_vllm
model_output_parser: qwen
model_server:
  name: vllm
  model: /path/to/Qwen3.5-9B
  trust_remote_code: true
  chat_template_kwargs:
    enable_thinking: false
  gdn_prefill_backend: triton
generation_kwargs:
  max_new_tokens: 8
  temperature: 0
  max_game_steps: 6
```

This keeps task action semantics in the base task and model-family formatting in
the model-specific override.

## Request Shape

`ConfigurableTask` and `ConfigurableMessagesTask` build a
`generate_until_game` request with this argument order:

```text
(
  ctx,
  generation_kwargs,
  doc_to_visual,
  model_server,
  loop_worker,
  game_env,
  observation_parser,
  model_output_parser,
  action_parser,
  lmms_eval_specific_kwargs,
  doc_id,
  task_name,
  split,
)
```

`run_generate_until_game()` still accepts older 10- and 12-element shapes and
defaults `model_output_parser` to `identity` for compatibility.

## Tracing

Each episode is serialized as JSON. Per step, the runner records:

- `raw_model_output`: the exact first text block returned by the model server.
- `model_output`: the normalized first text block after `ModelOutputParser`.
- `action`: the parsed `GameAction`.
- `parse_error`: parser failure text, if any.
- `reward`, `done`, and environment `info`.

This makes it possible to debug whether a failure belongs to model generation,
model-output normalization, task action parsing, or environment dynamics.

## Extension Guide

### Adding a Model Backend

Implement `ModelServer.generate()` or `generate_batch()` and register it:

```python
register_model_server("my_backend", MyModelServer)
```

The backend should convert `AgentInput` typed content into the backend's native
request format and return `AgentOutput` without task-specific parsing.

### Adding a Model Output Parser

Implement `ModelOutputParser` when a model family has reusable output wrappers:

```python
class MyModelOutputParser(ModelOutputParser):
    def parse(self, output, state, agent_id=None):
        ...

register_model_output_parser("my_model_family", MyModelOutputParser)
```

Keep this parser model-facing only. It may add structured metadata, but it
should not decide whether `MOVE_FORWARD` or `ATTACK` is a valid task action.

### Adding a Task Action Parser

Use `ActionNameParser` for simple finite action sets. Add a custom
`ActionParser` when the task action schema is structured, multi-agent, spatial,
or otherwise richer than a single action name.

### Adding a Game Task

Implement:

- `GameEnv` for state transitions and metrics.
- `ObservationParser` for task-specific observations.
- `process_results` and metric aggregation functions.
- YAML wiring for `game_env`, `observation_parser`, `model_output_parser`, and
  `action_parser`.

## Notes on Non-text Models

`AgentInput` and `AgentOutput` are lists of typed `ContentBlock` objects, not
just strings. Text is one block type. Image, video, audio, tensor, embedding,
latent, logits, or model-specific policy outputs can be represented by other
block types.

Parser implementations should therefore avoid assuming that all useful data is
in `first_text()`. Text parsers can use it as a convenience, but non-text
models should be able to communicate through blocks and metadata without
changing the loop contract.
