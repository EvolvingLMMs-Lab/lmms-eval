# Agentic Task Contract

This document defines the task-level contract consumed by lmms-eval agentic
rollout workers and external training runtimes such as lmms-engine.

The goal is that a new agentic environment is implemented once in lmms-eval,
then reused by evaluation and online RL without adding environment-specific
code to the training engine.

## YAML Fields

An agentic task YAML should provide these fields:

```yaml
task: my_agentic_task
output_type: generate_until_game

doc_to_visual: !function utils.doc_to_visual
doc_to_text: !function utils.doc_to_text
doc_to_target: !function utils.doc_to_target

game_env: !function utils.env_manager
observation_parser:
  name: my_observation_parser
action_parser:
  name: my_action_parser

generation_kwargs:
  max_new_tokens: 64
  max_game_steps: 64

process_results: !function utils.process_results
metric_list:
  - metric: success
    aggregation: !function utils.aggregate_success
    higher_is_better: true
```

`game_env` must resolve to an `EnvManager`. Parser specs are resolved through
`AgenticFactory` and may be a built-in name, import path, callable, or dict with
`name` or `factory`.

`generation_kwargs.max_game_steps` is the default rollout horizon. Runtimes may
override it per run.

## Component Boundaries

Agentic components use open-ended payloads:

- `EnvManager.reset(doc, seed)` returns `EnvState`.
- `ObservationParser.parse(EnvState, ParserContext)` returns any model request
  payload, commonly `AgentInput`.
- `ModelServer.generate(...)` returns raw model output.
- `ModelOutputParser.parse(...)` returns normalized output, commonly
  `AgentOutput`.
- `ActionParser.parse(...)` returns `ParsedAction`.
- `EnvManager.step(GameAction)` returns `StepResult`.

Payload blocks are not restricted to text, image, or video. Tasks may use
tensors, embeddings, latent states, logits, or other structured objects as long
as their parser/model-server pair agrees on the schema.

## Data Contract

Evaluation data can be declared by the task YAML using lmms-eval's normal
dataset fields.

Training runtimes may inject their own docs instead of using the eval split.
Those docs should follow the same task-owned schema expected by `game_env`,
`doc_to_text`, and parsers. For example, lmms-engine uses:

```yaml
rl_config:
  task:
    task_name: my_agentic_task
    data_path: path/to/train.jsonl
```

The training runtime locates the lmms-eval task YAML by `task_name` or
`task_yaml`, full-loads the component specs, and replaces only the docs. It
should not add environment-specific code.

Docs should be JSON-serializable when they need to cross process boundaries.
Large media or state references should be represented by paths, URIs, or
task-owned lightweight descriptors.

## Metrics And Rewards

`EpisodeResult.metrics` should contain stable scalar metrics. RL runtimes can
select a reward metric by name, for example:

```yaml
algorithm:
  reward_key: success_reward
```

If no reward metric is selected, runtimes may fall back to summing step rewards.
Task authors should document the intended reward metric when an environment is
used for training.

## Checklist For New Agentic Tasks

1. Add an lmms-eval task YAML with `output_type: generate_until_game`.
2. Implement `game_env` in lmms-eval and return an `EnvManager`.
3. Provide observation/action parser specs that `AgenticFactory` can resolve.
4. Keep doc schema task-owned and JSON-serializable for distributed rollout.
5. Emit scalar metrics in `EpisodeResult.metrics`, including the recommended
   reward metric if the task is intended for RL.
6. Verify the task with `SyncEpisodeRolloutWorker` before connecting it to a
   training runtime.
