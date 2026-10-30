# Evaluating a policy in an interactive environment

ViZDoom and MiniGrid use `generate_until_agentic` with the same `--model` and
`--model_args` as other tasks. The evaluator calls the model's `generate_until`
method once per decision. Environments, prompts, action decoding, and episode
metrics live in each task directory.

For example, start an OpenAI-compatible vision model server, then run:

```bash
uv run python -m lmms_eval --model openai \
  --model_args model_version=Qwen/Qwen3.6-27B,base_url=http://127.0.0.1:8000/v1 \
  --tasks minigrid --limit 1 --batch_size 1 --log_samples \
  --output_path artifacts/minigrid
```

Install simulator dependencies with `uv pip install gymnasium minigrid` for
MiniGrid or `uv pip install vizdoom` for ViZDoom. Both imports happen only when
an episode begins. The `vizdoom` task uses `pufanyi/ViZDoom`; MiniGrid includes
three local scenario rows.

## Message callback

A chat task can provide `doc_to_messages` for the existing
`generate_until_agentic` output type. Calls without `round_idx` build the task
context and must have no simulator side effects. At execution, round zero
creates the episode and supplies its first observation. Later calls receive
all model outputs and the previous round's task-owned state:

```python
def doc_to_messages(doc, *, round_idx=None, previous_output=None,
                    previous_round_info=None, lmms_eval_specific_kwargs=None):
    # A normal message list when round_idx is None.
    # At runtime: (messages, terminal, updated_outputs, info).
    ...
```

The callback owns the full message list, including any conversation history.
Messages use the normal lmms-eval text/image/video schema, so model backends
perform their usual media encoding. A task can append its final JSON metric
summary to `updated_outputs` when the episode ends.

The optional `info` dictionary may carry:

- `close`: a zero-argument callback that the evaluator calls on every exit,
  including model failures and exhausted step budgets.
- `finalize`: a zero-argument callback returning the final response when
  `max_agentic_steps` is reached.
- `state`: a JSON-safe summary included by `--agentic_trace_mode full`.

Live simulator handles belong outside `state`. An initialization failure must
close any simulator that the callback created before raising, because the
evaluator has not received its cleanup callback yet.

Tasks with live simulators set `stateful: true`. Their model responses bypass
`--use_cache`; an episode failure becomes a failed sample with `env_error=1`
and the next episode can run. Stateless tasks keep their existing response
cache behavior. `reasoning_tags: none` protects terminal metric JSON that may
contain raw model replies.

## Simulator observations

ViZDoom uses a 320 by 240 first-person RGB view with the normal HUD. Privileged
state such as exact reward or game variables is kept out of the model prompt.
Each action advances five simulator tics; the next decision receives the five
frames produced by that action. The task accepts button names, JSON button
lists, and `press_buttons` tool calls.

MiniGrid supplies its RGB grid view, mission, facing direction, and carried
object. Its task-local decoder accepts discrete action names and common
aliases. Dataset rows select the scenario, seed, and simulator step limit.

Rollouts run serially through the existing evaluator. Use `--limit` to bound
initial runs and `--gen_kwargs max_agentic_steps=...` to cap decisions. These
tasks report simulator success and invalid actions; deterministic action
responses can validate the integration but do not measure model quality.
