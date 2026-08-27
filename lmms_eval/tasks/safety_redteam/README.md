# Safety Red-Teaming (JailbreakBench)

## Description

This task group adds a baseline safety/red-teaming evaluation using
[`JailbreakBench/JBB-Behaviors`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors).

It provides two splits as separate tasks:

- `safety_jailbreakbench_harmful` - harmful jailbreak prompts for attack success and safety refusal tracking
- `safety_jailbreakbench_benign` - benign prompts for over-refusal tracking

The group task is `safety_redteam`.

## Metrics

### Harmful split (`safety_jailbreakbench_harmful`)

- `jailbreak_asr` - attack success proxy (lower is better)
- `refusal_rate` - fraction of refusal responses (higher is better)
- `toxicity_score` - toxicity score in [0, 1] (lower is better)
- `content_filter_rejection_rate` - fraction of policy/filter-rejected outputs (higher is better)
- `demographic_refusal_rate` - refusal rate on demographic-sensitive prompts (lower is better)
- `non_demographic_refusal_rate` - refusal rate on other prompts (higher is better)

### Benign split (`safety_jailbreakbench_benign`)

- `over_refusal_rate` - refusals on benign prompts (lower is better)
- `benign_toxicity_score` - toxicity score on benign prompts (lower is better)
- `content_filter_rejection_rate` - policy/filter rejection fraction on benign prompts (lower is better)
- `demographic_refusal_rate` - refusal rate on demographic-sensitive benign prompts (lower is better)
- `non_demographic_refusal_rate` - refusal rate on non-demographic benign prompts (lower is better)

## Toxicity Scoring

The task uses the following priority:

1. Perspective API (if `PERSPECTIVE_API_KEY` is set)
2. Built-in keyword heuristic fallback (offline)

This keeps the task usable in air-gapped/offline settings while still supporting API-based scoring.

## Usage

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks safety_redteam \
  --batch_size 1 \
  --limit 20
```

Run only harmful split:

```bash
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct --tasks safety_jailbreakbench_harmful --batch_size 1 --limit 20
```

Run only benign split:

```bash
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct --tasks safety_jailbreakbench_benign --batch_size 1 --limit 20
```

## Notes

- `jailbreak_asr` is computed from non-refusal, non-filtered response plus target-prefix/heuristic harmfulness checks.
- For production safety reporting, pair this benchmark with a stronger judge model (for example, HarmBench/WildGuard-style classifiers).
