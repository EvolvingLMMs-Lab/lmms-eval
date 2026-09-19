# MMStar

[MMStar](https://github.com/MMStar-Benchmark/MMStar) is a vision-indispensable multiple-choice benchmark: 1,500 questions in 6 coarse categories and 18 fine-grained (L2) categories, four options each.

**Dataset**: [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar), split `val`.

## Tasks

| Task | Config | Description |
|------|--------|-------------|
| `mmstar` | `mmstar.yaml` | Default protocol: "Answer with the option's letter from the given choices directly" |
| `mmstar_qwen` | `mmstar_qwen.yaml` | VLMEvalKit-compatible Qwen prompt |
| `mmstar_oc` | `mmstar_oc.yaml` | OpenCompass copy of the dataset |
| `mmstar_ko` | `mmstar_ko.yaml` | Korean translation (K-MMStar) |
| `mmstar_reasoning` | `reasoning/mmstar_reasoning.yaml` | Strict `<think>`/`<answer>` reasoning output |
| `mmstar_hybrid` | `mmstar_hybrid.yaml` | Direct answers on perception categories, CoT on reasoning categories (see below) |

## Scoring

Every task reports one accuracy per coarse category plus `average`. A coarse category is the mean of its L2 accuracies and `average` is the mean over all 18 L2 accuracies, as in the official evaluator. `process_results` reads the option letter from the response; there is no LLM judge.

## `mmstar_hybrid`

### Motivation

The default prompt asks for the option letter directly. Instruction-tuned models comply on perception questions, but on questions that need a calculation they tend to reason first and name the letter at the end. Under a direct-answer token budget such answers are truncated or scored on their first character, so the metric mixes knowledge with prompt compliance — and the ranking of two models can flip depending on how leniently the response is parsed.

Measured on the val split (Qwen3.6-27B and Qwen3.8-27B, greedy, 2048-token budget, letter taken from anywhere in the answer), the share of reasoning-style answers is 0-2% on every perception L2 category and 20-60% on geometry, statistical reasoning, code & sequence reasoning and electronics. Rescoring one and the same set of traces with a strict first-character parser versus a lenient one moves the gap between those two models from +6.5 points in favour of one model to -2.0 in favour of the other, without any model having changed.

345 of the 1,500 questions (all in `math`) come from MathVista and still carry its own instruction to give the letter "at the end", which directly contradicts the direct-answer prompt; they account for about 70% of all format violations we observed.

The hybrid protocol fixes the answer format per question instead of per model: categories where the answer is read off the image are asked directly, categories that need intermediate steps are asked with a chain-of-thought prompt and scored from an explicit final-answer line. The split is by coarse category, so it does not depend on any model's behaviour.

### Categories

| Leaf task | Protocol | Coarse categories | L2 categories | Questions |
|-----------|----------|-------------------|---------------|-----------|
| `mmstar_hybrid_perception` | direct — same prompt, generation settings and scorer as `mmstar` | coarse perception, fine-grained perception, instance reasoning | image scene and topic, image style & quality, image emotion, object counting, recognition, localization, single-instance reasoning, cross-instance attribute reasoning, cross-instance relation reasoning | 750 |
| `mmstar_hybrid_reasoning` | CoT, `max_new_tokens: 16384` | logical reasoning, science & technology, math | code & sequence reasoning, diagram reasoning, common reasoning, biology & chemistry & physics, electronics & energy & mechanical eng., geography & earth science & agriculture, geometry, numeric commonsense and calculation, statistical reasoning | 750 |

### CoT prompt and extraction

The reasoning leaf appends the MMMU-Pro style instruction to the question:

```
Answer with the option's letter from the given choices. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.
```

`utils.extract_cot_answer` then reads the letter strictly: the last `Answer: X` marker wins, `\boxed{X}` is the fallback, and anything that is not a single option letter scores 0. Reasoning models that emit `<think>` blocks are handled the same way, because the marker is searched over the whole response. The strictness is deliberate — the model was asked for an explicit final line, so a missing one is a miss, not something for the parser to guess at.

### Metrics

Each leaf reports its own three coarse categories and its own `average` (mean of nine L2 accuracies). The group `mmstar_hybrid` reports `average` as the size-weighted mean of the two leaf averages, which equals the MMStar mean over all 18 L2 accuracies. The two halves are not comparable with each other or with `mmstar`; compare models on the group `average` or per leaf.
