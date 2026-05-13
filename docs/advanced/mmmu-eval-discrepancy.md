# Why Different Eval Frameworks Report Different MMMU Scores

> A code-level analysis of evaluation logic differences between lmms-eval and VLMEvalKit on MMMU_VAL, grounded in actual source code.

## TL;DR

When evaluating the same model (e.g. Qwen3-VL) on the same benchmark (MMMU_VAL, 900 questions), lmms-eval reports **50.67** while VLMEvalKit reports **54.78**. The ~4-point gap comes from differences in three places:

1. **Multiple-choice answer extraction** - different rule-based parsers with different fallback strategies
2. **Open-ended question evaluation** - rule-based substring matching vs. GPT-as-judge binary classification
3. **Random fallback vs. GPT fallback** - when parsing fails, lmms-eval guesses randomly; VLMEvalKit calls a judge model

For the **19 VLMEvalKit false positives**, this report uses a concrete definition: the model answer itself is wrong, but VLMEvalKit still assigns it as correct through one of these evaluator paths:

- **FP-A (MCQ letter inference path)**: `can_infer_option()` extracts a standalone option letter near the tail of the response and maps it to a choice.
- **FP-B (MCQ text inference path)**: `can_infer_text()` identifies a unique option-text substring in the response and maps it to that choice.
- **FP-C (Open-ended judge path)**: GPT binary judging marks the response as the standard answer in the "standard answer vs. others" conversion.

Out of 39 differential cases, 19 are VLMEvalKit false positives (wrong answers marked correct), and 10 are cases where GPT judge successfully recovered answers that rule-based parsing missed. VLMEvalKit's higher score is partly genuine (recovering correct but non-standard answers) and partly overestimation (marking wrong answers as correct).

---

## Background

A community contributor ([mathCrazyy](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)) conducted a systematic comparison of lmms-eval, Qwen3VL, and VLMEvalKit evaluation results on `mmmu_val`. Qwen3VL and VLMEvalKit share the same evaluation logic, so we treat them as one system. This document grounds the contributor's findings in actual source code from both frameworks.

---

## 1. What lmms-eval Actually Sends to Qwen3-VL

### System Prompt

lmms-eval uses a fixed system prompt for Qwen3-VL, defined in the model class constructor:

```python
# Source: lmms_eval/models/simple/qwen3_vl.py, line 50
system_prompt: Optional[str] = "You are a helpful assistant."
```

This system prompt is injected as the first message in every conversation:

```python
# Source: lmms_eval/models/simple/qwen3_vl.py, line 229
message = [{"role": "system", "content": self.system_prompt}]
```

### Task-Level Prompts

MMMU uses model-specific prompt templates defined in the task YAML. There are two configurations:

**Default prompt** (used for most models):

```yaml
# Source: lmms_eval/tasks/mmmu/mmmu_val.yaml, lines 17-21
lmms_eval_specific_kwargs:
  default:
    prompt_type: "format"
    multiple_choice_prompt: "Answer with the option's letter from the given choices directly."
    open_ended_prompt: "Answer the question using a single word or phrase."
```

**Qwen3-VL specific prompt** (used when model name matches `qwen3_vl`):

```yaml
# Source: lmms_eval/tasks/mmmu/mmmu_val.yaml, lines 22-26
  qwen3_vl:
    format: "qwen3_vl"
    pre_prompt: "Question: "
    post_prompt: "Answer with the option letter only."
    open_ended_prompt: "Please select the correct answer from the options above."
```

### Full Input Example (Multiple-Choice)

For a multiple-choice question about Art History with 4 options and 1 image, lmms-eval constructs the following message sequence for Qwen3-VL. The processor's `apply_chat_template()` then converts it to the final token stream:

```
┌─────────────────────────────────────────────────────┐
│ Message 1: system                                   │
│ "You are a helpful assistant."                       │
├─────────────────────────────────────────────────────┤
│ Message 2: user                                     │
│ [image_1]                                           │
│                                                     │
│ Question: Who painted this artwork?                  │
│ Options:                                            │
│ A. Leonardo da Vinci                                │
│ B. Michelangelo                                     │
│ C. Raphael                                          │
│ D. Donatello                                        │
│ Answer with the option letter only.                 │
└─────────────────────────────────────────────────────┘
```

The code that builds this (for qwen3_vl format):

```python
# Source: lmms_eval/tasks/mmmu/utils.py, lines 81-99
def mmmu_doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    open_ended_prompt = lmms_eval_specific_kwargs.get("open_ended_prompt", "")

    question = doc["question"]
    options = parse_options(ast.literal_eval(doc["options"]))
    question_type = doc["question_type"]

    if question_type == "multiple-choice":
        prompt = f"{pre_prompt}{question}\nOptions:\n{options}\n{post_prompt}"
    else:
        # open ended question
        prompt = f"{pre_prompt}{question}\nOptions:\n{options}\n{open_ended_prompt}"
```

Note: For the Qwen3-VL format, **even open-ended questions include the options list** and use the prompt `"Please select the correct answer from the options above."` This effectively converts open-ended questions into multiple-choice format for Qwen3-VL. This is a deliberate design choice from the Qwen3-VL technical report.

### Full Input Example (Open-Ended, Default Format)

For non-Qwen3-VL models using the default format, open-ended questions look different:

```
┌─────────────────────────────────────────────────────┐
│ Message 1: system                                   │
│ "You are a helpful assistant."                       │
├─────────────────────────────────────────────────────┤
│ Message 2: user                                     │
│ [image_1]                                           │
│                                                     │
│ What is the value of x in this equation?             │
│                                                     │
│ Answer the question using a single word or phrase.  │
└─────────────────────────────────────────────────────┘
```

No options are shown. The model must generate a free-form answer.

### Generation Parameters

```yaml
# Source: lmms_eval/tasks/mmmu/_default_template_yaml
generation_kwargs:
  max_new_tokens: 128
```

The model class also sets:

```python
# Source: lmms_eval/models/simple/qwen3_vl.py, lines 332-337
default_gen_kwargs = {
    "max_new_tokens": 128,
    "temperature": 0.0,   # greedy decoding
    "top_p": None,
    "num_beams": 1,
}
```

---

## 2. Multiple-Choice Parsing: Code-Level Comparison

This is where the biggest evaluation difference lives.

### lmms-eval: `parse_mmmu_multi_choice_response()`

Source: [`lmms_eval/tasks/_task_utils/mmmu_mcq_utils.py`, lines 17-65](../lmms_eval/tasks/_task_utils/mmmu_mcq_utils.py)

```python
def parse_mmmu_multi_choice_response(response, all_choices, index2ans):
    # Step 1: Strip punctuation, pad with spaces
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    # Step 2: Hierarchical pattern matching
    # Priority 1: Match (A) format
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)

    # Priority 2: Match "A " format (letter + space)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    # Priority 3: Match "A." format (letter + period)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # Priority 4: Option text matching (only if response > 5 words)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)

    # Fallback: RANDOM GUESS
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)

    # Tiebreaker: last occurrence wins
    elif len(candidates) > 1:
        # ... picks the candidate with the largest rfind position
```

**Key characteristics:**
- Pure rule-based, no external model calls
- When all rules fail: **random guess** from available choices
- Option text matching only activates for responses longer than 5 words
- Cannot infer the option letter when the model only restates the option content in a short response

### VLMEvalKit: `can_infer_option()` + `can_infer_text()`

Source: [`vlmeval/utils/matching_util.py`](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/utils/matching_util.py)

```python
def can_infer_option(answer, choices):
    # Step 1: Replace punctuation with spaces
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')
    splits = [x.strip() for x in answer_mod.split()]

    # Step 2: Count how many choice letters appear
    count = count_choice(splits, choices)

    # Step 3: If exactly one choice found, verify it's near the end
    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch  # Letter found once, near the end

    return False  # Cannot infer


def can_infer_text(answer, choices):
    answer = answer.lower()
    # Guard: skip if answer is much longer than all options combined
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    # Check if exactly one option text appears in the answer
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]  # Unique text match
    return False
```

**The VLMEvalKit pipeline** (in order):
1. `can_infer_option()` - Find a unique option letter near the end of response
2. `can_infer_text()` - Find a unique option text substring in the response
3. **GPT judge model** - Call GPT (e.g., gpt-3.5-turbo-0613) to extract the answer
4. If judge is unavailable, return `"Z"` (marked wrong)

### Side-by-Side Comparison

| Aspect | lmms-eval | VLMEvalKit |
|---|---|---|
| Approach | Hierarchical pattern matching | Word tokenization + position check |
| Letter patterns | `(A)`, `A `, `A.` | Letter as standalone token near end |
| Text matching | Only if response > 5 words | Always attempted; guarded by length ratio |
| Fallback | `random.choice()` | GPT judge model |
| Final fallback | Random letter (uniform) | `"Z"` (always wrong) |
| External deps | None | Requires GPT API |

### Concrete Example

Consider a model response: `"Choose the option of $2,616,000"`

- **lmms-eval**: No `(A)`, no `A `, no `A.` found. Response is > 5 words, so it tries option text matching. If option B's text is `"$2,616,000"`, it matches -> returns `B`. **This works.**
- **VLMEvalKit**: `can_infer_option()` finds no letter. `can_infer_text()` finds `"$2,616,000"` matches only option B -> returns `B`. **Also works.**

Now consider: `"So the correct answer is C."`

- **lmms-eval**: Strips the trailing `.`, pads -> `" So the correct answer is C "`. Matches `"C "` -> returns `C`. **Works.**
- **VLMEvalKit**: `can_infer_option()` tokenizes, finds `"C"` once, near end -> returns `C`. **Also works.**

Now consider: `"The answer is approximately 3.14"`  (open-ended, but let's say it was MCQ with option C = `"3.14"`)

- **lmms-eval**: No letter patterns found. Response > 5 words, tries text matching: `"3.14"` in response -> returns `C`. **Works.**
- **VLMEvalKit**: `can_infer_option()` fails. `can_infer_text()`: answer length check passes, `"3.14"` found in answer -> returns `C`. **Works.**

The divergence happens in edge cases - short responses, ambiguous formatting, multiple possible matches.

---

## 3. Open-Ended Question Evaluation: Fundamentally Different Philosophies

### lmms-eval: Rule-Based Substring and Numeric Matching

Source: [`lmms_eval/tasks/mmmu/utils.py`, lines 473-542](../lmms_eval/tasks/mmmu/utils.py)

The pipeline:

```
Model Response
    │
    ▼
get_key_subresponses()          ← Extract segments after indicator words
    │                               ("answer", "so", "is", "therefore", "result")
    ▼
extract_numbers()               ← Regex: commas, scientific notation, decimals
    │
    ▼
normalize_str()                 ← Lowercase strings; round numbers to 2 decimals
    │
    ▼
eval_open()                     ← Check: substring match OR exact numeric match
```

Evaluation criterion (`eval_open`, lines 342-368):
- For string predictions: **any predicted segment contains the gold answer as a substring** -> correct
- For numeric predictions: **exact equality after normalization** -> correct

### VLMEvalKit: Binary Classification via GPT Judge

According to the contributor's analysis, VLMEvalKit handles open-ended questions in MMMU_VAL differently:

> Open-ended questions in MMMU_VAL are not evaluated independently by a scoring model; instead, they are automatically reduced to a binary-choice question of "standard answer vs. others".

This means VLMEvalKit converts open-ended questions into a simple "is this the standard answer?" binary classification, judged by GPT.

### Why This Matters

lmms-eval's rule-based approach is **deterministic and reproducible** - no API calls, no model variance. But it can miss correct answers that are phrased differently (e.g., "approximately three point one four" vs. "3.14").

VLMEvalKit's GPT-judge approach can handle semantic equivalence but introduces:
- **Non-determinism** from GPT's own variability
- **API dependency** (cost, availability, version differences)
- **Potential over-acceptance** when the judge is too lenient

---

## 4. Statistical Analysis of the 39 Differential Cases

Out of 900 questions in MMMU_VAL, 39 received different judgments between the two frameworks:

| Situation | Count | % of 39 | Impact on /900 | Cumulative |
|---|---|---|---|---|
| **VLMEvalKit False Positive** (wrong answer marked correct) | 19 | 48.7% | 2.1% | |
| **GPT Judge Extraction Success** (lmms-eval rule fails, GPT succeeds) | 10 | 25.6% | 1.1% | 3.6% overestimate |
| **Open-ended A/B conversion correct** | 2 | 5.1% | 0.2% | |
| **Random divergence** (both frameworks guess, different outcomes) | 2 | 5.1% | 0.2% | |
| **lmms-eval rule correct** (lmms-eval rules outperform VLMEvalKit) | 6 | 15.4% | 0.6% | 0.6% underestimate |

**Net effect**: VLMEvalKit scores ~4.1 points higher. Of that:
- ~3.6 points come from false positives + lenient GPT judge matching (overestimation)
- ~0.6 points are offset by cases where lmms-eval's rules are actually more accurate

---

## 5. Implications for Benchmark Users

### When comparing scores across papers

If Paper A uses lmms-eval and Paper B uses VLMEvalKit for MMMU_VAL, **expect 2-4 points of systematic difference** that has nothing to do with model quality. This is entirely from evaluation pipeline differences.

### Which framework to trust?

Neither is perfect. The tradeoffs:

| | lmms-eval | VLMEvalKit |
|---|---|---|
| **False positive rate** | Lower (conservative) | Higher (lenient) |
| **False negative rate** | Higher (misses non-standard formats) | Lower (GPT recovers) |
| **Reproducibility** | Deterministic (no API calls) | Non-deterministic (GPT variance) |
| **Cost** | Free | Requires GPT API calls |
| **Academic safety** | Safer (underestimate < overestimate) | Riskier (overestimate is worse for credibility) |

For academic evaluation, **false positives are more dangerous than false negatives** - overestimating model capability is worse than underestimating it. lmms-eval's conservative approach is the safer default.

### Recommendations

1. **Always report which evaluation framework you used** and its version
2. **Do not directly compare numbers** from lmms-eval and VLMEvalKit papers
3. **For the fairest comparison**, re-evaluate all models under the same framework
4. **If you need maximum answer recovery**, consider adding an optional GPT-judge fallback to lmms-eval (but report both scores)
5. **Check sample-level outputs** (`--log_samples`) when scores seem unexpected

---

## Appendix: Source Code Reference

| Component | File | Key Function |
|---|---|---|
| MCQ parsing (shared) | `lmms_eval/tasks/_task_utils/mmmu_mcq_utils.py` | `parse_mmmu_multi_choice_response()` |
| Open-ended parsing | `lmms_eval/tasks/mmmu/utils.py` | `parse_open_response()` |
| Open-ended evaluation | `lmms_eval/tasks/mmmu/utils.py` | `eval_open()` |
| MCQ evaluation | `lmms_eval/tasks/mmmu/utils.py` | `eval_multi_choice()` |
| Result processing entry | `lmms_eval/tasks/mmmu/utils.py` | `mmmu_process_results()` |
| String normalization | `lmms_eval/tasks/mmmu/utils.py` | `normalize_str()` |
| Random fallback | `lmms_eval/tasks/_task_utils/mmmu_mcq_utils.py` | Line 48: `random.choice(all_choices)` |
| Qwen3-VL system prompt | `lmms_eval/models/simple/qwen3_vl.py` | Line 50: `system_prompt="You are a helpful assistant."` |
| Qwen3-VL message construction | `lmms_eval/models/simple/qwen3_vl.py` | Line 229: system message injection |
| MMMU task config | `lmms_eval/tasks/mmmu/mmmu_val.yaml` | `lmms_eval_specific_kwargs` |
| Qwen3-VL task config | `lmms_eval/tasks/mmmu/mmmu_val_qwen.yaml` | Qwen3-VL-specific prompt format |
| VLMEvalKit option inference | `vlmeval/utils/matching_util.py` | `can_infer_option()` |
| VLMEvalKit text inference | `vlmeval/utils/matching_util.py` | `can_infer_text()` |

---

*This analysis is based on a community contribution by [@mathCrazyy](https://github.com/mathCrazyy) and grounded in the source code of both lmms-eval and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). The VLMEvalKit source was retrieved from the `main` branch as of February 2026.*
