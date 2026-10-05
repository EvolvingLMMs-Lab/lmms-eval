# Shared MCQ extraction contract

This corpus specifies the observable behavior of
`lmms_eval.tasks._task_utils.mcq_extract.extract_mcq_answer`. Expected answers
are reviewed independently of the implementation. A parser's current output is
not a reason to change a fixture's expected answer.

This is a general MCQ contract shared by many benchmarks, not an MMStar-specific
parser. Cover two-, three-, four-, five-, eight-, and fourteen-option tasks,
restricted/noncontiguous alphabets, multilingual replies, and different
generation styles. Consumer tests span multiple benchmark wrappers and the
verification pipeline; MMStar is only one real model E2E entry point.

## Public interface

- Inputs are a response string and optional single ASCII letter choices.
- Missing/empty responses return `""`. `None` or an empty choices list uses
  the default `A` through `H`, preserving the existing API.
- Choices are case insensitive; outputs are uppercase and belong to the
  supplied choices. Custom alphabets, including `I` through `N`, are supported.
- Reasoning stripping is an upstream responsibility. This parser does not
  replace the separate reasoning scorer or a benchmark's official scorer.

## Evidence and precedence

1. Answer containers (`<answer>...</answer>` and `\boxed{...}`) scope parsing
   to their content. The last container wins, including an invalid/ambiguous
   final container; its closing tag must never become an answer candidate.
   Unfinished LaTeX braces, nested answer containers, and mismatched answer
   tags abstain. Ordinary nested presentation wrappers inside a single box,
   such as `\boxed{\text{B}}`, remain valid.
2. Explicit declarations (`Answer: B`, `the answer is b`, `I choose C`,
   `Option A is correct`, and existing Chinese/Korean/Japanese phrases) take
   precedence over implicit labels and incidental mentions. The last explicit
   declaration wins. Match the choice adjacent to the declaration, rather
   than scanning arbitrary following prose for any letter. An explicit final
   choice outside the allowed alphabet returns empty, without falling back
   to an earlier distractor.
   Legacy open markers `<answer> c` and `answer> d` remain supported. Literal
   multilingual markers are Chinese `答案是`, `答案为`, `选`; Korean `옵션`,
   `정답은`, `답은`, `답:`; and Japanese `答えは`. Allow whitespace and common
   punctuation between each marker and its adjacent choice token.
   Fullwidth parentheses around ASCII choices are also formatting. In an
   explicit Korean/Japanese declaration, the grammatical suffixes `입니다`
   and `です` may immediately follow the choice (`정답은 b입니다.`,
   `答えはBです。`). This scoped exception does not authorize arbitrary Unicode
   word suffixes or implicit choice letters embedded in Unicode words.
3. Without explicit evidence, accept bare letters, a single leading option
   label (`B. ...`, `(b) ...`, `C: ...`, `D) ...`), standalone parenthesized
   choices, and an unambiguous trailing uppercase choice. Multiple distinct
   implicit labels or a complete option list return empty rather than guessing.
   Unambiguous trailing choices occur on a standalone final line or follow a
   comma/semicolon. Merely ending prose with `Vitamin B` or `the value of C`
   is insufficient. Markdown emphasis and inline-code wrappers around a choice
   (`**B**`, `*B*`, or `` `B` ``) preserve its meaning.
4. A response beginning with `A`/`a` followed by ordinary noun prose is
   ambiguous with the English article and returns empty. Answer assertions
   such as `A is correct`, `A because ...`, `A seems correct`, and punctuated
   labels remain accepted. Lowercase letters in ordinary prose (`plan b`,
   `the value of c`, articles) are not implicit answers.
5. Reject multiple alternative selections (`A or B`, `B and C`, `A/B`) in a
   single answer declaration/container. A correction with a later explicit
   declaration is different: `I choose A. Actually, the answer is B` returns B.
6. Choice tokens must not be embedded in words, acronyms, identifiers, decimal
   numbers, URLs, or Unicode words. Dots, colons, or closing parentheses after
   a word do not turn its final letter into an option label. Do not infer an
   answer from an arbitrary uppercase letter mentioned in prose.

These rules deliberately prefer explicit final answers and abstention over
the old unrestricted substring fallback. They cannot infer author intent in
every ambiguous sentence. Keep ambiguous examples and their policy rationale
visible in the corpus.

## Fixture format and coverage

`cases.json` is UTF-8 JSON with `schema_version: 1` and a `cases` array. Each
case has these required fields:

| Field | Meaning |
| --- | --- |
| `id` | Unique stable ID shown in pytest failure output |
| `category` | Coverage group, such as `regression`, `phrase`, or `negative` |
| `response` | Exact model output string, or null for the empty-input guard |
| `choices` | Explicit choice list, or null for defaults |
| `expected` | Reviewed uppercase letter or the empty string |
| `rationale` | Why this response does or does not select a choice |

Optional `base_expected` records an observed output at the reproduction base;
it is evidence, not the desired answer. Do not invent historical outputs.

Benchmark-inspired adversarial cases use `category: hack_<benchmark>`, with
stable IDs prefixed the same way. Their optional `benchmark_context` provides
the parser name, attack mechanism, and exact synthetic option text used for
native-parser replay. The option keys must match `choices`. `expected` still
specifies this shared extractor's contract, independently of what the native
benchmark heuristic returns.

Coverage includes the six #1554 review examples; all existing positive and
negative tests; structured formats; case/whitespace/Markdown variants;
multilingual phrases; explicit corrections and precedence; alternatives and
option lists; invalid and extended choice alphabets; acronyms, prose,
punctuation, identifiers and Unicode boundaries; and empty inputs.

`test_mcq_extract.py` validates fixture structure and runs every case
individually against both the shared utility and `MCQExtractor`. Additional
tests exercise consumer scoring, constructor/per-call choice overrides, and
letter substitution across a small documented grammar. These are CPU tests
without model downloads. They establish the extraction contract, not model
E2E validation.

## Benchmark attack fixtures

There are 52 additional adversarial fixtures across six benchmark categories:

| Category | Cases | Native parsing path | Mechanisms |
| --- | ---: | --- | --- |
| `hack_mmmu` | 12 | `_task_utils/mmmu_mcq_utils.py:parse_mmmu_multi_choice_response` | Bracket priority, echoed options, missing token boundaries, option-text matching, random fallback |
| `hack_mmmu_pro` | 8 | `_task_utils/mmmu_mcq_utils.py:parse_mmmu_pro_multi_choice_response` | Unbounded `Answer:` substrings, lowercase corrections, invalid-answer fallback, ten-choice alternatives |
| `hack_videommmu` | 8 | `_task_utils/mmmu_mcq_utils.py:parse_videommmu_multi_choice_response` | Period/colon priority, acronym suffixes, echoed options, conflicting containers |
| `hack_mmbench` | 8 | `mmbench/mmbench_evals.py:MMBench_Evaluator.can_infer` | Prefetch punctuation buckets, short articles, unoffered E, negated option text |
| `hack_seedbench` | 8 | `seedbench/utils.py:seed_process_result` | First-character truncation, words/articles, alternatives, later corrections |
| `hack_ai2d` | 8 | `ai2d/utils.py:MultiChoiceRegexFilter.apply` | First punctuated label, echoed options, unoffered labels, invalid/ambiguous final answers |

The fixtures are fabricated output strings with explicit synthetic option
context. They do not contain held-out answers or model-generated evidence, and
they are not a benchmark accuracy experiment. They include abstention cases
such as `Answer: JUNK`, as well as positive controls where a later explicit
answer or container must override a distracting label. Uppercase/lowercase,
two or more conflicting labels, and four/five/eight/ten choices are covered.

Run the actual local native parsing paths and the shared extractor:

```bash
env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  WANDB_MODE=disabled \
  ./.venv/bin/python test/eval/mcq_extract/probe_benchmarks.py \
  --output /tmp/mcq-hack-benchmark-observations.json
./.venv/bin/python -m pytest -q test/eval/test_mcq_extract.py -k hack_
```

The probe preserves raw outputs and runs the randomized MMMU-family paths
with seeds 0, 1, 7, and 42. It records MMBench's actual prefetch result, which
runs before static or API judging; it does not call the judge API. For AI2D it
records the actual configured filter output, before exact-match scoring. Raw
non-label strings and `False` remain visible, and are treated as no selected
choice only when summarizing agreement. SEEDBench's case-insensitive scoring
means a returned lowercase letter is summarized as its uppercase choice.
The two unoffered-E fixtures are boundary controls: a returned E cannot earn
credit against an A-D target, so these are not score-inflation demonstrations.

Native observations are diagnostic evidence, not assertions that preserve
undesired behavior. Future native-parser fixes may change those observations;
the shared utility and verifier must continue satisfying the reviewed fixture
expectations. No benchmark scorer is changed by adding these test data.

The original `validation.json` and `validation.md` retain the historical
219-case before/after evidence and four-sample real model smoke test from
#1564. The benchmark attack fixtures expand the current corpus to 271 cases;
their observations and validation are recorded separately in
`benchmark_validation.json` and `benchmark_validation.md`.

## Reproduction and verification

Use the repository's uv-managed environment. Reproduction base:
`bc27dfacaa2d23f3203544e1c0cc3141964167e3` (merged #1554).

```bash
./.venv/bin/python test/eval/mcq_extract/reproduce.py \
  --revision bc27dfacaa2d23f3203544e1c0cc3141964167e3 \
  --output /tmp/mcq-before.json
./.venv/bin/python test/eval/mcq_extract/reproduce.py \
  --output /tmp/mcq-after.json
./.venv/bin/python -m pytest -q \
  test/eval/test_mcq_extract.py test/eval/test_mcq_extract_consumers.py
uv run pre-commit run --all-files
```

The reproduction helper records each actual output and a mismatch summary.
It exits with status 1 when outputs differ from the reviewed expectations.
Before/after JSON reports must use the same corpus. Real CLI model E2E
validation must additionally exercise a consumer such as plain `mmstar`;
`mmstar_reasoning` uses a different parser and is insufficient.

## Existing library comparison

The repository already depends on [Math-Verify](https://github.com/huggingface/Math-Verify).
Its `StringExtractionConfig` supports configured choice strings and anchored
answer extraction. On the original 219-case corpus, installed version 0.9.0 matched 104/219
cases with uppercase strings, 135/219 with both cases configured, and 113/219
with both cases and unanchored extraction disabled. This compares agreement
with this contract; it is not a claim about either parser's general accuracy.
Rerunning the comparison now includes the added fixtures, so its denominator
will be 271 rather than the historical 219.

The differences include closed answer tags, alternatives (`A or B`), and
invalid final declarations that must mask earlier valid answers. Those are
project-level policies, so replacing the utility directly or using this
extractor as an unrestricted fallback would undo intentional abstention.
After this utility has already selected a bounded letter, another string
extractor would only duplicate the token check. Keep the core format parser
dependency-free; mathematical equivalence remains a separate verifier.

[Inspect's `answer` scorer](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html#answer)
illustrates an explicit `ANSWER:` protocol, while its `choice` scorer works
with its multiple-choice solver and task state. The
[lm-evaluation-harness choice filter](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/filters/extraction.py)
uses configurable regular expressions and document choices. These are useful
design references, but adopting either full framework would not remove the
policy layer described above. Only Math-Verify was executed for this comparison.

The comparison's complete source is committed and can be rerun independently:

```bash
./.venv/bin/python test/eval/mcq_extract/compare_math_verify.py \
  --output /tmp/mcq-math-verify-comparison.json
```
