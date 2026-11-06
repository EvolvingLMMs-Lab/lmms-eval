# LLM Judge

`lmms_eval.llm_judge` scores predictions with a remote judge. Tasks select the backend with `API_TYPE` (or `get_server(server_name=...)`).

| `API_TYPE` | Provider | Credentials |
| --- | --- | --- |
| `openai` (default) | OpenAI-compatible chat completions (lab default: OpenRouter) | `OPENAI_API_KEY`, optional `OPENAI_API_URL` or `OPENAI_API_BASE` |
| `azure` | Azure OpenAI | `AZURE_API_KEY`, `AZURE_ENDPOINT`, `AZURE_API_VERSION` |
| `minimax` / `async_minimax` | MiniMax chat completions | `MINIMAX_API_KEY` |
| `jev` / `typesafe` | TypeSafe Jev | `TYPESAFE_API_KEY` |
| `async_jev` / `async_typesafe` | Jev, async | `TYPESAFE_API_KEY` |
| `local` | An OpenAI-compatible local server | `OPENAI_API_URL` |

`async_*` names are the non-blocking clients. A missing key makes `is_available()` false and `evaluate` raises; it does not stop process startup.

## TypeSafe Jev

Jev is a System One model ([models](https://docs.typesafe.ai/models), [System One](https://docs.typesafe.ai/concepts/system-one)). It does not write text. `POST https://api.typesafe.ai/v1/systemone` takes a text `state` plus typed `questions` and returns probabilities. Input is text only (64k tokens per request, 32k per question). Model ids: `jev-latest` (default), `jev-1.14.0`, `jev-1.13.0`.

```bash
export TYPESAFE_API_KEY=...
export API_TYPE=jev
# Optional. A gpt-* MODEL_VERSION is ignored and jev-latest is sent.
export MODEL_VERSION=jev-latest
# Optional yes/no cutoff in [0, 1]. Default 0.5.
export JEV_NOUL_THRESHOLD=0.5
```

`ServerConfig.noul_threshold` overrides the environment variable. `TYPESAFE_MODEL` (or `JEV_MODEL`) overrides a non-Jev `model_name`. The native path requires `TYPESAFE_API_KEY`. It does not read `OPENROUTER_API_KEY`.

### OpenRouter

OpenRouter does not serve System One. The only Jev id it lists is `typesafe/jev-router`. It has no `jev-latest`, no `jev-1.14.0`, and no `POST /v1/systemone`. That id is an OpenAI-compatible `chat/completions` router: OpenRouter picks an underlying model for each request. A test call was answered by `openai/gpt-6-luna`. The response is plain text. There is no `noul`, `score`, or `choice` probability, and a later call can be a different model, so the judgment is not reproducible.

Use the existing OpenAI judge (`API_TYPE=openai`), not `API_TYPE=jev`. `OpenAIProvider` and `AsyncOpenAIProvider` read these variables:

| Variable | Role |
| --- | --- |
| `API_TYPE` | `openai` (or `async_openai` for the async client). |
| `OPENAI_API_KEY` | Bearer token. Put the OpenRouter key here. The judge does not read `OPENROUTER_API_KEY`. |
| `OPENAI_API_URL` or `OPENAI_API_BASE` | API root. Either name works. Do not append `/chat/completions`; the client adds that path. **Default is `https://openrouter.ai/api/v1`.** Set `https://api.openai.com/v1` to use official OpenAI. |
| `MODEL_VERSION` | Chat model id. Set `typesafe/jev-router`. The provider does not read this variable. Tasks such as `k12` copy it into `ServerConfig.model_name`, and that string is sent as `model`. |

```bash
export API_TYPE=openai
export OPENAI_API_KEY=...
export OPENAI_API_URL=https://openrouter.ai/api/v1
export MODEL_VERSION=typesafe/jev-router
```

The task prompt is sent as a normal chat completion, and the task parses the generated text. This route does not apply the noul threshold, and it does not return `parsed_result["probability"]`. For calibrated `noul`, `score`, and `choice` answers, use `API_TYPE=jev` with `TYPESAFE_API_KEY` as described above.

### How judge calls are mapped

| Judge call | Jev question | String the task parses |
| --- | --- | --- |
| Binary correct / incorrect (`evaluate_binary`, `judge_type="binary"`, or a prompt that asks for Yes/No, 1/0, True/False, Correct/Incorrect) | `noul` | `Yes`/`No`, `1`/`0`, `True`/`False`, `Correct`/`Incorrect`, or `<label>True</label>` / `<label>False</label>`, chosen from `output_format` or the prompt. Positive when `noul >= threshold`. |
| Numeric score (`judge_type="score"`, `score_range`, or a prompt that states a scale) | `score` | The level index mapped back onto that scale, as a bare number (`0.4`, `10`). |
| Rubric (`evaluate_with_rubric`) | one `score` per item | JSON object of scores. A list value is that item's ordered levels (2–10). |
| Options (`Request.options` or `judge_type="choice"`) | `choice` (at most 255 options) | The chosen option string. |
| Pairwise (`evaluate_comparative`, or a prompt with two responses and a scale) | two `score` questions | `score1 score2` on one line. |

The yes-probability is on the response and on the binary helper:

- `response.parsed_result["probability"]` (also `noul`, `threshold`, `decision`)
- `evaluate_binary(...)["probability"]`

Choice and score answers also expose `confidence` and `probabilities` on `parsed_result` when Jev sends them.

A judge request that includes an image (`Request.images`, an `image` / `image_url` content part, or a `data:image/` payload) raises `JevTextOnlyError` before any HTTP call. A prompt that asks the model to extract a free-form answer (`Extracted answer:`) raises `JevUnsupportedPromptError`. Both are ordinary exceptions. Tasks that already catch a failed judge call record that sample as unscored and continue.

Timeouts, HTTP 429, 529, and 5xx are retried with exponential backoff. A numeric `Retry-After` header sets the delay. Other 4xx responses fail immediately with `JevAPIError` and the status code. An empty `answers` object, or a decision field that is missing or null, raises `ValueError` (`Jev returned an empty or unusable answer`) instead of calling `str.strip` on `None`. One failed judgment does not clear the client; the next sample still runs.

### Tasks that can use Jev

These read the canonical string, or the parsed bool / int from `evaluate_binary`, and do not send the judge an image. Set `API_TYPE=jev`.

- `k12` and `olympiadbench` testmini (`evaluate_binary`; `0/1` and `yes/no`).
- `mathverse` scoring via `score_answer` (`evaluate_binary`, `0/1`). The separate answer-extraction step cannot.
- `mathvision` when the verifier is `OpenAIVerifier(judge_type="binary")`.
- `voicebench` open QA (prompt asks for `Yes` / `No`; the task reads `response.content`).
- `emma` and `stare` (prompt ends in `Correct_or_not`; the task accepts `Correct` and rejects `Incorrect`).
- `mmvet` and `mmvetv2`. The prompt states a 0–1 correctness score. Jev returns that number, not a completed few-shot row. `float()` on the first token accepts it.
- `spatialtreebench` `gpteval`. The prompt states a 0–1 scale (including `0分到1分`). The task parser falls back to the first number in the reply. Jev does not write the `reason` string.
- Text pairwise prompts that state a scale and do not attach an image: `evaluate_comparative`, and `OpenAIVerifier(judge_type="comparative")` without images. `multilingual-llava-bench-in-the-wild` and `llava-in-the-wild` (`utils_ko`) are in this group when their rule text contains a scale such as "1 to 10". If the rule text has no scale, the call raises `JevUnsupportedPromptError` and those tasks record `-1`.

### Tasks that cannot use Jev

- Image judges: `holisafe_llm_judge`, `llava_wilder`, and `mavis`. The call raises `JevTextOnlyError`. HoliSafe and MAVIS already fall back when the judge fails.
- Free-text extraction: `mathvista` answer extraction and the `mathverse` extract step (`Extracted answer:`).
- `omnispatial` `eval_type=llm` asks for `True` / `False`, but the task treats any reply containing the substring `true` as correct, and `false` contains `true`. The canonical `False` would be scored as correct. Do not switch this task until that parser is exact.
- `mme_cot` and `internal_eval` (`d170_en`, `d170_cn`, `dc100_en`, `dc200_cn`) call `evaluate_binary`, which Jev answers, then call `.lower()` or `.strip()` on the parsed bool or int. That raises for every backend, and the sample is scored 0. Fixing those tasks is separate from this provider.
- `mmrefine` calls the OpenAI client directly, not this registry.
- `wenet_speech` constructs a judge server and does not call it.

### Live check

Mocks do not count as an end-to-end run. With `TYPESAFE_API_KEY` set, score a text-judged task on 8 samples:

```bash
export TYPESAFE_API_KEY
export API_TYPE=jev
export MODEL_VERSION=jev-latest
uv run python -m lmms_eval \
  --model openai \
  --model_args model_version=gpt-4o-mini \
  --tasks k12 \
  --limit 8 \
  --batch_size 1
```

`k12` still needs a model that can answer its items. The judge traffic goes to Jev, not to that model. The model under test may be multimodal; the judge request for `k12` is text.
