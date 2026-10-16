# OmniCompass

OmniCompass evaluates temporal grounding, spatial grounding and video question
answering with paired Text and Text+Image target specifications. The VL track
contains 820 videos. The AV track uses 240 videos with their complete audio.

- [Paper](https://arxiv.org/abs/2610.18829)
- [Dataset](https://huggingface.co/datasets/VCLab-PolyU/OmniCompass)
- [Reference implementation](https://github.com/PolyU-VCLab/OmniCompass)

## Tasks and metrics

| Group | Tasks |
| --- | --- |
| `omnicompass_vl` | `omnicompass_vl_tg`, `omnicompass_vl_sg`, `omnicompass_vl_qa` |
| `omnicompass_av` | `omnicompass_av_tg`, `omnicompass_av_sg`, `omnicompass_av_qa` |

Each task runs both target specifications. It reports `text`, `text_image` and
the paired Cross-Specification Consistency Score, `cs`, on a 0 to 1 scale.
Multiply by 100 to compare with the paper's percentage tables. The three tasks
use temporal IoU, mean spatial IoU at the two given keyframes, and QA accuracy.
CS uses the reference implementation's paired scoring rule.
The group rows report Overall scores as the unweighted mean of the three tasks.

The default is 32 uniformly sampled frames and a 32,768-token output budget.
Each frame is preceded by its decoded timestamp. Text+Image includes one
additional reference image; it does not count toward the video frame budget.
AV also receives the complete audio on the original video timeline.

Completed empty or unparseable answers score zero. Content-policy rejections
follow the official exclusion rule. Transport errors leave the evaluation
incomplete and cause a failure; they must be retried before reporting results.
Output-limit metadata is retained by the structured-response backends.
The task does not silently reduce the frame count or output budget.

## Install and data

From the lmms-eval repository root:

```bash
uv sync
```

Install FFmpeg for AV extraction. PyAV and Pillow provide video-frame decoding.
Set `FFMPEG_BINARY` to a working executable if FFmpeg is not on `PATH`.
GPU backends may need additional model-specific dependencies.

The first run downloads seven small metadata files from the public dataset and
checks their hashes. Media archives are fetched as selected samples need them,
verified and extracted to the local cache. A single video can require a
multi-gigabyte TAR download; a full run needs substantial disk space for both
archives and extracted files. Existing verified files are reused.

The default dataset revision is pinned in `_core/hf_data.py`. No Hugging Face
token is required for the public release. To use an existing full or partial
download, set `OMNICOMPASS_DATA` to the directory containing `release.json`,
the six metadata files, and the selected `videos/` and `references/` files.
Local mode verifies files and never downloads missing media.

## API evaluation

Use a chat-completions endpoint that accepts multiple images. Provide the key
through the environment, not through `--model_args` or a committed config.

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_API_BASE="https://YOUR_PROVIDER/v1"
export MODEL_ID="YOUR_MODEL"
export OMNICOMPASS_CACHE="$HOME/.cache/omnicompass"
export OMNICOMPASS_NFRAME=32
export OMNICOMPASS_VIDEO_IDS=oc_004

uv run python -m lmms_eval \
  --model openai \
  --model_args "model_version=$MODEL_ID,return_response_json=true,strict_generation=true,image_detail=high,token_limit_field=max_completion_tokens,timeout=180,max_retries=1,num_concurrent=1" \
  --tasks omnicompass_vl \
  --batch_size 1 --fail_on_error \
  --gen_kwargs max_new_tokens=32768,temperature=0,do_sample=False \
  --use_cache outputs/omnicompass-vl/cache \
  --output_path outputs/omnicompass-vl/results \
  --log_samples
```

This selects one real video and sends six requests: three tasks times two
specifications. Remove `OMNICOMPASS_VIDEO_IDS` for the full track.
Use `token_limit_field=max_tokens` if required by the endpoint.
`return_response_json=true` preserves response status, usage and finish reason
for the official scorer and failure handling.
`strict_generation=true` applies the specified request timeout and makes the
wrapper own retries; permanent 4xx errors stop immediately. Without this option,
the backend retains its upstream SDK timeout and retry behavior.
`--fail_on_error` requests a nonzero CLI exit on failure; it is off by default.

For AV, use an endpoint that accepts **images and base64 MP3 input together**:

```bash
export OMNICOMPASS_OPENAI_AUDIO=input_audio
export OMNICOMPASS_VIDEO_IDS=oc_595

uv run python -m lmms_eval \
  --model openai \
  --model_args "model_version=$MODEL_ID,audio_format=input_audio,audio_encoding=data-uri,return_response_json=true,strict_generation=true,image_detail=high,token_limit_field=max_tokens,timeout=180,max_retries=1,num_concurrent=1" \
  --tasks omnicompass_av \
  --batch_size 1 --fail_on_error \
  --gen_kwargs max_new_tokens=32768,temperature=0,do_sample=False \
  --use_cache outputs/omnicompass-av/cache \
  --output_path outputs/omnicompass-av/results \
  --log_samples
```

The AV opt-in asserts that the selected endpoint supports this input contract.
Image-only endpoints cannot evaluate AV. The integration does not drop audio.
This example uses the data-URI encoding required by DashScope. For endpoints
that accept bare Base64, use `audio_encoding=base64`, the backend default.

## Local GPU evaluation

Use a chat backend that preserves the interleaved images and timestamps.
Legacy `doc_to_visual` backends are rejected.

```bash
export OMNICOMPASS_NFRAME=32
export OMNICOMPASS_VIDEO_IDS=oc_004

uv run python -m lmms_eval \
  --model qwen3_5 \
  --model_args pretrained=Qwen/Qwen3.5-9B,enable_thinking=True,context_length=65536,return_response_json=True,strict_generation=True,apply_image_pixel_limits=True \
  --tasks omnicompass_vl \
  --batch_size 1 --fail_on_error \
  --gen_kwargs max_new_tokens=32768,temperature=0,do_sample=False \
  --use_cache outputs/omnicompass-gpu/cache \
  --output_path outputs/omnicompass-gpu/results \
  --log_samples
```

The context must accommodate both input tokens and the requested output budget.
`context_length` declares a capacity limit; it does not extend model support.
An explicit `context_length` or `strict_generation=True` enables the preflight
capacity check. Without either option, the original backend behavior remains.
`apply_image_pixel_limits=True` applies the backend's configured
`min_pixels`/`max_pixels` to chat images. It is off by default so existing
benchmarks retain their original image preprocessing and token accounting.
Use `enable_thinking=False` to disable thinking on this backend. API endpoints
keep their provider default unless configured explicitly. Supported API switches
are `enable_thinking_kwarg=true` or `false`, with
`thinking_parameter=chat_template` or `enable_thinking` to match the provider.

The AV task declares separate-audio support for the Qwen2.5-Omni and Qwen3-Omni
chat backends. Hardware and model-specific execution must be verified separately.
The API smoke test does not validate native GPU inference.

## Sampling, subsets and resume

| Variable | Purpose |
| --- | --- |
| `OMNICOMPASS_NFRAME` | Fixed frame count, such as 32, 64 or 128. Default: 32. |
| `OMNICOMPASS_FPS` | Sampling frequency, such as 1. Unset `OMNICOMPASS_NFRAME` first. |
| `OMNICOMPASS_VIDEO_IDS` | Comma-separated public video IDs. Keeps both specifications for each selected task. |
| `OMNICOMPASS_CACHE` | Media, audio and frame cache directory. |
| `OMNICOMPASS_DATA` | Read-only local release directory. Disables media downloads. |
| `OMNICOMPASS_REVISION` | Dataset commit or ref; a ref is resolved to a commit. |
| `OMNICOMPASS_MAX_DOWNLOAD_BYTES` | Maximum bytes per media archive; 0 forbids archive downloads. |
| `OMNICOMPASS_MODEL_REVISION` | Optional deployment/version identifier for response-cache isolation. |

Use video IDs for smoke tests. A request-level `--limit` can cut a Text/Text+Image
pair in half; incomplete pairs are rejected. This prevents reporting CS on a
different sample set.

Repeat the same command and `--use_cache` path to reuse completed deterministic
responses. OmniCompass checkpoints each answer by default. Failed transport
responses are not reused. Changing sampling, task configuration, model arguments
or recorded runtime versions creates a different cache identity. Use separate
result directories for distinct experiments.

The ordered `(question_id, setting)` selection is part of the task fingerprint.
Reusing a response-cache directory with different `OMNICOMPASS_VIDEO_IDS` is safe:
changed selections miss, and returning to the same selection can reuse its answers.
Overlapping subsets do not share answers. Caches created before this selection
fingerprint was added are not reused. This is conservative isolation, not
cross-subset deduplication.

Managed media-cache files that fail checksum verification are atomically replaced
from the verified archive under the download lock, subject to the download budget.
Files under explicit `OMNICOMPASS_DATA` remain read-only and fail on corruption.
Provider policy refusals use the same marker rules for caching and scoring;
transport failures remain retryable. Data-URI audio includes `audio/mpeg` or
`audio/wav`; live API compatibility has only been tested on the DashScope-compatible
deployment described in the PR.

Strict response validation, per-answer checkpointing, stable restart identities
and writer locks apply to OmniCompass. Unrelated benchmark requests keep the
upstream cache validity rules and scratch layout. Python tasks receive the model
name only when their class explicitly declares `MODEL_NAME_AWARE = True`.

Local model directories are content-hashed at cache initialization, including
weight shards and tokenizer files. This reads all recognized artifacts and may
be expensive for large models or multiple ranks; that startup cost has not been
benchmarked. Do not substitute timestamp-only hashes when files can change in place.

## Implementation and tests

`_core/` contains the OmniCompass reference prompts, sampler, parsers and scorer,
plus download and cache-identity helpers. It keeps evaluation independent of
another source checkout. Changes to this code must preserve the reference
protocol and have regression coverage.

```bash
PYTHONPATH=. uv run pytest -q \
  test/tasks/test_omnicompass_download.py \
  test/tasks/test_omnicompass_protocol.py \
  test/tasks/test_omnicompass_selection_cache.py \
  test/tasks/test_omnicompass_frames.py \
  test/cache/test_omnicompass_durability.py \
  test/cache/test_response_cache.py \
  test/models/test_openai.py \
  test/models/test_qwen3_vl_simple.py \
  test/models/test_omnicompass_generation.py
```

Unit tests use fixtures. Real-model validation is reported separately in the PR,
including the exact command, selected IDs, backend, hardware and emitted scores.
