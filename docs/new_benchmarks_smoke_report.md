# New Benchmarks Unified Smoke Test

## Commands

### Non-audio benchmarks with Gemini 3 Flash

```bash
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"

uv run python -m lmms_eval eval \
  --model openai_compatible \
  --model_args "model_version=google/gemini-3-flash-preview,max_frames_num=4,num_concurrent=1,max_retries=2,timeout=60" \
  --tasks repcount,countix,ovr_kinetics,ssv2 \
  --limit 1 \
  --output_path outputs/smoke_gemini3_flash_non_audio \
  --log_samples \
  --process_with_media \
  --verbosity INFO \
  --force_simple
```

### Audio benchmarks with OpenRouter Omni-capable model

```bash
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"

uv run python -m lmms_eval eval \
  --model openai_compatible \
  --model_args "model_version=google/gemini-2.5-flash,max_frames_num=4,num_concurrent=1,max_retries=2,timeout=60" \
  --tasks vggsound,av_asr \
  --limit 1 \
  --output_path outputs/smoke_openrouter_omni_audio \
  --log_samples \
  --process_with_media \
  --verbosity INFO \
  --force_simple
```

## Final Metrics Table

| Benchmark | Model | Metric | Value |
| --- | --- | --- | ---: |
| repcount | `google/gemini-3-flash-preview` | mae_norm | 0.9804 |
| repcount | `google/gemini-3-flash-preview` | obo | 0.0000 |
| countix | `google/gemini-3-flash-preview` | mae_norm | 2.2581 |
| countix | `google/gemini-3-flash-preview` | obo | 0.0000 |
| ovr_kinetics | `google/gemini-3-flash-preview` | mae | 8.0000 |
| ovr_kinetics | `google/gemini-3-flash-preview` | obo | 0.0000 |
| ssv2 | `google/gemini-3-flash-preview` | acc | 0.0000 |
| vggsound | `google/gemini-2.5-flash` | acc | 0.0000 |
| av_asr | `google/gemini-2.5-flash` | wer | 100.0000 |

Sources:

- `outputs/smoke_gemini3_flash_non_audio/google__gemini-3-flash-preview/20260224_201715_results.json`
- `outputs/smoke_openrouter_omni_audio/google__gemini-2.5-flash/20260224_201826_results.json`

## Per-dataset Input / Output

| Benchmark | Model | Input | Target | Model Output |
| --- | --- | --- | --- | --- |
| repcount | `google/gemini-3-flash-preview` | How many times is the action repeated in this video? Answer with a single integer. | `5` | `10` |
| countix | `google/gemini-3-flash-preview` | Count the number of repetitions in this clip. Answer with a single integer. | `3` | `10` |
| ovr_kinetics | `google/gemini-3-flash-preview` | How many times does jumping happen in this video? Answer with a single integer. | `2` | `The video shows jumping 10 times.` |
| ssv2 | `google/gemini-3-flash-preview` | What action is being performed in this video? Answer with the action label only. | `moving drawer of night stand` | `Dodgeball` |
| vggsound | `google/gemini-2.5-flash` | What is the main sound event in this clip? Answer with the sound class label only. | `playing acoustic guitar` | `Speech` |
| av_asr | `google/gemini-2.5-flash` | Transcribe the speech in this video. | `hello world` | `[ Silence ]` |

Sample files:

- `outputs/smoke_gemini3_flash_non_audio/google__gemini-3-flash-preview/20260224_201715_samples_repcount.jsonl`
- `outputs/smoke_gemini3_flash_non_audio/google__gemini-3-flash-preview/20260224_201715_samples_countix.jsonl`
- `outputs/smoke_gemini3_flash_non_audio/google__gemini-3-flash-preview/20260224_201715_samples_ovr_kinetics.jsonl`
- `outputs/smoke_gemini3_flash_non_audio/google__gemini-3-flash-preview/20260224_201715_samples_ssv2.jsonl`
- `outputs/smoke_openrouter_omni_audio/google__gemini-2.5-flash/20260224_201826_samples_vggsound.jsonl`
- `outputs/smoke_openrouter_omni_audio/google__gemini-2.5-flash/20260224_201826_samples_av_asr.jsonl`
