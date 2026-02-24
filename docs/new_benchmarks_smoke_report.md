# New Benchmarks Unified Smoke Test

## Command

```bash
uv run python -m lmms_eval eval \
  --model dummy_video_reader \
  --model_args "response=1,fail_on_missing=true" \
  --tasks repcount,countix,ovr_kinetics,ssv2,vggsound,av_asr \
  --limit 1 \
  --output_path outputs/smoke_new_bench \
  --log_samples \
  --process_with_media \
  --verbosity INFO
```

## Final Metrics Table

| Benchmark | Metric | Value |
| --- | --- | ---: |
| repcount | mae_norm | 0.7843 |
| repcount | obo | 0.0000 |
| countix | mae_norm | 0.6452 |
| countix | obo | 0.0000 |
| ovr_kinetics | mae | 1.0000 |
| ovr_kinetics | obo | 1.0000 |
| ssv2 | acc | 0.0000 |
| vggsound | acc | 0.0000 |
| av_asr | wer | 100.0000 |

Source: `outputs/smoke_new_bench/20260224_200855_results.json`

## Per-dataset Input / Output

| Benchmark | Input | Target | Model Output |
| --- | --- | --- | --- |
| repcount | How many times is the action repeated in this video? Answer with a single integer. | `5` | `1` |
| countix | Count the number of repetitions in this clip. Answer with a single integer. | `3` | `1` |
| ovr_kinetics | How many times does jumping happen in this video? Answer with a single integer. | `2` | `1` |
| ssv2 | What action is being performed in this video? Answer with the action label only. | `moving drawer of night stand` | `1` |
| vggsound | What is the main sound event in this clip? Answer with the sound class label only. | `playing acoustic guitar` | `1` |
| av_asr | Transcribe the speech in this video. | `hello world` | `1` |

Sample files:

- `outputs/smoke_new_bench/20260224_200855_samples_repcount.jsonl`
- `outputs/smoke_new_bench/20260224_200855_samples_countix.jsonl`
- `outputs/smoke_new_bench/20260224_200855_samples_ovr_kinetics.jsonl`
- `outputs/smoke_new_bench/20260224_200855_samples_ssv2.jsonl`
- `outputs/smoke_new_bench/20260224_200855_samples_vggsound.jsonl`
- `outputs/smoke_new_bench/20260224_200855_samples_av_asr.jsonl`
