# Omni-Modal Model Evaluation Results

**Date:** 2025-12-26
**Branch:** add-omni-models
**Evaluation Limit:** 10 samples per task

## Models Added

1. **Qwen3-Omni** (`qwen3_omni`)
   - Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
   - Uses `Qwen3OmniMoeForConditionalGeneration` and `Qwen3OmniMoeProcessor`
   - Supports: images, video, audio, and mixed audio+image inputs

2. **Video-SALMONN** (`video_salmonn`)
   - Model: `tsinghua-ee/video-SALMONN-2_plus_7B`
   - Base: `Qwen/Qwen2.5-VL-7B-Instruct` with LoRA adapters (PEFT)
   - Supports: images, video (with embedded audio tracks)
   - Note: Does NOT support standalone audio files

3. **Uni-MoE** (`uni_moe`)
   - Model: `HIT-TMG/Uni-MoE-2.0-Omni`
   - Requires custom installation from https://github.com/HITsz-TMG/Uni-MoE
   - **Status:** Not tested (requires external dependencies)

## Evaluation Results

### Summary Table

| Model | Task | Score | Notes |
|-------|------|-------|-------|
| qwen3_omni | mme | 0 | Image task, limited samples |
| qwen3_omni | mmau | **60%** | Audio understanding task |
| qwen3_omni | omni_bench | **70%** | Audio+image task |
| video_salmonn | mme | 170 | Image task |
| video_salmonn | mmau | 30% | Audio ignored (not supported) |
| video_salmonn | omni_bench | 40% | Audio ignored (not supported) |
| uni_moe | - | - | Requires custom installation |

### Key Fixes Applied

1. **Mixed modality handling**: Fixed flatten logic to preserve `[audio, image]` groupings
2. **Tuple output handling**: Model returns `(text_ids, audio)` tuple even when `return_audio=False`
3. **Audio preprocessing**: Added stereo-to-mono conversion and 16kHz resampling
4. **AudioDecoder support**: Handle new datasets library audio format

## Installation Notes

### qwen3_omni
```bash
pip install qwen-omni-utils
# Uses sdpa attention implementation (flash_attention_2 requires additional installation)
python -m lmms_eval --model qwen3_omni --model_args pretrained=Qwen/Qwen3-Omni-30B-A3B-Instruct,attn_implementation=sdpa --tasks <task>
```

### video_salmonn
```bash
pip install transformers peft qwen-vl-utils
python -m lmms_eval --model video_salmonn --model_args pretrained=tsinghua-ee/video-SALMONN-2_plus_7B --tasks <task>
```

**Important**: video_salmonn does NOT support standalone audio inputs. Use video files with embedded audio tracks instead.

### uni_moe
```bash
git clone https://github.com/HITsz-TMG/Uni-MoE
export PYTHONPATH=/path/to/Uni-MoE/Uni-MoE-2:$PYTHONPATH
python -m lmms_eval --model uni_moe --model_args pretrained=HIT-TMG/Uni-MoE-2.0-Omni --tasks <task>
```

## Notes

- All evaluations were run with `--limit 10` for testing purposes
- Results may vary with full dataset evaluation
- uni_moe requires manual setup of external dependencies before use
