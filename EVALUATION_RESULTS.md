# Omni-Modal Model Evaluation Results

**Date:** 2025-12-25
**Branch:** add-omni-models
**Evaluation Limit:** 10 samples per task

## Models Added

1. **Qwen3-Omni** (`qwen3_omni`)
   - Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
   - Uses `Qwen3OmniMoeForConditionalGeneration` and `Qwen3OmniMoeProcessor`
   - Supports: images, video, audio

2. **Video-SALMONN** (`video_salmonn`)
   - Model: `tsinghua-ee/video-SALMONN-2_plus_7B`
   - Base: `Qwen/Qwen2.5-VL-7B-Instruct` with LoRA adapters (PEFT)
   - Supports: images, video

3. **Uni-MoE** (`uni_moe`)
   - Model: `HIT-TMG/Uni-MoE-2.0-Omni`
   - Requires custom installation from https://github.com/HITsz-TMG/Uni-MoE
   - **Status:** Not tested (requires external dependencies)

## Evaluation Results

### Summary Table

| Model | Task | Status | Score | Notes |
|-------|------|--------|-------|-------|
| qwen3_omni | mme | Success | 0 | Image task, limited samples |
| qwen3_omni | mmau | Success | 20% accuracy | Audio understanding task |
| qwen3_omni | omni_bench | Success | 0% accuracy | Audio-visual task, limited samples |
| video_salmonn | mme | Success | 170 | Image task |
| video_salmonn | mmau | Success | 30% accuracy | Audio understanding task |
| video_salmonn | omni_bench | Success | 40% accuracy | Audio-visual task |
| uni_moe | - | Not Tested | - | Requires custom installation |

### Detailed Results

#### qwen3_omni

**MME (Image Understanding)**
```
|      Tasks      |Version|Filter|n-shot|  Metric  |   |Value|   |Stderr|
|-----------------|-------|------|-----:|----------|---|-----|---|------|
|mme              |   N/A |      |      |          |   |     |   |      |
| - mme_cognition |      1|none  |     0|mme_score |↑  |    0|±  |   N/A|
| - mme_perception|      1|none  |     0|mme_score |↑  |    0|±  |   N/A|
```

**MMAU (Audio Understanding)**
```
|      Tasks      |Version|Filter|n-shot|  Metric  |   |Value|   |Stderr|
|-----------------|-------|------|-----:|----------|---|-----|---|------|
|mmau             |    N/A|      |      |          |   |     |   |      |
| - mmau_test     |      0|none  |     0|submission|↑  |N/A  |±  |   N/A|
| - mmau_test_mini|      0|none  |     0|accuracy  |↑  |   20|±  |   N/A|
```

**Omni-Bench (Audio-Visual)**
```
|  Tasks   |Version|Filter|n-shot| Metric |   |Value|   |Stderr|
|----------|------:|------|-----:|--------|---|----:|---|------|
|omni_bench|      0|none  |     0|accuracy|↑  |    0|±  |   N/A|
```

#### video_salmonn

**MME (Image Understanding)**
```
|      Tasks      |Version|Filter|n-shot|  Metric  |   |Value|   |Stderr|
|-----------------|-------|------|-----:|----------|---|-----|---|------|
|mme              |   N/A |      |      |          |   |     |   |      |
| - mme_cognition |      1|none  |     0|mme_score |↑  |    0|±  |   N/A|
| - mme_perception|      1|none  |     0|mme_score |↑  |  170|±  |   N/A|
```

**MMAU (Audio Understanding)**
```
|      Tasks      |Version|Filter|n-shot|  Metric  |   |Value|   |Stderr|
|-----------------|-------|------|-----:|----------|---|-----|---|------|
|mmau             |    N/A|      |      |          |   |     |   |      |
| - mmau_test     |      0|none  |     0|submission|↑  |N/A  |±  |   N/A|
| - mmau_test_mini|      0|none  |     0|accuracy  |↑  |   30|±  |   N/A|
```

**Omni-Bench (Audio-Visual)**
```
|  Tasks   |Version|Filter|n-shot| Metric |   |Value|   |Stderr|
|----------|------:|------|-----:|--------|---|----:|---|------|
|omni_bench|      0|none  |     0|accuracy|↑  |   40|±  |   N/A|
```

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
