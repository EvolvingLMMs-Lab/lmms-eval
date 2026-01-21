# UniWorld Evaluation Scripts Usage Guide

## 📋 Overview

UniWorld-V1 is a unified vision-language model that supports both understanding and generation tasks. This guide shows you how to evaluate UniWorld on various benchmarks.

## 🚀 Quick Start

### 1. Setup UniWorld

First, clone the UniWorld repository at your lmms-eval root:

```bash
cd /path/to/lmms-eval
git clone https://github.com/LanguageBind/UniWorld.git
```

The model will be automatically downloaded from HuggingFace when you run evaluation.

### 2. Run ChartQA100 Evaluation

**Simple usage (single GPU):**
```bash
bash uniworld.sh
```

**Multi-GPU (2 GPUs):**
```bash
bash uniworld.sh "0,1"
```

**Custom model path:**
```bash
bash uniworld.sh "0" "./path/to/UniWorld-V1"
```

**Full customization:**
```bash
bash uniworld.sh "0,1" "LanguageBind/UniWorld-V1" "./my_output"
```

## 📊 Available Scripts

### 1. `uniworld.sh` - ChartQA100 Specific

**Purpose:** Evaluate UniWorld on ChartQA100 benchmark

**Syntax:**
```bash
bash uniworld.sh [GPU_IDS] [MODEL_PATH] [OUTPUT_PATH]
```

**Parameters:**
- `GPU_IDS`: GPU IDs (default: `"0"`)
- `MODEL_PATH`: Model path or HF ID (default: `"LanguageBind/UniWorld-V1"`)
- `OUTPUT_PATH`: Output directory (default: `"./logs/uniworld_chartqa100"`)

**Examples:**
```bash
# Single GPU, default settings
bash uniworld.sh

# 2 GPUs
bash uniworld.sh "0,1"

# 4 GPUs
bash uniworld.sh "0,1,2,3"

# Custom output path
bash uniworld.sh "0" "LanguageBind/UniWorld-V1" "/mnt/data/uniworld_results"
```

### 2. `uniworld_general.sh` - Multi-Task Support

**Purpose:** Evaluate UniWorld on any task (understanding or generation)

**Syntax:**
```bash
bash uniworld_general.sh [GPU_IDS] [TASK] [OUTPUT_PATH] [MODEL_PATH]
```

**Parameters:**
- `GPU_IDS`: GPU IDs (default: `"0"`)
- `TASK`: Task name or comma-separated list (default: `"chartqa100"`)
- `OUTPUT_PATH`: Output directory (default: `"./logs/uniworld_{TASK}"`)
- `MODEL_PATH`: Model path or HF ID (default: `"LanguageBind/UniWorld-V1"`)

**Examples:**

```bash
# ChartQA100
bash uniworld_general.sh "0" "chartqa100" "./logs/chartqa"

# Uni-MMMU Jigsaw Visual CoT (with image generation)
bash uniworld_general.sh "0,1" "uni_mmmu_jigsaw100_visual_cot" "./logs/jigsaw"

# Uni-MMMU Maze Visual CoT (multi-GPU)
bash uniworld_general.sh "0,1" "uni_mmmu_maze100_visual_cot" "/mnt/data/maze"

# Uni-MMMU Sliding Visual CoT
bash uniworld_general.sh "0,1" "uni_mmmu_sliding54_visual_cot" "./logs/sliding"

# Multiple tasks at once
bash uniworld_general.sh "0" "chartqa100,mmbench" "./logs/multi_task"
```

## 🎯 Task-Specific Examples

### Understanding Tasks (Text Generation)

```bash
# ChartQA
bash uniworld_general.sh "0" "chartqa100" "./logs/chartqa"

# MMBench
bash uniworld_general.sh "0" "mmbench" "./logs/mmbench"

# ScienceQA
bash uniworld_general.sh "0" "scienceqa_img" "./logs/scienceqa"
```

### Generation Tasks (Visual CoT)

For Visual CoT tasks, the script automatically:
- Detects `visual_cot` in the task name
- Creates an `images/` subdirectory for generated images
- Adds `image_output_dir` to model args

```bash
# Jigsaw Puzzle (2×2 completion)
bash uniworld_general.sh "0,1" "uni_mmmu_jigsaw100_visual_cot" "/mnt/data/jigsaw"

# Maze Navigation (step-by-step images)
bash uniworld_general.sh "0,1" "uni_mmmu_maze100_visual_cot" "/mnt/data/maze"

# Sliding Puzzle (move visualization)
bash uniworld_general.sh "0,1" "uni_mmmu_sliding54_visual_cot" "/mnt/data/sliding"
```

## 🔧 Configuration Options

### Multi-GPU Setup

UniWorld uses model parallelism to distribute the model across GPUs:

```bash
# 2 GPUs - each GPU holds ~50% of model
bash uniworld.sh "0,1"

# 4 GPUs - each GPU holds ~25% of model
bash uniworld.sh "0,1,2,3"
```

### Custom Model Path

If you have a local copy of UniWorld:

```bash
bash uniworld.sh "0" "./path/to/local/UniWorld-V1"
```

### Environment Variables

The scripts automatically set:
- `CUDA_VISIBLE_DEVICES`: Controls which GPUs to use
- `MASTER_PORT`: Port for distributed training (29601 for uniworld.sh, 29602 for uniworld_general.sh)
- `NCCL_P2P_DISABLE=1`: Disables P2P for compatibility
- `NCCL_IB_DISABLE=1`: Disables InfiniBand
- `GLOO_USE_IPV6=0`: Disables IPv6

## 📂 Output Structure

After running evaluation, you'll get:

```
output_path/
├── results.json                    # Final metrics
├── samples_*.jsonl                 # Individual sample results
└── images/                         # (Visual CoT tasks only)
    ├── task_name_0.png
    ├── task_name_1.png
    └── ...
```

## 🐛 Troubleshooting

### OOM (Out of Memory) Issues

If you encounter OOM errors:

1. **Use more GPUs** (model parallelism):
   ```bash
   bash uniworld.sh "0,1,2,3"  # 4 GPUs
   ```

2. **Reduce batch size** (edit script, change `BATCH_SIZE`):
   ```bash
   BATCH_SIZE=1  # Already at minimum for generation tasks
   ```

3. **Use quantization** (if supported):
   ```bash
   # Not yet implemented for UniWorld, but can be added
   ```

### Model Loading Issues

If model fails to load:

1. **Check UniWorld repo is cloned**:
   ```bash
   ls UniWorld/  # Should show UniWorld-V1/ directory
   ```

2. **Verify model path**:
   ```bash
   # For HuggingFace (downloads automatically)
   bash uniworld.sh "0" "LanguageBind/UniWorld-V1"
   
   # For local path
   bash uniworld.sh "0" "./UniWorld/UniWorld-V1"
   ```

### Port Conflicts

If you get "port already in use" errors:

Edit the script and change `MASTER_PORT`:
```bash
export MASTER_PORT=29700  # Use a different port
```

## 📊 Expected Performance

| Task | Metric | Expected Score |
|------|--------|----------------|
| ChartQA100 | Relaxed Accuracy | ~70-75% |
| Jigsaw (Visual CoT) | Text Accuracy | ~60-70% |
| Maze (Visual CoT) | Text Exact | ~40-50% |
| Sliding (Visual CoT) | Text Exact | ~30-40% |

*Note: Actual scores may vary based on model version and evaluation settings*

## 🔗 Related Scripts

- `logs` - Bagel Maze evaluation (multi-GPU)
- `examples/models/bagel_maze_quantized.sh` - Bagel with quantization
- `examples/models/uniworld_chartqa.sh` - Original UniWorld ChartQA script
- `examples/models/uniworld_unimmmu_jigsaw.sh` - Original UniWorld Jigsaw script

## 📝 Notes

1. **Visual CoT tasks** automatically generate intermediate images
2. **Batch size** should stay at 1 for generation tasks
3. **Mixed precision** (BF16) is enabled by default for faster inference
4. **Model parallelism** is automatic - model is distributed across all visible GPUs

---

For more details, see:
- UniWorld repo: https://github.com/LanguageBind/UniWorld
- lmms-eval docs: `docs/README.md`
