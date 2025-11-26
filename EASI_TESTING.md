# Testing Bagel on EASI Spatial Reasoning Benchmarks

This guide explains how to evaluate the Bagel model on EASI spatial reasoning benchmarks integrated into lmms-eval.

## Benchmarks Included

| Benchmark | Task Name | Description |
|-----------|-----------|-------------|
| **MMSI** | `mmsibench` | Multi-modal Spatial Intelligence Benchmark |
| **OmniSpatial** | `embspatialbench` | Embodied spatial reasoning in indoor environments |
| **MindCube** | `mindcubebench` | 3D spatial reasoning with cube arrangements |
| **SpatialViz** | `spatial457` | 7-level hierarchical spatial understanding |

## Prerequisites

1. **Install lmms-eval** (if not already done):
   ```bash
   cd G:\Uni-MMMU\lmms-eval
   uv sync
   ```

2. **Clone Bagel repository** (if not already done):
   ```bash
   cd G:\Uni-MMMU\lmms-eval
   git clone https://github.com/ByteDance-Seed/Bagel.git
   ```

3. **Download Bagel model weights**:
   - Download from: [Bagel releases](https://github.com/ByteDance-Seed/Bagel)
   - Note the path to the model directory (e.g., `/path/to/BAGEL-7B-MoT`)

## Quick Start

### Option 1: Python Script (Recommended)

```bash
# Test all benchmarks
python test_bagel_easi.py --model_path /path/to/BAGEL-7B-MoT

# Quick test with 10 samples per task
python test_bagel_easi.py --model_path /path/to/BAGEL-7B-MoT --limit 10

# Test specific benchmark
python test_bagel_easi.py --model_path /path/to/BAGEL-7B-MoT --tasks mmsibench

# Specify output directory
python test_bagel_easi.py --model_path /path/to/BAGEL-7B-MoT --output_dir ./my_results
```

### Option 2: Bash Script

```bash
# Set model path
export MODEL_PATH=/path/to/BAGEL-7B-MoT

# Run all benchmarks
bash test_bagel_easi.sh

# Quick test (set limit)
export LIMIT=10
bash test_bagel_easi.sh
```

### Option 3: Direct Command

Test individual benchmarks directly:

```bash
# MMSIBench (MMSI)
python -m lmms_eval \
    --model bagel \
    --model_args pretrained=/path/to/BAGEL-7B-MoT \
    --tasks mmsibench \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./logs/

# EmbSpatialBench (OmniSpatial)
python -m lmms_eval \
    --model bagel \
    --model_args pretrained=/path/to/BAGEL-7B-MoT \
    --tasks embspatialbench \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./logs/

# MindCubeBench (MindCube)
python -m lmms_eval \
    --model bagel \
    --model_args pretrained=/path/to/BAGEL-7B-MoT \
    --tasks mindcubebench \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./logs/

# Spatial457 (SpatialViz)
python -m lmms_eval \
    --model bagel \
    --model_args pretrained=/path/to/BAGEL-7B-MoT \
    --tasks spatial457 \
    --batch_size 1 \
    --device cuda:0 \
    --output_path ./logs/
```

## Command Line Options

### Python Script Options

```
--model_path      Path to Bagel model (required)
--model_name      Model name (default: bagel)
--tasks           Specific tasks to run (default: all)
--batch_size      Batch size (default: 1)
--device          Device (default: cuda:0)
--output_dir      Output directory (default: ./logs/bagel_easi_results)
--limit           Limit samples per task (default: None)
```

### Environment Variables (Bash Script)

```bash
MODEL_PATH        Path to Bagel model (required)
BATCH_SIZE        Batch size (default: 1)
OUTPUT_DIR        Output directory (default: ./logs/bagel_easi_results)
DEVICE            Device (default: cuda:0)
LIMIT             Limit samples per task (optional)
```

## Output

Results will be saved to the output directory with:
- Evaluation logs: `{task}_{timestamp}.log`
- Result files: JSON files with predictions and scores
- Summary: Aggregated metrics for each benchmark

## Expected Metrics

Each benchmark reports different metrics:

- **MMSIBench**: `accuracy` - Overall accuracy across all spatial categories
- **EmbSpatialBench**: `accuracy` - Accuracy on embodied spatial reasoning
- **MindCubeBench**: `accuracy` - Accuracy on 3D cube reasoning
- **Spatial457**: `accuracy`, `L1_single_acc`, `L2_objects_acc`, etc. - Per-level accuracies

## Troubleshooting

### Model Path Not Found
```
ERROR: Model path does not exist: /path/to/BAGEL-7B-MoT
```
**Solution**: Verify the model path and ensure the model is downloaded.

### Bagel Repository Not Found
```
WARNING: Bagel repository not found at .../Bagel
```
**Solution**: Clone the Bagel repository:
```bash
cd G:\Uni-MMMU\lmms-eval
git clone https://github.com/ByteDance-Seed/Bagel.git
```

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**:
- Reduce batch size: `--batch_size 1`
- Use smaller model variant
- Free up GPU memory

### Import Errors
```
ModuleNotFoundError: No module named 'xxx'
```
**Solution**: Ensure environment is set up:
```bash
cd G:\Uni-MMMU\lmms-eval
uv sync
```

## Example Output

```
==========================================
Configuration
==========================================
Model: bagel
Model Path: /path/to/BAGEL-7B-MoT
Tasks: mmsibench, embspatialbench, mindcubebench, spatial457
Batch Size: 1
Device: cuda:0
Output Directory: ./logs/bagel_easi_results
==========================================

============================================================
Task: mmsibench
Description: MMSI - Multi-modal Spatial Intelligence
Log: ./logs/bagel_easi_results/mmsibench_20231125_143022.log
============================================================
...
✓ mmsibench completed successfully

Results:
{
    "accuracy": 0.65,
    "Pos-Cam-Cam": 0.70,
    "Pos-Obj-Obj": 0.62,
    ...
}

==========================================
Summary
==========================================
Total tasks: 4
Completed: 4
Failed: 0

✓ All tasks completed successfully!

Results saved to: ./logs/bagel_easi_results
```

## Notes

- First run will download datasets automatically
- Evaluation can take several hours for full benchmarks
- Use `--limit` for quick testing during development
- Results are cached to avoid re-evaluation
