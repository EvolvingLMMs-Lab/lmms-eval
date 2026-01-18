# Uni-MMMU Task

Uni-MMMU is a unified multimodal benchmark that tests visual reasoning capabilities through three task types: Jigsaw, Maze, and Sliding Puzzle.

## Task Types

### Jigsaw
- **Input**: Reference 2x2 image with bottom-right cell hidden + 2 candidate patches
- **Goal**: Choose which candidate correctly completes the image
- **Output**: JSON with choice (0 or 1)

### Maze
- **Input**: Maze image with start (blue dot) and goal (green frame)
- **Goal**: Find the path from start to goal
- **Output**: JSON array of moves ["right", "down", ...]

### Sliding Puzzle
- **Input**: 3x3 sliding puzzle with 8 colored tiles and 1 empty space (red)
- **Goal**: Find the sequence of moves to solve the puzzle
- **Output**: JSON array of moves ["down", "right", ...]

## Visual CoT (Chain-of-Thought) Mode

The Visual CoT versions (`*_visual_cot`) are **aligned with the original Uni-MMMU benchmark**. They require **interleaved generation** - the model generates intermediate images as part of its reasoning process.

### Generation Flow (Aligned with Original)

**Jigsaw**:
```
Input (reference + candidates)
    → gen_image (Candidate 0 completion)
    → gen_image (Candidate 1 completion)
    → gen_text (final answer with choice)
```

**Maze** (k = number of ground truth steps):
```
Input (initial maze image)
    → [gen_text (planning) → gen_image (step state)] × k
    → gen_text (final answer with move sequence)
```

**Sliding** (k = number of ground truth steps):
```
Input (initial puzzle image)
    → [gen_text (planning) → gen_image (step state)] × k
    → gen_text (final answer with move sequence)
```

### Key Differences from Standard Mode

| Aspect | Standard Mode | Visual CoT Mode |
|--------|--------------|-----------------|
| Output | Text only | Images + Text |
| Generation | Single text generation | Interleaved text/image |
| Step count | N/A | Dynamic from ground truth |
| Model requirement | Understanding only | Interleaved generation |

### Model Requirements

To run Visual CoT tasks, the model must support **interleaved text-image generation**:

1. **Context Management**: Maintain context across text and image generations
2. **Conditional Generation**: Generate images conditioned on previous context (text + images)
3. **Text-to-Image**: Generate intermediate visualization images
4. **Final Answer**: Generate final text answer after all images

### Bagel Model Implementation

The Bagel model (`lmms_eval/models/simple/bagel.py`) implements interleaved generation via `generate_uni_mmmu_interleaved()`:

```python
# Generation flow for Jigsaw:
# 1. Add input images to context
# 2. Add prompt to context
# 3. Generate image with Candidate 0
# 4. Add generated image to context
# 5. Generate image with Candidate 1
# 6. Add generated image to context
# 7. Generate final text answer

# Generation flow for Maze/Sliding:
# 1. Add input image to context
# 2. Add prompt to context
# 3. For each step:
#    a. Generate planning text
#    b. Add planning text to context
#    c. Generate step image
#    d. Add step image to context
# 4. Generate final text answer
```

### YAML Configuration

Visual CoT tasks use `bagel_interleaved` in `lmms_eval_specific_kwargs`:

```yaml
lmms_eval_specific_kwargs:
  bagel_interleaved:
    task_type: "jigsaw"  # or "maze" or "sliding"
    num_images: 2        # Fallback; actual count from ground truth for maze/sliding
    cfg_text_scale: 4.0  # Generation parameters
    cfg_interval: 0.4
    timestep_shift: 3.0
    num_timesteps: 50
```

For maze and sliding tasks, `num_images` is **dynamically determined** from the ground truth:
- Maze: `len(doc["steps"])`
- Sliding: `len(doc["steps_words"])`

## Running the Tasks

### Standard Mode (Text-only Understanding)
```bash
python -m lmms_eval \
    --model bagel \
    --model_args pretrained=/path/to/BAGEL-7B-MoT,mode=understanding \
    --tasks uni_mmmu_jigsaw100,uni_mmmu_maze100,uni_mmmu_sliding54 \
    --batch_size 1
```

### Visual CoT Mode (Interleaved Generation)
```bash
python -m lmms_eval \
    --model bagel \
    --model_args pretrained=/path/to/BAGEL-7B-MoT,mode=generation \
    --tasks uni_mmmu_jigsaw100_visual_cot,uni_mmmu_maze100_visual_cot,uni_mmmu_sliding54_visual_cot \
    --batch_size 1
```

## Metrics

| Task | Metric | Description |
|------|--------|-------------|
| Jigsaw | `jigsaw_text_acc` | Accuracy of choice (0 or 1) |
| Maze | `maze_text_exact` | Exact match of move sequence |
| Maze | `maze_text_frame_acc` | Per-step accuracy of moves |
| Sliding | `sliding_text_exact` | Exact match of move sequence |
| Sliding | `sliding_text_frame_acc` | Per-step accuracy of moves |

## Reference

This implementation is aligned with the original Uni-MMMU benchmark:
- Repository: https://github.com/xxx/Uni-MMMU
- Evaluation script: `eval_ummmu.py`
- Sample generation code: `sample_code_example/gpt/`
