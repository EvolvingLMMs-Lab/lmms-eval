# MMSI-Bench (Multi-Modal Spatial Intelligence)

## 1. Task Overview

MMSI-Bench evaluates multi-modal spatial intelligence across 5 task categories:
- **Attribute (Appearance)**: Visual appearance attributes (color, shape, texture, orientation, count)
- **Attribute (Measurement)**: Measurement-related attributes
- **Motion (Object)**: Object motion understanding
- **Motion (Camera)**: Camera motion understanding
- **MSR (Multi-image Spatial Reasoning)**: Multi-image spatial reasoning

## 2. Task Files

### 2.1 Direct Generation Tasks

| Task | YAML File | Description |
|------|-----------|-------------|
| Attribute (Appr.) | `mmsi_attribute_appr.yaml` | Appearance attribute recognition |
| Attribute (Meas.) | `mmsi_attribute_meas.yaml` | Measurement attribute recognition |
| Motion (Obj.) | `mmsi_motion_obj.yaml` | Object motion understanding |
| Motion (Cam.) | `mmsi_motion_cam.yaml` | Camera motion understanding |
| MSR | `mmsi_msr.yaml` | Multi-image spatial reasoning |

### 2.2 Visual CoT Tasks

| Task | YAML File | Description |
|------|-----------|-------------|
| Attribute (Appr.) Visual CoT | `mmsi_attribute_appr_visual_cot.yaml` | Two-stage with visualization |
| Attribute (Meas.) Visual CoT | `mmsi_attribute_meas_visual_cot.yaml` | Two-stage with visualization |
| Motion (Obj.) Visual CoT | `mmsi_motion_obj_visual_cot.yaml` | Two-stage with visualization |
| Motion (Cam.) Visual CoT | `mmsi_motion_cam_visual_cot.yaml` | Two-stage with visualization |
| MSR Visual CoT | `mmsi_msr_visual_cot.yaml` | Two-stage with visualization |

## 3. Prompt Design

### 3.1 Direct Generation

Questions are composed using `pre_prompt + question + post_prompt` format.

**Example (Attribute Appearance):**
```
{pre_prompt}Based on the visual content, please answer the following question about appearance attributes.

{question}

{post_prompt}Based on your visual observation, answer with the option's letter from the given choices directly. Enclose the option's letter within ``.
```

### 3.2 Visual CoT (Two-Stage)

Visual CoT uses `[GEN_PROMPT]...[/GEN_PROMPT][QUESTION]...[/QUESTION]` format.

#### Stage 1: Generation Prompt (No Candidate Leakage)

Each category has a specialized generation prompt. These prompts **do NOT contain answer candidates** to avoid information leakage.

**Attribute (Appearance):**
```
Create a visualization that highlights and labels the visual appearance attributes
(color, shape, texture, orientation, count) in the scene. Use annotations, bounding
boxes, and labels to make object features and counts clearly visible.
```

**Attribute (Measurement):**
```
Create a visualization that highlights and labels measurement attributes in the
scene. Use annotations to show sizes, distances, proportions, and scales clearly.
```

**Motion (Object):**
```
Create a visualization that tracks and highlights object motion in the scene.
Use arrows, trajectories, and annotations to show movement direction, speed,
and path of objects.
```

**Motion (Camera):**
```
Create a visualization that illustrates camera motion characteristics. Use
arrows and annotations to show camera movement type (pan, tilt, zoom, dolly),
direction, and magnitude.
```

**MSR (Multi-image Spatial Reasoning):**
```
Create a visualization that highlights spatial relationships across the images.
Use annotations, connecting lines, and labels to show relative positions,
orientations, and spatial correspondence between elements.
```

#### Stage 2: Question Prompt (Aligned with Direct Generation)

```
You are given the original image(s) and a visualization highlighting [attributes/motion/spatial relationships].
Use both to analyze [specific aspects].

{question}

Based on your visual observation, answer with the option's letter from the given choices directly.
Enclose the option's letter within ``.
```

## 4. Key Functions

| Function | Purpose |
|----------|---------|
| `msr_doc_to_text` | Build prompt for direct generation |
| `msr_doc_to_text_with_gen_prompt` | Build two-stage prompt for Visual CoT |
| `msr_doc_to_visual` | Extract images from document |
| `msr_process_results` | Process and score results |
| `msr_aggregate_results` | Aggregate category-wise scores |

## 5. Evaluation Metrics

- **Accuracy**: Exact match of predicted option letter (A/B/C/D) with ground truth
- **Category-wise Accuracy**: Accuracy aggregated per task category

## 6. Stage 1 Original Image Handling

Visual CoT models receive the original image in Stage 1 through the model implementation:
- `azure_trapi_visual_cot`: Uses `images.edit` API with original image
- `bagel_visual_cot`: Passes original image to `generate_text_and_image`
- `nano_banana_visual_cot`: Uses `images.edit` API with original image

The generation prompt is defined in YAML configuration (`lmms_eval_specific_kwargs.generation_prompt`), NOT in the Python utils, allowing flexible prompt customization per model.

## 7. Running Evaluation

```bash
# Direct generation
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=/path/to/model \
    --tasks mmsi_attribute_appr \
    --batch_size 4 \
    --output_path ./logs/

# Visual CoT
python -m lmms_eval \
    --model azure_trapi_visual_cot \
    --model_args save_intermediate=true \
    --tasks mmsi_attribute_appr_visual_cot \
    --batch_size 1 \
    --output_path ./logs/
```
