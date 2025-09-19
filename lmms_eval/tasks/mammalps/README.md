# MammAlps Tasks

The `mammalps` directory defines evaluation tasks for the **MammAlps** video dataset introduced in the [MammAlps paper](https://arxiv.org/pdf/2503.18223).  
MammAlps targets video understanding of alpine wildlife across three subtasks:

- **Animal recognition** – identify the species visible in a clip.
- **Action recognition** – list fine‑grained movements or behaviors.
- **Activity recognition** – describe higher‑level activities (e.g., foraging, resting).

## Contents

- **dataset_builder.py** – Converts `_cot.jsonl` annotations into Hugging Face–ready datasets and can optionally copy video clips. Supports predefined datasets (`animalkingdom`, `mammalnet`, `mammalps`) or custom configurations.
- **mammalps.yaml** – Groups the three MammAlps subtasks so they can be run together.
- **mammalps_animal.yaml**, **mammalps_action.yaml**, **mammalps_activity.yaml** – Task configs pointing to `luciehmct/mammalps`; they wire up helper functions from `utils.py` and score predictions with a strict Jaccard index.
- **utils.py** – Shared utilities:
  - `mammalps_doc_to_visual` downloads a clip if missing.
  - `mammalps_doc_to_text` / `mammalps_doc_to_target` extract prompts and answers for each subtask.
  - `mammalps_process_results` parses the “Final answer: [...]” list, logs predictions, and computes the Jaccard metric.
  - `mammalps_jaccard_aggregation` averages scores across examples.

## Building Datasets

```bash
# Build all datasets with separate train/test directories
python3 dataset_builder.py

# Build only the MammAlps unified dataset
python3 dataset_builder.py --dataset mammalps --unified

# Build and upload a unified dataset
python3 dataset_builder.py --dataset mammalps --unified
```

## Key options

| Flag             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--dataset / -d` | animalkingdom, mammalnet, mammalps, or all (default: all)                   |
| `--split / -s`   | test, train, or both (default: both)                                        |
| `--unified / -u` | Combine train/test into one directory                                       |

---

## Dataset Structure

Input directories must follow:

```bash
{base_dir}/
├── clips/
│   ├── video1.mp4
│   └── ...
├── test/
│   ├── action_recognition_cot.jsonl
│   ├── animal_recognition_cot.jsonl
│   └── ...
└── train/
    ├── action_recognition_cot.jsonl
    ├── animal_recognition_cot.jsonl
    └── ...
```

Unified outputs are saved under {DatasetName}_HF_Dataset_Unified/ with separate train/ and test/ subfolders.
Each record looks like:

```bash
{
  "id": "original_id",
  "clip": "clips/video.mp4",
  "video_id": "video_id",
  "task": {
    "prompt": "...",
    "answer": ["..."]
  }
}
```

## Evaluation

Each subtask configuration loads the required records from  
`luciehmct/mammalps` and runs helper functions in `utils.py`:

- **`doc_to_visual`** → fetches the clip and supplies its path to the model.  
- **`doc_to_text`** → provides the prompt for the current subtask.  
- **`doc_to_target`** → returns the expected label list.  
- **`process_results`** → parses the model response, collects the final answer list, and logs the prompt, response, ground truth, and Jaccard score to  
  `results/<model>_<timestamp>/mammalps_<subtask>.jsonl`.  
- **`mammalps_jaccard_metric`** → computes the strict Jaccard index.  
- **`mammalps_jaccard_aggregation`** → averages scores across examples.  

Example Commands
```bash
# 1. Action recognition with InternVL3
python -m lmms_eval \
  --model internvl3 \
  --model_args "pretrained=OpenGVLab/InternVL3-8B,modality=video,num_frame=32,use_temporal_context=True" \
  --tasks mammalps_action \
  --batch_size 1 \
  --output_path "$OUT_DIR" \
  --verbosity INFO

# 2. Animal recognition with InternVL3
python -m lmms_eval \
  --model internvl3 \
  --model_args "pretrained=OpenGVLab/InternVL3-8B,modality=video,num_frame=32,use_temporal_context=True" \
  --tasks mammalps_animal \
  --batch_size 1 \
  --output_path "$OUT_DIR"

# 3. Activity recognition with InternVL3
python -m lmms_eval \
  --model internvl3 \
  --model_args "pretrained=OpenGVLab/InternVL3-8B,modality=video,use_temporal_context=True" \
  --tasks mammalps_activity \
  --batch_size 1 \
  --output_path "$OUT_DIR"

# 4. Run all three MammAlps subtasks together
python -m lmms_eval \
  --model internvl3 \
  --model_args "pretrained=OpenGVLab/InternVL3-8B,modality=video,num_frame=32,use_temporal_context=True" \
  --tasks mammalps \
  --batch_size 1 \
  --output_path "$OUT_DIR"
```

`use_temporal_context` is a flag in the InternVL3 model that lets you embed richer timing information in video prompts.

When enabled (the default), each frame is annotated with its timestamp and the overall video length. The placeholder <video> in the original prompt is replaced by a list like “The video is L second(s) long… Frame‑1 at second t₁: <image>, …” so the model can reason about temporal relationships between frames. If disabled, frames are simply numbered (“Frame1: <image>”), without any notion of when they occur.

## Citation

If you use the MammAlps dataset or these evaluation tasks, please cite the original paper:

**MammAlps, arXiv:2503.18223**  
<https://arxiv.org/pdf/2503.18223>

---

## Sources

- `lmms_eval/tasks/mammalps/dataset_builder.py`  
- `lmms_eval/tasks/mammalps/mammalps.yaml`  
- `lmms_eval/tasks/mammalps/mammalps_action.yaml`  
- `lmms_eval/tasks/mammalps/mammalps_activity.yaml`  
- `lmms_eval/tasks/mammalps/utils.py`

