# AnimalKingdom Tasks

The `animalkingdom` directory defines evaluation tasks for the **AnimalKingdom** video dataset introduced in the [AnimalKingdom paper]( https://arxiv.org/abs/2204.08129).  
AnimalKingdom targets video understanding of alpine wildlife across three subtasks:

- **Animal recognition** – identify the species visible in a clip.
- **Action recognition** – list fine‑grained movements or behaviors.
- **Activity recognition** – describe higher‑level activities (e.g., foraging, resting).

## Contents

- **dataset_builder.py** – Converts `_cot.jsonl` annotations into Hugging Face–ready datasets and can optionally copy video clips. Supports predefined datasets (`MammAlps`, `mammalnet`, `AnimalKingdom`) or custom configurations.
- **animalkingdom.yaml** – Groups the three AnimalKingdom subtasks so they can be run together.
- **animalkingdom_animal.yaml**, **animalkingdom_action.yaml**, **animalkingdom_activity.yaml** – Task configs pointing to `luciehmct/AnimalKingdom`; they wire up helper functions from `utils.py` and score predictions with a strict Jaccard index.
- **utils.py** – Shared utilities:
  - `animalkingdom_doc_to_visual` downloads a clip if missing.
  - `animalkingdom_doc_to_text` / `animalkingdom_doc_to_target` extract prompts and answers for each subtask.
  - `animalkingdom_process_results` parses the “Final answer: [...]” list, logs predictions, and computes the Jaccard metric.
  - `animalkingdom_jaccard_aggregation` averages scores across examples.

## Evaluation

Each subtask configuration loads the required records from  
`luciehmct/animalkingdom-test` and runs helper functions in `utils.py`:

- **`doc_to_visual`** → fetches the clip and supplies its path to the model.  
- **`doc_to_text`** → provides the prompt for the current subtask.  
- **`doc_to_target`** → returns the expected label list.  
- **`process_results`** → parses the model response, collects the final answer list, and logs the prompt, response, ground truth, and Jaccard score to  
  `results/<model>_<timestamp>/animalkingdom_<subtask>.jsonl`.  
- **`animalkingdom_jaccard_metric`** → computes the strict Jaccard index.  
- **`animalkingdom_jaccard_aggregation`** → averages scores across examples.  

Example Commands
```bash
# 1. Action recognition with InternVL3
python -m lmms_eval \
  --model internvl3 \
  --model_args "pretrained=OpenGVLab/InternVL3-8B,modality=video,num_frame=32,use_temporal_context=True" \
  --tasks animalkingdom_action \
  --batch_size 1 \
  --output_path "$OUT_DIR" \
  --verbosity INFO

# 2. Animal recognition with InternVL3
python -m lmms_eval \
  --model internvl3 \
  --model_args "pretrained=OpenGVLab/InternVL3-8B,modality=video,num_frame=32,use_temporal_context=True" \
  --tasks animalkingdom_animal \
  --batch_size 1 \
  --output_path "$OUT_DIR"

# 3. Activity recognition with InternVL3
python -m lmms_eval \
  --model internvl3 \
  --model_args "pretrained=OpenGVLab/InternVL3-8B,modality=video,use_temporal_context=True" \
  --tasks animalkingdom_activity \
  --batch_size 1 \
  --output_path "$OUT_DIR"

# 4. Run all three AnimalKingdom subtasks together
python -m lmms_eval \
  --model internvl3 \
  --model_args "pretrained=OpenGVLab/InternVL3-8B,modality=video,num_frame=32,use_temporal_context=True" \
  --tasks animalkingdom \
  --batch_size 1 \
  --output_path "$OUT_DIR"
```

`use_temporal_context` is a flag in the InternVL3 model that lets you embed richer timing information in video prompts.

When enabled (the default), each frame is annotated with its timestamp and the overall video length. The placeholder <video> in the original prompt is replaced by a list like “The video is L second(s) long… Frame‑1 at second t₁: <image>, …” so the model can reason about temporal relationships between frames. If disabled, frames are simply numbered (“Frame1: <image>”), without any notion of when they occur.


## Sources

- `lmms_eval/tasks/animalkingdom/dataset_builder.py`  
- `lmms_eval/tasks/animalkingdom/animalkingdom.yaml`  
- `lmms_eval/tasks/animalkingdom/animalkingdom_action.yaml`  
- `lmms_eval/tasks/animalkingdom/animalkingdom_activity.yaml`  
- `lmms_eval/tasks/animalkingdom/utils.py`

