# CC-OCR V2

CC-OCR V2 evaluates recognition, document parsing, grounding, information extraction, and visual question answering. Its public test split contains 7,093 examples across 74 scenarios. Some questions span multiple pages. [Benchmark and evaluator](https://github.com/Eioss/CC-OCR-V2/tree/58c6f20c464ad8471e49f5faab2b126c7e39556a) · [Dataset](https://huggingface.co/datasets/Eioss/CC-OCR-V2/tree/f80226d958d748daa0bf805ed6f69c2d41823e80)

## Prepare the data

In the source checkout, install the task dependencies and activate the environment.

```bash
uv sync --locked --extra cc_ocr_v2
source .venv/bin/activate
```

From the directory where you will run evaluations, prepare the dataset once.

```bash
python -m lmms_eval.tasks.cc_ocr_v2.prepare_dataset --output-dir ./cc_ocr_v2_data
```

Preparation reads the pinned raw release, checks its file inventory, and writes a Hugging Face DatasetDict with a single `test` split. Each row contains the original question, answer, ordered pages, and a source key identifying its track, subtask, scenario, and sample. Use a new output directory. Set `HF_HUB_OFFLINE=1` to prepare from a complete cached snapshot, or reuse the prepared dataset directly for an offline evaluation. The task loads `./cc_ocr_v2_data` relative to the evaluation process's working directory.

## Run

From the same directory, run the native Qwen3.8 chat backend on a CUDA device that can hold the model and its generation cache.

```bash
MODEL_PATH="$(hf download Qwen/Qwen3.8-27B --quiet --revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0)"
python -m lmms_eval \
  --model qwen3_8 \
  --model_args "pretrained=${MODEL_PATH},device_map=cuda:0,attn_implementation=sdpa,system_prompt=,enable_thinking=false,strict_generation=true,apply_image_pixel_limits=true,min_pixels=65536,max_pixels=16777216,context_length=262144" \
  --tasks cc_ocr_v2 --batch_size 1 --log_samples --fail_on_error --output_path ./results
```

The task passes all pages in order, followed by the original question in one user turn. Preparation converts each page to RGB and encodes it as JPEG at quality 95 with optimization, following the authors' inference script. The command uses an empty system prompt and disables thinking. An empty generation counts as a model answer. The task raises an error if a backend reports an exhausted request or budget. The authors use temperature zero and omit a response-token limit. The task uses greedy decoding with a limit of 24,576 new tokens.

## Scores

All metrics are on a 0–1 scale. Multiply by 100 for comparison with the published leaderboard. Recognition and extraction pool token or field counts **within each scenario** before computing F1. Parsing and QA also weight scenarios equally. Grounding first averages text and object scenarios separately, then gives both subtasks equal weight. `cc_ocr_v2_average` averages the five track scores. A run with a missing track has no overall score. [Official aggregation](https://github.com/Eioss/CC-OCR-V2/blob/58c6f20c464ad8471e49f5faab2b126c7e39556a/scripts/eval_all_tasks_one_model.sh) · [Published scores](https://github.com/Eioss/CC-OCR-V2/blob/58c6f20c464ad8471e49f5faab2b126c7e39556a/all_models_summary.csv)

Per-sample records expose `row_score` and any counts needed by the official metric. The track and overall scores use hierarchical aggregation, so the task does not report row-wise CLT standard errors.
