# Task Configuration

The `lmms_eval` is meant to be an extensible and flexible framework within which many different evaluation tasks can be defined. All tasks in the new version of the harness are built around a YAML configuration file format.

These YAML configuration files, along with the current codebase commit hash, are intended to be shareable such that providing the YAML config enables another researcher to precisely replicate the evaluation setup used by another, in the case that the prompt or setup differs from standard `lmms_eval` task implementations.

While adding a standard evaluation task on a new dataset can be occasionally as simple as swapping out a Hugging Face dataset path in an existing file, more specialized evaluation setups also exist. Here we'll provide a crash course on the more advanced logic implementable in YAML form available to users.

## Good Reference Tasks

Contributing a new task can be daunting! Luckily, much of the work has often been done for you in a different, similarly evaluated task. Good examples of task implementations to study include:

Generation-based tasks:

- MME (`lmms_eval/tasks/mme/mme.yaml`)

```yaml
dataset_path: lmms-lab/MME
dataset_kwargs:
  token: True
task: "mme"
test_split: test
output_type: generate_until
doc_to_visual: !function utils.mme_doc_to_visual
doc_to_text: !function utils.mme_doc_to_text
doc_to_target: "answer"
generation_kwargs:
  max_new_tokens: 16
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false
# The return value of process_results will be used by metrics
process_results: !function utils.mme_process_results
# Note that the metric name can be either a registed metric function (such as the case for GQA) or a key name returned by process_results
metric_list:
  - metric: mme_percetion_score
    aggregation: !function utils.mme_aggregate_results
    higher_is_better: true
  - metric: mme_cognition_score
    aggregation: !function utils.mme_aggregate_results
    higher_is_better: true
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer the question using a single word or phrase."
  qwen_vl:  
    pre_prompt: ""
    post_prompt: " Answer:"
metadata:
  - version: 0.0
```

You can pay special attention to the `process_results` and `metric_list` fields, which are used to define how the model output is post-processed and scored.
Also, the `lmms_eval_specific_kwargs` field is used to define model-specific prompt configurations. The default is set to follow Llava.

PPL-based tasks:
- Seedbench (`lmms_eval/tasks/seedbench/seedbench_ppl.yaml`)

```yaml
dataset_path: lmms-lab/SEED-Bench
dataset_kwargs:
  token: True
task: "seedbench_ppl"
test_split: test
output_type: multiple_choice
doc_to_visual: !function utils.seed_doc_to_visual
doc_to_text: !function utils.seed_doc_to_text_mc
doc_to_choice : !function utils.seed_doc_to_choice
doc_to_target: !function utils.seed_doc_to_mc_target
# Note that the metric name can be either a registed metric function (such as the case for GQA) or a key name returned by process_results
metric_list:
  - metric: acc
metadata:
  - version: 0.0
```

## Configurations

Tasks are configured via the `TaskConfig` object. Below, we describe all fields usable within the object, and their role in defining a task.

### Parameters

Task naming + registration:
- **task** (`str`, defaults to None) — name of the task.
- **group** (`str`, *optional*) — name of the task group(s) a task belongs to. Enables one to run all tasks with a specified tag or group name at once.

Dataset configuration options:
- **dataset_path** (`str`) — The name of the dataset as listed by HF in the datasets Hub.
- **dataset_name**  (`str`, *optional*, defaults to None) — The name of what HF calls a “config” or sub-task of the benchmark. If your task does not contain any data instances, just leave this to default to None. (If you're familiar with the HF `datasets.load_dataset` function, these are just the first 2 arguments to it.)
- **dataset_kwargs** (`dict`, *optional*) — Auxiliary arguments that `datasets.load_dataset` accepts. This can be used to specify arguments such as `data_files` or `data_dir` if you want to use local datafiles such as json or csv.
- **training_split** (`str`, *optional*) — Split in the dataset to use as the training split.
- **validation_split** (`str`, *optional*) — Split in the dataset to use as the validation split.
- **test_split** (`str`, *optional*) — Split in the dataset to use as the test split.
- **fewshot_split** (`str`, *optional*) — Split in the dataset to draw few-shot exemplars from. assert that this not None if num_fewshot > 0. **This function is not well tested so far**
- **process_docs** (`Callable`, *optional*) — Optionally define a function to apply to each HF dataset split, to preprocess all documents before being fed into prompt template rendering or other evaluation steps. Can be used to rename dataset columns, or to process documents into a format closer to the expected format expected by a prompt template.

Prompting / in-context formatting options:
- **doc_to_text** (`Union[Callable, str]`, *optional*) — Column name or function to process a sample into the appropriate input for the model
- **doc_to_visial** (`Union[Callable, str]`, *optional*) — Function to process a sample into the appropriate input images for the model.
- **doc_to_target** (`Union[Callable, str]`, *optional*) — Column name or or function to process a sample into the appropriate target output for the model. For multiple choice tasks, this should return an index into
- **doc_to_choice** (`Union[Callable, str]`, *optional*) — Column name or or function to process a sample into a list of possible string choices for `multiple_choice` tasks. Left undefined for `generate_until` tasks.

Runtime configuration options:
- **num_fewshot** (`int`, *optional*, defaults to 0) — Number of few-shot examples before the input. **This function is not well tested so far**
- **batch_size** (`int`, *optional*, defaults to 1) — Batch size. 

**So far some models (such as qwen) may not support batch size > 1. Some models (such as llava) will generate different scores for different batch sizes. We recommend setting batch size to 1 for final benchmarking runs.** 

Scoring details:
- **metric_list** (`str`, *optional*, defaults to None) — A list of metrics to use for evaluation.
- **output_type** (`str`, *optional*, defaults to "generate_until") — Selects the type of model output for the given task. Options are `generate_until`, `loglikelihood`, and `multiple_choice`.
- **generation_kwargs** (`dict`, *optional*) — Auxiliary arguments for the `generate` function from HF transformers library. Advanced keyword arguments may not be supported for non-HF LM classes.
