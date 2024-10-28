# Task Configuration

The `lmms_eval` is meant to be an extensible and flexible framework within which many different evaluation tasks can be defined. All tasks in the new version of the harness are built around a YAML configuration file format.

These YAML configuration files, along with the current codebase commit hash, are intended to be shareable such that providing the YAML config enables another researcher to precisely replicate the evaluation setup used by another, in the case that the prompt or setup differs from standard `lmms_eval` task implementations.

While adding a standard evaluation task on a new dataset can be occasionally as simple as swapping out a Hugging Face dataset path in an existing file, more specialized evaluation setups also exist. Here we'll provide a crash course on the more advanced logic implementable in YAML form available to users.

## Configurations

Tasks are configured via the `TaskConfig` object. Below, we describe all fields usable within the object, and their role in defining a task.

### Parameters

Task naming + registration:
- **task** (`str`, defaults to None) — name of the task.
- **group** (`str`, *optional*) — name of the task group(s) a task belongs to. Enables one to run all tasks with a specified tag or group name at once. This would be deprecated in the future, and we recommend using `tag` to replace it.
- **task_alias** (`str`, defaults to None) - Alias of the task name that will be printed in the final table results.
- **tag** (`str`, *optional*) — name of the task tags(s) a task belongs to. Enables one to run all tasks with a specified tag name at once. This is a improved naming rule over `group`.

Dataset configuration options:
- **dataset_path** (`str`) — The name of the dataset as listed by HF in the datasets Hub.
- **dataset_name**  (`str`, *optional*, defaults to None) — The name of what HF calls a `config` or `subset` of the benchmark. If your task does not contain any data instances, just leave this to default to None. (If you're familiar with the HF `datasets.load_dataset` function, these are just the first 2 arguments to it.)
- **dataset_kwargs** (`dict`, *optional*) — Auxiliary arguments that `datasets.load_dataset` accepts. This can be used to specify arguments such as `data_files` or `data_dir` if you want to use local datafiles such as json or csv.
- **test_split** (`str`, *optional*) — Split in the dataset to use as the test split. This is required for denoting the `split` of the HF dataset.
- **training_split** (`str`, *optional*) — Split in the dataset to use as the training split.
- **validation_split** (`str`, *optional*) — Split in the dataset to use as the validation split.
- **fewshot_split** (`str`, *optional*) — Split in the dataset to draw few-shot exemplars from. assert that this not None if num_fewshot > 0. **This function is not well tested so far**
- **process_docs** (`Callable`, *optional*) — Optionally define a function to apply to each HF dataset split, to preprocess all documents before being fed into prompt template rendering or other evaluation steps. Can be used to rename dataset columns, or to process documents into a format closer to the expected format expected by a prompt template.

Prompting / in-context formatting options:
- **doc_to_text** (`Union[Callable, str]`, *optional*) — Column name or function to process a sample into the appropriate input for the model. 

  For multi-round generation, (e.g., MMSearch), the function accepts additional parameters about the round index, previous round information and previous model output. It should return the input image for the next round, input text for the next round, a boolean indicating if round inference should terminate, model outputs from all rounds, and extra information from previous rounds.
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

Other:
- **metadata** (`dict`, *optional*) — An optional field where arbitrary metadata can be passed. Most tasks should include a `version` key in this field that is used to denote the version of the yaml config. Other special metadata keys are: `num_fewshot`, to override the printed `n-shot` table column for a task.

## Using Yaml Configurations to Define Tasks

We recomment to browse existing tasks in the `lmms_eval/tasks` folder to get a sense of the different options available. 

Here we will provide some explainations on the existing tasks and how to define new tasks. Here we use MME as an example.

```yaml
dataset_path: lmms-lab/MME # The name of the dataset as listed by HF in the datasets Hub.
dataset_kwargs:
  token: True # Auxiliary arguments that `datasets.load_dataset` accepts. This can be used to specify arguments such as `data_files` or `data_dir` if you want to use local datafiles such as json or csv.
task: "mme" # The name of the task, this should be registered in the task manager. If successful, you can call lmms_eval with this task name by setting `--tasks mme`.
test_split: test # The split of the dataset to use as the test split.
output_type: generate_until # The type of model output for the given task. Options are `generate_until`, `loglikelihood`, and `multiple_choice`.
doc_to_visual: !function utils.mme_doc_to_visual # The function to process a sample into the appropriate input for the model. 
doc_to_text: !function utils.mme_doc_to_text # The function to process a sample into the appropriate target output for the model.
doc_to_target: "answer" # The function to process a sample into a list of possible string choices for `multiple_choice` tasks.
generation_kwargs: # Auxiliary arguments for the `generate` function from HF transformers library. This would be used in different models files.
  max_new_tokens: 16
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false
# The return value of process_results will be used by metrics
process_results: !function utils.mme_process_results
# Note that the metric name can be either a registed metric function (such as the case for GQA) or a key name returned by process_results
# e.g. Following metrics `mme_perception_score` is custom defined. 
# So `mme_process_results` function should return the dict `{"mme_perception_score": {sub_k:sub_v, ..., } }`
# And the `mme_aggregate_results` function could get the dict `{sub_k:sub_v, ..., }`, and use the information to gather the final accuracy.
metric_list:
  - metric: mme_percetion_score # The name of the metric to use for evaluation. The process_results function should return the metric name and the metric value, in format of `{metric_name: results}`. And the aggregation function will use the results to get the final score.
    aggregation: !function utils.mme_aggregate_results # The name of the aggregation function to use for evaluation.
    higher_is_better: true # Whether the metric is better when the value is higher.
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

### Embedded Python Code

As above example shown, you can use python functions for certain arguments by using the `!function` operator after the argument name followed by `<filename>.<pythonfunctionname>`. This feature can be used for the following arguments:
1. `doc_to_text`
2. `doc_to_target`
3. `doc_to_choice`
4. `aggregation` for a `metric` in `metric_list`

You can base a YAML on another YAML file as a template. This can be handy when you need to just change the prompt for `doc_to_text` but keep the rest the same or change `filters` to compare which is better. Simply use `include` in the YAML file and write the name of the template you want to base from. This assumes that the base temeplate is in the same directory. 

Otherwise, You will need to define the full path.

```yaml
include: <YAML filename or with full path>
...
```

### Passing Arguments to Metrics

Metrics can be defined in the `metric_list` argument when building the YAML config. Multiple metrics can be listed along with any auxiliary arguments. For example, setting the [`exact_match` metric](https://github.com/huggingface/evaluate/tree/main/metrics/exact_match), auxiliary arguments such as `ignore_case`, `ignore_punctuation`, `regexes_to_ignore` can be listed as well. They will be added to the metric function as `kwargs`. Some metrics have predefined values for `aggregation` and `higher_is_better` so listing the metric name only can be sufficient.

```yaml
metric_list:
  - metric: acc
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "\\$"
```

### Natively Supported Metrics

Here we list all metrics currently supported natively in `lmms_eval`:

Metrics:
* `acc` (accuracy)
* `acc_norm` (length-normalized accuracy)
* `acc_all` (accuracy metric where all answers must be correct for each question)
* `anls` (average Normalized Levenshtein Similarity, used for evaluating text similarity)
* `acc_mutual_info` (baseline loglikelihood - normalized accuracy)
* `by_pass` (by-pass score, dont calculate anything, just return the model output as the result)
* `exact_match` (exact match score, bind to `output_type: generate_until` and `aggregation: mean`)
* `perplexity`
* `word_perplexity` (perplexity per word)
* `byte_perplexity` (perplexity per byte)
* `bits_per_byte`
* `brier_score` (a scoring rule for probabilistic predictions)
* `matthews_corrcoef` (Matthews correlation coefficient)
* `f1` (F1 score)
* `bleu`
* `chrf`
* `ter`

Aggregation functions:
* `mean`
* `median`
* `perplexity`
* `weighted_perplexity`
* `bits_per_byte`

### Adding a Multiple Choice Metric

Adding a multiple choice metric has a few steps. To get it working you need to:

1. register a metric function
2. register an aggregation function
3. update the `Task` definition to make sure the correct arguments are passed

The default metric and aggregation functions are in `lm_eval/api/metrics.py`, and you can add a function there if it's for general use. The metrics are towards the bottom of the file and look like this:

```python
    @register_metric(
        metric="mcc",
        higher_is_better=True,
        output_type="multiple_choice",
        aggregation="matthews_corrcoef",
    )
    def mcc_fn(items):  # This is a passthrough function
        return items
```
Note that many of these are passthrough functions, and for multiple choice (at least) this function is never actually called.

Aggregation functions are defined towards the top of the file, here's an example:

    @register_aggregation("matthews_corrcoef")
    def matthews_corrcoef(items):
        unzipped_list = list(zip(*items))
        golds = unzipped_list[0]
        preds = unzipped_list[1]
        return sklearn.metrics.matthews_corrcoef(golds, preds)

This function returns a single numeric value. The input is defined in `Task.process_results` in `lm_eval/api/task.py`. There's a section that looks like this:

```python
    result_dict = {
        **({"acc": acc} if "acc" in use_metric else {}),
        **({"f1": (gold, pred)} if "f1" in use_metric else {}),
        **({"mcc": (gold, pred)} if "mcc" in use_metric else {}),
        **({"acc_norm": acc_norm} if "acc_norm" in use_metric else {}),
        **({"exact_match": exact_match} if "exact_match" in use_metric else {}),
    }
```

The value here determines the input to the aggregation function, though the name used matches the metric function. These metrics all have simple needs and just need the accuracy or gold and predicted values, but immediately below this there are examples of metrics with more complicated needs you can use as reference.

## Good Reference Tasks

Contributing a new task can be daunting! Luckily, much of the work has often been done for you in a different, similarly evaluated task. Good examples of task implementations to study include:

**Generation-based tasks:**

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
# e.g. Following metrics `mme_perception_score` is custom defined. 
# So `mme_process_results` function should return the dict `{"mme_perception_score": {sub_k:sub_v, ..., } }`
# And the `mme_aggregate_results` function could get the dict `{sub_k:sub_v, ..., }`, and use the information to gather the final accuracy.
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

And other tasks can be:
- MMBench (`lmms_eval/tasks/mmbench/mmbench.yaml`) (Group: `mmbench`)

**Notes:**
You can pay special attention to the process_results and metric_list fields, which define how the model output is post-processed and scored.

**`process_results`** is executed in parallel (multi-GPU). We recommend using it to collect and parse model outputs into formatted results. If your evaluation requires external models (e.g., GPT-4) as a judge or answer extractor, we also suggest integrating the judging process within this function.

**`aggregate_results`** is executed in the main process (rank 0). We recommend using it to calculate the final score or accuracy.

Also, the `lmms_eval_specific_kwargs` field is used to define model-specific prompt configurations. The default is set to follow Llava.

**Generation-based Tasks (GPT-Eval)**

You can check the following tasks to see how we incoporate GPT4 as judge model into our evaluation pipeline.

- LLaVA-In-The-Wild (https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/llava-in-the-wild/llava-in-the-wild.yaml)

**PPL-based tasks:**
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

**Multi-round-generation-based tasks:**

- MMSearch(`lmms_eval/tasks/mmsearch/mmsearch_end2end.yaml`)

```yaml
dataset_path: CaraJ/MMSearch
dataset_name: end2end
dataset_kwargs:
  token: False
task: "mmsearch_end2end"
test_split: end2end
output_type: generate_until_multi_round # Note that here we use the new output_type here for multi-round generation. It basicly follows generate_until but incorporate multi-round inference
doc_to_visual: !function lmms_eval_utils.mmsearch_end2end_doc_to_visual
doc_to_text: !function lmms_eval_utils.mmsearch_end2end_doc_to_text
doc_to_target: "answer"
generation_kwargs:
  until:
    - "ASSISTANT:"
  max_new_tokens: 512
  temperature: 0
  top_p: 0
  num_beams: 1
  do_sample: false
process_results: !function lmms_eval_utils.mmsearch_end2end_process_results
metric_list:
  - metric: end2end_f1_score
    aggregation: !function lmms_eval_utils.mmsearch_aggregate_results_f1_score
    higher_is_better: true
  - metric: requery_score
    aggregation: !function lmms_eval_utils.mmsearch_aggregate_results_req_score
    higher_is_better: true
lmms_eval_specific_kwargs: # Note that here we cache the result of every sample whenever the it is inferenced
  middle_resules_dir: /data1/zrr/jdz/mmsearch/mmsearch_middile_results
  result_cache_dir: /data1/zrr/jdz/mmsearch/mmsearch_result_cache_dir

```
