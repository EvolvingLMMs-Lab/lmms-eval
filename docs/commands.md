# User Guide
This document details the interface exposed by `lmms_eval` and provides details on what flags are available to users.

## Command-line Interface


Equivalently, running the library can be done via the `lmms_eval` entrypoint at the command line.

This mode supports a number of command-line arguments, the details of which can be also be seen via running with `-h` or `--help`:

- `--model` : Selects which model type or provider is evaluated. Must be a string corresponding to the name of the model type/provider being used. See [the main README](https://github.com/EleutherAI/lm-evaluation-harness/tree/main#model-apis-and-inference-servers) for a full list of enabled model names and supported libraries or APIs.

* `--model_args` : Controls parameters passed to the model constructor. Accepts a string containing comma-separated keyword arguments to the model class of the format `"arg1=val1,arg2=val2,..."`, such as, for example `--model_args pretrained=liuhaotian/llava-v1.5-7b,batch_size=1`. For a full list of what keyword arguments, see the initialization of the corresponding model class in `lmms_eval/models/`.

* `--tasks` : Determines which tasks or task groups are evaluated. Accepts a comma-separated list of task names or task group names. Must be solely comprised of valid tasks/groups. You can use `--tasks list` to see all the available tasks. If you add your own tasks but not shown on the list, you can try to set `--verbosity=DEBUG` to view the error message. You can also use `--tasks list_with_num` to check every tasks and the number of question each task contains. However, `list_with_num` will download all the available datasets and may require lots of memory and time.

- `--num_fewshot` : Sets the number of few-shot examples to place in context. Must be an integer.

- `--gen_kwargs` : takes an arg string in same format as `--model_args` and creates a dictionary of keyword arguments. These will be passed to the models for all called `generate_until` (free-form or greedy generation task) tasks, to set options such as the sampling temperature or `top_p` / `top_k`. For a list of what args are supported for each model type, reference the respective library's documentation (for example, the documentation for `transformers.AutoModelForCausalLM.generate()`.) These kwargs will be applied to all `generate_until` tasks called--we do not currently support unique gen_kwargs or batch_size values per task in a single run of the library. To control these on a per-task level, set them in that task's YAML file.

- `--batch_size` : Sets the batch size used for evaluation. Can be a positive integer or `"auto"` to automatically select the largest batch size that will fit in memory, speeding up evaluation. One can pass `--batch_size auto:N` to re-select the maximum batch size `N` times during evaluation. This can help accelerate evaluation further, since `lm-eval` sorts documents in descending order of context length.

- `--max_batch_size` : Sets the maximum batch size to try to fit in memory, if `--batch_size auto` is passed.

- `--device` : Sets which device to place the model onto. Must be a string, for example, `"cuda", "cuda:0", "cpu", "mps"`. Defaults to "cuda", and can be ignored if running multi-GPU or running a non-local model type.

- `--output_path` : A string of the form `dir/file.jsonl` or `dir/`. Provides a path where high-level results will be saved, either into the file named or into the directory named. If `--log_samples` is passed as well, then per-document outputs and metrics will be saved into the directory as well.

- `--log_samples` : If this flag is passed, then the model's outputs, and the text fed into the model, will be saved at per-document granularity. Must be used with `--output_path`.

- `--limit` : Accepts an integer, or a float between 0.0 and 1.0 . If passed, will limit the number of documents to evaluate to the first X documents (if an integer) per task or first X% of documents per task. Useful for debugging, especially on costly API models.

- `--use_cache` : Should be a path where a sqlite db file can be written to. Takes a string of format `/path/to/sqlite_cache_` in order to create a cache db at `/path/to/sqlite_cache_rank{i}.db` for each process (0-NUM_GPUS). This allows results of prior runs to be cached, so that there is no need to re-run results in order to re-score or re-run a given (model, task) pair again.

- `--cache_requests` : Can be "true", "refresh", or "delete". "true" means that the cache should be used. "refresh" means that you wish to regenerate the cache, which you should run if you change your dataset configuration for a given task. "delete" will delete the cache. Cached files are stored under lm_eval/cache/.cache unless you specify a different path via the environment variable: `LM_HARNESS_CACHE_PATH`. e.g. `LM_HARNESS_CACHE_PATH=~/Documents/cache_for_lm_harness`.

- `--check_integrity` : If this flag is used, the library tests for each task selected are run to confirm task integrity.

- `--write_out` : Used for diagnostic purposes to observe the format of task documents passed to a model. If this flag is used, then prints the prompt and gold target string for the first document of each task.

- `--show_config` : If used, prints the full `lm_eval.api.task.TaskConfig` contents (non-default settings the task YAML file) for each task which was run, at the completion of an evaluation. Useful for when one is modifying a task's configuration YAML locally to transmit the exact configurations used for debugging or for reproducibility purposes.

- `--include_path` : Accepts a path to a folder. If passed, then all YAML files containing `lm-eval` compatible task configurations will be added to the task registry as available tasks. Used for when one is writing config files for their own task in a folder other than `lm_eval/tasks/`.

- `--system_instruction`: Specifies a system instruction string to prepend to the prompt.

- `--apply_chat_template` : This flag specifies whether to apply a chat template to the prompt. It can be used in the following ways:
	- `--apply_chat_template` : When used without an argument, applies the only available chat template to the prompt. For Hugging Face models, if no dedicated chat template exists, the default chat template will be applied.
	- `--apply_chat_template template_name` : If the model has multiple chat templates, apply the specified template to the prompt.

    For Hugging Face models, the default chat template can be found in the [`default_chat_template`](https://github.com/huggingface/transformers/blob/fc35907f95459d7a6c5281dfadd680b6f7b620e3/src/transformers/tokenization_utils_base.py#L1912) property of the Transformers Tokenizer.

- `--fewshot_as_multiturn` : If this flag is on, the Fewshot examples are treated as a multi-turn conversation. Questions are provided as user content and answers are provided as assistant responses. Requires `--num_fewshot` to be set to be greater than 0, and `--apply_chat_template` to be on.

- `--predict_only`: Generates the model outputs without computing metrics. Use with `--log_samples` to retrieve decoded results.

* `--seed`: Set seed for python's random, numpy and torch.  Accepts a comma-separated list of 3 values for python's random, numpy, and torch seeds, respectively, or a single integer to set the same seed for all three.  The values are either an integer or 'None' to not set the seed. Default is `0,1234,1234` (for backward compatibility).  E.g. `--seed 0,None,8` sets `random.seed(0)` and `torch.manual_seed(8)`. Here numpy's seed is not set since the second value is `None`.  E.g, `--seed 42` sets all three seeds to 42.

* `--wandb_args`:  Tracks logging to Weights and Biases for evaluation runs and includes args passed to `wandb.init`, such as `project` and `job_type`. Full list [here](https://docs.wandb.ai/ref/python/init). e.g., ```--wandb_args project=test-project,name=test-run```

* `--hf_hub_log_args` : Logs evaluation results to Hugging Face Hub. Accepts a string with the arguments separated by commas. Available arguments:
    * `hub_results_org` - organization name on Hugging Face Hub, e.g., `EleutherAI`. If not provided, the results will be pushed to the owner of the Hugging Face token,
    * `hub_repo_name` - repository name on Hugging Face Hub (deprecated, `details_repo_name` and `results_repo_name` should be used instead), e.g., `lm-eval-results`,
    * `details_repo_name` - repository name on Hugging Face Hub to store details, e.g., `lm-eval-results`,
    * `results_repo_name` - repository name on Hugging Face Hub to store results, e.g., `lm-eval-results`,
    * `push_results_to_hub` - whether to push results to Hugging Face Hub, can be `True` or `False`,
    * `push_samples_to_hub` - whether to push samples results to Hugging Face Hub, can be `True` or `False`. Requires `--log_samples` to be set,
    * `public_repo` - whether the repository is public, can be `True` or `False`,
    * `leaderboard_url` - URL to the leaderboard, e.g., `https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard`.
    * `point_of_contact` - Point of contact for the results dataset, e.g., `yourname@example.com`.
    * `gated` - whether to gate the details dataset, can be `True` or `False`.

## External Library Usage

We also support using the library's external API for use within model training loops or other scripts.

`lmms_eval` supplies two functions for external import and use: `lmms_eval.evaluate()` and `lmms_eval.simple_evaluate()`.

`simple_evaluate()` can be used by simply creating an `lmms_eval.api.model.LM` subclass that implements the methods described in the [Model Guide](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs/model_guide.md), and wrapping your custom model in that class as follows:

```python
import lmms_eval
...

my_model = initialize_my_model() # create your model (could be running finetuning with some custom modeling code)
...
# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LMM.loglikelihood()`
# - `Your_LMM.generate_until()`
lmm_obj = Your_LMM(model=my_model, batch_size=16)

# indexes all tasks from the `lmms_eval/tasks` subdirectory.
# Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
# to include a set of tasks in a separate directory.
task_manager = lmms_eval.tasks.TaskManager()

# Setting `task_manager` to the one above is optional and should generally be done
# if you want to include tasks from paths other than ones in `lmms_eval/tasks`.
# `simple_evaluate` will instantiate its own task_manager if it is set to None here.
results = lmms_eval.simple_evaluate( # call simple_evaluate
    model=lmm_obj,
    tasks=["taskname1", "taskname2"],
    num_fewshot=0,
    task_manager=task_manager,
    ...
)
```

See the `simple_evaluate()` and `evaluate()` functions in [lmms_eval/evaluator.py](../lmms_eval/evaluator.py#:~:text=simple_evaluate) for a full description of all arguments available. All keyword arguments to simple_evaluate share the same role as the command-line flags described previously.

Additionally, the `evaluate()` function offers the core evaluation functionality provided by the library, but without some of the special handling and simplification + abstraction provided by `simple_evaluate()`.

As a brief example usage of `evaluate()`:

```python
import lmms_eval

# suppose you've defined a custom lm_eval.api.Task subclass in your own external codebase
from my_tasks import MyTask1
...

# create your model (could be running finetuning with some custom modeling code)
my_model = initialize_my_model()
...

# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`
lmm_obj = Your_LMM(model=my_model, batch_size=16)

# optional: the task_manager indexes tasks including ones
# specified by the user through `include_path`.
task_manager = lmms_eval.tasks.TaskManager(
    include_path="/path/to/custom/yaml"
    )

# To get a task dict for `evaluate`
task_dict = lmms_eval.tasks.get_task_dict(
    [
        "mmlu", # A stock task
        "my_custom_task", # A custom task
        {
            "task": ..., # A dict that configures a task
            "doc_to_text": ...,
            },
        MyTask1 # A task object from `lm_eval.task.Task`
        ],
    task_manager # A task manager that allows lm_eval to
                 # load the task during evaluation.
                 # If none is provided, `get_task_dict`
                 # will instantiate one itself, but this
                 # only includes the stock tasks so users
                 # will need to set this if including
                 # custom paths is required.
    )

results = evaluate(
    lm=lmm_obj,
    task_dict=task_dict,
    ...
)
```

## Usage with SRT API

> install sglang

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang;
pip install -e "python[srt]"
python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```

> run sglang backend service with the following command

```bash

CKPT_PATH=$1
TASK=$2
MODALITY=$3
TP_SIZE=$4
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

python3 -m lmms_eval \
    --model srt_api \
    --model_args modality=$MODALITY,model_version=$CKPT_PATH,tp=$TP_SIZE,host=127.0.0.1,port=30000,timeout=600 \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/

```

You may need to install some dependencies for the above command to work (if you encounter some errors).

```bash
pip install httpx==0.23.3
pip install protobuf==3.20
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```


## Usage with SRT API

> install sglang

```bash
git clone https://github.com/sgl-project/sglang.git
# Current version is tested on #1222
cd sglang;
pip install -e "python[srt]"

# Install FlashInfer CUDA kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

> run sglang backend service with the following command

```bash
# After update, there is no need to use an extra command to setup backend server
# the server will be initialized in the init process

# launch lmms-eval srt_api model
CKPT_PATH=$1
TASK=$2
MODALITY=$3
TP_SIZE=$4
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

python3 -m lmms_eval \
    --model srt_api \
    --model_args modality=$MODALITY,model_version=$CKPT_PATH,tp=$TP_SIZE,host=127.0.0.1,port=30000,timeout=600 \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
```

You may need to install some dependencies for the above command to work (if you encounter some errors).

```bash
pip install httpx==0.23.3
pip install protobuf==3.20
```

## Regression Test

Now after each PR, we need to run the regression test to make sure the performance of the model is not degraded.

```bash
python3 tools/regression.py
```

```bash
Already on 'dev/fix_output_path'

|task|llava-onevision-qwen2-0.5b-ov|
|--|--|
|ocrbench (dev/fix_output_path)|0.70 ± 0.70|
|mmmu_val (dev/fix_output_path)|50.00 ± 50.00|
|ai2d (dev/fix_output_path)|50.00 ± 50.00|
|muirbench (dev/fix_output_path)|12.50 ± 12.50|
|videomme (dev/fix_output_path)|2500.00 ± 2500.00|

|branch|runtime|%|
|--|--|--|
|dev/fix_output_path|87.7s|100%|
```