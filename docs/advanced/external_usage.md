# External Usage

`lmms_eval` can be used in two ways: as a **CLI tool** for quick tasks like browsing
benchmarks and launching the Web UI, or as a **Python library** for programmatic
access to tasks, datasets, and evaluations.

## Installation

```bash
# From PyPI
pip install lmms-eval
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install "lmms-eval[all]"
```

---

# Part I - CLI

## 1) Preview Available Tasks
```bash
# Flat list of every registered name (tasks + groups + tags)
lmms-eval tasks list
# Markdown table of task groups only
lmms-eval tasks groups

# Markdown table of leaf tasks only (with config path and output type)
lmms-eval tasks subtasks
# Tags only
lmms-eval tasks tags
```

Example output for `tasks subtasks` (truncated):

```
| Task | Config Location | Output Type |
|------|-----------------|-------------|
| mme  | lmms_eval/tasks/mme/mme.yaml | generate_until |
| mmmu_val | lmms_eval/tasks/mmmu/mmmu_val.yaml | generate_until |
| ...  | ... | ... |
```

These commands only read YAML configs - no dataset download happens.

## 2) List Available Models

```bash
# Show all registered model backends (chat, simple, dual-mode)
lmms-eval models

# Include aliases
lmms-eval models --aliases
```

## 3) Launch the Web UI
The Web UI provides a browser-based interface for configuring and running
evaluations interactively. Requires Node.js 18+ (auto-built on first launch).

```bash
# Start the Web UI (opens browser automatically)
lmms-eval ui
# Custom port
lmms-eval ui --port 3000
```

## 4) Interactive Evaluation Wizard

Run `lmms-eval eval` with no arguments to launch a step-by-step wizard that
guides you through model selection, task selection, and options:

```bash
lmms-eval eval
```

The wizard lets you search/filter tasks, shows a command preview, and runs
the evaluation after confirmation.

## 5) Direct Evaluation

Pass arguments directly (same flags as before, fully backward-compatible):

```bash
# New style (with eval subcommand)
lmms-eval eval --model qwen2_5_vl --tasks mme --batch_size 1 --limit 8

# Old style (still works, routes to eval automatically)
lmms-eval --model qwen2_5_vl --tasks mme --batch_size 1 --limit 8
```

## 6) Start the HTTP Eval Server

```bash
lmms-eval serve --host 0.0.0.0 --port 8000
```

## 7) Other Commands

```bash
# Version and environment info
lmms-eval version

# Statistical power analysis for benchmark planning
lmms-eval power --effect-size 0.03 --tasks mme

# Terminal UI (requires textual package)
lmms-eval tui
```

---

# Part II - Python Library

Beyond the CLI, `lmms_eval` can be imported as a Python library. This lets
external projects list benchmarks, load task definitions, download datasets,
iterate over samples, and run evaluations - all programmatically.

## 8) List Available Benchmarks (Python)

Use `TaskManager` to index all built-in tasks without downloading any data:

```python
from lmms_eval.tasks import TaskManager
tm = TaskManager()
# All registered names (tasks + groups + tags)
print(tm.all_tasks)
print(tm.all_subtasks)  # e.g. ['mme', 'mmmu_val', 'mathvista', ...]
print(tm.all_groups)
print(tm.list_all_tasks())
```
No dataset download happens at this stage. `TaskManager` only reads YAML configs
from the `lmms_eval/tasks/` directory to build its index.
## 9) Load a Task and Download Its Dataset
`get_task_dict` instantiates task objects. During construction each task calls
`download()`, which triggers `datasets.load_dataset()` under the hood.
```python
from lmms_eval.tasks import TaskManager, get_task_dict
tm = TaskManager()
task_dict = get_task_dict(["mme"], task_manager=tm)
task = task_dict["mme"]
```
After this call the HuggingFace dataset has been downloaded (or loaded from
cache) and is stored in `task.dataset` as a `datasets.DatasetDict`.
## 10) Iterate Over Benchmark Samples

Each task exposes its splits through accessor methods:

```python
# Check which splits exist
task.has_test_docs()        # True / False
task.has_validation_docs()  # True / False
task.has_training_docs()    # True / False
test_data = task.test_docs()           # full dataset with images/audio
test_data_lite = task.test_docs_no_media()  # same rows, media columns removed
for doc in test_data:
    print(doc.keys())  # e.g. dict_keys(['question', 'answer', 'image', ...])
    break
```
There is also a convenience property that returns whichever split the task uses
for evaluation (test if available, otherwise validation):
```python
eval_data = task.eval_docs            # datasets.Dataset
eval_data_lite = task.eval_docs_no_media  # without media columns
```

## 11) Access Task Configuration

Every task carries its full YAML config as a `TaskConfig` dataclass:

```python
cfg = task.config
cfg.test_split             # "test"
cfg.output_type            # "generate_until"
cfg.metric_list            # [{"metric": "mme_perception_score", ...}, ...]
cfg.generation_kwargs      # {"max_new_tokens": 16, "temperature": 0, ...}
cfg.lmms_eval_specific_kwargs  # per-model prompt variants
```
You can also read a raw YAML config without instantiating the task (and
therefore without downloading data):
```python
raw = tm._get_config("mme")  # returns the parsed YAML as a dict
```

## 12) Load Tasks from a Custom Path
External projects can maintain their own task YAMLs and load them alongside
(or instead of) the built-in tasks:
```python
# Include custom tasks in addition to built-in ones
tm = TaskManager(include_path="/path/to/my/tasks")
tm = TaskManager(include_path="/path/to/my/tasks", include_defaults=False)
tm = TaskManager(include_path=["/path/a", "/path/b"])
```
Task YAMLs in the custom directory follow the same format as built-in tasks.
See the [Task Guide](../guides/task_guide.md) for the full specification.
## 13) Run an Evaluation Programmatically

`simple_evaluate` is the same function the CLI calls internally:

```python
from lmms_eval.evaluator import simple_evaluate
    model="qwen2_5_vl",
    model_args={"pretrained": "Qwen/Qwen2.5-VL-3B-Instruct"},
    tasks=["mme"],
    batch_size=1,
    limit=8,           # set to None for full evaluation
    log_samples=True,  # save per-sample outputs
)
# results["results"] contains per-task metrics
# results["samples"] contains per-sample model outputs (if log_samples=True)
print(results["results"]["mme"])
```

Key parameters:
| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Registered model name (e.g. `"qwen2_5_vl"`, `"vllm"`, `"openai"`) |
| `model_args` | `str \| dict` | Model constructor arguments |
| `tasks` | `list` | Task names, dicts, or Task objects |
| `limit` | `int \| float` | Cap the number of samples per task (useful for testing) |
| `batch_size` | `int` | Inference batch size |
| `task_manager` | `TaskManager` | Pre-configured TaskManager (optional) |
| `gen_kwargs` | `str` | Override generation parameters |
| `predict_only` | `bool` | Generate outputs without computing metrics |
## 14) Remote Evaluation via HTTP Server
For async workflows (e.g. triggering evaluations during training), use the
eval server and client:
```python
# Server side
from lmms_eval.entrypoints import ServerArgs, launch_server
```

```python
# Client side
from lmms_eval.entrypoints import EvalClient
client = EvalClient("http://eval-server:8000")
# Submit a non-blocking evaluation job
job = client.evaluate(
    model="qwen2_5_vl",
    tasks=["mme", "mmmu_val"],
    model_args={"pretrained": "Qwen/Qwen2.5-VL-7B-Instruct"},
)
# Poll or wait for results
result = client.wait_for_job(job["job_id"])
print(result["result"])
```
An async client (`AsyncEvalClient`) is also available for use in async
training loops. See the [v0.6 release notes](lmms-eval-0.6.md) for full
server API documentation.
---

## Quick Reference

| What you need | CLI / Import | Downloads data? |
|---------------|--------------|-----------------|
| List tasks | `lmms-eval tasks list` | No |
| Task table | `lmms-eval tasks subtasks` | No |
| List models | `lmms-eval models` | No |
| Interactive wizard | `lmms-eval eval` (no args) | No |
| Direct evaluation | `lmms-eval eval --model X --tasks Y` | **Yes** |
| Web UI | `lmms-eval ui` | No |
| HTTP server | `lmms-eval serve` | Server-side |
| Power analysis | `lmms-eval power` | No |
| Version info | `lmms-eval version` | No |
| List benchmarks (Python) | `TaskManager().all_subtasks` | No |
| Read raw YAML config | `TaskManager()._get_config(name)` | No |
| Instantiate task + download | `get_task_dict([name])` | **Yes** |
| Iterate samples | `task.test_docs()` | (at construction) |
| Full evaluation (Python) | `simple_evaluate(...)` | **Yes** |
| Remote evaluation (Python) | `EvalClient(url).evaluate(...)` | Server-side |
## Data Flow

```
TaskManager()
  └─ initialize_tasks()        # scan lmms_eval/tasks/**/*.yaml
       └─ index: {name -> yaml_path, type}
  └─ TaskManager.load_task_or_group()
       └─ ConfigurableTask(config)
            └─ download()              # datasets.load_dataset("lmms-lab/MME")
                 └─ self.dataset        # DatasetDict with all splits
                 └─ self.dataset_no_image  # same, media columns stripped
task.config       ->  TaskConfig dataclass     # all YAML fields as attributes
```