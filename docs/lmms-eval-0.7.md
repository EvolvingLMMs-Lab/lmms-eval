# LMMs-Eval v0.7

## Overview

v0.7 focuses on making lmms-eval easier to use, share, and reproduce. The theme is **operational simplicity** - fewer flags to remember, fewer things that can go wrong, and a single file that captures your entire experiment.

---

## 1. Better One-Line Evaluation

Running an evaluation used to mean assembling a long command with many flags. Sharing that command with a teammate meant copy-pasting a fragile shell one-liner, hoping the environment variables were set, and trusting the args were right. Reproducing a result from a paper meant reverse-engineering the setup from a results JSON that only stored a few fields.

v0.7 replaces all of that with a single YAML file:

```bash
python -m lmms_eval --config configs/my_experiment.yaml
```

One file. Everything is in it - model, tasks, generation parameters, environment variables. No separate `export` commands, no long CLI flags. Ship the YAML, reproduce the result.

### 1.1 What Goes in the YAML

A config file maps directly to CLI arguments, plus an optional `env` section for environment variables:

```yaml
# configs/example_local.yaml
env:
  HF_HOME: "${HF_HOME:-~/.cache/huggingface}"
  HF_HUB_ENABLE_HF_TRANSFER: "1"

model: qwen2_5_vl
model_args: "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto"
tasks: "mme,mmmu_val"
batch_size: 1
num_fewshot: 0
seed: "42,42,42,42"
gen_kwargs: "temperature=0,max_new_tokens=1024"
output_path: "results/"
log_samples: true
```

For API models with credentials:

```yaml
# configs/example_api.yaml
env:
  OPENAI_API_KEY: "${OPENAI_API_KEY}"
  HF_HOME: "${HF_HOME:-~/.cache/huggingface}"

model: openai
model_args: "model=gpt-4o,max_retries=5"
tasks: "mme,mmmu_val"
batch_size: 1
gen_kwargs: "temperature=0,max_new_tokens=1024"
output_path: "results/"
log_samples: true
```

For batch evaluation (multiple models in one run):

```yaml
# configs/example_batch.yaml
- model: qwen2_5_vl
  model_args: "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto"
  tasks: "mme"
  batch_size: 1
  output_path: "results/qwen25vl/"
  log_samples: true

- model: qwen3_vl
  model_args: "pretrained=Qwen/Qwen3-VL-4B-Instruct,device_map=auto"
  tasks: "mme"
  batch_size: 1
  output_path: "results/qwen3vl/"
  log_samples: true
```

### 1.2 Environment Variables

The `env` section sets environment variables before evaluation starts. This solves the "works on my machine" problem - credentials and paths are part of the config, not floating in someone's `.bashrc`.

```yaml
env:
  OPENAI_API_KEY: "sk-..."           # Literal value
  HF_TOKEN: "${HF_TOKEN}"            # Expand from shell environment
  HF_HOME: "${HF_HOME:-/data/cache}" # Expand with default fallback
```

**Variable expansion**: Values containing `${VAR}` are expanded using the current shell environment. The `${VAR:-default}` syntax provides fallback defaults. This lets you keep secrets out of the YAML while still documenting which variables are needed.

**Sensitive value masking**: Keys containing `KEY`, `TOKEN`, `SECRET`, or `PASSWORD` are masked in log output (`Config env: OPENAI_API_KEY=********`). The actual values are set correctly in `os.environ`; only the log is masked.

### 1.3 CLI Override Priority

The priority chain is: **defaults < YAML < CLI**. CLI arguments always win.

```bash
# YAML says limit: null, but you want a quick test
python -m lmms_eval --config configs/example_local.yaml --limit 5

# YAML says batch_size: 1, override for faster throughput
python -m lmms_eval --config configs/example_local.yaml --batch_size 4
```

This lets you keep a "canonical" config in the YAML and override individual values at the command line without editing the file. The override detection works by comparing each CLI argument against argparse defaults - only arguments that differ from defaults are treated as explicit overrides.

### 1.4 Schema Validation

Unknown keys in the YAML now raise an error with the list of valid keys:

```
ValueError: Unknown keys in config file: ['modle', 'taks'].
Valid keys are: ['batch_size', 'config', 'device', 'gen_kwargs', 'limit', 'log_samples', 'model', 'model_args', ...]
```

Previously, typos like `modle` instead of `model` were silently accepted via `setattr` and ignored at runtime, leading to confusing evaluation failures.

### 1.5 Full Experiment Reproducibility

Results JSON now includes `resolved_cli_args` - the complete resolved configuration after merging defaults, YAML, and CLI overrides:

```json
{
  "config": {
    "model": "qwen2_5_vl",
    "model_args": "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto",
    "resolved_cli_args": {
      "model": "qwen2_5_vl",
      "model_args": "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto",
      "tasks": "mme,mmmu_val",
      "batch_size": 1,
      "num_fewshot": 0,
      "seed": "42,42,42,42",
      "gen_kwargs": "temperature=0,max_new_tokens=1024",
      "output_path": "results/",
      "log_samples": true,
      "limit": null,
      "..."
    }
  }
}
```

This means you can reconstruct the exact YAML config from any results file. No more guessing what flags were used.

### 1.6 Web UI Integration

The Web UI now supports YAML import and export:

- **Export**: Configure an evaluation in the Web UI, then click "Export YAML" to download the config as a `.yaml` file. Share it with teammates or commit it to your repo.
- **Import**: Click "Import YAML" to load a config file into the Web UI form. Useful for reviewing or tweaking a config before running.

The `env` section maps to the Web UI's environment variables field. The round-trip is lossless - export a YAML, import it back, and the form state is identical.

### 1.7 Example Configs

Three example configs are included in `configs/`:

| File | Use Case |
|------|----------|
| `configs/example_local.yaml` | Local GPU evaluation (Qwen2.5-VL) |
| `configs/example_api.yaml` | API model evaluation (OpenAI-compatible) |
| `configs/example_batch.yaml` | Multiple models in a single run |

Use them as templates:

```bash
cp configs/example_local.yaml configs/my_experiment.yaml
# Edit to your needs, then run
python -m lmms_eval --config configs/my_experiment.yaml
```
