# Contributing to lmms-eval

Thanks for your interest in contributing to lmms-eval! This guide covers everything you need to get started.

## Getting Started

### Prerequisites

- Python >= 3.9
- [uv](https://docs.astral.sh/uv/) (package manager)
- Git

### Setup

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
uv sync
```

### Running a Quick Test

```bash
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks mme \
    --batch_size 1 \
    --limit 8
```

## Development Workflow

1. **Fork** the repository
2. **Create a branch** from `main` (`git checkout -b my-feature`)
3. **Make changes** and test locally
4. **Run linting** before committing:
   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit run --all-files
   ```
5. **Commit** with a descriptive message (see [Commit Style](#commit-style))
6. **Open a pull request** against `main`

### Contributor Funnel and Labels

To make onboarding predictable, we use a lightweight funnel:

`Discover -> First Run -> First Issue -> First PR -> Repeat PRs -> Maintainer Track`

Issue labels to use and follow:

- `good first issue` - Small task, scoped for first-time contributors
- `help wanted` - Community contribution requested with maintainer support
- `priority:high` - Important work for upcoming release
- `needs reproduction` - Missing minimal repro before triage
- `needs decision` - Maintainer decision required
- `blocked` - Waiting on dependency or external change

Target response times:

- Triage first response: within 48 hours
- First PR review: within 72 hours

### Commit Style

Use conventional commit prefixes:

- `feat:` - New feature or benchmark
- `fix:` - Bug fix
- `refactor:` - Code restructuring without behavior change
- `docs:` - Documentation only
- `test:` - Adding or updating tests
- `chore:` - Maintenance (deps, CI, configs)

Examples:
```
feat: add SWE-bench Verified benchmark
fix: handle empty response in MMMU parsing
refactor: extract shared flatten() to model_utils
```

### Code Style

- **Formatter**: Black (line length 240) + isort (handled by pre-commit)
- **Naming**: PEP 8 - `snake_case` for functions/variables, `PascalCase` for classes
- **Type hints**: Required for all new code
- **Docstrings**: Required for public APIs
- **Line length**: 88 characters recommended, 240 max (enforced by Black)
- **Imports**: Use `from lmms_eval.imports import optional_import` for model-specific packages

## How to Contribute

### Adding a New Benchmark / Task

This is the most common contribution. Each benchmark lives in its own directory under `lmms_eval/tasks/`.

1. **Create the directory**:
   ```
   lmms_eval/tasks/my_benchmark/
       my_benchmark.yaml        # Task config
       utils.py                 # Processing functions
       _default_template_yaml   # Shared defaults (optional)
   ```

2. **Write the YAML config**:
   ```yaml
   task: my_benchmark
   dataset_path: hf-org/my-dataset     # HuggingFace dataset
   test_split: test
   output_type: generate_until

   doc_to_visual: !function utils.doc_to_visual
   doc_to_text: !function utils.doc_to_text
   doc_to_messages: !function utils.doc_to_messages
   doc_to_target: "answer"

   process_results: !function utils.process_results

   metric_list:
     - metric: accuracy
       aggregation: mean
       higher_is_better: true

   generation_kwargs:
     max_new_tokens: 128
   ```

3. **Implement the functions in `utils.py`**:
   - `doc_to_visual(doc)` - Extract images/video/audio from a dataset sample
   - `doc_to_text(doc, lmms_eval_specific_kwargs)` - Format the text prompt
   - `doc_to_messages(doc, lmms_eval_specific_kwargs)` - Build structured chat messages (recommended)
   - `process_results(doc, results)` - Parse model output and compute metrics

4. **Test your benchmark**:
   ```bash
   python -m lmms_eval --model qwen2_5_vl \
       --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
       --tasks my_benchmark --limit 8
   ```

See `docs/task_guide.md` for a detailed walkthrough.

### Adding a New Model

Models live under `lmms_eval/models/chat/` (recommended) or `lmms_eval/models/simple/` (legacy).

1. **Create `models/chat/my_model.py`**
2. **Inherit from `lmms`**, set `is_simple = False`
3. **Implement required methods**: `generate_until`, `loglikelihood`, `generate_until_multi_round`
4. **Register** in `models/__init__.py`:
   ```python
   AVAILABLE_CHAT_TEMPLATE_MODELS = {"my_model": "MyModel", ...}
   ```
5. **Use `optional_import`** for model-specific dependencies:
   ```python
   from lmms_eval.imports import optional_import
   MyLib, _has_mylib = optional_import("mylib", "MyLib")
   ```

See `docs/model_guide.md` for details and `CLAUDE.md` for architecture patterns.

### Fixing Bugs

1. Check existing [issues](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues) for duplicates
2. If the bug is new, open an issue first using the **Bug Report** template
3. Reference the issue in your PR

### Improving Documentation

Documentation improvements are always welcome. Key docs:
- `README.md` - Project overview (available in 16 languages under `docs/`)
- `docs/task_guide.md` - How to add benchmarks
- `docs/model_guide.md` - How to add models
- `CLAUDE.md` - Architecture reference

## Package Management

**Always use `uv`, never `pip`.**

```bash
uv sync                    # Install from lockfile
uv add package             # Add a dependency
uv remove package          # Remove a dependency
uv run tool                # Run a tool in the environment
```

## Questions?

- **Discord**: [discord.gg/zdkwKUqrPy](https://discord.gg/zdkwKUqrPy)
- **Issues**: [GitHub Issues](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
- **Quick-start**: [Evaluate Your Model in 5 Minutes](docs/quickstart.md)
