# HiPhO

HiPhO is a multimodal benchmark built from 13 high school physics olympiad exams of 2024 and 2025.

- Paper: https://arxiv.org/abs/2509.07894
- Dataset: https://huggingface.co/datasets/HY-Wan/HiPhO

## Usage
```bash
python -m lmms_eval --tasks hipho --model <model_name> --batch_size 1
```

Run a single exam with a task such as `hipho_ipho_2025`.

## Scoring

- Predicted answers are read from `\boxed{...}`, otherwise from a `Final Answer:` line, otherwise from the last line of the response. They are split at top-level commas and semicolons.
- Every reference answer is stored as `\boxed{...}`. It is compared after the box, `$`, `\text{}`, `\left`/`\right`, case and spaces are removed.
- A response answer matches a reference answer when the two are equal after that cleanup, or when both are numbers within a relative or absolute tolerance of `1e-3`. `x = 5` can be answered as `x = 5` or `5`, and `x = 7` does not match `x = 5`.
- Each reference answer is matched at most once. It earns its share of the question's `points`, so a question scores between 0 and 1.
- The `hipho_score` metric is 100 times the mean question score. It is not weighted by the marks of each question, so it is not the official exam score.
