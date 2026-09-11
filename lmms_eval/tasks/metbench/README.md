# MET-Bench

[Paper](https://arxiv.org/abs/2502.10886) · [Website](https://vanyacohen.com/MET-Bench/) · [Reference evaluator](https://github.com/vanyacohen/MET-Bench)

MET-Bench evaluates entity tracking in Minecraft, Chess, and Shell Game using parallel text and image inputs. The `metbench` tag runs all six tasks with chain-of-thought prompts. Chess and Shell Game use ten actions; Minecraft predicts the next state from an initial state and an action.

## Data

The pinned Hugging Face configurations contain 500 unique examples per domain. `evaluation` includes images, while `evaluation_text_only` contains the same examples without images. Examples retain their order and identifiers from the full test split. Chess and Shell Game are deduplicated by initial state and ten-action prefix; Minecraft is deduplicated by initial state, action, and ordered candidate states.

| Domain | Dataset | Text task | Image task |
|---|---|---|---|
| Minecraft | [🤗 Minecraft](https://huggingface.co/datasets/vanyacohen/MET-Bench-Minecraft) | `metbench_minecraft_text` | `metbench_minecraft_image` |
| Chess | [🤗 Chess](https://huggingface.co/datasets/vanyacohen/MET-Bench-Chess) | `metbench_chess_text` | `metbench_chess_image` |
| Shell Game | [🤗 Shell Game](https://huggingface.co/datasets/vanyacohen/MET-Bench-Shell) | `metbench_shell_text` | `metbench_shell_image` |

## Run

Install lmms-eval following its contributor guide. Credentials are read from the environment. For an OpenAI-compatible API:

```bash
uv run python -m lmms_eval \
  --model openai \
  --model_args model_version=YOUR_MODEL_ID \
  --tasks metbench \
  --batch_size 1 \
  --log_samples \
  --output_path results/metbench
```

Set `OPENAI_API_KEY`; use `OPENAI_API_BASE` for another compatible provider. Add `--limit 2` for a small check or select one task by name. Image tasks use the chat interface to preserve the order of interleaved text and images and require a model that supports multiple images.

Each task reports `acc` from 0 to 1. Minecraft and Shell Game measure answer accuracy; Chess measures the fraction of board squares with the correct piece or empty state. The framework reports standard error across examples, with one board score per Chess trial. Prompts and answer parsing match the reference evaluator. Target states are used only for scoring.

## Citation

See the [paper](https://arxiv.org/abs/2502.10886) and [reference repository](https://github.com/vanyacohen/MET-Bench#citation) for citation details.
