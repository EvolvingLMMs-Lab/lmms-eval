# SPaRC (visual)

[Paper](https://aclanthology.org/2025.emnlp-main.526/) | [arXiv](https://arxiv.org/abs/2505.16686) | [Homepage](https://sparc.gipplab.org/) | [Code](https://github.com/lkaesberg/SPaRC) | [Dataset](https://huggingface.co/datasets/lkaesberg/SPaRC)

SPaRC (EMNLP 2025) has 1,000 2D grid pathfinding puzzles based on the game *The Witness*. The model must
draw a line from a start node to an end node that obeys every rule on the grid (dots, gaps, squares,
stars, triangles, polyshapes). The test split has 500 puzzles at difficulty levels 1 to 5.

This task shows each puzzle to the model as an image, together with the puzzle's text description. This
is the visual mode of the reference code (`sparc --visual`, plot type `original`, prompt
`single_shot_visual`). The visual setting is studied in the follow-up paper
[Is Visual Prompting All You Need? Studying VLM Spatial Reasoning under Progressive Visual Scaffolds](https://arxiv.org/abs/2608.21170)
(Findings of EMNLP 2026). This task covers only the plain image. It does not include the visual scaffolds
from that paper.

## Installation

The dataset has no image column. Each image is drawn from the puzzle data by the
[`sparc-visualization`](https://pypi.org/project/sparc-visualization/) package, the same renderer the
reference code uses. Its only dependency is Pillow.

```bash
pip install "sparc-visualization==0.2.2"
```

## Tasks

- `sparc_visual`: all 500 test puzzles (main variant). Puzzles are ordered round-robin by difficulty
  level, so `--limit 10` gives 2 puzzles per level.
- `sparc_visual_hard`: only the 89 level-5 puzzles, which the paper calls "hard".

## Metrics

- `acc`: the share of puzzles where the extracted path exactly matches one of the puzzle's stored
  solutions. The dataset stores every valid solution.
- `acc_level_1` to `acc_level_5` (`sparc_visual` only): `acc` for each difficulty level (86 / 118 / 121 /
  86 / 89 puzzles). A level that is not in a `--limit` subset reports `NaN`.
- `invalid_path` (lower is better): the share of answers that are not a valid line from start to end.
  An answer is invalid if it has the wrong start or end, is not connected, crosses itself, or passes
  through a rule cell. This is the "Invalid Path" column of Table 7 in the paper.

## Prompt and scoring

`utils.py` holds copies of the reference prompt
([`sparc/prompts/single_shot_visual.py`](https://github.com/lkaesberg/SPaRC/blob/9f2a3d6/sparc/prompts/single_shot_visual.py))
and validator ([`sparc/validation.py`](https://github.com/lkaesberg/SPaRC/blob/9f2a3d6/sparc/validation.py)).
The messages follow `process_puzzle_visual` in
[`sparc/process_puzzle.py`](https://github.com/lkaesberg/SPaRC/blob/9f2a3d6/sparc/process_puzzle.py): a
system prompt, then one user turn with the image first and the text second. The model explains its
reasoning, then writes `####` followed by the coordinate list. The path is read from the text after the
last `####`. If there is no `####`, the validator reads coordinates from the whole response, as the
reference code does.

We checked the copies on all 500 test puzzles. The prompts equal the reference prompts, all 3,218 stored
solutions score 1, and the images are pixel-identical to the images the reference code sends.

Generation is greedy (`temperature: 0`) with `max_new_tokens: 32768`, because reasoning models often
write more than 10k tokens per puzzle. The paper does not state a temperature, and the reference command
defaults to `--temperature 1.0`. Use `--gen_kwargs` to change these settings.

## Usage

```bash
pip install "sparc-visualization==0.2.2"
python -m lmms_eval \
    --model openai \
    --model_args model_version=google/gemini-2.5-flash-lite,base_url=https://openrouter.ai/api/v1,num_concurrent=10 \
    --tasks sparc_visual \
    --batch_size 1 --limit 10 --log_samples --output_path ./lmms_runs/sparc_visual
```

The `openai` backend reads the API key from `OPENAI_API_KEY`.

## Citation

```bibtex
@inproceedings{kaesberg-etal-2025-sparc,
  title     = "{SP}a{RC}: A Spatial Pathfinding Reasoning Challenge",
  author    = "Kaesberg, Lars Benedikt and Wahle, Jan Philip and Ruas, Terry and Gipp, Bela",
  booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
  month     = nov,
  year      = "2025",
  address   = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2025.emnlp-main.526/",
  doi       = "10.18653/v1/2025.emnlp-main.526",
  pages     = "10359--10390"
}
```
