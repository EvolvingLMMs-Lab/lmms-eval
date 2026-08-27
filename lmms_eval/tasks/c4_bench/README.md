# C4 Bench for lmms-eval

This task pack integrates [C4 Bench](https://arxiv.org/abs/2608.06501) with
lmms-eval. It loads the published
[C4-Eval](https://huggingface.co/datasets/sci-m-wang/C4-Eval) records and
downloads each original image from the Hub on first use.

## Run

```bash
lmms-eval \
  --model <lmms-eval-backend> \
  --model_args <backend-arguments> \
  --tasks c4_bench \
  --log_samples \
  --output_path c4_results
```

`c4_bench` runs the four primary task forms, `H0`, `H1`, `H4`, and `E0`, and
reports their size-weighted `c4_exact_match` over 884 instances. Run
`c4_bench_e1` separately for gold-answer explanation analysis, or
`c4_bench_explanations` to report JSON validity across `E0` and `E1`.

The adapter uses each published question verbatim. It intentionally does not
set `max_new_tokens`, `max_tokens`, or an artificial context limit. The empty
`until` list only prevents lmms-eval from injecting its default double-newline
stop sequence. Model-native reasoning output is preserved for the official
conservative answer parser.

For an official run, record the model id, backend, hardware, native context
length, output-token setting, sampling configuration, chat template or
reasoning overrides, task scope, sharding, and deviations from model defaults.

## Citation

```bibtex
@misc{wang2026mllmsdecodecreativeleap,
      title={Can MLLMs Decode the Creative Leap? Introducing C4 for Cross-Concept Understanding},
      author={Ming Wang and Yuqing Zhang and Tingna Xie and Xiangju Li and Xiaocui Yang and Daling Wang and Shi Feng and Yifei Zhang},
      year={2026},
      eprint={2608.06501},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.06501},
}
```
