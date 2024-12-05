# MixEval-X: Any-to-Any Evaluations from Real-World Data Mixtures

[Homepage](https://mixeval-x.github.io/) / [arXiv](https://arxiv.org/abs/2410.13754)

## Usage

Here is the list of tasks in MixEval-X:

```
mix_evals_image2text
 ├── mix_evals_image2text_freeform ---------- 998 rows
 └── mix_evals_image2text_mc ---------------- 990 rows
mix_evals_image2text_hard
 ├── mix_evals_image2text_freeform_hard ----- 498 rows
 └── mix_evals_image2text_mc_hard ----------- 500 rows
mix_evals_video2text
 ├── mix_evals_video2text_freeform ---------- 968 rows
 └── mix_evals_video2text_mc ---------------- 634 rows
mix_evals_video2text_hard
 ├── mix_evals_video2text_freeform_hard ----- 499 rows
 └── mix_evals_video2text_mc_hard ----------- 324 rows
mix_evals_audio2text
 └── mix_evals_audio2text_freeform ---------- 962 rows
mix_evals_audio2text_hard
 └── mix_evals_audio2text_freeform_hard ----- 505 rows
```

The HowToQA and Social-IQ-2.0 was removed from the Video2Text benchmark pool due to annotation issues. A key advantage of MixEval-X is its capacity for self-refinement, enabling the benchmark pool to adapt and grow with time.

You can run the command:

```bash
lmms-eval   --model=<MODEL> \
            --model_args=<MODEL_ARGS> \
            --tasks=<TASK> \
            --batch_size=1 \
            --log_samples \
            --output_path=./logs/
```

Models are listed at [here](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/0589d0fba2efbcb526321f23ab0587599fd3c4c9/lmms_eval/models/__init__.py#L13).

For example, to evaluate `llava_vid` on `mix_evals_video2text` (including `mix_evals_video2text_freeform` and `mix_evals_video2text_mc`):

```bash
lmms-eval   --model=llava_vid \
            --model_args=pretrained=lmms-lab/LLaVA-NeXT-Video-7B \
            --tasks=mix_evals_video2text \
            --batch_size=1 \
            --log_samples \
            --output_path=./logs/
```

For more details, please refer to the [readme](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main) and [documentation](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/docs).

## Final Results Calculation

The final results are calculated by the weighted average of the results from the two tasks in each benchmark pool. The weights are determined by the number of rows in each task. For example, the final results for `mix_evals_video2text` are calculated as follows:

```python
NUM_ROWS = {
    "mix_evals_video2text_freeform": 968,
    "mix_evals_video2text_mc": 634,
}

results = {
    "mix_evals_video2text_freeform": 0.5,
    "mix_evals_video2text_mc": 0.6,
}

final_result = sum([results[task] * NUM_ROWS[task] for task in NUM_ROWS]) / sum(NUM_ROWS.values())
```

## Citation

```bib
@article{ni2024mixevalx,
    title={MixEval-X: Any-to-Any Evaluations from Real-World Data Mixtures},
    author={Ni, Jinjie and Song, Yifan and Ghosal, Deepanway and Li, Bo and Zhang, David Junhao and Yue, Xiang and Xue, Fuzhao and Zheng, Zian and Zhang, Kaichen and Shah, Mahir and Jain, Kabir and You, Yang and Shieh, Michael},
    journal={arXiv preprint arXiv:2410.13754},
    year={2024}
}

@article{ni2024mixeval,
    title={MixEval: Deriving Wisdom of the Crowd from LLM Benchmark Mixtures},
    author={Ni, Jinjie and Xue, Fuzhao and Yue, Xiang and Deng, Yuntian and Shah, Mahir and Jain, Kabir and Neubig, Graham and You, Yang},
    journal={arXiv preprint arXiv:2406.06565},
    year={2024}
}
```
