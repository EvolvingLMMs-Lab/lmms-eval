# Multi-Crit

Multi-Crit evaluates whether multimodal judges can follow diverse, fine-grained criteria when comparing two candidate responses to an image-grounded prompt and recognize cross-criterion preference conflicts.


- Project and reference implementation: <https://github.com/tyxiong23/Multi-Crit>
- Dataset: <https://huggingface.co/datasets/txiong23/Multi-Crit>
- Paper: [*Multi-Crit: Benchmarking Multimodal Judges on Pluralistic
  Criteria-Following*](https://openaccess.thecvf.com/content/CVPR2026/papers/Xiong_Multi-Crit_Benchmarking_Multimodal_Judges_on_Pluralistic_Criteria-Following_CVPR_2026_paper.pdf) (CVPR 2026 Highlight)

## Tasks

| Task | Hugging Face split | Rows |
| --- | --- | ---: |
| `multi_crit_open_ended` | `open_ended` | 1,000 |
| `multi_crit_reasoning` | `reasoning` | 425 |
| `multi_crit` | both tasks above | 1,425 |

The group reports each split separately; it intentionally does not mix the two
criterion sets into one aggregate.

## Evaluation protocol

If the direct parser cannot find `Response 1 is better.` or
`Response 2 is better.` in the final three lines, the task uses an OpenAI-compatible verifier. Its default model is `gpt-4o-mini`; override it with
`MULTI_CRIT_VERIFIER_MODEL`. Set `OPENAI_API_KEY` and set `OPENAI_API_URL` to
either an API base URL ending in `/v1` or the full `/chat/completions` endpoint.
The verifier uses the reference settings `temperature=0`, `max_tokens=50`, one
request, and a 120-second timeout. Its raw completion is retained in logged
metric records. A verifier failure is recorded as `error` and therefore does
not count as a correct preference.

## Metrics

All reported values are percentages rounded to two decimal places.

- `overall`: micro-average over criterion rows after reference deduplication.
- criterion metrics: accuracy for each of the five split-specific criteria.
- `macro_avg`: mean of the already-rounded criterion accuracies.
- `pluralistic_acc` (PAcc): percentage of prompts for which every present
  criterion is correct.
- `tradeoff_sensitivity` (TOS): prompt-macro average over prompts containing at
  least one human-label conflict. A prompt succeeds if the judge detects any
  conflicting criterion pair. Prompts without conflicts are excluded.
- `conflict_matching` (CMR): pair-micro average over all conflicting criterion
  pairs. Numerators and denominators are summed globally before division. For
  example, results `1/2` and `2/100` aggregate as `3/102`, not as the mean of
  the two prompt-level ratios.

TOS and CMR are undefined when their eligible denominator is zero and are then
reported as `N/A`. Because `--limit` selects flattened criterion rows rather
than complete prompts, small limited runs are useful for smoke testing but do
not produce interpretable prompt/pair metrics.

## Example commands

```bash
python -m lmms_eval \
  --model openai \
  --model_args model_version=gpt-4o-2024-08-06 \
  --tasks multi_crit \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/
```

```bash
accelerate launch --multi_gpu --num_machines 1 --num_processes 4 -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=sdpa \
  --tasks multi_crit \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/
```

## Citation

```bibtex
@InProceedings{Xiong_2026_multicrit,
    author    = {Xiong, Tianyi and Ge, Yi and Li, Ming and Zhang, Zuolong and Kulkarni, Pranav and Wang, Kaishen and He, Qi and Zhu, Zeying and Liu, Chenxi and Chen, Ruibo and Zheng, Tong and Chen, Yanshuo and Wang, Xiyao and Zhang, Renrui and Chen, Wenhu and Huang, Heng},
    title     = {Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {8641-8652}
}
```
