# MMVP (Multimodal Visual Patterns)

MMVP is a benchmark that focuses on identifying "CLIP-blind pairs" — images perceived as similar by CLIP despite clear visual differences. It tests VLMs across 9 basic visual patterns including orientation, direction, color, counting, etc.

## Dataset

- **Source**: [`lmms-lab-eval/MMVP`](https://huggingface.co/datasets/lmms-lab-eval/MMVP)
- **Original**: [`MMVP/MMVP`](https://huggingface.co/datasets/MMVP/MMVP) (broken — only contains images, missing text annotations)
- **Samples**: 300 (150 pairs), each pair has the same question with opposite correct answers (A/B)

## Ground Truth Corrections

The original `MMVP/MMVP` dataset contained annotation errors where two pairs had their answers swapped. These corrections are applied directly in the `lmms-lab-eval/MMVP` dataset based on visual verification in [issue #1018](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/1018) and the [original MMVP issue #30](https://github.com/tsb0601/MMVP/issues/30):

| Index | Question | Original GT | Corrected GT | Reason |
|:-----:|:---------|:-----------:|:------------:|:-------|
| 99 | Does the elephant have long or short tusks? | (a) Long | **(b) Short** | Image shows short tusks |
| 100 | Does the elephant have long or short tusks? | (b) Short | **(a) Long** | Image shows long tusks |
| 279 | Is the elderly person standing or sitting? | (a) Standing | **(b) Sitting** | Image shows person sitting on bench |
| 280 | Is the elderly person standing or sitting? | (b) Sitting | **(a) Standing** | Image shows person standing |

## Metrics

- **mmvp_accuracy**: Percentage of correctly answered questions (300 samples)
- **mmvp_pair_accuracy**: Percentage of pairs where both questions are answered correctly (150 pairs). This stricter metric better captures genuine visual understanding vs. lucky guessing.

## References

```bibtex
@inproceedings{tong2024eyes,
  title={Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs},
  author={Tong, Shengbang and Liu, Zhuang and Zhai, Yuexiang and Ma, Yi and LeCun, Yann and Xie, Saining},
  booktitle={CVPR},
  year={2024}
}
```
