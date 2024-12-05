# MixEval-X: Any-to-Any Evaluations from Real-World Data Mixtures

[Homepage](https://mixeval-x.github.io/) / [arXiv](https://arxiv.org/abs/2410.13754)

## Usage

Tasks:

```
mix_evals_image2text
 ├── mix_evals_image2text_freeform
 └── mix_evals_image2text_mc
mix_evals_image2text_hard
 ├── mix_evals_image2text_freeform_hard
 └── mix_evals_image2text_mc_hard
mix_evals_video2text
 ├── mix_evals_video2text_freeform
 └── mix_evals_video2text_mc
mix_evals_video2text_hard
 ├── mix_evals_video2text_freeform_hard
 └── mix_evals_video2text_mc_hard
mix_evals_audio2text
 └── mix_evals_audio2text_freeform
mix_evals_audio2text_hard
 └── mix_evals_audio2text_freeform_hard
```

```bash
lmms-eval --model=llava_vid --model_args=pretrained=lmms-lab/LLaVA-NeXT-Video-7B --tasks=mix_evals_video2text --batch_size=1 --log_samples --output_path=./logs/
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
