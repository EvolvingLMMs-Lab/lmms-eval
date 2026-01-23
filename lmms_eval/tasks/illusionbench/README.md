# IllusionBench

IllusionBench is a benchmark for evaluating visual illusion understanding in Vision-Language Models.

**Paper**: [IllusionBench: A Large-scale and Comprehensive Benchmark for Visual Illusion Understanding in Vision-Language Models](https://arxiv.org/abs/2501.00848)

**Dataset**: [lmms-lab-eval/IllusionBench](https://huggingface.co/datasets/lmms-lab-eval/IllusionBench)

## Overview

- 5,357 question-answer pairs from 1,000+ images
- Covers classic cognitive illusions and real-world scene illusions
- Question types: True/False and Multiple Choice
- Categories: Classic Cognitive, Trap, Real Scene, Ishihara, No Illusion

## Usage

```bash
python -m lmms_eval --tasks illusionbench --model <model_name>
```

## Citation

```bibtex
@article{zhang2025illusionbench,
  title={IllusionBench: A Large-scale and Comprehensive Benchmark for Visual Illusion Understanding in Vision-Language Models},
  author={Zhang, Yiming and Zhang, Zicheng and Wei, Xinyi and Liu, Xiaohong and Zhai, Guangtao and Min, Xiongkuo},
  journal={arXiv preprint arXiv:2501.00848},
  year={2025}
}
```
