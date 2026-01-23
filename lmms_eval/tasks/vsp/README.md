# VSP (Visual Spatial Planning)

VSP evaluates visual spatial planning capabilities in multimodal large language models.

**Paper:** [VSP: Diagnosing the Dual Challenges of Perception and Reasoning in Spatial Planning Tasks for MLLMs](https://arxiv.org/abs/2407.01863)

**GitHub:** [UCSB-NLP-Chang/Visual-Spatial-Planning](https://github.com/UCSB-NLP-Chang/Visual-Spatial-Planning)

## Subtasks

- `vsp_google_map`: Navigation task - find path from start (S) to goal (G)
- `vsp_collision`: Collision detection - estimate time for car/person to reach goal

## Dataset

The dataset is available at [AnonyCAD/VSP](https://huggingface.co/datasets/AnonyCAD/VSP) on HuggingFace.

## Usage

```bash
python -m lmms_eval --model <model> --tasks vsp --limit 10
```

## Citation

```bibtex
@article{wu2024vsp,
  title={VSP: Diagnosing the Dual Challenges of Perception and Reasoning in Spatial Planning Tasks for MLLMs},
  author={Wu, Qiucheng and Zhao, Handong and Saxon, Michael and Bui, Trung and Wang, William Yang and Zhang, Yang and Chang, Shiyu},
  journal={arXiv preprint arXiv:2407.01863},
  year={2024}
}
```
