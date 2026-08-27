# Spatial457

Spatial457 is a diagnostic benchmark for 6D spatial reasoning in large multimodal models (CVPR 2025 Highlight).

## Paper

[Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models](https://arxiv.org/abs/2502.08636)

## Dataset

- HuggingFace: [RyanWW/Spatial457](https://huggingface.co/datasets/RyanWW/Spatial457)
- GitHub: [XingruiWang/Spatial457](https://github.com/XingruiWang/Spatial457)

## Capabilities Evaluated

- Multi-object understanding
- 2D spatial localization
- 3D spatial localization
- 3D orientation estimation

## Task Levels

| Level | Task | Description |
|-------|------|-------------|
| L1 | single | Single object attribute recognition |
| L2 | objects | Multiple object attribute recognition |
| L3 | 2d_spatial | 2D spatial relationships from camera view |
| L4 | occ | Occlusion relationships |
| L4 | pose | Object facing direction in 3D space |
| L5 | 6d_spatial | 3D spatial relationships from object perspective |
| L5 | collision | Potential collision prediction |

## Dataset

The dataset is hosted at [lmms-lab-eval/Spatial457](https://huggingface.co/datasets/lmms-lab-eval/Spatial457).

## Usage

```bash
python -m lmms_eval --tasks spatial457 --model <model_name>
```

Or run individual levels:

```bash
python -m lmms_eval --tasks spatial457_l1_single --model <model_name>
```

## Citation

```bibtex
@inproceedings{wang2025spatial457,
  title     = {Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models},
  author    = {Wang, Xingrui and Ma, Wufei and Zhang, Tiezheng and de Melo, Celso M and Chen, Jieneng and Yuille, Alan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```
