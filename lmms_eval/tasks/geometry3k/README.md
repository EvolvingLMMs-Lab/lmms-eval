# Geometry3K

Geometry3K is a benchmark for evaluating geometry problem solving with 3,002 high school multi-choice problems combining text descriptions and diagrams for symbolic reasoning assessment. Problems span diverse geometric shapes including lines, triangles, quadrilaterals, polygons, and circles.

## Dataset

- **Dataset**: [Yang130/geometry3k_4choices_mixed](https://huggingface.co/datasets/Yang130/geometry3k_4choices_mixed)
- **Split**: test (601 samples)
- **Format**: Multiple choice (A/B/C/D)

## Paper

Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning (ACL 2021)

- Paper: https://aclanthology.org/2021.acl-long.528/
- Project: https://lupantech.github.io/inter-gps

## Usage

```bash
python -m lmms_eval --tasks geometry3k --model <model_name> --model_args <args>
```

## Citation

```bibtex
@inproceedings{lu-etal-2021-inter,
    title = "{I}nter-{GPS}: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning",
    author = "Lu, Pan and Gong, Ran and Jiang, Shibiao and Qiu, Liang and Huang, Siyuan and Liang, Xiaodan and Zhu, Song-Chun",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.528/",
    pages = "6774--6786",
}
```
