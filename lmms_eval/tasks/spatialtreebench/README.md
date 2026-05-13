# SpatialTreeBench (`spatialtreebench`)

SpatialTreeBench is a hierarchical benchmark for evaluating spatial capabilities in multimodal models, from low-level perception to higher-level simulation and agentic behavior.

## Name Mapping in lmms-eval

Model cards and reports may use different benchmark names. In lmms-eval, the following name variants all refer to this existing task:

- `TreeBench`
- `SpatialTreeBench`
- `Spatial-TreeBench`

Canonical lmms-eval task name: `spatialtreebench`

Task config: `lmms_eval/tasks/spatialtreebench/spatialtreebench.yaml`

Dataset configured in YAML: `LongfeiLi/SpatialTree-Bench`

## References

- Paper: [SpatialTree: How Spatial Abilities Branch Out in MLLMs](https://arxiv.org/abs/2512.20617)
- Project page: [spatialtree.github.io](https://spatialtree.github.io/)
- Dataset: [LongfeiLi/SpatialTree-Bench](https://huggingface.co/datasets/LongfeiLi/SpatialTree-Bench)

## Usage

```bash
python -m lmms_eval \
  --model <model_name> \
  --model_args <key=value,...> \
  --tasks spatialtreebench \
  --batch_size 1
```

Use `--tasks spatialtreebench` even when source materials mention `TreeBench` or `Spatial-TreeBench`.
