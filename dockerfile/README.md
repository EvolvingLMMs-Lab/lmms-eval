# Docker image for LLaVA-OneVision-2 evaluation

This Dockerfile reproduces the `lmms-eval-ov2:26.05` image used by the
`llava-onevision-2` branch sweep experiments. It is based on
`nvcr.io/nvidia/pytorch:25.04-py3` (PyTorch 2.7.0a0, CUDA 12.9, Python 3.12)
and pins the exact `transformers` / `qwen-vl-utils` versions required for
result reproducibility.

## Build

```bash
# CN mirrors (default, aliyun)
docker build -t lmms-eval-ov2:26.05 -f dockerfile/Dockerfile .

# Disable CN mirrors (official PyPI / Ubuntu archive)
docker build --build-arg USE_CN_MIRROR=0 \
    -t lmms-eval-ov2:26.05 -f dockerfile/Dockerfile .
```

## Run

The image does **not** bake in the `lmms-eval` source. Mount this repo into
`/workspace/lmms-eval` and install in editable mode on first launch:

```bash
docker run --gpus all --rm -it \
    -v $(pwd):/workspace/lmms-eval \
    -v /path/to/hf_cache:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -w /workspace/lmms-eval \
    lmms-eval-ov2:26.05 \
    bash -lc "pip install -e . --no-deps && bash"
```

## Pinned versions (do not change without re-running the sweep)

- `transformers==5.7.0` — LLaVA-OneVision-2-8B-Instruct's
  `configuration_llava_onevision2.py` uses `PreTrainedConfig`
  (transformers 5.x API); imports fail under 4.x.
- `qwen-vl-utils==0.0.14`
- `av<16.0.0`
- `flash-attn` (built against the base image's PyTorch 2.7)
