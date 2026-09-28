# lmms-eval runtime for Jetson (default: Orin, JetPack 6 / L4T R36.4, CUDA 12.6).
# Thor (JetPack 7): docker build --build-arg BASE_IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor ...
# The base image ships Tegra-built torch, transformers, flash-attn and vLLM;
# we only add lmms-eval's pure-Python deps and pin the GPU stack so pip cannot replace it.
# The repo itself is bind-mounted at its host path at run time (see run_eval.sh).
ARG BASE_IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin
FROM ${BASE_IMAGE}
# Extra apt options, e.g. APT_OPTS="-o Acquire::Check-Date=false" on a board whose clock is not synchronized
# (apt then rejects the Ubuntu Release files as "not valid yet").
ARG APT_OPTS=""

COPY jetson/docker/requirements-jetson.txt /tmp/requirements-jetson.txt
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|transformers|tokenizers|vllm|flash[-_]attn|numpy|triton|xformers|accelerate|av)==' > /tmp/gpu-pins.txt \
    && cat /tmp/gpu-pins.txt \
    && pip install -c /tmp/gpu-pins.txt -r /tmp/requirements-jetson.txt \
    && python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

# Java for pycocoevalcap (PTB tokenizer, METEOR) used by captioning tasks such as coco2017_cap_val_lite.
RUN apt-get ${APT_OPTS} update && apt-get ${APT_OPTS} install -y --no-install-recommends default-jre-headless && rm -rf /var/lib/apt/lists/*

# The base venv's interpreter lives under /root; open it up so run_eval.sh can run as the host user.
RUN chmod 755 /root

# Base image points the HF caches at /data/models/huggingface; use the host's shared /opt/hf-cache instead.
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_CACHE=/opt/hf-cache/hub \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache/hub \
    TRANSFORMERS_CACHE=/opt/hf-cache/hub
ENTRYPOINT []
