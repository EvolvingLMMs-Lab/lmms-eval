FROM nvcr.io/nvidia/pytorch:25.04-py3

# ============================================================
# Mirror switches (set to 0 to disable and use official sources)
# Override at build time:
#   docker build --build-arg USE_CN_MIRROR=0 ...
# ============================================================
ARG USE_CN_MIRROR=1
ARG APT_MIRROR=http://mirrors.aliyun.com/ubuntu/
ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Swap apt sources to Tsinghua mirror when USE_CN_MIRROR=1
# Ubuntu 24.04 (noble) uses DEB822 format at /etc/apt/sources.list.d/ubuntu.sources
RUN if [ "$USE_CN_MIRROR" = "1" ]; then \
        echo ">>> Using CN apt mirror: $APT_MIRROR" && \
        if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
            sed -i "s|http://archive.ubuntu.com/ubuntu/|${APT_MIRROR}|g; s|http://security.ubuntu.com/ubuntu/|${APT_MIRROR}|g" \
                /etc/apt/sources.list.d/ubuntu.sources; \
        fi && \
        if [ -f /etc/apt/sources.list ]; then \
            sed -i "s|http://archive.ubuntu.com/ubuntu/|${APT_MIRROR}|g; s|http://security.ubuntu.com/ubuntu/|${APT_MIRROR}|g" \
                /etc/apt/sources.list; \
        fi; \
    fi

# Install system dependencies and ffmpeg in one layer, then clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    xz-utils \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Configure pip to use CN mirror when USE_CN_MIRROR=1
RUN if [ "$USE_CN_MIRROR" = "1" ]; then \
        echo ">>> Using CN pip index: $PIP_INDEX" && \
        pip config set global.index-url "$PIP_INDEX" && \
        pip config set global.trusted-host "$(echo $PIP_INDEX | awk -F[/:] '{print $4}')" && \
        pip config set global.timeout 300 && \
        pip config set global.retries 5; \
    fi

# Set working directory
WORKDIR /workspace
# At runtime, mount this repo to /workspace/lmms-eval and run:
#   pip install -e . --no-deps
ENV PYTHONPATH=/workspace/lmms-eval

# Install Python packages and clean up in one layer.
# Pin transformers and qwen-vl-utils for result reproducibility.
# transformers >= 5.0 is required: LLaVA-OneVision-2-8B-Instruct's
# configuration_llava_onevision2.py uses PreTrainedConfig (5.x API).
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --no-deps \
        peft accelerate datasets evaluate sentence_transformers && \
    pip install --no-cache-dir \
        "transformers==5.7.0" "qwen-vl-utils==0.0.14" \
        multiprocess dill xxhash pyarrow fsspec huggingface_hub \
        httpx jsonlines aiohttp numexpr \
        pytablewriter sacrebleu sqlitedict timm einops ftfy openai \
        opencv-python-headless "av<16.0.0" hf_transfer nltk sentencepiece \
        yt-dlp tqdm-multiprocess transformers-stream-generator \
        zstandard pyyaml sympy latex2sympy2 mpmath Jinja2 openpyxl loguru \
        tenacity wandb tiktoken pydantic packaging zss protobuf \
        python-dotenv \
        decord fastapi uvicorn more-itertools editdistance librosa soundfile \
        spacy anls rouge Levenshtein google-generativeai \
        duckduckgo_search langchain langchain-community beautifulsoup4 python-multipart \
        trimesh scipy shapely && \
    pip install --no-cache-dir flash-attn --no-build-isolation && \
    rm -rf ~/.cache/pip /tmp/* /root/.cache

# Set default command
CMD ["/bin/bash"]
