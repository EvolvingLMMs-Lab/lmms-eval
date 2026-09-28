# lmms-eval + TensorRT Edge-LLM C++ runtime (llm_build, visual_build, llm_inference) for Jetson.
#
# Orin (JetPack 6.2, L4T r36.4) - defaults (Edge-LLM v0.6.0; ONNX must be exported on an x86 GPU host):
#   docker build -f jetson/frameworks/trt_edgellm/Dockerfile -t lmms-eval-trt-edgellm:latest jetson/frameworks/trt_edgellm
# Thor (JetPack 7.0/7.1, L4T r38.x, CUDA 13.0) - tested with Edge-LLM v0.10.1 on L4T r38.4:
#   docker build --build-arg BASE_IMAGE=lmms-eval-jetson:latest --build-arg EDGELLM_REF=v0.10.1 \
#     --build-arg EMBEDDED_TARGET=jetson-thor --build-arg CUDA_CTK_VERSION=13.0 --build-arg ENABLE_CUTE_DSL=ALL \
#     --build-arg L4T_RELEASE=r38.4 --build-arg L4T_SOC=som \
#     -f jetson/frameworks/trt_edgellm/Dockerfile -t lmms-eval-trt-edgellm:latest jetson/frameworks/trt_edgellm
#   L4T_RELEASE / L4T_SOC: the rXX.Y part and SoC directory in /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.
#   CUDA_CTK_VERSION / ENABLE_CUTE_DSL: Edge-LLM >= 0.8 CMake options (see its installation guide); empty = not passed.
ARG BASE_IMAGE=lmms-eval-jetson:latest
FROM ${BASE_IMAGE}

ARG EDGELLM_REF=v0.6.0
ARG EMBEDDED_TARGET=jetson-orin
ARG L4T_RELEASE=r36.4
ARG L4T_SOC=t234
ARG CUDA_CTK_VERSION=""
ARG ENABLE_CUTE_DSL=""
# e.g. "-o Acquire::Check-Date=false" when the board clock is not synchronized (see jetson/docker/Dockerfile)
ARG APT_OPTS=""

# TensorRT headers/libs from NVIDIA's L4T apt repo (same versions JetPack installs on the host).
# The build below points CMake at these packaged headers explicitly: the Thor vLLM base image also carries an
# unpackaged copy of other TensorRT headers in /usr/include, and finding NvInfer.h there adds -isystem /usr/include,
# which breaks libstdc++'s #include_next <math.h> in every CUDA file. The base image also exports CUDAARCHS=110,
# which would override the toolchain's arch-specific target (110a, needed by the FP4 kernels); it is unset here.
RUN apt-get ${APT_OPTS} update && apt-get ${APT_OPTS} install -y --no-install-recommends ca-certificates curl gnupg \
    && curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | gpg --dearmor -o /usr/share/keyrings/nvidia-l4t.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-l4t.gpg] https://repo.download.nvidia.com/jetson/common ${L4T_RELEASE} main" > /etc/apt/sources.list.d/nvidia-l4t.list \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-l4t.gpg] https://repo.download.nvidia.com/jetson/${L4T_SOC} ${L4T_RELEASE} main" >> /etc/apt/sources.list.d/nvidia-l4t.list \
    && apt-get ${APT_OPTS} update && apt-get ${APT_OPTS} install -y --no-install-recommends libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${EDGELLM_REF} https://github.com/NVIDIA/TensorRT-Edge-LLM.git /opt/TensorRT-Edge-LLM \
    && cd /opt/TensorRT-Edge-LLM && git submodule update --init --recursive --depth 1 \
    && TRT_INCLUDE_DIR=$(dirname "$(dpkg -L libnvinfer-headers-dev | grep '/NvInfer.h$')") \
    && unset CUDAARCHS CUDA_ARCHITECTURES \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR=/usr \
        -DTensorRT_INCLUDE_DIR="$TRT_INCLUDE_DIR" \
        -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
        -DEMBEDDED_TARGET=${EMBEDDED_TARGET} \
        ${CUDA_CTK_VERSION:+-DCUDA_CTK_VERSION=${CUDA_CTK_VERSION}} \
        ${ENABLE_CUTE_DSL:+-DENABLE_CUTE_DSL=${ENABLE_CUTE_DSL}} \
    && cmake --build build -j"$(nproc)"
# (No `llm_inference --help` check here: the CUDA driver is only mounted at `docker run --runtime nvidia`.)

# The builders and llm_inference load the plugin library from build/... relative to the working directory
# unless EDGELLM_PLUGIN_PATH is set.
ENV PATH=/opt/TensorRT-Edge-LLM/build/examples/llm:/opt/TensorRT-Edge-LLM/build/examples/multimodal:$PATH \
    EDGELLM_PLUGIN_PATH=/opt/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
