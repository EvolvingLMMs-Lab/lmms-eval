# TensorRT Edge-LLM

[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) runs Qwen2.5-VL as TensorRT engines through a C++ runner. It takes three steps: export to ONNX, build engines on the device, then evaluate.

| | Orin (JetPack 6.2) | Thor (JetPack 7.1, tested) |
|---|---|---|
| Edge-LLM release | v0.6.0 | v0.10.1 |
| Export | x86 GPU host only (compute capability 8.0+, about 8–16 GB VRAM for 3B, 20–48 GB for 7B) | on the device |
| Precisions | fp16, int4_awq (no FP8 / NVFP4 on SM87) | fp16, int4_awq, fp8, nvfp4 |

## Thor workflow

```bash
export TRT_WORKSPACE=/opt/models/trt-edgellm      # ONNX and engines: <workspace>/<model>-<precision>/{onnx,engines}

# 1. Runtime image, on top of lmms-eval-jetson (L4T_RELEASE / L4T_SOC: the rXX.Y part and the SoC directory
#    in /etc/apt/sources.list.d/nvidia-l4t-apt-source.list)
docker build --build-arg BASE_IMAGE=lmms-eval-jetson:latest --build-arg EDGELLM_REF=v0.10.1 \
  --build-arg EMBEDDED_TARGET=jetson-thor --build-arg CUDA_CTK_VERSION=13.0 --build-arg ENABLE_CUTE_DSL=ALL \
  --build-arg L4T_RELEASE=r38.4 --build-arg L4T_SOC=som \
  -f jetson/frameworks/trt_edgellm/Dockerfile -t lmms-eval-trt-edgellm:latest jetson/frameworks/trt_edgellm

# 2. Export (quantize if needed) -> $TRT_WORKSPACE/<model>-<precision>/onnx; builds the export image on first use
jetson/scripts/download_assets.sh hf 3b
jetson/frameworks/trt_edgellm/export.sh 3b fp8

# 3. Build engines on the device (REMOVE_ONNX=1 deletes the ONNX afterwards)
jetson/frameworks/trt_edgellm/build_engines.sh 3b fp8

# 4. Evaluate like any other framework
jetson/scripts/run_eval.sh trt_edgellm 3b-fp8 mme 8
jetson/scripts/run_eval.sh trt_edgellm 3b-fp8 mme
```

The script headers list the other options: `EDGELLM_REF`, `BASE_IMAGE`, `GPU_FLAGS` (x86: `--gpus all`), `KEEP_QUANTIZED`, `AWQ_SOURCE`.

## How it works

- **Export** (`export.sh`, image `Dockerfile.export` on `nvcr.io/nvidia/pytorch:26.05-py3`): `tensorrt-edgellm-quantize llm` for fp8 / nvfp4 / int4_awq (default text calibration; LM head and vision encoder unquantized), then `tensorrt-edgellm-export`. The LLM and the vision encoder are exported separately, and the vision encoder always comes from the base checkpoint in FP16, so every precision uses the same vision encoder.
- **Build** (`build_engines.sh`): `llm_build` (batch 1, input ≤ 2560 tokens, KV cache 3072) and `visual_build` with the model's image-token range (256–2048 for Qwen2.5-VL).
- **Evaluate** (lmms-eval `trt_edgellm` backend): all requests go into one input JSON, `llm_inference` runs once, and its profile is saved as `trt_profile.json`. Latency is therefore available only as averages (vision encoder, prefill, per-token decode), not per sample. Failed requests are scored as empty answers and counted in the profile as `failed_requests`. The encoder cache is off (`encoder_cache_budget_bytes=0`), as for the other frameworks.

## Known issues (Thor, Edge-LLM v0.10.1)

- **int4_awq from Qwen's AWQ checkpoint** (`AWQ_SOURCE=qwen`, the checkpoint vLLM `awq` uses): text prompts work, but every image prompt returns an empty answer. The default quantizes the base checkpoint with Edge-LLM's own AWQ instead, so Edge-LLM int4_awq and vLLM awq are **different** 4-bit models.
- **nvfp4** (3B, quantized here): garbage output even for text-only prompts, while fp8 from the same pipeline is fine. Not yet separated from the quantization; NVIDIA's pre-quantized `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` would tell.
- **Images with a side above 4096 px** are rejected by the runner ("exceeds the GPU-resize budget"), e.g. 64 of MME's landmark photos. `trt_edgellm.sh` sets `max_image_side=4096`, which downscales them first (`TRT_MAX_IMAGE_SIDE` to change). The `+no-downscale` MME runs on Thor predate this fix and lose about 100 perception points on landmark.
- **Build-image quirks**, handled in the Dockerfiles: the Thor base image ships stray TensorRT 10.13.2 headers in `/usr/include` (the include dir is set from the apt package), exports `CUDAARCHS=110` (unset, since the FP4 kernels need `110a`), and the plugin path must be set (`EDGELLM_PLUGIN_PATH`).
