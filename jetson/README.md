# lmms-eval on Jetson: framework benchmarks

Scripts for running lmms-eval benchmarks for Qwen2.5-VL 3B/7B on Jetson with four inference frameworks, measuring accuracy, latency and memory the same way for each. Tested on a Jetson AGX Orin 32GB (JetPack 6.2, L4T R36.4, CUDA 12.6, MAXN). For Jetson Thor, see [THOR.md](THOR.md).

| Framework | `<framework>` | Precisions (`<size>-<precision>`) | How it runs | Status |
|---|---|---|---|---|
| Hugging Face transformers (PyTorch) | `hf` | `bf16` | in process, lmms-eval `qwen2_5_vl` backend; reference implementation | tested |
| vLLM | `vllm` | `bf16`, `awq` (Qwen's AWQ 4-bit) | in process, lmms-eval `vllm` backend | scripted |
| llama.cpp | `llamacpp` | `q4_k_m`, `q8_0`, `f16` (GGUF + f16 vision projector) | `llama-server` container + lmms-eval `openai` backend with streaming | scripted |
| TensorRT Edge-LLM | `trt_edgellm` | `fp16`, `int4_awq` | C++ runner via lmms-eval `trt_edgellm` backend | **untested**: ONNX export needs an x86 GPU host or Jetson Thor |

The first precision listed is the default. Every framework sees the same image resolution range: 256–2048 visual tokens, the lmms-eval defaults for Qwen2.5-VL.

## Quick start

```bash
# 1. Build the images once (and again after dependency changes)
docker build -f jetson/Dockerfile -t lmms-eval-jetson:latest .
docker build -f jetson/frameworks/llamacpp/Dockerfile -t llamacpp-jetson:b11135 jetson/frameworks/llamacpp   # llama.cpp only

# 2. Download the model(s) for a framework + MME into /opt/hf-cache (shared, reused by other projects)
jetson/download_assets.sh <framework> <size>[-<precision>]

# 3. Run: jetson/run_eval.sh <framework> <size>[-<precision>] [tasks=mme] [limit]
jetson/run_eval.sh <framework> <size>[-<precision>] mme 8     # smoke test on 8 samples first
jetson/run_eval.sh <framework> <size>[-<precision>] mme       # full MME (2374 samples)

# 4. Refresh the comparison table
python3 jetson/summarize.py            # -> jetson/results/SUMMARY.md

# Or run a whole comparison unattended (runs one after another, then writes SUMMARY.md):
jetson/run_matrix.sh mme 8                                   # smoke test of the default matrix
jetson/run_matrix.sh mme                                     # full MME: hf, vllm, llamacpp x 3b/7b x precisions
jetson/run_matrix.sh mme "" -- vllm 3b-awq llamacpp 3b-q8_0  # a custom list of <framework> <size-precision> pairs
```

Stop other GPU jobs before measuring. They take unified memory, and 7B bf16 does not fit next to them. For example: `docker stop vllm-orchestrator`, then `docker start vllm-orchestrator` afterwards.

## Examples per framework

**Hugging Face transformers (bf16)**
```bash
jetson/download_assets.sh hf 3b
jetson/run_eval.sh hf 3b mme          # Qwen2.5-VL-3B-Instruct, bf16
jetson/run_eval.sh hf 7b mme          # Qwen2.5-VL-7B-Instruct, bf16
```

**vLLM (bf16 and AWQ 4-bit)**
```bash
jetson/download_assets.sh vllm 3b-awq
jetson/run_eval.sh vllm 3b mme        # bf16
jetson/run_eval.sh vllm 3b-awq mme    # AWQ 4-bit
jetson/run_eval.sh vllm 7b-awq mme
VLLM_GPU_MEM=0.65 jetson/run_eval.sh vllm 7b mme   # override vLLM's memory fraction
```

**llama.cpp (GGUF)**
```bash
jetson/download_assets.sh llamacpp 3b-q4_k_m
jetson/run_eval.sh llamacpp 3b-q4_k_m mme
jetson/run_eval.sh llamacpp 3b-q8_0 mme
jetson/run_eval.sh llamacpp 7b-q4_k_m mme
```
`run_eval.sh` starts `llama-server` on port 8090 (`LLAMACPP_PORT`), waits for `/health`, runs the eval, then stops the server and saves its log as `server.log`.

**TensorRT Edge-LLM (FP16 and INT4 AWQ)**, not yet tested (see [below](#tensorrt-edge-llm-workflow)):
```bash
# on an x86 GPU host or Jetson Thor:
jetson/frameworks/trt_edgellm/export.sh 3b fp16 ./trt-edgellm-workspace
# copy ./trt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-fp16/onnx -> /opt/models/trt-edgellm/Qwen2.5-VL-3B-Instruct-fp16/onnx on the Jetson, then there:
docker build -f jetson/frameworks/trt_edgellm/Dockerfile -t lmms-eval-trt-edgellm:latest jetson/frameworks/trt_edgellm
jetson/frameworks/trt_edgellm/build_engines.sh 3b fp16
jetson/run_eval.sh trt_edgellm 3b-fp16 mme
jetson/run_eval.sh trt_edgellm 3b-int4_awq mme
```

**Other tasks and options**
```bash
jetson/run_eval.sh hf 3b mme,pope 50                                        # several tasks, 50 samples each
EXTRA_MODEL_ARGS=max_pixels=802816 jetson/run_eval.sh hf 7b mme             # extra --model_args
OFFLINE=0 jetson/run_eval.sh hf 3b mme                                      # allow Hub downloads during the run

# Variants get their own result dir <framework>-<precision>+<RUN_TAG>:
RUN_TAG=pil EXTRA_MODEL_ARGS=pass_pil_images=True jetson/run_eval.sh vllm 3b mme                 # vLLM: PIL images, no PNG/base64
RUN_TAG=png1 EVAL_ENV=LMMS_IMAGE_PNG_COMPRESS_LEVEL=1 jetson/run_eval.sh llamacpp 3b-q8_0 mme    # fast lossless PNG over HTTP

# Other datasets: download once (dataset:<repo>:<config>), then run
jetson/download_assets.sh dataset:lmms-lab-encoder/LMMs-Eval-Lite:coco2017_cap_val
jetson/run_eval.sh hf 3b coco2017_cap_val_lite                              # captioning, needs Java (in the image)
```

**Image transport overhead.** The default vLLM and OpenAI-compatible paths encode every image as PNG/base64 on the CPU. For MME's large `landmark` photos (up to 4592×3056) that takes about 2 s per image on Orin, which lengthens wall time but not the per-sample model latency. `pass_pil_images=True` (vLLM) removes the encoding. `LMMS_IMAGE_PNG_COMPRESS_LEVEL=1` makes the PNG about 3.6× faster and is still lossless. The runs in [experiments/followups_20260923.sh](experiments/followups_20260923.sh) measure both.

## What gets measured

| Metric | Source |
|---|---|
| Scores (MME perception / cognition) | lmms-eval, identical task code for every framework |
| TTFT (time to first token) per sample | from the model call to the first generated token: vision encoder + prefill + first decode step. HF: generation streamer. vLLM: engine request metrics. llama.cpp: first streamed chunk (includes HTTP). TensorRT: runner averages only |
| Answer time per sample | whole model call (all generated tokens) |
| Decode ms/token | (answer − TTFT) / (output tokens − 1) |
| Preprocess time per sample | prompt/media preparation before the model call |
| RAM, GPU+SoC power | `tegrastats` sampled at 1 Hz for the whole run |

Per-sample numbers are in each run's `lmms_eval/*_samples_*.jsonl` under `token_counts`: `input_tokens`, `output_tokens`, `preprocess_seconds`, `time_to_first_token_seconds`, `generation_seconds`. MME answers are one word (about 2 tokens), so TTFT dominates the answer time on this benchmark. Use a longer-answer task to compare decode speed.

## Layout

```
jetson/
  Dockerfile, requirements-jetson.txt   lmms-eval image (Jetson vLLM base: torch 2.10, transformers 4.57.3, flash-attn, vLLM)
  run_eval.sh                           one run: <framework> <size>[-<precision>] [tasks] [limit]
  run_matrix.sh                         many runs in sequence + summary
  download_assets.sh                    models/datasets into /opt/hf-cache (offline-ready)
  summarize.py                          results/SUMMARY.md
  compute_flops.sh, flops.py            analytical FLOPs per sample (vision / prefill / decode); run_eval.sh calls it
  frameworks/
    common.sh                           argument parsing + Hub cache helpers
    hf.sh, vllm.sh, llamacpp.sh, trt_edgellm.sh   one file per framework: precisions, assets, backend args, server start/stop
    llamacpp/Dockerfile                 llama.cpp b11135 built for sm_87
    trt_edgellm/                        Dockerfile (runtime), export.sh (x86/Thor), build_engines.sh (device)
  results/<model>/<task>/<framework>-<precision>/<timestamp>[_limitN]/
    run_info.txt      board, L4T, power mode, git commit, image ids, other running containers, exact command
    run.log           full console output (+ server.log for llama.cpp, trt_profile.json for TensorRT)
    tegrastats.log    1 Hz RAM / GPU / power
    flops.json        analytical FLOPs per sample: vision encoder, LLM prefill, LLM decode (+ flops.log)
    lmms_eval/        *_results.json (scores, config), *_samples_<task>.jsonl (per-sample outputs + latency)
```

To add a framework, create `jetson/frameworks/<name>.sh` defining `FW_PRECISIONS`, `fw_assets`, `fw_setup` and, for servers, `fw_start` and `fw_stop`. The existing files show the pattern.

## TensorRT Edge-LLM workflow

[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) (v0.6.0 for JetPack 6.2) supports Qwen2.5-VL. On Orin it runs FP16, INT8 and INT4, but not FP8 or NVFP4. The pipeline has three steps:
1. **Export** (`export.sh`): quantize (INT4 AWQ) and export the LLM and vision encoder to ONNX. This needs an x86 GPU host (compute capability 8.0+, about 8–16 GB VRAM for 3B, 20–48 GB for 7B) or Jetson Thor. It cannot run on Orin.
2. **Build** (`build_engines.sh`, on the device): `llm_build` and `visual_build` create TensorRT engines under `/opt/models/trt-edgellm/<model>-<precision>/engines`.
3. **Evaluate** (`run_eval.sh trt_edgellm ...`): the `trt_edgellm` backend writes every request into one input JSON, runs `llm_inference` once, and saves the runner's profile as `trt_profile.json`. Latency is available only as averages.

On Thor (JetPack 7), build `jetson/Dockerfile` on a Thor base image, then build the TensorRT image with `--build-arg EMBEDDED_TARGET=jetson-thor` and that board's `L4T_RELEASE` / `L4T_SOC`. The Dockerfile header lists these arguments.

## Notes and workarounds

- The container runs as your user with group `mlusers`, so files in `/opt/hf-cache` stay shared.
- Runs are offline by default. `download_assets.sh` also prepares the `datasets` Arrow cache, which avoids the HF login that the MME task config (`token: True`) would otherwise require.
- Models load from the local snapshot path, because transformers 4.57.3 queries the Hub for repo IDs even in offline mode.
- lmms-eval exits 0 even when evaluation fails, so `run_eval.sh` checks the log for `Error during evaluation`.
- The repo is mounted at its host path, because `lmms_eval/llm_judge/factory.py` fails when the repo is mounted at a shallow path such as `/workspace`.
