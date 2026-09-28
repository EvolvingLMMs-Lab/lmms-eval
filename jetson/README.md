# lmms-eval on Jetson: inference-framework benchmarks

This folder benchmarks vision-language models on NVIDIA Jetson with lmms-eval. It runs the same model and tasks through four inference frameworks and measures accuracy, latency, memory, power and FLOPs the same way for each. So far the model is Qwen2.5-VL-3B/7B-Instruct, on a Jetson AGX Orin 32GB and a Jetson AGX Thor.

| Framework | `<framework>` | Precisions (`<size>-<precision>`, first = default) | How it runs |
|---|---|---|---|
| Hugging Face transformers | `hf` | `bf16` | in process, lmms-eval `qwen2_5_vl` backend; the reference |
| vLLM | `vllm` | `bf16`, `awq` (official AWQ 4-bit checkpoint) | in process, lmms-eval `vllm` backend |
| llama.cpp | `llamacpp` | `q4_k_m`, `q8_0`, `f16` (GGUF + f16 vision projector) | `llama-server` container + lmms-eval `openai` backend, streaming |
| TensorRT Edge-LLM | `trt_edgellm` | `fp16`, `int4_awq`, `fp8`, `nvfp4` (fp8/nvfp4: Thor only) | C++ runner via lmms-eval `trt_edgellm` backend; see [docs/tensorrt-edgellm.md](docs/tensorrt-edgellm.md) |

## Results

| Board | Summary | Notes (setup, deviations, failures) |
|---|---|---|
| Jetson AGX Orin 32GB (MAXN) | [results/orin/SUMMARY.md](results/orin/SUMMARY.md) | full MME for HF only; vLLM and llama.cpp as 8-sample smoke runs |
| Jetson AGX Thor (120 W) | [results/thor/SUMMARY.md](results/thor/SUMMARY.md) | [results/thor/NOTES.md](results/thor/NOTES.md) |

Each summary has one table of scores, latency, memory and power per run, and one table of FLOPs per sample (vision encoder / LLM prefill / LLM decode) with the achieved TFLOP/s. How everything is measured, and which settings keep the frameworks comparable: [docs/methodology.md](docs/methodology.md).

## Quick start

All commands run from the repo root. Results go to `jetson/results/<board>` (`orin`, `thor`, detected from the board; override with `RESULTS_DIR`).

```bash
# 1. Build the images once (Thor: see docs/setup-thor.md for the build arguments)
docker build -f jetson/docker/Dockerfile -t lmms-eval-jetson:latest .
docker build -f jetson/frameworks/llamacpp/Dockerfile -t llamacpp-jetson:b11135 jetson/frameworks/llamacpp   # llama.cpp only

# 2. Download the model for a framework (+ MME) into the shared HF cache (/opt/hf-cache); runs are offline afterwards
jetson/scripts/download_assets.sh vllm 3b-awq
jetson/scripts/download_assets.sh dataset:lmms-lab-encoder/GQA:testdev_balanced_instructions   # other datasets: dataset:<repo>[:<config>]

# 3. One run: <framework> <size>[-<precision>] [tasks=mme] [limit]
jetson/scripts/run_eval.sh vllm 3b-awq mme 8      # smoke test on 8 samples first
jetson/scripts/run_eval.sh vllm 3b-awq mme        # full MME

# 4. Many runs in a row (SUMMARY.md is refreshed at the end), or a whole experiment
jetson/scripts/run_matrix.sh mme "" -- hf 3b vllm 3b-awq llamacpp 3b-q8_0
jetson/experiments/qwen2_5_vl/gqa.sh              # SMOKE=1 for 8 samples per run
python3 jetson/scripts/summarize.py               # rebuild SUMMARY.md at any time
```

Useful options (environment variables of `run_eval.sh`; its header lists them all):

```bash
ATTN=sdpa jetson/scripts/run_eval.sh hf 7b mme                                   # HF without flash-attn
VLLM_GPU_MEM=0.3 jetson/scripts/run_eval.sh vllm 7b mme                          # vLLM memory fraction
EXTRA_MODEL_ARGS=max_pixels=802816 jetson/scripts/run_eval.sh hf 7b mme          # extra --model_args
RUN_TAG=pil EXTRA_MODEL_ARGS=pass_pil_images=True jetson/scripts/run_eval.sh vllm 3b mme   # variant -> <framework>-<precision>+pil
OFFLINE=0 jetson/scripts/run_eval.sh hf 3b mme                                   # allow Hub downloads during the run
```

Stop other GPU jobs before measuring: they take unified memory and skew latency. `run_eval.sh` waits until `nvidia-smi` shows no other GPU process (`WAIT_GPU_IDLE=0` to skip) and records other containers and GPU processes in `run_info.txt`.

## Experiments

One script per question, under [experiments/qwen2_5_vl/](experiments/qwen2_5_vl/):

| Script | Question |
|---|---|
| `mme.sh` | Main comparison: MME (2,374 yes/no questions) for every framework and precision, 3B and 7B |
| `gqa.sh` | Short-answer VQA: GQA testdev-balanced (12,578 questions), same matrix |
| `followups.sh` | Image transport overhead (PIL vs PNG/base64), decode-heavy captioning (COCO), image-resolution sweep |

## Layout

```
jetson/
  README.md
  docs/
    methodology.md          what is measured and how; fairness settings; FLOPs; result files
    setup-thor.md           building and running on Jetson Thor (JetPack 7)
    tensorrt-edgellm.md     ONNX export, engine build, evaluation with TensorRT Edge-LLM
    agent-prompt-thor.md    the prompt used to run the Thor benchmark unattended with Claude Code
  docker/                   lmms-eval image (Dockerfile, requirements-jetson.txt)
  scripts/
    run_eval.sh             one run: start servers, eval in the container, tegrastats, FLOPs, slim the run dir
    run_matrix.sh           many runs in a row + summary
    download_assets.sh      models / datasets into the shared HF cache (offline-ready)
    summarize.py            results/<board>/SUMMARY.md
    compute_flops.sh        analytical FLOPs per sample (runs models/<model>/flops.py in the container)
    slim_run.py             shrink a run dir for git (strip progress bars, gzip per-sample files)
  frameworks/
    common.sh               argument parsing, board / model resolution, Hub cache helpers
    hf.sh, vllm.sh, llamacpp.sh, trt_edgellm.sh   per framework: precisions, assets, backend args, server start/stop
    llamacpp/Dockerfile     llama.cpp server image
    trt_edgellm/            runtime + export images, export.sh, build_engines.sh
  models/
    qwen2_5_vl/
      model.sh              sizes -> Hub repos (base, AWQ, GGUF), image-token range
      flops.py              FLOPs model of Qwen2.5-VL (vision encoder, LLM prefill, decode)
  experiments/qwen2_5_vl/   mme.sh, gqa.sh, followups.sh
  results/<board>/          SUMMARY.md, NOTES.md, logs/, <model>/<task>/<framework>-<precision>[+tag]/<run id>/
```

## Extending

- **Framework**: add `frameworks/<name>.sh` defining `FW_PRECISIONS`, `fw_assets`, `fw_setup` and, for servers, `fw_start` / `fw_stop`. Use the model variables (`HF_REPO`, `IMAGE_MIN_PIXELS`, ...) rather than repo names.
- **Model**: add `models/<model>/model.sh` (same variables as `models/qwen2_5_vl/model.sh`) and run with `MODEL=<model>`. `flops.py` is architecture-specific, so a new model family needs its own; without one, set `FLOPS=0`.
- **Task**: any lmms-eval task works. Download its dataset first with `download_assets.sh dataset:<repo>[:<config>]`, since runs are offline.
