# Jetson Thor run notes

Unattended run following `jetson/THOR_PROMPT.md`, started 2026-09-23.

## Environment
- Board: Jetson AGX Thor, L4T R38.4 (kernel 6.8.12-tegra), 14 CPU cores, 125.8 GB unified memory.
- **Power mode: 120W (nvpmodel mode 1, the board default), not MAXN.** `sudo -n` needs a password, so
  `nvpmodel -m 0` was not possible. Orin ran at MAXN. `jetson_clocks` not used (as on Orin).
- Shared machine: another user (`fabien`) was running a host (non-Docker) lmms-eval job at the start
  (HF Qwen2.5-VL-7B on GQA, ~9 GB GPU memory). It was not stopped. `run_eval.sh` now waits until no other
  GPU compute process is listed by `nvidia-smi` before each run and records any in `run_info.txt`.
  No running Docker containers at the start, so none were stopped.
- `/opt/hf-cache` exists and is group-writable (`mlusers`): used as `HF_CACHE`.
- `/opt/models` does not exist and `/opt` is root-owned; without sudo it cannot be created.
  **`TRT_WORKSPACE` falls back to `$HOME/opt-fallback/models/trt-edgellm`.**
- `gh` is not installed and the HTTPS remote has no credentials; pushes go over SSH
  (`git push git@github.com:michalszczepanski91/lmms-eval.git jetson-benchmark`), which works.
- Disk: 65 GB free on / at the start (93% used), 33 GB after building the images and downloading the models.
  Edge-LLM ONNX exports are deleted once their engines are built, and engine sets are deleted after their runs
  when space runs low. (`~/dev/quantize-work`, 35 GB, belongs to other work and was left alone.)
- **System clock is ~7h13m slow** (NTP active but not synchronized, RTC at 1970; no sudo to fix). Run IDs
  (timestamps) and commit dates are therefore ~7 h early. apt inside `docker build` rejects the Ubuntu Release
  files as "not valid yet", hence the new `APT_OPTS` build argument (see below).
- Orin baseline available in `jetson/results`: full MME only for HF bf16 3B and 7B; vLLM bf16/AWQ and llama.cpp
  Q8_0 have 8-sample smoke runs only, llama.cpp Q4_K_M none, and the follow-ups were not run on Orin. Thor vs
  Orin comparisons are therefore complete only for HF.

## Images and versions
- `lmms-eval-jetson:latest` on `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor`: torch 2.10.0 (CUDA, sm_110),
  transformers 4.57.3, vLLM 0.19.0+cu130, flash-attn 2.8.4, CUDA 13.0. Same torch/transformers/vLLM as Orin;
  flash-attn is present, so HF runs use `flash_attention_2` as on Orin (no `ATTN=sdpa`). No pins loosened.
- `llamacpp-jetson:b11135`: llama.cpp b11135, `CUDA_ARCH=110`, built without changes.
- TensorRT Edge-LLM **v0.10.1** (2026-09-03), the latest release; its support matrix lists Jetson Thor on
  JetPack 7.0/7.1 with CUDA 13.0 as official. This board: L4T R38.4 (JetPack 7.1), CUDA 13.0, TensorRT 10.13.3.
  - Runtime image `lmms-eval-trt-edgellm:latest`: `lmms-eval-jetson` + TensorRT 10.13 dev packages from the L4T
    r38.4 apt repo (`common` + `som`), Edge-LLM built with `EMBEDDED_TARGET=jetson-thor CUDA_CTK_VERSION=13.0
    ENABLE_CUTE_DSL=ALL`.
  - Export/quantize image `lmms-eval-trt-export:v0.10.1`: `nvcr.io/nvidia/pytorch:26.05-py3` (torch
    2.12.0a0 NGC build) + Edge-LLM's `tools` extra (transformers 5.14.1, modelopt 0.45.0), keeping NGC torch.

## Script and backend changes
- `run_eval.sh`: `WAIT_GPU_IDLE` (default 1) waits for other GPU compute processes (via `nvidia-smi`, so host
  jobs of other users are seen too); `run_info.txt` lists them. Skipped where `nvidia-smi` is unavailable.
- `jetson/Dockerfile`, `trt_edgellm/Dockerfile`: `APT_OPTS` build argument (default empty), used here as
  `APT_OPTS="-o Acquire::Check-Date=false"` because of the clock.
- `run_matrix.sh`: messages use `$RESULTS_DIR` instead of a hard-coded `jetson/results`.
- `summarize.py`: power is read per board. Orin has `VDD_GPU_SOC`, Thor `VDD_GPU` (GPU only) and
  `VDD_CPU_SOC_MSS`, so the old `VDD_GPU_SOC` regex gave no power on Thor. Now: "GPU rail" (VDD_GPU_SOC or
  VDD_GPU) and "module" (sum of GPU, CPU/SoC and 5V system rails, comparable across boards). The header shows
  the power mode from `run_info.txt` instead of a fixed "MAXN"; the "other" column includes other GPU processes.
- TensorRT Edge-LLM (untested scripts, rewritten against v0.10.1):
  - `export.sh`: Edge-LLM 0.8 removed `tensorrt-edgellm-export-llm/-export-visual/-quantize-llm`; now
    `tensorrt-edgellm-quantize llm` (fp8/nvfp4) + `tensorrt-edgellm-export <ckpt> <out>` (LLM + visual).
    int4_awq exports Qwen's official AWQ checkpoint (the same one vLLM awq uses) instead of re-quantizing.
    Checkpoints are read from the shared `$HF_CACHE` (the old script downloaded a second copy into the workspace).
    Runs in a prebuilt image (`Dockerfile.export`, built on demand) as the calling user.
  - `Dockerfile.export` (new): Edge-LLM's extras pin `torch==2.13.0`; installed without torch/torchvision so the
    container's CUDA torch stays. Needs the git submodules (license file of `3rdParty/nlohmannJson`).
  - `trt_edgellm/Dockerfile`: optional `CUDA_CTK_VERSION` / `ENABLE_CUTE_DSL` build args (required CMake options
    since Edge-LLM 0.8; empty = not passed, so the Orin v0.6.0 build is unchanged).
  - Backend `trt_edgellm`: new `encoder_cache_budget_bytes` option. Edge-LLM 0.10 caches vision-encoder outputs
    across requests by default (256 MiB); `trt_edgellm.sh` passes 0, matching vLLM/llama.cpp with caching off.
  - `trt_edgellm/Dockerfile`: the Thor vLLM base image carries an unpackaged copy of TensorRT 10.13.2 headers in
    `/usr/include` (the apt package has 10.13.3 in `/usr/include/aarch64-linux-gnu`). Edge-LLM's FindTensorRT picked
    `/usr/include`, the resulting `-isystem /usr/include` broke libstdc++'s `#include_next <math.h>` in every CUDA
    file. Fixed by passing `-DTensorRT_INCLUDE_DIR` from `dpkg -L libnvinfer-headers-dev`.
  - `trt_edgellm/Dockerfile`: the base image exports `CUDAARCHS=110`, overriding Edge-LLM's toolchain target
    `110a`; the FP4 kernels then fail in ptxas (`cvt with .e2m1x2 not supported on sm_110`). Unset for the build.
  - `trt_edgellm/Dockerfile`: `EDGELLM_PLUGIN_PATH` set; otherwise `llm_build`/`llm_inference` look for
    `build/libNvInfer_edgellm_plugin.so` relative to the working directory ("Plugin not found" for AttentionPlugin).
  - `export.sh`: int4_awq failed in the visual export (`hidden_size // None`): Qwen's AWQ repos store a vision_config
    without the fields that have defaults (num_heads, depth, ...). The LLM is now exported from the LLM checkpoint
    (`--skip-visual`) and the vision encoder always from the base checkpoint (`--skip-llm`). The AWQ repos' 390
    vision tensors were checked to be bit-identical to the base checkpoint's, so this changes nothing numerically,
    and every precision uses the same FP16 vision encoder.
  - `export.sh`: `HF_TOKEN_PATH` points away from `/opt/hf-cache/token` (another user's, unreadable), which the
    quantizer's calibration download (public `abisee/cnn_dailymail`) otherwise tried to read; `USER`/`LOGNAME` are
    passed (torch needs a user name for uid 1001 inside the container).
  - `build_engines.sh`: `REMOVE_ONNX=1` deletes the ONNX export after a successful build (disk).
  - Build times on Thor (with the other user's job running): export fp16/int4_awq ~2 min; quantize+export fp8/nvfp4
    ~9 min; engine build ~2 min. 3B engine sets: fp16 7.7 GB, fp8 5.1 GB, int4_awq 3.8 GB, nvfp4 4.0 GB.
  - Checked with a two-request probe of `llm_inference` (red panda image): output fields `output_text`,
    `request_idx`, `finish_reason` and profile keys `prefill`, `generation`, `multimodal`, `stages`, `wall_clock`
    match the backend and `summarize.py`; warmup runs are not counted in the profile.
  - Backend: the runner exits 1 if any request failed but still writes all responses, with the error text as
    `output_text` and `finish_reason: "error"`. The backend used `check=True` (one bad sample would abort the run)
    and would have scored error text as answers. Failed requests are now empty answers, counted in the profile as
    `failed_requests`. `test/models/test_trt_edgellm.py` covers this and the encoder-cache flag.

## Log
- Phase 2, Edge-LLM 3B full MME (fp16/int4_awq/fp8) first attempt: perception ~1450 vs 1547 for HF on Orin. Per
  category everything matched HF except landmark (64% vs 96%): 128 requests (64 landmark photos x 2 questions)
  failed in the runner with "Raw image 2848x4288 exceeds the GPU-resize budget of 4096x4096 pixels", so they scored
  as empty answers (the backend's failed-request handling kept the run alive). Fix: backend option
  `max_image_side` (4096 in `trt_edgellm.sh`) downscales such images (bicubic, aspect ratio kept) before the runner;
  every framework resizes them to <= 2048 visual tokens (~1.6 MP) anyway. The first runs are kept as
  `trt_edgellm-<p>+no-downscale`; the three runs are repeated after the main matrix.
- Phase 1 (smoke, `mme 8`): the other user's job ended ~19:55 (board clock); all 10 HF/vLLM/llama.cpp combinations
  finished with sensible Yes/No answers and per-sample timing. Edge-LLM 3B: fp16 and fp8 correct.
  - `run_eval.sh` fix: lmms-eval writes the trt_edgellm result files directly into the run dir (no model-named
    subdir), which run_eval.sh did not move into `lmms_eval/`, so summarize.py skipped them.
  - int4_awq exported from Qwen's AWQ checkpoint (attempt 1, kept as `trt_edgellm-int4_awq+qwen-awq-ckpt`):
    answers "Hello" to every MME question. Probe: text-only prompts are answered correctly, every image prompt
    ends immediately with an empty answer (patching `vision_config.dtype` in the engine config changed nothing).
    Attempt 2: AWQ via `tensorrt-edgellm-quantize` on the base checkpoint (now the default, `AWQ_SOURCE=qwen` for the
    old path) - correct answers. Note: Edge-LLM int4_awq is therefore not the same AWQ checkpoint as vLLM awq.
  - nvfp4 (3B, quantized here with `tensorrt-edgellm-quantize --quantization nvfp4`): garbage output even for a
    text-only prompt ("嚯嚯。 本 1.111..."); fp8 from the same pipeline is fine. Deferred: test the 7B with NVIDIA's
    pre-quantized `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` to separate the kernels from the 3B quantization.
- Disk reached 4.5 GB free after the 3B Edge-LLM engines + calibration data; deleted my calibration dataset cache
  (2.1 GB, re-downloaded for 7B). Plan: evaluate the Edge-LLM 3B engines early in phase 2 and then delete them.
- Phase 2 (main MME matrix) stopped at llama.cpp: `llamacpp 3b-q8_0` hung at sample 935/2374 for ~27 h (llama-server
  stuck on one request, one CPU core at 100%, `run.log` not growing). Killed on 2026-09-25 together with the matrix, so
  llama.cpp 3B q4_k_m and 7B q8_0 / q4_k_m full MME were never run. The hung run dir
  (`Qwen2.5-VL-3B-Instruct/mme/llamacpp-q8_0/20260924-015342`) is kept as evidence; it has no results.
  `THOR_PROMPT.md` now has a watchdog rule (kill a run whose `run.log` hasn't grown for 30 min).
- FLOPs (2026-09-25): `jetson/flops.py` / `compute_flops.sh` compute analytical FLOPs per sample (vision encoder,
  LLM prefill, LLM decode) into `flops.json`; `run_eval.sh` runs it after every eval, older runs were backfilled.
  Validated against torch's `FlopCounterMode` on a scaled-down random Qwen2.5-VL built from the real transformers
  code (eager attention): vision exact on 4 image shapes; LLM exact except that `flops.py` counts causal attention
  as L(L+1)/2 (what FlashAttention-style kernels compute), while eager attention computes the full L x L.
- Phase 5 GQA (2026-09-25, testdev-balanced, 12,578 questions): HF bf16 3B/7B, vLLM bf16/AWQ 3B/7B, Edge-LLM 3B
  fp16/fp8/int4_awq. llama.cpp left out until its hang is understood.
  - **HF 7B GQA timings are not valid**: 427 min, ~1.9 s per sample (TTFT ~0.94 s) from start to end, against
    0.28 s TTFT on MME with 2.4x more image tokens and 83-87 min for vLLM 7B GQA. GPU rail only ~11 W, i.e. the GPU
    was mostly idle. tegrastats shows one CPU core pinned at 100% @ 2.6 GHz by something outside the run during the
    GQA runs (also seen during the llama.cpp hang, not during MME); gone by 2026-09-28, cause unknown. Accuracy is
    unaffected (0.6088); the run is to be repeated.
- Result size (2026-09-28): `jetson/slim_run.py` (run by `run_eval.sh` after every run) strips per-request progress
  bars and Edge-LLM profile-switch lines from `run.log`, gzips `tegrastats.log` and `*_samples_*.jsonl`, and moves
  per-sample FLOPs to `flops_samples.jsonl.gz`. results-thor went from 218 MB to 16 MB; SUMMARY.md is unchanged.
  Thor `run.log` / `tegrastats.log` had never been committed (`.gitignore` only re-included `results/**/*.log`);
  fixed, so they are committed from now on.
- Reorganization (2026-09-28): `jetson/` split into `scripts/`, `docker/`, `docs/`, `models/<model>/` (Qwen2.5-VL repos,
  image range, FLOPs) and `experiments/<model>/`; results moved to `jetson/results/<board>/` (this file:
  `results/thor/`, Orin: `results/orin/`), picked from the board automatically. Paths above are the old ones.
  Orchestration logs moved to `logs/`.
