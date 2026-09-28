# Methodology

How the Jetson benchmarks measure accuracy, speed, memory, power and compute, and which settings keep the frameworks comparable.

## Keeping the frameworks comparable

| Setting | Why |
|---|---|
| Same lmms-eval task code for every framework | Scores differ only because of the model runtime, not the prompt or the metric |
| Same image resolution range: 256–2048 visual tokens (Qwen2.5-VL: 200,704–1,605,632 pixels) | Image size drives the vision encoder and prefill cost. HF/vLLM pass the limits to the processor, llama.cpp `--image-min/max-tokens`, Edge-LLM `visual_build --min/maxImageTokens`; all read them from `models/<model>/model.sh` |
| Batch size 1, one request at a time | Edge latency, not server throughput |
| Cross-request caching off (vLLM prefix + multimodal cache, llama.cpp prompt cache, Edge-LLM encoder cache) | MME asks two questions per image, so a cache would skip the vision encoder and most of the prefill for every second sample, which the HF reference cannot do |
| Offline runs from a shared, pre-downloaded HF cache | No download time in the measurements; identical checkpoints |
| No other GPU process during a run | `run_eval.sh` waits for an idle GPU and records other containers and GPU processes in `run_info.txt` |
| Power mode recorded, `jetson_clocks` not used | Orin ran at MAXN, Thor at its default 120 W (no sudo); compare like with like |

Remaining differences are properties of the frameworks and are reported, not removed: quantization (AWQ, GGUF Q4/Q8, FP8), image transport (PNG/base64 over HTTP for llama.cpp and vLLM's default path; `+pil` / `+png1` variants measure that overhead), and Edge-LLM downscaling images with a side above 4096 px before its runner (`max_image_side`; the other frameworks shrink such images to ≤ 2048 visual tokens anyway).

## Metrics

| Metric | Source |
|---|---|
| Scores | lmms-eval task metrics (MME perception / cognition, GQA exact match, COCO CIDEr, ...) |
| TTFT per sample | From the model call to the first generated token: vision encoder + prefill + first decode step. HF: generation streamer. vLLM: engine request metrics. llama.cpp: first streamed chunk (includes HTTP). Edge-LLM: runner averages only (vision encoder + prefill) |
| Answer time per sample | Whole model call, all generated tokens |
| Decode ms/token | (answer − TTFT) / (output tokens − 1) |
| Wall time | Whole lmms-eval run, including image loading and preprocessing |
| RAM | Whole-board unified memory from `tegrastats` at 1 Hz: base (before model load) and peak; other processes included |
| Power | `tegrastats` rails averaged over the run. GPU rail: `VDD_GPU_SOC` on Orin (GPU + SoC), `VDD_GPU` on Thor (GPU only). Module: GPU + CPU/SoC + 5 V system rails, comparable across boards |
| FLOPs per sample | Analytical, see below |
| Achieved TFLOP/s | (vision + prefill FLOPs) / mean TTFT, and total FLOPs / mean answer time. A lower bound on GPU throughput, since TTFT includes preprocessing (and HTTP for llama.cpp) |

Per-sample numbers are in each run's `lmms_eval/*_samples_<task>.jsonl.gz` under `token_counts`: `input_tokens`, `output_tokens`, `preprocess_seconds`, `time_to_first_token_seconds`, `generation_seconds`. MME and GQA answers are one word (about 2 tokens), so TTFT dominates; use a longer-answer task such as COCO captioning to compare decode speed.

## FLOPs

`models/qwen2_5_vl/flops.py` computes FLOPs per sample from the model's `config.json`, each sample's image grid and its prompt / output token counts. `scripts/compute_flops.sh` runs it in the eval container after every run (`FLOPS=0` to skip) and writes `flops.json`.

- **Counted**: matrix multiplies at 2 FLOPs per multiply-accumulate: every linear layer (patch embedding, attention projections, MLPs, patch merger, LM head) plus attention scores and attention-weighted values. Norms, activations, rotary embeddings and softmax are left out (< 1%).
- **Vision encoder**: ViT with window attention over 8×8-patch windows except the full-attention blocks, then the 2×2 patch merger into the LLM width.
- **Prefill**: the LLM over the whole prompt (text + image tokens) plus the LM head for the first output token. Causal attention counts L(L+1)/2 query–key pairs, the work FlashAttention-style kernels do; an implementation that computes the full L×L matrix and masks it does up to 2× the attention FLOPs.
- **Decode**: one LLM step per further output token against the KV cache, plus the LM head.
- **Token counts**: those recorded by the backend. Edge-LLM records none, so they are recounted with the tokenizer (≈ 0.1% difference in prefill).
- **Same for every framework and precision**: quantization changes the cost per FLOP, not the count. A FLOPs difference between frameworks on the same task means different images or prompts.
- **Validated** against PyTorch's `FlopCounterMode` on a scaled-down random Qwen2.5-VL built from the real transformers classes (eager attention, 4 image shapes): vision encoder exact, LLM exact up to the causal convention above.

## Result files

Each run writes `results/<board>/<model>/<tasks>/<framework>-<precision>[+tag]/<run id>[_limitN]/`:

| File | Content |
|---|---|
| `run_info.txt` | board, L4T, power mode, git commit, image IDs, other containers / GPU processes, exact lmms-eval command, versions |
| `run.log` | console output, without per-request progress bars (see `slim_run.py`) |
| `server.log` / `trt_profile.json` | llama-server log / Edge-LLM runner profile |
| `tegrastats.log.gz` | 1 Hz RAM, CPU, GPU, temperatures, power rails |
| `flops.json`, `flops_samples.jsonl.gz`, `flops.log` | FLOPs summary per task / per sample / log |
| `lmms_eval/*_results.json` | scores and the full lmms-eval config |
| `lmms_eval/*_samples_<task>.jsonl.gz` | per-sample input, answer, target, score and `token_counts` |

Run IDs with `_limitN` are smoke tests; `summarize.py` leaves them out unless `--include-smoke`.

## Implementation notes

- Containers run as the calling user, plus group `mlusers` if it exists (`SHARED_GROUP`), so files in the shared cache stay group-writable.
- `download_assets.sh` also prepares the `datasets` Arrow cache, which works offline and avoids the HF login that task configs with `token: True` (MME, GQA) would otherwise require.
- Models load from local snapshot paths, because transformers 4.57.3 queries the Hub for repo IDs even in offline mode.
- lmms-eval exits 0 when evaluation fails, so `run_eval.sh` checks the log for `Error during evaluation`.
- The repo is mounted at its host path: `lmms_eval/llm_judge/factory.py` fails when it is mounted at a shallow path such as `/workspace`.
- Out of memory on Jetson shows up as `NVML_SUCCESS == r INTERNAL ASSERT FAILED` in PyTorch's CUDA allocator.
- `docker build` has no GPU driver, so the Dockerfiles do not run CUDA binaries.
