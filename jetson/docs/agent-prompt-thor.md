# Task: run the full Qwen2.5-VL framework benchmark on this Jetson Thor, unattended

You are on a Jetson AGX Thor, in a clone of `github.com/michalszczepanski91/lmms-eval`, branch `jetson-benchmark`. The same benchmarks already ran on a Jetson AGX Orin 32GB. Their results are in `jetson/results/orin/` (read its `SUMMARY.md`). Your job is to produce the same results on Thor, plus the TensorRT Edge-LLM runs that Orin couldn't do, with as little human involvement as possible. The user is away. Don't wait for answers: make reasonable decisions, record them, and keep going. Stop and report only if something blocks every remaining step.

Read `jetson/README.md`, `jetson/docs/setup-thor.md`, `jetson/docs/methodology.md` and `jetson/results/thor/NOTES.md` first. They describe the harness, the Thor-specific build steps and what already happened on this board. This prompt adds the rules and the order.

## Status (2026-09-28)

Done on Thor: phase 0 (setup), phase 1 (smoke), phase 2 for HF / vLLM (3B, 7B) and Edge-LLM 3B fp16 / fp8 / int4_awq (the Edge-LLM runs as `+no-downscale`, before the `max_image_side` fix), phase 5 GQA for HF / vLLM / Edge-LLM 3B. **Still to do**, in this order:
1. Re-run HF 7B GQA (the 2026-09-25 run's timings are invalid, see NOTES.md; its accuracy is fine).
2. Phase 2 llama.cpp: 3B q8_0, 3B q4_k_m, 7B q8_0, 7B q4_k_m on full MME (the 3B q8_0 run hung once; watch it).
3. Edge-LLM 3B fp16 / fp8 / int4_awq full MME again with the downscale fix.
4. Phase 3 follow-ups.
5. Phase 4 for 7B (fp16, int4_awq, fp8) and the nvfp4 check with `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4`.
6. GQA for llama.cpp and for any new Edge-LLM engines.
7. Phase 6 wrap-up.

## Ground rules

- **Shared files go in `/opt`.** This machine keeps reusable assets in `/opt`:
  - Hugging Face cache (models, datasets): `/opt/hf-cache` (`HF_CACHE`)
  - TensorRT Edge-LLM workspace (ONNX, engines): `/opt/models/trt-edgellm` (`TRT_WORKSPACE`)
  - Check that these exist and are writable. If not, try `sudo -n mkdir -p` and `sudo -n chown`/`chmod g+ws` for your user, plus group `mlusers` if it exists. If sudo isn't available non-interactively, use `$HOME/opt-fallback/...` instead, and report that prominently at the end.
  - Results and scripts stay in the repo: results go to `jetson/results/thor` automatically (board detected).
- **Export these in every shell you use:**
  ```bash
  export HF_CACHE=/opt/hf-cache TRT_WORKSPACE=/opt/models/trt-edgellm SHARED_GROUP=mlusers
  ```
- **Keep the measurements clean.**
  - Set MAXN (`sudo -n nvpmodel -m 0`; if that fails, record the current `nvpmodel -q` and continue). Don't run `jetson_clocks`; it wasn't used on Orin.
  - Before benchmarking, list running containers and GPU users (`docker ps`, `tegrastats` RAM). Stop other containers using the GPU or large amounts of memory, and record their names in `jetson/results/thor/NOTES.md`. On Orin a co-resident vLLM server caused an out-of-memory failure. Restart everything you stopped (`docker start <name>`) when you're done, even if you finish with failures.
  - Never run two benchmarks, or a benchmark and a Docker build, at the same time. Downloads during a run are OK.
- **Don't disturb running scripts.** Bash reads scripts as it executes them. Never edit `jetson/scripts/run_eval.sh` in place while a run is active: write a copy and `mv` it over. Don't `git stash`, `git checkout` or reset the working tree while anything runs.
- **Long jobs:** start them in the background and wait on completion (for example an `until grep ...; do sleep 60; done` loop); don't poll every few seconds. A full-MME run takes 30–120 min, a full-GQA run 35 min to 4 h.
- **Watch for hangs.** If a run's `run.log` hasn't grown for 30 min while its process is still alive, it's stuck (on Thor a llama.cpp MME run once hung for 27 h on a single request). Kill that `run_eval.sh` (its exit trap stops llama-server), save `docker logs` of the server first, log it in `NOTES.md` and let the matrix go on.
- **Commit and push after each phase:** `git add jetson/results/thor <changed scripts> && git commit && git push`. Use conventional commit messages. Commit only files you created or changed, plus results.
- **Fixes:** when a step fails, read the log (`run.log`, `server.log`), fix the cause in the scripts or backends, and re-run. At most 3 attempts per item; then log it as failed in `NOTES.md` and move on. Keep fixes general (for example a build argument, or detection of the Thor image's versions), not Thor-only hacks, and describe each in `NOTES.md` and the commit message. Run `uvx ruff check` / `uvx ruff format` on Python you change, and the relevant tests in `test/models` inside the image.

## Lessons from Orin (already handled in the scripts; don't undo)

- Runs are offline: `download_assets.sh` prepares the datasets' Arrow cache, which avoids the HF login that the MME task (`token: True`) would otherwise require. Models load from local snapshot paths, because transformers 4.57.3 queries the Hub even in offline mode.
- lmms-eval exits 0 on evaluation errors; `run_eval.sh` checks the log instead.
- vLLM and llama.cpp run with cross-request prefix/image caching **off**. MME asks two questions per image, and caching would favour those frameworks unfairly. Image limits are 256–2048 visual tokens in every framework.
- Out of memory on Jetson shows up as `NVML_SUCCESS == r INTERNAL ASSERT FAILED` in PyTorch's CUDA allocator.
- The default image transport (PNG/base64) is slow for MME's large `landmark` photos. `+pil` (vLLM) and `+png1` (llama.cpp) variants exist; the follow-up experiments measure the difference.
- `docker build` has no GPU driver, so don't run CUDA binaries during builds.
- llama.cpp with Qwen2.5-VL-3B: with fp16 cuBLAS accumulation, one MME image (`commonsense_reasoning/0064.png`) overflows, and **every later request returns `????????????????`** until the server restarts. `llamacpp.sh` now sets `GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16`. Scan every llama.cpp run's samples for `"??"` answers anyway, and check that per-category scores look plausible, not just the totals. Runs hit by this are kept with an `INVALID` file (the reason inside), which `summarize.py` skips.
- The vLLM 7B bf16 memory budget is 21.0 GB (0.7 of the Orin 32GB): 18.4 GB leaves no KV cache after profiling. The Thor 7B bf16 runs of 2026-09-24/25 used 18.4 GB and worked.
- **FLOPs** are computed analytically after every run: `run_eval.sh` calls `jetson/scripts/compute_flops.sh`, which writes `flops.json` (vision encoder / LLM prefill / LLM decode / total per sample) and `flops.log` to the run dir, on CPU. Then `slim_run.py` shrinks the run dir for git (per-sample files gzipped, progress bars stripped from `run.log`). `summarize.py` puts them in the "Compute (FLOPs)" table with achieved TFLOP/s. For runs without `flops.json`, run `jetson/scripts/compute_flops.sh <run dir> ...`. FLOPs are identical across frameworks and precisions (up to prompt-token counting), so a big difference between frameworks points at different image sizes or prompts: investigate it.

## Phases

**0. Setup**
1. `git pull`. Then check `gh auth status`; if it isn't logged in, you can't push. Keep committing locally, and put "push pending: run `gh auth login` then `git push`" at the top of the final report.
2. Prepare `/opt` as described above. Set MAXN. Stop other GPU containers and record them.
3. Build the images as in `jetson/docs/setup-thor.md` §2 (`BASE_IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor`, llama.cpp `CUDA_ARCH=110`). Run the version check from the same section. If `flash_attn` is missing, export `ATTN=sdpa` for every HF run and record it. If the lmms-eval requirements conflict with the Thor image, loosen the offending pins in `jetson/docker/requirements-jetson.txt`; don't replace the image's torch or vLLM.
4. Download everything: `jetson/docs/setup-thor.md` §3.

**1. Smoke tests** — `jetson/scripts/run_matrix.sh mme 8`. Every combination must finish, give sensible answers, and record per-sample timing (`token_counts` in `lmms_eval/*_samples_mme.jsonl.gz`). Fix failures before continuing. Commit and push.

**2. Main comparison** — `jetson/scripts/run_matrix.sh mme`. This is HF bf16, vLLM bf16/AWQ and llama.cpp Q8_0/Q4_K_M, each for 3B and 7B, with the default transport, as on Orin. Then `python3 jetson/scripts/summarize.py`, commit and push.

**3. Follow-ups** — `jetson/experiments/qwen2_5_vl/followups.sh`: transport fixes, COCO captioning and resolution sweep, on 3B. Thor has plenty of memory, so afterwards also run the COCO captioning for 7B: `hf 7b`, `vllm 7b` and `vllm 7b-awq` with `RUN_TAG=pil EXTRA_MODEL_ARGS=pass_pil_images=True`, and `llamacpp 7b-q8_0` and `7b-q4_k_m` with `RUN_TAG=png1 EVAL_ENV=LMMS_IMAGE_PNG_COMPRESS_LEVEL=1`. Summarize, commit and push.

**4. TensorRT Edge-LLM** — follow `jetson/docs/tensorrt-edgellm.md`.
1. Find the Edge-LLM release that supports JetPack 7 / Thor (GitHub releases of `NVIDIA/TensorRT-Edge-LLM`, and its installation guide). Use the matching PyTorch container. Record both in `NOTES.md`.
2. Export, build engines and evaluate `3b` with `fp16`, `int4_awq`, `fp8` and `nvfp4`, then `7b` with the same precisions. Smoke-test each (`mme 8`) before the full run.
3. The lmms-eval `trt_edgellm` backend and these scripts were written from the Edge-LLM docs and have never run on real hardware. Expect to fix things: the input/output JSON fields, profile keys, CLI flags, and build-image package names. Check the runner's real output against `lmms_eval/models/chat/trt_edgellm.py` and `jetson/scripts/summarize.py` (`trt_profile_stats`), fix them, and keep `test/models/test_trt_edgellm.py` passing and updated.
4. If Edge-LLM can't produce engines at all, document exactly where it fails in `NOTES.md` and continue.

Summarize, commit and push.

**5. GQA** — visual question answering, `gqa` task = GQA testdev-balanced, 12,578 questions on 398 images, exact match on a single word or phrase.
1. Download `dataset:lmms-lab-encoder/GQA:testdev_balanced_instructions` and `dataset:lmms-lab-encoder/GQA:testdev_balanced_images` with `download_assets.sh` (skip if `$HF_CACHE/datasets/lmms-lab-encoder___gqa/testdev_balanced_*` exist). The first run's task load joins images to questions (a few minutes, cached afterwards).
2. Smoke test: `jetson/scripts/run_matrix.sh gqa 8 -- hf 3b`. Check that the answers are sensible and that `flops.json` exists.
3. Full runs: `jetson/experiments/qwen2_5_vl/gqa.sh` (HF, vLLM, llama.cpp x 3B/7B, Edge-LLM 3B), or `jetson/scripts/run_matrix.sh gqa "" -- <framework> <size-precision> ...` for a subset; add every TensorRT Edge-LLM engine that gave sensible MME answers. About 9 runs x 1–4 h: expect roughly a day.

Summarize, commit and push.

**6. Wrap-up**
1. Restart the containers you stopped. Restore the power mode if you changed it.
2. `python3 jetson/scripts/summarize.py` → `jetson/results/thor/SUMMARY.md`.
3. Write `jetson/results/thor/REPORT.md` for the user. Lead with the answer, then support it:
   - A table per model (3B, 7B): framework/precision → MME perception + cognition, TTFT p50/p90, answer p50, decode ms/token, wall time, peak RAM, average power.
   - **Thor vs Orin** for the same framework/precision/transport, as a speedup and a score difference.
   - COCO captioning: CIDEr/BLEU-4 and decode speed per framework (this is where decode speed shows).
   - Resolution sweep: accuracy vs TTFT for 256/512/1024/2048 tokens.
   - TensorRT Edge-LLM results by precision (averages only), or where it failed.
   - GQA: exact match per framework/precision next to TTFT, answer latency and wall time; does the MME ranking hold?
   - Compute: FLOPs per sample split into vision encoder / prefill / decode / total for 3B and 7B on MME and GQA, and achieved TFLOP/s per framework/precision (which share of the work is the vision encoder, and how close each framework gets to the GPU's peak).
   - Everything that deviated from Orin (`ATTN=sdpa`, loosened pins, fallback paths, failed items), and all fixes made to scripts or backends.
4. Commit and push. End with a short message giving the report path, the headline findings, and anything that needs the user's action.
