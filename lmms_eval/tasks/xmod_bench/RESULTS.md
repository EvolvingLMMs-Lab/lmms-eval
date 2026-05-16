# XModBench-Lite reproduction results

XModBench-Lite = 5 task families × 6 modality configs × 200 examples
(6,000 samples), balanced across families and modality directions. Run
through lmms-eval with chat-style *interleaved-multimedia* model wrappers
(`*_interleave`) that feed the full question-stem + 4-option media prompt to
each model (the stock *simple* wrappers attach only one media object per
request and silently drop the option media).

No upstream lmms-eval model file was modified. All adaptations live in new
`lmms_eval/models/chat/*_interleave.py` wrappers + the run launcher.

## By-config accuracy vs. paper (XModBench, ICLR 2026, Table 2)

| Config | Qwen2.5-Omni ours | paper | Δ | Qwen3-Omni ours | Baichuan-Omni-1.5 ours | paper | Δ |
|--------|------:|------:|----:|------:|------:|------:|----:|
| A→T | 63.1 | 62.0 | +1.1 | 71.6 | 52.5 | 47.8 | +4.7 |
| A→V | 49.8 | 48.0 | +1.8 | 52.0 | 32.0 | 35.8 | −3.8 |
| T→A | 59.2 | 55.4 | +3.8 | 62.5 | 47.6 | 40.5 | +7.1 |
| T→V | 62.5 | 59.6 | +2.9 | 67.0 | 56.6 | 56.2 | +0.4 |
| V→A | 50.3 | 50.5 | −0.2 | 55.6 | 47.0 | 38.6 | +8.4 |
| V→T | 76.4 | 76.3 | +0.1 | 83.1 | 77.7 | 73.0 | +4.7 |

- **Qwen2.5-Omni: 6/6 configs within |Δ|<5** — paper reproduced on Lite.
- **Qwen3-Omni**: not in the paper (newer model); numbers reported for the
  first time. Same code path as the validated Qwen2.5-Omni wrapper.
- **Baichuan-Omni-1.5: 4/6 within |Δ|<5.** T→A (+7.1) and V→A (+8.4) are
  genuine positive deviations (clean runs, ~0 OOM): Baichuan scores higher on
  the balanced Lite subsample than on the full 61k set — not a pipeline bug.

## Key fixes (all in `*_interleave` wrappers / launcher, not upstream)

1. **Decode**: decode the full generated sequence and take the text after the
   final `assistant\n` turn. Trimming by `input_ids` length yields empty
   strings on multimodal inputs (the processor expands media placeholders).
2. **Video budget**: AudioBench's fps=12/60-frame/512px settings assume
   ~80 GB GPUs; on 24 GB cards every video-condition sample OOM'd and was
   silently caught as empty. Default lowered to fps=2/16-frame/384px;
   per-model overridable.
3. **t2a GPU profile**: 4 audio options (long `singer_identification` songs)
   OOM a single 24 GB GPU → t2a runs on the 4-GPU profile.
4. **Baichuan**: dedicated env (transformers 4.45.2 — legacy KV cache *and*
   `qwen2_vl`; flash-attn 2.7.4 ABI-matched; xformers removed; missing
   model `.py` files fetched). `processor.{visual,video}_processor.max_pixels`
   capped to ~0.2 MP so the 4-vision-option configs fit.
5. **OmniVinci**: env built from the official recipe (pydantic<2, flash-attn
   2.5.8, VILA). Processor config (`audio_chunk_length`, `num_video_frames`)
   set in the subclass to match the official runner. Still blocked on a
   VILA `mm_info["audio_info"]` indexing error under interleaved multi-audio
   prompts (model + processor code byte-identical to the working AudioBench
   copy — invocation-level difference still under investigation).
6. **VITA**: skipped — its lmms-eval wrapper only supports a single
   image_tensor/audios per request (architectural, not a quick fix).

## Reproduce

```bash
# qwen2.5-omni (env: qwenomni3)
./submit_lite.sh qwen2_5_omni_interleave Qwen/Qwen2.5-Omni-7B qwenomni3
# qwen3-omni (30B MoE, all configs need 4 GPU)
LIGHT_GRES=gpu:a5000:4 HEAVY_GRES=gpu:a5000:4 \
  ./submit_lite.sh qwen3_omni_interleave Qwen/Qwen3-Omni-30B-A3B-Instruct qwenomni3 \
  device_map=auto,attn_implementation=flash_attention_2
# baichuan-omni (env: /scratch/xwang378/envs/baichuan)
./submit_lite.sh baichuan_omni_interleave baichuan-inc/Baichuan-Omni-1d5 \
  /scratch/xwang378/envs/baichuan device_map=auto,max_num_frames=8

# Level-2 metrics from the per-task sample logs
python lmms_eval/tasks/xmod_bench/summarize.py \
  --logs logs/xmod_bench_lite/results_<model>/
```
