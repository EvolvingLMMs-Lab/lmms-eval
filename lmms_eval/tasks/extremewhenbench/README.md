# ExtremeWhenBench — lmms-eval task

Hour-scale natural-language temporal grounding benchmark. 2,273 questions
over 194 hour-long videos (mean 75.7 min, max 9 hr) sourced from LVBench,
MLVU, and VideoMME. Each prediction is a `[start, end]` interval in seconds;
scoring is mIoU and R@{0.3, 0.5, 0.7}.

Companion to the paper *Natural-Language Temporal Grounding in Hour-Long
Videos is a Search Problem: A Benchmark and Empirical Decomposition*.

## Files

```
extremewhenbench/
├── extremewhenbench.yaml   # task config
├── utils.py                # doc_to_visual, doc_to_text, IoU/R@τ aggregation
└── README.md               # this file
```

The HF dataset (`min1321/extreme-when-bench`) ships annotations only.
Videos are reused from the existing lmms-eval caches of the
source-corpus tasks (`lvbench`, `mlvu`, `videomme`) — nothing is
re-distributed.

## Video resolution

`utils.py:ewb_doc_to_visual` resolves each video by `source_corpus + video_id`:

| Corpus    | Default lookup path                                       |
| --------- | --------------------------------------------------------- |
| LVBench   | `$HF_HOME/lvbench/all_videos/{video_id}.mp4`              |
| MLVU      | `$HF_HOME/mlvu/{video_id}.mp4`                            |
| VideoMME  | `$HF_HOME/videomme/data/{video_id}.mp4`                   |

If the file is not under those paths, the resolver also walks one
sub-directory level (matches the actual layout of the source-corpus HF
caches). Override per corpus with:

```bash
export EWB_LVBENCH_PATH=/abs/path/to/lvbench/videos
export EWB_MLVU_PATH=/abs/path/to/mlvu/videos
export EWB_VIDEOMME_PATH=/abs/path/to/videomme/videos
```

Optional speed-up: set `EWB_PREEXTRACTED_DIR` to a directory of
lightweight pre-encoded videos (e.g., uniform 1024-frame, 384×N re-encodes
named `{video_id}.mp4`). The resolver tries it first, then falls back.

## Running

### Recommended: vLLM serve + openai adapter (matches paper)

The paper uses vLLM's server-side video decoding (with frame timestamps
attached) for hour-long temporal grounding. lmms-eval's default video
preprocessing extracts frames client-side, which loses the absolute-time
signal that the model needs for hour-scale videos. To reproduce the
paper number, point lmms-eval's `openai` adapter at a vLLM serve with
the `pass_video_url` and `enable_thinking_kwarg` options (added by a
companion lmms-eval PR — see `patches/chat_openai_patched.py`).

```bash
export HF_HOME=/path/to/your/hf-cache   # used by both vLLM and lmms-eval

# 1) Make sure source-corpus videos are cached under $HF_HOME
#    (skip any corpus you have already downloaded)
python -c "from datasets import load_dataset; load_dataset('lmms-lab/LVBench')"
python -c "from datasets import load_dataset; load_dataset('sy1998/MLVU_dev')"
python -c "from datasets import load_dataset; load_dataset('lmms-lab/Video-MME')"

# 2) Serve Qwen3.5-9B (Qwen3_5ForConditionalGeneration)
#    --allowed-local-media-path must cover wherever the videos live.
#    Pointing it at HF_HOME covers all three source-corpus caches above.
vllm serve /path/to/Qwen3.5-9B \
    --served-model-name qwen3.5-9b \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enforce-eager \
    --allowed-local-media-path "$HF_HOME"

# 3) Run lmms-eval against the server
python -m lmms_eval \
    --model openai \
    --model_args "model=qwen3.5-9b,base_url=http://localhost:8000/v1,api_key=EMPTY,\
                  pass_video_url=True,max_frames_num=768,enable_thinking_kwarg=False,\
                  num_concurrent=32" \
    --gen_kwargs "max_new_tokens=128,temperature=0,top_p=0.8" \
    --tasks extremewhenbench \
    --batch_size 1 \
    --output_path ./logs/ewb \
    --log_samples
```

Expected on full 2,273 q: `mIoU ≈ 0.047 ± 0.005` for Qwen3.5-9B at
`num_frames=768`.

### Other models

Any model adapter that talks to a vLLM serve via OpenAI-compatible API
should work the same way. For non-vLLM backends, the `pass_video_url`
option is ignored — frames are extracted client-side and timestamps are
injected as text annotations (lmms-eval default).

## Notes

- The prompt template includes the original video duration via
  `{dur:.0f}-second long video`. This is **required** for hour-scale
  performance — without explicit duration, the model defaults to short-clip
  output (mIoU collapses by ~40%).
- `parse_failures` count as `IoU = 0` (strict; matches the paper convention).
- The 5 (qid, video_id, source_corpus, question, correct_interval) fields
  are sufficient for evaluation; `event_summary`, `category`, and
  `youtube_url` are provided for downstream analysis.

## Citation

```bibtex
@inproceedings{seo2026hourlong,
  title  = {Natural-Language Temporal Grounding in Hour-Long Videos
            is a Search Problem: A Benchmark and Empirical Decomposition},
  author = {Seo, Sukmin and Kim, Geewook},
  year   = {2026}
}
```
